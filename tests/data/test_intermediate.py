"""Tests for building the intermediate dataset (raw text -> standard, typed, flagged)."""

import datetime as dt
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from data.datasets import DatasetConfig
from data.intermediate import EVENT_ID, build_intermediate, standardise

SOURCE: dict[str, Any] = {
    "url": "https://example.org/toy.zip",
    "archive": "toy.zip",
    "md5": "0" * 32,
    "member": "toy.csv",
    "max_uncompressed_bytes": 10_000,
    "doi": "10.0/toy",
    "version": "v1",
    "license": "CC0-1.0",
    "citation": "Toy (2026).",
    "homepage": "https://example.org/toy",
}
CONFIG: dict[str, Any] = {
    "name": "toy",
    "description": "Toy.",
    "source": SOURCE,
    "columns": {
        "user_id": "User",
        "timestamp": "When",
        "login_success": "Ok",
        "ip": "IP",
        "asn": "ASN",
        "region": "Region",
        "rtt_ms": "RTT",
        "user_agent": "UA",
    },
    "labels": {"label_ato": "ATO"},
    "null_markers": {"region": ["-"]},
    "identifier_dtypes": {"user_id": "Int64"},
    "artefacts": [
        {"name": "asn_is_placeholder", "field": "asn", "kind": "at_least", "value": 500_000,
         "meaning": "fake"},
        {"name": "ip_is_placeholder", "field": "ip", "kind": "in_network", "value": "10.0.0.0/8",
         "meaning": "fake"},
    ],
}  # fmt: skip

FIRST_USER_ID = -4324475583306591935  # a real-looking 64-bit pseudonym from the RBA file

ROWS: dict[str, list[str | None]] = {
    "index": ["0", "1", "2", "3"],
    "When": [
        "2020-02-03 12:43:30.772",
        "2020-02-03 12:44:00.000",
        "2020-02-04 08:00:00.001",
        "2020-02-05 23:59:59.999",
    ],
    "User": [str(FIRST_USER_ID), "42", "42", "7"],
    "Ok": ["True", "False", "TRUE", "false"],
    "IP": ["10.0.65.171", "100.1.2.3", "194.87.207.6", "10.255.255.255"],
    "ASN": ["29695", "507727", "500000", "499999"],
    "Region": ["-", "Nan", None, "Vestland"],
    "RTT": [None, "1123.0", "30", None],
    "UA": ["Mozilla/5.0 (iPhone)", "curl/8", "Mozilla/5.0 (X11)", "-"],
    "ATO": ["False", "True", "False", "False"],
}


def _config(**overrides: Any) -> DatasetConfig:
    return DatasetConfig.model_validate({**CONFIG, **overrides})


def _build(rows: dict[str, list[str | None]] = ROWS, **overrides: Any) -> pl.DataFrame:
    raw = pl.LazyFrame(rows, schema=dict.fromkeys(rows, pl.String))
    return standardise(raw, _config(**overrides)).collect()


def test_columns_are_standard_and_ordered() -> None:
    frame = _build()
    assert frame.columns == [
        EVENT_ID,
        "user_id",
        "timestamp",
        "login_success",
        "ip",
        "asn",
        "user_agent",
        "region",
        "rtt_ms",
        "label_ato",
        "asn_is_placeholder",
        "ip_is_placeholder",
    ]
    assert "index" not in frame.columns, "unmapped raw columns are dropped"
    assert frame[EVENT_ID].to_list() == [0, 1, 2, 3]


def test_types() -> None:
    schema = _build().schema
    assert schema["user_id"] == pl.Int64
    assert schema["timestamp"] == pl.Datetime("ms")
    assert schema["login_success"] == pl.Boolean
    assert schema["label_ato"] == pl.Boolean
    assert schema["rtt_ms"] == pl.Float64
    assert schema["user_agent"] == pl.String
    assert schema["asn"] == pl.Categorical, "an ASN is an identifier, never a number"
    assert schema["ip"] == pl.Categorical


def test_values_are_parsed() -> None:
    frame = _build()
    assert frame["user_id"][0] == FIRST_USER_ID
    assert frame["timestamp"][0] == dt.datetime(2020, 2, 3, 12, 43, 30, 772000)
    assert frame["login_success"].to_list() == [True, False, True, False]
    assert frame["rtt_ms"].to_list() == [None, 1123.0, 30.0, None]
    assert frame["label_ato"].to_list() == [False, True, False, False]


def test_null_markers_apply_only_to_their_column() -> None:
    frame = _build()
    assert frame["region"].to_list() == [None, "Nan", None, "Vestland"], "'Nan' is a real region"
    assert frame["user_agent"][3] == "-", "'-' is only a null marker where declared"


def test_artefact_flags() -> None:
    frame = _build()
    assert frame["asn_is_placeholder"].to_list() == [False, True, True, False]
    assert frame["ip_is_placeholder"].to_list() == [True, False, False, True], (
        "'100.1.2.3' starts with '10' but is not in 10.0.0.0/8"
    )


def test_artefact_flags_are_never_null() -> None:
    rows = {**ROWS, "IP": ["not-an-ip", "::1", "10.0.0.1", None]}
    frame = _build(rows)
    assert frame["ip_is_placeholder"].to_list() == [False, False, True, False]


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [("Ok", "maybe"), ("When", "03/02/2020"), ("User", "abc"), ("RTT", "fast")],
)
def test_unparseable_values_fail_loudly(column: str, bad_value: str) -> None:
    rows = {**ROWS, column: [bad_value, *ROWS[column][1:]]}
    with pytest.raises(pl.exceptions.PolarsError):
        _build(rows)


def test_string_identifiers_by_default() -> None:
    assert _build(identifier_dtypes={})["user_id"].dtype == pl.String


@pytest.mark.parametrize(
    "overrides",
    [
        {"null_markers": {"country": ["-"]}},  # country is not mapped
        {"identifier_dtypes": {"asn": "Int64"}},  # asn is categorical, not an identifier
        {"artefacts": [{"name": "x", "field": "asn", "kind": "at_least", "value": "big",
                        "meaning": "m"}]},
        {"artefacts": [{"name": "x", "field": "ip", "kind": "in_network", "value": "not/a/net",
                        "meaning": "m"}]},
    ],
)  # fmt: skip
def test_invalid_settings_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _config(**overrides)


def test_build_intermediate_writes_parquet(tmp_path: Path) -> None:
    raw_file = tmp_path / "toy.csv"
    pl.DataFrame(ROWS).write_csv(raw_file)
    out = build_intermediate(_config(), raw_file, tmp_path / "out" / "toy.parquet")
    assert out.is_file()
    assert not list(out.parent.glob("*.part"))
    stored = pl.read_parquet(out)
    assert stored.equals(_build())
