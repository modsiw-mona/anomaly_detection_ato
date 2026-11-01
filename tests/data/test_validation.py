"""Tests for the data validation rules of the intermediate dataset."""

import warnings
from typing import Any

import polars as pl
import pytest
from pandera.errors import SchemaErrors, SchemaWarning

from data.datasets import DatasetConfig, load_dataset_config
from data.intermediate import standardise
from data.validation import build_schema, validate_intermediate
from tests.data.test_intermediate import CONFIG, ROWS

RULES: dict[str, Any] = {
    "timestamp_range": ["2020-02-01", "2021-03-01"],
    "unique_together": ["user_id", "timestamp"],
    "allowed_values": {"region": ["Nan", "Vestland"]},
    "patterns": {"asn": "^[0-9]+$"},
    "numeric_ranges": {"rtt_ms": [1, 8_600_000]},
}
# Row 1 of ROWS is a takeover on a failed login; this version keeps every rule satisfied.
Rows = dict[str, list[str | None]]
CLEAN_ROWS: Rows = {**ROWS, "ATO": ["False", "False", "True", "False"]}


def _config() -> DatasetConfig:
    return DatasetConfig.model_validate({**CONFIG, "validation": RULES})


def _frame(rows: Rows = CLEAN_ROWS) -> pl.DataFrame:
    raw = pl.LazyFrame(rows, schema=dict.fromkeys(rows, pl.String))
    return standardise(raw, _config()).collect()


def _failed_checks(frame: pl.DataFrame) -> set[str]:
    with pytest.raises(SchemaErrors) as caught:
        validate_intermediate(frame, _config())
    return set(caught.value.failure_cases["check"].to_list())


def test_valid_frame_passes_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = validate_intermediate(_frame(), _config())
    assert out.height == len(CLEAN_ROWS["User"])


def test_takeover_on_failed_login_is_a_warning_not_an_error() -> None:
    with pytest.warns(SchemaWarning, match="takeover_on_successful_login"):
        validate_intermediate(_frame(ROWS), _config())


@pytest.mark.parametrize(
    ("column", "value", "check"),
    [
        ("When", "2019-12-31 23:59:59.000", "timestamp_range"),
        ("Region", "Atlantis", "allowed_values"),
        ("ASN", "AS123", "pattern"),
        ("RTT", "-5", "numeric_range"),
    ],
)
def test_dataset_rules_catch_their_violation(column: str, value: str, check: str) -> None:
    rows: Rows = {**CLEAN_ROWS, column: [value, *CLEAN_ROWS[column][1:]]}
    assert check in _failed_checks(_frame(rows))


def test_duplicate_user_and_timestamp_is_an_error() -> None:
    rows: Rows = {
        **CLEAN_ROWS,
        "User": ["42", "42", "42", "7"],
        "When": [CLEAN_ROWS["When"][1]] * 4,
    }
    assert any("unique" in check for check in _failed_checks(_frame(rows)))


def test_required_field_cannot_be_null() -> None:
    frame = _frame().with_columns(
        pl.when(pl.col("event_id") == 0).then(None).otherwise(pl.col("user_id")).alias("user_id")
    )
    assert "not_nullable" in _failed_checks(frame)


def test_wrong_storage_type_is_an_error() -> None:
    frame = _frame().with_columns(pl.col("rtt_ms").cast(pl.Int64))
    assert any(check.startswith("dtype") for check in _failed_checks(frame))


def test_unexpected_column_is_an_error() -> None:
    assert "column_in_schema" in _failed_checks(_frame().with_columns(pl.lit(1).alias("extra")))


def test_rba_schema_covers_every_intermediate_column() -> None:
    schema = build_schema(load_dataset_config("rba"))
    assert set(schema.columns) == {
        "event_id", "user_id", "timestamp", "login_success", "ip", "asn", "device_type",
        "user_agent", "country", "region", "city", "os", "browser", "rtt_ms", "label_ato",
        "label_attack_ip", "asn_is_placeholder", "ip_is_placeholder",
    }  # fmt: skip
