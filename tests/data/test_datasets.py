"""Tests for dataset configuration loading and validation."""

import copy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from data.datasets import DatasetConfig, find_repo_root, load_dataset_config

VALID: dict[str, Any] = {
    "name": "toy",
    "description": "Toy dataset for tests.",
    "source": {
        "url": "https://example.org/toy.zip",
        "archive": "toy.zip",
        "md5": "0" * 32,
        "member": "toy/toy.csv",
        "max_uncompressed_bytes": 1_000,
        "doi": "10.0/toy",
        "version": "v1",
        "license": "CC0-1.0",
        "citation": "Toy (2026).",
        "homepage": "https://example.org/toy",
    },
    "columns": {
        "user_id": "User",
        "timestamp": "When",
        "login_success": "Ok",
        "asn": "ASN",
    },
    "labels": {"label_ato": "Is ATO"},
}


def _with(path: tuple[str, ...], value: Any) -> dict[str, Any]:
    """Return a deep copy of VALID with the nested key at ``path`` replaced (or removed)."""
    data = copy.deepcopy(VALID)
    node = data
    for key in path[:-1]:
        node = node[key]
    if value is None:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return data


def test_valid_config_loads() -> None:
    config = DatasetConfig.model_validate(VALID)
    assert config.rename_map() == {
        "User": "user_id",
        "When": "timestamp",
        "Ok": "login_success",
        "ASN": "asn",
        "Is ATO": "label_ato",
    }


def test_rba_config_in_repo_is_valid() -> None:
    config = load_dataset_config("rba")
    assert config.name == "rba"
    assert config.source.url.scheme == "https"
    assert {"user_id", "timestamp", "login_success"} <= config.columns.keys()
    assert {"label_ato", "label_attack_ip"} == config.labels.keys()
    assert {artefact.name for artefact in config.artefacts} == {
        "asn_is_placeholder",
        "ip_is_placeholder",
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("columns", "user_id"), None),  # required field missing
        (("columns",), {"user_id": "U", "timestamp": "T", "login_success": "S"}),  # no context
        (("columns", "label_ato"), "Is ATO"),  # label mapped as a column
        (("labels", "asn"), "ASN"),  # non-label under [labels]
        (("source", "url"), "http://example.org/toy.zip"),  # not https
        (("source", "md5"), "not-a-checksum"),
        (("source", "archive"), "../toy.zip"),  # a path, not a file name
        (("unexpected_key",), "typo"),  # unknown keys are rejected
    ],
)
def test_invalid_configs_are_rejected(path: tuple[str, ...], value: Any) -> None:
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate(_with(path, value))


@pytest.mark.parametrize("name", ["../rba", "RBA", "", "rba/../x"])
def test_dataset_name_is_validated(name: str) -> None:
    with pytest.raises(ValueError, match="Invalid dataset name"):
        load_dataset_config(name)


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset_config("absent", conf_dir=tmp_path)


def test_find_repo_root_from_nested_directory() -> None:
    root = find_repo_root()
    assert (root / "pyproject.toml").is_file()
    assert find_repo_root(root / "src" / "data") == root
