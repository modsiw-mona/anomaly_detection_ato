"""End-to-end tests of the feature pipeline on a toy raw dataset in a temporary data directory."""

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from pandera.errors import SchemaErrors

from data.datasets import DatasetConfig
from data.features import features_path
from data.split import split_path
from pipelines.feature_pipeline import run as pipeline
from tests.data.test_features import ALL_COLUMNS, WINDOW, _config

N_USERS, LOGINS, TAKEOVER_USERS = 300, 4, 30
STEPS = ["raw data", "intermediate dataset", "feature table", "split"]


def _raw_csv(path: Path, seed: int = 0) -> Path:
    """A raw file in the source's own column names, every value as text."""
    rng = np.random.default_rng(seed)
    start = dt.datetime(2020, 3, 1)
    rows: list[dict[str, Any]] = []
    for user in range(N_USERS):
        for i in range(LOGINS):
            when = start + dt.timedelta(minutes=int(rng.integers(0, 200_000)))
            rows.append(
                {
                    "U": str(user + 1),
                    "T": when.strftime("%Y-%m-%d %H:%M:%S.000"),
                    "S": "True" if rng.random() < 0.9 else "False",  # noqa: PLR2004
                    "IP": f"10.0.{rng.integers(0, 4)}.{rng.integers(1, 9)}",
                    "ASN": str(rng.choice(["100", "200", "300"])),
                    "D": str(rng.choice(["mobile", "desktop"])),
                    "UA": str(rng.choice(["UA1", "UA2", "UA3"])),
                    "C": str(rng.choice(["NO", "SE", "US"])),
                    "R": "R1",
                    "CI": "C1",
                    "OS": str(rng.choice(["iOS", "Android"])),
                    "B": str(rng.choice(["Safari", "Chrome"])),
                    "RTT": str(rng.integers(10, 500)),
                    "ATO": "True" if user < TAKEOVER_USERS and i == LOGINS - 1 else "False",
                    "AIP": "True" if rng.random() < 0.05 else "False",  # noqa: PLR2004
                }
            )
    assert set(ALL_COLUMNS.values()) <= rows[0].keys()
    pl.DataFrame(rows).write_csv(path)
    return path


@pytest.fixture
def config() -> DatasetConfig:
    """The toy dataset configuration, with the label window set."""
    return _config(**WINDOW)


@pytest.fixture
def data_dir(tmp_path: Path, config: DatasetConfig, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary data directory where the download step writes a toy raw CSV."""

    def download(cfg: DatasetConfig, raw_dir: Path) -> Path:
        raw_dir.mkdir(parents=True, exist_ok=True)
        return _raw_csv(raw_dir / Path(cfg.source.member).name)

    monkeypatch.setattr(pipeline, "download_dataset", download)
    return tmp_path


def _report(config: DatasetConfig, data_dir: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(pipeline.report_path(config, data_dir).read_text())
    return loaded


def _no_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object) -> None:
        raise AssertionError("validate-only must not download or rebuild")

    for name in ("download_dataset", "build_intermediate", "write_features", "write_split"):
        monkeypatch.setattr(pipeline, name, fail)


def test_pipeline_builds_and_validates_every_stage(config: DatasetConfig, data_dir: Path) -> None:
    """A run builds and validates every stage, then reports the steps and the drift table."""
    report = pipeline.run(config, data_dir, drift_limit=1.01)
    assert report["status"] == "passed"
    assert [s["step"] for s in report["steps"]] == [*STEPS, "drift between development and test"]
    assert features_path(config, data_dir).is_file()
    assert split_path(config, data_dir).is_file()
    assert _report(config, data_dir) == report
    split = next(s for s in report["steps"] if s["step"] == "split")
    assert split["users"] == pl.read_parquet(features_path(config, data_dir))["user_id"].n_unique()
    assert {d["column"] for d in report["drift"]} >= {"prior_successes", "new_country", "attack_ip"}


def test_validate_only_checks_existing_outputs_without_rebuilding(
    config: DatasetConfig, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``validate_only`` checks the existing outputs without downloading or rebuilding them."""
    pipeline.run(config, data_dir, drift_limit=1.01)
    _no_rebuild(monkeypatch)
    report = pipeline.run(config, data_dir, validate_only=True, drift_limit=1.01)
    assert report["status"] == "passed"
    assert report["steps"][0]["step"] == "intermediate dataset"


def test_a_broken_output_stops_the_pipeline_and_the_report_says_where(
    config: DatasetConfig, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An output that breaks its schema stops the run, and the report names the step."""
    pipeline.run(config, data_dir, drift_limit=1.01)
    path = features_path(config, data_dir)
    pl.read_parquet(path).with_columns(pl.lit(99, pl.Int8).alias("hour")).write_parquet(path)
    _no_rebuild(monkeypatch)
    with pytest.raises(SchemaErrors):
        pipeline.run(config, data_dir, validate_only=True, drift_limit=1.01)
    report = _report(config, data_dir)
    assert report["status"] == "failed"
    assert [s["step"] for s in report["steps"]] == ["intermediate dataset"]
    assert report["failed_step"] == "feature table"
    assert "in_range(0, 23)" in report["error"]


def test_a_rebuild_that_fails_its_checks_keeps_the_previous_output(
    config: DatasetConfig, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rebuild that fails validation keeps the previous output and leaves no partial file."""
    pipeline.run(config, data_dir, drift_limit=1.01)
    path = features_path(config, data_dir)
    before = pl.read_parquet(path)

    def reject(*_: object) -> pl.DataFrame:
        raise ValueError("rejected")

    monkeypatch.setattr(pipeline, "validate_features", reject)
    with pytest.raises(ValueError, match="rejected"):
        pipeline.run(config, data_dir, drift_limit=1.01)
    assert pl.read_parquet(path).equals(before)
    assert not path.with_name(path.name + ".part").exists()
    report = _report(config, data_dir)
    assert report["failed_step"] == "feature table"
    assert report["error"] == "ValueError: rejected"


def test_drift_over_the_limit_stops_the_pipeline(config: DatasetConfig, data_dir: Path) -> None:
    """Drift over the limit stops the run and keeps the drift table in the report."""
    with pytest.raises(pipeline.DriftError, match="drift"):
        pipeline.run(config, data_dir, drift_limit=0.0)
    report = _report(config, data_dir)
    assert report["status"] == "failed"
    assert report["drift"], "the drift table is kept to see which features moved"


@pytest.mark.parametrize(
    ("error", "code"),
    [(None, 0), (pipeline.DriftError("x"), 1), (SchemaErrors.__new__(SchemaErrors), 1)],
)
def test_main_exit_code(
    monkeypatch: pytest.MonkeyPatch, error: Exception | None, code: int
) -> None:
    """``main`` exits with 0 on success and 1 on a drift or schema failure."""

    def fake_run(*_: object, **__: object) -> dict[str, Any]:
        if error is not None:
            raise error
        return {}

    monkeypatch.setattr(pipeline, "run", fake_run)
    if isinstance(error, SchemaErrors):
        error.failure_cases = pl.DataFrame({"check": ["c"], "column": ["x"]})
    assert pipeline.main(["--validate-only"]) == code
