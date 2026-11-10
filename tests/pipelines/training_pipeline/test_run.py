"""End-to-end tests of the training pipeline on a toy feature table in a temporary data directory."""

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from data.datasets import DatasetConfig
from data.features import build_features, features_path
from data.split import load_side, write_split
from inference.predict import Detector, alert_thresholds
from model.validation import ModelValidationError
from pipelines.training_pipeline import run as pipeline
from pipelines.training_pipeline.config import TrainingConfig
from pipelines.training_pipeline.run import cross_validation_summary
from tests.data.test_features import WINDOW, _config, _events

N_USERS, LOGINS, TAKEOVER_USERS, FIRST_LOGIN_TAKEOVERS = 300, 4, 45, 15
TRAINING: dict[str, Any] = {
    "name": "toy forest",
    "model": "random forest",
    "feature_groups": ["history", "user_novelty", "global_rarity", "sharing", "failures",
                       "attack_ip", "missingness", "device", "time", "campaign", "travel"],
    "balance_campaign": True,
    "negative_share": 0.5,
    "seed": 1,
    "folds": {"n_folds": 3, "repeats": 1, "seed": 1},
    "budgets": {"review / block": 0.01, "step-up": 0.1},
    "baseline": {"model": "logistic regression",
                 "feature_groups": ["history", "global_rarity", "attack_ip"]},
    "segments": {"column": "country", "value": "RO", "name": "Romania"},
    "gates": {"min_interval_low": 0.0, "min_recall_at_1pct": 0.0, "min_gain_over_baseline": 0.0,
              "alert_rate_ratio": [0.01, 100.0], "min_segment_recall_at_1pct": 0.0,
              "max_seconds_per_login_in_batch": 1.0, "max_seconds_single_login": 10.0},
}  # fmt: skip


def _toy_events(seed: int = 0) -> list[dict[str, Any]]:
    """Logins where a takeover comes from a rare country and network on an attack-listed IP."""
    rng = np.random.default_rng(seed)
    start = dt.datetime(2020, 3, 1)
    events: list[dict[str, Any]] = []
    for user in range(N_USERS):
        times = sorted(int(m) for m in rng.integers(0, 200_000, size=LOGINS))
        for i, minute in enumerate(times):
            takeover = user < TAKEOVER_USERS and i == (
                0 if user < FIRST_LOGIN_TAKEOVERS else LOGINS - 1
            )
            events.append(
                {
                    "user_id": user + 1,
                    "timestamp": start + dt.timedelta(minutes=minute),
                    "login_success": True,
                    "country": "RO" if takeover and user % 3 else str(rng.choice(["NO", "SE"])),
                    "asn": "999" if takeover else str(rng.choice(["100", "200"])),
                    "ip": f"9.9.9.{user}" if takeover else f"1.1.{user}.1",
                    "label_ato": takeover,
                    "label_attack_ip": takeover,
                }
            )
    events.sort(key=lambda e: str(e["timestamp"]))
    return events


@pytest.fixture
def config() -> DatasetConfig:
    """The toy dataset configuration, with the label window set."""
    return _config(**WINDOW)


@pytest.fixture
def training() -> TrainingConfig:
    """The toy training configuration, validated."""
    return TrainingConfig.model_validate(TRAINING)


@pytest.fixture
def data_dir(tmp_path: Path, config: DatasetConfig) -> Path:
    """A temporary data directory with the toy feature table and its split."""
    path = features_path(config, tmp_path)
    path.parent.mkdir(parents=True)
    build_features(_events(_toy_events()), config).collect().write_parquet(path)
    write_split(config, tmp_path)
    return tmp_path


def _sha(path: Path) -> str:
    return pipeline.sha256_of(path)


def test_trains_validates_and_saves_the_files_the_inference_module_reads(
    config: DatasetConfig, training: TrainingConfig, data_dir: Path
) -> None:
    """A run saves the model, metadata and out-of-fold scores the detector reads, and a record."""
    record = pipeline.run(config, training, data_dir)
    assert record["status"] == "passed"
    paths = pipeline.model_paths(config, data_dir)
    assert all(p.is_file() for p in paths.values())
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["joblib_sha256"] == _sha(paths["model"]) == record["model"]["joblib_sha256"]
    assert set(metadata["thresholds"]) == set(training.budgets)
    oof = pl.read_parquet(paths["out_of_fold"])
    assert oof.height == record["data"]["development_logins"]
    assert oof["event_id"].is_sorted()

    detector = Detector.from_files(paths["model"].parent, config.name, metadata["thresholds"])
    dev = pl.read_parquet(features_path(config, data_dir)).filter(
        pl.col("event_id").is_in(oof["event_id"].implode())
    )
    assert detector.score(dev).shape == (dev.height,)

    names = {g["name"] for g in record["gates"]}
    assert {"pr_auc_interval_low", "gain_over_baseline", "seconds_single_login"} <= names
    assert {"segment_recall_at_1pct[Romania]", "segment_recall_at_1pct[first login]"} <= names
    assert record["data"]["features_sha256"] == _sha(features_path(config, data_dir))
    assert record["model"]["same_as_previous"] is None, "no earlier model"
    runs = list((data_dir / "08_reporting" / "training_runs").glob("*.json"))
    assert len(runs) == 1
    assert json.loads(pipeline.report_path(config, data_dir).read_text())["status"] == "passed"


def test_the_same_configuration_gives_the_same_model(
    config: DatasetConfig, training: TrainingConfig, data_dir: Path
) -> None:
    """Training twice with the same configuration gives a byte-identical model."""
    first = pipeline.run(config, training, data_dir)
    second = pipeline.run(config, training, data_dir)
    assert second["model"]["same_as_previous"] is True
    assert second["model"]["joblib_sha256"] == first["model"]["joblib_sha256"]
    # the forest averages its trees in parallel, so scores may differ in the last bit (~1e-16)
    assert second["thresholds"] == pytest.approx(first["thresholds"], rel=1e-12)


def test_a_failing_gate_saves_nothing_and_keeps_the_previous_model(
    config: DatasetConfig, training: TrainingConfig, data_dir: Path
) -> None:
    """A failed validation gate saves nothing, keeps the previous model and reports the gate."""
    pipeline.run(config, training, data_dir)
    paths = pipeline.model_paths(config, data_dir)
    before = {name: _sha(path) for name, path in paths.items()}
    strict = training.model_copy(
        update={"gates": training.gates.model_copy(update={"max_seconds_single_login": 1e-12})}
    )
    with pytest.raises(ModelValidationError, match="seconds_single_login"):
        pipeline.run(config, strict, data_dir)
    assert {name: _sha(path) for name, path in paths.items()} == before
    record = json.loads(pipeline.report_path(config, data_dir).read_text())
    assert record["status"] == "failed"
    assert any(g["name"] == "seconds_single_login" and not g["passed"] for g in record["gates"])
    assert record["error"].startswith("ModelValidationError")


def test_the_test_side_is_never_read(
    config: DatasetConfig,
    training: TrainingConfig,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training reads only the development side of the split."""
    real = load_side
    sides: list[str] = []

    def spy(cfg: DatasetConfig, folder: Path, side: str) -> pl.LazyFrame:
        sides.append(side)
        return real(cfg, folder, side)

    monkeypatch.setattr(pipeline, "load_side", spy)
    pipeline.run(config, training, data_dir)
    assert sides == ["dev"]


def test_more_repeats_refine_the_estimate_but_not_the_thresholds(
    config: DatasetConfig, training: TrainingConfig, data_dir: Path
) -> None:
    """More repeats add folds but change neither the thresholds nor the model."""
    one = pipeline.run(config, training, data_dir)
    three = pipeline.run(
        config,
        training.model_copy(update={"folds": training.folds.model_copy(update={"repeats": 3})}),
        data_dir,
    )
    assert (one["cross_validation"]["folds"], three["cross_validation"]["folds"]) == (3, 9)
    assert three["thresholds"] == pytest.approx(one["thresholds"], rel=1e-12), "repeat 0 sets them"
    assert three["model"]["same_as_previous"] is True


@pytest.mark.parametrize("fails", [False, True])
def test_main_exit_code(monkeypatch: pytest.MonkeyPatch, *, fails: bool) -> None:
    """``main`` exits with 0 on success and 1 when a validation gate fails."""

    def fake_run(*_: object) -> dict[str, Any]:
        if fails:
            raise ModelValidationError([])
        return {
            "cross_validation": {"pr_auc (5 folds)": 0.5, "folds": 5},
            "model": {"same_as_previous": True},
        }

    monkeypatch.setattr(pipeline, "run", fake_run)
    assert pipeline.main(["--repeats", "1"]) == int(fails)


NOTEBOOK_06_KEYS = {
    "dataset", "selected", "model", "campaign_balanced", "columns", "negative_share",
    "trained_on", "joblib_sha256", "joblib_megabytes", "cv_15_folds",
}  # fmt: skip
NOTEBOOK_06_CV_KEYS = {
    "pr_auc (15 folds)", "std within repeat", "std across repeats", "recall_0_1pct",
    "recall_1pct", "precision_1pct", "pr_auc_romania", "pr_auc_other", "recall_1pct_romania",
    "recall_1pct_other", "pooled_low", "pooled_high",
}  # fmt: skip


def test_fifteen_folds_write_the_metadata_notebooks_07_and_08_read(
    config: DatasetConfig, training: TrainingConfig, data_dir: Path
) -> None:
    """Five folds by three repeats write the metadata keys that notebooks 07 and 08 read."""
    fifteen = training.model_copy(
        update={"folds": training.folds.model_copy(update={"n_folds": 5, "repeats": 3})}
    )
    pipeline.run(config, fifteen, data_dir)
    metadata = json.loads(pipeline.model_paths(config, data_dir)["metadata"].read_text())
    assert set(metadata) >= NOTEBOOK_06_KEYS
    assert set(metadata["cv_15_folds"]) == NOTEBOOK_06_CV_KEYS
    assert metadata["selected"] == "toy forest"


def test_each_gate_measures_what_it_names(
    config: DatasetConfig, training: TrainingConfig, data_dir: Path
) -> None:
    """The interval, recall, gain and alert-rate gates match values recomputed from the files."""
    record = pipeline.run(config, training, data_dir)
    gate = {g["name"]: g["value"] for g in record["gates"]}
    rows = record["repeats"]
    assert gate["pr_auc_interval_low"] == round(np.mean([r["pooled_low"] for r in rows]), 4)
    assert gate["recall_at_1pct"] == round(np.mean([r["recall_1pct"] for r in rows]), 4)
    gain = rows[0]["pr_auc"] / record["baseline_repeat_0"]["pr_auc"] - 1
    assert gate["gain_over_baseline"] == pytest.approx(gain, abs=1e-4)

    paths = pipeline.model_paths(config, data_dir)
    metadata = json.loads(paths["metadata"].read_text())
    detector = Detector.from_files(paths["model"].parent, config.name, metadata["thresholds"])
    oof = pl.read_parquet(paths["out_of_fold"])
    dev = (
        pl.read_parquet(features_path(config, data_dir))
        .join(oof.select("event_id"), on="event_id", how="semi")
        .sort("event_id")
    )
    saved_model_scores = detector.score(dev)  # the final model, not the out-of-fold scores
    for tier, threshold in metadata["thresholds"].items():
        rate = float(np.mean(saved_model_scores >= threshold))
        assert gate[f"alert_rate[{tier}]"] == pytest.approx(rate, abs=1e-6)
    oof_rate = float(np.mean(oof["score"].to_numpy() >= metadata["thresholds"]["step-up"]))
    assert metadata["thresholds"] == pytest.approx(
        alert_thresholds(oof["score"].to_numpy(), training.budgets)
    )
    assert oof_rate == pytest.approx(training.budgets["step-up"], abs=0.01)


def test_a_failure_while_saving_keeps_the_previous_files(
    config: DatasetConfig,
    training: TrainingConfig,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write error while saving keeps the previous model files and leaves no partial file."""
    pipeline.run(config, training, data_dir)
    paths = pipeline.model_paths(config, data_dir)
    before = {name: _sha(path) for name, path in paths.items()}

    def disk_full(*_: object, **__: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", disk_full)
    with pytest.raises(OSError, match="disk full"):
        pipeline.run(config, training, data_dir)
    assert {name: _sha(path) for name, path in paths.items()} == before
    assert not [p for path in paths.values() for p in path.parent.glob("*.part")]


def test_test_side_rows_do_not_change_the_model(
    config: DatasetConfig, training: TrainingConfig, data_dir: Path
) -> None:
    """Changing test-side rows, labels included, leaves the trained model byte-identical."""
    first = pipeline.run(config, training, data_dir)
    path = features_path(config, data_dir)
    split = pl.read_parquet(data_dir / "05_model_input" / f"{config.name}_split.parquet")
    test_users = split.filter(pl.col("split") == "test")["user_id"].implode()
    garbage = pl.read_parquet(path).with_columns(
        pl.when(pl.col("user_id").is_in(test_users))
        .then(pl.lit(1.0))
        .otherwise(pl.col("global_share_asn"))
        .alias("global_share_asn"),
        pl.when(pl.col("user_id").is_in(test_users))
        .then(~pl.col("label_ato"))
        .otherwise(pl.col("label_ato"))
        .alias("label_ato"),
    )
    garbage.write_parquet(path)
    second = pipeline.run(config, training, data_dir)
    assert second["data"]["features_sha256"] != first["data"]["features_sha256"]
    assert second["model"]["joblib_sha256"] == first["model"]["joblib_sha256"]


def test_a_configuration_that_does_not_fit_the_data_fails_before_training(
    config: DatasetConfig, training: TrainingConfig, data_dir: Path
) -> None:
    """A segment column missing from the data stops the run before any training."""
    wrong = training.model_copy(
        update={"segments": training.segments.model_copy(update={"column": "campaign_name"})}
    )
    with pytest.raises(pipeline.ConfigurationError, match="campaign_name"):
        pipeline.run(config, wrong, data_dir)
    record = json.loads(pipeline.report_path(config, data_dir).read_text())
    assert record["seconds"].get("out_of_fold") is None, "stopped before any training"


def test_cross_validation_summary_averages_repeats_like_notebook_06() -> None:
    """``cross_validation_summary`` averages repeats; one repeat has no spread across repeats."""
    rows = [
        {"pr_auc": 0.4, "pr_auc_std": 0.1, "recall_1pct": 0.9, "pooled_low": 0.30001},
        {"pr_auc": 0.5, "pr_auc_std": 0.2, "recall_1pct": 0.8, "pooled_low": 0.4},
    ]
    summary = cross_validation_summary(rows, 10)
    assert summary == {
        "pr_auc (10 folds)": 0.45,
        "std within repeat": 0.15,
        "std across repeats": round(float(np.std([0.4, 0.5], ddof=1)), 4),
        "recall_1pct": 0.85,
        "pooled_low": 0.35,
    }
    assert cross_validation_summary(rows[:1], 5)["std across repeats"] is None


@pytest.mark.parametrize("repeats", ["0", "-1"])
def test_repeats_must_be_positive(repeats: str) -> None:
    """``main`` rejects a repeat count below 1."""
    with pytest.raises(SystemExit):
        pipeline.main(["--repeats", repeats])
