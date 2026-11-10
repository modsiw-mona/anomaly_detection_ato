"""End-to-end tests of the inference pipeline on a toy model and feature table."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pandera.errors import SchemaErrors

from data.datasets import DatasetConfig
from data.drift import drift_report
from data.features import build_features, features_path
from data.split import load_side, write_split
from inference.predict import ALLOW, Detector
from pipelines.inference_pipeline import run as pipeline
from pipelines.training_pipeline import run as training_pipeline
from pipelines.training_pipeline.config import TrainingConfig
from tests.data.test_features import WINDOW, _config, _events
from tests.pipelines.training_pipeline.test_run import TRAINING, _toy_events


@pytest.fixture(scope="module")
def config() -> DatasetConfig:
    """The toy dataset configuration, with the label window set."""
    return _config(**WINDOW)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory, config: DatasetConfig) -> Path:
    """A data directory with features, split and a model saved by the training pipeline."""
    root = tmp_path_factory.mktemp("data")
    path = features_path(config, root)
    path.parent.mkdir(parents=True)
    build_features(_events(_toy_events()), config).collect().write_parquet(path)
    write_split(config, root)
    training_pipeline.run(config, TrainingConfig.model_validate(TRAINING), root)
    return root


@pytest.fixture
def data_dir(trained: Path, tmp_path: Path) -> Path:
    """A fresh copy of the trained directory for each test."""
    for file in trained.rglob("*"):
        if file.is_file():
            target = tmp_path / file.relative_to(trained)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file.read_bytes())
    return tmp_path


def test_scores_and_tiers_every_login_of_the_side(config: DatasetConfig, data_dir: Path) -> None:
    """The run scores every test login like the saved detector and tiers it by its thresholds."""
    report = pipeline.run(config, data_dir)
    assert report["status"] in {"passed", "passed with drift warning"}
    paths = pipeline.output_paths(config, data_dir, "test")
    scores, predictions = pl.read_parquet(paths["scores"]), pl.read_parquet(paths["predictions"])
    assert scores.columns == ["event_id", "score"], (
        "the file notebook 07 wrote and the bundle reads"
    )
    test = load_side(config, data_dir, "test").collect()
    assert scores["event_id"].equals(test["event_id"])

    detector = Detector.load(data_dir, config.name)
    expected = detector.score(test)
    np.testing.assert_allclose(scores["score"].to_numpy(), expected, rtol=0, atol=1e-12)
    assert list(predictions["tier"]) == [detector.tier(s) for s in predictions["score"]]

    tiers = report["tiers"]
    assert sum(t["logins"] for t in tiers.values()) == report["rows"] == test.height
    assert set(tiers) == {*detector.thresholds, ALLOW}
    assert report["model"]["joblib_sha256"] == detector.metadata["joblib_sha256"]
    assert json.loads(pipeline.report_path(config, data_dir).read_text())["rows"] == test.height


def test_labels_are_never_used(config: DatasetConfig, data_dir: Path) -> None:
    """Flipping the takeover labels changes neither the scores nor the tiers."""
    pipeline.run(config, data_dir)
    paths = pipeline.output_paths(config, data_dir, "test")
    before = pl.read_parquet(paths["predictions"])
    path = features_path(config, data_dir)
    pl.read_parquet(path).with_columns(~pl.col("label_ato")).write_parquet(path)
    pipeline.run(config, data_dir)
    after = pl.read_parquet(paths["predictions"])
    assert after["tier"].equals(before["tier"])
    np.testing.assert_allclose(after["score"], before["score"], rtol=0, atol=1e-12)


def test_an_invalid_batch_is_not_scored_and_no_earlier_scores_stay(
    config: DatasetConfig, data_dir: Path
) -> None:
    """A batch that fails validation is not scored, and an earlier run's outputs are removed."""
    pipeline.run(config, data_dir)  # an earlier, successful run
    assert pipeline.output_paths(config, data_dir, "test")["scores"].is_file()
    path = features_path(config, data_dir)
    pl.read_parquet(path).with_columns(pl.lit(99, pl.Int8).alias("hour")).write_parquet(path)
    with pytest.raises(SchemaErrors):
        pipeline.run(config, data_dir)
    assert not any(p.exists() for p in pipeline.output_paths(config, data_dir, "test").values())
    report = json.loads(pipeline.report_path(config, data_dir).read_text())
    assert report["status"] == "failed"
    assert report["error"].startswith("SchemaErrors")


def test_drift_is_reported_but_the_batch_is_still_scored(
    config: DatasetConfig, data_dir: Path
) -> None:
    """Drift on the model's inputs is reported as a warning, and the batch is still scored."""
    report = pipeline.run(config, data_dir, drift_limit=1e-9)
    assert report["status"] == "passed with drift warning"
    assert report["drift"]["features_over_the_limit"]
    assert report["drift"]["largest_score"] > 0, "test against development, not against itself"
    assert pipeline.output_paths(config, data_dir, "test")["scores"].is_file()

    columns = Detector.load(data_dir, config.name).columns
    expected = drift_report(
        load_side(config, data_dir, "dev").select(columns).collect(),
        load_side(config, data_dir, "test").select(columns).collect(),
        columns,
        limit=1e-9,
    )
    assert [row["column"] for row in report["drift"]["table"]] == columns, "the model's inputs"
    assert report["drift"]["table"] == expected.to_dicts()


def test_a_model_that_does_not_match_its_metadata_is_refused(
    config: DatasetConfig, data_dir: Path
) -> None:
    """The run refuses a model file whose SHA-256 differs from its metadata."""
    model = data_dir / "06_models" / f"{config.name}_model.joblib"
    model.write_bytes(model.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        pipeline.run(config, data_dir)


def test_the_development_side_is_marked_in_sample(config: DatasetConfig, data_dir: Path) -> None:
    """Scoring the development side marks the report in-sample, with no drift."""
    report = pipeline.run(config, data_dir, side="dev")
    assert report["in_sample"] is True
    assert report["drift"]["largest_score"] == 0


def test_an_empty_batch_is_refused(
    config: DatasetConfig, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run refuses a side with no logins."""
    real = load_side

    def empty_test(cfg: DatasetConfig, folder: Path, side: str) -> pl.LazyFrame:
        frame = real(cfg, folder, side)
        return frame.head(0) if side == "test" else frame

    monkeypatch.setattr(pipeline, "load_side", empty_test)
    with pytest.raises(ValueError, match="no logins"):
        pipeline.run(config, data_dir)


def test_unknown_side(config: DatasetConfig, data_dir: Path) -> None:
    """The run refuses a side other than development or test."""
    with pytest.raises(ValueError, match="side"):
        pipeline.run(config, data_dir, side="holdout")


@pytest.mark.parametrize("fails", [False, True])
def test_main_exit_code(monkeypatch: pytest.MonkeyPatch, *, fails: bool) -> None:
    """``main`` exits with 0 on success and 1 on a schema failure."""

    def fake_run(*_: object, **__: object) -> dict[str, object]:
        if fails:
            error = SchemaErrors.__new__(SchemaErrors)
            error.failure_cases = pl.DataFrame({"check": ["c"], "column": ["x"]})
            raise error
        return {
            "rows": 10,
            "status": "passed",
            "tiers": {"step-up": {"share_with_stricter": 0.01}, "allow": {}},
        }

    monkeypatch.setattr(pipeline, "run", fake_run)
    assert pipeline.main([]) == int(fails)
