"""Shared fixtures: a toy data directory shaped like the real one (no real data needed)."""

import json
from pathlib import Path

import joblib
import polars as pl
import pytest

from data.features import DEFAULT_GROUPS, build_features, feature_columns
from model.preprocessing import to_model_frame
from model.selection import train_supervised
from tests.data.test_features import _config, _events, _random_events


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A data directory shaped like the real one, for dataset ``rba``, built from toy events."""
    root = tmp_path_factory.mktemp("data")
    frame = build_features(_events(_random_events(1_500, seed=5)), _config()).collect()
    # takeovers = logins from an attack IP, so the "IP on an attack list" switch must move the score
    frame = frame.with_columns(pl.col("attack_ip").alias("label_ato"))
    columns = feature_columns((*DEFAULT_GROUPS, "campaign", "travel"))
    pipeline = train_supervised("random forest", frame, columns, negative_share=0.5)
    for folder in ("04_feature", "05_model_input", "06_models", "07_model_output"):
        (root / folder).mkdir()
    frame.write_parquet(root / "04_feature" / "rba_features.parquet")
    users = frame.select(pl.col("user_id").unique().sort())
    side = pl.when(pl.int_range(pl.len()) % 2 == 0).then(pl.lit("test")).otherwise(pl.lit("dev"))
    users.with_columns(side.alias("split")).write_parquet(
        root / "05_model_input" / "rba_split.parquet"
    )  # half the users on each side
    joblib.dump(pipeline, root / "06_models" / "rba_model.joblib")
    (root / "06_models" / "rba_model.json").write_text(json.dumps({"columns": columns}))
    scores = pipeline.predict_proba(to_model_frame(frame, columns))[:, 1]
    for name in ("dev_oof", "test_scores"):
        pl.DataFrame({"event_id": frame["event_id"], "score": scores}).write_parquet(
            root / "07_model_output" / f"rba_{name}.parquet"
        )
    return root
