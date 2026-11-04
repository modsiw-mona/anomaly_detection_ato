"""The Streamlit demo runs end to end on a toy data directory (no real data needed)."""

import json
from pathlib import Path

import joblib
import polars as pl
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from data.features import DEFAULT_GROUPS, build_features, feature_columns
from model.preprocessing import to_model_frame
from model.selection import train_supervised
from tests.data.test_features import _config, _events, _random_events

APP = Path(__file__).parents[2] / "notebooks" / "7-deploy" / "app.py"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A data directory shaped like the real one, for dataset ``rba``, built from toy events."""
    root = tmp_path_factory.mktemp("data")
    frame = build_features(_events(_random_events(1_500, seed=5)), _config()).collect()
    # takeovers = logins from an attack IP, so the "IP on an attack list" what-if must move the score
    frame = frame.with_columns(pl.col("attack_ip").alias("label_ato"))
    columns = feature_columns((*DEFAULT_GROUPS, "campaign", "travel"))
    pipeline = train_supervised("random forest", frame, columns, negative_share=0.5)
    for folder in ("04_feature", "05_model_input", "06_models", "07_model_output"):
        (root / folder).mkdir()
    frame.write_parquet(root / "04_feature" / "rba_features.parquet")
    users = frame.select(pl.col("user_id").unique()).with_columns(pl.lit("test").alias("split"))
    users.write_parquet(root / "05_model_input" / "rba_split.parquet")
    joblib.dump(pipeline, root / "06_models" / "rba_model.joblib")
    (root / "06_models" / "rba_model.json").write_text(json.dumps({"columns": columns}))
    scores = pipeline.predict_proba(to_model_frame(frame, columns))[:, 1]
    for name in ("dev_oof", "test_scores"):
        pl.DataFrame({"event_id": frame["event_id"], "score": scores}).write_parquet(
            root / "07_model_output" / f"rba_{name}.parquet"
        )
    return root


def test_demo_scores_a_login_and_reacts_to_what_if(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ATO_DATA_DIR", str(data_dir))
    st.cache_resource.clear()
    st.cache_data.clear()
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception, app.exception
    assert app.metric[0].label == "Takeover score"
    before = app.metric[0].value
    attack = next(t for t in app.toggle if t.label == "IP address on an attack list")
    attack.set_value(not attack.value).run()
    assert not app.exception, app.exception
    assert app.metric[0].value != before, "the what-if change reaches the model"
    assert app.metric[0].delta is not None
    assert app.metric[0].delta.lstrip("+-") != "0.0000 vs the real login"
    labels = {t.label for t in app.toggle}
    assert "First successful login of this account" not in labels, (
        "a fact, shown in the table, not a switch"
    )


def test_demo_explains_missing_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATO_DATA_DIR", str(tmp_path))
    st.cache_resource.clear()  # the model loaded by the other test is cached per process
    st.cache_data.clear()
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert app.error, "a clear message instead of a crash"
