"""Tests for the inference module: thresholds, tiers, saving/loading and explanations."""

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest

from data.features import build_features, feature_columns
from inference.predict import ALLOW, BUDGETS, Detector, _raw_value, alert_thresholds, format_value
from model.preprocessing import to_model_frame
from model.selection import train_supervised
from tests.data.test_features import _config, _events, _random_events

TOP = 4


@pytest.fixture(scope="module")
def features() -> pl.DataFrame:
    frame = build_features(_events(_random_events(1_500, seed=5)), _config()).collect()
    return frame.with_columns(pl.col("new_country").fill_null(value=False).alias("label_ato"))


@pytest.fixture(scope="module")
def saved(features: pl.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A data directory laid out like the real one, with a small trained model."""
    root = tmp_path_factory.mktemp("data")
    columns = feature_columns()
    pipeline = train_supervised("random forest", features, columns, negative_share=0.5)
    (root / "06_models").mkdir()
    (root / "07_model_output").mkdir()
    joblib.dump(pipeline, root / "06_models" / "toy_model.joblib")
    (root / "06_models" / "toy_model.json").write_text(json.dumps({"columns": columns}))
    scores = pipeline.predict_proba(to_model_frame(features, columns))
    pl.DataFrame({"event_id": features["event_id"], "score": scores[:, 1]}).write_parquet(
        root / "07_model_output" / "toy_dev_oof.parquet"
    )
    return root


def test_budgets_must_be_strictest_first() -> None:
    with pytest.raises(ValueError, match="strictest"):
        alert_thresholds(np.linspace(0, 1, 100), {"loose": 0.01, "strict": 0.001})


def test_thresholds_follow_the_budgets() -> None:
    scores = np.linspace(0, 1, 100_001)
    thresholds = alert_thresholds(scores)
    assert list(thresholds) == list(BUDGETS), "strictest tier first"
    for tier, share in BUDGETS.items():
        assert (scores >= thresholds[tier]).mean() == pytest.approx(share, abs=1e-4)


def test_tiers(saved: Path) -> None:
    detector = Detector.load(saved, "toy")
    block, step_up = detector.thresholds["review / block"], detector.thresholds["step-up"]
    assert detector.tier(block) == "review / block"
    assert block > step_up, "the stricter tier has the higher threshold"
    assert detector.tier((block + step_up) / 2) == "step-up"
    assert detector.tier(-1.0) == ALLOW
    assert detector.riskier_than(-1.0) == 0.0
    assert detector.riskier_than(2.0) == 1.0
    rank = detector.riskier_than(step_up)
    assert 0.95 <= rank <= 1 - BUDGETS["step-up"] + 1e-9, "the step-up threshold is the top 1%"  # noqa: PLR2004


def test_loaded_detector_scores_and_explains(saved: Path, features: pl.DataFrame) -> None:
    detector = Detector.load(saved, "toy")
    rows = features.filter(pl.col("label_ato")).head(3)
    scores = detector.score(rows)
    assert scores.shape == (3,)
    assert ((scores >= 0) & (scores <= 1)).all()
    explanations = detector.explain(rows, top=TOP)
    assert len(explanations) == rows.height
    for explanation in explanations:
        assert explanation.height == TOP
        assert explanation.columns == ["input", "value", "contribution"]
        assert explanation["contribution"].abs().is_sorted(descending=True)


def test_explanations_name_the_original_value() -> None:
    raw = pl.DataFrame(
        {
            "asn_prior_attack_share": [None, 0.5],
            "device_type": ["mobile", "desktop"],
            "hour": [3.0, 4.0],
        }
    )
    assert _raw_value(raw, "missingindicator_asn_prior_attack_share", 0) == "ASN never seen"
    assert _raw_value(raw, "missingindicator_asn_prior_attack_share", 1) == "ASN seen before"
    assert _raw_value(raw, "asn_prior_attack_share", 1) == "50.00%"
    assert _raw_value(raw, "device_type_mobile", 1) == "desktop"
    assert _raw_value(raw, "hour_sin", 0) == "3"


def test_values_are_formatted_for_people() -> None:
    assert format_value("global_share_country", 0.0017767) == "0.18%"
    assert format_value("global_share_ip", 0.00003) == "< 0.01%"
    assert format_value("global_share_ip", 0.0) == "0.00%"
    assert format_value("first_success", value=True) == "yes"
    assert format_value("hours_since_prev_success", None) == "—"
    assert format_value("hours_since_prev_success", 937.456) == "937.46"
    assert format_value("prior_successes", 12.0) == "12"


def test_explanations_relative_to_typical_logins(saved: Path, features: pl.DataFrame) -> None:
    detector = Detector.load(saved, "toy")
    rows = features.head(5)
    absolute = detector.explain(rows, top=100)[0]
    detector.set_reference(rows)
    relative = detector.explain(rows, top=100)
    total = sum(float(e["contribution"].sum()) for e in relative)
    assert total == pytest.approx(0.0, abs=1e-6), (
        "contributions relative to the same rows average out"
    )
    assert not absolute.equals(relative[0])


def test_explanations_add_up_to_the_score(saved: Path, features: pl.DataFrame) -> None:
    detector = Detector.load(saved, "toy")
    rows = features.filter(pl.col("label_ato")).head(2)
    reference = features.filter(~pl.col("label_ato")).head(20)
    everything = 10 * len(detector.columns)
    detector.explain(rows, top=1)  # builds the explainer
    base = float(np.atleast_1d(detector._explainer.expected_value)[-1])
    scores = detector.score(rows)
    for i, e in enumerate(detector.explain(rows, top=everything)):
        assert float(e["contribution"].sum()) + base == pytest.approx(scores[i], abs=1e-6)
    detector.set_reference(reference)
    for i, e in enumerate(detector.explain(rows, top=everything)):
        expected = scores[i] - float(detector.score(reference).mean())
        assert float(e["contribution"].sum()) == pytest.approx(expected, abs=1e-6), (
            "relative to the reference"
        )


def test_explanations_show_flags_as_yes_no(saved: Path, features: pl.DataFrame) -> None:
    detector = Detector.load(saved, "toy")
    explanation = detector.explain(features.head(1), top=10 * len(detector.columns))[0]
    flag = explanation.filter(pl.col("input") == "attack_ip")["value"][0]
    assert flag in {"yes", "no"}
