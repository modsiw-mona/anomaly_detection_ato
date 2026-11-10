"""Tests for the model validation gates."""

import numpy as np
import pytest

from model import validation
from model.evaluation import Metrics


def _metrics(pr_auc: float, recall_at_1pct: float) -> Metrics:
    return {
        "pr_auc": pr_auc,
        "roc_auc": 0.9,
        "prevalence": 0.001,
        "recall_at_0_1pct": recall_at_1pct,
        "recall_at_1pct": recall_at_1pct,
        "precision_at_1pct": 0.01,
        "friction_at_1pct": 0.01,
    }


@pytest.mark.parametrize(
    ("low", "recall", "passed"),
    [(0.06, 0.8, [True, True]), (0.05, 0.8, [False, True]), (0.4, 0.79, [True, False])],
)
def test_selection_rule(low: float, recall: float, passed: list[bool]) -> None:
    gates = validation.selection_rule(low, recall, min_low=0.05, min_recall=0.8)
    assert [g["passed"] for g in gates] == passed
    assert all(g["blocking"] for g in gates)


@pytest.mark.parametrize(
    ("model", "baseline", "passed"), [(0.45, 0.40, True), (0.43, 0.40, False), (0.1, 0.0, True)]
)
def test_gain_over_baseline(model: float, baseline: float, *, passed: bool) -> None:
    assert validation.gain_over_baseline(model, baseline, min_gain=0.1)["passed"] is passed


def test_alert_rates_compare_the_flagged_share_with_each_budget() -> None:
    scores = np.arange(1_000) / 1_000
    budgets = {"block": 0.01, "step-up": 0.1}
    on_budget = validation.alert_rates(
        scores, {"block": 0.99, "step-up": 0.9}, budgets, ratio=(0.5, 2.0)
    )
    assert [g["value"] for g in on_budget] == [0.01, 0.1]
    assert all(g["passed"] for g in on_budget)
    too_many = validation.alert_rates(scores, {"block": 0.9}, budgets, ratio=(0.5, 2.0))
    assert not too_many[0]["passed"], "10x the block budget"


def test_segment_recall_skips_the_overall_row() -> None:
    segments = {"all": _metrics(0.4, 0.9), "A": _metrics(0.5, 0.95), "B": _metrics(0.1, 0.4)}
    gates = validation.segment_recall(segments, min_recall=0.5)
    assert [(g["name"], g["passed"]) for g in gates] == [
        ("segment_recall_at_1pct[A]", True),
        ("segment_recall_at_1pct[B]", False),
    ]


def test_weakest_segment_is_reported_not_blocking() -> None:
    segments = {"all": _metrics(0.4, 0.9), "A": _metrics(0.5, 0.95), "B": _metrics(0.1, 0.4)}
    gate = validation.weakest_segment(segments)
    assert gate["value"] == pytest.approx(0.25)
    assert not gate["passed"]
    assert not gate["blocking"]
    assert "B" in gate["condition"]


def test_scoring_time_in_batch_and_for_one_login() -> None:
    assert validation.batch_scoring_time(2e-6, max_seconds=0.001)["passed"]
    assert not validation.batch_scoring_time(0.01, max_seconds=0.001)["passed"]
    assert validation.single_login_latency(0.03, max_seconds=0.2)["passed"]
    assert not validation.single_login_latency(0.5, max_seconds=0.2)["passed"]


def test_overfitting_gap_and_calibration_are_reported() -> None:
    y = np.array([True, False, False, True, False])
    perfect = np.array([0.9, 0.1, 0.2, 0.8, 0.3])
    noisy = np.array([0.4, 0.5, 0.2, 0.8, 0.3])
    gap = validation.overfitting_gap(y, perfect, noisy)
    assert gap["value"] > 0
    assert gap["passed"] and not gap["blocking"]
    brier = validation.calibration(y, perfect)
    assert brier["value"] == pytest.approx(np.mean((perfect - y) ** 2), rel=1e-3)
    assert not brier["blocking"]


def test_check_raises_only_for_failed_blocking_gates() -> None:
    reported = validation.weakest_segment({"all": _metrics(0.4, 0.9), "A": _metrics(0.1, 0.9)})
    validation.check([reported])  # failed, but only reported
    blocking = validation.batch_scoring_time(1.0, max_seconds=0.001)
    with pytest.raises(validation.ModelValidationError, match="seconds_per_login") as caught:
        validation.check([reported, blocking])
    assert caught.value.failed == [blocking]
