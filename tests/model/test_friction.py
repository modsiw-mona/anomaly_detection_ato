"""Tests for the per-user friction metric (re-authentication rate per legitimate user)."""

import numpy as np
import polars as pl
import pytest

from model.friction import (
    bootstrap_friction,
    challenged_counts,
    threshold_for_rate,
    user_friction,
)

HALF, TWO_THIRDS = 0.5, 2 / 3
TAKEOVER_SHARE = 0.03
TOY_USERS = 3


def _toy() -> pl.DataFrame:
    """User a: one of two logins high; b: none; c: its only login high. Two takeovers."""
    return pl.DataFrame(
        {
            "user_id": ["a", "a", "b", "b", "c", "x", "y"],
            "label_ato": [False, False, False, False, False, True, True],
            "score": [0.9, 0.1, 0.1, 0.2, 0.95, 0.92, 0.5],
        }
    )


def test_threshold_is_the_score_of_the_kth_takeover() -> None:
    scores = np.array([0.05, 0.9, 0.1, 0.8])
    assert threshold_for_rate(scores, 0.5) == pytest.approx(0.8)
    assert threshold_for_rate(scores, 0.75) == pytest.approx(0.1)
    assert threshold_for_rate(scores, 0.7) == pytest.approx(0.1), "the next step up"
    assert threshold_for_rate(scores, 1.0) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="rate"):
        threshold_for_rate(scores, 0)
    with pytest.raises(ValueError, match="no takeovers"):
        threshold_for_rate(np.array([]), 0.5)


def test_rate_per_user_counts_only_legitimate_logins() -> None:
    logins, counts = challenged_counts(_toy(), [0.6], score="score")
    assert len(logins) == TOY_USERS, "takeover users x and y are not legitimate users"
    assert sorted(counts[:, 0] / logins) == [0.0, 0.5, 1.0]


def test_friction_at_a_fixed_threshold_and_at_a_detection_rate() -> None:
    result = user_friction(_toy(), thresholds={"fixed": 0.6}, rates=(0.5,))
    fixed = result["fixed"]
    assert fixed["detected"] == pytest.approx(HALF), "only the 0.92 takeover"
    assert fixed["median"] == pytest.approx(HALF)
    assert fixed["mean"] == pytest.approx(HALF)
    assert fixed["ever"] == pytest.approx(TWO_THIRDS)
    assert fixed["users"] == TOY_USERS
    rate = result["detect 50.0%"]
    assert rate["threshold"] == pytest.approx(0.92)
    assert rate["ever"] == pytest.approx(1 / 3), "only user c scores above 0.92"


def test_detecting_every_takeover_can_cost_more_friction() -> None:
    result = user_friction(_toy(), rates=(0.5, 1.0))
    assert result["detect 100.0%"]["mean"] >= result["detect 50.0%"]["mean"]


def test_null_scores_are_rejected() -> None:
    frame = _toy().with_columns(
        pl.when(pl.col("user_id") == "b").then(None).otherwise(pl.col("score")).alias("score")
    )
    with pytest.raises(ValueError, match="nulls"):
        user_friction(frame)


def test_bootstrap_interval_contains_the_estimate_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    users = np.repeat(np.arange(400), 3)
    label = rng.random(users.size) < TAKEOVER_SHARE
    score = np.where(label, rng.beta(5, 2, users.size), rng.beta(1, 8, users.size))
    frame = pl.DataFrame({"user_id": users, "label_ato": label, "score": score})
    point = user_friction(frame, rates=(0.8,))["detect 80.0%"]
    first = bootstrap_friction(frame, rates=(0.8,), n_boot=300)
    again = bootstrap_friction(frame, rates=(0.8,), n_boot=300)
    assert first.equals(again)
    row = first.row(0, named=True)
    assert row["mean low"] <= point["mean"] <= row["mean high"]
    assert row["ever low"] <= point["ever"] <= row["ever high"]
