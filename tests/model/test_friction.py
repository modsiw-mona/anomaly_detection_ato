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


def test_a_login_scoring_exactly_the_threshold_is_challenged() -> None:
    frame = pl.DataFrame(
        {"user_id": ["a", "a", "x"], "label_ato": [False, False, True], "score": [0.5, 0.1, 0.5]}
    )
    result = user_friction(frame, rates=(1.0,))["detect 100.0%"]
    assert result["threshold"] == pytest.approx(0.5)
    assert result["mean"] == pytest.approx(HALF), "the tie at 0.5 counts as asked"


def test_bootstrap_looks_up_the_right_threshold() -> None:
    """Two takeovers, detect both: a replicate's threshold is the lower score (0.4) unless both draws
    are the higher one (1 in 4). A central interval (30th to 70th percentile) then holds only the
    friction at 0.4, whose value differs from the friction at 0.8."""
    frame = pl.DataFrame(
        {
            "user_id": ["a", "a", "b", "c", "c", "x", "y"],
            "label_ato": [False, False, False, False, False, True, True],
            "score": [0.9, 0.1, 0.5, 0.65, 0.2, 0.8, 0.4],
        }
    )
    direct = user_friction(frame, thresholds={"high": 0.8, "low": 0.4}, rates=())
    assert direct["low"]["mean"] != pytest.approx(direct["high"]["mean"])
    row = bootstrap_friction(frame, rates=(1.0,), n_boot=400, level=0.4).row(0, named=True)
    for end in ("mean low", "mean high"):
        assert row[end] == pytest.approx(direct["low"]["mean"]), end
    for end in ("ever low", "ever high"):
        assert row[end] == pytest.approx(direct["low"]["ever"]), end


def test_bad_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="NaN"):
        user_friction(_toy().with_columns(pl.when(pl.col("user_id") == "b").then(float("nan"))
                                          .otherwise(pl.col("score")).alias("score")))  # fmt: skip
    with pytest.raises(ValueError, match="no legitimate"):
        user_friction(_toy().filter(pl.col("label_ato")))
    with pytest.raises(ValueError, match="both named"):
        user_friction(_toy(), thresholds={"detect 50.0%": 0.3}, rates=(0.5,))


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
