"""Tests for the evaluation protocol: folds, budget metrics and the PR-AUC bootstrap."""

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score

from model.evaluation import (
    assign_folds,
    average_precision_from_positives,
    bootstrap_pr_auc,
    evaluate,
)

N_FOLDS, N_REPEATS = 5, 3
TAKEOVER_ROW_SHARE, PREVALENCE, RARE_PREVALENCE = 0.5, 0.02, 0.005


def _features(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    users = np.repeat(np.arange(500), rng.integers(1, 5, size=500))
    label = np.isin(users, np.arange(23)) & (
        rng.random(users.size) < TAKEOVER_ROW_SHARE
    )  # ~23 takeover users
    return pl.DataFrame({"user_id": users, "label_ato": label})


def test_folds_group_users_and_spread_takeovers_evenly() -> None:
    features = _features()
    folds = assign_folds(features, n_folds=N_FOLDS, n_repeats=N_REPEATS)
    assert folds["user_id"].is_unique().all()
    for repeat in range(N_REPEATS):
        per_fold = folds.filter("has_takeover").group_by(f"fold_{repeat}").len()["len"].to_numpy()
        assert per_fold.max() - per_fold.min() <= 1, "takeover users spread evenly across folds"
    assert not folds["fold_0"].equals(folds["fold_1"]), "repeats use different partitions"
    assert folds.equals(assign_folds(features, n_folds=N_FOLDS, n_repeats=N_REPEATS))


@pytest.mark.parametrize("seed", range(5))
def test_average_precision_matches_scikit_learn_with_ties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.random(5_000) < PREVALENCE
    score = rng.integers(0, 20, size=y.size).astype(float) + 3 * y  # many ties, some signal
    ours = average_precision_from_positives(score[y], np.sort(score[~y]))
    assert ours == pytest.approx(average_precision_score(y, score), abs=1e-12)


def test_budget_metrics_on_a_perfect_and_a_constant_score() -> None:
    y = np.zeros(10_000, dtype=bool)
    y[:5] = True
    perfect = evaluate(y, y.astype(float))
    assert perfect["recall_at_0_1pct"] == 1.0, "5 takeovers fit in the top 10"
    assert perfect["pr_auc"] == 1.0
    constant = evaluate(y, np.zeros(y.size))
    assert constant["recall_at_1pct"] == pytest.approx(0.01), "ties are shared out at random"
    assert constant["pr_auc"] == pytest.approx(5 / 10_000)
    assert constant["friction_at_1pct"] == pytest.approx((100 - 0.05) / 9_995)


def test_bootstrap_interval_contains_the_estimate_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    y = rng.random(20_000) < RARE_PREVALENCE
    score = rng.normal(size=y.size) + 2 * y
    point = average_precision_score(y, score)
    low, high = bootstrap_pr_auc(y, score, n_boot=500)
    assert low < point < high
    assert (low, high) == bootstrap_pr_auc(y, score, n_boot=500)
