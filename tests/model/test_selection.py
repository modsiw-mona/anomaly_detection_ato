"""Tests for model selection: out-of-fold scoring, leave-group-out and the candidates."""

from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score

from data.features import build_features, feature_columns
from model.evaluation import assign_folds
from model.selection import (
    SUPERVISED,
    Scorer,
    campaign_weights,
    candidate,
    fit_isolation_forest,
    fit_pu_bagging,
    fit_supervised,
    fold_per_login,
    group_metrics,
    out_of_fold,
)
from tests.data.test_features import _config, _events, _random_events

LEARNT_OVER_CHANCE = 10
NEGATIVE_SHARE = 0.5
N_FOLDS = 3


@pytest.fixture(scope="module")
def features() -> pl.DataFrame:
    frame = build_features(_events(_random_events(1_500, seed=5)), _config()).collect()
    # a learnable signal: takeovers are logins from a new country for the user
    return frame.with_columns(pl.col("new_country").fill_null(value=False).alias("label_ato"))


@pytest.fixture(scope="module")
def folds(features: pl.DataFrame) -> pl.DataFrame:
    return assign_folds(features, n_folds=N_FOLDS, n_repeats=1)


@pytest.fixture(scope="module")
def fold(features: pl.DataFrame, folds: pl.DataFrame) -> np.ndarray:
    return fold_per_login(features, folds, "fold_0")


class Spy:
    """A fitter that records what it was trained on and scores by event id."""

    def __init__(self) -> None:
        self.trained_on: list[pl.DataFrame] = []

    def __call__(self, train: pl.DataFrame, columns: Sequence[str]) -> Scorer:
        self.trained_on.append(train)
        return lambda frame: frame["event_id"].cast(pl.Float64).to_numpy()


def test_out_of_fold_scores_every_login_without_training_on_its_user(
    features: pl.DataFrame, folds: pl.DataFrame, fold: np.ndarray
) -> None:
    spy = Spy()
    scores = out_of_fold(features, fold, spy, feature_columns())
    assert np.array_equal(scores, features["event_id"].cast(pl.Float64).to_numpy())
    assert len(spy.trained_on) == N_FOLDS
    fold_of = dict(zip(folds["user_id"], folds["fold_0"], strict=True))
    for k, train in enumerate(spy.trained_on):
        assert all(fold_of[u] != k for u in train["user_id"].unique()), "a fold saw its own users"


def test_hidden_takeovers_never_reach_training(features: pl.DataFrame, fold: np.ndarray) -> None:
    hide = (features["device_type"] == "mobile").to_numpy()
    hidden_ids = set(features.filter(pl.Series(hide) & pl.col("label_ato"))["event_id"])
    assert hidden_ids, "the test needs hidden takeovers"
    spy = Spy()
    out_of_fold(features, fold, spy, feature_columns(), hide=hide)
    trained = set().union(*(set(t["event_id"]) for t in spy.trained_on))
    assert not hidden_ids & trained
    legit_hidden = set(features.filter(pl.Series(hide) & ~pl.col("label_ato"))["event_id"])
    assert legit_hidden <= trained, "only takeovers are hidden, legitimate logins stay"


@pytest.mark.parametrize("name", [*SUPERVISED, "PU bagging"])
def test_supervised_candidates_learn_a_real_signal(
    name: str, features: pl.DataFrame, fold: np.ndarray
) -> None:
    def fit(train: pl.DataFrame, columns: Sequence[str]) -> Scorer:  # toy data: keep more legit
        if name == "PU bagging":
            return fit_pu_bagging(train, columns, n_bags=3, negative_share=NEGATIVE_SHARE)
        return fit_supervised(name, train, columns, negative_share=NEGATIVE_SHARE)

    scores = out_of_fold(features, fold, fit, feature_columns())
    y = features["label_ato"].to_numpy()
    assert average_precision_score(y, scores) > LEARNT_OVER_CHANCE * y.mean()


def test_isolation_forest_scores_every_login(features: pl.DataFrame) -> None:
    score = fit_isolation_forest(features, feature_columns())(features)
    assert score.shape == (features.height,)
    assert np.isfinite(score).all()


def test_every_user_needs_a_fold(features: pl.DataFrame, folds: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="fold"):
        fold_per_login(features, folds.head(1), "fold_0")


def test_candidates_by_name() -> None:
    for name in (*SUPERVISED, "PU bagging", "isolation forest"):
        assert callable(candidate(name))
    with pytest.raises(KeyError):
        candidate("nope")


def test_group_metrics_compare_each_group_against_all_legitimate_logins() -> None:
    y = np.array([True, True, False, False, False, False])
    score = np.array([0.9, 0.1, 0.5, 0.4, 0.3, 0.2])
    groups = {"first": np.array([True, False, False, False, False, False])}
    result = group_metrics(y, score, groups)
    assert result["first"]["pr_auc"] == 1.0, "the first takeover outranks every legitimate login"
    assert result["first"]["prevalence"] == pytest.approx(1 / 5)
    assert result["all"]["prevalence"] == pytest.approx(2 / 6)


def test_campaign_weights_balance_the_dominant_group_against_the_rest() -> None:
    sample = pl.DataFrame(
        {
            "label_ato": [True, True, True, True, False, False],
            "country": ["RO", "RO", "RO", "NO", "NO", "RO"],
        }
    )
    w = campaign_weights(sample)
    assert w[:3].sum() == pytest.approx(w[3]), "RO takeovers weigh as much as all the others"
    assert w[:4].sum() == pytest.approx(4), "takeovers keep their total weight"
    assert (w[4:] == 1).all(), "legitimate logins are untouched"
    assert (
        campaign_weights(sample.with_columns(pl.lit(value=False).alias("label_ato"))) == 1
    ).all()


def test_campaign_balanced_training_learns(features: pl.DataFrame, fold: np.ndarray) -> None:
    frame = features.with_columns(pl.col("country").cast(pl.String))

    def fit(train: pl.DataFrame, columns: Sequence[str]) -> Scorer:
        return fit_supervised(
            "random forest", train, columns, balance_campaign=True, negative_share=NEGATIVE_SHARE
        )

    scores = out_of_fold(frame, fold, fit, feature_columns())
    y = frame["label_ato"].to_numpy()
    assert average_precision_score(y, scores) > LEARNT_OVER_CHANCE * y.mean()
