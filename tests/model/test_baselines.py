"""Tests for the baselines: rules, training sample and the logistic model."""

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score

from data.features import build_features
from model.baselines import fit_logistic, predict_scores, rule_scores, training_sample
from tests.data.test_features import _config, _events, _random_events

NEGATIVE_SHARE = 0.25
LEARNT_OVER_CHANCE = 10  # ~9 training takeovers: demand 10x the PR-AUC of chance


@pytest.fixture(scope="module")
def features() -> pl.DataFrame:
    frame = build_features(_events(_random_events(1_500, seed=5)), _config()).collect()
    # a learnable signal: takeovers are logins from a new country for the user
    return frame.with_columns(pl.col("new_country").fill_null(value=False).alias("label_ato"))


def test_rule_scores(features: pl.DataFrame) -> None:
    scores = rule_scores(features)
    assert not scores["dummy (constant)"].any()
    new_country = features["new_country"].fill_null(value=False).to_numpy()
    assert np.array_equal(scores["rule: new country"], new_country.astype(float))
    combined = scores["rule: new country"] + scores["rule: attack IP"]
    assert np.array_equal(scores["rule: new country + attack IP"], combined)


def test_training_sample_keeps_every_takeover(features: pl.DataFrame) -> None:
    sample = training_sample(features, negative_share=NEGATIVE_SHARE)
    assert sample["label_ato"].sum() == features["label_ato"].sum()
    legit_kept = (~sample["label_ato"].to_numpy()).sum() / (~features["label_ato"].to_numpy()).sum()
    assert legit_kept == pytest.approx(NEGATIVE_SHARE, abs=0.02)
    assert sample.equals(training_sample(features, negative_share=NEGATIVE_SHARE)), "seeded"


def test_logistic_baseline_learns_a_real_signal(features: pl.DataFrame) -> None:
    alternate = features.with_row_index("_row")
    train = alternate.filter(pl.col("_row") % 2 == 0).drop("_row")
    test = alternate.filter(pl.col("_row") % 2 == 1).drop("_row")
    assert train["label_ato"].sum() > 0, "both halves need takeovers"
    assert test["label_ato"].sum() > 0, "both halves need takeovers"
    model = fit_logistic(train, negative_share=NEGATIVE_SHARE)
    scores = predict_scores(model, test)
    y = test["label_ato"].to_numpy()
    assert scores.shape == (test.height,)
    chance = y.mean()  # the PR-AUC of a random score
    assert average_precision_score(y, scores) > LEARNT_OVER_CHANCE * chance, "signal not learnt"
