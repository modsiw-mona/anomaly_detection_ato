"""Tests for the training configuration file and its validation."""

from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from data.features import DEFAULT_GROUPS
from inference.predict import BUDGETS
from model import baselines, evaluation
from model.selection import SELECTED, SUPERVISED
from pipelines.training_pipeline.config import (
    TrainingConfig,
    load_training_config,
    training_config_path,
)


def _raw() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(training_config_path("rba").read_text())
    return loaded


def test_rba_configuration_is_the_one_selected_in_issue_6() -> None:
    config = load_training_config("rba")
    assert config.model == SELECTED
    assert config.feature_groups == (*DEFAULT_GROUPS, "campaign", "travel")
    assert config.balance_campaign
    assert config.negative_share == baselines.NEGATIVE_SHARE
    assert config.seed == baselines.SEED
    assert config.name == "random forest (leaf 2) + campaign + travel, balanced"
    assert (config.folds.n_folds, config.folds.repeats, config.folds.seed) == (
        5,
        3,
        evaluation.SEED,
    )
    assert config.budgets == BUDGETS
    assert config.baseline.model == "logistic regression"
    assert config.baseline.feature_groups == DEFAULT_GROUPS


def test_the_selected_forest_is_the_tuned_one() -> None:
    model = SUPERVISED[SELECTED](["prior_successes"], 1).named_steps["model"]
    assert (model.min_samples_leaf, model.max_features, model.n_estimators) == (2, "sqrt", 300)
    assert model.class_weight == "balanced_subsample"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"model": "magic forest"}, "unknown model"),
        ({"feature_groups": ["history", "telepathy"]}, "unknown feature groups"),
        ({"budgets": {"step-up": 0.01, "review / block": 0.001}}, "strictest"),
        ({"negative_share": 1.5}, "less than 1"),
        ({"surprise": 1}, "Extra inputs"),
    ],
)
def test_invalid_configuration_is_rejected(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        TrainingConfig.model_validate({**_raw(), **change})


def test_budgets_must_name_the_response_tiers() -> None:
    with pytest.raises(ValidationError, match="response tiers"):
        TrainingConfig.model_validate({**_raw(), "budgets": {"block": 0.001, "step-up": 0.01}})


def test_alert_rate_ratio_must_bracket_one() -> None:
    raw = _raw()
    raw["gates"] = {**raw["gates"], "alert_rate_ratio": [1.5, 2.0]}
    with pytest.raises(ValidationError, match="alert_rate_ratio"):
        TrainingConfig.model_validate(raw)
