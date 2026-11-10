"""Configuration of the training pipeline: ``conf/training/<dataset>.yaml``, validated with pydantic.

It fixes everything a training run depends on (model, feature groups, sampling, seeds, folds, alert
budgets, the baseline and the limits of the model validation gates), so a run is reproducible from
the file alone and a change shows up in the repository's history.
"""

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from data.datasets import find_repo_root
from data.features import FEATURE_GROUPS
from inference.predict import BUDGETS as TIERS
from model.selection import SUPERVISED

Share = Annotated[float, Field(gt=0, lt=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _known_groups(groups: tuple[str, ...]) -> tuple[str, ...]:
    unknown = set(groups) - FEATURE_GROUPS.keys()
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    if not groups:
        raise ValueError("at least one feature group is needed")
    return groups


def _known_model(name: str) -> str:
    if name not in SUPERVISED:
        raise ValueError(f"unknown model {name!r}; known: {sorted(SUPERVISED)}")
    return name


class Folds(_Strict):
    """User-grouped cross-validation on the development side (``model.evaluation.assign_folds``)."""

    n_folds: int = Field(ge=2)
    repeats: int = Field(ge=1)
    seed: int


class Baseline(_Strict):
    """The model the selected one must beat, scored on the same folds."""

    model: str
    feature_groups: tuple[str, ...]

    _check_model = field_validator("model")(_known_model)
    _check_groups = field_validator("feature_groups")(_known_groups)


class Segments(_Strict):
    """Takeovers of the dominant campaign (``column == value``) against all the others."""

    column: str
    value: str
    name: str


class Gates(_Strict):
    """Limits of the model validation gates (``model.validation``)."""

    min_interval_low: float = Field(ge=0, le=1)
    min_recall_at_1pct: float = Field(ge=0, le=1)
    min_gain_over_baseline: float = Field(ge=0)
    alert_rate_ratio: tuple[float, float]
    min_segment_recall_at_1pct: float = Field(ge=0, le=1)
    max_seconds_per_login_in_batch: float = Field(gt=0)
    max_seconds_single_login: float = Field(gt=0)

    @field_validator("alert_rate_ratio")
    @classmethod
    def _ordered(cls, ratio: tuple[float, float]) -> tuple[float, float]:
        if not 0 < ratio[0] <= 1 <= ratio[1]:
            raise ValueError("alert_rate_ratio must be [low, high] with 0 < low <= 1 <= high")
        return ratio


class TrainingConfig(_Strict):
    """One training configuration (see ``conf/training/rba.yaml``)."""

    name: str  # how the configuration is called in results (notebook 06: "selected")
    model: str
    feature_groups: tuple[str, ...]
    balance_campaign: bool
    negative_share: Share
    seed: int
    folds: Folds
    budgets: dict[str, Share]
    baseline: Baseline
    segments: Segments
    gates: Gates

    _check_model = field_validator("model")(_known_model)
    _check_groups = field_validator("feature_groups")(_known_groups)

    @model_validator(mode="after")
    def _budgets(self) -> TrainingConfig:
        shares = list(self.budgets.values())
        if not shares or shares != sorted(shares):
            raise ValueError("budgets must be listed strictest (smallest) first")
        if set(self.budgets) != set(TIERS):
            raise ValueError(f"budgets must name the response tiers {list(TIERS)}")
        return self


def training_config_path(dataset: str, conf_dir: Path | None = None) -> Path:
    """Where the training configuration of ``dataset`` lives."""
    return (conf_dir or find_repo_root() / "conf") / "training" / f"{dataset}.yaml"


def load_training_config(dataset: str, conf_dir: Path | None = None) -> TrainingConfig:
    """Read and validate ``conf/training/<dataset>.yaml``."""
    raw = yaml.safe_load(training_config_path(dataset, conf_dir).read_text())
    return TrainingConfig.model_validate(raw)
