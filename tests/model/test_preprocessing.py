"""Tests for the scikit-learn preprocessing of the feature table."""

import numpy as np
import polars as pl
import pytest

from data.features import FEATURE_GROUPS, build_features, feature_columns
from model.preprocessing import build_preprocessor, to_model_frame
from tests.data.test_features import HAND_BUILT, _config, _events, _random_events

ALL_GROUPS = tuple(FEATURE_GROUPS)


@pytest.fixture(scope="module")
def features() -> pl.DataFrame:
    return build_features(
        _events(HAND_BUILT + _random_events(300, seed=3)[9:]), _config()
    ).collect()


def test_default_feature_set_transforms_without_missing_values(features: pl.DataFrame) -> None:
    columns = feature_columns()
    out = build_preprocessor(columns).fit_transform(to_model_frame(features, columns))
    assert out.shape[0] == features.height
    assert not np.isnan(out).any()


def test_every_group_can_be_used_alone(features: pl.DataFrame) -> None:
    for group in ALL_GROUPS:
        columns = list(FEATURE_GROUPS[group])
        out = build_preprocessor(columns).fit_transform(to_model_frame(features, columns))
        assert out.shape[1] >= 1, group
        assert not np.isnan(out).any(), group


def test_output_names_are_traceable(features: pl.DataFrame) -> None:
    columns = feature_columns(("history", "device", "time"))
    pre = build_preprocessor(columns).fit(to_model_frame(features, columns))
    names = set(pre.get_feature_names_out())
    assert {"prior_successes", "first_success", "hour_sin", "weekday_cos"} <= names
    assert any(name.startswith("device_type_") for name in names)


def test_unseen_category_at_transform_time(features: pl.DataFrame) -> None:
    columns = ["device_type"]
    pre = build_preprocessor(columns).fit(to_model_frame(features, columns))
    unseen = to_model_frame(
        features.head(1).with_columns(pl.lit("smart-fridge").alias("device_type")), columns
    )
    assert pre.transform(unseen).sum() == 0, "unknown categories encode as all zeros"


def test_fitted_statistics_come_from_training_rows_only(features: pl.DataFrame) -> None:
    columns = ["prior_successes"]
    train, other = features.head(20), features.tail(20)
    pre = build_preprocessor(columns).fit(to_model_frame(train, columns))
    scaler = pre.named_transformers_["counts"][-1]
    expected = np.log1p(train["prior_successes"].cast(pl.Float64).to_numpy()).mean()
    assert scaler.mean_[0] == pytest.approx(expected)
    pre.transform(to_model_frame(other, columns))  # transforming other rows never refits
    assert scaler.mean_[0] == pytest.approx(expected)


def test_unsupported_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="Unsupported"):
        to_model_frame(pl.DataFrame({"weird": ["text"]}), ["weird"])
