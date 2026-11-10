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
    """Feature table of the hand-built events plus random ones, shared by the module."""
    return build_features(
        _events(HAND_BUILT + _random_events(300, seed=3)[9:]), _config()
    ).collect()


def test_default_feature_set_transforms_without_missing_values(features: pl.DataFrame) -> None:
    """The default feature set transforms to a matrix with no missing values."""
    columns = feature_columns()
    out = build_preprocessor(columns).fit_transform(to_model_frame(features, columns))
    assert out.shape[0] == features.height
    assert not np.isnan(out).any()


def test_every_group_can_be_used_alone(features: pl.DataFrame) -> None:
    """Each feature group alone transforms to at least one column with no missing values."""
    for group in ALL_GROUPS:
        columns = list(FEATURE_GROUPS[group])
        out = build_preprocessor(columns).fit_transform(to_model_frame(features, columns))
        assert out.shape[1] >= 1, group
        assert not np.isnan(out).any(), group


def test_output_names_are_traceable(features: pl.DataFrame) -> None:
    """Output column names trace back to their source features."""
    columns = feature_columns(("history", "device", "time"))
    pre = build_preprocessor(columns).fit(to_model_frame(features, columns))
    names = set(pre.get_feature_names_out())
    assert {"prior_successes", "first_success", "hour_sin", "weekday_cos"} <= names
    assert any(name.startswith("device_type_") for name in names)


def test_unseen_category_at_transform_time(features: pl.DataFrame) -> None:
    """A category not seen in fitting encodes as all zeros."""
    columns = ["device_type"]
    pre = build_preprocessor(columns).fit(to_model_frame(features, columns))
    unseen = to_model_frame(
        features.head(1).with_columns(pl.lit("smart-fridge").alias("device_type")), columns
    )
    assert pre.transform(unseen).sum() == 0, "unknown categories encode as all zeros"


def test_fitted_statistics_come_from_training_rows_only(features: pl.DataFrame) -> None:
    """Scaling statistics come from the fitted rows and do not change on transform."""
    columns = ["prior_successes"]
    train, other = features.head(20), features.tail(20)
    pre = build_preprocessor(columns).fit(to_model_frame(train, columns))
    scaler = pre.named_transformers_["counts"][-1]
    expected = np.log1p(train["prior_successes"].cast(pl.Float64).to_numpy()).mean()
    assert scaler.mean_[0] == pytest.approx(expected)
    pre.transform(to_model_frame(other, columns))  # transforming other rows never refits
    assert scaler.mean_[0] == pytest.approx(expected)


def test_unsupported_type_is_rejected() -> None:
    """``to_model_frame`` rejects a column that is neither numeric, boolean nor categorical."""
    with pytest.raises(TypeError, match="Unsupported"):
        to_model_frame(pl.DataFrame({"weird": ["text"]}), ["weird"])


EXPECTED_FAMILY = {
    "history": "counts", "global_rarity": "shares", "sharing": "counts", "failures": "counts",
    "attack_ip": "booleans", "missingness": "booleans", "artefacts": "booleans",
}  # fmt: skip


def _family_of(columns: list[str]) -> dict[str, str]:
    transformer = build_preprocessor(columns)
    return {
        c: name for name, _, cols in transformer.transformers for c in cols if name != "never_seen"
    }


def test_every_feature_lands_in_its_family() -> None:
    """``build_preprocessor`` places every feature in the transformer family of its kind."""
    family = _family_of(feature_columns(ALL_GROUPS))
    assert family.keys() == set(feature_columns(ALL_GROUPS)), "every feature is placed"
    for column in FEATURE_GROUPS["campaign"]:
        expected = "rates" if column.endswith("_share") else "counts"
        assert family[column] == expected, column
    assert family["user_prior_countries"] == "counts"
    assert family["country_change_1h"] == "booleans"
    assert family["hours_since_prev_success"] == "counts"
    for column in FEATURE_GROUPS["user_novelty"]:
        assert family[column] == ("counts" if column == "n_new_key_signals" else "booleans"), column
    for group, expected in EXPECTED_FAMILY.items():
        for column in FEATURE_GROUPS[group]:
            if column not in {"first_success"}:
                assert family[column] == expected, column


def test_never_seen_network_is_not_a_clean_record() -> None:
    """A network never seen before gets its own indicator instead of a 0% attack share."""
    frame = pl.DataFrame({"asn_prior_attack_share": [None, 0.0, 0.5, 1.0]})
    out = build_preprocessor(["asn_prior_attack_share"]).fit_transform(frame)
    never_seen = out[:, -1]
    assert never_seen.tolist() == [1.0, 0.0, 0.0, 0.0], "a separate 'never seen' indicator"
    assert out[0, 0] != out[1, 0], "unseen is not imputed as 0% attack"


def test_unknown_feature_is_rejected() -> None:
    """``build_preprocessor`` rejects a feature that belongs to no family."""
    with pytest.raises(ValueError, match="family"):
        build_preprocessor(["something_new"])
