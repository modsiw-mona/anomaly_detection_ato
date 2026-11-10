"""Tests for the validation rules of the feature table and the split."""

from collections.abc import Callable

import polars as pl
import pytest
from pandera.errors import SchemaErrors

from data import features as feature_definitions
from data.feature_validation import build_feature_schema, validate_features, validate_split
from data.features import build_features
from data.split import assign_split
from tests.data.test_features import (
    CATCH_ALL,
    HAND_BUILT,
    WINDOW,
    _config,
    _events,
    _random_events,
)

Mutation = Callable[[pl.DataFrame], pl.DataFrame]
RowPicker = Callable[[pl.DataFrame], int]


@pytest.fixture
def features() -> pl.DataFrame:
    frame = build_features(_events(_random_events(400, seed=3)), _config(**WINDOW)).collect()
    assert frame["first_success"].any() and not frame["first_success"].all()
    return frame


def _failed_checks(frame: pl.DataFrame) -> set[str]:
    with pytest.raises(SchemaErrors) as caught:
        validate_features(frame, _config(**WINDOW))
    return set(caught.value.failure_cases["check"].to_list())


def _returning(frame: pl.DataFrame) -> int:
    return int(frame["first_success"].arg_min() or 0)


def _first(frame: pl.DataFrame) -> int:
    return int(frame["first_success"].arg_max() or 0)


def _network_seen_before(frame: pl.DataFrame) -> int:
    return int((frame["global_share_asn"] > 0).arg_max() or 0)


def _on_row(column: str, value: pl.Expr, row: RowPicker = _returning) -> Mutation:
    """Replace ``column`` with ``value`` on one row, picked by ``row``."""

    def mutate(frame: pl.DataFrame) -> pl.DataFrame:
        at_row = pl.int_range(pl.len()) == row(frame)
        return frame.with_columns(
            pl.when(at_row).then(value).otherwise(pl.col(column)).alias(column)
        )

    return mutate


def test_features_built_by_the_pipeline_pass(features: pl.DataFrame) -> None:
    assert validate_features(features, _config(**WINDOW)).height == features.height


def test_hand_built_features_pass() -> None:
    frame = build_features(_events(HAND_BUILT), _config(**WINDOW)).collect()
    assert validate_features(frame, _config(**WINDOW)).height == frame.height


@pytest.mark.parametrize(
    ("mutation", "check"),
    [
        (_on_row("hour", pl.lit(24, pl.Int8)), "in_range(0, 23)"),
        (_on_row("weekday", pl.lit(0, pl.Int8)), "in_range(1, 7)"),
        (_on_row("global_share_asn", pl.lit(1.5)), "in_range(0, 1)"),
        (_on_row("hours_since_prev_success", pl.lit(-1.0)), "greater_than_or_equal_to(0)"),
        (_on_row("n_new_key_signals", pl.lit(9, pl.UInt8)), "less_than_or_equal_to(5)"),
        (_on_row("first_success", pl.lit(value=True)), "first_login_has_no_earlier_logins"),
        (
            _on_row("new_country", pl.lit(value=True), _first),
            "new_country_null_exactly_on_first_login",
        ),
        (
            _on_row("hours_since_prev_success", pl.lit(None, pl.Float64)),
            "hours_since_prev_success_null_exactly_on_first_login",
        ),
        (
            _on_row("user_prior_countries", pl.lit(2, pl.UInt32), _first),
            "first_login_has_no_earlier_countries",
        ),
        (_on_row("country_change_1h", pl.lit(None, pl.Boolean)), "travel_known_after_first_login"),
        (_on_row("new_country", ~pl.col("new_country")), "n_new_key_signals_counts_them"),
        (
            _on_row("asn_prior_attack_share", pl.lit(None, pl.Float64), _network_seen_before),
            "attack_share_known_with_fail_share",
        ),
        (_on_row("fails_1h", pl.col("fails_24h") + 1), "fails_24h_include_fails_1h"),
        (
            _on_row("asn_fails_prev24h", pl.col("asn_attempts_prev24h") + 1),
            "network_fails_within_attempts",
        ),
        (_on_row("rtt_missing", ~pl.col("rtt_missing")), "rtt_missing_flags_null_rtt"),
        (
            _on_row("asn_prior_users", pl.lit(0, pl.UInt32), _network_seen_before),
            "asn_seen_before_iff_shared",
        ),
        (
            _on_row("timestamp", pl.col("timestamp").min() - pl.duration(days=1)),
            "rows_in_time_order",
        ),
        (_on_row("event_id", pl.col("event_id").first()), "field_uniqueness"),
        (_on_row("user_id", pl.lit(CATCH_ALL, pl.Int64)), "not_a_catch_all_account"),
    ],
)
def test_each_rule_catches_its_violation(
    features: pl.DataFrame, mutation: Mutation, check: str
) -> None:
    assert check in _failed_checks(mutation(features))


def test_login_outside_the_label_window_is_an_error(features: pl.DataFrame) -> None:
    late = features.with_columns(pl.col("timestamp").dt.offset_by("1y"))
    assert "label_window" in _failed_checks(late)


def test_unexpected_column_and_wrong_type_are_errors(features: pl.DataFrame) -> None:
    assert "column_in_schema" in _failed_checks(features.with_columns(pl.lit(1).alias("extra")))
    retyped = features.with_columns(pl.col("prior_successes").cast(pl.Int64))
    assert any(check.startswith("dtype") for check in _failed_checks(retyped))


def test_a_dataset_without_some_fields_still_validates(features: pl.DataFrame) -> None:
    fewer = features.drop("rtt_ms", "rtt_missing", "country", "country_change_1h")
    assert validate_features(fewer, _config(**WINDOW)).height == features.height


def test_a_new_feature_needs_a_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    groups = {**feature_definitions.FEATURE_GROUPS, "new": ("brand_new_feature",)}
    monkeypatch.setattr("data.feature_validation.FEATURE_GROUPS", groups)
    with pytest.raises(RuntimeError, match="brand_new_feature"):
        build_feature_schema(_config())


# --- split ---


def _split_failures(features: pl.DataFrame, split: pl.DataFrame) -> set[str]:
    with pytest.raises(SchemaErrors) as caught:
        validate_split(features.lazy(), split)
    return set(caught.value.failure_cases["check"].to_list())


@pytest.fixture
def many_users() -> pl.DataFrame:
    """Feature-shaped rows of 200 users, 20 of them with a takeover."""
    users = [u for u in range(200) for _ in range(3)]
    labels = [u < 20 and i == 0 for u in range(200) for i in range(3)]  # noqa: PLR2004
    return pl.DataFrame({"user_id": users, "label_ato": labels})


def test_split_built_by_the_pipeline_passes(many_users: pl.DataFrame) -> None:
    split = assign_split(many_users.lazy())
    checked = validate_split(many_users.lazy(), split)
    assert checked.height == 200  # noqa: PLR2004


def test_user_without_a_side_is_an_error(many_users: pl.DataFrame) -> None:
    split = assign_split(many_users.lazy()).filter(pl.col("user_id") != 5)  # noqa: PLR2004
    assert "not_nullable" in _split_failures(many_users, split)


def test_user_not_in_the_feature_table_is_an_error(many_users: pl.DataFrame) -> None:
    split = assign_split(many_users.lazy())
    extra = pl.DataFrame({"user_id": [999], "has_takeover": [False], "split": ["dev"]})
    assert "not_nullable" in _split_failures(many_users, pl.concat([split, extra]))


def test_takeover_flag_must_agree_with_the_labels(many_users: pl.DataFrame) -> None:
    split = assign_split(many_users.lazy()).with_columns(
        pl.when(pl.col("user_id") == 0)
        .then(pl.lit(value=False))
        .otherwise(pl.col("has_takeover"))
        .alias("has_takeover")
    )
    assert "takeover_flag_agrees_with_labels" in _split_failures(many_users, split)


def test_moving_a_user_breaks_the_test_share(many_users: pl.DataFrame) -> None:
    split = assign_split(many_users.lazy())
    moved = split.filter(pl.col("split") == "test")["user_id"][0]
    split = split.with_columns(
        pl.when(pl.col("user_id") == moved)
        .then(pl.lit("dev"))
        .otherwise(pl.col("split"))
        .alias("split")
    )
    assert "test_share_per_stratum" in _split_failures(many_users, split)


def test_unknown_side_is_an_error(many_users: pl.DataFrame) -> None:
    split = assign_split(many_users.lazy()).with_columns(
        pl.when(pl.col("user_id") == 0)
        .then(pl.lit("holdout"))
        .otherwise(pl.col("split"))
        .alias("split")
    )
    assert "isin(['dev', 'test'])" in _split_failures(many_users, split)
