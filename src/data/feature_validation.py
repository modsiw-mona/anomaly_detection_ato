"""Data validation rules for the feature table and the development / test split (Pandera, polars).

The intermediate dataset has its own schema (:mod:`data.validation`). The feature pipeline also
validates what it builds from it, before the next step uses it:

* **Feature table** (:func:`build_feature_schema`): every column has the type the builder writes, a
  value in its range (shares in [0, 1], counts never negative, hour and weekday), and nulls only
  where a feature is undefined: a user's **first** successful login has no earlier login to compare
  with. The rows are successful logins of real accounts inside the label window, in time order, and
  the relations that follow from the definitions hold, for example "a first login has no earlier
  logins" or "failures in the last 24 hours include those of the last hour".
* **Split** (:func:`validate_split`): every user of the feature table is on exactly one side, the
  takeover flag of each user agrees with the labels, and each stratum (users with and without a
  takeover) has the pre-registered test share.

Each rule is a definition, so it holds for any dataset, not a measurement of this one.

Usage::

    validate_features(frame, config)  # raises pandera.errors.SchemaErrors listing every failure
    validate_split(features, split)  # the same, for the user -> side assignment
"""

import datetime as dt
from collections.abc import Callable

import pandera.polars as pa
import polars as pl
from pandera.api.polars.types import PolarsData

from data.datasets import DatasetConfig
from data.features import (
    CONTEXT_COLUMNS,
    FAILURE_WINDOWS,
    FEATURE_GROUPS,
    KEY_SIGNALS,
    LABEL,
    NETWORK_SIGNALS,
    NOVELTY_SIGNALS,
    RARITY_SIGNALS,
    TIME,
    USER,
)
from data.intermediate import EVENT_ID, storage_dtype
from data.split import SEED, TEST_SHARE, assign_split

FrameCheck = Callable[[PolarsData], pl.LazyFrame]

FIRST = "first_success"
COUNTS = (
    "prior_successes",
    "ip_prior_users",
    "asn_prior_users",
    "user_agent_prior_users",
    *FAILURE_WINDOWS.values(),
    "user_prior_countries",
)
BURST_COUNTS = ("asn_attempts_prev24h", "asn_fails_prev24h")  # unknown when the network is
SHARES = (
    *(f"global_share_{s}" for s in RARITY_SIGNALS),
    *(f"{s}_prior_{k}_share" for s in NETWORK_SIGNALS for k in ("fail", "attack")),
)
FLAGS = (  # always known
    FIRST,
    "attack_ip",
    "region_missing",
    "city_missing",
    "ip_is_placeholder",
    "asn_is_placeholder",
    "rtt_missing",
)
UNDEFINED_ON_FIRST = (  # nothing earlier to compare with: null on a first login, known after it
    "hours_since_prev_success",
    *(f"new_{s}" for s in NOVELTY_SIGNALS),
    "n_new_key_signals",
)
TRAVEL = "country_change_1h"  # unknown on a first login; may also be when a country is unknown
SHARED_BY = {
    "ip": "ip_prior_users",
    "asn": "asn_prior_users",
    "user_agent": "user_agent_prior_users",
}


def _between(low: float, high: float) -> pa.Check:
    return pa.Check.in_range(low, high)


def _feature_columns() -> dict[str, pa.Column]:
    columns: dict[str, pa.Column] = {}
    for name in COUNTS:
        columns[name] = pa.Column(pl.UInt32, required=False)  # unsigned: never negative
    for name in BURST_COUNTS:
        columns[name] = pa.Column(pl.UInt32, nullable=True, required=False)
    for name in SHARES:
        columns[name] = pa.Column(pl.Float64, _between(0, 1), nullable=True, required=False)
    for name in FLAGS:
        columns[name] = pa.Column(pl.Boolean, required=False)
    for signal in NOVELTY_SIGNALS:
        columns[f"new_{signal}"] = pa.Column(pl.Boolean, nullable=True, required=False)
    columns |= {
        "hours_since_prev_success": pa.Column(
            pl.Float64, pa.Check.ge(0), nullable=True, required=False
        ),
        "n_new_key_signals": pa.Column(
            pl.UInt8, pa.Check.le(len(KEY_SIGNALS)), nullable=True, required=False
        ),
        TRAVEL: pa.Column(pl.Boolean, nullable=True, required=False),
        "hour": pa.Column(pl.Int8, _between(0, 23), required=False),
        "weekday": pa.Column(pl.Int8, _between(1, 7), required=False),  # Monday = 1
        "device_type": pa.Column(pl.Categorical, nullable=True, required=False),
        "rtt_ms": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True, required=False),
    }
    known = {c for group in FEATURE_GROUPS.values() for c in group}
    missing = known - columns.keys()
    if missing:  # a new feature must get a rule here
        raise RuntimeError(f"no validation rule for features {sorted(missing)}")
    return columns


def _rule(rule: pl.Expr, needs: tuple[str, ...]) -> FrameCheck:
    """A row-wise rule over several columns; it passes when a needed column is absent."""

    def check(data: PolarsData) -> pl.LazyFrame:
        if not set(needs) <= set(data.lazyframe.collect_schema()):
            return data.lazyframe.select(pl.lit(value=True))
        return data.lazyframe.select(rule.fill_null(value=True))

    return check


def _frame_checks() -> list[pa.Check]:
    first = pl.col(FIRST)
    rules: dict[str, tuple[pl.Expr, tuple[str, ...]]] = {
        "first_login_has_no_earlier_logins": (
            first == (pl.col("prior_successes") == 0),
            (FIRST, "prior_successes"),
        ),
        "first_login_has_no_earlier_countries": (
            ~first | (pl.col("user_prior_countries") == 0),
            (FIRST, "user_prior_countries"),
        ),
        "rows_in_time_order": (pl.col(TIME).diff() >= pl.duration(), (TIME,)),
        "fails_24h_include_fails_1h": (
            pl.col("fails_24h") >= pl.col("fails_1h"),
            ("fails_1h", "fails_24h"),
        ),
        "network_fails_within_attempts": (
            pl.col("asn_fails_prev24h") <= pl.col("asn_attempts_prev24h"),
            ("asn_fails_prev24h", "asn_attempts_prev24h"),
        ),
        "rtt_missing_flags_null_rtt": (
            pl.col("rtt_missing") == pl.col("rtt_ms").is_null(),
            ("rtt_missing", "rtt_ms"),
        ),
        "travel_null_on_first_login": (~first | pl.col(TRAVEL).is_null(), (FIRST, TRAVEL)),
        **{
            f"{column}_null_exactly_on_first_login": (
                pl.col(column).is_null() == first,
                (FIRST, column),
            )
            for column in UNDEFINED_ON_FIRST
        },
        **{  # an earlier login with the value <-> an earlier account with it
            f"{signal}_seen_before_iff_shared": (
                (pl.col(f"global_share_{signal}") > 0) == (pl.col(users) > 0),
                (f"global_share_{signal}", users),
            )
            for signal, users in SHARED_BY.items()
        },
    }
    checks = [pa.Check(_rule(rule, needs), name=name) for name, (rule, needs) in rules.items()]
    return [
        *checks,
        pa.Check(_counts_key_signals, name="n_new_key_signals_counts_them"),
        pa.Check(_attack_share_known_with_fail_share, name="attack_share_known_with_fail_share"),
    ]


def _counts_key_signals(data: PolarsData) -> pl.LazyFrame:
    """The count of new key signals adds up the key signals the dataset has (any of them)."""
    present = set(data.lazyframe.collect_schema())
    key = [f"new_{s}" for s in KEY_SIGNALS if f"new_{s}" in present]
    if "n_new_key_signals" not in present or not key:
        return data.lazyframe.select(pl.lit(value=True))
    counted = pl.col("n_new_key_signals") == pl.sum_horizontal(key).cast(pl.UInt8)
    return data.lazyframe.select(counted.fill_null(value=True))


def _attack_share_known_with_fail_share(data: PolarsData) -> pl.LazyFrame:
    """Both shares of a network come from the same earlier attempts: known or unknown together."""
    present = set(data.lazyframe.collect_schema())
    pairs = [
        (pl.col(f"{s}_prior_attack_share"), pl.col(f"{s}_prior_fail_share"))
        for s in NETWORK_SIGNALS
        if {f"{s}_prior_attack_share", f"{s}_prior_fail_share"} <= present
    ]
    if not pairs:
        return data.lazyframe.select(pl.lit(value=True))
    return data.lazyframe.select(pl.all_horizontal(a.is_null() == f.is_null() for a, f in pairs))


def _not_in(values: list[int] | list[str]) -> pa.Check:
    return pa.Check(
        lambda data: data.lazyframe.select(~pl.col(data.key).is_in(values)),
        name="not_a_catch_all_account",
    )


def _in_window(start: dt.date, end: dt.date) -> pa.Check:
    start_at, end_at = dt.datetime.combine(start, dt.time()), dt.datetime.combine(end, dt.time())
    return pa.Check(
        lambda data: data.lazyframe.select(pl.col(data.key).is_between(start_at, end_at, "left")),
        name="label_window",
    )


def build_feature_schema(config: DatasetConfig) -> pa.DataFrameSchema:
    """Pandera schema for the feature table of ``config``."""
    catch_all: list[int] | list[str] = list(config.catch_all_user_ids)
    if storage_dtype(USER, config) == pl.Int64():
        catch_all = [int(i) for i in catch_all]
    user_checks = [_not_in(catch_all)] if catch_all else []
    columns: dict[str, pa.Column] = {
        EVENT_ID: pa.Column(pl.UInt32, unique=True),
        USER: pa.Column(storage_dtype(USER, config), checks=user_checks),
        TIME: pa.Column(
            storage_dtype(TIME, config),
            checks=[_in_window(*config.label_window)] if config.label_window else [],
        ),
        LABEL: pa.Column(pl.Boolean),
        **{
            c: pa.Column(storage_dtype(c, config), nullable=True, required=False)
            for c in CONTEXT_COLUMNS
        },
        **_feature_columns(),
    }
    return pa.DataFrameSchema(
        columns, checks=_frame_checks(), strict=True, name=f"{config.name}_features"
    )


def validate_features(frame: pl.DataFrame, config: DatasetConfig) -> pl.DataFrame:
    """Validate the feature table against :func:`build_feature_schema`, collecting every failure.

    Raises:
        pandera.errors.SchemaErrors: If any rule fails (all failures are listed).
    """
    validated: pl.DataFrame = build_feature_schema(config).validate(frame, lazy=True)
    return validated


def validate_split(
    features: pl.LazyFrame,
    split: pl.DataFrame,
    *,
    test_share: float = TEST_SHARE,
    seed: int = SEED,
) -> pl.DataFrame:
    """Check the user -> side assignment against the feature table it was computed from.

    Besides coverage, flags and shares, every user must be on the side the pre-registered rule
    (:func:`data.split.assign_split` with ``test_share`` and ``seed``) gives, so an edited,
    re-seeded or stale split is caught.

    Returns:
        One row per user: ``user_id``, ``labelled`` (has a takeover in the feature table, null if
        the user is not in it), ``has_takeover``, ``split`` (null if the user has no side) and
        ``registered`` (the side of the pre-registered rule).

    Raises:
        pandera.errors.SchemaErrors: If any rule fails (all failures are listed).
    """
    users = features.group_by(USER).agg(pl.col(LABEL).any().alias("labelled")).collect()
    registered = assign_split(features, test_share=test_share, seed=seed).select(
        USER, pl.col("split").alias("registered")
    )
    frame = users.join(split, on=USER, how="full", coalesce=True).join(
        registered, on=USER, how="left"
    )
    expected = {  # the same rounding as data.split.assign_split
        flag: round(test_share * frame.filter(pl.col("labelled") == flag).height)
        for flag in (True, False)
    }

    def stratum_share(data: PolarsData) -> pl.LazyFrame:
        in_test = (pl.col("split") == "test").sum().over("labelled")
        wanted = pl.col("labelled").replace_strict(expected, return_dtype=pl.Int64)
        return data.lazyframe.select((in_test == wanted).fill_null(value=True))

    def pre_registered(data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select(
            (pl.col("split") == pl.col("registered")).fill_null(value=True)
        )

    def agrees(data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select(
            (pl.col("has_takeover") == pl.col("labelled")).fill_null(value=True)
        )

    schema = pa.DataFrameSchema(
        {
            USER: pa.Column(unique=True),
            "labelled": pa.Column(pl.Boolean),  # null: a user without logins in the feature table
            "has_takeover": pa.Column(pl.Boolean),
            "split": pa.Column(pl.String, pa.Check.isin(["dev", "test"])),  # null: no side
            "registered": pa.Column(pl.String, nullable=True),  # null: user not in the features
        },
        checks=[
            pa.Check(agrees, name="takeover_flag_agrees_with_labels"),
            pa.Check(stratum_share, name="test_share_per_stratum"),
            pa.Check(pre_registered, name="matches_the_pre_registered_assignment"),
        ],
        strict=True,
        name="split",
    )
    validated: pl.DataFrame = schema.validate(frame, lazy=True)
    return validated
