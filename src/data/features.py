"""Behavioural features for every successful login, computed only from the past.

Stage ``data/02_intermediate`` -> ``data/04_feature``. One row per **successful login of a real
account** (catch-all IDs excluded) inside the dataset's label window. Every feature of a login at
time *t* uses only events strictly before *t*, computed over the **whole** timeline, so months
outside the label window still serve as history. ``tests/data/test_features.py`` checks this by
recomputing each row's features on data cut at that row and requiring identical values.

Feature groups (see :data:`FEATURE_GROUPS`), each motivated by the EDA (notebook 3-analysis):

* ``history`` — earlier successful logins of the user and time since the last one; first logins are
  where per-user behaviour says nothing (39% of takeovers).
* ``user_novelty`` — first time *this user* succeeds with this country / ASN / IP / device / OS /
  browser / user agent / region / city, and how many key signals are new at once. Null on a user's
  first successful login, where novelty is undefined.
* ``global_rarity`` — share of all earlier successful logins (all users) with this value: the signal
  available even on a first login.
* ``sharing`` — distinct accounts that succeeded from this IP / ASN before.
* ``failures`` — failed attempts on the same account in the hour / day before.
* ``attack_ip``, ``missingness``, ``device``, ``time`` — context of the attempt.
* ``artefacts`` (placeholder flags) and ``rtt`` — kept for ablations only (EDA: artefacts of the
  synthesis; RTT present for one takeover).
"""

import argparse
import datetime as dt
import logging
from pathlib import Path

import polars as pl

from data.datasets import DatasetConfig, find_repo_root, load_dataset_config
from data.intermediate import EVENT_ID, intermediate_path, mapped_fields

logger = logging.getLogger(__name__)

USER, TIME, SUCCESS, LABEL = "user_id", "timestamp", "login_success", "label_ato"
NOVELTY_SIGNALS = (
    "country", "asn", "ip", "device_type", "os", "browser", "user_agent", "region", "city",
)  # fmt: skip
KEY_SIGNALS = ("country", "asn", "device_type", "os", "browser")  # EDA: strongest, not redundant
RARITY_SIGNALS = ("country", "asn", "ip", "device_type", "os", "browser", "user_agent")
SHARING_SIGNALS = ("ip", "asn")
FAILURE_WINDOWS = {"1h": "fails_1h", "24h": "fails_24h"}

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "history": ("prior_successes", "first_success", "hours_since_prev_success"),
    "user_novelty": (*(f"new_{s}" for s in NOVELTY_SIGNALS), "n_new_key_signals"),
    "global_rarity": tuple(f"global_share_{s}" for s in RARITY_SIGNALS),
    "sharing": tuple(f"{s}_prior_users" for s in SHARING_SIGNALS),
    "failures": tuple(FAILURE_WINDOWS.values()),
    "attack_ip": ("attack_ip",),
    "missingness": ("region_missing", "city_missing"),
    "device": ("device_type",),
    "time": ("hour", "weekday"),
    "artefacts": ("ip_is_placeholder", "asn_is_placeholder"),
    "rtt": ("rtt_ms", "rtt_missing"),
}
DEFAULT_GROUPS = (
    "history", "user_novelty", "global_rarity", "sharing", "failures", "attack_ip",
    "missingness", "device", "time",
)  # fmt: skip
ID_COLUMNS = (EVENT_ID, USER, TIME)


def feature_columns(groups: tuple[str, ...] = DEFAULT_GROUPS) -> list[str]:
    """Feature column names of the given groups, in group order."""
    return [column for group in groups for column in FEATURE_GROUPS[group]]


def _catch_all_ids(config: DatasetConfig) -> list[int] | list[str]:
    ids = config.catch_all_user_ids
    return [int(i) for i in ids] if config.identifier_dtypes.get(USER) == "Int64" else list(ids)


def real_account_attempts(intermediate: pl.LazyFrame, config: DatasetConfig) -> pl.LazyFrame:
    """All attempts on real accounts (catch-all IDs removed), in user and time order."""
    return intermediate.filter(~pl.col(USER).is_in(_catch_all_ids(config))).sort(
        USER, TIME, EVENT_ID
    )


def _failures_before(attempts: pl.LazyFrame) -> pl.LazyFrame:
    """Failed attempts on the same account in each window ending at every attempt (event_id keyed).

    The window includes the attempt itself; for a successful login that adds nothing, because the
    login itself is not a failure.
    """
    failed = attempts.with_columns((~pl.col(SUCCESS)).cast(pl.UInt32).alias("_failed"))
    result = failed.select(EVENT_ID)
    for period, name in FAILURE_WINDOWS.items():
        window = failed.rolling(index_column=TIME, period=period, group_by=USER).agg(
            pl.col("_failed").sum().alias(name), pl.col(EVENT_ID).last()
        )
        result = result.join(window.select(EVENT_ID, name), on=EVENT_ID, how="left")
    return result


def _user_features(successes: pl.LazyFrame, available: set[str]) -> pl.LazyFrame:
    """History and novelty against the user's own earlier successful logins (user/time order)."""
    first = pl.col("prior_successes") == 0
    novelty = [
        pl.when(first)
        .then(None)
        .otherwise(pl.col(s).is_first_distinct().over(USER))
        .alias(f"new_{s}")
        for s in NOVELTY_SIGNALS
        if s in available
    ]
    previous = pl.col(TIME).shift(1).over(USER)
    frame = successes.with_columns(
        pl.int_range(pl.len()).over(USER).cast(pl.UInt32).alias("prior_successes")
    )
    frame = frame.with_columns(
        first.alias("first_success"),
        ((pl.col(TIME) - previous).dt.total_seconds() / 3600).alias("hours_since_prev_success"),
        *novelty,
    )
    key = [f"new_{s}" for s in KEY_SIGNALS if s in available]
    return frame.with_columns(
        pl.when(first)
        .then(None)
        .otherwise(pl.sum_horizontal(key))
        .cast(pl.UInt8)
        .alias("n_new_key_signals")
    )


def _global_features(successes: pl.LazyFrame, available: set[str]) -> pl.LazyFrame:
    """Rarity and sharing across all users' earlier successful logins (global time order)."""
    ordered = successes.sort(TIME, EVENT_ID)
    prior_total = pl.int_range(pl.len())
    rarity = [  # 0/0 on the very first login is undefined: null, not NaN
        (pl.int_range(pl.len()).over(s) / prior_total).fill_nan(None).alias(f"global_share_{s}")
        for s in RARITY_SIGNALS
        if s in available
    ]
    sharing = []
    for s in SHARING_SIGNALS:
        if s in available:
            new_pair = pl.struct(s, USER).is_first_distinct().cast(pl.UInt32)
            sharing.append((new_pair.cum_sum().over(s) - new_pair).alias(f"{s}_prior_users"))
    return ordered.select(EVENT_ID, *rarity, *sharing)


def build_features(intermediate: pl.LazyFrame, config: DatasetConfig) -> pl.LazyFrame:
    """Feature table: one row per successful login of a real account inside the label window."""
    available = set(mapped_fields(config))
    attempts = real_account_attempts(intermediate, config)
    successes = attempts.filter(pl.col(SUCCESS))

    frame = (
        _user_features(successes, available)
        .join(_global_features(successes, available), on=EVENT_ID, how="left")
        .join(_failures_before(attempts), on=EVENT_ID, how="left")
    )
    context = [
        pl.col("label_attack_ip").alias("attack_ip"),
        pl.col("region").is_null().alias("region_missing"),
        pl.col("city").is_null().alias("city_missing"),
        pl.col(TIME).dt.hour().alias("hour"),
        pl.col(TIME).dt.weekday().alias("weekday"),
        pl.col("rtt_ms").is_null().alias("rtt_missing"),
    ]
    frame = frame.with_columns(c for c in context if set(c.meta.root_names()) <= available)
    if config.label_window:
        start, end = (dt.datetime.combine(d, dt.time()) for d in config.label_window)
        frame = frame.filter((pl.col(TIME) >= start) & (pl.col(TIME) < end))

    present = [c for c in feature_columns(tuple(FEATURE_GROUPS)) if c in frame.collect_schema()]
    return frame.select(*ID_COLUMNS, LABEL, *present).sort(TIME, EVENT_ID)


def features_path(config: DatasetConfig, data_dir: Path) -> Path:
    """Where the feature table for ``config`` lives."""
    return data_dir / "04_feature" / f"{config.name}_features.parquet"


def write_features(config: DatasetConfig, data_dir: Path) -> Path:
    """Build the feature table from the intermediate dataset and write it (``.part`` then rename)."""
    out = features_path(config, data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".part")
    intermediate = pl.scan_parquet(intermediate_path(config, data_dir))
    build_features(intermediate, config).collect().write_parquet(partial, compression="zstd")
    partial.replace(out)
    logger.info("Wrote %s", out.name)
    return out


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="rba", help="Name of conf/dataset/<name>.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_dataset_config(args.dataset)
    out = write_features(config, find_repo_root() / "data")
    print(out.relative_to(find_repo_root()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
