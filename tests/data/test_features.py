"""Tests for the behavioural features: hand-checked values and the no-future-information guarantee."""

import datetime as dt
from typing import Any

import numpy as np
import polars as pl
import pytest

from data.datasets import DatasetConfig
from data.features import DEFAULT_GROUPS, FEATURE_GROUPS, build_features, feature_columns
from tests.data.test_intermediate import SOURCE

ALL_COLUMNS = {
    "user_id": "U", "timestamp": "T", "login_success": "S", "ip": "IP", "asn": "ASN",
    "device_type": "D", "user_agent": "UA", "country": "C", "region": "R", "city": "CI",
    "os": "OS", "browser": "B", "rtt_ms": "RTT",
}  # fmt: skip
CATCH_ALL = 99


def _config(**overrides: Any) -> DatasetConfig:
    return DatasetConfig.model_validate(
        {
            "name": "toy",
            "description": "Toy.",
            "source": SOURCE,
            "columns": ALL_COLUMNS,
            "labels": {"label_ato": "ATO", "label_attack_ip": "AIP"},
            "identifier_dtypes": {"user_id": "Int64"},
            "catch_all_user_ids": [str(CATCH_ALL)],
            **overrides,
        }
    )


def _events(rows: list[dict[str, Any]]) -> pl.LazyFrame:
    """An intermediate-shaped frame; unspecified fields get neutral defaults."""
    defaults = {
        "ip": "1.1.1.1", "asn": "100", "device_type": "mobile", "user_agent": "UA1", "country": "NO",
        "region": "R1", "city": "C1", "os": "iOS", "browser": "Safari", "rtt_ms": None,
        "label_ato": False, "label_attack_ip": False, "ip_is_placeholder": False,
        "asn_is_placeholder": False,
    }  # fmt: skip
    full = [{"event_id": i, **defaults, **row} for i, row in enumerate(rows)]
    return pl.LazyFrame(full).with_columns(
        pl.col("event_id").cast(pl.UInt32),
        pl.col("user_id").cast(pl.Int64),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        pl.col("rtt_ms").cast(pl.Float64),
        *(pl.col(c).cast(pl.Categorical) for c in ("ip", "asn", "device_type", "country", "region", "city", "os", "browser")),
    )  # fmt: skip


def at(minutes: int) -> dt.datetime:
    return dt.datetime(2020, 3, 1) + dt.timedelta(minutes=minutes)


HAND_BUILT = [
    {"user_id": 1, "timestamp": dt.datetime(2020, 1, 15), "login_success": True},  # before window
    {"user_id": 1, "timestamp": at(0), "login_success": True},
    {"user_id": 2, "timestamp": at(10), "login_success": False},
    {"user_id": 2, "timestamp": at(20), "login_success": True, "device_type": "desktop"},
    {"user_id": 1, "timestamp": at(30), "login_success": False},
    {"user_id": CATCH_ALL, "timestamp": at(35), "login_success": True, "country": "XX"},
    {"user_id": 1, "timestamp": at(40), "login_success": True, "country": "SE", "asn": "200",
     "ip": "2.2.2.2", "label_ato": True, "label_attack_ip": True},
    {"user_id": 1, "timestamp": at(50), "login_success": True},
    {"user_id": 1, "timestamp": dt.datetime(2020, 12, 5), "login_success": True},  # after window
]  # fmt: skip
WINDOW = {"label_window": ["2020-02-01", "2020-12-01"]}
# Event ids of HAND_BUILT, named by role
USER_1_IN_WINDOW, USER_2_FIRST, TAKEOVER, USER_1_RETURNS = 1, 3, 6, 7
NEW_KEY_SIGNALS_IN_TAKEOVER = 2  # country + asn
SUCCESSES_BEFORE_TAKEOVER = 2  # January + event 1


@pytest.fixture
def hand_built() -> pl.DataFrame:
    return build_features(_events(HAND_BUILT), _config(**WINDOW)).collect()


def test_population_is_successful_logins_of_real_accounts_inside_the_window(
    hand_built: pl.DataFrame,
) -> None:
    assert hand_built["event_id"].to_list() == [
        USER_1_IN_WINDOW,
        USER_2_FIRST,
        TAKEOVER,
        USER_1_RETURNS,
    ]
    assert CATCH_ALL not in hand_built["user_id"].to_list()
    assert hand_built["timestamp"].is_sorted()


def test_history_counts_logins_before_the_window(hand_built: pl.DataFrame) -> None:
    row = hand_built.filter(pl.col("event_id") == USER_1_IN_WINDOW).row(0, named=True)
    assert row["prior_successes"] == 1, "the January login is history"
    assert row["first_success"] is False
    assert row["new_country"] is False


def test_first_success_has_no_novelty(hand_built: pl.DataFrame) -> None:
    row = hand_built.filter(pl.col("event_id") == USER_2_FIRST).row(0, named=True)
    assert row["first_success"] is True
    assert row["prior_successes"] == 0
    assert row["new_country"] is None
    assert row["n_new_key_signals"] is None
    assert row["hours_since_prev_success"] is None


def test_novelty_and_failures_for_the_takeover(hand_built: pl.DataFrame) -> None:
    row = hand_built.filter(pl.col("event_id") == TAKEOVER).row(0, named=True)
    assert row["label_ato"] is True
    assert row["attack_ip"] is True
    assert (row["new_country"], row["new_asn"], row["new_ip"]) == (True, True, True)
    assert (row["new_device_type"], row["new_os"], row["new_browser"]) == (False, False, False)
    assert row["n_new_key_signals"] == NEW_KEY_SIGNALS_IN_TAKEOVER
    assert row["fails_1h"] == 1, "the failure 10 minutes earlier on the same account"
    assert row["prior_successes"] == SUCCESSES_BEFORE_TAKEOVER
    assert row["hours_since_prev_success"] == pytest.approx(40 / 60)


def test_returning_to_a_known_value_is_not_new(hand_built: pl.DataFrame) -> None:
    row = hand_built.filter(pl.col("event_id") == USER_1_RETURNS).row(0, named=True)
    assert row["new_country"] is False
    assert row["new_asn"] is False


def test_global_rarity_and_sharing_ignore_the_catch_all(hand_built: pl.DataFrame) -> None:
    row = hand_built.filter(pl.col("event_id") == TAKEOVER).row(0, named=True)
    # earlier successes of real accounts: events 0, 1, 3 (all NO); the catch-all's XX is ignored
    assert row["global_share_country"] == 0.0, "SE never seen before"
    assert row["ip_prior_users"] == 0
    shared = hand_built.filter(pl.col("event_id") == USER_2_FIRST).row(0, named=True)
    assert shared["global_share_country"] == 1.0, "every earlier success was NO"
    assert shared["ip_prior_users"] == 1, "user 1 used this IP before user 2"


SUCCESS_RATE, ATTACK_IP_RATE = 0.6, 0.1


def _random_events(n: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    minutes = np.cumsum(rng.integers(1, 90, size=n))
    return [
        {
            "user_id": int(rng.choice([1, 2, 3, 4, 5, 6, CATCH_ALL])),
            "timestamp": at(int(m)),
            "login_success": bool(rng.random() < SUCCESS_RATE),
            "country": str(rng.choice(["NO", "SE", "US", "RU"])),
            "asn": str(rng.choice(["100", "200", "300"])),
            "ip": str(rng.choice(["1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4"])),
            "device_type": str(rng.choice(["mobile", "desktop"])),
            "label_attack_ip": bool(rng.random() < ATTACK_IP_RATE),
        }
        for m in minutes
    ]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_feature_uses_information_from_the_future(seed: int) -> None:
    events = _random_events(160, seed)
    config = _config()
    full = build_features(_events(events), config).collect()
    for event_id in full["event_id"].gather_every(7).to_list():
        cut = _events(events[: event_id + 1])  # everything up to and including this login
        truncated = build_features(cut, config).collect().filter(pl.col("event_id") == event_id)
        expected = full.filter(pl.col("event_id") == event_id)
        assert truncated.equals(expected), f"features of event {event_id} depend on later events"


def test_feature_groups() -> None:
    columns = feature_columns()
    assert len(columns) == len(set(columns))
    assert "attack_ip" in columns
    assert not set(FEATURE_GROUPS["artefacts"]) & set(columns), "artefacts only behind ablations"
    assert not set(FEATURE_GROUPS["rtt"]) & set(columns), "RTT only behind ablations"
    assert set(DEFAULT_GROUPS) <= FEATURE_GROUPS.keys()
