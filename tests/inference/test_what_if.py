"""Rules that keep a what-if login consistent (inference.what_if)."""

import os
from pathlib import Path

import polars as pl
import pytest

from data.features import build_features
from inference.what_if import (
    GAPS,
    IMPLIED_BY,
    NEW_IF_NEVER_SEEN,
    SIGNALS,
    TRAVEL_GAP,
    UNKNOWN_IF,
    gap_of,
    required_signals,
    with_attack_share,
)
from tests.data.test_features import _config, _events, _random_events

RETURNING = {
    "prior_successes": 3,
    "hours_since_prev_success": 50.0,
    "region_missing": False,
    "city_missing": False,
    "global_share_country": 0.2,
    "global_share_asn": 0.01,
    "asn_prior_users": 40,
    "global_share_ip": 0.001,
    "ip_prior_users": 2,
    "global_share_device_type": 0.3,
    "global_share_os": 0.1,
    "global_share_browser": 0.05,
    "global_share_user_agent": 0.001,
    "user_agent_prior_users": 9,
    "attack_ip": False,
    "ip_prior_attack_share": 0.0,
    **{s: False for s in SIGNALS if s != "attack_ip"},
}


def _login(**changes: object) -> pl.DataFrame:
    return pl.DataFrame([{**RETURNING, **changes}])


def test_first_login_has_no_history_rules() -> None:
    login = _login(prior_successes=0)
    assert required_signals(login, login, {}, None) == {}


def test_never_seen_network_is_new_for_the_account_and_brings_a_new_ip() -> None:
    login = _login()
    located = _login(asn_prior_users=0, global_share_asn=0.0)
    required = required_signals(login, located, {}, "1 day to 1 week")
    assert required["new_asn"][0] is True
    assert required["new_ip"][0] is True, "a new network always means a new IP address"


def test_implied_signals_follow_their_cause() -> None:
    login = _login()
    required = required_signals(login, login, {"new_country": True, "new_browser": True}, None)
    assert required["new_region"][0] is True
    assert required["new_user_agent"][0] is True
    assert "new_city" not in required, "a new region does not always mean a new city"


def test_no_region_rule_when_the_region_is_unknown() -> None:
    login = _login(region_missing=True)
    assert "new_region" not in required_signals(login, login, {"new_country": True}, None)


@pytest.mark.parametrize(("gap", "fixed"), [(TRAVEL_GAP, False), ("1 day to 1 week", True)])
def test_travel_needs_a_recent_previous_login(gap: str, fixed: bool) -> None:
    login = _login()
    required = required_signals(login, login, {"country_change_1h": True}, gap)
    assert ("country_change_1h" in required) is fixed
    if fixed:
        assert required["country_change_1h"][0] is False


def test_gap_ranges() -> None:
    assert gap_of(None) is None
    assert gap_of(0.5) == TRAVEL_GAP
    assert gap_of(1.0) == "1 hour to 1 day"
    assert gap_of(10_000.0) == "over a month"
    assert next(iter(GAPS)) == TRAVEL_GAP


def test_attack_share_follows_the_attack_flag() -> None:
    assert with_attack_share(_login(attack_ip=True))["ip_prior_attack_share"][0] == 1.0
    assert with_attack_share(_login(ip_prior_attack_share=1.0))["ip_prior_attack_share"][0] == 0.0
    never_seen = _login(attack_ip=True, ip_prior_attack_share=None)
    assert with_attack_share(never_seen)["ip_prior_attack_share"][0] is None


def test_rules_never_contradict_logins_built_by_the_feature_pipeline() -> None:
    """Every returning login the pipeline builds satisfies the rules that follow from the feature definitions.

    The "always together" rules come from the data (an IP belongs to one network, a user agent string encodes its
    device, OS and browser, a region belongs to one country); random toy logins need not follow them, so they are
    checked on the real data below.
    """
    frame = build_features(_events(_random_events(3_000, seed=11)), _config()).collect()
    returning = frame.filter(pl.col("prior_successes") > 0)
    assert returning.height > 100  # noqa: PLR2004
    for row in range(returning.height):
        login = returning[row]
        switches = {s: bool(login[s][0]) for s in SIGNALS if s in login.columns}
        gap = gap_of(login["hours_since_prev_success"][0])
        for signal, (value, reason) in required_signals(login, login, switches, gap).items():
            if signal in IMPLIED_BY:
                continue
            assert switches[signal] == value, (signal, reason, login.row(0, named=True))


DATA_DIR = Path(os.environ.get("ATO_DATA_DIR", Path(__file__).parents[2] / "data"))
FEATURES = DATA_DIR / "04_feature" / "rba_features.parquet"


@pytest.mark.skipif(not FEATURES.is_file(), reason="needs the RBA feature table (make features)")
def test_every_rule_holds_on_the_rba_data() -> None:
    """No returning RBA login breaks a rule, so a what-if login that follows them stays realistic."""
    columns = {c for cs in NEW_IF_NEVER_SEEN.values() for c in cs} | set(UNKNOWN_IF.values())
    columns |= {*SIGNALS, *NEW_IF_NEVER_SEEN, "prior_successes", "hours_since_prev_success"}
    columns |= {"ip_prior_attack_share"}
    data = pl.scan_parquet(FEATURES).select(sorted(columns)).filter(pl.col("prior_successes") > 0)
    broken = [
        *(
            ~pl.col(flag) & pl.any_horizontal(pl.col(c) == 0 for c in cols)
            for flag, cols in NEW_IF_NEVER_SEEN.items()
        ),
        *(
            pl.any_horizontal(pl.col(c) for c in causes)
            & ~pl.col(flag)
            & ~pl.col(UNKNOWN_IF.get(flag, "prior_successes")).cast(pl.Boolean)
            for flag, causes in IMPLIED_BY.items()
        ),
        pl.col("country_change_1h") & (pl.col("hours_since_prev_success") >= GAPS[TRAVEL_GAP]),
        pl.col("attack_ip") & (pl.col("ip_prior_attack_share") != 1),
        ~pl.col("attack_ip") & (pl.col("ip_prior_attack_share") != 0),
    ]
    counts = data.select(rule.fill_null(False).sum().alias(str(i)) for i, rule in enumerate(broken))
    assert set(counts.collect().row(0)) == {0}
