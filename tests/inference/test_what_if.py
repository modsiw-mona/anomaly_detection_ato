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
    typical_row,
    with_attack_share,
)
from tests.data.test_features import _config, _events, _random_events

RETURNING = {
    "prior_successes": 3,
    "user_prior_countries": 2,
    "asn_prior_attack_share": 0.3,
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


def test_new_country_soon_after_the_previous_login_is_travel() -> None:
    login = _login(user_prior_countries=2)
    required = required_signals(login, login, {"new_country": True}, TRAVEL_GAP)
    assert required["country_change_1h"][0] is True


def test_with_one_earlier_country_travel_follows_new_country() -> None:
    login = _login(user_prior_countries=1)
    on = required_signals(login, login, {"new_country": True}, TRAVEL_GAP)
    assert on["country_change_1h"][0] is True
    assert on["new_region"][0] is True
    off = required_signals(login, login, {"country_change_1h": True}, TRAVEL_GAP)
    assert off["country_change_1h"][0] is False, "no new country, so no country change"


@pytest.mark.parametrize("countries", [1, 2])
@pytest.mark.parametrize("gap", [TRAVEL_GAP, "1 day to 1 week"])
def test_new_country_turned_on_by_the_viewer_is_never_locked(countries: int, gap: str) -> None:
    login = _login(user_prior_countries=countries)
    assert "new_country" not in required_signals(login, login, {"new_country": True}, gap)


def test_travel_turned_on_by_the_viewer_stays_free_when_it_can_happen() -> None:
    """Several earlier countries and a recent previous login: travel to a known country is possible."""
    login = _login(user_prior_countries=2)
    assert "country_change_1h" not in required_signals(
        login, login, {"country_change_1h": True}, TRAVEL_GAP
    )


def test_attack_ip_must_agree_with_its_network() -> None:
    clean = _login(asn_prior_attack_share=0.0)
    assert required_signals(clean, clean, {}, None)["attack_ip"][0] is False
    attack_only = _login(asn_prior_attack_share=1.0, prior_successes=0)
    assert required_signals(attack_only, attack_only, {}, None)["attack_ip"][0] is True
    unseen_ip = _login(asn_prior_attack_share=0.0, ip_prior_attack_share=None)
    assert "attack_ip" not in required_signals(unseen_ip, unseen_ip, {}, None)


@pytest.mark.parametrize(
    ("group", "signals"),
    [("network", {"new_asn", "new_ip"}), ("browser", {"new_browser", "new_os", "new_user_agent"}),
     ("device", {"new_device_type", "new_os", "new_browser", "new_user_agent"})],
)  # fmt: skip
def test_with_one_earlier_login_a_replaced_value_is_new(group: str, signals: set[str]) -> None:
    login = _login(prior_successes=1)
    required = required_signals(login, login, {}, None, {group})
    assert {s for s, (value, _) in required.items() if value} == signals
    assert not {
        s for s, (v, _) in required_signals(_login(), _login(), {}, None, {group}).items() if v
    }


def test_a_replaced_value_is_not_forced_new_when_the_real_one_was_new() -> None:
    """The real value was already new, so the replacement could be the earlier one."""
    login = _login(prior_successes=1, new_device_type=True)
    required = required_signals(login, login, {"new_device_type": True}, None, {"device"})
    assert "new_device_type" not in required


def test_with_one_earlier_country_a_replaced_country_is_new() -> None:
    login = _login(user_prior_countries=1)
    assert required_signals(login, login, {}, None, {"country"})["new_country"][0] is True


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


def test_typical_row_is_one_real_login() -> None:
    pool = pl.DataFrame({"event_id": [1, 2, 3], "share": [0.3, 0.1, 0.2], "users": [30, 10, 20]})
    assert typical_row(pool, "share", ["share", "users"]) == {"share": 0.2, "users": 20}
    assert typical_row(pool.clear(), "share", ["share"]) == {}


def test_rules_never_contradict_logins_built_by_the_feature_pipeline() -> None:
    """Every returning login the pipeline builds satisfies the rules that follow from the feature definitions.

    The "always together" and attack-IP rules come from the data (an IP belongs to one network, a user agent string
    encodes its device, OS and browser, a region belongs to one country, the attack list flags every attempt of an
    IP); random toy logins need not follow them, so they are checked on the real data below.
    """
    frame = build_features(_events(_random_events(3_000, seed=11)), _config()).collect()
    returning = frame.filter(pl.col("prior_successes") > 0)
    assert returning.height > 100  # noqa: PLR2004
    for row in range(returning.height):
        login = returning[row]
        switches = {s: bool(login[s][0]) for s in SIGNALS if s in login.columns}
        gap = gap_of(login["hours_since_prev_success"][0])
        for signal, (value, reason) in required_signals(login, login, switches, gap).items():
            if signal in IMPLIED_BY or signal == "attack_ip":
                continue
            assert switches[signal] == value, (signal, reason, login.row(0, named=True))


DATA_DIR = Path(os.environ.get("ATO_DATA_DIR", Path(__file__).parents[2] / "data"))
FEATURES = DATA_DIR / "04_feature" / "rba_features.parquet"


@pytest.mark.skipif(not FEATURES.is_file(), reason="needs the RBA feature table (make features)")
def test_every_rule_holds_on_the_rba_data() -> None:
    """No returning RBA login breaks a rule, so a what-if login that follows them stays realistic."""
    columns = {c for cs in NEW_IF_NEVER_SEEN.values() for c in cs} | set(UNKNOWN_IF.values())
    columns |= {*SIGNALS, *NEW_IF_NEVER_SEEN, "prior_successes", "hours_since_prev_success"}
    columns |= {"ip_prior_attack_share", "asn_prior_attack_share", "user_prior_countries"}
    columns |= {"global_share_asn", "asn_prior_users"}
    data = pl.scan_parquet(FEATURES).select(sorted(columns)).filter(pl.col("prior_successes") > 0)

    def known(flag: str) -> pl.Expr:
        column = UNKNOWN_IF.get(flag)
        return ~pl.col(column) if column else pl.lit(True)

    recent = pl.col("hours_since_prev_success") < GAPS[TRAVEL_GAP]
    attack_share, network_share = pl.col("ip_prior_attack_share"), pl.col("asn_prior_attack_share")
    broken = {
        **{
            f"{flag} never seen": ~pl.col(flag) & pl.any_horizontal(pl.col(c) == 0 for c in cols)
            for flag, cols in NEW_IF_NEVER_SEEN.items()
        },
        **{
            f"{flag} implied": pl.any_horizontal(pl.col(c) for c in causes)
            & ~pl.col(flag)
            & known(flag)
            for flag, causes in IMPLIED_BY.items()
        },
        "travel needs a recent login": pl.col("country_change_1h") & ~recent,
        "new country soon is travel": pl.col("new_country") & recent & ~pl.col("country_change_1h"),
        "one country: travel is new": pl.col("country_change_1h")
        & ~pl.col("new_country")
        & (pl.col("user_prior_countries") == 1),
        "attack IP share 1": pl.col("attack_ip") & (attack_share != 1),
        "other IP share 0": ~pl.col("attack_ip") & (attack_share != 0),
        "attack IP, clean network": (attack_share > 0) & (network_share == 0),
        "clean IP, attack-only network": (attack_share == 0) & (network_share == 1),
        "never-seen network with accounts": (pl.col("global_share_asn") == 0)
        & (pl.col("asn_prior_users") > 0),
        "network seen before without accounts": (pl.col("global_share_asn") > 0)
        & (pl.col("asn_prior_users") == 0),
        "network seen before, attack history unknown": (pl.col("global_share_asn") > 0)
        & network_share.is_null(),
    }
    counts = data.select(rule.fill_null(False).sum().alias(name) for name, rule in broken.items())
    assert counts.collect().row(0, named=True) == dict.fromkeys(broken, 0)
