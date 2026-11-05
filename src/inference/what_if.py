"""Rules that keep a *what if* login consistent when its signals are changed (issue #8 demo).

Each rule holds without exception in the RBA feature table (9.8 M logins; checked on returning-account
logins, where the account-history signals exist):

* **Never seen before, so new for the account too.** If nobody had used a country, network, IP, device type,
  operating system, browser or user agent before (its share of earlier logins, or the accounts seen on it,
  is zero), it is new for this account as well.
* **Always together.** A new network means a new IP address; a new device type, operating system or browser
  means a new user agent string. A new country means a new region when the region is known: this one comes
  from the data's geography (a region belongs to one country), not from how the features are computed.
* **Travel.** A country change within an hour needs the previous login less than an hour earlier; a new
  country less than an hour after the previous login is a country change; if the account had used only one
  country, a country change is a new country.
* **Attack-listed IP.** Every earlier attempt from an attack-listed IP is an attack attempt (share 1), and
  none from another IP (share 0); an IP never seen before has no share. An IP's earlier attempts are part of
  its network's, so an attack-listed IP needs a network with some attack history, and a network whose earlier
  attempts all came from attack IPs has only attack-listed IPs.

Two more follow from the definitions when a *where from* control replaces a value: if the account had only one
earlier login, a different network, browser or device type is new for it; if it had used only one country, a
different country is new.

On a first login the account-history signals do not exist, so no rule applies to them.
"""

from collections.abc import Collection, Mapping

import polars as pl

NETWORK_SIGNALS = {"attack_ip": "IP address on an attack list"}
HISTORY_SIGNALS = {  # compared with the account's earlier logins (on = the signal is present)
    "new_country": "New country",
    "new_region": "New region",
    "new_city": "New city",
    "new_asn": "New network (ASN)",
    "new_ip": "New IP address",
    "new_device_type": "New device type",
    "new_os": "New operating system",
    "new_browser": "New browser",
    "new_user_agent": "New exact user agent",
    "country_change_1h": "Country changed within an hour of the previous login",
}
SIGNALS = {**NETWORK_SIGNALS, **HISTORY_SIGNALS}
NEW_IF_NEVER_SEEN = {  # nobody had used it before (a zero here), so it is new for this account too
    "new_country": ("global_share_country",),
    "new_asn": ("global_share_asn", "asn_prior_users"),
    "new_ip": ("global_share_ip", "ip_prior_users"),
    "new_device_type": ("global_share_device_type",),
    "new_os": ("global_share_os",),
    "new_browser": ("global_share_browser",),
    "new_user_agent": ("global_share_user_agent", "user_agent_prior_users"),
}
IMPLIED_BY = {  # always present when one of its causes is
    "new_region": ("new_country",),
    "new_ip": ("new_asn",),
    "new_user_agent": ("new_device_type", "new_os", "new_browser"),
}
UNKNOWN_IF = {"new_region": "region_missing", "new_city": "city_missing"}  # no rule when unknown
GAPS = {  # time since the account's previous login: label -> upper bound in hours
    "under an hour": 1.0,
    "1 hour to 1 day": 24.0,
    "1 day to 1 week": 24.0 * 7,
    "1 week to 1 month": 24.0 * 30,
    "over a month": float("inf"),
}
TRAVEL = "country_change_1h"
TRAVEL_GAP = "under an hour"
ATTACK_SHARE = "ip_prior_attack_share"
NEVER_SEEN_REASON = "Nobody had used it before, so it is new for this account too."
TRAVEL_REASON = "Needs the previous login less than an hour earlier."
NEW_COUNTRY_TRAVEL_REASON = (
    "A new country less than an hour after the previous login is a country change."
)
ONE_COUNTRY_REASON = "The account had used only one country, so another country is a new one."
SINGLE_LOGIN_REASON = "The account had one earlier login, so a different value is a new one."
NEW_IF_MOVED = {  # where-from groups and the account-novelty signal of the value they replace
    "network": ("new_asn", "new_ip"),
    "browser": ("new_browser",),
    "device": ("new_device_type",),
}

Required = dict[str, tuple[bool, str]]


def gap_of(hours: float | None) -> str | None:
    """The time-since-previous-login range of a login (None on a first login)."""
    if hours is None:
        return None
    return next(name for name, upper in GAPS.items() if hours < upper)


def unknown(login: pl.DataFrame, signal: str) -> bool:
    """Whether the signal cannot be judged for this login (e.g. the region is unknown)."""
    column = UNKNOWN_IF.get(signal)
    return column is not None and column in login.columns and bool(login[column][0])


def required_signals(  # noqa: C901, PLR0912 - one place for every rule, in the order they apply
    login: pl.DataFrame,
    located: pl.DataFrame,
    switches: Mapping[str, bool],
    gap: str | None,
    moved: Collection[str] = (),
) -> Required:
    """Signals a rule fixes, with the value and the reason.

    ``login`` is the real login; ``located`` the login after the *where from* controls; ``switches`` the
    viewer's own switch values (a value a rule forced must not be passed here, so a rule never feeds itself);
    ``gap`` the chosen time since the previous login; ``moved`` the *where from* groups whose value the viewer
    changed (``network``, ``country``, ``browser``, ``device``). Rules are applied until none adds anything.
    """
    required: Required = {}

    def is_on(signal: str) -> bool:
        return required[signal][0] if signal in required else bool(switches.get(signal))

    def fix(signal: str, value: bool, reason: str) -> bool:
        if signal not in login.columns or signal in required:
            return False
        required[signal] = (value, reason)
        return True

    # An IP's earlier attempts are part of its network's: an attack-listed IP needs a network with some attack
    # history, and a network with only attack attempts has only attack-listed IPs (for an IP seen before).
    if ATTACK_SHARE in located.columns and located[ATTACK_SHARE][0] is not None:
        network_share = located["asn_prior_attack_share"][0]
        if network_share == 0:
            fix(
                "attack_ip",
                False,
                "This network had no attack attempts before, so its IPs are not on the list.",
            )
        elif network_share == 1:
            fix(
                "attack_ip",
                True,
                "Every earlier attempt from this network came from attack-listed IPs.",
            )
    if login["prior_successes"][0] == 0:
        return required  # no account history: the history signals do not exist
    for signal, columns in NEW_IF_NEVER_SEEN.items():
        if any(located[c][0] == 0 for c in columns if c in located.columns):
            fix(signal, True, NEVER_SEEN_REASON)
    if login["prior_successes"][0] == 1:  # one earlier login: a different value is a new one
        for group, signals in NEW_IF_MOVED.items():
            if group in moved:
                for signal in signals:
                    fix(signal, True, SINGLE_LOGIN_REASON)
    one_country = "user_prior_countries" in login.columns and login["user_prior_countries"][0] == 1
    if one_country and "country" in moved:
        fix("new_country", True, ONE_COUNTRY_REASON)
    if gap != TRAVEL_GAP:
        fix(TRAVEL, False, TRAVEL_REASON)
    added = True
    while added:  # rules that depend on each other, until nothing changes
        added = False
        for signal, causes in IMPLIED_BY.items():
            present = [c for c in causes if is_on(c)]
            if present and not unknown(login, signal):
                added |= fix(signal, True, f"Always on when '{SIGNALS[present[0]]}' is on.")
        if gap == TRAVEL_GAP and is_on("new_country"):
            added |= fix(TRAVEL, True, NEW_COUNTRY_TRAVEL_REASON)
        if one_country and is_on(TRAVEL):
            added |= fix("new_country", True, ONE_COUNTRY_REASON)
    return required


def with_attack_share(edited: pl.DataFrame) -> pl.DataFrame:
    """Set the IP's share of earlier attack attempts from the attack-IP flag (unchanged if never seen)."""
    if ATTACK_SHARE not in edited.columns or edited[ATTACK_SHARE][0] is None:
        return edited
    share = 1.0 if edited["attack_ip"][0] else 0.0
    return edited.with_columns(pl.lit(share).cast(edited[ATTACK_SHARE].dtype).alias(ATTACK_SHARE))
