"""Account-takeover risk demo (issue #8): pick a scenario, play with its signals, see the response.

Run from the repository root (after ``make data intermediate features split`` and notebooks 06 and 07):

    make demo

Scenarios are real logins from the **test side** of the split (users the model never saw): for each kind
of outcome (takeover blocked, takeover asked for a second factor, takeover missed when there is one, false
alarm, ordinary login) the demo offers a returning account and an account's first login, each the median-score
login of its group. The response tiers of issue #7 are shown as *block* (review / block) and *ask for 2FA*
(step-up authentication: a second factor).

Every signal is a control set to the scenario's real value. *Where the login comes from* switches move a
group of inputs at once: switching one replaces that group with the inputs of one real test login on the other
side of the threshold, the one most similar to the scenario's login on the inputs the viewer did not change
(``inference.what_if.nearest_row``), so the relations between them hold and nothing else moves more than
needed. *Compared with the account's earlier logins* controls are the time since the previous login and the
account-novelty flags. Rules between signals (``inference.what_if``) fix the signals that must follow from the
others; those are greyed out with the reason.
"""

import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pandas as pd
import polars as pl
import streamlit as st

REPO_ROOT = next(
    p for p in (Path.cwd(), *Path(__file__).resolve().parents) if (p / "pyproject.toml").is_file()
)
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from data.datasets import load_dataset_config  # noqa: E402
from data.split import load_side  # noqa: E402
from inference.predict import ALLOW, NEVER_SEEN_PREFIX, Detector  # noqa: E402
from inference.what_if import (  # noqa: E402
    GAPS,
    HISTORY_SIGNALS,
    NETWORK_SIGNALS,
    SIGNALS,
    Values,
    gap_of,
    nearest_row,
    required_signals,
    typical_row,
    unknown,
    with_attack_share,
)

DATA_DIR = Path(os.environ.get("ATO_DATA_DIR", REPO_ROOT / "data"))
DATASET = os.environ.get("ATO_DATASET", "rba")
TIER_STYLE = {"review / block": "error", "step-up": "warning", ALLOW: "success"}
RESPONSE = {"review / block": "block", "step-up": "ask for 2FA (a second factor)", ALLOW: "allow"}


@dataclass(frozen=True)
class Context:
    """A switch over a group of inputs, on when ``condition`` holds for the login."""

    text: str
    columns: tuple[str, ...]
    condition: pl.Expr


CONTEXT = {  # thresholds sit between dev takeovers and dev legitimate logins (notebook 07 / EDA)
    "rare_network": Context(
        "Rare network (ASN): under 0.1% of all earlier logins",
        ("global_share_asn", "global_share_ip", "asn_prior_users", "ip_prior_users"),
        pl.col("global_share_asn") < 0.001,  # noqa: PLR2004
    ),
    "quiet_network": Context(
        "Network barely active in the previous 24 hours (under 50 attempts)",
        ("asn_attempts_prev24h", "asn_fails_prev24h"),
        pl.col("asn_attempts_prev24h") < 50,  # noqa: PLR2004
    ),
    "attack_network": Context(
        "Most of the network's earlier attempts came from attack IPs",
        (
            "asn_prior_attack_share",
            "asn_prior_fail_share",
            "ip_prior_attack_share",
            "ip_prior_fail_share",
        ),
        pl.col("asn_prior_attack_share") >= 0.5,  # noqa: PLR2004
    ),
    "rare_country": Context(
        "Rare country: under 1% of all earlier logins",
        ("global_share_country", "country"),  # the country is shown, not a model input
        pl.col("global_share_country") < 0.01,  # noqa: PLR2004
    ),
    "rare_browser": Context(
        "Rare browser: under 0.1% of all earlier logins",
        (
            "global_share_browser",
            "global_share_os",
            "global_share_user_agent",
            "user_agent_prior_users",
        ),
        pl.col("global_share_browser") < 0.001,  # noqa: PLR2004
    ),
}
NETWORK = ("rare_network", "quiet_network", "attack_network")  # one network: moved together
DEVICE = "device_type"
DEVICES = ("desktop", "mobile", "tablet")
GAP = "gap"
CLIENT = "rare_browser"  # moved together with the device type
ASN = (
    "ASN (autonomous system number): the number that identifies a network on the internet, such as an "
    "internet provider's, a company's or a cloud host's."
)
DEFINITIONS = {  # shown on hover, only where the switch's name uses a technical term
    "rare_network": f"Less than 0.1% of all earlier logins came from this network. {ASN}",
    "attack_network": "At least half of the earlier attempts from this network came from IP addresses on "
    "the dataset's list of addresses used in credential attacks (such as password guessing).",
    "rare_country": "Less than 1% of all earlier logins came from this country.",
    "rare_browser": "Less than 0.1% of all earlier logins used this browser version.",
    DEVICE: "In this data, desktop logins were about 10 times more often takeovers than mobile ones.",
    "attack_ip": "The dataset's list of IP addresses used in credential attacks (such as password guessing).",
    "new_asn": f"The account had never logged in from this network. {ASN}",
    "new_user_agent": "User agent: the text a browser sends to identify itself (browser and version, "
    "operating system, device). The account had never sent this exact one.",
    "country_change_1h": "Also called impossible travel: a jump between countries a person can hardly make.",
}
KEY_NOVELTY = ("new_country", "new_asn", "new_device_type", "new_os", "new_browser")
MIN_SCORE_CHANGE = 5e-5  # below this the 4-decimal score does not move

LABELS = {  # plain-language names for the inputs shown on screen
    "country": "Country",
    "device_type": "Device type",
    "first_success": "First successful login of this account",
    "prior_successes": "Earlier successful logins of this account",
    "hours_since_prev_success": "Hours since the account's previous login",
    "global_share_country": "Share of all earlier logins from this country",
    "global_share_asn": "Share of all earlier logins from this network (ASN)",
    "global_share_ip": "Share of all earlier logins from this IP",
    "global_share_browser": "Share of all earlier logins with this browser",
    "global_share_os": "Share of all earlier logins with this OS",
    "global_share_device_type": "Share of all earlier logins with this device type",
    "global_share_user_agent": "Share of all earlier logins with this exact user agent",
    "asn_prior_attack_share": "Share of this network's earlier attempts from attack IPs",
    "asn_prior_fail_share": "Share of this network's earlier attempts that failed",
    "ip_prior_attack_share": "Share of this IP's earlier attempts from attack IPs",
    "ip_prior_fail_share": "Share of this IP's earlier attempts that failed",
    "asn_attempts_prev24h": "Attempts from this network in the previous 24 hours",
    "asn_fails_prev24h": "Failed attempts from this network in the previous 24 hours",
    "asn_prior_users": "Accounts seen on this network before",
    "ip_prior_users": "Accounts seen on this IP before",
    "user_agent_prior_users": "Accounts seen with this exact user agent before",
    "user_prior_countries": "Countries this account used before",
    "n_new_key_signals": "Number of new key signals at once",
    "fails_1h": "Failed attempts on this account in the previous hour",
    "fails_24h": "Failed attempts on this account in the previous 24 hours",
    "region_missing": "Region unknown",
    "city_missing": "City unknown",
    **{
        c: f"{text} for this account" for c, text in HISTORY_SIGNALS.items() if c.startswith("new_")
    },
    "country_change_1h": HISTORY_SIGNALS["country_change_1h"],
    **NETWORK_SIGNALS,
}
ENCODED = {"device_type_": "Device type is ", "hour_": "Hour of day", "weekday_": "Day of week"}

KINDS = {  # stable URL slug -> button label
    "block": "Takeover → block",
    "2fa": "Takeover → ask for 2FA",
    "missed": "Takeover the model misses",
    "false-alarm": "Legitimate login flagged (false alarm)",
    "ordinary": "Ordinary legitimate login",
}
SCENARIOS = {"returning": "Returning account", "first": "First login of a new account"}

REFERENCE_SIZE = 300  # ordinary logins that define "typical" for the explanations
MIN_EFFECT = 1e-4  # contributions smaller than this are not shown


@st.cache_resource
def detector() -> Detector:
    return Detector.load(DATA_DIR, DATASET)


@st.cache_data
def test_logins() -> pl.DataFrame:
    """Test-side logins with their saved scores (issue #7)."""
    config = load_dataset_config(DATASET)
    scores = pl.read_parquet(DATA_DIR / "07_model_output" / f"{DATASET}_test_scores.parquet")
    return load_side(config, DATA_DIR, "test").collect().join(scores, on="event_id", how="inner")


@st.cache_resource
def explained_detector() -> Detector:
    """The detector, explaining relative to a fixed sample of ordinary (allowed, legitimate) test logins.

    The sample comes from the test side, which issue #7 already evaluated; it only sets what "typical" means.
    """
    det = copy.copy(detector())  # keep the shared cached detector unchanged
    logins = test_logins()
    ordinary = logins.filter(~pl.col("label_ato") & (pl.col("score") < det.thresholds["step-up"]))
    det.set_reference(ordinary.sample(n=min(REFERENCE_SIZE, ordinary.height), seed=0))
    return det


def context_state(login: pl.DataFrame, name: str) -> bool:
    """Whether a *where the login comes from* switch is on for this login (a missing value: off)."""
    return bool(login.select(CONTEXT[name].condition.fill_null(False)).item())


def _on(name: str, state: bool) -> pl.Expr:
    condition = CONTEXT[name].condition.fill_null(False)
    return condition if state else ~condition


def _columns(logins: pl.DataFrame, *names: str) -> list[str]:
    return [c for n in names for c in CONTEXT[n].columns if c in logins.columns]


class Typical(TypedDict):
    """Inputs of typical test logins for each position of the switches (a plain dict, so it can be cached)."""

    others: dict[str, dict[bool, Values]]
    gap: dict[str, float | None]


@st.cache_data
def typical_values() -> Typical:
    """The country inputs of a typical real test login on each side of its threshold, and a typical time gap.

    The country switch is the only input of its group, so there is nothing to keep similar.
    """
    logins = test_logins()
    others = {
        name: {
            state: typical_row(
                logins.filter(_on(name, state)), CONTEXT[name].columns[0], _columns(logins, name)
            )
            for state in (True, False)
        }
        for name in CONTEXT
        if name not in (*NETWORK, CLIENT)
    }
    hours, lower, gap = pl.col("hours_since_prev_success"), 0.0, {}
    for name, upper in GAPS.items():
        within = logins.filter((hours >= lower) & (hours < upper))
        gap[name] = typical_row(
            within, "hours_since_prev_success", ["hours_since_prev_success"]
        ).get("hours_since_prev_success")
        lower = upper
    return Typical(others=others, gap=gap)


def _scenario_login(logins: pl.DataFrame, event_id: int) -> pl.DataFrame:
    return logins.filter(pl.col("event_id") == event_id)


@st.cache_data
def network_donor(event_id: int, wanted: tuple[bool, ...]) -> Values:
    """Network inputs for the network switches in position ``wanted``.

    The three switches describe one network, so the inputs come from one real test login matching all three
    (every combination occurs in the test data): the one most similar to the scenario's login on the inputs of
    the switches the viewer left as they were.
    """
    logins = test_logins()
    login = _scenario_login(logins, event_id)
    real = tuple(context_state(login, n) for n in NETWORK)
    kept = [n for n, w, r in zip(NETWORK, wanted, real, strict=True) if w == r]
    pool = logins.filter(*(_on(n, w) for n, w in zip(NETWORK, wanted, strict=True)))
    return nearest_row(
        pool, login, _columns(logins, *kept), _columns(logins, *NETWORK), by="global_share_asn"
    )


@st.cache_data
def client_donor(event_id: int, device: str, rare: bool) -> Values:
    """Device and browser inputs from one real test login with this device type and browser rarity.

    Device type and browser describe one client. When only the device type changed, the login is the one
    whose browser is most like the scenario's.
    """
    logins = test_logins()
    login = _scenario_login(logins, event_id)
    kept = [CLIENT] if rare == context_state(login, CLIENT) else []
    pool = logins.filter((pl.col(DEVICE) == device) & _on(CLIENT, rare))
    return nearest_row(
        pool,
        login,
        _columns(logins, *kept),
        ["global_share_device_type", *_columns(logins, CLIENT)],
        by="global_share_browser",
    )


def kinds(logins: pl.DataFrame, det: Detector) -> dict[str, pl.DataFrame]:
    """Test-side logins grouped by ground truth and response tier, keyed by slug (empty groups left out)."""
    score, takeover = pl.col("score"), pl.col("label_ato")
    block, step_up = det.thresholds["review / block"], det.thresholds["step-up"]
    groups = {
        "block": logins.filter(takeover & (score >= block)),
        "2fa": logins.filter(takeover & (score >= step_up) & (score < block)),
        "missed": logins.filter(takeover & (score < step_up)),
        "false-alarm": logins.filter(~takeover & (score >= step_up)),
        "ordinary": logins.filter(~takeover & (score < step_up)),
    }
    return {slug: frame for slug, frame in groups.items() if not frame.is_empty()}


def representative(pool: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """The median-score returning-account login and first login of a group (ties: fixed pseudo-random order)."""
    first = pl.col("prior_successes") == 0
    picks = {}
    for name, frame in (("returning", pool.filter(~first)), ("first", pool.filter(first))):
        if frame.is_empty():
            continue
        ordered = frame.sort(pl.col("score"), pl.col("event_id").hash(seed=0))
        picks[name] = ordered[(ordered.height - 1) // 2]
    return picks


@st.cache_data
def scenarios() -> tuple[dict[str, int], dict[str, dict[str, pl.DataFrame]]]:
    """Group sizes and the scenario logins of each kind."""
    groups = kinds(test_logins(), explained_detector())
    return (
        {slug: frame.height for slug, frame in groups.items()},
        {slug: representative(frame) for slug, frame in groups.items()},
    )


def _key(name: str) -> str:
    return f"signal-{name}"


def _real(login: pl.DataFrame, column: str) -> bool:
    value = login[column][0]
    return bool(value) if value is not None else False


def load_signals(login: pl.DataFrame) -> None:
    """Set every control to the login's real values."""
    state = st.session_state
    for name in CONTEXT:
        state[_key(name)] = context_state(login, name)
    for column in SIGNALS:
        if column in login.columns:
            state[_key(column)] = _real(login, column)
    state[_key(DEVICE)] = login[DEVICE][0]
    state[_key(GAP)] = gap_of(login["hours_since_prev_success"][0])
    state["forced"] = {}  # signal -> the value the viewer had given it before a rule fixed it
    state["loaded"] = int(login["event_id"][0])


def pick_scenario(
    counts: dict[str, int], picks: dict[str, dict[str, pl.DataFrame]]
) -> pl.DataFrame:
    """Top of the page: one button per kind of outcome, then its scenarios; returns the chosen login."""
    slugs = list(picks)
    if (
        st.session_state.get("kind") not in slugs
    ):  # ?example=<slug>&scenario=<name> opens one directly
        wanted = st.query_params.get("example", slugs[0])
        st.session_state["kind"] = wanted if wanted in slugs else slugs[0]
    for column, slug in zip(st.columns(len(slugs)), slugs, strict=True):
        chosen = slug == st.session_state["kind"]
        if column.button(
            KINDS[slug],
            key=f"kind-{slug}",
            type="primary" if chosen else "secondary",
            width="stretch",
        ):
            st.session_state["kind"] = slug
            st.rerun()
    kind = st.session_state["kind"]
    options = list(picks[kind])
    requested = st.query_params.get("scenario")
    scenario = st.radio(
        "Scenario",
        options,
        index=options.index(requested) if requested in options else 0,
        format_func=SCENARIOS.__getitem__,
        horizontal=True,
        key=f"scenario-{kind}",
    )
    st.query_params.update({"example": kind, "scenario": scenario})  # the URL reopens this scenario
    login = picks[kind][scenario]
    st.caption(
        f"A real test login, the typical (median-score) one of its scenario; {counts[kind]:,} test logins "
        f"of this kind in all. {describe(login)}"
    )
    return login


def describe(login: pl.DataFrame) -> str:
    """One line about the scenario's real login."""
    prior = int(login["prior_successes"][0])
    if prior == 0:
        account = "the account's first successful login"
    else:
        account = f"{prior:,} earlier login{'s' if prior > 1 else ''}"
    truth = "**takeover**" if login["label_ato"][0] else "**legitimate**"
    return (
        f"{account[0].upper()}{account[1:]}, from {login['country'][0]} on {login['device_type'][0]}, "
        f"{login['timestamp'][0]:%Y-%m-%d %H:%M}. Ground truth: {truth}."
    )


def label(column: str) -> str:
    """Plain-language name of an input; never-seen indicators and encodings read from their source."""
    if column.startswith(NEVER_SEEN_PREFIX):
        source = column.removeprefix(NEVER_SEEN_PREFIX)
        return f"{source.split('_', 1)[0].upper()} never seen before (for: {LABELS.get(source, source)})"
    for prefix, text in ENCODED.items():
        if column.startswith(prefix) and column not in LABELS:
            return f"{text}{column.removeprefix(prefix)}" if text.endswith(" ") else text
    return LABELS.get(column, column)


def locate(login: pl.DataFrame) -> pl.DataFrame:
    """The login with the *where the login comes from* controls applied (network, country, browser, device)."""
    state = st.session_state
    event_id = int(login["event_id"][0])
    located = login.clone()
    wanted = tuple(bool(state[_key(n)]) for n in NETWORK)
    if wanted != tuple(context_state(login, n) for n in NETWORK):
        located = _with_values(located, network_donor(event_id, wanted))
    for name, values in typical_values()["others"].items():
        if bool(state[_key(name)]) != context_state(login, name):
            located = _with_values(located, values[bool(state[_key(name)])])
    device, rare = state[_key(DEVICE)], bool(state[_key(CLIENT)])
    if device != login[DEVICE][0] or rare != context_state(login, CLIENT):
        located = _with_values(located, {DEVICE: device, **client_donor(event_id, device, rare)})
    return located


def current_required(login: pl.DataFrame, located: pl.DataFrame) -> dict[str, tuple[bool, str]]:
    """The signals the rules fix for the current controls (see ``inference.what_if``).

    A signal a rule forced counts with the value the viewer had given it, so a rule never feeds itself.
    """
    state = st.session_state
    forced = state["forced"]
    switches = {
        signal: forced[signal] if signal in forced else bool(state.get(_key(signal)))
        for signal in SIGNALS
    }
    moved = {
        "network": tuple(bool(state[_key(n)]) for n in NETWORK)
        != tuple(context_state(login, n) for n in NETWORK),
        "country": bool(state[_key("rare_country")]) != context_state(login, "rare_country"),
        "browser": bool(state[_key("rare_browser")]) != context_state(login, "rare_browser"),
        "device": state[_key(DEVICE)] != login[DEVICE][0],
    }
    return required_signals(
        login, located, switches, state.get(_key(GAP)), {g for g, m in moved.items() if m}
    )


def follow_rules(required: dict[str, tuple[bool, str]]) -> None:
    """Set the signals a rule fixes; give a signal back the viewer's value once no rule fixes it any more."""
    state = st.session_state
    forced = state["forced"]  # signal -> the value the viewer had given it
    for signal, (value, _reason) in required.items():
        current = bool(state.get(_key(signal)))
        if signal not in forced and current != value:
            forced[signal] = current
        state[_key(signal)] = value
    for signal in [s for s in forced if s not in required]:
        state[_key(signal)] = forced.pop(signal)


def signal_panel(login: pl.DataFrame) -> pl.DataFrame:
    """Left column: every signal as a control; returns the login with the controls applied."""
    if st.session_state.get("loaded") != int(login["event_id"][0]):
        load_signals(login)
    first_login = bool(login["prior_successes"][0] == 0)
    located = locate(login)
    required = current_required(login, located)
    follow_rules(required)

    st.subheader("Signals")
    st.caption(
        "On = the signal is present in this login. Change any of them to see the score move."
    )
    st.markdown("**Where the login comes from**")
    for name, context in CONTEXT.items():
        st.toggle(context.text, key=_key(name), help=_help(name))
    devices = [*DEVICES, *([login[DEVICE][0]] if login[DEVICE][0] not in DEVICES else [])]
    st.radio("Device type", devices, horizontal=True, key=_key(DEVICE), help=_help(DEVICE))
    for column, text in NETWORK_SIGNALS.items():
        if column in login.columns:
            fixed = required[column][1] if column in required else None
            st.toggle(text, key=_key(column), disabled=fixed is not None, help=_help(column, fixed))
    st.markdown("**Compared with the account's earlier logins**")
    if first_login:
        st.caption(
            "This is the account's first successful login: there is no earlier login to compare with, "
            "so these signals do not exist yet (greyed out)."
        )
    else:
        st.select_slider("Previous login of this account", list(GAPS), key=_key(GAP))
    for column, text in HISTORY_SIGNALS.items():
        if column not in login.columns:
            continue
        if unknown(login, column):
            reason: str | None = f"{text.removeprefix('New ').capitalize()} unknown for this login."
        else:
            reason = required[column][1] if column in required else None
        st.toggle(
            text,
            key=_key(column),
            disabled=first_login or reason is not None,
            help=_help(column, reason),
        )
    st.button("Back to the real login", on_click=load_signals, args=(login,))
    return apply_signals(login, located, first_login)


def _help(signal: str, fixed: str | None = None) -> str | None:
    """The signal's definition (only for technical terms), and why it is greyed out when a rule fixes it."""
    parts = [DEFINITIONS.get(signal), f"**Greyed out:** {fixed}" if fixed else None]
    return "\n\n".join(p for p in parts if p) or None


def apply_signals(login: pl.DataFrame, located: pl.DataFrame, first_login: bool) -> pl.DataFrame:
    """The located login with the signal switches applied, keeping dependent inputs consistent.

    Only controls moved away from the real value change the login (a missing value stays missing).
    """
    state = st.session_state
    edited = located
    moved = False
    for column in SIGNALS:
        if column not in login.columns or (first_login and column in HISTORY_SIGNALS):
            continue
        if state[_key(column)] != _real(login, column):
            edited = edited.with_columns(pl.lit(state[_key(column)]).alias(column))
            moved = True
    if moved and not first_login:
        key = [c for c in KEY_NOVELTY if c in edited.columns]
        edited = edited.with_columns(
            pl.sum_horizontal(key).cast(pl.UInt8).alias("n_new_key_signals")
        )
    if not first_login and state[_key(GAP)] != gap_of(login["hours_since_prev_success"][0]):
        hours = typical_values()["gap"][state[_key(GAP)]]
        if hours is not None:
            edited = edited.with_columns(pl.lit(hours).alias("hours_since_prev_success"))
    share = "ip_prior_attack_share"
    touched = state[_key("attack_ip")] != _real(login, "attack_ip") or (
        share in login.columns and located[share][0] != login[share][0]
    )
    if touched:  # only re-derive what a control changed, so the real login is never altered
        edited = with_attack_share(edited)
    return edited


def _with_values(login: pl.DataFrame, values: Values) -> pl.DataFrame:
    return login.with_columns(
        pl.lit(value).cast(login[column].dtype).alias(column) for column, value in values.items()
    )


def changed_inputs(det: Detector, login: pl.DataFrame, edited: pl.DataFrame) -> list[str]:
    return [c for c in det.columns if c in login.columns and edited[c][0] != login[c][0]]


def risk_panel(det: Detector, login: pl.DataFrame, edited: pl.DataFrame) -> None:
    """Right column: score, rank, response tier and the inputs behind the score."""
    score, original = float(det.score(edited)[0]), float(det.score(login)[0])
    changed = changed_inputs(det, login, edited)
    moved = abs(score - original) >= MIN_SCORE_CHANGE
    tier = det.tier(score)
    st.subheader("Risk")
    st.metric(
        "Takeover score",
        f"{score:.4f}",
        delta=f"{score - original:+.4f} from the real login's {original:.4f}"
        if changed and moved
        else None,
        delta_color="inverse",
    )
    if changed and not moved:
        st.caption(
            "These changes alone do not move the score of this login; try combining more signals."
        )
    rank = det.riskier_than(score)
    rank_text = "more than 99.99%" if rank > 0.9999 else f"{rank:.2%}"  # noqa: PLR2004
    st.caption(f"Riskier than **{rank_text}** of development logins.")
    getattr(st, TIER_STYLE[tier])(f"Response: **{RESPONSE[tier]}**", icon=None)
    st.caption(
        f"Ask for 2FA from {det.thresholds['step-up']:.4f}, the top 1% of development logins; "
        f"block from {det.thresholds['review / block']:.4f}, the top 0.1%."
    )
    st.subheader("Why")
    everything = det.explain(edited, top=len(det.columns) * 3)[0]  # every input, by |contribution|
    shown_first = everything.filter(pl.col("input").is_in(changed))  # always show what was changed
    rest = everything.filter(~pl.col("input").is_in(changed))
    raises = rest.filter(pl.col("contribution") > MIN_EFFECT).head(4)  # riskier first
    lowers = rest.filter(pl.col("contribution") < -MIN_EFFECT).head(3)
    effect = (
        pl.when(pl.col("contribution") > 0).then(pl.lit("↑ raises")).otherwise(pl.lit("↓ lowers"))
    )
    st.table(
        pl.concat([shown_first, raises, lowers])
        .with_columns(pl.col("input").map_elements(label, return_dtype=pl.String))
        .with_columns(effect.alias("effect"), pl.col("contribution").round(3)),
    )
    st.caption(
        "Tree SHAP values relative to a typical login: any input you changed, then the inputs that make this "
        "login riskier than a typical one, then those that make it safer."
    )
    comparison_panel(det, login, score, original)


def _switch(value: object) -> str:
    return "on" if value else "off"


def signal_rows(login: pl.DataFrame) -> list[tuple[str, str, str]]:
    """Every control: its name, the scenario's real value and the value it has now."""
    state = st.session_state
    rows = [
        (context.text, _switch(context_state(login, name)), _switch(state[_key(name)]))
        for name, context in CONTEXT.items()
    ]
    rows.append(("Device type", str(login[DEVICE][0]), str(state[_key(DEVICE)])))
    signals = dict(NETWORK_SIGNALS)
    if login["prior_successes"][0] > 0:  # a first login has no earlier login to compare with
        gap = gap_of(login["hours_since_prev_success"][0])
        rows.append(("Previous login of this account", str(gap), str(state[_key(GAP)])))
        signals |= HISTORY_SIGNALS
    rows += [
        (text, _switch(_real(login, column)), _switch(state[_key(column)]))
        for column, text in signals.items()
        if column in login.columns
    ]
    return rows


def comparison_panel(det: Detector, login: pl.DataFrame, score: float, original: float) -> None:
    """The scenario's real signals next to the viewer's, with the score of each as the last row."""
    st.subheader("Real login and your changes")
    st.caption(
        "The signals you can change here, as they were in the scenario's real login and as they are now "
        "(changed ones in bold; a greyed-out switch may have changed with them), and the model's score for each."
    )

    def scored(value: float) -> str:
        return f"{value:.4f} ({RESPONSE[det.tier(value)]})"

    rows = [*signal_rows(login), ("Takeover score", scored(original), scored(score))]
    table = pd.DataFrame(rows, columns=["signal", "real login", "with your changes"])
    changed = table["real login"] != table["with your changes"]
    st.table(
        table.style.apply(
            lambda row: ["font-weight: bold" if changed[row.name] else ""] * len(row), axis=1
        ).hide(axis="index")
    )


def main() -> None:
    st.set_page_config(page_title="ATO risk demo", layout="wide")
    st.markdown(  # larger category buttons
        "<style>[class*='st-key-kind-'] button {min-height: 3.5rem; font-size: 1.05rem;}</style>",
        unsafe_allow_html=True,
    )
    st.title("Account-takeover risk — demo")
    st.caption(
        "Proof of concept for the thesis *Detección de Account Takeover mediante identificación automática de "
        "anomalías comportamentales en registros de autenticación*. Synthetic data (RBA dataset, Wiefling et al. 2022); "
        "logins from users the model never saw in training."
    )
    try:
        det = explained_detector()
        counts, picks = scenarios()
    except FileNotFoundError as missing:
        st.error(
            f"Model or data not found ({missing.filename}). Run the pipeline and notebooks 06 and 07 first; see README."
        )
        st.stop()
    if not picks:
        st.info("No logins to show.")
        st.stop()
    login = pick_scenario(counts, picks)
    left, right = st.columns([1, 1.2])
    with left:
        edited = signal_panel(login)
    with right:
        risk_panel(det, login, edited)


main()
