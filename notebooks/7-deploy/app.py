"""Account-takeover risk demo (issue #8): pick a scenario, play with its signals, see the response.

Run from the repository root (after ``make data intermediate features split`` and notebooks 06 and 07):

    make demo

Scenarios are real logins from the **test side** of the split (users the model never saw): for each kind
of outcome (takeover sent to review, takeover sent to step-up, false alarm, ordinary login) the demo
offers a returning account and an account's first login, each the median-score login of its group.

Every signal is a switch set to the scenario's real value. *Where the login comes from* switches move a
group of inputs at once (for example, "rare network" moves the network's share of all logins and the
number of accounts seen on it): switching one replaces that group with the typical (median) values of
test logins on the other side of the threshold. *Compared with the account's earlier logins* switches
are the account-novelty flags themselves.
"""

import copy
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import streamlit as st

REPO_ROOT = next(
    p for p in (Path.cwd(), *Path(__file__).resolve().parents) if (p / "pyproject.toml").is_file()
)
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from data.datasets import load_dataset_config  # noqa: E402
from data.split import load_side  # noqa: E402
from inference.predict import ALLOW, NEVER_SEEN_PREFIX, Detector, format_value  # noqa: E402

DATA_DIR = Path(os.environ.get("ATO_DATA_DIR", REPO_ROOT / "data"))
DATASET = os.environ.get("ATO_DATASET", "rba")
TIER_STYLE = {"review / block": "error", "step-up": "warning", ALLOW: "success"}


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
        ("global_share_country",),
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
NETWORK = (
    "rare_network",
    "quiet_network",
    "attack_network",
)  # describe one network: moved together
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
IMPLIED_BY = {  # a signal that is always present when another one is (no exception in the data)
    "new_ip": ("new_asn",),
    "new_user_agent": ("new_device_type", "new_os", "new_browser"),
}
KEY_NOVELTY = ("new_country", "new_asn", "new_device_type", "new_os", "new_browser")
MIN_SCORE_CHANGE = 5e-5  # below this the 4-decimal score does not move

KEY_COLUMNS = [
    "event_id", "timestamp", "country", "device_type", "first_success", "prior_successes",
    "hours_since_prev_success", "global_share_country", "global_share_asn", "asn_prior_attack_share",
]  # fmt: skip
LABELS = {  # plain-language names for the inputs shown on screen
    "event_id": "Login id (reference only, not a model input)",
    "timestamp": "Time",
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
    "review": "Takeover → review / block",
    "step-up": "Takeover → step-up only",
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


Values = dict[str, float | None]


@st.cache_data
def context_values() -> tuple[dict[tuple[bool, ...], Values], dict[str, dict[bool, Values]]]:
    """Median inputs of test logins in each switch position.

    The network switches describe one network, so their inputs come from logins matching all three at once
    (every combination occurs in the test data); country and browser are independent of the network.
    """
    logins = test_logins()
    on = {name: context.condition.fill_null(False) for name, context in CONTEXT.items()}

    def medians(names: tuple[str, ...], where: pl.Expr) -> Values:
        columns = [c for n in names for c in CONTEXT[n].columns if c in logins.columns]
        return logins.filter(where).select(pl.col(columns).median()).row(0, named=True)

    network = {
        pattern: medians(
            NETWORK,
            pl.all_horizontal(
                on[n] if state else ~on[n] for n, state in zip(NETWORK, pattern, strict=True)
            ),
        )
        for pattern in itertools.product((True, False), repeat=len(NETWORK))
    }
    others = {
        name: {state: medians((name,), on[name] if state else ~on[name]) for state in (True, False)}
        for name in CONTEXT
        if name not in NETWORK
    }
    return network, others


def kinds(logins: pl.DataFrame, det: Detector) -> dict[str, pl.DataFrame]:
    """Test-side logins grouped by ground truth and response tier, keyed by slug (empty groups left out)."""
    score, takeover = pl.col("score"), pl.col("label_ato")
    block, step_up = det.thresholds["review / block"], det.thresholds["step-up"]
    groups = {
        "review": logins.filter(takeover & (score >= block)),
        "step-up": logins.filter(takeover & (score >= step_up) & (score < block)),
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


def load_signals(login: pl.DataFrame) -> None:
    """Set every switch to the login's real values."""
    for name in CONTEXT:
        st.session_state[_key(name)] = context_state(login, name)
    for column in SIGNALS:
        if column in login.columns:
            value = login[column][0]
            st.session_state[_key(column)] = bool(value) if value is not None else False
    st.session_state["forced"] = set()  # signals switched on only because a signal they follow was
    st.session_state["loaded"] = int(login["event_id"][0])


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


def follow_causes(login: pl.DataFrame) -> None:
    """Switch on a signal whose cause is on; put it back to its real value when its cause goes off again."""
    state = st.session_state
    forced = state["forced"]
    for child, parents in IMPLIED_BY.items():  # e.g. a new network always means a new IP address
        if any(state.get(_key(p)) for p in parents):
            if not state.get(_key(child)):
                state[_key(child)] = True
                forced.add(child)
        elif child in forced:  # its cause was switched off again: back to the real value
            real = login[child][0]
            state[_key(child)] = bool(real) if real is not None else False
            forced.discard(child)


def signal_panel(login: pl.DataFrame) -> pl.DataFrame:
    """Left column: every signal as a switch; returns the login with the switches applied."""
    if st.session_state.get("loaded") != int(login["event_id"][0]):
        load_signals(login)
    first_login = bool(login["prior_successes"][0] == 0)
    state = st.session_state
    follow_causes(login)

    st.subheader("Signals")
    st.caption(
        "On = the signal is present in this login. Change any of them to see the score move."
    )
    st.markdown("**Where the login comes from**")
    for name, context in CONTEXT.items():
        st.toggle(context.text, key=_key(name))
    for column, text in NETWORK_SIGNALS.items():
        if column in login.columns:
            st.toggle(text, key=_key(column))
    st.markdown("**Compared with the account's earlier logins**")
    if first_login:
        st.caption(
            "This is the account's first successful login: there is no earlier login to compare with, "
            "so these signals do not exist yet (greyed out)."
        )
    for column, text in HISTORY_SIGNALS.items():
        if column not in login.columns:
            continue
        forced_by = [p for p in IMPLIED_BY.get(column, ()) if state.get(_key(p))]
        help_text = f"Always on when '{SIGNALS[forced_by[0]]}' is on." if forced_by else None
        st.toggle(text, key=_key(column), disabled=first_login or bool(forced_by), help=help_text)
    st.button("Back to the real login", on_click=load_signals, args=(login,))
    return apply_signals(login, first_login)


def apply_signals(login: pl.DataFrame, first_login: bool) -> pl.DataFrame:
    """The login with the switches' values, keeping the inputs that depend on them consistent.

    Only switches moved away from the real value change the login (a missing value stays missing).
    """
    state = st.session_state
    edited = login.clone()
    network, others = context_values()
    wanted = tuple(bool(state[_key(n)]) for n in NETWORK)
    if wanted != tuple(context_state(login, n) for n in NETWORK):
        edited = _with_values(edited, network[wanted])
    for name, values in others.items():
        if bool(state[_key(name)]) != context_state(login, name):
            edited = _with_values(edited, values[bool(state[_key(name)])])
    moved = False
    for column in SIGNALS:
        if column not in login.columns or (first_login and column in HISTORY_SIGNALS):
            continue
        real = login[column][0]
        if state[_key(column)] != (bool(real) if real is not None else False):
            edited = edited.with_columns(pl.lit(state[_key(column)]).alias(column))
            moved = True
    if moved and not first_login:
        key = [c for c in KEY_NOVELTY if c in edited.columns]
        edited = edited.with_columns(
            pl.sum_horizontal(key).cast(pl.UInt8).alias("n_new_key_signals")
        )
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
        delta=f"{score - original:+.4f} vs the real login" if changed and moved else None,
        delta_color="inverse",
    )
    if changed and not moved:
        st.caption(
            "These changes alone do not move the score of this login; try combining more signals."
        )
    rank = det.riskier_than(score)
    rank_text = "more than 99.99%" if rank > 0.9999 else f"{rank:.2%}"  # noqa: PLR2004
    st.caption(f"Riskier than **{rank_text}** of development logins.")
    getattr(st, TIER_STYLE[tier])(f"Response: **{tier}**", icon=None)
    st.caption(
        f"Step-up (second factor) from {det.thresholds['step-up']:.4f}, the top 1% of development logins; "
        f"review / block from {det.thresholds['review / block']:.4f}, the top 0.1%."
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


def details_panel(login: pl.DataFrame) -> None:
    with st.expander("Login details (the real login)"):
        shown = [c for c in KEY_COLUMNS if c in login.columns]
        st.table(
            pl.DataFrame(
                {
                    "field": [label(c) for c in shown],
                    "value": [format_value(c, login[c][0]) for c in shown],
                }
            ),
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
        details_panel(login)
    with right:
        risk_panel(det, login, edited)


main()
