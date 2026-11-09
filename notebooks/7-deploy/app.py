"""Account-takeover risk demo (issue #8): pick a real login, change its inputs, see the model's answer.

Run from the repository root (after ``make data intermediate features split`` and notebooks 06 and 07):

    make demo

Scenarios are real logins from the **test side** of the split (users the model never saw): for each kind
of outcome (takeover blocked, takeover asked for a second factor, takeover missed when there is one, false
alarm, ordinary login) the demo offers a returning account and an account's first login, each the median-score
login of its group. The response tiers of issue #7 are shown as *block* (review / block) and *ask for 2FA*
(step-up authentication: a second factor).

Every control is one model input, set to the real login's value: yes/no inputs are switches and number inputs
are sliders over plain steps (the real value included), each with where takeovers and legitimate logins usually
are. The model re-scores exactly the inputs shown; nothing is taken from another login. Strict relations of the
data (``inference.what_if``) grey out the inputs that follow from others, with the reason. The table on the
right shows how the score is built: tree-SHAP contributions per control, in the order of the controls, adding up
from a typical ordinary login's score to this one's.
"""

import copy
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path

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
    NEVER_SEEN_NETWORK,
    SIGNALS,
    gap_of,
    required_signals,
    typical_row,
    unknown,
    with_attack_share,
)

DATA_DIR = Path(os.environ.get("ATO_DATA_DIR", REPO_ROOT / "data"))
DATASET = os.environ.get("ATO_DATASET", "rba")
THESIS_EN = (
    "Account takeover (ATO) detection through automatic identification of behavioural anomalies in "
    "authentication records"
)
REPOSITORY = "https://github.com/modsiw-mona/anomaly_detection_ato"
TIER_STYLE = {"review / block": "error", "step-up": "warning", ALLOW: "success"}
RESPONSE = {"review / block": "block", "step-up": "ask for 2FA (a second factor)", ALLOW: "allow"}
SHORT = {"review / block": "block", "step-up": "ask for 2FA", ALLOW: "allow"}  # in the table
KINDS = {  # stable URL slug -> button label, and what the kind means
    "block": ("Takeover → block", "A real takeover the model scores high enough to block."),
    "2fa": (
        "Takeover → ask for 2FA",
        "A real takeover the model catches by asking for a second factor.",
    ),
    "missed": ("Takeover the model misses", "A real takeover the model lets through."),
    "false-alarm": (
        "Legitimate login flagged (false alarm)",
        "A real user's login the model asks for 2FA by mistake.",
    ),
    "ordinary": (
        "Ordinary legitimate login",
        "A real user's login the model lets through, as it should.",
    ),
}
SCENARIOS = {"returning": "Returning account", "first": "First login of a new account"}
ASN = (
    "ASN (autonomous system number): the number that identifies a network on the internet, such as an "
    "internet provider's, a company's or a cloud host's."
)
ATTACK = (
    "Attack IPs: the dataset's list of IP addresses used in credential attacks (such as password "
    "guessing)."
)
DEVICE, DEVICES, GAP = "device_type", ("desktop", "mobile", "tablet"), "gap"


@dataclass(frozen=True)
class Slider:
    """A number input of the model, set on plain steps."""

    label: str
    steps: tuple[float, ...]
    share: bool  # shown as a percentage
    help: str
    group: str = "network"  # the where-from group it belongs to (see ``inference.what_if``)


SLIDERS = {
    "global_share_asn": Slider(
        "How common the network is: share of all earlier logins from it",
        (0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.13, 0.3),
        share=True,
        help=f"{ASN} A big home or mobile internet provider carries a large share of all logins; a "
        "small hosting company or VPN network, almost none. 0% means nobody had logged in from it before.",
    ),
    "asn_prior_users": Slider(
        "Different accounts that had logged in from this network",
        (0, 5, 20, 100, 1_000, 10_000, 100_000, 500_000),
        share=False,
        help="How many different user accounts had already logged in successfully from this network "
        "before this login. A big internet provider has hundreds of thousands; a network that only a "
        "handful of accounts ever used is unusual.",
    ),
    "asn_attempts_prev24h": Slider(
        "Login attempts from this network in the 24 hours before",
        (0, 5, 50, 500, 5_000, 20_000, 100_000),
        share=False,
        help="All login attempts from this network, by any account, successful or failed, in the 24 "
        "hours before this login. Busy providers have thousands a day; a quiet network, almost none.",
    ),
    "asn_prior_attack_share": Slider(
        "Share of the network's earlier attempts that came from attack IPs",
        (0, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0),
        share=True,
        help=f"{ATTACK} Of all earlier login attempts from this network, the share made from those "
        "addresses.",
    ),
    "global_share_country": Slider(
        "How common the country is: share of all earlier logins from it",
        (0, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.65),
        share=True,
        help="Most logins to this service come from a few countries; a country with a tiny share is "
        "unusual for the service as a whole (not only for this account).",
        group="country",
    ),
    "global_share_browser": Slider(
        "How common the browser version is: share of all earlier logins with it",
        (0, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2),
        share=True,
        help="The browser and its exact version (for example Chrome 87). Current versions of popular "
        "browsers are common; old or unusual versions are rare.",
        group="browser",
    ),
}
DEFINITIONS = {  # hover text, only where a name uses a technical term
    "attack_ip": ATTACK,
    "new_asn": f"The account had never logged in from this network. {ASN}",
    "new_user_agent": "User agent: the text a browser sends to identify itself (browser and version, "
    "operating system, device). The account had never sent this exact one.",
    "country_change_1h": "Also called impossible travel: a jump between countries a person can hardly "
    "make.",
    DEVICE: "In this data, desktop logins were about 10 times more often takeovers than mobile ones.",
}
KEY_NOVELTY = ("new_country", "new_asn", "new_device_type", "new_os", "new_browser")
OTHER = "Other inputs (kept as in the real login)"
START = "Typical ordinary login (starting point)"
REFERENCE_SIZE = 300  # ordinary logins that define "typical" for the explanations
MIN_SCORE_CHANGE = 5e-5  # below this the 4-decimal score does not move
MIN_EFFECT = 1e-4  # contributions smaller than this read as no effect
SMALL_SHARE, LARGE_SHARE = 0.01, 0.1  # percentage formats: two significant digits below 1%


# ---------- data and model ---------------------------------------------------------------------------


@st.cache_resource
def detector() -> Detector:
    return Detector.load(DATA_DIR, DATASET)


@st.cache_data
def test_logins() -> pl.DataFrame:
    """Test-side logins with their saved scores (issue #7)."""
    config = load_dataset_config(DATASET)
    scores = pl.read_parquet(DATA_DIR / "07_model_output" / f"{DATASET}_test_scores.parquet")
    return load_side(config, DATA_DIR, "test").collect().join(scores, on="event_id", how="inner")


@st.cache_data
def dev_logins() -> pl.DataFrame:
    """Development-side logins: the number inputs, the label and the out-of-fold score (issue #6)."""
    config = load_dataset_config(DATASET)
    scores = pl.read_parquet(DATA_DIR / "07_model_output" / f"{DATASET}_dev_oof.parquet")
    dev = load_side(config, DATA_DIR, "dev").select("event_id", "label_ato", *SLIDERS).collect()
    return dev.join(scores, on="event_id", how="inner")


@st.cache_resource
def explained_detector() -> Detector:
    """The detector, explaining relative to a fixed sample of ordinary (allowed, legitimate) test logins."""
    det = copy.copy(detector())  # keep the shared cached detector unchanged
    logins = test_logins()
    ordinary = logins.filter(~pl.col("label_ato") & (pl.col("score") < det.thresholds["step-up"]))
    det.set_reference(ordinary.sample(n=min(REFERENCE_SIZE, ordinary.height), seed=0))
    return det


def _median(frame: pl.DataFrame, column: str) -> float | None:
    value = frame[column].median()
    return float(value) if isinstance(value, int | float) else None


@st.cache_data
def usual_values() -> dict[str, tuple[float | None, float | None]]:
    """Median of each number input among development takeovers and legitimate logins."""
    dev = dev_logins()
    ato, legit = dev.filter(pl.col("label_ato")), dev.filter(~pl.col("label_ato"))
    return {c: (_median(ato, c), _median(legit, c)) for c in SLIDERS}


@st.cache_data
def device_shares() -> dict[str, float]:
    """The typical share of all earlier logins of each device type (moves with the device type)."""
    logins = test_logins()
    shares = {}
    for device in DEVICES:
        value = _median(logins.filter(pl.col(DEVICE) == device), "global_share_device_type")
        if value is not None:
            shares[device] = value
    return shares


@st.cache_data
def score_bins() -> list[tuple[float, float, int, int]]:
    """Development logins by score range: (low, high, logins, takeovers), the tiers split further."""
    det, dev = detector(), dev_logins()
    edges = [0.0, det.thresholds["step-up"], det.thresholds["review / block"], 0.2, 0.5, 1.000001]
    bins = []
    for low, high in itertools.pairwise(edges):
        part = dev.filter((pl.col("score") >= low) & (pl.col("score") < high))
        bins.append((low, high, part.height, int(part["label_ato"].sum())))
    return bins


@st.cache_data
def typical_gap_hours() -> dict[str, float | None]:
    """Hours since the previous login of a typical test login in each time range."""
    logins, hours, lower, out = test_logins(), pl.col("hours_since_prev_success"), 0.0, {}
    for name, upper in GAPS.items():
        within = logins.filter((hours >= lower) & (hours < upper))
        out[name] = typical_row(
            within, "hours_since_prev_success", ["hours_since_prev_success"]
        ).get("hours_since_prev_success")
        lower = upper
    return out


def kinds(logins: pl.DataFrame, det: Detector) -> dict[str, pl.DataFrame]:
    """Test-side logins grouped by ground truth and response tier, keyed by slug (empty ones left out)."""
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
    """The median-score returning-account login and first login of a group (ties: fixed order)."""
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


# ---------- formatting -------------------------------------------------------------------------------


def fmt(column: str, value: object) -> str:
    """An input value for people: shares as percentages, switches as on/off."""
    if value is None:
        return "no earlier attempts" if column == "asn_prior_attack_share" else "—"
    if isinstance(value, bool):
        return "on" if value else "off"
    if column in SLIDERS and SLIDERS[column].share and isinstance(value, int | float):
        return _percent(column, float(value))
    if isinstance(value, int | float):
        return f"{value:,.0f}"
    return str(value)


def _percent(column: str, share: float) -> str:
    """A share as a percentage; small shares keep two significant digits so they never read as 0."""
    if share == 0:
        return "0% (never seen)" if column.startswith("global_share_") else "0%"
    if share < SMALL_SHARE:
        return f"{share * 100:.2g}%"
    return f"{share:.0%}" if share >= LARGE_SHARE else f"{share:.1%}"


def _key(name: str) -> str:
    return f"in-{name}"


def _real(login: pl.DataFrame, column: str) -> bool:
    value = login[column][0]
    return bool(value) if value is not None else False


# ---------- choosing a login -------------------------------------------------------------------------


def introduction() -> None:
    st.markdown(
        f"This demo accompanies the master's thesis *{THESIS_EN}* ([code and results]({REPOSITORY})). "
        "A machine-learning model gives each successful login a **takeover score** and a **response**: "
        "allow it, ask for a second factor (2FA), or block it. The logins are real records of a public "
        "synthetic dataset (RBA dataset, Wiefling et al. 2022), from users the model never saw in training."
    )
    st.markdown(
        "**How to use it**\n"
        "1. **Choose one of the kinds of authentication records** below.\n"
        "2. **Choose a scenario:** an account that had logged in before, or an account's first login.\n"
        "3. **See an example:** the left side fills with the inputs of a real login of that kind, and the "
        "right side shows the model's score and response for it.\n"
        "4. **Play with the model:** change the login's inputs on the left and watch the score and the "
        "recommended response change. **Back to the real login** undoes your changes.\n"
        "5. **Compare in the table** *How the score is built*: each input's real value next to yours, and "
        "how much it raises (+) or lowers (-) the score. The last row compares the two scores."
    )


def load_inputs(login: pl.DataFrame) -> None:
    """Set every control to the login's real values."""
    state = st.session_state
    for column in SLIDERS:
        state[_key(column)] = login[column][0]
    for column in SIGNALS:
        if column in login.columns:
            state[_key(column)] = _real(login, column)
    state[_key(DEVICE)] = login[DEVICE][0]
    state[_key(GAP)] = gap_of(login["hours_since_prev_success"][0])
    state["forced"] = {}  # switch -> the value the viewer had given it before a rule fixed it
    state["held"] = {}  # slider -> the value the viewer had given it before a rule fixed it
    state["loaded"] = int(login["event_id"][0])


def pick_scenario(
    counts: dict[str, int], picks: dict[str, dict[str, pl.DataFrame]]
) -> pl.DataFrame:
    """Top of the page: one button per kind of outcome, then its scenarios; returns the chosen login."""
    slugs = list(picks)
    if st.session_state.get("kind") not in slugs:  # ?example=<slug>&scenario=<name> opens one
        wanted = st.query_params.get("example", slugs[0])
        st.session_state["kind"] = wanted if wanted in slugs else slugs[0]
    for column, slug in zip(st.columns(len(slugs)), slugs, strict=True):
        chosen = slug == st.session_state["kind"]
        name, text = KINDS[slug]
        if column.button(
            name, key=f"kind-{slug}", type="primary" if chosen else "secondary", width="stretch"
        ):
            st.session_state["kind"] = slug
            st.rerun()
        column.caption(text)
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
    st.query_params.update({"example": kind, "scenario": scenario})  # the URL reopens it
    login = picks[kind][scenario]
    st.caption(describe(login, counts[kind]))
    return login


def describe(login: pl.DataFrame, count: int) -> str:
    """One line about the scenario's real login."""
    prior = int(login["prior_successes"][0])
    if prior == 0:
        account = "the account's first successful login"
    else:
        account = f"{prior:,} earlier login{'s' if prior > 1 else ''} of this account"
    truth = "**takeover**" if login["label_ato"][0] else "**legitimate**"
    return (
        f"**The login:** {account}, from {login['country'][0]} on a {login[DEVICE][0]}, "
        f"{login['timestamp'][0]:%Y-%m-%d}. Ground truth: {truth}. It is the typical (median-score) "
        f"one of the {count:,} test logins of this kind."
    )


# ---------- the inputs -------------------------------------------------------------------------------


def network_rules(login: pl.DataFrame) -> None:
    """Keep the network inputs to combinations the data has (see ``inference.what_if``)."""
    state = st.session_state
    held = state["held"]
    users, attack = _key("asn_prior_users"), _key("asn_prior_attack_share")
    if state[_key("global_share_asn")] == 0:  # never seen: no accounts either
        if "asn_prior_users" not in held:
            held["asn_prior_users"] = state[users]
        state[users] = 0
        return
    if "asn_prior_users" in held:
        state[users] = held.pop("asn_prior_users")
    if state[users] == 0:  # seen before: at least one account
        real = login["asn_prior_users"][0]
        state[users] = real if real else SLIDERS["asn_prior_users"].steps[1]
    if state[attack] is None:  # seen before: a known attack history
        real = login["asn_prior_attack_share"][0]
        state[attack] = real if real is not None else 0.0


def edited_login(login: pl.DataFrame) -> pl.DataFrame:
    """The real login with the number inputs and the device type of the controls.

    An IP is part of its network, so it never has a bigger share of logins or more accounts than the
    network, and a network never has more failed attempts than attempts.
    """
    state = st.session_state
    edited = login
    for column in SLIDERS:
        if state[_key(column)] != login[column][0]:
            value = pl.lit(state[_key(column)]).cast(login[column].dtype)
            edited = edited.with_columns(value.alias(column))
    edited = edited.with_columns(
        pl.min_horizontal("global_share_ip", "global_share_asn").alias("global_share_ip"),
        pl.min_horizontal("ip_prior_users", "asn_prior_users")
        .cast(login["ip_prior_users"].dtype)
        .alias("ip_prior_users"),
        pl.min_horizontal("asn_fails_prev24h", "asn_attempts_prev24h")
        .cast(login["asn_fails_prev24h"].dtype)
        .alias("asn_fails_prev24h"),
    )
    device = state[_key(DEVICE)]
    if device != login[DEVICE][0]:
        edited = edited.with_columns(pl.lit(device).cast(login[DEVICE].dtype).alias(DEVICE))
        if device in device_shares():
            share = pl.lit(device_shares()[device]).cast(login["global_share_device_type"].dtype)
            edited = edited.with_columns(share.alias("global_share_device_type"))
    return edited


def current_required(login: pl.DataFrame, located: pl.DataFrame) -> dict[str, tuple[bool, str]]:
    """The switches the rules fix for the current controls (see ``inference.what_if``).

    A switch a rule forced counts with the value the viewer had given it, so a rule never feeds itself.
    """
    state = st.session_state
    forced = state["forced"]
    switches = {s: forced[s] if s in forced else bool(state.get(_key(s))) for s in SIGNALS}
    moved = {spec.group for c, spec in SLIDERS.items() if state[_key(c)] != login[c][0]}
    if state[_key(DEVICE)] != login[DEVICE][0]:
        moved.add("device")
    return required_signals(login, located, switches, state.get(_key(GAP)), moved)


def follow_rules(required: dict[str, tuple[bool, str]]) -> None:
    """Set the switches a rule fixes; give one back the viewer's value once no rule fixes it."""
    state = st.session_state
    forced = state["forced"]
    for signal, (value, _reason) in required.items():
        current = bool(state.get(_key(signal)))
        if signal not in forced and current != value:
            forced[signal] = current
        state[_key(signal)] = value
    for signal in [s for s in forced if s not in required]:
        state[_key(signal)] = forced.pop(signal)


def _help(name: str, fixed: str | None = None) -> str | None:
    """The definition (only for technical terms), and why a control is greyed out."""
    parts = [DEFINITIONS.get(name), f"**Greyed out:** {fixed}" if fixed else None]
    return "\n\n".join(p for p in parts if p) or None


def slider(login: pl.DataFrame, column: str) -> None:
    """A number input on plain steps (the real value included), and where its usual values are."""
    state = st.session_state
    spec, real = SLIDERS[column], login[column][0]
    steps: list[float | None] = sorted({*spec.steps, *([real] if real is not None else [])})
    never_seen = state[_key("global_share_asn")] == 0
    if column == "asn_prior_users" and not never_seen:
        steps = [s for s in steps if s != 0]  # a network seen before had accounts
    if column == "asn_prior_attack_share" and never_seen:
        steps = [None, *steps]  # failed attempts only, or none at all
    locked = column == "asn_prior_users" and never_seen
    help_text = f"{spec.help}\n\n**Greyed out:** {NEVER_SEEN_NETWORK}" if locked else spec.help
    st.select_slider(
        spec.label,
        steps,
        key=_key(column),
        format_func=lambda v: fmt(column, v),
        help=help_text,
        disabled=locked,
    )
    takeover, legit = usual_values()[column]
    st.caption(
        f"Takeovers usually: **{fmt(column, takeover)}** · legitimate logins usually: "
        f"**{fmt(column, legit)}**"
    )


def input_panel(login: pl.DataFrame) -> pl.DataFrame:
    """Left column: every input as a control; returns the login with the controls applied."""
    if st.session_state.get("loaded") != int(login["event_id"][0]):
        load_inputs(login)
    first_login = bool(login["prior_successes"][0] == 0)
    network_rules(login)
    located = edited_login(login)
    required = current_required(login, located)
    follow_rules(required)

    st.subheader("The login's inputs")
    st.caption(
        "Each control is one input the model receives, set to the real login's value. Move any of "
        "them: the model re-scores the login with exactly these values. Greyed-out controls follow from "
        "the others (the ? says why)."
    )
    st.markdown("**Where the login comes from**")
    for column in SLIDERS:
        slider(login, column)
    devices = [*DEVICES, *([login[DEVICE][0]] if login[DEVICE][0] not in DEVICES else [])]
    st.radio("Device type", devices, horizontal=True, key=_key(DEVICE), help=_help(DEVICE))
    fixed = required["attack_ip"][1] if "attack_ip" in required else None
    st.toggle(
        "IP address on an attack list",
        key=_key("attack_ip"),
        disabled=fixed is not None,
        help=_help("attack_ip", fixed),
    )
    st.markdown("**Compared with the account's earlier logins**")
    if first_login:
        st.caption(
            "This is the account's first successful login: there is no earlier login to compare with, "
            "so these inputs do not exist yet (greyed out)."
        )
    else:
        st.select_slider("Time since this account's previous login", list(GAPS), key=_key(GAP))
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
    st.button("Back to the real login", on_click=load_inputs, args=(login,))
    return apply_switches(login, located, first_login)


def apply_switches(login: pl.DataFrame, located: pl.DataFrame, first_login: bool) -> pl.DataFrame:
    """The located login with the switches and the time gap applied, keeping dependent inputs consistent.

    Only controls moved away from the real value change the login (a missing value stays missing).
    """
    state, edited, moved = st.session_state, located, False
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
        hours = typical_gap_hours()[state[_key(GAP)]]
        if hours is not None:
            edited = edited.with_columns(pl.lit(hours).alias("hours_since_prev_success"))
    if state[_key("attack_ip")] != _real(login, "attack_ip"):
        edited = with_attack_share(edited)
    return edited


# ---------- the model's answer -----------------------------------------------------------------------


def frequency(score: float) -> str:
    """What a score this high meant in the dataset (development logins, out-of-fold scores)."""
    for low, high, logins, takeovers in score_bins():
        if low <= score < high and logins:
            return (
                f"In the dataset, **{takeovers:,} of {logins:,}** logins with a score in this range "
                f"({low:.4f} to {min(high, 1):.4f}) were takeovers (**{_percent('', takeovers / logins)}**). "
                "Overall, about 1 login in 70,000 is a takeover."
            )
    return ""


def control_rows(login: pl.DataFrame) -> dict[str, str]:
    """Login column -> the row (control) it belongs to, in the order of the controls on the left."""
    rows = {c: s.label for c, s in SLIDERS.items()}
    rows |= {DEVICE: "Device type", "global_share_device_type": "Device type"}
    rows["attack_ip"] = "IP address on an attack list"
    if login["prior_successes"][0] > 0:
        rows["hours_since_prev_success"] = "Time since this account's previous login"
        rows |= {c: text for c, text in HISTORY_SIGNALS.items() if c in login.columns}
        rows["n_new_key_signals"] = "Number of new signals at once (follows the switches)"
    return rows


def source_of(name: str, columns: list[str]) -> str:
    """The login column a model input comes from (one-hot, cyclic and never-seen encodings)."""
    if name.startswith(NEVER_SEEN_PREFIX):
        return name.removeprefix(NEVER_SEEN_PREFIX)
    if name in columns:
        return name
    return next((c for c in columns if name.startswith(f"{c}_")), name)


def _effect(contribution: float) -> str:
    if contribution > MIN_EFFECT:
        return "raises"
    return "lowers" if contribution < -MIN_EFFECT else ""


def score_table(
    det: Detector, login: pl.DataFrame, edited: pl.DataFrame, score: float, original: float
) -> pd.DataFrame:
    """One row per control (in the controls' order), the other inputs, then the score."""
    names, values = det.contributions(edited)
    rows_of, columns = control_rows(login), list(login.columns)
    part: dict[str, float] = {}
    for name, value in zip(names, values[0], strict=True):
        row = rows_of.get(source_of(name, columns), OTHER)
        part[row] = part.get(row, 0.0) + float(value)
    table = [(START, "", "", f"{score - float(values[0].sum()):.3f}", "")]
    shown = {START}
    for column, row in rows_of.items():
        if row in shown:
            continue
        shown.add(row)
        if column == "hours_since_prev_success":
            real, yours = gap_of(login[column][0]), st.session_state[_key(GAP)]
        else:
            real, yours = fmt(column, login[column][0]), fmt(column, edited[column][0])
        table.append(
            (row, str(real), str(yours), f"{part.get(row, 0.0):+.3f}", _effect(part.get(row, 0.0)))
        )
    other = part.get(OTHER, 0.0)
    table.append((OTHER, "", "", f"{other:+.3f}", _effect(other)))
    delta = score - original
    arrow = "↑" if delta > 0 else "↓"
    change = "no change" if abs(delta) < MIN_SCORE_CHANGE else f"{arrow} {delta:+.4f}"
    table.append(
        (
            "Takeover score",
            f"{original:.4f} ({SHORT[det.tier(original)]})",
            f"{score:.4f} ({SHORT[det.tier(score)]})",
            f"= {score:.3f}",
            change,
        )
    )
    return pd.DataFrame(
        table, columns=["input", "real login", "with your changes", "contribution", "effect"]
    )


def risk_panel(det: Detector, login: pl.DataFrame, edited: pl.DataFrame) -> None:
    """Right column: score, what it means, the response, and how the score is built."""
    score, original = float(det.score(edited)[0]), float(det.score(login)[0])
    delta = score - original
    tier = det.tier(score)
    st.subheader("The model's answer")
    st.metric(
        "Takeover score",
        f"{score:.4f}",
        delta=f"{delta:+.4f} from the real login's {original:.4f}"
        if abs(delta) >= MIN_SCORE_CHANGE
        else None,
        delta_color="inverse",
    )
    st.markdown(
        frequency(score),
        help="The 7.9 million logins of the dataset's training part, each scored by a model that had "
        "not seen it during training.",
    )
    getattr(st, TIER_STYLE[tier])(f"Response: **{RESPONSE[tier]}**", icon=None)
    st.caption(
        f"Ask for 2FA from {det.thresholds['step-up']:.4f}, the riskiest 1% of the dataset's logins; "
        f"block from {det.thresholds['review / block']:.4f}, the riskiest 0.1%. The service sets these "
        "alert budgets; the scores they fall on were fixed before the final test."
    )
    st.subheader("How the score is built")
    table = score_table(det, login, edited, score, original)
    last = len(table) - 1
    if delta >= MIN_SCORE_CHANGE:
        total = "background-color: #fdecea; color: #b42318; font-weight: bold"
    elif delta <= -MIN_SCORE_CHANGE:
        total = "background-color: #e7f6ec; color: #1e7a34; font-weight: bold"
    else:
        total = "font-weight: bold"

    def style(row: pd.Series) -> list[str]:
        if row.name == last:
            return [total] * len(row)
        changed = row["real login"] != row["with your changes"]
        return ["font-weight: bold" if changed else ""] * len(row)

    st.table(table.style.apply(style, axis=1).hide(axis="index"))
    st.caption(
        "Start from a typical ordinary login's score, add each contribution, and you get the score "
        "(tree SHAP). Rows follow the controls on the left; changed inputs in bold. The last row is red "
        "when your changes raised the score and green when they lowered it."
    )


def main() -> None:
    st.set_page_config(page_title="ATO risk demo", layout="wide")
    st.markdown(  # larger kind buttons
        "<style>[class*='st-key-kind-'] button {min-height: 3.5rem; font-size: 1.05rem;}</style>",
        unsafe_allow_html=True,
    )
    st.title("Account-takeover risk demo")
    try:
        det = explained_detector()
        counts, picks = scenarios()
    except FileNotFoundError as missing:
        st.error(
            f"Model or data not found ({missing.filename}). Run the pipeline and notebooks 06 and 07 "
            "first; see README."
        )
        st.stop()
    if not picks:
        st.info("No logins to show.")
        st.stop()
    introduction()
    login = pick_scenario(counts, picks)
    left, right = st.columns([1, 1.2])
    with left:
        edited = input_panel(login)
    with right:
        risk_panel(det, login, edited)


main()
