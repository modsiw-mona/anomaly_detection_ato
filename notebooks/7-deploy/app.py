"""Account-takeover risk demo (issues #8 and #30): a login page that shows the proportional response.

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
data (``inference.what_if``) grey out the inputs that follow from others, with the reason.

The page is a simulated sign-in: the left side is the login attempt (where and how the person logs in), the right
side a generic sign-in card. **Log in** shows the screen the person would see for the response tier (welcome, a
code request, or the sign-in held) and, below it, what the system saw: the score, how often such scores were
takeovers, the thresholds, and how the score is built (tree-SHAP contributions per control, adding up from a
typical ordinary login's score to this one's). Nobody is authenticated: the screens only show the response.
A guided tour dims the page and highlights one part at a time (pure CSS; a tiny script only scrolls to it).
"""

import html
import json
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

from inference.bundle import DemoBundle, build_bundle, fetch  # noqa: E402
from inference.predict import ALLOW, NEVER_SEEN_PREFIX, Detector  # noqa: E402
from inference.what_if import (  # noqa: E402
    GAPS,
    HISTORY_SIGNALS,
    NEVER_SEEN_NETWORK,
    SIGNALS,
    gap_of,
    required_signals,
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
RESPONSE = {
    "review / block": "block until verified",
    "step-up": "ask for 2FA (a second factor)",
    ALLOW: "allow",
}
SHORT = {"review / block": "block", "step-up": "ask for 2FA", ALLOW: "allow"}  # in the table
KINDS = {  # stable URL slug -> button label (says what the login is and what the model did)
    "block": "Takeover: the model blocks it",
    "2fa": "Takeover: the model asks for 2FA",
    "missed": "Takeover: the model misses it",
    "false-alarm": "Real user: the model asks for 2FA by mistake",
    "ordinary": "Real user: the model lets them in",
}
SCENARIOS = {
    "returning": "An account that had logged in before",
    "first": "An account's first login",
}
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
        help="The network is the internet provider, company or cloud host the login comes through (its "
        "ASN, autonomous system number). This is the share of all earlier logins to the service that came "
        "through the same network: a big home or mobile provider carries a large share, a small hosting "
        "company or a VPN almost none, and 0% means nobody had logged in through it before.",
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
ACCENT = "#6B3FA0"
RED, ORANGE, YELLOW, GREEN, GREY = (
    ("#b42318", "#fdecea"),  # (text and border, background)
    ("#b54708", "#fff1e0"),
    ("#8a6100", "#fff8d6"),
    ("#1e7a34", "#e7f6ec"),
    ("#52514e", "#f1f1ef"),
)
TIER_COLOR = {"review / block": RED, "step-up": ORANGE, ALLOW: GREEN}
KIND_COLOR = {"block": RED, "2fa": ORANGE, "missed": GREY, "false-alarm": YELLOW, "ordinary": GREEN}
SIMULATION = "Simulation: this demo does not sign anyone in; it only shows the response the model would trigger."


@dataclass(frozen=True)
class TourStep:
    """One step of the guided tour: the part of the page it highlights and what it says."""

    target: str  # key of the container to highlight
    title: str
    text: str


TOUR = (
    TourStep(
        "tour-kinds",
        "1. Pick a login",
        "Choose what kind of login to test: a takeover the model blocks or asks for 2FA, a real user asked for "
        "2FA by mistake, or a real user let in. Then choose the account: one that had logged in before, or its "
        "first login.",
    ),
    TourStep(
        "tour-attempt",
        "2. The login attempt",
        "Where and how the person logs in: the network, the country, the browser, the device, and what is new "
        "for the account. Every control is one input of the model; greyed-out ones follow from the others.",
    ),
    TourStep(
        "tour-card",
        "3. Log in",
        "A takeover always uses the account's valid credentials, so the password alone cannot tell the owner "
        "from an attacker. Log in to see what the model decides.",
    ),
    TourStep(
        "tour-response",
        "4. What the person sees",
        "A response proportional to the risk: the person gets in, is asked to confirm it is them (a second "
        "factor an attacker does not have), or the sign-in is blocked.",
    ),
    TourStep(
        "tour-system",
        "5. What the model saw",
        "The takeover score, how often scores like it were takeovers in the data, and the two thresholds. The "
        "table shows how much each input raised or lowered the score.",
    ),
    TourStep(
        "tour-attempt",
        "6. Try your own",
        "Change any input and log in again to see how the response changes. Reset to the real login undoes "
        "your changes. That is the tour.",
    ),
)
RESPONSE_STEPS = (
    3,
    4,
)  # the steps that highlight the response: the tour logs in so it is on the page
OTHER = "Other inputs (kept as in the real login)"
START = "Typical ordinary login (starting point)"
MIN_SCORE_CHANGE = 5e-5  # below this the 4-decimal score does not move
MIN_EFFECT = 1e-4  # contributions smaller than this read as no effect
SMALL_SHARE, LARGE_SHARE = 0.01, 0.1  # percentage formats: two significant digits below 1%


# ---------- data and model ---------------------------------------------------------------------------


BUNDLE_RECORD = Path(__file__).with_name("bundle.json")  # the published demo bundle and its SHA-256


def _published_bundle() -> Path:
    """Download the published bundle (checked file by file against ``bundle.json``). ``make demo`` sets
    ``ATO_DEMO_OFFLINE``, so a local run never downloads anything."""
    if os.environ.get("ATO_DEMO_OFFLINE") or not BUNDLE_RECORD.exists():
        raise FileNotFoundError(2, "no data and no published bundle", str(DATA_DIR))
    record = json.loads(BUNDLE_RECORD.read_text())
    if not {f"{DATASET}_model.joblib", f"{DATASET}_model.json"} <= set(record["files"]):
        raise ValueError(f"the published bundle has no model for dataset {DATASET!r}")
    return fetch(record)


@st.cache_resource
def demo() -> tuple[Detector, DemoBundle]:
    """The model and the demo bundle: built from the full data when it is here (``ATO_DATA_DIR``), else read
    from ``ATO_BUNDLE_DIR``, else the published bundle. Explanations are relative to the bundle's reference."""
    if (DATA_DIR / "07_model_output" / f"{DATASET}_dev_oof.parquet").exists():
        det = Detector.load(DATA_DIR, DATASET)
        bundle = build_bundle(DATA_DIR, DATASET, list(SLIDERS), DEVICES, det)
    else:
        folder = Path(os.environ.get("ATO_BUNDLE_DIR") or _published_bundle())
        bundle = DemoBundle.read(folder)
        det = Detector.from_files(folder, DATASET, bundle.thresholds)
    det.set_reference(bundle.reference)
    return det, bundle


def explained_detector() -> Detector:
    return demo()[0]


def usual_values() -> dict[str, tuple[float | None, float | None]]:
    """Median of each number input among development takeovers and legitimate logins."""
    return demo()[1].usual


def device_shares() -> dict[str, float]:
    """The typical share of all earlier logins of each device type (moves with the device type)."""
    return demo()[1].device_shares


def score_bins() -> list[tuple[float, float, int, int]]:
    """Development logins by score range: (low, high, logins, takeovers), the tiers split further."""
    return demo()[1].score_bins


def typical_gap_hours() -> dict[str, float | None]:
    """Hours since the previous login of a typical test login in each time range."""
    return demo()[1].gap_hours


def scenarios() -> tuple[dict[str, int], dict[str, dict[str, pl.DataFrame]]]:
    """Group sizes and the scenario logins of each kind."""
    bundle = demo()[1]
    return bundle.counts, bundle.picks()


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
    text, button = st.columns([6, 1], vertical_alignment="bottom")
    text.markdown(
        f"This demo accompanies the master's thesis *{THESIS_EN}* ([code and results]({REPOSITORY})). A model "
        "scores every login and responds in proportion to the risk: it lets the person in, asks for a second "
        "factor (2FA), or blocks the sign-in. The logins are real records (RBA dataset, Wiefling et al. 2022)."
    )
    button.button(
        "Take the guided tour", key="tour-start", on_click=_tour_go, args=(0,), width="stretch"
    )
    st.markdown(
        "**How to use it**\n"
        "1. **Pick a login** below: what kind of record, and which account.\n"
        "2. **Press Log in** to see what the person sees, and what the model saw.\n"
        "3. **Change the attempt** on the left and log in again. **Reset to the real login** undoes it."
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


def _choice_buttons(options: dict[str, str], chosen: str, prefix: str) -> str | None:
    """A row of equal buttons, the chosen one highlighted; returns the option clicked, if any."""
    clicked = None
    for column, (slug, label) in zip(st.columns(len(options)), options.items(), strict=True):
        if column.button(
            label,
            key=f"{prefix}-{slug}",
            type="primary" if slug == chosen else "secondary",
            width="stretch",
        ):
            clicked = slug
    return clicked


def pick_scenario(
    counts: dict[str, int], picks: dict[str, dict[str, pl.DataFrame]]
) -> pl.DataFrame:
    """Top of the page: the kind of record, then the account; shows the chosen login and returns it."""
    state = st.session_state
    slugs = list(picks)
    if state.get("kind") not in slugs:  # ?example=<slug>&scenario=<name> opens one
        wanted = st.query_params.get("example", slugs[0])
        state["kind"] = wanted if wanted in slugs else slugs[0]
    st.markdown("**What kind of record?**")
    clicked = _choice_buttons({s: KINDS[s] for s in slugs}, state["kind"], "kind")
    if clicked:
        state["kind"] = clicked
        st.rerun()
    kind = state["kind"]
    options = list(picks[kind])
    if state.get(f"scenario-{kind}") not in options:
        requested = st.query_params.get("scenario")
        state[f"scenario-{kind}"] = requested if requested in options else options[0]
    st.markdown("**Which account?**")
    clicked = _choice_buttons(
        {o: SCENARIOS[o] for o in options}, state[f"scenario-{kind}"], "scenario"
    )
    if clicked:
        state[f"scenario-{kind}"] = clicked
        st.rerun()
    scenario = state[f"scenario-{kind}"]
    st.query_params.update({"example": kind, "scenario": scenario})  # the URL reopens it
    login = picks[kind][scenario]
    selected_login(login, kind, scenario)
    return login


def selected_login(login: pl.DataFrame, kind: str, scenario: str) -> None:
    """The chosen login at a glance: what really happened (highlighted), the account, where from and when."""
    prior = int(login["prior_successes"][0])
    account = "first login" if prior == 0 else f"{prior:,} earlier login{'s' if prior > 1 else ''}"
    takeover = bool(login["label_ato"][0])
    color, tint = RED if takeover else GREEN
    truth = (
        f"<span style='background:{tint};color:{color};padding:0.1rem 0.6rem;border-radius:999px;"
        f"font-weight:600'>{'A takeover' if takeover else 'The real user'}</span>"
    )
    facts = (
        ("What really happened", truth),
        ("Account", html.escape(account)),
        ("From", html.escape(f"{login['country'][0]}, on a {login[DEVICE][0]}")),
        ("When", f"{login['timestamp'][0]:%d %b %Y}"),
    )
    cells = "".join(
        f"<div><div style='font-size:0.85rem;color:#6b6b6b;margin-bottom:0.2rem'>{name}</div>"
        f"<div style='font-weight:600'>{value}</div></div>"
        for name, value in facts
    )
    with st.container(border=True, key="selected"):
        st.markdown(
            f"**The selected login:** {KINDS[kind]} · {SCENARIOS[scenario].lower()}\n\n"
            f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:0.8rem'>"
            f"{cells}</div>",
            unsafe_allow_html=True,
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
    takeover, legit = usual_values()[column]
    usual = f"**Takeovers usually:** {fmt(column, takeover)}. **Legitimate logins usually:** {fmt(column, legit)}."
    help_text = "\n\n".join(
        [spec.help, usual, *([f"**Greyed out:** {NEVER_SEEN_NETWORK}"] if locked else [])]
    )
    st.select_slider(
        spec.label,
        steps,
        key=_key(column),
        format_func=lambda v: fmt(column, v),
        help=help_text,
        disabled=locked,
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

    st.subheader("2. The login attempt")
    st.caption(
        "Where and how this person is logging in. Each control is one input the model receives, set to the "
        "selected login's real values; change any of them and log in again. Greyed-out controls follow from "
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
    st.button("Reset to the real login", key="back", on_click=load_inputs, args=(login,))
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


# ---------- what the system saw -----------------------------------------------------------------------


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


def attempt_id(edited: pl.DataFrame) -> int:
    """Identifies the login attempt as the model sees it: changes when any input changes."""
    return int(edited.hash_rows(seed=0)[0])


def username(login: pl.DataFrame) -> str:
    return f"user-{str(login['user_id'][0])[-4:]}"


def sign_in_card(login: pl.DataFrame, attempt: int) -> bool:
    """The generic sign-in card, with a hint on what to do next; True once this exact attempt was submitted."""
    state = st.session_state
    event = int(login["event_id"][0])
    submitted = state.get("submitted")
    changed = isinstance(submitted, tuple) and submitted[0] == event and submitted[1] != attempt
    if changed and state.get("notified") != attempt:
        st.toast("The login attempt changed. Press Log in again to see the new response.")
        state["notified"] = attempt
    with st.container(key="tour-card", border=True):
        st.markdown("#### Example SSO")
        st.caption("Sign in to continue")
        st.text_input("Username", value=username(login), disabled=True)
        st.text_input("Password", value="correct-password", type="password", disabled=True)
        if st.button(
            "Log in again" if changed else "Log in", key="login", type="primary", width="stretch"
        ):
            state["submitted"] = submitted = (event, attempt)
            state["pressed"] = state.get("pressed", 0) + 1  # replays the card's reaction
            state["scroll"] = True
            st.toast("Login checked. The response is below.")
        st.caption(
            "A takeover always uses the account's valid credentials, so only where and how the login "
            "happens can tell the owner from an attacker."
        )
    if submitted == (event, attempt):
        st.info("Next: change anything in the login attempt on the left, then log in again.")
        return True
    if changed:
        st.info("The login attempt changed. Log in again to see the new response.")
    else:
        st.info("Next: press **Log in** to test this login.")
    return False


def card_reaction(tier: str) -> None:
    """Colour the sign-in card by the response; a refused login shakes, like a real one, once per press."""
    color, tint = TIER_COLOR[tier]
    name = f"refused-{st.session_state.get('pressed', 0)}"  # a new name replays the animation
    shake = (
        f"@keyframes {name} {{0%, 100% {{transform: translateX(0)}} 20%, 60% {{transform: translateX(-9px)}} "
        f"40%, 80% {{transform: translateX(9px)}}}}"
        if tier != ALLOW
        else ""
    )
    animation = f"animation: {name} 0.45s ease-in-out 1;" if tier != ALLOW else ""
    st.markdown(
        f"<style>{shake} .st-key-tour-card {{outline: 3px solid {color}; outline-offset: 1px; "
        f"border-radius: 0.6rem; background-color: {tint}; {animation}}}</style>",
        unsafe_allow_html=True,
    )


def message(colors: tuple[str, str], title: str, text: str) -> None:
    """A sign-in message as a real service shows it: a coloured box with a bold title."""
    color, tint = colors
    st.markdown(
        f"<div style='background:{tint};border-left:5px solid {color};border-radius:0.5rem;"
        f"padding:0.8rem 1rem;margin-bottom:0.6rem'><div style='color:{color};font-weight:700;"
        f"font-size:1.05rem'>{title}</div><div style='color:#31333f;margin-top:0.2rem'>{text}</div></div>",
        unsafe_allow_html=True,
    )


def response_screen(tier: str, attempt: int) -> None:
    """What the person would see, by response tier (simulated), with the whole 2FA flow."""
    state = st.session_state
    with st.container(key="tour-response", border=True):
        st.markdown("**What the person sees**")
        if tier == ALLOW:
            message(GREEN, "Welcome back.", "You are signed in.")
        elif tier == "step-up":
            message(
                ORANGE,
                "Sign-in on hold.",
                "Before you continue, confirm it's you with Face ID or the 6-digit code we sent to your phone.",
            )
            code = st.text_input(
                "Verification code",
                key=f"code-{attempt}",  # a new attempt starts with an empty field
                max_chars=6,
                placeholder="any 6 digits",
            )
            attacker = st.toggle(
                "Simulate an attacker: they do not have the phone, so the code is wrong",
                key=f"attacker-{attempt}",
            )
            if st.button("Verify", key="verify"):
                state[f"verified-{attempt}"] = True
            if state.get(f"verified-{attempt}"):
                if attacker:
                    message(
                        RED,
                        "Verification failed.",
                        "This sign-in is blocked and the account's owner is alerted.",
                    )
                elif code:
                    message(GREEN, "Identity verified.", "You are signed in.")
                else:
                    st.caption("Enter the code first: any 6 digits work in this demo.")
        else:
            message(
                RED,
                "Sign-in blocked.",
                "For your security, this sign-in was stopped. Our security team is reviewing it and will "
                "contact you on your recovery email.",
            )
        st.caption(SIMULATION)


def system_view(det: Detector, login: pl.DataFrame, edited: pl.DataFrame, score: float) -> None:
    """What the model saw: the score, what it means, the thresholds and how the score is built."""
    original = float(det.score(login)[0])
    delta = score - original
    tier = det.tier(score)
    with st.container(key="tour-system"):
        st.subheader("What the model saw")
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
        message(TIER_COLOR[tier], f"Response: {RESPONSE[tier]}", position(det, score))
        st.caption(
            f"Ask for 2FA from {det.thresholds['step-up']:.4f}, the riskiest 1% of the dataset's logins; "
            f"block from {det.thresholds['review / block']:.4f}, the riskiest 0.1%. The service sets these "
            "alert budgets; the scores they fall on were fixed before the final test."
        )
        with st.container(key="score-table"):
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


def position(det: Detector, score: float) -> str:
    """Where a score falls against the two thresholds."""
    block, step_up = det.thresholds["review / block"], det.thresholds["step-up"]
    if score >= block:
        return f"{score:.4f} is at or above the block threshold, {block:.4f}: the riskiest 0.1%"
    if score >= step_up:
        return f"{score:.4f} is between the 2FA threshold, {step_up:.4f}, and the block threshold, {block:.4f}"
    return f"{score:.4f} is below the 2FA threshold, {step_up:.4f}"


def login_panel(det: Detector, login: pl.DataFrame, edited: pl.DataFrame) -> None:
    """Right column: the sign-in card; once submitted, the response and what the model saw."""
    attempt = attempt_id(edited)
    if _tour_step() in RESPONSE_STEPS:  # the tour logs in to show the response
        st.session_state["submitted"] = (int(login["event_id"][0]), attempt)
    st.subheader("3. Log in")
    if not sign_in_card(login, attempt):
        return
    score = float(det.score(edited)[0])
    tier = det.tier(score)
    card_reaction(tier)
    response_screen(tier, attempt)
    system_view(det, login, edited, score)
    if st.session_state.pop("scroll", False) and _tour_step() is None:
        # with changes, the table compares the two logins; otherwise show the response
        scroll_to("score-table" if attempt != attempt_id(login) else "tour-response")


def scroll_to(key: str) -> None:
    """Scroll the page so the container with this key is in view (a tiny same-origin script)."""
    st.iframe(
        f"""<script>
        try {{
            const target = window.parent.document.querySelector(".st-key-{key}");
            if (target) {{ target.scrollIntoView({{behavior: "smooth", block: "center"}}); }}
        }} catch (error) {{}}
        </script>""",
        height=1,  # invisible: it only runs the scroll
    )


# ---------- the guided tour --------------------------------------------------------------------------


def _tour_step() -> int | None:
    step = st.session_state.get("tour")
    return step if isinstance(step, int) else None


def _tour_go(step: int | None) -> None:
    st.session_state["tour"] = step


def tour() -> None:
    """The current tour step: dim the page except the highlighted part, and a box with Back / Next / Skip.

    The white backgrounds assume the light theme, which ``.streamlit/config.toml`` sets."""
    step = _tour_step()
    if step is None:
        return
    current = TOUR[step]
    st.markdown(
        f"""<style>
        .st-key-{current.target} {{
            position: relative; z-index: 1000; border-radius: 0.6rem;
            background-color: #ffffff;
            box-shadow: 0 0 0 0.6rem #ffffff,
                0 0 0 calc(0.6rem + 4px) {ACCENT}, 0 0 0 200vmax rgba(20, 12, 32, 0.55);
        }}
        .st-key-tour-box {{
            position: fixed; right: 1.5rem; bottom: 1.5rem; z-index: 1001; width: min(26rem, 90vw);
            padding: 1rem 1.2rem; border-radius: 0.8rem; background-color: #ffffff;
            box-shadow: 0 0.5rem 2rem rgba(0, 0, 0, 0.35); border-top: 4px solid {ACCENT};
        }}
        </style>""",
        unsafe_allow_html=True,
    )
    with st.container(key="tour-box"):
        st.markdown(f"**{current.title}**")
        st.write(current.text)
        st.caption(f"Step {step + 1} of {len(TOUR)}")
        back, forward, skip = st.columns(3)
        back.button(
            "Back", key="tour-back", on_click=_tour_go, args=(step - 1,), disabled=step == 0
        )
        last = step == len(TOUR) - 1
        forward.button(
            "Done" if last else "Next",
            key="tour-next",
            type="primary",
            on_click=_tour_go,
            args=(None if last else step + 1,),
        )
        skip.button("Skip", key="tour-skip", on_click=_tour_go, args=(None,))
    scroll_to(current.target)


def page_style() -> None:
    """Kind buttons in the colour of their outcome; the tour button unlike the action buttons."""
    rules = [
        "[class*='st-key-kind-'] button, [class*='st-key-scenario-'] button {min-height: 2.8rem;}"
    ]
    for slug, (color, tint) in KIND_COLOR.items():
        rules += [
            f".st-key-kind-{slug} button {{border: 2px solid {color}; color: {color}; background: #ffffff;}}",
            f".st-key-kind-{slug} button:hover {{background: {tint}; color: {color}; border-color: {color};}}",
            f".st-key-kind-{slug} button[data-testid='stBaseButton-primary'] "
            f"{{background: {color}; color: #ffffff; border-color: {color};}}",
        ]
    rules.append(
        f".st-key-tour-start button {{border: 1.5px dashed {ACCENT}; border-radius: 999px; "
        f"background: #f4effa; color: {ACCENT}; font-size: 0.9rem; min-height: 2.2rem;}}"
    )
    st.markdown(f"<style>{' '.join(rules)}</style>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="ATO risk demo", layout="wide")
    page_style()
    st.title("Account-takeover risk demo")
    try:
        with st.spinner(
            "Loading the model and the example logins (a first visit downloads about 10 MB)..."
        ):
            det = explained_detector()
            counts, picks = scenarios()
    except FileNotFoundError as missing:
        st.error(
            f"Model or data not found ({missing.filename}). Run the pipeline and notebooks 06 and 07 "
            "first; see README."
        )
        st.stop()
    except (OSError, ValueError) as failure:  # the published bundle: not downloaded or not verified
        st.error(
            f"The demo's data could not be loaded ({failure}). Please reload in a few minutes."
        )
        st.stop()
    if not picks:
        st.info("No logins to show.")
        st.stop()
    introduction()
    st.subheader("1. Pick a login")
    with st.container(key="tour-kinds"):
        login = pick_scenario(counts, picks)
    left, right = st.columns([1, 1.2])
    with left, st.container(key="tour-attempt"):
        edited = input_panel(login)
    with right:
        login_panel(det, login, edited)
    tour()


main()
