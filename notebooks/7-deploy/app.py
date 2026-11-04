"""Account-takeover risk demo (issue #8): score a login, choose a response, show why.

Run from the repository root (after ``make data intermediate features split`` and notebooks 06 and 07):

    make demo

Logins come from the **test side** of the split (users the model never saw). Pick one, see its risk
score, the response tier (allow / step-up / review-block) and the inputs that moved the score; then
change a few behavioural signals ("what if this login came from a known country?") and re-score.
"""

import copy
import os
import sys
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
HISTORY_SIGNALS = {
    "new_country",
    "new_asn",
    "new_device_type",
    "country_change_1h",
}  # need earlier logins
KEY_NOVELTY = (
    "new_country",
    "new_asn",
    "new_device_type",
    "new_os",
    "new_browser",
)  # n_new_key_signals
WHAT_IF = {  # behaviour signals that can be switched (positive wording: on = the signal is present)
    "new_country": "New country for this account",
    "new_asn": "New network (ASN) for this account",
    "new_device_type": "New device type for this account",
    "attack_ip": "IP address on an attack list",
    "country_change_1h": "Country changed within an hour of the previous login",
}
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
    **{
        f"new_{s}": f"New {s.replace('_', ' ')} for this account"
        for s in ("ip", "os", "browser", "user_agent", "region", "city")
    },
    **WHAT_IF,
}


@st.cache_resource
def detector() -> Detector:
    return Detector.load(DATA_DIR, DATASET)


REFERENCE_SIZE = 300  # ordinary logins that define "typical" for the explanations
MIN_EFFECT = 1e-4  # contributions smaller than this are not shown


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


@st.cache_data
def test_logins() -> pl.DataFrame:
    """Test-side logins with their saved scores (issue #7)."""
    config = load_dataset_config(DATASET)
    scores = pl.read_parquet(DATA_DIR / "07_model_output" / f"{DATASET}_test_scores.parquet")
    return load_side(config, DATA_DIR, "test").collect().join(scores, on="event_id", how="inner")


EXAMPLES = {  # stable URL slug -> label
    "review": "Takeover → review / block",
    "step-up": "Takeover → step-up only",
    "missed": "Takeover the model misses",
    "false-alarm": "Legitimate login flagged (false alarm)",
    "ordinary": "Ordinary legitimate login",
}


def examples(logins: pl.DataFrame, det: Detector) -> dict[str, pl.DataFrame]:
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


def _int_param(name: str) -> int:
    try:
        return int(st.query_params.get(name, 0))
    except ValueError:
        return 0


def pick_login(groups: dict[str, pl.DataFrame], det: Detector) -> pl.DataFrame:
    """Sidebar: choose a kind of example and its number; returns that one login."""
    with st.sidebar:
        st.header("Pick a login")
        # ?example=<slug>&n=<number> opens a given example directly (slugs: see EXAMPLES)
        slugs = list(groups)
        wanted = st.query_params.get("example", slugs[0])
        start_kind = slugs.index(wanted) if wanted in slugs else 0
        kind = st.radio("Example", slugs, index=start_kind, format_func=EXAMPLES.__getitem__)
        pool = groups[kind]
        st.caption(f"{pool.height:,} logins of this kind on the test side")
        last = max(pool.height - 1, 0)
        start_n = min(max(_int_param("n"), 0), last)
        pick = st.number_input("Example number", min_value=0, max_value=last, value=start_n, step=1)
        st.divider()
        st.markdown(
            f"**Thresholds** (top 1% / 0.1% of development scores)\n\n"
            f"- step-up ≥ `{det.thresholds['step-up']:.4f}`\n"
            f"- review / block ≥ `{det.thresholds['review / block']:.4f}`"
        )
    # a fixed pseudo-random order, so example 0 is a typical login rather than the dataset's first day
    return pool.sort(pl.col("event_id").hash(seed=0))[int(pick)]


def label(column: str) -> str:
    """Plain-language name of an input; never-seen indicators and encodings read from their source."""
    if column.startswith(NEVER_SEEN_PREFIX):
        source = column.removeprefix(NEVER_SEEN_PREFIX)
        return f"{source.split('_', 1)[0].upper()} never seen before (for: {LABELS.get(source, source)})"
    return LABELS.get(column, column)


def login_panel(login: pl.DataFrame) -> pl.DataFrame:
    """Left column: the login's key fields and its behaviour signals; returns the edited login."""
    st.subheader("The login")
    shown = [c for c in KEY_COLUMNS if c in login.columns]
    st.table(
        pl.DataFrame(
            {
                "field": [label(c) for c in shown],
                "value": [format_value(c, login[c][0]) for c in shown],
            }
        ),
    )
    st.markdown(f"**Ground truth:** {'takeover' if login['label_ato'][0] else 'legitimate'}")
    st.subheader("Behaviour signals — what if…")
    first_login = (
        bool(login["prior_successes"][0] == 0) if "prior_successes" in login.columns else False
    )
    if first_login:
        st.info(
            "This is the account's **first** successful login. There is no earlier login to compare it with, "
            "so the account-history signals (new country, network or device, country change) do not apply. "
            "Its score comes from global signals: how rare its country, network and browser are among all "
            "logins, and the network's attack history (see *Why*)."
        )
    edited = login.clone()
    for column, text in WHAT_IF.items():
        if column not in login.columns or (first_login and column in HISTORY_SIGNALS):
            continue
        current = bool(login[column][0]) if login[column][0] is not None else False
        value = st.toggle(text, value=current, key=f"{column}-{login['event_id'][0]}")
        edited = edited.with_columns(pl.lit(value).alias(column))
    if "n_new_key_signals" in login.columns and not first_login:
        key = [c for c in KEY_NOVELTY if c in edited.columns]
        edited = edited.with_columns(
            pl.sum_horizontal(key).cast(pl.UInt8).alias("n_new_key_signals")
        )
    st.caption(
        "On = the signal is present in this login. Switch one to see how the score and the response would "
        "change; the login's other features stay as they were."
    )
    return edited


def risk_panel(det: Detector, login: pl.DataFrame, edited: pl.DataFrame) -> None:
    """Right column: score, rank, response tier and the inputs behind the score."""
    score, original = float(det.score(edited)[0]), float(det.score(login)[0])
    toggled = [c for c in WHAT_IF if c in login.columns and bool(edited[c][0]) != bool(login[c][0])]
    changed = bool(toggled)
    tier = det.tier(score)
    style = TIER_STYLE[tier]
    st.subheader("Risk")
    st.metric(
        "Takeover score",
        f"{score:.4f}",
        delta=f"{score - original:+.4f} vs the real login" if changed else None,
        delta_color="inverse",
    )
    rank = det.riskier_than(score)
    rank_text = "more than 99.99%" if rank > 0.9999 else f"{rank:.2%}"  # noqa: PLR2004
    st.caption(f"Riskier than **{rank_text}** of development logins.")
    getattr(st, style)(f"Response: **{tier}**", icon=None)
    st.subheader("Why")
    everything = det.explain(edited, top=len(det.columns) * 3)[0]  # every input, by |contribution|
    changed_inputs = everything.filter(
        pl.col("input").is_in(toggled)
    )  # always show what was changed
    rest = everything.filter(~pl.col("input").is_in(toggled))
    raises = rest.filter(pl.col("contribution") > MIN_EFFECT).head(4)  # riskier first
    lowers = rest.filter(pl.col("contribution") < -MIN_EFFECT).head(3)
    effect = (
        pl.when(pl.col("contribution") > 0).then(pl.lit("↑ raises")).otherwise(pl.lit("↓ lowers"))
    )
    table = pl.concat([changed_inputs, raises, lowers])
    st.table(
        table.with_columns(
            pl.col("input").map_elements(label, return_dtype=pl.String),
        ).with_columns(effect.alias("effect"), pl.col("contribution").round(3)),
    )
    st.caption(
        "Tree SHAP values relative to a typical login: any input you changed, then the inputs that make this "
        "login riskier than a typical one, then those that make it safer."
    )


def main() -> None:
    st.set_page_config(page_title="ATO risk demo", layout="wide")
    st.title("Account-takeover risk — demo")
    st.caption(
        "Proof of concept for the thesis *Detección de Account Takeover mediante identificación automática de "
        "anomalías comportamentales en registros de autenticación*. Synthetic data (RBA dataset, Wiefling et al. 2022); "
        "logins from users the model never saw in training."
    )
    try:
        det, logins = explained_detector(), test_logins()
    except FileNotFoundError as missing:
        st.error(
            f"Model or data not found ({missing.filename}). Run the pipeline and notebooks 06 and 07 first; see README."
        )
        st.stop()
    groups = examples(logins, det)
    if not groups:
        st.info("No logins to show.")
        st.stop()
    login = pick_login(groups, det)
    left, right = st.columns([1, 1.2])
    with left:
        edited = login_panel(login)
    with right:
        risk_panel(det, login, edited)


main()
