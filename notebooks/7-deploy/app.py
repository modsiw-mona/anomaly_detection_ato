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
from inference.predict import ALLOW, Detector, format_value  # noqa: E402

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
WHAT_IF = {
    "first_success": "First successful login of this account",
    "new_country": "Country never used by this account before",
    "new_asn": "Network (ASN) never used by this account before",
    "new_device_type": "Device type never used by this account before",
    "attack_ip": "IP address on an attack list",
    "country_change_1h": "Country changed less than an hour after the previous login",
}
KEY_COLUMNS = [
    "event_id", "timestamp", "country", "device_type", "prior_successes", "hours_since_prev_success",
    "global_share_country", "asn_prior_attack_share",
]  # fmt: skip


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


def login_panel(login: pl.DataFrame) -> pl.DataFrame:
    """Left column: the login's key fields and the what-if toggles; returns the edited login."""
    st.subheader("The login")
    shown = [c for c in KEY_COLUMNS if c in login.columns]
    st.dataframe(
        pl.DataFrame({"field": shown, "value": [format_value(c, login[c][0]) for c in shown]}),
        width="stretch",
        hide_index=True,
    )
    st.markdown(f"**Ground truth:** {'takeover' if login['label_ato'][0] else 'legitimate'}")
    st.subheader("What if…")
    first_login = (
        bool(login["prior_successes"][0] == 0) if "prior_successes" in login.columns else False
    )
    edited = login.clone()
    for column, label in WHAT_IF.items():
        if column in login.columns:
            current = bool(login[column][0]) if login[column][0] is not None else False
            # history-based signals do not exist on an account's first login; first_success is a fact
            locked = column == "first_success" or (first_login and column in HISTORY_SIGNALS)
            value = st.toggle(
                label, value=current, key=f"{column}-{login['event_id'][0]}", disabled=locked
            )
            edited = edited.with_columns(pl.lit(value).alias(column))
    if "n_new_key_signals" in login.columns and not first_login:
        key = [c for c in KEY_NOVELTY if c in edited.columns]
        edited = edited.with_columns(
            pl.sum_horizontal(key).cast(pl.UInt8).alias("n_new_key_signals")
        )
    st.caption(
        "One signal at a time; the login's other features stay as they were. Signals that need earlier logins "
        "are locked on an account's first login."
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
    st.dataframe(
        pl.concat([changed_inputs, raises, lowers]).with_columns(
            effect.alias("effect"), pl.col("contribution").round(3)
        ),
        width="stretch",
        hide_index=True,
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
    left, right = st.columns([1, 1])
    with left:
        edited = login_panel(login)
    with right:
        risk_panel(det, login, edited)


main()
