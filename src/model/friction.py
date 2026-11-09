"""Friction per user: how often a legitimate user would be asked to re-authenticate.

Two studies on this dataset, Wiefling et al. (2022, ACM TOPS) and Rotter et al. (2025, ACM AISec),
measure risk-based authentication **per user**, not per login. They set the score threshold so that a
given share of takeovers is detected (the true-positive rate), then compute, for each legitimate user,
the share of that user's logins that would have been asked to re-authenticate, and report the
distribution over users (the median user first). This module computes the same metric for any score.

* **Rate per user** -- legitimate logins of the user scoring at or above the threshold, over all the
  user's legitimate logins. Takeover logins are left out of the rate; they set the threshold.
* **Thresholds** -- fixed ones (the alert budgets set on development data), or the threshold for a
  **detection rate**: the score of the k-th highest takeover, with k = ceil(rate x takeovers), i.e. the
  highest threshold that still detects that share. With few takeovers the realised share is the next
  step up (for example 27 of 28 = 96.4% for a 94.2% target).
* **Uncertainty** -- as :func:`model.evaluation.bootstrap_pr_auc`: the takeovers are resampled and the
  legitimate logins held fixed (with hundreds of thousands of users, their sampling variation is
  negligible). Resampling moves the threshold for a detection rate, so the interval shows how much
  the friction depends on which takeovers happened to be in the data.
"""

from collections.abc import Mapping, Sequence
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import polars as pl

from model.evaluation import LABEL, SEED, USER

# Detection rates at which Rotter et al. (2025) report friction for the dataset's real takeovers.
REAUTH_RATES = (0.942, 0.897, 0.828, 0.782)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class Friction(TypedDict):
    """Friction of one threshold: detection and the per-user re-authentication distribution."""

    threshold: float
    detected: float  # share of takeovers scoring at or above the threshold
    median: float  # share of logins re-authenticated, for the median legitimate user
    mean: float
    p90: float
    p99: float
    ever: float  # share of legitimate users asked at least once
    users: int


def threshold_for_rate(positive_scores: FloatArray, rate: float) -> float:
    """Highest threshold that detects at least ``rate`` of the takeovers."""
    if not 0 < rate <= 1:
        raise ValueError("rate must be in (0, 1]")
    if positive_scores.size == 0:
        raise ValueError("no takeovers to set the threshold on")
    ordered = np.sort(positive_scores)[::-1]
    k = int(np.ceil(rate * ordered.size - 1e-9))  # tolerance: 0.75 * 4 must stay 3
    return float(ordered[k - 1])


def _check(frame: pl.DataFrame, score: str) -> None:
    if frame[score].null_count():
        raise ValueError(f"{score} has nulls: drop the logins the score does not cover first")


def challenged_counts(
    frame: pl.DataFrame, thresholds: Sequence[float], *, score: str = "score"
) -> tuple[IntArray, IntArray]:
    """Per legitimate user: number of logins, and number at or above each threshold (users x T)."""
    _check(frame, score)
    legit = frame.filter(~pl.col(LABEL))
    per_user = legit.group_by(USER, maintain_order=True).agg(  # same order every run
        pl.len().alias("_n"),
        *[(pl.col(score) >= t).sum().alias(f"_t{j}") for j, t in enumerate(thresholds)],
    )
    counts = per_user.select([f"_t{j}" for j in range(len(thresholds))]).to_numpy()
    return per_user["_n"].to_numpy().astype(np.int64), counts.astype(np.int64).reshape(
        per_user.height, len(thresholds)
    )


def _summary(rates: FloatArray) -> dict[str, float]:
    median, p90, p99 = np.quantile(rates, [0.5, 0.9, 0.99])
    return {
        "median": float(median),
        "mean": float(rates.mean()),
        "p90": float(p90),
        "p99": float(p99),
        "ever": float((rates > 0).mean()),
    }


def user_friction(
    frame: pl.DataFrame,
    *,
    thresholds: Mapping[str, float] | None = None,
    rates: Sequence[float] = REAUTH_RATES,
    score: str = "score",
) -> dict[str, Friction]:
    """Friction at each fixed threshold (by name) and at the threshold of each detection rate.

    ``frame`` holds ``user_id``, ``label_ato`` and ``score`` for every login to evaluate.
    """
    _check(frame, score)
    positives = frame.filter(pl.col(LABEL))[score].to_numpy()
    named = dict(thresholds or {})
    named |= {f"detect {r:.1%}": threshold_for_rate(positives, r) for r in rates}
    logins, counts = challenged_counts(frame, list(named.values()), score=score)
    result: dict[str, Friction] = {}
    for j, (name, t) in enumerate(named.items()):
        summary = _summary(counts[:, j] / logins)
        result[name] = Friction(
            threshold=t,
            detected=float((positives >= t).mean()),
            median=summary["median"],
            mean=summary["mean"],
            p90=summary["p90"],
            p99=summary["p99"],
            ever=summary["ever"],
            users=int(logins.size),
        )
    return result


def bootstrap_friction(  # noqa: PLR0913 — the knobs of one bootstrap
    frame: pl.DataFrame,
    *,
    rates: Sequence[float] = REAUTH_RATES,
    score: str = "score",
    n_boot: int = 2_000,
    seed: int = SEED,
    level: float = 0.95,
) -> pl.DataFrame:
    """Bootstrap interval of the mean rate and of the share of users ever asked, per detection rate.

    Every threshold a replicate can pick is one of the takeovers' scores, so the per-user counts are
    computed once for each of them and each replicate only looks them up.
    """
    _check(frame, score)
    positives = frame.filter(pl.col(LABEL))[score].to_numpy()
    candidates = np.unique(positives)
    logins, counts = challenged_counts(frame, candidates.tolist(), score=score)
    per_user = counts / logins[:, None]
    mean_at = per_user.mean(axis=0)
    ever_at = (counts > 0).mean(axis=0)
    rng = np.random.default_rng(seed)
    tail = (1 - level) / 2
    rows = []
    for rate in rates:
        picks = np.array(
            [
                np.searchsorted(
                    candidates, threshold_for_rate(rng.choice(positives, positives.size), rate)
                )
                for _ in range(n_boot)
            ]
        )
        mean_low, mean_high = np.quantile(mean_at[picks], [tail, 1 - tail])
        ever_low, ever_high = np.quantile(ever_at[picks], [tail, 1 - tail])
        rows.append(
            {
                "detection rate": rate,
                "mean low": float(mean_low),
                "mean high": float(mean_high),
                "ever low": float(ever_low),
                "ever high": float(ever_high),
            }
        )
    return pl.DataFrame(rows)
