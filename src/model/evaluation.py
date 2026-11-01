"""Evaluation protocol for extremely imbalanced takeover detection (development plan, section 5).

* **Folds** — repeated k-fold on the development side, **grouped by user** (all logins of a user in
  the same fold) and **stratified by label at user level** (takeover users spread evenly).
* **Metrics** — PR-AUC (average precision) as the primary metric; recall, precision and *friction*
  (share of legitimate logins challenged) at **alert budgets** — the top fraction of logins a
  security team could review. Rule-based scores have many ties, so budget metrics use the
  **expected value under random tie-breaking** instead of an arbitrary order.
* **Uncertainty** — mean ± std across folds, and a bootstrap interval for PR-AUC that resamples the
  **takeovers** (the scarce class). With millions of legitimate logins their sampling variation is
  negligible, so they are held fixed; this makes each replicate cost O(positives).
"""

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

USER, LABEL = "user_id", "label_ato"
SEED = 20261102
BUDGETS = (0.001, 0.01)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def assign_folds(
    features: pl.DataFrame, *, n_folds: int = 5, n_repeats: int = 3, seed: int = SEED
) -> pl.DataFrame:
    """Fold of every user for each repeat (columns ``fold_0`` … ``fold_{n_repeats-1}``)."""
    users = features.group_by(USER).agg(pl.col(LABEL).any().alias("has_takeover")).sort(USER)
    flags = users["has_takeover"].to_numpy()
    rng = np.random.default_rng(seed)
    folds = {}
    for repeat in range(n_repeats):
        fold = np.empty(users.height, dtype=np.int8)
        for flag in (True, False):
            members = rng.permutation(np.flatnonzero(flags == flag))
            fold[members] = np.arange(members.size) % n_folds
        folds[f"fold_{repeat}"] = fold
    return users.with_columns(pl.Series(name, values) for name, values in folds.items())


def _budget_counts(y: BoolArray, score: FloatArray, budget: float) -> tuple[float, int]:
    """Expected true positives among the top ``budget`` share, and the number flagged."""
    n = y.size
    k = max(1, int(np.ceil(budget * n)))
    threshold = np.partition(score, n - k)[n - k]
    above, tied = score > threshold, score == threshold
    take_from_ties = k - int(above.sum())
    tp = y[above].sum() + y[tied].sum() * take_from_ties / tied.sum()
    return float(tp), k


class Metrics(TypedDict):
    """Metrics of one score on one evaluation set."""

    pr_auc: float
    roc_auc: float
    prevalence: float
    recall_at_0_1pct: float
    recall_at_1pct: float
    precision_at_1pct: float
    friction_at_1pct: float


def evaluate(y: BoolArray, score: FloatArray) -> Metrics:
    """All protocol metrics for one score vector (higher = more suspicious)."""
    positives = int(y.sum())
    tp_01, _ = _budget_counts(y, score, BUDGETS[0])
    tp_1, flagged_1 = _budget_counts(y, score, BUDGETS[1])
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "prevalence": positives / y.size,
        "recall_at_0_1pct": tp_01 / positives,
        "recall_at_1pct": tp_1 / positives,
        "precision_at_1pct": tp_1 / flagged_1,
        "friction_at_1pct": (flagged_1 - tp_1) / (y.size - positives),
    }


def average_precision_from_positives(
    positive_scores: FloatArray, negative_scores_sorted: FloatArray
) -> float:
    """Average precision computed from the positives' view (identical to scikit-learn's AP).

    AP is the mean, over positives, of the precision at the threshold equal to that positive's
    score, counting every item scoring at least as high. Needs negatives sorted ascending.
    """
    pos = np.sort(positive_scores)[::-1]
    tp_ge = np.searchsorted(-pos, -pos, side="right")  # positives scoring >= each positive
    fp_ge = negative_scores_sorted.size - np.searchsorted(negative_scores_sorted, pos, side="left")
    return float(np.mean(tp_ge / (tp_ge + fp_ge)))


def bootstrap_pr_auc(
    y: BoolArray, score: FloatArray, *, n_boot: int = 2_000, seed: int = SEED, level: float = 0.95
) -> tuple[float, float]:
    """Bootstrap interval for PR-AUC, resampling positives (negatives held fixed)."""
    rng = np.random.default_rng(seed)
    positives, negatives = score[y], np.sort(score[~y])
    replicates = [
        average_precision_from_positives(rng.choice(positives, positives.size), negatives)
        for _ in range(n_boot)
    ]
    tail = (1 - level) / 2
    low, high = np.quantile(replicates, [tail, 1 - tail])
    return float(low), float(high)


def summarise(per_fold: Sequence[Metrics]) -> pl.DataFrame:
    """Mean, std, min and max of every metric across folds."""
    frame = pl.DataFrame(per_fold)
    return pl.DataFrame(
        {
            "metric": frame.columns,
            "mean": [frame[c].mean() for c in frame.columns],
            "std": [frame[c].std() for c in frame.columns],
            "min": [frame[c].min() for c in frame.columns],
            "max": [frame[c].max() for c in frame.columns],
        }
    )
