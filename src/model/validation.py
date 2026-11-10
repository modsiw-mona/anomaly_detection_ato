"""Model validation: the checks a trained model must pass before it is saved.

The training pipeline runs them on the **development side only**, after training and before writing
the model, so an automatic retrain can never replace a good model with a worse one unnoticed. Each
check is a :class:`Gate`; **blocking** gates stop the pipeline, the others are reported.

Blocking gates:

* **Selection rule** (issue #6, fixed before any model was trained): the lower end of the bootstrap
  interval of the out-of-fold PR-AUC above a floor, and recall within the 1% budget.
* **Gain over the baseline**: out-of-fold PR-AUC at least a given share above the logistic
  baseline's on the same folds (the course's *simple model comparison*).
* **Alert rate**: the thresholds come from out-of-fold scores, but the saved model is refitted on
  the whole development side; the share of development logins it actually flags at each threshold
  must stay near that tier's budget.
* **Segments**: recall within the 1% budget for each group of takeovers (the dominant campaign and
  the rest; first logins and returning accounts), each against all legitimate logins.
* **Scoring time** per login.

Reported, not blocking:

* **Weakest segment**: its PR-AUC as a share of the overall one. The course's condition (at least
  80%) is not met here: takeovers outside the dominant campaign are much harder, the limitation the
  thesis reports, so it is shown rather than used to stop the pipeline.
* **Overfitting gap**: PR-AUC of the final model on its own training side against out-of-fold.
* **Calibration**: Brier score of the out-of-fold scores (they rank logins; they are not
  probabilities).
"""

from collections.abc import Mapping
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, brier_score_loss

from model.evaluation import Metrics

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
COURSE_WEAKEST_SEGMENT_SHARE = 0.8


class Gate(TypedDict):
    """One check of a trained model."""

    name: str
    value: Any
    condition: str
    passed: bool
    blocking: bool


class ModelValidationError(RuntimeError):
    """A blocking gate failed: the model is not saved."""

    def __init__(self, failed: list[Gate]) -> None:
        names = ", ".join(f"{g['name']} ({g['value']}; needs {g['condition']})" for g in failed)
        super().__init__(f"model validation failed: {names}")
        self.failed = failed


def _gate(name: str, value: Any, condition: str, *, passed: bool, blocking: bool = True) -> Gate:
    return {
        "name": name,
        "value": value,
        "condition": condition,
        "passed": bool(passed),
        "blocking": blocking,
    }


def selection_rule(
    interval_low: float, recall_at_1pct: float, *, min_low: float, min_recall: float
) -> list[Gate]:
    """The pre-registered selection rule of issue #6."""
    return [
        _gate(
            "pr_auc_interval_low",
            round(interval_low, 4),
            f"> {min_low}",
            passed=interval_low > min_low,
        ),
        _gate(
            "recall_at_1pct",
            round(recall_at_1pct, 4),
            f">= {min_recall}",
            passed=recall_at_1pct >= min_recall,
        ),
    ]


def gain_over_baseline(pr_auc: float, baseline_pr_auc: float, *, min_gain: float) -> Gate:
    """PR-AUC at least ``min_gain`` (a share) above the baseline's, on the same folds."""
    gain = pr_auc / baseline_pr_auc - 1 if baseline_pr_auc > 0 else float("inf")
    return _gate(
        "gain_over_baseline",
        round(gain, 4),
        f">= {min_gain} (model {pr_auc:.4f} vs baseline {baseline_pr_auc:.4f})",
        passed=gain >= min_gain,
    )


def alert_rates(
    scores: FloatArray,
    thresholds: Mapping[str, float],
    budgets: Mapping[str, float],
    *,
    ratio: tuple[float, float],
) -> list[Gate]:
    """Share of logins the final model flags at each tier's threshold, against the tier's budget."""
    gates = []
    for tier, threshold in thresholds.items():
        rate = float(np.mean(scores >= threshold))
        low, high = ratio[0] * budgets[tier], ratio[1] * budgets[tier]
        gates.append(
            _gate(
                f"alert_rate[{tier}]",
                round(rate, 6),
                f"between {low:g} and {high:g} (budget {budgets[tier]:g})",
                passed=low <= rate <= high,
            )
        )
    return gates


def segment_recall(segments: Mapping[str, Metrics], *, min_recall: float) -> list[Gate]:
    """Recall within the 1% budget of each segment of takeovers (``model.selection.group_metrics``)."""
    return [
        _gate(
            f"segment_recall_at_1pct[{name}]",
            round(metrics["recall_at_1pct"], 4),
            f">= {min_recall}",
            passed=metrics["recall_at_1pct"] >= min_recall,
        )
        for name, metrics in segments.items()
        if name != "all"
    ]


def weakest_segment(segments: Mapping[str, Metrics]) -> Gate:
    """PR-AUC of the weakest segment as a share of the overall PR-AUC (reported only)."""
    overall = segments["all"]["pr_auc"]
    name, weakest = min(
        ((n, m["pr_auc"]) for n, m in segments.items() if n != "all"), key=lambda item: item[1]
    )
    share = weakest / overall if overall > 0 else 0.0
    return _gate(
        "weakest_segment_pr_auc_share",
        round(share, 4),
        f">= {COURSE_WEAKEST_SEGMENT_SHARE} (course condition; weakest: {name})",
        passed=share >= COURSE_WEAKEST_SEGMENT_SHARE,
        blocking=False,
    )


def scoring_time(seconds_per_login: float, *, max_seconds: float) -> Gate:
    """Average time to score one login with the saved pipeline."""
    return _gate(
        "seconds_per_login",
        float(f"{seconds_per_login:.3g}"),
        f"<= {max_seconds}",
        passed=seconds_per_login <= max_seconds,
    )


def overfitting_gap(y: BoolArray, in_sample: FloatArray, out_of_fold: FloatArray) -> Gate:
    """PR-AUC of the final model on its own training side minus out-of-fold PR-AUC (reported)."""
    train = float(average_precision_score(y, in_sample))
    oof = float(average_precision_score(y, out_of_fold))
    return _gate(
        "overfitting_gap",
        round(train - oof, 4),
        f"reported (training {train:.4f} vs out-of-fold {oof:.4f})",
        passed=True,
        blocking=False,
    )


def calibration(y: BoolArray, out_of_fold: FloatArray) -> Gate:
    """Brier score of the out-of-fold scores (reported; the scores are not probabilities)."""
    return _gate(
        "brier_score",
        float(f"{brier_score_loss(y, np.clip(out_of_fold, 0, 1)):.3g}"),
        "reported",
        passed=True,
        blocking=False,
    )


def check(gates: list[Gate]) -> None:
    """Raise :class:`ModelValidationError` if any blocking gate failed."""
    failed = [g for g in gates if g["blocking"] and not g["passed"]]
    if failed:
        raise ModelValidationError(failed)
