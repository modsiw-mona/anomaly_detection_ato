"""Baselines for issue #5: a dummy, single-signal rules from the EDA, and a simple trained model.

* **Dummy** — the same score for every login: the "no skill" reference.
* **Rules** — new country for the user; attack IP; both (score 0, 1 or 2). On a first login "new
  country" is undefined and counts as not new, like Freeman's score (see ``model.freeman``).
* **Logistic regression** — the simplest trained model on the default features, used for learning
  and scalability curves. For speed it trains on every takeover plus a random share of legitimate
  logins, with **balanced class weights**; it is always *evaluated* on complete, unsampled folds
  (see ``model.evaluation``).

  Weighting matters more than the model here. Re-weighting the sample back to the real class
  proportions (≈1 takeover per 70,000 logins) let the L2 penalty wash out the ~90 training takeovers:
  PR-AUC 0.0001 on the five folds of the first repeat, worse than the rules. Balanced class weights
  give 0.033 on the same folds (notebook 5-models).
"""

from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data.features import LABEL, feature_columns
from data.intermediate import EVENT_ID
from model.preprocessing import build_preprocessor, to_model_frame

FloatArray = np.typing.NDArray[np.float64]
NEGATIVE_SHARE = 0.05
SEED = 20261102


def rule_scores(frame: pl.DataFrame) -> dict[str, FloatArray]:
    """Scores of the dummy and the single-signal rules for every row of ``frame``.

    Args:
        frame: Feature rows with the ``new_country`` and ``attack_ip`` columns. A null
            ``new_country`` (first login) counts as not new.

    Returns:
        One score array per baseline name (``dummy (constant)``, ``rule: new country``,
        ``rule: attack IP``, ``rule: new country + attack IP``), each with one value per row of
        ``frame``; higher means more suspicious.
    """
    new_country = frame["new_country"].fill_null(value=False).cast(pl.Float64).to_numpy()
    attack_ip = frame["attack_ip"].cast(pl.Float64).to_numpy()
    return {
        "dummy (constant)": np.zeros(frame.height),
        "rule: new country": new_country,
        "rule: attack IP": attack_ip,
        "rule: new country + attack IP": new_country + attack_ip,
    }


def training_sample(
    train: pl.DataFrame, *, negative_share: float = NEGATIVE_SHARE, seed: int = SEED
) -> pl.DataFrame:
    """Every takeover plus a random ``negative_share`` of legitimate logins.

    The seed picks row *positions*, so ``train`` is first put in a canonical order (by event id):
    the same seed then picks the same logins whatever order the rows arrived in (polars joins and
    group-bys do not guarantee row order).

    Args:
        train: Training feature rows with the ``event_id`` and ``label_ato`` columns.
        negative_share: Fraction (0 to 1) of the legitimate logins to keep.
        seed: Seed of the random sample of legitimate logins.

    Returns:
        The takeover rows followed by the sampled legitimate rows, same columns as ``train``.
    """
    train = train.sort(EVENT_ID)
    positives = train.filter(pl.col(LABEL))
    negatives = train.filter(~pl.col(LABEL)).sample(fraction=negative_share, seed=seed)
    return pl.concat([positives, negatives])


def logistic_baseline(columns: Sequence[str]) -> Pipeline:
    """Unfitted preprocessing + logistic regression pipeline.

    Args:
        columns: Feature columns the preprocessor handles (see ``model.preprocessing``).

    Returns:
        A ``preprocess`` then ``model`` pipeline; the logistic regression uses balanced class
        weights.
    """
    return Pipeline(
        [
            ("preprocess", build_preprocessor(columns)),
            ("model", LogisticRegression(max_iter=2_000, class_weight="balanced")),
        ]
    )


def fit_logistic(
    train: pl.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    negative_share: float = NEGATIVE_SHARE,
    seed: int = SEED,
) -> Pipeline:
    """Fit the logistic baseline on a sample of ``train`` (all takeovers kept).

    Args:
        train: Training feature rows, with the ``label_ato`` label.
        columns: Feature columns to use; None means the default ``feature_columns()``.
        negative_share: Fraction (0 to 1) of the legitimate logins kept in the sample.
        seed: Seed of the sample of legitimate logins.

    Returns:
        The fitted pipeline (see :func:`logistic_baseline`).
    """
    columns = list(columns or feature_columns())
    sample = training_sample(train, negative_share=negative_share, seed=seed)
    pipeline = logistic_baseline(columns)
    pipeline.fit(to_model_frame(sample, columns), sample[LABEL].to_numpy())
    return pipeline


def predict_scores(
    pipeline: Pipeline, frame: pl.DataFrame, columns: Sequence[str] | None = None
) -> FloatArray:
    """Takeover scores (log-odds) for every row of ``frame``.

    Args:
        pipeline: A fitted pipeline with a ``decision_function``, such as the one from
            :func:`fit_logistic`.
        frame: Feature rows to score.
        columns: Feature columns the pipeline was fitted on; None means the default
            ``feature_columns()``.

    Returns:
        One log-odds score per row of ``frame``, in the same order; higher means more suspicious.
    """
    columns = list(columns or feature_columns())
    scores: FloatArray = pipeline.decision_function(to_model_frame(frame, columns))
    return scores
