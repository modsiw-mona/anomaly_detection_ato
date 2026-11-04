"""Model selection for issue #6: candidate families, out-of-fold scoring and per-group metrics.

Every candidate is a scikit-learn ``Pipeline`` (preprocessing + model) trained the same way as the
logistic baseline (``model.baselines``): every takeover plus a random share of legitimate logins,
evaluated on complete, unsampled folds.

* **Supervised** — logistic regression (the bar), random forest, gradient-boosted trees
  (scikit-learn's ``HistGradientBoostingClassifier``, the LightGBM-style algorithm with no system
  dependency).
* **Unsupervised** — Isolation Forest, trained on logins without labels; the takeovers are used only
  to evaluate it.
* **PU bagging** — the average of several gradient-boosting models, each trained against a different
  random sample of the "legitimate" logins. Unlabelled is not the same as legitimate (some
  takeovers were never detected), and bagging over the unlabelled side is the standard remedy
  (Bekker & Davis, 2020).

**Per-group evaluation.** One campaign dominates the labels (dataset card: 56% of takeover rows come
from country ``RO``), so metrics are also reported for each group of takeovers against all
legitimate logins, and :func:`out_of_fold` can **hide** a group from training (leave-campaign-out):
a detector that only memorises one campaign is useless against the next attacker.
"""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data.features import LABEL, USER
from model.baselines import NEGATIVE_SHARE, SEED, training_sample
from model.evaluation import Metrics, evaluate
from model.preprocessing import build_preprocessor, to_model_frame

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]
Scorer = Callable[[pl.DataFrame], FloatArray]
UNSUPERVISED_SAMPLE = 500_000  # logins used to fit Isolation Forest
PU_BAGS = 10


def _pipeline(columns: Sequence[str], model: object) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(columns)), ("model", model)])


def logistic_regression(columns: Sequence[str], seed: int = SEED) -> Pipeline:
    """The bar from issue #5."""
    model = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed)
    return _pipeline(columns, model)


def random_forest(columns: Sequence[str], seed: int = SEED) -> Pipeline:
    """Random forest; class weights rebalanced inside every bootstrap sample."""
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    return _pipeline(columns, model)


def gradient_boosting(columns: Sequence[str], seed: int = SEED) -> Pipeline:
    """Gradient-boosted trees, regularised for ~100 training takeovers (no early stopping: too few
    takeovers to hold some out)."""
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    return _pipeline(columns, model)


SUPERVISED: dict[str, Callable[[Sequence[str], int], Pipeline]] = {
    "logistic regression": logistic_regression,
    "random forest": random_forest,
    "gradient boosting": gradient_boosting,
}


def _score_with(pipeline: Pipeline, columns: Sequence[str]) -> Scorer:
    def score(frame: pl.DataFrame) -> FloatArray:
        x = to_model_frame(frame, columns)
        if hasattr(pipeline, "predict_proba"):
            result: FloatArray = pipeline.predict_proba(x)[:, 1]
        else:
            result = pipeline.decision_function(x)
        return result

    return score


def fit_supervised(
    name: str,
    train: pl.DataFrame,
    columns: Sequence[str],
    *,
    negative_share: float = NEGATIVE_SHARE,
    seed: int = SEED,
) -> Scorer:
    """Fit one supervised candidate on every takeover + a sample of legitimate logins."""
    sample = training_sample(train, negative_share=negative_share, seed=seed)
    pipeline = SUPERVISED[name](columns, seed)
    pipeline.fit(to_model_frame(sample, columns), sample[LABEL].to_numpy())
    return _score_with(pipeline, columns)


def fit_pu_bagging(
    train: pl.DataFrame,
    columns: Sequence[str],
    *,
    n_bags: int = PU_BAGS,
    negative_share: float = NEGATIVE_SHARE,
    seed: int = SEED,
) -> Scorer:
    """Average of ``n_bags`` gradient-boosting models, each against a different unlabelled sample."""
    members = [
        fit_supervised(
            "gradient boosting", train, columns, negative_share=negative_share, seed=seed + bag
        )
        for bag in range(n_bags)
    ]

    def score(frame: pl.DataFrame) -> FloatArray:
        return np.mean([member(frame) for member in members], axis=0)

    return score


def fit_isolation_forest(
    train: pl.DataFrame, columns: Sequence[str], *, seed: int = SEED
) -> Scorer:
    """Isolation Forest on unlabelled logins; higher score = more anomalous."""
    sample = train.sort("event_id").sample(n=min(UNSUPERVISED_SAMPLE, train.height), seed=seed)
    model = IsolationForest(
        n_estimators=300, max_samples=min(4_096, sample.height), n_jobs=-1, random_state=seed
    )
    pipeline = _pipeline(columns, model)
    pipeline.fit(to_model_frame(sample, columns))

    def score(frame: pl.DataFrame) -> FloatArray:
        result: FloatArray = -pipeline.score_samples(to_model_frame(frame, columns))
        return result

    return score


Fitter = Callable[[pl.DataFrame, Sequence[str]], Scorer]


def candidate(name: str) -> Fitter:
    """Fitting function of a candidate by name (supervised, ``PU bagging`` or ``isolation forest``)."""
    if name in SUPERVISED:
        return lambda train, columns: fit_supervised(name, train, columns)
    if name == "PU bagging":
        return fit_pu_bagging
    if name == "isolation forest":
        return fit_isolation_forest
    raise KeyError(name)


def fold_per_login(dev: pl.DataFrame, fold_of_user: pl.DataFrame, column: str) -> IntArray:
    """Fold of every login of ``dev``, from a user -> fold table (``model.evaluation.assign_folds``)."""
    folds = dev.select(USER).join(fold_of_user.select(USER, column), on=USER, how="left")
    if folds[column].null_count():
        raise ValueError("every user needs a fold")
    result: IntArray = folds[column].to_numpy().astype(np.int64)
    return result


def out_of_fold(
    dev: pl.DataFrame,
    fold: IntArray,
    fit: Fitter,
    columns: Sequence[str],
    *,
    hide: BoolArray | None = None,
) -> FloatArray:
    """Score every development login with a model trained on the other folds.

    ``fold`` gives each login's fold (:func:`fold_per_login`). Takeovers where ``hide`` is True are
    removed from **training** (they are still scored): that is how the leave-campaign-out test hides
    one group of takeovers. Legitimate logins are never hidden.
    """
    hidden = np.zeros(dev.height, dtype=bool) if hide is None else hide & dev[LABEL].to_numpy()
    scores = np.empty(dev.height)
    for k in np.unique(fold):
        held_out = fold == k
        scorer = fit(dev.filter(pl.Series(~held_out & ~hidden)), columns)
        scores[held_out] = scorer(dev.filter(pl.Series(held_out)))
    return scores


def group_metrics(
    y: BoolArray, score: FloatArray, groups: Mapping[str, BoolArray]
) -> dict[str, Metrics]:
    """Metrics of each group of takeovers against **all** legitimate logins (plus ``all``)."""
    result = {"all": evaluate(y, score)}
    for name, members in groups.items():
        keep = ~y | (y & members)
        result[name] = evaluate(y[keep], score[keep])
    return result
