"""Model selection for issue #6: candidate families, out-of-fold scoring and per-group metrics.

Every candidate is a scikit-learn ``Pipeline`` (preprocessing + model) trained the same way as the
logistic baseline (``model.baselines``): every takeover plus a random share of legitimate logins,
evaluated on complete, unsampled folds.

* **Supervised** — logistic regression (the bar), random forest, and gradient-boosted trees in three
  implementations: scikit-learn's ``HistGradientBoostingClassifier``, LightGBM and XGBoost (the last
  two need the OpenMP runtime, ``brew install libomp`` on macOS).
* **Unsupervised** — Isolation Forest, trained on logins without labels; the takeovers are used only
  to evaluate it.
* **PU bagging** — the average of several gradient-boosting models, each trained against a different
  random sample of the "legitimate" logins. Unlabelled is not the same as legitimate (some
  takeovers were never detected), and bagging over the unlabelled side is the standard remedy
  (Bekker & Davis, 2020).

**Campaign balancing.** :func:`campaign_weights` gives the dominant campaign and all other takeovers
the same total weight in training.

**Per-group evaluation.** One campaign dominates the labels (dataset card: 56% of takeover rows come
from country ``RO``), so metrics are also reported for each group of takeovers against all
legitimate logins, and :func:`out_of_fold` can **hide** a group from training (leave-campaign-out):
a detector that only memorises one campaign is useless against the next attacker.
"""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

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
CAMPAIGN_GROUP = (
    "country"  # context column that identifies a campaign (data.features.CONTEXT_COLUMNS)
)


def _pipeline(columns: Sequence[str], model: object) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(columns)), ("model", model)])


def logistic_regression(columns: Sequence[str], seed: int = SEED) -> Pipeline:
    """Logistic regression with balanced class weights, the bar from issue #5.

    Args:
        columns: Feature columns the preprocessor handles (see ``model.preprocessing``).
        seed: Random state of the model.

    Returns:
        An unfitted ``preprocess`` then ``model`` pipeline.
    """
    model = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed)
    return _pipeline(columns, model)


def random_forest(columns: Sequence[str], seed: int = SEED) -> Pipeline:
    """Random forest; class weights rebalanced inside every bootstrap sample.

    Args:
        columns: Feature columns the preprocessor handles (see ``model.preprocessing``).
        seed: Random state of the model.

    Returns:
        An unfitted ``preprocess`` then ``model`` pipeline.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    return _pipeline(columns, model)


def gradient_boosting(columns: Sequence[str], seed: int = SEED) -> Pipeline:
    """Gradient-boosted trees, regularised for ~100 training takeovers.

    No early stopping: there are too few takeovers to hold some out.

    Args:
        columns: Feature columns the preprocessor handles (see ``model.preprocessing``).
        seed: Random state of the model.

    Returns:
        An unfitted ``preprocess`` then ``model`` pipeline.
    """
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


def lightgbm(columns: Sequence[str], seed: int = SEED) -> Pipeline:
    """LightGBM with the capacity of :func:`gradient_boosting`, plus row and column subsampling.

    Same number of trees, learning rate, leaves, minimum leaf size and L2 regularisation as
    :func:`gradient_boosting`.

    Args:
        columns: Feature columns the preprocessor handles (see ``model.preprocessing``).
        seed: Random state of the model.

    Returns:
        An unfitted ``preprocess`` then ``model`` pipeline.
    """
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=40,
        reg_lambda=1.0,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )
    return _pipeline(columns, model)


def xgboost(columns: Sequence[str], seed: int = SEED) -> Pipeline:
    """XGBoost (histogram method, leaf-wise growth) with the same capacity as :func:`lightgbm`.

    Args:
        columns: Feature columns the preprocessor handles (see ``model.preprocessing``).
        seed: Random state of the model.

    Returns:
        An unfitted ``preprocess`` then ``model`` pipeline.
    """
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        tree_method="hist",
        grow_policy="lossguide",
        max_leaves=15,
        max_depth=0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=seed,
    )
    return _pipeline(columns, model)


def forest(leaf: int, features: str | float) -> Callable[[Sequence[str], int], Pipeline]:
    """Random forest with a given minimum leaf size and features per split (the tuning grid of #6).

    Args:
        leaf: Minimum number of samples in a leaf (``min_samples_leaf``).
        features: Features considered at each split (``max_features``): a name such as ``sqrt``,
            or a fraction of the features.

    Returns:
        A builder with the signature of the other candidates: ``(columns, seed)`` to an unfitted
        pipeline.
    """

    def build(columns: Sequence[str], seed: int = SEED) -> Pipeline:
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=leaf,
            max_features=features,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        return _pipeline(columns, model)

    return build


SELECTED = "random forest (leaf 2, features sqrt)"  # issue #6: the selected model
SUPERVISED: dict[str, Callable[[Sequence[str], int], Pipeline]] = {
    "logistic regression": logistic_regression,
    "random forest": random_forest,
    "gradient boosting": gradient_boosting,
    "LightGBM": lightgbm,
    "XGBoost": xgboost,
    SELECTED: forest(2, "sqrt"),
}


def score_with(pipeline: Pipeline, columns: Sequence[str]) -> Scorer:
    """Scoring function of a fitted pipeline.

    Args:
        pipeline: A fitted pipeline with ``predict_proba`` or, failing that, ``decision_function``.
        columns: Feature columns the pipeline was fitted on.

    Returns:
        A function from a feature frame to one score per row: the probability of the takeover
        class, or the decision function when the pipeline has no ``predict_proba``.
    """

    def score(frame: pl.DataFrame) -> FloatArray:
        x = to_model_frame(frame, columns)
        if hasattr(pipeline, "predict_proba"):
            result: FloatArray = pipeline.predict_proba(x)[:, 1]
        else:
            result = pipeline.decision_function(x)
        return result

    return score


def campaign_weights(sample: pl.DataFrame, group: str = CAMPAIGN_GROUP) -> FloatArray:
    """Sample weights that stop one campaign from drowning out every other takeover.

    The takeovers of the **dominant group** (the ``group`` value with the most takeovers in the
    training sample, e.g. country ``RO``) and all other takeovers each get half of the takeovers'
    total weight; legitimate logins keep weight 1. Without it, the RO campaign (56% of takeover rows)
    dominates training: hiding it from training *doubled* the PR-AUC on the other takeovers.

    Args:
        sample: Training rows with ``label_ato`` and the ``group`` column.
        group: Column that identifies a campaign; nulls form their own group.

    Returns:
        One weight per row of ``sample``. All ones when there is no takeover or the takeovers
        belong to a single group.
    """
    y = sample[LABEL].to_numpy()
    weights = np.ones(sample.height)
    groups = sample[group].cast(pl.String).fill_null("").to_numpy()
    if not y.any():
        return weights
    values, counts = np.unique(groups[y], return_counts=True)
    if values.size < 2:  # noqa: PLR2004 — a single group: nothing to balance
        return weights
    dominant = groups == values[np.argmax(counts)]
    for members in (y & dominant, y & ~dominant):
        weights[members] = (y.sum() / 2) / members.sum()
    return weights


def train_supervised(  # noqa: PLR0913 — the knobs of one training run
    name: str,
    train: pl.DataFrame,
    columns: Sequence[str],
    *,
    balance_campaign: bool = False,
    negative_share: float = NEGATIVE_SHARE,
    seed: int = SEED,
) -> Pipeline:
    """Fitted pipeline of one supervised candidate: every takeover + a sample of legitimate logins.

    Args:
        name: Candidate name, a key of ``SUPERVISED``.
        train: Training feature rows, with the ``label_ato`` label.
        columns: Feature columns to use.
        balance_campaign: Whether to weight the training rows with :func:`campaign_weights`.
        negative_share: Fraction (0 to 1) of the legitimate logins kept in the training sample.
        seed: Seed of the sample and random state of the model.

    Returns:
        The fitted pipeline.

    Raises:
        KeyError: If ``name`` is not a supervised candidate.
    """
    sample = training_sample(train, negative_share=negative_share, seed=seed)
    pipeline = SUPERVISED[name](columns, seed)
    fit_params = {"model__sample_weight": campaign_weights(sample)} if balance_campaign else {}
    pipeline.fit(to_model_frame(sample, columns), sample[LABEL].to_numpy(), **fit_params)
    return pipeline


def fit_supervised(  # noqa: PLR0913
    name: str,
    train: pl.DataFrame,
    columns: Sequence[str],
    *,
    balance_campaign: bool = False,
    negative_share: float = NEGATIVE_SHARE,
    seed: int = SEED,
) -> Scorer:
    """Scorer of :func:`train_supervised`.

    Args:
        name: Candidate name, a key of ``SUPERVISED``.
        train: Training feature rows, with the ``label_ato`` label.
        columns: Feature columns to use.
        balance_campaign: Whether to weight the training rows with :func:`campaign_weights`.
        negative_share: Fraction (0 to 1) of the legitimate logins kept in the training sample.
        seed: Seed of the sample and random state of the model.

    Returns:
        A function from a feature frame to one score per row (see :func:`score_with`).

    Raises:
        KeyError: If ``name`` is not a supervised candidate.
    """
    pipeline = train_supervised(
        name,
        train,
        columns,
        balance_campaign=balance_campaign,
        negative_share=negative_share,
        seed=seed,
    )
    return score_with(pipeline, columns)


def fit_pu_bagging(
    train: pl.DataFrame,
    columns: Sequence[str],
    *,
    n_bags: int = PU_BAGS,
    negative_share: float = NEGATIVE_SHARE,
    seed: int = SEED,
) -> Scorer:
    """Average of ``n_bags`` gradient-boosting models, each against a different unlabelled sample.

    Args:
        train: Training feature rows, with the ``label_ato`` label.
        columns: Feature columns to use.
        n_bags: Number of models; model ``i`` uses seed ``seed + i``.
        negative_share: Fraction (0 to 1) of the unlabelled logins in each model's sample.
        seed: Seed of the first model.

    Returns:
        A function from a feature frame to one score per row: the mean of the models'
        probabilities.
    """
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
    """Isolation Forest on unlabelled logins; higher score = more anomalous.

    The labels are not used: the model is fitted on a random sample of at most
    ``UNSUPERVISED_SAMPLE`` logins of ``train``.

    Args:
        train: Training feature rows.
        columns: Feature columns to use.
        seed: Seed of the sample and random state of the model.

    Returns:
        A function from a feature frame to one score per row: the negated Isolation Forest
        ``score_samples``.
    """
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


def candidate(name: str, *, balance_campaign: bool = False) -> Fitter:
    """Fitting function of a candidate by name (supervised, ``PU bagging`` or ``isolation forest``).

    Args:
        name: A key of ``SUPERVISED``, ``PU bagging`` or ``isolation forest``.
        balance_campaign: Whether to weight training with :func:`campaign_weights`; supervised
            candidates only.

    Returns:
        A function ``(train, columns)`` that fits the candidate and returns its scorer.

    Raises:
        ValueError: If ``balance_campaign`` is set for a candidate that is not supervised.
        KeyError: If ``name`` is not a known candidate.
    """
    if name in SUPERVISED:
        return lambda train, columns: fit_supervised(
            name, train, columns, balance_campaign=balance_campaign
        )
    if balance_campaign:
        raise ValueError(f"campaign balancing is only implemented for {sorted(SUPERVISED)}")
    if name == "PU bagging":
        return fit_pu_bagging
    if name == "isolation forest":
        return fit_isolation_forest
    raise KeyError(name)


def fold_per_login(dev: pl.DataFrame, fold_of_user: pl.DataFrame, column: str) -> IntArray:
    """Fold of every login of ``dev``, from a user -> fold table (``model.evaluation.assign_folds``).

    Args:
        dev: Development logins with ``user_id``.
        fold_of_user: One row per user, with ``user_id`` and the fold column.
        column: Name of the fold column to use, for example ``fold_0``.

    Returns:
        The fold number of each login of ``dev``, in the same order.

    Raises:
        ValueError: If a user of ``dev`` has no fold.
    """
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

    Args:
        dev: Development feature rows, with the ``label_ato`` label.
        fold: Fold number of each row of ``dev``.
        fit: Fitting function of the candidate (see :func:`candidate`).
        columns: Feature columns to use.
        hide: Per row of ``dev``, True for takeovers to leave out of training; None hides none.

    Returns:
        One out-of-fold score per row of ``dev``, in the same order.
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
    """Metrics of each group of takeovers against **all** legitimate logins (plus ``all``).

    Args:
        y: True where the login is a takeover.
        score: One score per login, aligned with ``y``.
        groups: Per group name, a mask over the logins marking the group's takeovers.

    Returns:
        The metrics (see ``model.evaluation.evaluate``) of ``all`` the takeovers and of each group.
    """
    result = {"all": evaluate(y, score)}
    for name, members in groups.items():
        keep = ~y | (y & members)
        result[name] = evaluate(y[keep], score[keep])
    return result
