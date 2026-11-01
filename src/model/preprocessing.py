"""scikit-learn preprocessing for the feature table.

The behavioural features (``data.features``) are computed once, over the whole timeline. What *can*
learn from data — imputation values, scaling parameters, category lists — lives here, inside a
scikit-learn ``ColumnTransformer`` that is fitted on training folds only. Fitting it on the full
dataset would leak test-set statistics into training.

Each feature is handled by its family:

* imputers keep a column even when a training fold has no value for it (RTT is 93% missing), so
  the output shape never depends on the fold;
* **counts and durations** (``prior_successes``, ``hours_since_prev_success``, ``*_prior_users``,
  ``fails_*``, ``n_new_key_signals``) — median imputation, ``log1p`` (heavy-tailed), standard scaling;
* **global shares** (``global_share_*``) — median imputation, log (rarity spans orders of magnitude),
  standard scaling;
* **booleans** (``new_*``, ``first_success``, ``attack_ip``, missingness and artefact flags) — nulls
  become 0. A null novelty only happens on a first login, which ``first_success`` already marks;
* **categories** (``device_type``) — one-hot, with ``missing`` and unseen values handled;
* **cyclic time** (``hour``, ``weekday``) — sine and cosine, so 23 h sits next to 0 h;
* **raw numbers** (``rtt_ms``) — median imputation, ``log1p``, standard scaling.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

COUNTS = (
    "prior_successes", "hours_since_prev_success", "ip_prior_users", "asn_prior_users",
    "fails_1h", "fails_24h", "n_new_key_signals",
)  # fmt: skip
SHARE_PREFIX = "global_share_"
CATEGORICAL = ("device_type",)
CYCLIC = {"hour": 24, "weekday": 7}
RAW_NUMERIC = ("rtt_ms",)
SHARE_FLOOR = 1e-7  # a never-seen value has share 0; log needs a floor


def _log_share(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.log(np.maximum(x, SHARE_FLOOR))


def _cyclic(period: int) -> FunctionTransformer:
    def encode(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        angle = 2 * np.pi * np.asarray(x, dtype=float) / period
        return np.hstack([np.sin(angle), np.cos(angle)])

    def names(_transformer: object, columns: Sequence[str]) -> list[str]:
        return [f"{c}_{fn}" for fn in ("sin", "cos") for c in columns]

    return FunctionTransformer(encode, feature_names_out=names)


def to_model_frame(features: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    """Select ``columns`` with dtypes scikit-learn handles: numbers as Float64, categories as text."""
    exprs = []
    for column in columns:
        dtype = features.schema[column]
        if column in CATEGORICAL:
            exprs.append(pl.col(column).cast(pl.String).fill_null("missing"))
        elif dtype.is_numeric() or dtype == pl.Boolean:
            exprs.append(pl.col(column).cast(pl.Float64))
        else:
            raise TypeError(f"Unsupported feature {column!r} of type {dtype}")
    return features.select(exprs)


def build_preprocessor(columns: Sequence[str]) -> ColumnTransformer:
    """Unfitted ColumnTransformer for the given feature columns (any subset of the feature groups)."""
    log_counts = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    families: dict[str, list[str]] = {"counts": [], "shares": [], "booleans": [], "raw": []}
    transformers = []
    for column in columns:
        if column in COUNTS:
            families["counts"].append(column)
        elif column.startswith(SHARE_PREFIX):
            families["shares"].append(column)
        elif column in RAW_NUMERIC:
            families["raw"].append(column)
        elif column in CATEGORICAL:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            transformers.append((column, encoder, [column]))
        elif column in CYCLIC:
            transformers.append((column, _cyclic(CYCLIC[column]), [column]))
        else:
            families["booleans"].append(column)

    pipelines = {
        "counts": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), log_counts, StandardScaler()
        ),
        "shares": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            FunctionTransformer(_log_share, feature_names_out="one-to-one"),
            StandardScaler(),
        ),
        "booleans": SimpleImputer(strategy="constant", fill_value=0.0),
        "raw": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            StandardScaler(),
        ),
    }
    transformers += [(name, pipelines[name], cols) for name, cols in families.items() if cols]
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
