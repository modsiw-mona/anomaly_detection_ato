"""Score logins with the selected model: risk score, response tier and the signals behind it.

The detector is the ``.joblib`` pipeline saved in issue #6 (preprocessing + model) plus **alert
thresholds** taken from the out-of-fold development scores (issue #7): a login in the top 0.1% of
development scores goes to *review / block*, the top 1% to *step-up* (second factor), the rest is
allowed. Explanations are exact tree SHAP values, computed only when asked for.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.pipeline import Pipeline

from model.preprocessing import to_model_frame

FloatArray = npt.NDArray[np.float64]
BUDGETS: dict[str, float] = {"review / block": 0.001, "step-up": 0.01}  # strictest first
ALLOW = "allow"


def alert_thresholds(
    dev_scores: FloatArray, budgets: Mapping[str, float] = BUDGETS
) -> dict[str, float]:
    """Score threshold of each tier: the score above which that share of development logins falls.

    Tiers must be listed strictest first (smallest budget first): :meth:`Detector.tier` returns the
    first tier whose threshold a score reaches.
    """
    shares = list(budgets.values())
    if shares != sorted(shares):
        raise ValueError("budgets must be listed strictest (smallest) first")
    return {tier: float(np.quantile(dev_scores, 1 - share)) for tier, share in budgets.items()}


@dataclass
class Detector:
    """A fitted pipeline, its input columns and the tier thresholds."""

    pipeline: Pipeline
    columns: list[str]
    thresholds: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    reference_scores: FloatArray = field(default_factory=lambda: np.empty(0))  # sorted
    _explainer: Any = field(default=None, repr=False)
    _reference: FloatArray | None = field(default=None, repr=False)

    @classmethod
    def load(cls, data_dir: Path, dataset: str = "rba") -> Detector:
        """Load the model of ``dataset`` from ``data_dir/06_models`` and its thresholds."""
        models = data_dir / "06_models"
        metadata = json.loads((models / f"{dataset}_model.json").read_text())
        dev_scores = pl.read_parquet(data_dir / "07_model_output" / f"{dataset}_dev_oof.parquet")
        reference = np.sort(dev_scores["score"].to_numpy())
        return cls(
            pipeline=joblib.load(models / f"{dataset}_model.joblib"),
            columns=list(metadata["columns"]),
            thresholds=alert_thresholds(reference),
            metadata=metadata,
            reference_scores=reference,
        )

    def riskier_than(self, score: float) -> float:
        """Share of development logins (out-of-fold) with a lower score than ``score``."""
        if self.reference_scores.size == 0:
            return float("nan")
        lower = np.searchsorted(self.reference_scores, score, side="left")
        return float(lower / self.reference_scores.size)

    def score(self, frame: pl.DataFrame) -> FloatArray:
        """Takeover score of every row (probability-like, higher = more suspicious)."""
        result: FloatArray = self.pipeline.predict_proba(to_model_frame(frame, self.columns))[:, 1]
        return result

    def tier(self, score: float) -> str:
        """Response tier for one score: the strictest tier whose threshold it reaches."""
        for name, threshold in self.thresholds.items():
            if score >= threshold:
                return name
        return ALLOW

    def _shap(self, frame: pl.DataFrame) -> tuple[list[str], FloatArray]:
        import shap  # noqa: PLC0415 — heavy import, only needed for explanations

        preprocess, model = (
            self.pipeline.named_steps["preprocess"],
            self.pipeline.named_steps["model"],
        )
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(model)
        names = [n.split("__", 1)[-1] for n in preprocess.get_feature_names_out()]
        values = np.asarray(
            self._explainer.shap_values(preprocess.transform(to_model_frame(frame, self.columns)))
        )
        values = values[..., 1] if values.ndim == 3 else values  # noqa: PLR2004 — (rows, inputs, classes)
        return names, values

    def set_reference(self, typical: pl.DataFrame) -> None:
        """Explain relative to these logins (e.g. a sample of ordinary ones) instead of the model's
        average training login, which campaign balancing and undersampling make far riskier than usual.

        The mean tree-SHAP vector of the reference is subtracted, so contributions still add up to
        ``score(login) - mean score(reference)``; per-input values are an approximation of SHAP computed
        against that background. Units are the model's output (probabilities for the random forest).
        """
        _, values = self._shap(typical)
        self._reference = values.mean(axis=0)

    def explain(self, frame: pl.DataFrame, top: int = 6) -> list[pl.DataFrame]:
        """For each row, the ``top`` inputs that moved its score most (tree SHAP), with their values.

        With :meth:`set_reference`, a contribution is relative to a typical login: positive means
        "this input makes the login riskier than a typical one".
        """
        names, values = self._shap(frame)
        if self._reference is not None:
            values = values - self._reference
        explanations = []
        for i in range(frame.height):
            order = np.argsort(-np.abs(values[i]))[:top]
            explanations.append(
                pl.DataFrame(
                    {
                        "input": [names[j] for j in order],
                        "value": [_raw_value(frame, names[j], i) for j in order],
                        "contribution": values[i][order],
                    }
                )
            )
        return explanations


NEVER_SEEN_PREFIX = "missingindicator_"  # name scikit-learn gives the "never seen" indicators


SMALL_SHARE = 1e-4


def _format_share(value: float) -> str:
    """A share as a percentage; tiny non-zero shares as "< 0.01%" so they do not read as 0."""
    return "< 0.01%" if 0 < value < SMALL_SHARE else f"{value:.2%}"


def format_value(name: str, value: object) -> str:
    """A feature value for people: shares as percentages, flags as yes/no, missing as a dash."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if "share" in name:
            return _format_share(value)
        return f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}"
    if hasattr(value, "strftime"):
        return f"{value:%Y-%m-%d %H:%M:%S}"
    return str(value)


def _raw_value(raw: pl.DataFrame, name: str, row: int) -> str:
    """The login's original value behind a model input (e.g. ``hour`` for ``hour_sin``)."""
    if name.startswith(NEVER_SEEN_PREFIX):
        source = name.removeprefix(NEVER_SEEN_PREFIX)
        what = source.split("_", 1)[0].upper()  # ASN or IP
        return f"{what} never seen" if raw[source][row] is None else f"{what} seen before"
    if name in raw.columns:
        return format_value(name, raw[name][row])
    for column in raw.columns:  # one-hot and cyclic encodings: device_type_mobile, hour_sin
        if name.startswith(f"{column}_"):
            return format_value(column, raw[column][row])
    return ""


def describe(columns: Sequence[str], row: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of ``row`` that is model input, in model order (for display)."""
    return {c: row[c] for c in columns if c in row}
