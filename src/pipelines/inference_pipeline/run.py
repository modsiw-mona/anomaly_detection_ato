"""Inference pipeline: saved model + a batch of logins -> takeover scores and response tiers.

The third of the three FTI pipelines. It scores a batch of logins in one go (batch inference), the
way a security team would score the previous day's logins; the demo scores one login at a time
with the same inference module (``inference.predict``).

1. **Model**: loaded through :class:`inference.predict.Detector`, which refuses a model file whose
   SHA-256 differs from its metadata and uses the thresholds saved with it.
2. **Batch**: a side of the split from the feature table (the test side by default), read
   **without its labels** and validated with the feature table's schema
   (:func:`data.feature_validation.validate_features`) before scoring.
3. **Drift**: the batch against the development side the model learned from, on the model's inputs
   (:mod:`data.drift`). Drift is **reported, not blocking**: a service keeps answering and the
   report tells the team to look; the feature pipeline is where drift between the sides stops a run.
4. **Scores and tiers**: every login's score and response tier (allow, step-up, review / block).

Outputs: ``data/07_model_output/<dataset>_<side>_scores.parquet`` (``event_id``, ``score``: the file
notebook 07 wrote and the demo bundle reads), ``<dataset>_<side>_predictions.parquet`` (plus the
tier), written together, and a report in ``data/08_reporting/<dataset>_inference_pipeline.json`` (rows, logins per tier
against each budget, drift, model SHA-256, timings). The labels are never used: measuring the
model on the test side is the final evaluation's job (notebook 07), done once. The outputs of an
earlier run are removed when a run starts, so a failed run leaves no stale scores behind.

Usage::

    make inference_pipeline                 # DATASET=rba, SIDE=test by default
"""

import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Any

import polars as pl
from pandera.errors import SchemaErrors

from data.datasets import DatasetConfig, find_repo_root, load_dataset_config
from data.drift import DRIFT_LIMIT, drift_report
from data.feature_validation import validate_features
from data.features import LABEL
from data.intermediate import publish
from data.split import load_side
from inference.predict import ALLOW, BUDGETS, Detector

logger = logging.getLogger(__name__)

SIDES = ("dev", "test")


def output_paths(config: DatasetConfig, data_dir: Path, side: str) -> dict[str, Path]:
    """The scores (as notebook 07 wrote them) and the predictions with their tiers."""
    folder = data_dir / "07_model_output"
    return {
        "scores": folder / f"{config.name}_{side}_scores.parquet",
        "predictions": folder / f"{config.name}_{side}_predictions.parquet",
    }


def report_path(config: DatasetConfig, data_dir: Path) -> Path:
    """The report of the latest inference run."""
    return data_dir / "08_reporting" / f"{config.name}_inference_pipeline.json"


def _read_batch(config: DatasetConfig, data_dir: Path, side: str) -> pl.DataFrame:
    """The logins of one side, without the label column: scoring can never use it."""
    batch = load_side(config, data_dir, side).drop(LABEL).collect()
    if batch.is_empty():
        raise ValueError(f"the {side} side has no logins to score")
    return batch


def _write_together(frames: dict[str, pl.DataFrame], paths: dict[str, Path]) -> None:
    """Write every file next to its target first, then rename them all: never a mixed pair."""
    partial = {name: path.with_name(path.name + ".part") for name, path in paths.items()}
    try:
        for name, frame in frames.items():
            paths[name].parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(partial[name])
    except BaseException:
        for path in partial.values():
            path.unlink(missing_ok=True)
        raise
    for name in frames:
        publish(partial[name], paths[name])


def run(
    config: DatasetConfig,
    data_dir: Path,
    *,
    side: str = "test",
    drift_limit: float = DRIFT_LIMIT,
) -> dict[str, Any]:
    """Score one side of the split with the saved model; return the report.

    The outputs of an earlier run are removed first, so a failed run never leaves scores of another
    model next to the current one. Scoring the development side is in-sample (the model was trained
    on it) and is marked so in the report.

    Raises:
        pandera.errors.SchemaErrors: If the batch breaks the feature table's schema (nothing is
            scored).
        ValueError: If ``side`` is unknown, the batch is empty, or the model file does not match
            its metadata.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    started = dt.datetime.now(dt.UTC)
    paths = output_paths(config, data_dir, side)
    for path in paths.values():
        path.unlink(missing_ok=True)
    report: dict[str, Any] = {
        "dataset": config.name,
        "started": started.isoformat(timespec="seconds"),
        "side": side,
        "in_sample": side == "dev",
        "status": "failed",
        "seconds": {},
    }
    try:
        detector = Detector.load(data_dir, config.name)
        report["model"] = {
            "joblib_sha256": detector.metadata.get("joblib_sha256"),
            "selected": detector.metadata.get("selected"),
            "thresholds": detector.thresholds,
        }

        clock = time.perf_counter()
        batch = _read_batch(config, data_dir, side)
        validate_features(batch, config, labelled=False)
        report["rows"] = batch.height
        report["seconds"]["read_and_validate"] = round(time.perf_counter() - clock, 1)

        clock = time.perf_counter()
        columns = detector.columns  # only the model's inputs can move its scores
        development = load_side(config, data_dir, "dev").select(columns).collect()
        drift = drift_report(development, batch.select(columns), columns, limit=drift_limit)
        drifted = drift.filter(~pl.col("within_limit"))["column"].to_list()
        report["drift"] = {
            "against": "development side",
            "limit": drift_limit,
            "largest_score": round(float(drift.select(pl.col("score").max()).item()), 4),
            "features_over_the_limit": drifted,
            "table": drift.to_dicts(),
        }
        if drifted:
            logger.warning("Features drift from the development side: %s", drifted)
        report["seconds"]["drift"] = round(time.perf_counter() - clock, 1)

        clock = time.perf_counter()
        scores = detector.score(batch)
        tiers = detector.tiers(scores)
        report["seconds"]["scoring"] = round(time.perf_counter() - clock, 1)
        predictions = pl.DataFrame({"event_id": batch["event_id"], "score": scores, "tier": tiers})
        _write_together(
            {"scores": predictions.select("event_id", "score"), "predictions": predictions}, paths
        )

        budgets = detector.metadata.get("budgets") or BUDGETS
        counts = dict(predictions.group_by("tier").len().iter_rows())
        report["tiers"] = {
            tier: {
                "logins": counts.get(tier, 0),
                "share": round(counts.get(tier, 0) / batch.height, 6),
                **({"budget": budgets[tier]} if tier in budgets else {}),
            }
            for tier in (*detector.thresholds, ALLOW)
        }
        flagged = 0
        for tier in detector.thresholds:  # strictest first: "and stricter" shares, as the budgets
            flagged += counts.get(tier, 0)
            report["tiers"][tier]["share_with_stricter"] = round(flagged / batch.height, 6)
        report["outputs"] = {name: path.name for name, path in paths.items()}
        report["status"] = "passed with drift warning" if drifted else "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["seconds"]["total"] = round((dt.datetime.now(dt.UTC) - started).total_seconds(), 1)
        out = report_path(config, data_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str) + "\n")
        logger.info("Wrote %s (%s)", out.name, report["status"])
    return report


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="rba", help="Name of conf/dataset/<name>.yaml")
    parser.add_argument("--side", default="test", choices=SIDES, help="Side of the split to score")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_dataset_config(args.dataset)
    data_dir = find_repo_root() / "data"
    try:
        report = run(config, data_dir, side=args.side)
    except SchemaErrors as errors:
        summary = errors.failure_cases.group_by("check", "column").len().sort("check", "column")
        logger.error("The batch breaks the feature schema; nothing was scored:\n%s", summary)  # noqa: TRY400
        return 1
    shares = ", ".join(
        f"{tier} {v['share_with_stricter']:.3%}"
        for tier, v in report["tiers"].items()
        if "share_with_stricter" in v
    )
    print(
        f"{report['rows']:,} {args.side} logins scored ({shares}); {report['status']}; "
        f"{report_path(config, data_dir).relative_to(find_repo_root())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
