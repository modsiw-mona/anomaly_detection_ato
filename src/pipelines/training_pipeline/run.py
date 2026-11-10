"""Training pipeline: feature table + split -> validated model, out-of-fold scores and alert thresholds.

The second of the three FTI pipelines. It trains **only the selected configuration**
(``conf/training/<dataset>.yaml``); comparing model families is experimentation, and stays in the
notebooks (5-models). Everything uses the **development side**; the test side is never read.

1. **Out-of-fold scores**: every development login scored by a model trained on the other folds,
   with the same user-grouped folds as the notebooks (``model.evaluation.assign_folds``). Repeat 0
   sets the thresholds; more repeats only refine the estimate.
2. **Baseline** on the same folds (the logistic regression of issue #5).
3. **Final model**: refitted on the whole development side.
4. **Alert thresholds**: the score above which each tier's budget of development logins falls, from
   the out-of-fold scores of repeat 0 (``inference.predict.alert_thresholds``).
5. **Model validation** (``model.validation``): blocking gates stop the run **before anything is
   saved**, so the previous model stays.
6. **Save**: the pipeline (``data/06_models/<dataset>_model.joblib``) with its metadata (columns,
   thresholds, metrics, SHA-256) and the out-of-fold scores (``data/07_model_output``), the files
   the inference module and the demo read.

Every run, passed or failed, writes a **run record** to ``data/08_reporting/training_runs/``
(parameters, metrics, gates, SHA-256 of the data and of the model, git commit, library versions,
and whether the model is identical to the previous one); the latest is also copied to
``data/08_reporting/<dataset>_training_pipeline.json``.

Usage::

    make training_pipeline                  # DATASET=rba by default
    make training_pipeline REPEATS=3        # the 15-fold estimate of issue #6 (three times slower)
"""

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from data.datasets import DatasetConfig, find_repo_root, load_dataset_config
from data.features import LABEL, feature_columns, features_path
from data.intermediate import publish
from data.split import load_side, split_path
from inference.predict import alert_thresholds
from model import validation
from model.evaluation import assign_folds, bootstrap_pr_auc, evaluate
from model.selection import (
    BoolArray,
    Fitter,
    FloatArray,
    IntArray,
    fit_supervised,
    fold_per_login,
    group_metrics,
    out_of_fold,
    score_with,
    train_supervised,
)
from pipelines.training_pipeline.config import TrainingConfig, load_training_config

logger = logging.getLogger(__name__)

BOOTSTRAP_REPLICATES = 1_000  # as in notebook 06
LIBRARIES = ("scikit-learn", "numpy", "polars", "pandas")


def model_paths(config: DatasetConfig, data_dir: Path) -> dict[str, Path]:
    """The files a training run writes, where the inference module and the demo read them."""
    return {
        "model": data_dir / "06_models" / f"{config.name}_model.joblib",
        "metadata": data_dir / "06_models" / f"{config.name}_model.json",
        "out_of_fold": data_dir / "07_model_output" / f"{config.name}_dev_oof.parquet",
    }


def report_path(config: DatasetConfig, data_dir: Path) -> Path:
    """The record of the latest training run."""
    return data_dir / "08_reporting" / f"{config.name}_training_pipeline.json"


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version() -> dict[str, Any]:
    """Git commit of the code and whether it had uncommitted changes (``None`` without git)."""
    git = shutil.which("git")
    if git is None:
        return {"commit": None, "uncommitted_changes": None}
    root = find_repo_root()

    def output(*args: str) -> str:
        result = subprocess.run([git, *args], cwd=root, capture_output=True, text=True, check=False)  # noqa: S603
        return result.stdout.strip() if result.returncode == 0 else ""

    commit = output("rev-parse", "HEAD") or None
    return {"commit": commit, "uncommitted_changes": bool(output("status", "--porcelain"))}


def _fitter(name: str, training: TrainingConfig, *, balance_campaign: bool) -> Fitter:
    return lambda train, columns: fit_supervised(
        name,
        train,
        columns,
        balance_campaign=balance_campaign,
        negative_share=training.negative_share,
        seed=training.seed,
    )


def _repeat_row(
    y: BoolArray, fold: IntArray, score: FloatArray, campaign: BoolArray, name: str
) -> dict[str, float]:
    """One repeat of cross-validation, measured as notebook 06 measures it."""
    per_fold = [evaluate(y[fold == k], score[fold == k]) for k in np.unique(fold)]
    groups = group_metrics(y, score, {name: campaign, "other": ~campaign})
    low, high = bootstrap_pr_auc(y, score, n_boot=BOOTSTRAP_REPLICATES)
    pr_aucs = [m["pr_auc"] for m in per_fold]
    return {
        "pr_auc": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs, ddof=1)),
        "recall_0_1pct": float(np.mean([m["recall_at_0_1pct"] for m in per_fold])),
        "recall_1pct": float(np.mean([m["recall_at_1pct"] for m in per_fold])),
        "precision_1pct": float(np.mean([m["precision_at_1pct"] for m in per_fold])),
        f"pr_auc_{name}": groups[name]["pr_auc"],
        "pr_auc_other": groups["other"]["pr_auc"],
        f"recall_1pct_{name}": groups[name]["recall_at_1pct"],
        "recall_1pct_other": groups["other"]["recall_at_1pct"],
        "pooled_low": low,
        "pooled_high": high,
    }


def cross_validation_summary(rows: list[dict[str, float]]) -> dict[str, float | None]:
    """Average of the repeats, with the names and rounding of notebook 06 (``cv_15_folds``)."""
    frame = pl.DataFrame(rows)
    summary: dict[str, Any] = {
        "pr_auc (15 folds)": frame["pr_auc"].mean(),
        "std within repeat": frame["pr_auc_std"].mean(),
        "std across repeats": frame["pr_auc"].std() if frame.height > 1 else None,
        **{c: frame[c].mean() for c in frame.columns if c not in {"pr_auc", "pr_auc_std"}},
    }
    return {k: None if v is None else round(float(v), 4) for k, v in summary.items()}


def _write_record(record: dict[str, Any], config: DatasetConfig, data_dir: Path) -> None:
    runs = data_dir / "08_reporting" / "training_runs"
    runs.mkdir(parents=True, exist_ok=True)
    stamp = record["started"].replace(":", "").replace("-", "").replace("+0000", "Z")
    text = json.dumps(record, indent=2, default=str) + "\n"
    (runs / f"{config.name}_{stamp}.json").write_text(text)
    report_path(config, data_dir).write_text(text)


def run(  # noqa: PLR0915 — one training run, step by step
    config: DatasetConfig,
    training: TrainingConfig,
    data_dir: Path,
) -> dict[str, Any]:
    """Train, validate and save the selected model; return the run record.

    Raises:
        model.validation.ModelValidationError: If a blocking gate fails (nothing is saved).
    """
    started = dt.datetime.now(dt.UTC)
    record: dict[str, Any] = {
        "dataset": config.name,
        "started": started.isoformat(timespec="seconds"),
        "status": "failed",
        "configuration": training.model_dump(mode="json"),
        "code": code_version(),
        "libraries": {
            "python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in LIBRARIES},
        },
        "data": {
            "features_sha256": sha256_of(features_path(config, data_dir)),
            "split_sha256": sha256_of(split_path(config, data_dir)),
        },
        "seconds": {},
    }
    paths = model_paths(config, data_dir)
    previous = (
        json.loads(paths["metadata"].read_text()).get("joblib_sha256")
        if paths["metadata"].is_file()
        else None
    )
    try:
        clock = time.perf_counter()
        dev = load_side(config, data_dir, "dev").collect()
        y = dev[LABEL].to_numpy()
        record["data"] |= {"development_logins": dev.height, "development_takeovers": int(y.sum())}
        user_folds = assign_folds(
            dev,
            n_folds=training.folds.n_folds,
            n_repeats=training.folds.repeats,
            seed=training.folds.seed,
        )
        folds = {
            r: fold_per_login(dev, user_folds, f"fold_{r}") for r in range(training.folds.repeats)
        }
        columns = feature_columns(training.feature_groups)
        model_fit = _fitter(training.model, training, balance_campaign=training.balance_campaign)
        oof = {r: out_of_fold(dev, folds[r], model_fit, columns) for r in folds}
        record["seconds"]["out_of_fold"] = round(time.perf_counter() - clock)

        clock = time.perf_counter()
        baseline_columns = feature_columns(training.baseline.feature_groups)
        baseline_fit = _fitter(training.baseline.model, training, balance_campaign=False)
        baseline_oof = out_of_fold(dev, folds[0], baseline_fit, baseline_columns)
        record["seconds"]["baseline"] = round(time.perf_counter() - clock)

        clock = time.perf_counter()
        pipeline = train_supervised(
            training.model,
            dev,
            columns,
            balance_campaign=training.balance_campaign,
            negative_share=training.negative_share,
            seed=training.seed,
        )
        record["seconds"]["final_fit"] = round(time.perf_counter() - clock)

        clock = time.perf_counter()
        in_sample = score_with(pipeline, columns)(dev)
        seconds_per_login = (time.perf_counter() - clock) / dev.height
        thresholds = alert_thresholds(oof[0], training.budgets)

        campaign = (
            dev[training.segments.column].cast(pl.String) == training.segments.value
        ).to_numpy()
        name = training.segments.name.lower()
        rows = [_repeat_row(y, folds[r], oof[r], campaign, name) for r in folds]
        metrics = cross_validation_summary(rows)
        baseline_row = _repeat_row(y, folds[0], baseline_oof, campaign, name)
        first = dev["first_success"].to_numpy()
        candidates = {
            training.segments.name: campaign,
            f"not {training.segments.name}": ~campaign,
            "first login": first,
            "returning account": ~first,
        }
        segments = group_metrics(  # a segment without takeovers has nothing to measure
            y, oof[0], {name: m for name, m in candidates.items() if (y & m).any()}
        )
        gates = [
            *validation.selection_rule(
                float(metrics["pooled_low"] or 0),
                float(metrics["recall_1pct"] or 0),
                min_low=training.gates.min_interval_low,
                min_recall=training.gates.min_recall_at_1pct,
            ),
            validation.gain_over_baseline(
                rows[0]["pr_auc"],
                baseline_row["pr_auc"],
                min_gain=training.gates.min_gain_over_baseline,
            ),
            *validation.alert_rates(
                in_sample, thresholds, training.budgets, ratio=training.gates.alert_rate_ratio
            ),
            *validation.segment_recall(
                segments, min_recall=training.gates.min_segment_recall_at_1pct
            ),
            validation.scoring_time(
                seconds_per_login, max_seconds=training.gates.max_seconds_per_login
            ),
            validation.weakest_segment(segments),
            validation.overfitting_gap(y, in_sample, oof[0]),
            validation.calibration(y, oof[0]),
        ]
        record |= {
            "cross_validation": {"folds": len(rows) * training.folds.n_folds, **metrics},
            "repeats": rows,
            "baseline_repeat_0": baseline_row,
            "segments": {name: dict(m) for name, m in segments.items()},
            "thresholds": thresholds,
            "gates": gates,
        }
        validation.check(gates)

        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        partial = paths["model"].with_name(paths["model"].name + ".part")
        joblib.dump(pipeline, partial)
        model_sha = sha256_of(partial)
        publish(partial, paths["model"])
        partial = paths["out_of_fold"].with_name(paths["out_of_fold"].name + ".part")
        pl.DataFrame({"event_id": dev["event_id"], "score": oof[0]}).write_parquet(partial)
        publish(partial, paths["out_of_fold"])
        fifteen = len(rows) * training.folds.n_folds == 15  # noqa: PLR2004 — notebook 06's estimate
        metadata = {  # the keys of notebook 06 first, so notebooks 07 and 08 still read it
            "dataset": config.name,
            "selected": training.name,
            "model": training.model,
            "campaign_balanced": training.balance_campaign,
            "columns": columns,
            "negative_share": training.negative_share,
            "trained_on": "development side (80% of users)",
            "joblib_sha256": model_sha,
            "joblib_megabytes": round(paths["model"].stat().st_size / 1e6, 2),
            **({"cv_15_folds": metrics} if fifteen else {}),
            "cross_validation": record["cross_validation"],
            "feature_groups": list(training.feature_groups),
            "budgets": training.budgets,
            "thresholds": thresholds,
            "training_run": record["started"],
        }
        paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
        record["model"] = {
            "joblib_sha256": model_sha,
            "same_as_previous": model_sha == previous if previous else None,
            "previous_joblib_sha256": previous,
        }
        record["status"] = "passed"
    finally:
        record["seconds"]["total"] = round((dt.datetime.now(dt.UTC) - started).total_seconds())
        _write_record(record, config, data_dir)
        logger.info("Run record written (%s)", record["status"])
    return record


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="rba", help="Name of conf/dataset/<name>.yaml")
    parser.add_argument(
        "--repeats", type=int, help="Cross-validation repeats (overrides the configuration)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_dataset_config(args.dataset)
    training = load_training_config(args.dataset)
    if args.repeats:
        folds = training.folds.model_copy(update={"repeats": args.repeats})
        training = training.model_copy(update={"folds": folds})
    data_dir = find_repo_root() / "data"
    try:
        record = run(config, training, data_dir)
    except validation.ModelValidationError as error:
        logger.error("%s", error)  # noqa: TRY400 — the gates are the message, not a traceback
        return 1
    cv = record["cross_validation"]
    print(
        f"PR-AUC {cv['pr_auc (15 folds)']:.4f} over {cv['folds']} folds; model saved "
        f"(same as before: {record['model']['same_as_previous']}); "
        f"{report_path(config, data_dir).relative_to(find_repo_root())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
