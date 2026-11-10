"""Training pipeline: feature table + split -> validated model, out-of-fold scores and alert thresholds.

The second of the three FTI pipelines. It trains **only the selected configuration**
(``conf/training/<dataset>.yaml``); comparing model families is experimentation, and stays in the
notebooks (5-models). Everything uses the **development side**; the test side is never read.

1. **Out-of-fold scores**: every development login scored by a model trained on the other folds,
   with the same user-grouped folds as the notebooks (``model.evaluation.assign_folds``; 5 folds x 3
   repeats by default, the 15-fold estimate of issue #6). Repeat 0 sets the thresholds; the other
   repeats only refine the estimate.
2. **Baseline** on the same folds (the logistic regression of issue #5).
3. **Final model**: refitted on the whole development side.
4. **Alert thresholds**: the score above which each tier's budget of development logins falls, from
   the out-of-fold scores of repeat 0 (``inference.predict.alert_thresholds``).
5. **Model validation** (``model.validation``): blocking gates stop the run **before anything is
   saved**, so the previous model stays.
6. **Save**: the pipeline (``data/06_models/<dataset>_model.joblib``) with its metadata (columns,
   thresholds, metrics, SHA-256) and the out-of-fold scores (``data/07_model_output``), the files
   the inference module and the demo read. All three are written first and renamed only then, the
   metadata last, so a failure leaves the previous model whole.

Every run, passed or failed, writes a **run record** to ``data/08_reporting/training_runs/``
(parameters, metrics, gates, SHA-256 of the data and of the model, git commit, library versions,
and whether the model is identical to the previous one); the latest is also copied to
``data/08_reporting/<dataset>_training_pipeline.json``.

Usage::

    make training_pipeline                  # DATASET=rba by default
    make training_pipeline REPEATS=1        # a quicker 5-fold run (notebooks 07 and 08 need the default 15)
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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline

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
LATENCY_SAMPLE = 100  # single-login scoring calls timed for the latency gate
LIBRARIES = ("scikit-learn", "numpy", "scipy", "joblib", "polars", "pandas")
NOTEBOOK_FOLDS = (
    15  # notebooks 07 and 08 read the 15-fold estimate of notebook 06 (``cv_15_folds``)
)


class ConfigurationError(ValueError):
    """The training configuration does not fit the data."""


def model_paths(config: DatasetConfig, data_dir: Path) -> dict[str, Path]:
    """The files a training run writes, where the inference module and the demo read them.

    Args:
        config: Dataset configuration; its ``name`` prefixes the file names.
        data_dir: The repository's ``data`` folder.

    Returns:
        In publishing order: ``model`` (the fitted pipeline, ``06_models/<dataset>_model.joblib``),
        ``out_of_fold`` (the out-of-fold development scores,
        ``07_model_output/<dataset>_dev_oof.parquet``) and ``metadata``
        (``06_models/<dataset>_model.json``, written last).
    """
    return {
        "model": data_dir / "06_models" / f"{config.name}_model.joblib",
        "out_of_fold": data_dir / "07_model_output" / f"{config.name}_dev_oof.parquet",
        "metadata": data_dir
        / "06_models"
        / f"{config.name}_model.json",  # last: it vouches for the rest
    }


def report_path(config: DatasetConfig, data_dir: Path) -> Path:
    """The record of the latest training run.

    Args:
        config: Dataset configuration; its ``name`` names the record file.
        data_dir: The repository's ``data`` folder.

    Returns:
        ``<data_dir>/08_reporting/<dataset>_training_pipeline.json``; every run also keeps its own
        copy in ``08_reporting/training_runs/``.
    """
    return data_dir / "08_reporting" / f"{config.name}_training_pipeline.json"


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, read in chunks.

    Args:
        path: File to hash; it is read 1 MiB at a time, so a large file is never loaded whole.

    Returns:
        The digest as a hexadecimal string.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version() -> dict[str, Any]:
    """Git commit of the code and whether tracked files had uncommitted changes.

    Returns:
        ``commit`` (the ``HEAD`` commit hash) and ``uncommitted_changes`` (True if a tracked file
        differs from that commit; untracked files are ignored). Both are None when git is not
        installed; each is None when its git command fails or takes more than 10 seconds.
    """
    git = shutil.which("git")
    if git is None:
        return {"commit": None, "uncommitted_changes": None}
    root = find_repo_root()

    def output(*args: str) -> str | None:
        try:
            result = subprocess.run(  # noqa: S603 — fixed arguments, resolved executable
                [git, *args], cwd=root, capture_output=True, text=True, check=False, timeout=10
            )
        except subprocess.TimeoutExpired:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = output("rev-parse", "HEAD")
    status = output("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "uncommitted_changes": None if status is None else bool(status)}


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


def cross_validation_summary(rows: list[dict[str, float]], folds: int) -> dict[str, float | None]:
    """Average of the repeats, with the names and rounding of notebook 06 (``cv_15_folds``).

    Args:
        rows: One dict of metrics per cross-validation repeat, with at least ``pr_auc`` (mean over
            the repeat's folds) and ``pr_auc_std`` (standard deviation over those folds).
        folds: Total number of folds over all repeats; it only names the PR-AUC key.

    Returns:
        ``pr_auc (<folds> folds)`` (mean PR-AUC of the repeats), ``std within repeat`` (mean of the
        per-repeat standard deviations), ``std across repeats`` (None with a single repeat) and
        the mean of every other metric, each rounded to 4 decimals.
    """
    frame = pl.DataFrame(rows)
    summary: dict[str, Any] = {
        f"pr_auc ({folds} folds)": frame["pr_auc"].mean(),
        "std within repeat": frame["pr_auc_std"].mean(),
        "std across repeats": frame["pr_auc"].std() if frame.height > 1 else None,
        **{c: frame[c].mean() for c in frame.columns if c not in {"pr_auc", "pr_auc_std"}},
    }
    return {k: None if v is None else round(float(v), 4) for k, v in summary.items()}


def segment_masks(dev: pl.DataFrame, training: TrainingConfig) -> dict[str, BoolArray]:
    """Logins of each group of takeovers: the dominant campaign and the rest; first and returning.

    Checked before any training, so a configuration that does not fit the data fails at once.

    Args:
        dev: Development logins, with the column named in ``training.segments`` and
            ``first_success``.
        training: Training configuration; ``segments`` gives the campaign's column, value and name.

    Returns:
        One boolean array over the logins of ``dev`` per group: ``<segments.name>``,
        ``not <segments.name>``, ``first login`` and ``returning account``. A null campaign value
        counts as outside the campaign, and a null ``first_success`` as a returning account.

    Raises:
        ConfigurationError: If ``dev`` lacks a column the segments need.
    """
    needed = {training.segments.column, "first_success"}
    missing = needed - set(dev.columns)
    if missing:
        raise ConfigurationError(f"columns needed for the segments are missing: {sorted(missing)}")
    campaign = (
        (dev[training.segments.column].cast(pl.String) == training.segments.value)
        .fill_null(value=False)
        .to_numpy()
    )
    first = dev["first_success"].fill_null(value=False).to_numpy()
    return {
        training.segments.name: campaign,
        f"not {training.segments.name}": ~campaign,
        "first login": first,
        "returning account": ~first,
    }


def single_login_seconds(scorer: Callable[[pl.DataFrame], FloatArray], dev: pl.DataFrame) -> float:
    """Median time to score one login on its own, as a login service or the demo does.

    Args:
        scorer: Function that scores a frame of logins, one score per row.
        dev: Logins to time; up to ``LATENCY_SAMPLE`` rows, evenly spaced over the frame, are
            scored one call each.

    Returns:
        The median wall-clock time of one call, in seconds.
    """
    rows = np.linspace(0, dev.height - 1, num=min(LATENCY_SAMPLE, dev.height), dtype=np.int64)
    times = []
    for row in rows:
        start = time.perf_counter()
        scorer(dev.slice(int(row), 1))
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def _save(
    paths: dict[str, Path], pipeline: Pipeline, oof: pl.DataFrame, metadata: dict[str, Any]
) -> str:
    """Write the three files next to their targets, then rename them, the metadata last.

    Nothing replaces a published file until all three are written; a failure on the way removes the
    partial files and leaves the previous model, scores and metadata as they were.
    """
    partial = {name: path.with_name(path.name + ".part") for name, path in paths.items()}
    try:
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, partial["model"])
        model_sha = sha256_of(partial["model"])
        oof.write_parquet(partial["out_of_fold"])
        metadata |= {
            "joblib_sha256": model_sha,
            "joblib_megabytes": round(partial["model"].stat().st_size / 1e6, 2),
        }
        partial["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    except BaseException:
        for path in partial.values():
            path.unlink(missing_ok=True)
        raise
    for name in ("model", "out_of_fold", "metadata"):
        publish(partial[name], paths[name])
    return model_sha


def _write_record(record: dict[str, Any], config: DatasetConfig, data_dir: Path) -> None:
    runs = data_dir / "08_reporting" / "training_runs"
    runs.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=2, default=str) + "\n"
    (runs / f"{config.name}_{record['run_id']}.json").write_text(text)
    report_path(config, data_dir).write_text(text)


def run(  # noqa: PLR0915 — one training run, step by step
    config: DatasetConfig,
    training: TrainingConfig,
    data_dir: Path,
) -> dict[str, Any]:
    """Train, validate and save the selected model; return the run record.

    Only the development side is read. Every run, passed or failed, writes its record to
    ``08_reporting/training_runs/`` and to :func:`report_path`; an error of the run is recorded in
    it and raised again, while an error writing the record is only logged.

    Args:
        config: Dataset configuration.
        training: The configuration to train: model, feature groups, sampling, seeds, folds,
            budgets, baseline, segments and gate limits.
        data_dir: The repository's ``data`` folder; the feature table and the split are read from
            it and the files of :func:`model_paths` written to it.

    Returns:
        The run record: ``dataset``, ``run_id``, ``started``, ``status`` (``passed``),
        ``configuration``, ``code`` (:func:`code_version`), ``libraries``, ``data`` (SHA-256 of
        the inputs, development logins and takeovers), ``seconds`` per stage, ``cross_validation``,
        ``repeats``, ``baseline_repeat_0``, ``segments``, ``thresholds``, ``gates`` and ``model``
        (SHA-256 of the saved file and whether it is the same as the previous one).

    Raises:
        ConfigurationError: If the configuration does not fit the data (checked before training).
        model.validation.ModelValidationError: If a blocking gate fails (nothing is saved).
    """
    started = dt.datetime.now(dt.UTC)
    paths = model_paths(config, data_dir)
    previous = sha256_of(paths["model"]) if paths["model"].is_file() else None
    record: dict[str, Any] = {
        "dataset": config.name,
        "run_id": started.strftime("%Y%m%dT%H%M%S%fZ"),
        "started": started.isoformat(timespec="seconds"),
        "status": "failed",
        "configuration": training.model_dump(mode="json"),
        "code": code_version(),
        "libraries": {
            "python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in LIBRARIES},
        },
        "data": {},
        "seconds": {},
    }
    try:
        record["data"] = {
            "features_sha256": sha256_of(features_path(config, data_dir)),
            "split_sha256": sha256_of(split_path(config, data_dir)),
        }
        clock = time.perf_counter()
        dev = load_side(config, data_dir, "dev").collect()
        y = dev[LABEL].to_numpy()
        record["data"] |= {"development_logins": dev.height, "development_takeovers": int(y.sum())}
        masks = segment_masks(dev, training)
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

        scorer = score_with(pipeline, columns)
        clock = time.perf_counter()
        in_sample = scorer(dev)
        batch_seconds = (time.perf_counter() - clock) / dev.height
        single_seconds = single_login_seconds(scorer, dev)
        thresholds = alert_thresholds(oof[0], training.budgets)

        name = training.segments.name.lower()
        campaign = masks[training.segments.name]
        n_folds = len(oof) * training.folds.n_folds
        rows = [_repeat_row(y, folds[r], oof[r], campaign, name) for r in folds]
        metrics = cross_validation_summary(rows, n_folds)
        baseline_row = _repeat_row(y, folds[0], baseline_oof, campaign, name)
        segments = group_metrics(  # a segment without takeovers has nothing to measure
            y, oof[0], {key: m for key, m in masks.items() if (y & m).any()}
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
            validation.batch_scoring_time(
                batch_seconds, max_seconds=training.gates.max_seconds_per_login_in_batch
            ),
            validation.single_login_latency(
                single_seconds, max_seconds=training.gates.max_seconds_single_login
            ),
            validation.weakest_segment(segments),
            validation.overfitting_gap(y, in_sample, oof[0]),
            validation.calibration(y, oof[0]),
        ]
        record |= {
            "cross_validation": {"folds": n_folds, **metrics},
            "repeats": rows,
            "baseline_repeat_0": baseline_row,
            "segments": {key: dict(m) for key, m in segments.items()},
            "thresholds": thresholds,
            "gates": gates,
        }
        validation.check(gates)

        metadata = {  # the keys of notebook 06 first, so notebooks 07 and 08 still read it
            "dataset": config.name,
            "selected": training.name,
            "model": training.model,
            "campaign_balanced": training.balance_campaign,
            "columns": columns,
            "negative_share": training.negative_share,
            "trained_on": "development side (80% of users)",
            **({"cv_15_folds": metrics} if n_folds == NOTEBOOK_FOLDS else {}),
            "cross_validation": record["cross_validation"],
            "feature_groups": list(training.feature_groups),
            "budgets": training.budgets,
            "thresholds": thresholds,
            "training_run": record["run_id"],
        }
        frame = pl.DataFrame({"event_id": dev["event_id"], "score": oof[0]})
        model_sha = _save(paths, pipeline, frame, metadata)
        record["model"] = {
            "joblib_sha256": model_sha,
            "same_as_previous": model_sha == previous if previous else None,
            "previous_joblib_sha256": previous,
        }
        record["status"] = "passed"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["seconds"]["total"] = round((dt.datetime.now(dt.UTC) - started).total_seconds())
        try:
            _write_record(record, config, data_dir)
            logger.info("Run record written (%s)", record["status"])
        except OSError:  # never hide the run's own error behind the record's
            logger.exception("Could not write the run record")
    return record


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Reads ``--dataset`` (default ``rba``) and ``--repeats`` (cross-validation repeats, overriding
    the configuration), runs :func:`run` on the repository's ``data`` folder and, on success,
    prints the PR-AUC, whether the model changed and the path of the run record.

    Args:
        argv: Command-line arguments; None reads them from sys.argv.

    Returns:
        0 if the model passed its gates and was saved; 1 if a blocking gate failed or the
        configuration does not fit the data (the reason is logged as an error).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="rba", help="Name of conf/dataset/<name>.yaml")
    parser.add_argument(
        "--repeats",
        type=_positive,
        help="Cross-validation repeats, overriding the configuration (notebooks 07 and 08 need the "
        "default 15-fold estimate)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_dataset_config(args.dataset)
    training = load_training_config(args.dataset)
    if args.repeats:
        raw = training.model_dump()
        raw["folds"]["repeats"] = args.repeats
        training = TrainingConfig.model_validate(raw)
    data_dir = find_repo_root() / "data"
    try:
        record = run(config, training, data_dir)
    except (validation.ModelValidationError, ConfigurationError) as error:
        logger.error("%s", error)  # noqa: TRY400 — the gates are the message, not a traceback
        return 1
    cv = record["cross_validation"]
    print(
        f"PR-AUC {cv[f'pr_auc ({cv["folds"]} folds)']:.4f} over {cv['folds']} folds; model saved "
        f"(same as before: {record['model']['same_as_previous']}); "
        f"{report_path(config, data_dir).relative_to(find_repo_root())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
