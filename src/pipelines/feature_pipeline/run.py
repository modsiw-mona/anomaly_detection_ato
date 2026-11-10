"""Feature pipeline: raw download -> intermediate dataset -> feature table -> development / test split.

The first of the three FTI pipelines (feature, training, inference). It runs the data steps in order
and **validates each output before the next step uses it**:

1. **Raw data**: download the archive, verify its checksum and extract the data file
   (:mod:`data.download`; a verified copy is not downloaded again).
2. **Intermediate dataset**: standard names, null markers and types (:mod:`data.intermediate`),
   validated with its Pandera schema (:mod:`data.validation`).
3. **Feature table**: past-only behavioural features (:mod:`data.features`), validated with
   :func:`data.feature_validation.validate_features`.
4. **Split**: users to the development or the test side (:mod:`data.split`), validated with
   :func:`data.feature_validation.validate_split`.
5. **Drift report**: the development and the test side compared feature by feature
   (:mod:`data.drift`); a feature over the limit stops the pipeline.

A rebuilt output replaces the previous one only after it passes its validation. Each run writes a
report (row counts, warnings, the drift table, timings and, on failure, the step and the error) to
``data/08_reporting/<dataset>_feature_pipeline.json``. With ``--validate-only`` the pipeline checks
the existing outputs of steps 2 to 5 without downloading or rebuilding anything.

Usage::

    make feature_pipeline                        # DATASET=rba by default
    make feature_pipeline VALIDATE_ONLY=1
"""

import argparse
import datetime as dt
import json
import logging
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
from pandera.errors import SchemaErrors, SchemaWarning

from data.datasets import DatasetConfig, find_repo_root, load_dataset_config
from data.download import download_dataset
from data.drift import DRIFT_LIMIT, drift_report
from data.feature_validation import validate_features, validate_split
from data.features import FEATURE_GROUPS, features_path, write_features
from data.intermediate import Check, build_intermediate, intermediate_path
from data.split import load_side, split_path, write_split
from data.validation import validate_intermediate

logger = logging.getLogger(__name__)


class DriftError(RuntimeError):
    """The development and test sides differ on a feature by more than the limit."""


def report_path(config: DatasetConfig, data_dir: Path) -> Path:
    """Where the report of a feature pipeline run for ``config`` lives."""
    return data_dir / "08_reporting" / f"{config.name}_feature_pipeline.json"


def _validated(check: Callable[[], pl.DataFrame]) -> dict[str, Any]:
    """Run a validation; return the rows it checked and the warning-level rules that failed."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SchemaWarning)
        rows = check().height
    warned = [w for w in caught if issubclass(w.category, SchemaWarning)]
    return {"rows": rows, "warnings": [str(w.message).split(" failure case")[0] for w in warned]}


def _feature_names(columns: list[str]) -> list[str]:
    return [c for group in FEATURE_GROUPS.values() for c in group if c in columns]


Step = Callable[[], dict[str, Any]]


def _build_or_check(
    path: Path,
    validate_only: bool,
    build: Callable[[Check], object],
    validate: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Validate the existing ``path``, or build it with the validation as the publish check."""
    result: dict[str, Any] = {}

    def check(file: Path) -> None:
        result.update(validate(file))

    if validate_only:
        check(path)
    else:
        build(check)
    return {"output": path.name, **result}


def _intermediate_step(config: DatasetConfig, data_dir: Path, validate_only: bool) -> Step:
    path = intermediate_path(config, data_dir)
    raw_file = data_dir / "01_raw" / Path(config.source.member).name
    return lambda: _build_or_check(
        path,
        validate_only,
        lambda check: build_intermediate(config, raw_file, path, check=check),
        lambda file: _validated(lambda: validate_intermediate(pl.read_parquet(file), config)),
    )


def _features_step(config: DatasetConfig, data_dir: Path, validate_only: bool) -> Step:
    return lambda: _build_or_check(
        features_path(config, data_dir),
        validate_only,
        lambda check: write_features(config, data_dir, check=check),
        lambda file: _validated(lambda: validate_features(pl.read_parquet(file), config)),
    )


def _split_summary(checked: pl.DataFrame) -> dict[str, Any]:
    sides = checked.group_by("split", "labelled").len().sort("split", "labelled")
    per_side = {
        f"{row['split']} / {'takeover' if row['labelled'] else 'no takeover'}": row["len"]
        for row in sides.iter_rows(named=True)
    }
    return {"users": checked.height, "users_per_side": per_side}


def _split_step(config: DatasetConfig, data_dir: Path, validate_only: bool) -> Step:
    def validate(file: Path) -> dict[str, Any]:
        features = pl.scan_parquet(features_path(config, data_dir))
        return _split_summary(validate_split(features, pl.read_parquet(file)))

    return lambda: _build_or_check(
        split_path(config, data_dir),
        validate_only,
        lambda check: write_split(config, data_dir, check=check),
        validate,
    )


def _drift_step(
    config: DatasetConfig, data_dir: Path, limit: float, report: dict[str, Any]
) -> Step:
    def drift() -> dict[str, Any]:
        dev, test = (load_side(config, data_dir, side) for side in ("dev", "test"))
        columns = _feature_names(dev.collect_schema().names())
        table = drift_report(
            dev.select(columns).collect(), test.select(columns).collect(), columns, limit=limit
        )
        report["drift"] = table.to_dicts()  # kept on failure, to see which features moved
        over = table.filter(~pl.col("within_limit"))
        if over.height:
            raise DriftError(f"features drift by {limit} or more: {over['column'].to_list()}")
        return {
            "features": table.height,
            "largest_score": round(float(table.select(pl.col("score").max()).item()), 4),
        }

    return drift


def _run_steps(report: dict[str, Any], steps: list[tuple[str, Step]]) -> None:
    """Run each step in order, timing it; on failure record the step and the error, then raise."""
    for name, action in steps:
        start = time.perf_counter()
        logger.info("Step: %s", name)
        try:
            result = action()
        except Exception as error:
            report["failed_step"] = name
            report["error"] = (
                _summary(error)
                if isinstance(error, SchemaErrors)
                else f"{type(error).__name__}: {error}"
            )
            raise
        report["steps"].append(
            {"step": name, "seconds": round(time.perf_counter() - start, 1), **result}
        )


def run(
    config: DatasetConfig,
    data_dir: Path,
    *,
    validate_only: bool = False,
    drift_limit: float = DRIFT_LIMIT,
) -> dict[str, Any]:
    """Run (or, with ``validate_only``, only check) the feature pipeline; return its report.

    A rebuilt output replaces the previous one only after it passes its validation, so a failed run
    leaves the earlier outputs in place.

    Raises:
        pandera.errors.SchemaErrors: If an output breaks a validation rule.
        DriftError: If a feature drifts between the two sides by ``drift_limit`` or more.
    """
    report: dict[str, Any] = {
        "dataset": config.name,
        "started": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "validate_only": validate_only,
        "status": "failed",
        "steps": [],
    }
    steps: list[tuple[str, Step]] = []
    if not validate_only:
        steps.append(
            ("raw data", lambda: {"output": download_dataset(config, data_dir / "01_raw").name})
        )
    steps += [
        ("intermediate dataset", _intermediate_step(config, data_dir, validate_only)),
        ("feature table", _features_step(config, data_dir, validate_only)),
        ("split", _split_step(config, data_dir, validate_only)),
        ("drift between development and test", _drift_step(config, data_dir, drift_limit, report)),
    ]
    try:
        _run_steps(report, steps)
        report["status"] = "passed"
    finally:  # the report of a failed run says where it stopped
        out = report_path(config, data_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        logger.info("Wrote %s (%s)", out.name, report["status"])
    return report


def _summary(errors: SchemaErrors) -> str:
    """Failing rows per check and column, as a short table."""
    return str(errors.failure_cases.group_by("check", "column").len().sort("check", "column"))


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="rba", help="Name of conf/dataset/<name>.yaml")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check the existing outputs without downloading or rebuilding them",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_dataset_config(args.dataset)
    data_dir = find_repo_root() / "data"
    try:
        run(config, data_dir, validate_only=args.validate_only)
    except SchemaErrors as errors:
        failure = f"Validation failed (failing rows per check):\n{_summary(errors)}"
    except DriftError as error:
        failure = str(error)
    else:
        print(report_path(config, data_dir).relative_to(find_repo_root()))
        return 0
    logger.error("%s", failure)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
