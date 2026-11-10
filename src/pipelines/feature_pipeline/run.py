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

Each run writes a report (row counts, warnings, the drift table, timings) to
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
from pandera.errors import SchemaErrors

from data.datasets import DatasetConfig, find_repo_root, load_dataset_config
from data.download import download_dataset
from data.drift import DRIFT_LIMIT, drift_report
from data.feature_validation import validate_features, validate_split
from data.features import FEATURE_GROUPS, features_path, write_features
from data.intermediate import build_intermediate, intermediate_path
from data.split import load_side, split_path, write_split
from data.validation import validate_intermediate

logger = logging.getLogger(__name__)


class DriftError(RuntimeError):
    """The development and test sides differ on a feature by more than the limit."""


def report_path(config: DatasetConfig, data_dir: Path) -> Path:
    """Where the report of a feature pipeline run for ``config`` lives."""
    return data_dir / "08_reporting" / f"{config.name}_feature_pipeline.json"


def _validated(check: Callable[[], pl.DataFrame]) -> tuple[int, list[str]]:
    """Run a validation; return the rows it checked and the warning-level rules that failed."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = check().height
    return rows, [str(w.message).split(" failure case")[0] for w in caught]


def _feature_names(frame: pl.DataFrame) -> list[str]:
    return [c for group in FEATURE_GROUPS.values() for c in group if c in frame.columns]


def run(
    config: DatasetConfig,
    data_dir: Path,
    *,
    validate_only: bool = False,
    drift_limit: float = DRIFT_LIMIT,
) -> dict[str, Any]:
    """Run (or, with ``validate_only``, only check) the feature pipeline; return its report.

    Raises:
        pandera.errors.SchemaErrors: If an output breaks a validation rule.
        DriftError: If a feature drifts between the two sides by ``drift_limit`` or more.
    """
    report: dict[str, Any] = {
        "dataset": config.name,
        "started": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "validate_only": validate_only,
        "steps": [],
    }

    def step(name: str, action: Callable[[], dict[str, Any]]) -> None:
        start = time.perf_counter()
        logger.info("Step: %s", name)
        result = action()
        report["steps"].append({"step": name, "seconds": round(time.perf_counter() - start, 1)})
        report["steps"][-1].update(result)

    def raw() -> dict[str, Any]:
        path = download_dataset(config, data_dir / "01_raw")
        return {"output": path.name}

    def intermediate() -> dict[str, Any]:
        path = intermediate_path(config, data_dir)
        if not validate_only:
            raw_file = data_dir / "01_raw" / Path(config.source.member).name
            build_intermediate(config, raw_file, path)
        rows, warned = _validated(lambda: validate_intermediate(pl.read_parquet(path), config))
        return {"output": path.name, "rows": rows, "warnings": warned}

    def features() -> dict[str, Any]:
        path = (
            features_path(config, data_dir) if validate_only else write_features(config, data_dir)
        )
        rows, warned = _validated(lambda: validate_features(pl.read_parquet(path), config))
        return {"output": path.name, "rows": rows, "warnings": warned}

    def split() -> dict[str, Any]:
        path = split_path(config, data_dir) if validate_only else write_split(config, data_dir)
        checked = validate_split(
            pl.scan_parquet(features_path(config, data_dir)), pl.read_parquet(path)
        )
        sides = checked.group_by("split", "labelled").len().sort("split", "labelled")
        per_side = {
            f"{row['split']} / {'takeover' if row['labelled'] else 'no takeover'}": row["len"]
            for row in sides.iter_rows(named=True)
        }
        return {"output": path.name, "users": checked.height, "users_per_side": per_side}

    def drift() -> dict[str, Any]:
        dev = load_side(config, data_dir, "dev").collect()
        test = load_side(config, data_dir, "test").collect()
        table = drift_report(dev, test, _feature_names(dev), limit=drift_limit)
        report["drift"] = table.to_dicts()
        over = table.filter(~pl.col("within_limit"))
        if over.height:
            raise DriftError(f"features drift by {drift_limit} or more: {over['column'].to_list()}")
        return {
            "features": table.height,
            "largest_score": round(float(table.select(pl.col("score").max()).item()), 4),
        }

    steps = [("raw data", raw)] if not validate_only else []
    steps += [
        ("intermediate dataset", intermediate),
        ("feature table", features),
        ("split", split),
        ("drift between development and test", drift),
    ]
    report["status"] = "failed"
    try:
        for name, action in steps:
            step(name, action)
        report["status"] = "passed"
    finally:  # the report of a failed run says where it stopped
        out = report_path(config, data_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        logger.info("Wrote %s (%s)", out.name, report["status"])
    return report


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
        failure = f"Validation failed:\n{errors.failure_cases.group_by('check', 'column').len()}"
    except DriftError as error:
        failure = str(error)
    else:
        print(report_path(config, data_dir).relative_to(find_repo_root()))
        return 0
    logger.error("%s", failure)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
