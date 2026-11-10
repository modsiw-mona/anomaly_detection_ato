"""Development / test split, by a rule fixed before looking at any result.

The EDA (issue #3) applied the pre-registered rule: the last 20% of the timeline holds no takeover,
so the final test set is a **random 20% of users**:

* **grouped by user** — all logins of a user fall on the same side, so a model cannot memorise a
  user's habits in training and meet them again in the test;
* **stratified by label at user level** — users with at least one takeover and users without are
  split separately, so the test set gets its fair share of the 138 takeover users.

The assignment is seeded and depends only on the set of users, never on row order. The test set is
used **once**, in issue #7; everything before works on the ``dev`` side only.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl

from data.datasets import DatasetConfig, find_repo_root, load_dataset_config
from data.features import LABEL, USER, features_path
from data.intermediate import EVENT_ID, Check, publish

logger = logging.getLogger(__name__)

SEED = 20261101
TEST_SHARE = 0.2


def assign_split(
    features: pl.LazyFrame, *, test_share: float = TEST_SHARE, seed: int = SEED
) -> pl.DataFrame:
    """Assign every user to ``dev`` or ``test`` (grouped by user, stratified by label).

    Args:
        features: Feature table; only ``user_id`` and ``label_ato`` are read.
        test_share: Share of the users of each stratum put on the test side, rounded to the
            nearest whole user.
        seed: Seed of the random generator that draws the test users.

    Returns:
        One row per user, sorted by ``user_id``: ``user_id``, ``has_takeover`` (at least one
        takeover login) and ``split`` (``dev`` or ``test``).

    Raises:
        ValueError: If ``test_share`` is not strictly between 0 and 1.
    """
    if not 0 < test_share < 1:
        raise ValueError("test_share must be between 0 and 1")
    users = (
        features.group_by(USER)
        .agg(pl.col(LABEL).any().alias("has_takeover"))
        .sort(USER)  # independent of row order
        .collect()
    )
    rng = np.random.default_rng(seed)
    parts = []
    for has_takeover in (True, False):
        stratum = users.filter(pl.col("has_takeover") == has_takeover)
        n_test = round(test_share * stratum.height)
        is_test = np.zeros(stratum.height, dtype=bool)
        is_test[rng.permutation(stratum.height)[:n_test]] = True
        parts.append(
            stratum.with_columns(
                pl.Series("split", np.where(is_test, "test", "dev"), dtype=pl.String)
            )
        )
    return pl.concat(parts).sort(USER)


def split_path(config: DatasetConfig, data_dir: Path) -> Path:
    """Where the user -> split assignment for ``config`` lives.

    Args:
        config: Dataset configuration; its ``name`` names the file.
        data_dir: Root of the data layout (``<repo>/data``).

    Returns:
        ``<data_dir>/05_model_input/<name>_split.parquet``.
    """
    return data_dir / "05_model_input" / f"{config.name}_split.parquet"


def load_side(config: DatasetConfig, data_dir: Path, side: str) -> pl.LazyFrame:
    """Feature rows of one side of the split (``dev`` or ``test``), ordered by ``event_id``.

    Joins do not guarantee row order; a fixed order keeps every seeded step downstream (sampling,
    bootstrap) reproducible from run to run.

    Args:
        config: Dataset configuration.
        data_dir: Root of the data layout (``<repo>/data``) holding the feature table and the
            split.
        side: ``dev`` or ``test``.

    Returns:
        The rows of the feature table whose user is on ``side``, with the feature table's
        columns, sorted by ``event_id``.

    Raises:
        ValueError: If ``side`` is neither ``dev`` nor ``test``.
    """
    if side not in {"dev", "test"}:
        raise ValueError("side must be 'dev' or 'test'")
    users = pl.scan_parquet(split_path(config, data_dir)).filter(pl.col("split") == side)
    return (
        pl.scan_parquet(features_path(config, data_dir))
        .join(users.select(USER), on=USER, how="semi")
        .sort(EVENT_ID)
    )


def write_split(config: DatasetConfig, data_dir: Path, *, check: Check | None = None) -> Path:
    """Compute the split from the feature table and write it.

    Written to ``.part`` and renamed once complete and accepted by ``check``, if given
    (:func:`data.intermediate.publish`).

    Args:
        config: Dataset configuration.
        data_dir: Root of the data layout (``<repo>/data``); the feature table is read from it
            and the split is written into it.
        check: Function given the path of the finished ``.part`` file; raising rejects the file
            and keeps any previous output. None publishes without a check.

    Returns:
        The path of the split file (see :func:`split_path`).
    """
    out = split_path(config, data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".part")
    assign_split(pl.scan_parquet(features_path(config, data_dir))).write_parquet(partial)
    publish(partial, out, check)
    logger.info("Wrote %s", out.name)
    return out


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: compute and write the split of ``--dataset``.

    Prints the path of the split file, relative to the repository.

    Args:
        argv: Command-line arguments; None reads them from sys.argv.

    Returns:
        The process exit code (0 on success). Failures raise instead of returning a code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="rba", help="Name of conf/dataset/<name>.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = write_split(load_dataset_config(args.dataset), find_repo_root() / "data")
    print(out.relative_to(find_repo_root()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
