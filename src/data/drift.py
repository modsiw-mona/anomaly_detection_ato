"""Distribution drift between two samples of the feature table, one feature at a time.

The feature pipeline compares the development and the test side of the split. Users are assigned at
random, so both sides should look alike: a large difference means the split, or the data, is not
what it should be. Only the features are compared, never the labels.

* **Numbers** (counts, shares, hours): the two-sample **Kolmogorov-Smirnov** statistic, the largest
  gap between the two empirical distribution functions, on the known values.
* **Yes/no and categories**: the **total variation distance**, half the sum of the absolute
  differences between the share of each value on the two sides.
* **Missing values**: the difference in the share of nulls, for both kinds.

A feature's score is the largest of its numbers, between 0 (identical) and 1. The limit, 0.2, is the
condition the course uses for feature drift (the Deepchecks *Feature Drift* check).
"""

import numpy as np
import polars as pl

DRIFT_LIMIT = 0.2


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic of two 1-D samples (0 if either is empty).

    Args:
        a: First sample of numbers, without missing values.
        b: Second sample of numbers, without missing values.

    Returns:
        The largest absolute gap between the two empirical distribution functions, between 0
        and 1; 0 if either sample is empty.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    a, b = np.sort(a), np.sort(b)
    grid = np.union1d(a, b)
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def total_variation(a: pl.Series, b: pl.Series) -> float:
    """Total variation distance between the value shares of two samples (nulls included).

    Values are compared as text, and null counts as one more value.

    Args:
        a: First sample.
        b: Second sample.

    Returns:
        Half the sum of the absolute differences between the share of each value in ``a`` and
        in ``b``, between 0 and 1; 0 if either sample is empty.
    """
    if a.len() == 0 or b.len() == 0:
        return 0.0
    shares = (
        pl.DataFrame({"value": a.cast(pl.String)})
        .group_by("value")
        .agg((pl.len() / a.len()).alias("a"))
        .join(
            pl.DataFrame({"value": b.cast(pl.String)})
            .group_by("value")
            .agg((pl.len() / b.len()).alias("b")),
            on="value",
            how="full",
            coalesce=True,
            nulls_equal=True,
        )
        .fill_null(0.0)
    )
    return 0.5 * float((shares["a"] - shares["b"]).abs().sum())


def _null_gap(a: pl.Series, b: pl.Series) -> float:
    if a.len() == 0 or b.len() == 0:
        return 0.0
    return abs(a.null_count() / a.len() - b.null_count() / b.len())


def drift_report(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    columns: list[str],
    *,
    limit: float = DRIFT_LIMIT,
) -> pl.DataFrame:
    """One row per column: method, score, the null-share gap and whether it is within ``limit``.

    Numeric columns are compared with :func:`ks_statistic` on their known values, the others
    with :func:`total_variation`.

    Args:
        reference: Sample to compare against, for example the development side of the split.
        current: Sample to compare, for example the test side of the split.
        columns: Columns to compare; each must be present in both frames.
        limit: Score from which a column counts as drifted.

    Returns:
        One row per column of ``columns``, in that order: ``column``, ``method``
        (``kolmogorov_smirnov`` or ``total_variation``), ``distance``, ``null_share_gap`` (the
        absolute difference in the share of nulls), ``score`` (the larger of the two) and
        ``within_limit`` (``score`` strictly below ``limit``).
    """
    rows = []
    for column in columns:
        a, b = reference[column], current[column]
        numeric = a.dtype.is_numeric()
        if numeric:
            known_a = a.drop_nulls().cast(pl.Float64).to_numpy()
            known_b = b.drop_nulls().cast(pl.Float64).to_numpy()
            distance = ks_statistic(known_a, known_b)
        else:
            distance = total_variation(a, b)
        nulls = _null_gap(a, b)
        score = max(distance, nulls)
        rows.append(
            {
                "column": column,
                "method": "kolmogorov_smirnov" if numeric else "total_variation",
                "distance": distance,
                "null_share_gap": nulls,
                "score": score,
                "within_limit": score < limit,
            }
        )
    return pl.DataFrame(rows, schema_overrides={"within_limit": pl.Boolean})
