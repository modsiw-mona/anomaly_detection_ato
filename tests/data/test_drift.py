"""Tests for the drift measures between two samples of the feature table."""

import numpy as np
import polars as pl
import pytest

from data.drift import DRIFT_LIMIT, drift_report, ks_statistic, total_variation


def test_ks_is_zero_for_identical_samples_and_one_for_disjoint_ones() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert ks_statistic(a, a.copy()) == 0
    assert ks_statistic(a, a + 10) == 1


def test_ks_hand_computed() -> None:
    # distribution functions at 1, 2, 3, 4: a = 1/3, 2/3, 1, 1; b = 0, 1/3, 2/3, 1
    assert ks_statistic(np.array([1.0, 2, 3]), np.array([2.0, 3, 4])) == pytest.approx(1 / 3)


def test_ks_ignores_order_and_handles_ties() -> None:
    a = np.array([3.0, 1, 1, 2])
    assert ks_statistic(a, np.array([1.0, 1, 2, 3])) == 0
    assert ks_statistic(np.array([0.0, 0, 0, 1]), np.array([0.0, 1, 1, 1])) == pytest.approx(0.5)


def test_total_variation_hand_computed() -> None:
    a = pl.Series(["x", "x", "y", "y"])
    b = pl.Series(["x", "y", "y", "y"])
    assert total_variation(a, b) == pytest.approx(0.25)
    assert total_variation(a, a) == 0
    assert total_variation(pl.Series(["x"]), pl.Series(["y"])) == 1


def test_total_variation_counts_nulls_as_a_value() -> None:
    a = pl.Series([True, None])
    b = pl.Series([True, True])
    assert total_variation(a, b) == pytest.approx(0.5)


def test_empty_samples_do_not_drift() -> None:
    assert ks_statistic(np.array([]), np.array([1.0])) == 0
    assert total_variation(pl.Series([], dtype=pl.String), pl.Series(["x"])) == 0


def test_report_uses_the_measure_of_each_kind_and_the_null_gap() -> None:
    reference = pl.DataFrame({"n": [1.0, 2.0, 3.0, None], "flag": [True, False, True, False]})
    current = pl.DataFrame({"n": [1.0, 2.0, 3.0, 4.0], "flag": [True, False, True, False]})
    report = drift_report(reference, current, ["n", "flag"]).rows_by_key("column", named=True)
    n, flag = report["n"][0], report["flag"][0]
    assert n["method"] == "kolmogorov_smirnov"
    assert flag["method"] == "total_variation"
    assert n["null_share_gap"] == pytest.approx(0.25)
    assert n["score"] == pytest.approx(max(n["distance"], 0.25))
    assert flag["score"] == 0
    assert flag["within_limit"]


def test_report_flags_a_feature_over_the_limit() -> None:
    reference = pl.DataFrame({"n": np.zeros(100)})
    current = pl.DataFrame({"n": np.r_[np.zeros(70), np.ones(30)]})  # 30% moved
    row = drift_report(reference, current, ["n"]).row(0, named=True)
    assert row["score"] == pytest.approx(0.3)
    assert row["score"] > DRIFT_LIMIT
    assert not row["within_limit"]
