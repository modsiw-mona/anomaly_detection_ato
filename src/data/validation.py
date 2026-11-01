"""Data validation rules for the intermediate dataset (Pandera, polars backend).

The schema is built from two sources, so it works for any dataset:

* **Generic rules**, derived from the canonical schema: every column has its storage type, required
  fields and labels are never null, ``event_id`` is unique, artefact flags are never null, and no
  unexpected column is present.
* **Dataset-specific rules**, read from the ``validation`` section of the translation file: allowed
  timestamp range, fields that together identify a row, closed sets of values, patterns and
  numeric ranges.

Rules that hold *almost* always are **warnings**, not errors: a confirmed takeover is expected on a
successful login, but the RBA file has one on a failed login, and that should be reported, not stop
the pipeline.

Usage::

    validate_intermediate(frame, config)  # raises pandera.errors.SchemaErrors listing every failure
"""

import datetime as dt
from collections.abc import Callable

import pandera.polars as pa
import polars as pl
from pandera.api.polars.types import PolarsData

from data.datasets import DatasetConfig
from data.intermediate import EVENT_ID, mapped_fields, storage_dtype
from data.schema import FIELDS_BY_NAME, Role

CheckFn = Callable[[PolarsData], pl.LazyFrame]


def _column_check(make: Callable[[pl.Expr], pl.Expr]) -> CheckFn:
    """Wrap a per-value condition so nulls pass (nullability is checked separately)."""

    def check(data: PolarsData) -> pl.LazyFrame:
        column = pl.col(data.key)
        return data.lazyframe.select(make(column) | column.is_null())

    return check


def _as_text(column: pl.Expr) -> pl.Expr:
    return column.cast(pl.String)


def _in_set(values: tuple[str, ...]) -> CheckFn:
    return _column_check(lambda c: _as_text(c).is_in(list(values)))


def _matches(pattern: str) -> CheckFn:
    return _column_check(lambda c: _as_text(c).str.contains(pattern))


def _between(low: float, high: float) -> CheckFn:
    return _column_check(lambda c: c.is_between(low, high))


def _in_period(start: dt.date, end: dt.date) -> CheckFn:
    start_at, end_at = dt.datetime.combine(start, dt.time()), dt.datetime.combine(end, dt.time())
    return _column_check(lambda c: (c >= start_at) & (c < end_at))


def _column_checks(field: str, config: DatasetConfig) -> list[pa.Check]:
    rules = config.validation
    checks: list[pa.Check] = []
    if field == "timestamp" and rules.timestamp_range:
        checks.append(pa.Check(_in_period(*rules.timestamp_range), name="timestamp_range"))
    if field in rules.allowed_values:
        checks.append(pa.Check(_in_set(rules.allowed_values[field]), name="allowed_values"))
    if field in rules.patterns:
        checks.append(pa.Check(_matches(rules.patterns[field]), name="pattern"))
    if field in rules.numeric_ranges:
        checks.append(pa.Check(_between(*rules.numeric_ranges[field]), name="numeric_range"))
    return checks


def _takeover_implies_success(data: PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select(~pl.col("label_ato") | pl.col("login_success"))


def build_schema(config: DatasetConfig) -> pa.DataFrameSchema:
    """Pandera schema for the intermediate dataset of ``config``."""
    columns: dict[str, pa.Column] = {EVENT_ID: pa.Column(pl.UInt32, unique=True)}
    for field in mapped_fields(config):
        role = FIELDS_BY_NAME[field].role
        columns[field] = pa.Column(
            storage_dtype(field, config),
            checks=_column_checks(field, config),
            nullable=role not in {Role.REQUIRED, Role.LABEL},
        )
    for artefact in config.artefacts:
        columns[artefact.name] = pa.Column(pl.Boolean)

    frame_checks: list[pa.Check] = []
    if {"label_ato", "login_success"} <= columns.keys():
        frame_checks.append(
            pa.Check(
                _takeover_implies_success,
                name="takeover_on_successful_login",
                raise_warning=True,
            )
        )
    return pa.DataFrameSchema(
        columns,
        checks=frame_checks,
        unique=list(config.validation.unique_together) or None,
        strict=True,
        name=f"{config.name}_intermediate",
    )


def validate_intermediate(frame: pl.DataFrame, config: DatasetConfig) -> pl.DataFrame:
    """Validate ``frame`` against :func:`build_schema`, collecting every failure.

    Raises:
        pandera.errors.SchemaErrors: If any error-level rule fails (all failures are listed).
    """
    validated: pl.DataFrame = build_schema(config).validate(frame, lazy=True)
    return validated
