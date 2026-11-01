"""Load and validate per-dataset configuration files (``conf/dataset/<name>.yaml``).

A dataset configuration is the "translation file" between a dataset's own column names and the
canonical schema in :mod:`data.schema`. It also records where the data comes from, how to verify
it, how to cite it, and its known artefacts.
"""

import datetime as dt
import ipaddress
import re
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from data.schema import FIELDS_BY_NAME, LABEL_FIELDS, Kind, check_requirements

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

Md5Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root: the closest ancestor containing ``pyproject.toml``.

    Args:
        start: Directory to search upward from. Defaults to this file's directory.

    Raises:
        FileNotFoundError: If no ancestor contains ``pyproject.toml``.
    """
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"No pyproject.toml found above {here}")


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Source(_Strict):
    """Where the dataset comes from and how to verify it."""

    url: HttpUrl
    archive: str = Field(description="File name to save the downloaded archive as.")
    md5: Md5Hex = Field(description="Published MD5 checksum of the archive.")
    member: str = Field(description="Path of the raw data file inside the archive.")
    max_uncompressed_bytes: int = Field(gt=0, description="Upper bound for the extracted size.")
    doi: str
    version: str
    license: str
    citation: str
    homepage: HttpUrl
    mirrors: tuple[HttpUrl, ...] = ()

    @field_validator("url")
    @classmethod
    def _https_only(cls, value: HttpUrl) -> HttpUrl:
        if value.scheme != "https":
            raise ValueError("Dataset downloads must use https")
        return value

    @field_validator("archive")
    @classmethod
    def _plain_file_name(cls, value: str) -> str:
        if Path(value).name != value:
            raise ValueError("archive must be a plain file name, not a path")
        return value


class Artefact(_Strict):
    """A known data-generation artefact that should be flagged, not silently trusted."""

    name: str = Field(description="Name of the boolean flag column this artefact produces.")
    field: str = Field(description="Canonical field the rule applies to.")
    kind: Literal["at_least", "in_network"]
    value: int | str
    meaning: str

    @field_validator("field")
    @classmethod
    def _known_field(cls, value: str) -> str:
        if value not in FIELDS_BY_NAME:
            raise ValueError(f"Unknown canonical field {value!r}")
        return value

    @model_validator(mode="after")
    def _value_matches_kind(self) -> Self:
        if self.kind == "at_least" and not isinstance(self.value, int):
            raise ValueError("'at_least' artefacts need an integer value")
        if self.kind == "in_network":
            if not isinstance(self.value, str):
                raise ValueError("'in_network' artefacts need a network such as '10.0.0.0/8'")
            ipaddress.IPv4Network(self.value)  # raises ValueError if invalid; IPv6 unsupported
        return self


IdentifierDtype = Literal["Int64", "String"]


class ValidationRules(_Strict):
    """Dataset-specific data rules, checked on the intermediate dataset (see data.validation).

    Generic rules (types, required fields, unique event IDs) come from the canonical schema; these
    capture what is specific to one dataset. Every key refers to a canonical field.
    """

    timestamp_range: tuple[dt.date, dt.date] | None = Field(
        default=None, description="Allowed [start, end) of the timestamp."
    )
    unique_together: tuple[str, ...] = Field(
        default=(), description="Fields that together identify a row."
    )
    allowed_values: dict[str, tuple[str, ...]] = Field(
        default_factory=dict,
        description="Closed set of values (nulls allowed if the field may be null).",
    )
    patterns: dict[str, str] = Field(
        default_factory=dict, description="Regular expression every non-null value must match."
    )
    numeric_ranges: dict[str, tuple[float, float]] = Field(
        default_factory=dict, description="Inclusive [min, max] for numeric fields."
    )

    @field_validator("patterns")
    @classmethod
    def _patterns_compile(cls, value: dict[str, str]) -> dict[str, str]:
        for pattern in value.values():
            re.compile(pattern)
        return value

    @model_validator(mode="after")
    def _ranges_are_ordered(self) -> Self:
        if self.timestamp_range and self.timestamp_range[0] >= self.timestamp_range[1]:
            raise ValueError("timestamp_range start must be before its end")
        if bad := [f for f, (lo, hi) in self.numeric_ranges.items() if lo > hi]:
            raise ValueError(f"numeric_ranges with min > max: {bad}")
        return self

    def fields(self) -> set[str]:
        """Every canonical field these rules refer to."""
        return {
            *self.unique_together,
            *self.allowed_values,
            *self.patterns,
            *self.numeric_ranges,
        }


class DatasetConfig(_Strict):
    """Validated contents of one ``conf/dataset/<name>.yaml`` file."""

    name: str
    description: str
    source: Source
    columns: dict[str, str] = Field(description="Canonical field -> column name in the dataset.")
    labels: dict[str, str] = Field(
        default_factory=dict, description="Canonical label field -> column name in the dataset."
    )
    timestamp_format: str = Field(
        default="%Y-%m-%d %H:%M:%S%.f", description="strftime-style format of the timestamp column."
    )
    null_markers: dict[str, tuple[str, ...]] = Field(
        default_factory=dict,
        description="Canonical field -> raw values that mean 'missing' in that column only.",
    )
    identifier_dtypes: dict[str, IdentifierDtype] = Field(
        default_factory=dict,
        description="Storage type for identifier fields (default String).",
    )
    artefacts: tuple[Artefact, ...] = ()
    catch_all_user_ids: tuple[str, ...] = Field(
        default=(),
        description="Raw user IDs that do not identify a real account (e.g. a bucket for attempts "
        "against unknown accounts). Excluded from per-user behaviour.",
    )
    validation: ValidationRules = Field(default_factory=ValidationRules)
    label_window: tuple[dt.date, dt.date] | None = Field(
        default=None,
        description="[start, end) of the period whose labels are trustworthy. Only logins in it are "
        "scored and evaluated; earlier and later events may still feed past-only history features.",
    )

    @model_validator(mode="after")
    def _check_mapping(self) -> Self:
        if overlap := self.columns.keys() & self.labels.keys():
            raise ValueError(f"Fields mapped both as columns and labels: {sorted(overlap)}")
        if misplaced := self.columns.keys() & LABEL_FIELDS:
            raise ValueError(f"Label fields must go under [labels]: {sorted(misplaced)}")
        if not_labels := self.labels.keys() - LABEL_FIELDS:
            raise ValueError(f"Not label fields: {sorted(not_labels)}")
        check_requirements(self.columns)
        mapped = self.columns.keys() | self.labels.keys()
        for setting, fields in (
            ("null_markers", self.null_markers.keys()),
            ("identifier_dtypes", self.identifier_dtypes.keys()),
            ("artefacts", {artefact.field for artefact in self.artefacts}),
            ("validation", self.validation.fields()),
        ):
            if unmapped := set(fields) - mapped:
                raise ValueError(f"{setting} refers to unmapped field(s): {sorted(unmapped)}")
        if not_ids := {
            f for f in self.identifier_dtypes if FIELDS_BY_NAME[f].kind is not Kind.IDENTIFIER
        }:
            raise ValueError(f"identifier_dtypes only applies to identifiers: {sorted(not_ids)}")
        if self.label_window and self.label_window[0] >= self.label_window[1]:
            raise ValueError("label_window start must be before its end")
        return self

    def rename_map(self) -> dict[str, str]:
        """Return the dataset-column -> canonical-field mapping, labels included."""
        return {source: canonical for canonical, source in {**self.columns, **self.labels}.items()}


def load_dataset_config(name: str, conf_dir: Path | None = None) -> DatasetConfig:
    """Load and validate ``conf/dataset/<name>.yaml`` (parsed with ``yaml.safe_load``).

    Args:
        name: Dataset name (lowercase letters, digits, ``_`` and ``-``).
        conf_dir: Directory holding the dataset files. Defaults to ``<repo>/conf/dataset``.

    Raises:
        ValueError: If ``name`` is not a valid dataset name.
        FileNotFoundError: If the configuration file does not exist.
        pydantic.ValidationError: If the file content is invalid.
    """
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid dataset name {name!r}")
    directory = conf_dir or find_repo_root() / "conf" / "dataset"
    with (directory / f"{name}.yaml").open(encoding="utf-8") as handle:
        raw: object = yaml.safe_load(handle)
    return DatasetConfig.model_validate(raw)
