"""Load and validate per-dataset configuration files (``conf/dataset/<name>.toml``).

A dataset configuration is the "translation file" between a dataset's own column names and the
canonical schema in :mod:`data.schema`. It also records where the data comes from, how to verify
it, how to cite it, and its known artefacts.
"""

import re
import tomllib
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from data.schema import FIELDS_BY_NAME, LABEL_FIELDS, check_requirements

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


class DatasetConfig(_Strict):
    """Validated contents of one ``conf/dataset/<name>.toml`` file."""

    name: str
    description: str
    source: Source
    columns: dict[str, str] = Field(description="Canonical field -> column name in the dataset.")
    labels: dict[str, str] = Field(
        default_factory=dict, description="Canonical label field -> column name in the dataset."
    )
    artefacts: tuple[Artefact, ...] = ()

    @model_validator(mode="after")
    def _check_mapping(self) -> Self:
        if overlap := self.columns.keys() & self.labels.keys():
            raise ValueError(f"Fields mapped both as columns and labels: {sorted(overlap)}")
        if misplaced := self.columns.keys() & LABEL_FIELDS:
            raise ValueError(f"Label fields must go under [labels]: {sorted(misplaced)}")
        if not_labels := self.labels.keys() - LABEL_FIELDS:
            raise ValueError(f"Not label fields: {sorted(not_labels)}")
        check_requirements(self.columns)
        return self

    def rename_map(self) -> dict[str, str]:
        """Return the dataset-column -> canonical-field mapping, labels included."""
        return {source: canonical for canonical, source in {**self.columns, **self.labels}.items()}


def load_dataset_config(name: str, conf_dir: Path | None = None) -> DatasetConfig:
    """Load and validate ``conf/dataset/<name>.toml``.

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
    with (directory / f"{name}.toml").open("rb") as handle:
        raw = tomllib.load(handle)
    return DatasetConfig.model_validate(raw)
