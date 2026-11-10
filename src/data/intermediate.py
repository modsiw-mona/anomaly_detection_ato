"""Build the intermediate dataset: raw file -> standard schema, real types, unified nulls, flags.

Stage ``data/01_raw`` -> ``data/02_intermediate`` of the data layout. Every step is driven by the
dataset's translation file (``conf/dataset/<name>.yaml``), so the same code serves any dataset that
meets the minimum data requirements:

1. Rename the dataset's columns to the standard field names; unmapped columns are dropped.
2. Turn the dataset's null markers into real nulls, **per column only** (a global rule would delete
   real values, e.g. the region "Nan" in Thailand).
3. Flag the declared artefacts as boolean columns, computed on the raw values.
4. Convert every field to the storage type of its kind. Conversions are strict: a value that does
   not parse stops the build instead of silently becoming null.
5. Add ``event_id``, the row's position in the raw file, so every record can be traced back.

The work is lazy and streamed with polars, so the full file is never loaded into memory.

Usage::

    make intermediate                                   # default dataset (rba)
    PYTHONPATH=src uv run python -m data.intermediate --dataset rba
"""

import argparse
import ipaddress
import logging
from collections.abc import Callable
from pathlib import Path

import polars as pl

from data.datasets import Artefact, DatasetConfig, find_repo_root, load_dataset_config
from data.schema import FIELDS, FIELDS_BY_NAME, Kind

logger = logging.getLogger(__name__)

EVENT_ID = "event_id"
_IPV4 = r"^(?P<a>\d{1,3})\.(?P<b>\d{1,3})\.(?P<c>\d{1,3})\.(?P<d>\d{1,3})$"


def scan_raw(raw_file: Path) -> pl.LazyFrame:
    """Lazily scan a raw CSV with every column as text, so no type is guessed.

    Args:
        raw_file: CSV file with a header row.

    Returns:
        A lazy frame with the file's own column names, every column of type ``String``.
    """
    return pl.scan_csv(raw_file, infer_schema=False)


def _ipv4_to_int(column: str) -> pl.Expr:
    """IPv4 text to its 32-bit integer value; null for anything that is not dotted IPv4."""
    octets = pl.col(column).str.extract_groups(_IPV4)
    value = pl.lit(0, dtype=pl.Int64)
    for name in ("a", "b", "c", "d"):
        value = value * 256 + octets.struct.field(name).cast(pl.Int64)
    return value


def artefact_flag(artefact: Artefact) -> pl.Expr:
    """Boolean expression that is True where ``artefact`` applies (never null).

    An ``at_least`` artefact applies where the field, read as an integer, is at least the
    declared value; an ``in_network`` artefact applies where the field is a dotted IPv4 address
    inside the declared network. A value that cannot be read that way counts as False.

    Args:
        artefact: Declared artefact; its ``field`` is read under its standard name.

    Returns:
        A polars expression named after ``artefact.name``.
    """
    match artefact.kind:
        case "at_least":
            flag = pl.col(artefact.field).cast(pl.Int64, strict=False) >= artefact.value
        case "in_network":
            network = ipaddress.IPv4Network(str(artefact.value))
            flag = _ipv4_to_int(artefact.field).is_between(
                int(network.network_address), int(network.broadcast_address)
            )
    return flag.fill_null(value=False).alias(artefact.name)


def mapped_fields(config: DatasetConfig) -> list[str]:
    """Standard fields the dataset provides (labels included), in canonical order.

    Args:
        config: Dataset configuration.

    Returns:
        The names of the canonical fields mapped in ``columns`` or ``labels``, in the order of
        :data:`data.schema.FIELDS`.
    """
    mapped = config.rename_map().values()
    return [f.name for f in FIELDS if f.name in mapped]


def storage_dtype(field: str, config: DatasetConfig) -> pl.DataType:
    """Storage type of a standard field in the intermediate dataset.

    Args:
        field: Canonical field name.
        config: Dataset configuration; it decides whether an identifier is stored as ``Int64``
            or ``String``.

    Returns:
        The polars type for the field's kind: ``Int64`` or ``String`` for identifiers,
        ``Datetime("ms")`` for the timestamp, ``Boolean``, ``Categorical``, ``Float64`` for
        numbers and ``String`` for text.

    Raises:
        KeyError: If ``field`` is not a canonical field.
    """
    match FIELDS_BY_NAME[field].kind:
        case Kind.IDENTIFIER:
            is_int = config.identifier_dtypes.get(field) == "Int64"
            return pl.Int64() if is_int else pl.String()
        case Kind.TIMESTAMP:
            return pl.Datetime("ms")
        case Kind.BOOLEAN:
            return pl.Boolean()
        case Kind.CATEGORICAL:
            return pl.Categorical()
        case Kind.NUMERIC:
            return pl.Float64()
        case Kind.TEXT:
            return pl.String()


def _typed(field: str, config: DatasetConfig) -> pl.Expr:
    """Strict conversion of a text column to the storage type of its kind."""
    column = pl.col(field)
    match FIELDS_BY_NAME[field].kind:
        case Kind.IDENTIFIER:
            dtype = config.identifier_dtypes.get(field, "String")
            typed = column.cast(pl.Int64, strict=True) if dtype == "Int64" else column
        case Kind.TIMESTAMP:
            typed = column.str.to_datetime(config.timestamp_format, time_unit="ms", strict=True)
        case Kind.BOOLEAN:
            typed = column.str.to_lowercase().replace_strict(
                {"true": True, "false": False}, return_dtype=pl.Boolean
            )
        case Kind.CATEGORICAL:
            typed = column.cast(pl.Categorical)
        case Kind.NUMERIC:
            typed = column.cast(pl.Float64, strict=True)
        case Kind.TEXT:
            typed = column
    return typed.alias(field)


def standardise(raw: pl.LazyFrame, config: DatasetConfig) -> pl.LazyFrame:
    """Apply steps 1-5 of the module docstring to a raw, all-text frame.

    The conversions are lazy: a value that does not parse raises only when the frame is
    collected or written.

    Args:
        raw: Raw dataset with its own column names, every column as text (see :func:`scan_raw`).
        config: Dataset configuration (the translation file).

    Returns:
        One row per raw row, in file order: ``event_id`` (the row's position in the raw file),
        every mapped field under its standard name and storage type, then one boolean column
        per declared artefact.
    """
    fields = mapped_fields(config)
    source_of = {canonical: source for source, canonical in config.rename_map().items()}

    frame = raw.with_row_index(EVENT_ID).select(
        EVENT_ID, *(pl.col(source_of[f]).alias(f) for f in fields)
    )
    if config.null_markers:
        frame = frame.with_columns(
            pl.when(pl.col(f).is_in(list(markers))).then(None).otherwise(pl.col(f)).alias(f)
            for f, markers in config.null_markers.items()
        )
    flags = [artefact_flag(a) for a in config.artefacts]
    if flags:
        frame = frame.with_columns(flags)
    return frame.with_columns(_typed(f, config) for f in fields)


Check = Callable[[Path], object]


def publish(partial: Path, out_file: Path, check: Check | None = None) -> None:
    """Rename a finished ``partial`` file to ``out_file``, after ``check`` (if any) accepts it.

    If ``check`` raises, the partial file is removed and ``out_file`` is left as it was, so a
    rejected rebuild never replaces a good output.

    Args:
        partial: Finished file to publish, usually ``<out_file>.part``.
        out_file: Final path; an existing file there is replaced.
        check: Function given the path of ``partial``; raising rejects it. None publishes
            without a check.

    Raises:
        BaseException: Whatever ``check`` raises, re-raised after ``partial`` is removed.
    """
    if check is not None:
        try:
            check(partial)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    partial.replace(out_file)


def intermediate_path(config: DatasetConfig, data_dir: Path) -> Path:
    """Where the intermediate parquet for ``config`` lives.

    Args:
        config: Dataset configuration; its ``name`` names the file.
        data_dir: Root of the data layout (``<repo>/data``).

    Returns:
        ``<data_dir>/02_intermediate/<name>.parquet``.
    """
    return data_dir / "02_intermediate" / f"{config.name}.parquet"


def build_intermediate(
    config: DatasetConfig, raw_file: Path, out_file: Path, *, check: Check | None = None
) -> Path:
    """Stream ``raw_file`` through :func:`standardise` into a zstd-compressed parquet file.

    The file is written to ``<out_file>.part`` and renamed only when complete and, if ``check`` is
    given, only after ``check`` accepts it (see :func:`publish`).

    Args:
        config: Dataset configuration (the translation file).
        raw_file: Raw CSV file, as extracted into ``data/01_raw``.
        out_file: Path of the parquet file to write; its directory is created if needed.
        check: Function given the path of the finished ``.part`` file; raising rejects the file
            and keeps any previous output. None publishes without a check.

    Returns:
        ``out_file``.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    partial = out_file.with_name(out_file.name + ".part")
    standardise(scan_raw(raw_file), config).sink_parquet(partial, compression="zstd")
    publish(partial, out_file, check)
    logger.info("Wrote %s", out_file.name)
    return out_file


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: build the intermediate dataset of ``--dataset``.

    Reads the raw file from ``data/01_raw`` and prints the path of the written parquet file,
    relative to the repository.

    Args:
        argv: Command-line arguments; None reads them from sys.argv.

    Returns:
        The process exit code (0 on success). Failures raise instead of returning a code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="rba", help="Name of conf/dataset/<name>.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = load_dataset_config(args.dataset)
    data_dir = find_repo_root() / "data"
    raw_file = data_dir / "01_raw" / Path(config.source.member).name
    out = build_intermediate(config, raw_file, intermediate_path(config, data_dir))
    print(out.relative_to(find_repo_root()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
