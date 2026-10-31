"""Download a dataset archive, verify it, and safely extract the raw data file.

Steps, all idempotent (re-running skips work that is already done and verified):

1. Download the archive over https into ``data/01_raw/``, computing its MD5 while streaming.
2. Reject the file if the checksum differs from the one the publisher declared.
3. Inspect the archive *before* extracting: refuse absolute paths, ``..`` components, symlinks,
   suspicious compression ratios (zip bombs) and oversized contents.
4. Extract only the one expected member, flattened into ``data/01_raw/``.

Nothing from the archive is ever executed.

Usage::

    make data                                         # default dataset (rba)
    PYTHONPATH=src uv run python -m data.download --dataset rba
"""

import argparse
import hashlib
import logging
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from data.datasets import DatasetConfig, find_repo_root, load_dataset_config

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1 << 20
MAX_COMPRESSION_RATIO = 200


class IntegrityError(RuntimeError):
    """A downloaded file does not match its published checksum."""


class UnsafeArchiveError(RuntimeError):
    """An archive contains entries that are unsafe or unexpected."""


def md5_of(path: Path) -> str:
    """Return the hex MD5 digest of a file, read in chunks.

    MD5 is used only to compare against the checksum the publisher declared, not for security.
    """
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path, expected_md5: str, *, timeout: float = 60.0) -> Path:
    """Download ``url`` to ``dest`` and verify its MD5, unless a verified copy already exists.

    The download goes to ``<dest>.part`` and is renamed only after the checksum matches, so an
    interrupted or corrupted download never looks complete.

    Raises:
        ValueError: If ``url`` is not https.
        IntegrityError: If the downloaded file's MD5 differs from ``expected_md5``.
    """
    if dest.is_file():
        if md5_of(dest) == expected_md5:
            logger.info("Already downloaded and verified: %s", dest)
            return dest
        logger.warning("Existing %s fails the checksum; downloading again", dest)
    if not url.startswith("https://"):
        raise ValueError(f"Refusing non-https download: {url}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    digest = hashlib.md5(usedforsecurity=False)
    logger.info("Downloading %s", url)
    with (
        urllib.request.urlopen(url, timeout=timeout) as response,  # noqa: S310  (https enforced above)
        partial.open("wb") as out,
    ):
        while chunk := response.read(CHUNK_BYTES):
            out.write(chunk)
            digest.update(chunk)

    actual = digest.hexdigest()
    if actual != expected_md5:
        partial.unlink()
        raise IntegrityError(f"MD5 mismatch for {url}: expected {expected_md5}, got {actual}")
    partial.replace(dest)
    logger.info("Downloaded and verified: %s (md5 %s)", dest, actual)
    return dest


def _check_entry(info: zipfile.ZipInfo) -> None:
    name = info.filename
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise UnsafeArchiveError(f"Absolute path in archive: {name!r}")
    if ".." in posix.parts or ".." in windows.parts:
        raise UnsafeArchiveError(f"Parent-directory reference in archive: {name!r}")
    if stat.S_ISLNK(info.external_attr >> 16):
        raise UnsafeArchiveError(f"Symbolic link in archive: {name!r}")
    if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
        raise UnsafeArchiveError(f"Suspicious compression ratio for {name!r} (possible zip bomb)")


def check_archive(archive: Path, member: str, max_uncompressed_bytes: int) -> zipfile.ZipInfo:
    """Inspect an archive without extracting it and return the entry for ``member``.

    Raises:
        UnsafeArchiveError: If any entry is unsafe, the total uncompressed size exceeds
            ``max_uncompressed_bytes``, or ``member`` is missing.
    """
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
    for info in infos:
        _check_entry(info)
    total = sum(info.file_size for info in infos)
    if total > max_uncompressed_bytes:
        raise UnsafeArchiveError(
            f"Archive expands to {total:,} bytes, above the {max_uncompressed_bytes:,} limit"
        )
    by_name = {info.filename: info for info in infos}
    if member not in by_name:
        raise UnsafeArchiveError(f"{member!r} not in archive; entries: {sorted(by_name)}")
    return by_name[member]


def extract_member(archive: Path, member: str, dest_dir: Path, max_uncompressed_bytes: int) -> Path:
    """Safely extract one member of ``archive`` into ``dest_dir`` (flattened to its file name).

    Skips extraction when a file of the expected size is already present. Zip CRC checks run
    while reading, and extraction stops if the member yields more bytes than its header declares.

    Raises:
        UnsafeArchiveError: If the archive fails :func:`check_archive` or the member's actual
            size differs from its header.
    """
    info = check_archive(archive, member, max_uncompressed_bytes)
    target = dest_dir / PurePosixPath(member).name
    if target.is_file() and target.stat().st_size == info.file_size:
        logger.info("Already extracted: %s", target)
        return target

    partial = target.with_name(target.name + ".part")
    written = 0
    with zipfile.ZipFile(archive) as zf, zf.open(info) as source, partial.open("wb") as out:
        while chunk := source.read(CHUNK_BYTES):
            written += len(chunk)
            if written > info.file_size:
                break
            out.write(chunk)
    if written != info.file_size:
        partial.unlink()
        raise UnsafeArchiveError(
            f"{member!r} yielded {written:,} bytes, header declares {info.file_size:,}"
        )
    partial.replace(target)
    logger.info("Extracted %s (%s bytes)", target, f"{written:,}")
    return target


def download_dataset(config: DatasetConfig, raw_dir: Path) -> Path:
    """Download, verify and extract a dataset; return the path of the raw data file."""
    source = config.source
    archive = fetch(str(source.url), raw_dir / source.archive, source.md5)
    return extract_member(archive, source.member, raw_dir, source.max_uncompressed_bytes)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="rba", help="Name of conf/dataset/<name>.toml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = load_dataset_config(args.dataset)
    raw_file = download_dataset(config, find_repo_root() / "data" / "01_raw")
    print(raw_file.relative_to(find_repo_root()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
