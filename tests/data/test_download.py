"""Tests for downloading, verifying and safely extracting dataset archives."""

import hashlib
import io
import stat
import zipfile
from pathlib import Path
from typing import Any

import pytest

from data import download
from data.datasets import DatasetConfig
from data.download import (
    IntegrityError,
    UnsafeArchiveError,
    check_archive,
    download_dataset,
    extract_member,
    fetch,
    md5_of,
)

URL = "https://example.org/toy.zip"
CSV = b"User,When,Ok,ASN\n1,2020-01-01,True,123\n"


def _md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> list[str]:
    """Make ``urlopen`` return ``payload``; return the list of URLs requested."""
    calls: list[str] = []

    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen)
    return calls


def _zip(path: Path, entries: dict[str | zipfile.ZipInfo, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def test_md5_of(tmp_path: Path) -> None:
    file = tmp_path / "f.bin"
    file.write_bytes(CSV)
    assert md5_of(file) == _md5(CSV)


def test_fetch_downloads_and_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _serve(monkeypatch, CSV)
    dest = fetch(URL, tmp_path / "toy.zip", _md5(CSV))
    assert dest.read_bytes() == CSV
    assert calls == [URL]
    assert not list(tmp_path.glob("*.part"))


def test_fetch_rejects_checksum_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, b"tampered")
    with pytest.raises(IntegrityError, match="MD5 mismatch"):
        fetch(URL, tmp_path / "toy.zip", _md5(CSV))
    assert not list(tmp_path.iterdir()), "nothing may be left behind after a failed download"


def test_fetch_skips_verified_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _serve(monkeypatch, b"should not be requested")
    dest = tmp_path / "toy.zip"
    dest.write_bytes(CSV)
    fetch(URL, dest, _md5(CSV))
    assert calls == []


def test_fetch_replaces_corrupt_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _serve(monkeypatch, CSV)
    dest = tmp_path / "toy.zip"
    dest.write_bytes(b"truncated")
    fetch(URL, dest, _md5(CSV))
    assert calls == [URL]
    assert dest.read_bytes() == CSV


def test_fetch_refuses_plain_http(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-https"):
        fetch("http://example.org/toy.zip", tmp_path / "toy.zip", _md5(CSV))


def test_extract_member_flattens_and_is_idempotent(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "a.zip", {"nested/toy.csv": CSV, "README.txt": b"hi"})
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    target = extract_member(archive, "nested/toy.csv", out_dir, 10_000)
    assert target == out_dir / "toy.csv"
    assert target.read_bytes() == CSV
    mtime = target.stat().st_mtime_ns
    assert extract_member(archive, "nested/toy.csv", out_dir, 10_000) == target
    assert target.stat().st_mtime_ns == mtime, "second run must not rewrite the file"


def _symlink_entry() -> zipfile.ZipInfo:
    info = zipfile.ZipInfo("link.csv")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    return info


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ("../evil.csv", "Parent-directory"),
        ("ok/../../evil.csv", "Parent-directory"),
        ("/etc/evil.csv", "Absolute path"),
        ("C:/evil.csv", "Absolute path"),
        (_symlink_entry(), "Symbolic link"),
    ],
)
def test_unsafe_entries_are_rejected(
    tmp_path: Path, entry: str | zipfile.ZipInfo, match: str
) -> None:
    archive = _zip(tmp_path / "a.zip", {"toy.csv": CSV, entry: b"x"})
    with pytest.raises(UnsafeArchiveError, match=match):
        check_archive(archive, "toy.csv", 10_000)


def test_zip_bomb_is_rejected(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "a.zip", {"toy.csv": b"\0" * 5_000_000})
    with pytest.raises(UnsafeArchiveError, match="compression ratio"):
        check_archive(archive, "toy.csv", 10_000_000)


def test_oversized_archive_is_rejected(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "a.zip", {"toy.csv": CSV})
    with pytest.raises(UnsafeArchiveError, match="limit"):
        check_archive(archive, "toy.csv", len(CSV) - 1)


def test_missing_member_is_rejected(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "a.zip", {"other.csv": CSV})
    with pytest.raises(UnsafeArchiveError, match="not in archive"):
        check_archive(archive, "toy.csv", 10_000)


def test_download_dataset_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _zip(tmp_path / "served.zip", {"toy/toy.csv": CSV}).read_bytes()
    _serve(monkeypatch, payload)
    source: dict[str, Any] = {
        "url": URL,
        "archive": "toy.zip",
        "md5": _md5(payload),
        "member": "toy/toy.csv",
        "max_uncompressed_bytes": 10_000,
        "doi": "10.0/toy",
        "version": "v1",
        "license": "CC0-1.0",
        "citation": "Toy (2026).",
        "homepage": "https://example.org/toy",
    }
    config = DatasetConfig.model_validate(
        {
            "name": "toy",
            "description": "Toy.",
            "source": source,
            "columns": {
                "user_id": "User",
                "timestamp": "When",
                "login_success": "Ok",
                "asn": "ASN",
            },
        }
    )
    raw_dir = tmp_path / "raw"
    raw_file = download_dataset(config, raw_dir)
    assert raw_file == raw_dir / "toy.csv"
    assert raw_file.read_bytes() == CSV
