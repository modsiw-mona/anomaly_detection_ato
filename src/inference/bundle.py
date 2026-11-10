"""The demo bundle: everything the demo needs, small enough to publish, built from the full data.

The demo explains real test logins with the selected model. Locally it reads the full feature table and score
files (millions of rows, gitignored); a public deployment cannot. The bundle keeps only what the demo uses:

* the **scenario logins**: per kind of outcome (takeover blocked, asked for 2FA, missed; false alarm; ordinary),
  the median-score login of a returning account and of a first login, with how many test logins each kind has;
* the **reference** logins the explanations are relative to (a fixed sample of ordinary test logins);
* small summaries: the alert thresholds, the usual value of each number input among development takeovers and
  legitimate logins, the typical device share, the typical gap of each time range, and the takeover share of
  development logins by score range.

Everything is derived from the public RBA dataset (CC BY 4.0). :func:`fetch` downloads a published bundle and
checks every file against the SHA-256 recorded in the repository before anything is loaded.
"""

import hashlib
import importlib.metadata
import itertools
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from data.datasets import load_dataset_config
from data.split import load_side
from inference.predict import Detector
from inference.what_if import GAPS, typical_row

SUMMARY, SCENARIOS, REFERENCE = "summary.json", "scenarios.parquet", "reference.parquet"
TIER_BLOCK, TIER_STEP_UP = "review / block", "step-up"
REFERENCE_SIZE = 300  # ordinary logins that define "typical" for the explanations
DEVICE = "device_type"
DEVICES = ("desktop", "mobile", "tablet")
NUMBER_INPUTS = (  # the demo's sliders, in their order on the page
    "global_share_asn",
    "asn_prior_users",
    "asn_attempts_prev24h",
    "asn_prior_attack_share",
    "global_share_country",
    "global_share_browser",
)
DOWNLOAD_TIMEOUT = 60  # seconds per file
ALLOWED_SCHEMES = ("https", "file")  # file: for tests and local copies
MAX_FILE_BYTES = 64 * 1024 * 1024  # the model is ~10 MB; anything far larger is not ours
RETRY_WAITS = (2.0, 5.0, 15.0)  # seconds before each new attempt, for temporary failures only
TRANSIENT_HTTP = frozenset({429, 500, 502, 503, 504})
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # file and release names: no paths, no hosts


@dataclass
class DemoBundle:
    """What the demo shows, without the full data."""

    thresholds: dict[str, float]
    counts: dict[str, int]  # test logins of each kind
    scenarios: pl.DataFrame  # columns "kind" and "scenario", then the login's columns and its score
    reference: pl.DataFrame
    usual: dict[str, tuple[float | None, float | None]]  # number input -> (takeovers, legitimate)
    device_shares: dict[str, float]
    gap_hours: dict[str, float | None]
    score_bins: list[tuple[float, float, int, int]]  # (low, high, logins, takeovers)

    def picks(self) -> dict[str, dict[str, pl.DataFrame]]:
        """Scenario logins by kind and scenario name, each a one-row frame, in the stored order.

        Returns:
            Per kind of outcome, per scenario name (``returning`` or ``first``), the login as a
            one-row frame without the ``kind`` and ``scenario`` columns.
        """
        result: dict[str, dict[str, pl.DataFrame]] = {}
        for i in range(self.scenarios.height):
            row = self.scenarios[i]
            kind, name = row["kind"][0], row["scenario"][0]
            result.setdefault(kind, {})[name] = row.drop("kind", "scenario")
        return result

    def write(self, folder: Path) -> list[Path]:
        """Write the bundle's three files into ``folder``.

        Args:
            folder: Destination folder; created if missing. Existing files are overwritten.

        Returns:
            The paths written: the scenarios and reference parquet files, then the JSON summary.
        """
        folder.mkdir(parents=True, exist_ok=True)
        self.scenarios.write_parquet(folder / SCENARIOS)
        self.reference.write_parquet(folder / REFERENCE)
        summary = {
            "thresholds": self.thresholds,
            "counts": self.counts,
            "usual": {k: list(v) for k, v in self.usual.items()},
            "device_shares": self.device_shares,
            "gap_hours": self.gap_hours,
            "score_bins": [list(b) for b in self.score_bins],
        }
        (folder / SUMMARY).write_text(json.dumps(summary, indent=2))
        return [folder / SCENARIOS, folder / REFERENCE, folder / SUMMARY]

    @classmethod
    def read(cls, folder: Path) -> DemoBundle:
        """Read a bundle written by :meth:`write`.

        Args:
            folder: Folder holding the bundle's three files.

        Returns:
            The bundle.
        """
        summary: dict[str, Any] = json.loads((folder / SUMMARY).read_text())
        return cls(
            thresholds={k: float(v) for k, v in summary["thresholds"].items()},
            counts={k: int(v) for k, v in summary["counts"].items()},
            scenarios=pl.read_parquet(folder / SCENARIOS),
            reference=pl.read_parquet(folder / REFERENCE),
            usual={k: (v[0], v[1]) for k, v in summary["usual"].items()},
            device_shares={k: float(v) for k, v in summary["device_shares"].items()},
            gap_hours=dict(summary["gap_hours"]),
            score_bins=[
                (float(a), float(b), int(c), int(d)) for a, b, c, d in summary["score_bins"]
            ],
        )


# ---------- building it from the full data ----------------------------------------------------------


def kinds(logins: pl.DataFrame, thresholds: Mapping[str, float]) -> dict[str, pl.DataFrame]:
    """Test-side logins grouped by ground truth and response tier, keyed by slug (empty ones left out).

    Args:
        logins: Logins with ``label_ato`` and their ``score``.
        thresholds: Score threshold of each tier; must hold the ``review / block`` and ``step-up``
            tiers.

    Returns:
        The non-empty groups among ``block``, ``2fa`` and ``missed`` (takeovers by tier), and
        ``false-alarm`` and ``ordinary`` (legitimate logins at or above, and below, the step-up
        threshold).
    """
    score, takeover = pl.col("score"), pl.col("label_ato")
    block, step_up = thresholds[TIER_BLOCK], thresholds[TIER_STEP_UP]
    groups = {
        "block": logins.filter(takeover & (score >= block)),
        "2fa": logins.filter(takeover & (score >= step_up) & (score < block)),
        "missed": logins.filter(takeover & (score < step_up)),
        "false-alarm": logins.filter(~takeover & (score >= step_up)),
        "ordinary": logins.filter(~takeover & (score < step_up)),
    }
    return {slug: frame for slug, frame in groups.items() if not frame.is_empty()}


def representative(pool: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """The median-score returning-account login and first login of a group (ties: fixed order).

    Args:
        pool: Logins of one group with ``score``, ``prior_successes`` and ``event_id``.

    Returns:
        ``returning`` and ``first``, each a one-row frame; a scenario with no login in ``pool`` is
        left out.
    """
    first = pl.col("prior_successes") == 0
    picks = {}
    for name, frame in (("returning", pool.filter(~first)), ("first", pool.filter(first))):
        if frame.is_empty():
            continue
        ordered = frame.sort(pl.col("score"), pl.col("event_id").hash(seed=0))
        picks[name] = ordered[(ordered.height - 1) // 2]
    return picks


def _median(frame: pl.DataFrame, column: str) -> float | None:
    value = frame[column].median()
    return float(value) if isinstance(value, int | float) else None


def _gap_hours(logins: pl.DataFrame) -> dict[str, float | None]:
    """Hours since the previous login of a typical test login in each time range."""
    hours, lower, out = pl.col("hours_since_prev_success"), 0.0, {}
    for name, upper in GAPS.items():
        within = logins.filter((hours >= lower) & (hours < upper))
        value = typical_row(within, "hours_since_prev_success", ["hours_since_prev_success"]).get(
            "hours_since_prev_success"
        )
        out[name] = float(value) if isinstance(value, int | float) else None
        lower = upper
    return out


def _score_bins(
    dev: pl.DataFrame, thresholds: Mapping[str, float]
) -> list[tuple[float, float, int, int]]:
    """Development logins by score range: (low, high, logins, takeovers), the tiers split further."""
    edges = [0.0, thresholds[TIER_STEP_UP], thresholds[TIER_BLOCK], 0.2, 0.5, 1.000001]
    bins = []
    for low, high in itertools.pairwise(edges):
        part = dev.filter((pl.col("score") >= low) & (pl.col("score") < high))
        bins.append((low, high, part.height, int(part["label_ato"].sum())))
    return bins


def build_bundle(
    data_dir: Path,
    dataset: str,
    number_inputs: Sequence[str] = NUMBER_INPUTS,
    devices: Sequence[str] = DEVICES,
    detector: Detector | None = None,
) -> DemoBundle:
    """Build the bundle from the full data.

    It uses the test side with its saved scores (issue #7) and the development side with its
    out-of-fold scores (issue #6).

    Args:
        data_dir: Root of the data folders (``05_model_input``, ``06_models``,
            ``07_model_output``...).
        dataset: Dataset name, as in the file names.
        number_inputs: Number inputs whose usual values among takeovers and legitimate logins are
            kept.
        devices: Device types whose typical global share is kept.
        detector: The detector whose thresholds define the tiers; None loads the saved one with
            :meth:`Detector.load`.

    Returns:
        The bundle.
    """
    det = detector or Detector.load(data_dir, dataset)
    config = load_dataset_config(dataset)
    outputs = data_dir / "07_model_output"
    test = (
        load_side(config, data_dir, "test")
        .collect()
        .join(
            pl.read_parquet(outputs / f"{dataset}_test_scores.parquet"),
            on="event_id",
            how="inner",
            maintain_order="left",  # the reference sample depends on the row order
        )
    )
    dev = (
        load_side(config, data_dir, "dev")
        .select("event_id", "label_ato", *number_inputs)
        .collect()
        .join(pl.read_parquet(outputs / f"{dataset}_dev_oof.parquet"), on="event_id", how="inner")
    )
    groups = kinds(test, det.thresholds)
    rows = [
        login.select(pl.lit(slug).alias("kind"), pl.lit(name).alias("scenario"), pl.all())
        for slug, frame in groups.items()
        for name, login in representative(frame).items()
    ]
    ordinary = test.filter(~pl.col("label_ato") & (pl.col("score") < det.thresholds[TIER_STEP_UP]))
    ato, legit = dev.filter(pl.col("label_ato")), dev.filter(~pl.col("label_ato"))
    shares = {
        d: _median(test.filter(pl.col(DEVICE) == d), "global_share_device_type") for d in devices
    }
    return DemoBundle(
        thresholds=dict(det.thresholds),
        counts={slug: frame.height for slug, frame in groups.items()},
        scenarios=pl.concat(rows, how="vertical") if rows else test.clear(),
        reference=ordinary.sample(n=min(REFERENCE_SIZE, ordinary.height), seed=0),
        usual={c: (_median(ato, c), _median(legit, c)) for c in number_inputs},
        device_shares={d: v for d, v in shares.items() if v is not None},
        gap_hours=_gap_hours(test),
        score_bins=_score_bins(dev, det.thresholds),
    )


# ---------- publishing and fetching ------------------------------------------------------------------


def sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB blocks.

    Args:
        path: The file.

    Returns:
        The digest as lowercase hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


PICKLE_LIBRARIES = (
    "scikit-learn",
    "numpy",
)  # the model file must be loaded with these exact versions


def manifest(files: Sequence[Path], base_url: str, release: str) -> dict[str, Any]:
    """The record committed to the repository.

    It holds where the published files are, their SHA-256, and the library versions the model file
    was saved with (the deployment must pin the same ones). The versions are read from the
    exporting environment, so export with the one the model was trained in.

    Args:
        files: The published files.
        base_url: URL of the folder the files are published under.
        release: Name of the release; a plain name, it also names the download folder.

    Returns:
        ``release``, ``base_url``, ``saved_with`` (library name to version) and ``files`` (file
        name to SHA-256), ready to be written as JSON.
    """
    return {
        "release": release,
        "base_url": base_url,
        "saved_with": {name: importlib.metadata.version(name) for name in PICKLE_LIBRARIES},
        "files": {f.name: sha256(f) for f in sorted(files, key=lambda f: f.name)},
    }


def _safe(name: object) -> str:
    text = str(name)
    if not SAFE_NAME.fullmatch(text) or text in {".", ".."}:
        raise ValueError(f"unsafe name in the bundle record: {text!r}")
    return text


def _transient(error: Exception) -> bool:
    """A failure worth retrying: a server error, rate limiting, a timeout or a dropped connection."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code in TRANSIENT_HTTP
    return isinstance(error, urllib.error.URLError | TimeoutError | ConnectionError)


def _read(url: str) -> bytes:
    """The body of ``url``, retrying temporary failures with growing waits (``RETRY_WAITS``)."""
    for wait in (*RETRY_WAITS, None):
        try:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:  # noqa: S310 — scheme checked
                data: bytes = response.read(MAX_FILE_BYTES + 1)
                return data
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if wait is None or not _transient(error):
                raise
            time.sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover


def _download(url: str, target: Path) -> None:
    """Write ``url`` to ``target``, a new file (never follows or reuses an existing one)."""
    if urllib.parse.urlparse(url).scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"refusing to download from {url}")
    data = _read(url)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"{url}: larger than {MAX_FILE_BYTES} bytes; not ours")
    target.unlink(missing_ok=True)  # a stale partial file (unlink never follows a link)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(target, flags, 0o600), "wb") as handle:
        handle.write(data)


def fetch(record: Mapping[str, Any], cache: Path | None = None) -> Path:
    """Download the files of a published bundle and check each SHA-256 before anything is loaded.

    Without ``cache``, the files go to a new private folder (``tempfile.mkdtemp``: readable by this user only),
    so no other user of the machine can swap a file between the check and the load; the model file is a
    pickle, so it must never be loaded unverified. With ``cache`` (a folder the caller owns), a file already
    there is reused only if its SHA-256 still matches. File and release names must be plain names.

    Args:
        record: The manifest (see :func:`manifest`): ``release``, ``base_url`` and ``files``.
        cache: Folder to keep downloads in, under a subfolder per release; None means a new private
            temporary folder.

    Returns:
        The folder holding the verified files.

    Raises:
        ValueError: If a file or release name is not a plain name, a URL scheme is not allowed, a
            file is too large, or a file does not match its SHA-256.
        urllib.error.URLError, TimeoutError, ConnectionError: If a download fails and the failure
            is not temporary (for example HTTP 404), or it still fails after the retries.
    """
    release = _safe(record["release"])
    folder = (cache / release) if cache else Path(tempfile.mkdtemp(prefix=f"ato-demo-{release}-"))
    folder.mkdir(parents=True, exist_ok=True)
    for raw_name, expected in record["files"].items():
        name = _safe(raw_name)
        target = folder / name
        if target.is_file() and not target.is_symlink() and sha256(target) == expected:
            continue
        partial = folder / f"{name}.part"
        _download(urllib.parse.urljoin(str(record["base_url"]), name), partial)
        if sha256(partial) != expected:
            partial.unlink()
            raise ValueError(f"{name}: SHA-256 does not match the repository's record; not loaded")
        partial.replace(target)
    return folder


def main(argv: list[str] | None = None) -> int:
    """Export the bundle and the model files for publishing, and write the manifest to commit.

    Args:
        argv: Command-line arguments (``--dataset``, ``--data-dir``, ``--out``, ``--release``,
            ``--base-url``, ``--manifest``); None reads them from ``sys.argv``.

    Returns:
        The exit status, 0 on success.
    """
    import argparse  # noqa: PLC0415
    import shutil  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="rba")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="where to write the record to commit"
    )
    args = parser.parse_args(argv)
    bundle = build_bundle(args.data_dir, args.dataset, NUMBER_INPUTS, DEVICES)
    files = bundle.write(args.out)
    for suffix in ("joblib", "json"):
        source = args.data_dir / "06_models" / f"{args.dataset}_model.{suffix}"
        files.append(Path(shutil.copy2(source, args.out / source.name)))
    record = json.dumps(manifest(files, args.base_url, args.release), indent=2) + "\n"
    args.manifest.write_text(record)  # only after everything above succeeded
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
