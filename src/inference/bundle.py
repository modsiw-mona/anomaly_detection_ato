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
import tempfile
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
        """Scenario logins by kind and scenario name, each a one-row frame, in the stored order."""
        result: dict[str, dict[str, pl.DataFrame]] = {}
        for i in range(self.scenarios.height):
            row = self.scenarios[i]
            kind, name = row["kind"][0], row["scenario"][0]
            result.setdefault(kind, {})[name] = row.drop("kind", "scenario")
        return result

    def write(self, folder: Path) -> list[Path]:
        """Write the bundle's three files into ``folder``."""
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
        """Read a bundle written by :meth:`write`."""
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
    """Test-side logins grouped by ground truth and response tier, keyed by slug (empty ones left out)."""
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
    """The median-score returning-account login and first login of a group (ties: fixed order)."""
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
    """Build the bundle from the full data: the test side with its saved scores (issue #7) and the
    development side with its out-of-fold scores (issue #6)."""
    det = detector or Detector.load(data_dir, dataset)
    config = load_dataset_config(dataset)
    outputs = data_dir / "07_model_output"
    test = (
        load_side(config, data_dir, "test")
        .collect()
        .join(
            pl.read_parquet(outputs / f"{dataset}_test_scores.parquet"), on="event_id", how="inner"
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
    """The record committed to the repository: where the published files are, their SHA-256, and the library
    versions the model file was saved with (the deployment must pin the same ones)."""
    return {
        "release": release,
        "base_url": base_url,
        "saved_with": {name: importlib.metadata.version(name) for name in PICKLE_LIBRARIES},
        "files": {f.name: sha256(f) for f in sorted(files, key=lambda f: f.name)},
    }


def fetch(record: Mapping[str, Any], cache: Path | None = None) -> Path:
    """Download the files of a published bundle into a cache folder, checking each SHA-256 first.

    A file already in the cache is used only if its SHA-256 still matches. A mismatch raises before the file
    is ever loaded (the model file is a pickle, so it must never be loaded unverified).
    """
    folder = (cache or Path(tempfile.gettempdir()) / "ato-demo") / str(record["release"])
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in record["files"].items():
        target = folder / name
        if target.exists() and sha256(target) == expected:
            continue
        url = urllib.parse.urljoin(str(record["base_url"]), urllib.parse.quote(name))
        if urllib.parse.urlparse(url).scheme not in ALLOWED_SCHEMES:
            raise ValueError(f"refusing to download from {url}")
        partial = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:  # noqa: S310 — scheme checked
            partial.write_bytes(response.read())
        if sha256(partial) != expected:
            partial.unlink()
            raise ValueError(f"{name}: SHA-256 does not match the repository's record; not loaded")
        partial.replace(target)
    return folder


def main(argv: list[str] | None = None) -> int:
    """Export the bundle and the model files for publishing, and print the manifest to commit."""
    import argparse  # noqa: PLC0415
    import shutil  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="rba")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args(argv)
    bundle = build_bundle(args.data_dir, args.dataset, NUMBER_INPUTS, DEVICES)
    files = bundle.write(args.out)
    for suffix in ("joblib", "json"):
        source = args.data_dir / "06_models" / f"{args.dataset}_model.{suffix}"
        files.append(Path(shutil.copy2(source, args.out / source.name)))
    print(json.dumps(manifest(files, args.base_url, args.release), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
