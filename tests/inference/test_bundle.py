"""The demo bundle: built from the data, written and read back unchanged, fetched only with matching SHA-256."""

import io
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from inference import bundle as bundle_module
from inference.bundle import NUMBER_INPUTS, DemoBundle, build_bundle, fetch, manifest, sha256

APP = Path(__file__).parents[2] / "notebooks" / "7-deploy" / "app.py"


@pytest.fixture(scope="module")
def bundle_dir(data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The bundle of the toy data, with the model files, as the export writes it."""
    folder = tmp_path_factory.mktemp("bundle")
    build_bundle(data_dir, "rba").write(folder)
    for suffix in ("joblib", "json"):
        shutil.copy2(data_dir / "06_models" / f"rba_model.{suffix}", folder)
    return folder


def test_bundle_reads_back_what_was_written(data_dir: Path, bundle_dir: Path) -> None:
    """A bundle read from disk equals the one built from the data, with scenarios for each kind."""
    built, read = build_bundle(data_dir, "rba"), DemoBundle.read(bundle_dir)
    assert list(read.thresholds.items()) == list(built.thresholds.items()), "strictest tier first"
    assert read.counts == built.counts
    assert read.scenarios.equals(built.scenarios)
    assert read.reference.equals(built.reference)
    assert read.usual == built.usual
    assert set(read.usual) == set(NUMBER_INPUTS)
    assert read.score_bins == built.score_bins
    assert read.device_shares == built.device_shares
    assert read.gap_hours == built.gap_hours
    picks = read.picks()
    assert set(picks) == set(read.counts), "a scenario for every kind of outcome"
    assert all(login.height == 1 for scenarios in picks.values() for login in scenarios.values())


def _app(monkeypatch: pytest.MonkeyPatch, **env: str) -> AppTest:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    st.cache_resource.clear()
    st.cache_data.clear()
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception, app.exception
    return app


def _pages(app: AppTest) -> dict[tuple[str, str], tuple[str, pd.DataFrame]]:
    """Score and table of every kind of outcome and scenario the page offers."""
    pages = {}
    for key in [b.key for b in app.button if b.key and b.key.startswith("kind-")]:
        app.button(key=key).click().run()
        for scenario in [b.key for b in app.button if b.key and b.key.startswith("scenario-")]:
            app.button(key=scenario).click().run()
            app.button(key="login").click().run()
            assert not app.exception, app.exception
            table: pd.DataFrame = app.table[0].value
            pages[(key, str(scenario))] = (app.metric[0].value, table)
    return pages


def test_the_demo_runs_from_the_bundle_alone(
    data_dir: Path, bundle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the full data, the bundle gives the same page for every kind and scenario."""
    from_data = _pages(_app(monkeypatch, ATO_DATA_DIR=str(data_dir)))
    app = _app(
        monkeypatch,
        ATO_DATA_DIR=str(tmp_path),
        ATO_BUNDLE_DIR=str(bundle_dir),
        ATO_DEMO_OFFLINE="1",
    )
    sliders = [s.key for s in app.select_slider if s.key != "in-gap"]
    assert sliders == [f"in-{c}" for c in NUMBER_INPUTS], "the export covers every number input"
    from_bundle = _pages(app)
    assert from_bundle.keys() == from_data.keys()
    for page, (score, table) in from_data.items():
        assert from_bundle[page][0] == score, page
        assert from_bundle[page][1].equals(table), page


def test_fetch_downloads_and_checks_every_file(bundle_dir: Path, tmp_path: Path) -> None:
    """``fetch`` downloads every listed file and replaces a cached file whose SHA-256 changed."""
    files = sorted(bundle_dir.iterdir())
    record = manifest(files, bundle_dir.as_uri() + "/", "test")
    folder = fetch(record, cache=tmp_path)
    assert sorted(p.name for p in folder.iterdir()) == sorted(f.name for f in files)
    assert all(sha256(folder / f.name) == sha256(f) for f in files)
    (folder / "summary.json").write_text("{}")  # a cached file that changed is downloaded again
    fetch(record, cache=tmp_path)
    assert sha256(folder / "summary.json") == sha256(bundle_dir / "summary.json")


def test_fetch_never_keeps_a_file_whose_sha256_differs(bundle_dir: Path, tmp_path: Path) -> None:
    """``fetch`` rejects a file whose SHA-256 differs and leaves neither it nor a partial file."""
    record = manifest(sorted(bundle_dir.iterdir()), bundle_dir.as_uri() + "/", "test")
    record["files"]["rba_model.joblib"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        fetch(record, cache=tmp_path)
    assert not (tmp_path / "test" / "rba_model.joblib").exists(), "the model is never left to load"
    assert not list((tmp_path / "test").glob("*.part")), "no partial file left behind"


@pytest.mark.parametrize(
    ("field", "value"), [("release", "../outside"), ("file", "../evil.txt"), ("file", "//host/x")]
)
def test_fetch_refuses_names_that_are_not_plain(tmp_path: Path, field: str, value: str) -> None:
    """``fetch`` refuses a release or file name that could point outside the cache folder."""
    record = {
        "release": "ok",
        "base_url": tmp_path.as_uri() + "/",
        "files": {"summary.json": "0" * 64},
    }
    if field == "release":
        record["release"] = value
    else:
        record["files"] = {value: "0" * 64}
    with pytest.raises(ValueError, match="unsafe name"):
        fetch(record, cache=tmp_path / "cache")


def test_fetch_without_a_cache_uses_a_new_private_folder(bundle_dir: Path) -> None:
    """Without a cache, ``fetch`` downloads into a new folder readable only by the current user."""
    record = manifest(sorted(bundle_dir.iterdir()), bundle_dir.as_uri() + "/", "test")
    first, second = fetch(record), fetch(record)
    assert first != second, "a new folder each time"
    assert first.stat().st_mode & 0o077 == 0, "readable by this user only"
    shutil.rmtree(first)
    shutil.rmtree(second)


def test_fetch_refuses_plain_http(tmp_path: Path) -> None:
    """``fetch`` refuses a base URL that uses plain HTTP."""
    record = {
        "release": "x",
        "base_url": "http://example.com/",
        "files": {"summary.json": "0" * 64},
    }
    with pytest.raises(ValueError, match="refusing"):
        fetch(record, cache=tmp_path)


def test_the_committed_record_lists_the_bundle_and_the_model() -> None:
    """The committed ``bundle.json`` points to the repository releases and hashes all five files."""
    record_file = APP.with_name("bundle.json")
    if not record_file.exists():
        pytest.skip("no bundle published yet")
    record = json.loads(record_file.read_text())
    assert record["base_url"].startswith(
        "https://github.com/modsiw-mona/anomaly_detection_ato/releases/"
    )
    assert set(record["files"]) == {
        "rba_model.joblib",
        "rba_model.json",
        "reference.parquet",
        "scenarios.parquet",
        "summary.json",
    }
    assert all(len(digest) == 64 for digest in record["files"].values())  # noqa: PLR2004 — SHA-256 hex


def test_the_deployment_pins_the_libraries_the_model_was_saved_with() -> None:
    """The deployment requirements pin the library versions the model was saved with.

    Streamlit Cloud installs ``requirements.txt``; the model is a pickle, so scikit-learn and numpy
    must be the versions it was saved with, whatever Dependabot later does to ``uv.lock``.
    """
    record_file = APP.with_name("bundle.json")
    if not record_file.exists():
        pytest.skip("no bundle published yet")
    saved_with = json.loads(record_file.read_text())["saved_with"]
    pins = dict(
        line.split(" ")[0].split("==")
        for line in APP.with_name("requirements.txt").read_text().splitlines()
        if "==" in line and not line.startswith("#")
    )
    assert {name: pins[name] for name in saved_with} == saved_with


class _Flaky:
    """Stands in for ``urlopen``: fails with the given HTTP codes first, then serves the local file."""

    def __init__(self, codes: list[int]) -> None:
        self.codes, self.calls = codes, 0

    def __call__(self, url: str, timeout: float) -> io.BytesIO:
        self.calls += 1
        if self.calls <= len(self.codes):
            raise urllib.error.HTTPError(url, self.codes[self.calls - 1], "error", {}, None)  # type: ignore[arg-type]
        return io.BytesIO(
            Path(urllib.request.url2pathname(url.removeprefix("file://"))).read_bytes()
        )


def _one_file_record(bundle_dir: Path) -> dict[str, object]:
    return manifest([bundle_dir / "summary.json"], bundle_dir.as_uri() + "/", "test")


def test_fetch_retries_temporary_server_errors(
    bundle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fetch`` retries temporary server errors, waiting longer before each new attempt."""
    flaky, waits = _Flaky([504, 503]), list[float]()
    monkeypatch.setattr("urllib.request.urlopen", flaky)
    monkeypatch.setattr("time.sleep", waits.append)
    folder = fetch(_one_file_record(bundle_dir), cache=tmp_path)
    assert sha256(folder / "summary.json") == sha256(bundle_dir / "summary.json")
    assert flaky.calls == len(flaky.codes) + 1, "two failures, then the file"
    assert waits == list(bundle_module.RETRY_WAITS[:2]), "growing waits between attempts"


def test_fetch_gives_up_after_the_last_retry(
    bundle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fetch`` raises the server error once every retry has failed."""
    flaky = _Flaky([504] * 10)
    monkeypatch.setattr("urllib.request.urlopen", flaky)
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(urllib.error.HTTPError, match="504"):
        fetch(_one_file_record(bundle_dir), cache=tmp_path)
    assert flaky.calls == len(bundle_module.RETRY_WAITS) + 1


def test_fetch_does_not_retry_a_missing_file(
    bundle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fetch`` raises a 404 at once, without retrying."""
    flaky = _Flaky([404])
    monkeypatch.setattr("urllib.request.urlopen", flaky)
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(urllib.error.HTTPError, match="404"):
        fetch(_one_file_record(bundle_dir), cache=tmp_path)
    assert flaky.calls == 1
