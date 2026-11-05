"""The Streamlit demo runs end to end on a toy data directory (no real data needed)."""

import json
from pathlib import Path

import joblib
import polars as pl
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from data.features import DEFAULT_GROUPS, build_features, feature_columns
from model.preprocessing import to_model_frame
from model.selection import train_supervised
from tests.data.test_features import _config, _events, _random_events

APP = Path(__file__).parents[2] / "notebooks" / "7-deploy" / "app.py"
SCENARIO_LABELS = {"returning": "Returning account", "first": "First login of a new account"}


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A data directory shaped like the real one, for dataset ``rba``, built from toy events."""
    root = tmp_path_factory.mktemp("data")
    frame = build_features(_events(_random_events(1_500, seed=5)), _config()).collect()
    # takeovers = logins from an attack IP, so the "IP on an attack list" what-if must move the score
    frame = frame.with_columns(pl.col("attack_ip").alias("label_ato"))
    columns = feature_columns((*DEFAULT_GROUPS, "campaign", "travel"))
    pipeline = train_supervised("random forest", frame, columns, negative_share=0.5)
    for folder in ("04_feature", "05_model_input", "06_models", "07_model_output"):
        (root / folder).mkdir()
    frame.write_parquet(root / "04_feature" / "rba_features.parquet")
    users = frame.select(pl.col("user_id").unique()).with_columns(pl.lit("test").alias("split"))
    users.write_parquet(root / "05_model_input" / "rba_split.parquet")
    joblib.dump(pipeline, root / "06_models" / "rba_model.joblib")
    (root / "06_models" / "rba_model.json").write_text(json.dumps({"columns": columns}))
    scores = pipeline.predict_proba(to_model_frame(frame, columns))[:, 1]
    for name in ("dev_oof", "test_scores"):
        pl.DataFrame({"event_id": frame["event_id"], "score": scores}).write_parquet(
            root / "07_model_output" / f"rba_{name}.parquet"
        )
    return root


def _open(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("ATO_DATA_DIR", str(data_dir))
    st.cache_resource.clear()
    st.cache_data.clear()
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception, app.exception
    return app


def _choose(app: AppTest, scenario: str) -> bool:
    """Open the first kind of outcome that offers ``scenario``; False if none does."""
    for key in [b.key for b in app.button if b.key and b.key.startswith("kind-")]:
        app.button(key=key).click().run()
        radio = app.radio[0]
        if SCENARIO_LABELS[scenario] in radio.options:
            radio.set_value(scenario).run()
            assert not app.exception, app.exception
            return True
    return False


def test_demo_scores_a_scenario_and_reacts_to_signals(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _open(data_dir, monkeypatch)
    assert app.metric[0].label == "Takeover score"
    assert not app.metric[0].delta, "the real login, before any change, is not 'changed'"
    assert not app.number_input, "scenarios are named, not numbered"
    assert _choose(app, "returning")
    labels = {t.label for t in app.toggle}
    assert {
        "New country", "New network (ASN)", "New IP address", "New exact user agent",
        "IP address on an attack list", "Country changed within an hour of the previous login",
        "Rare network (ASN): under 0.1% of all earlier logins", "Rare country: under 1% of all earlier logins",
    } <= labels  # fmt: skip
    assert "First successful login of this account" not in labels, "a fact, not a switch"
    before = app.metric[0].value
    attack = app.toggle(key="signal-attack_ip")
    attack.set_value(not attack.value).run()
    assert not app.exception, app.exception
    assert app.metric[0].value != before, "the change reaches the model"
    assert not app.metric[0].delta.lstrip("+-").startswith("0.0000")
    assert app.metric[0].delta.endswith(f"from the real login's {before}")
    app.button[-1].click().run()  # back to the real login
    assert app.metric[0].value == before
    assert not app.metric[0].delta


def test_implied_signal_follows_its_cause(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _open(data_dir, monkeypatch)
    assert _choose(app, "returning")
    app.toggle(key="signal-new_asn").set_value(True).run()
    assert not app.exception, app.exception
    new_ip = app.toggle(key="signal-new_ip")
    assert new_ip.value, "a new network always means a new IP address"
    assert new_ip.disabled


def test_where_from_switch_reaches_the_model(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _open(data_dir, monkeypatch)
    rare = app.toggle(key="signal-rare_network")
    rare.set_value(not rare.value).run()
    assert not app.exception, app.exception
    moved = bool(app.metric[0].delta)
    unmoved = any("do not move the score" in c.value for c in app.caption)
    assert moved or unmoved, "the switch changes the login's network inputs"
    why = app.table[-1].value  # the right column (Why) is drawn after the login details
    network_inputs = {
        "Share of all earlier logins from this network (ASN)",
        "Share of all earlier logins from this IP",
        "Accounts seen on this network before",
        "Accounts seen on this IP before",
        "Attempts from this network in the previous 24 hours",
        "Failed attempts from this network in the previous 24 hours",
        "Share of this network's earlier attempts from attack IPs",
        "Share of this network's earlier attempts that failed",
        "Share of this IP's earlier attempts from attack IPs",
        "Share of this IP's earlier attempts that failed",
    }
    assert why.iloc[0]["input"] in network_inputs, "changed network inputs are listed first in Why"


def test_undoing_a_cause_restores_the_implied_signal(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _open(data_dir, monkeypatch)
    assert _choose(app, "returning")
    before = app.toggle(key="signal-new_ip").value
    real_asn = app.toggle(key="signal-new_asn").value
    app.toggle(key="signal-new_asn").set_value(not real_asn).run()
    app.toggle(key="signal-new_asn").set_value(real_asn).run()
    assert not app.exception, app.exception
    assert app.toggle(key="signal-new_ip").value == before
    assert not app.metric[0].delta, "switching a signal on and off again is no change"


def test_changed_signal_is_listed_first(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _open(data_dir, monkeypatch)
    assert _choose(app, "returning")
    app.select_slider(key="signal-gap").set_value("under an hour").run()  # travel needs this
    travel = app.toggle(key="signal-country_change_1h")
    assert not travel.disabled
    travel.set_value(not travel.value).run()
    assert not app.exception, app.exception
    top = set(
        app.table[-1].value["input"].head(2)
    )  # the two changed inputs: travel and the time gap
    assert top == {
        "Country changed within an hour of the previous login",
        "Hours since the account's previous login",
    }


def test_travel_switch_follows_the_previous_login(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _open(data_dir, monkeypatch)
    assert _choose(app, "returning")
    app.select_slider(key="signal-gap").set_value("under an hour").run()
    app.toggle(key="signal-country_change_1h").set_value(True).run()
    app.select_slider(key="signal-gap").set_value("over a month").run()
    assert not app.exception, app.exception
    travel = app.toggle(key="signal-country_change_1h")
    assert travel.disabled, (
        "no country change within an hour when the previous login is a month old"
    )
    assert not travel.value


def test_a_released_switch_gets_back_the_viewers_value(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _open(data_dir, monkeypatch)
    assert _choose(app, "returning")
    app.select_slider(key="signal-gap").set_value("under an hour").run()
    app.toggle(key="signal-country_change_1h").set_value(True).run()  # the viewer's choice
    app.select_slider(key="signal-gap").set_value("over a month").run()  # a rule turns it off
    app.select_slider(key="signal-gap").set_value(
        "under an hour"
    ).run()  # the rule no longer applies
    assert not app.exception, app.exception
    assert app.toggle(key="signal-country_change_1h").value, "back to what the viewer had chosen"


def test_a_switch_the_viewer_turns_on_stays_in_their_hands(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _open(data_dir, monkeypatch)
    assert _choose(app, "returning")
    app.select_slider(key="signal-gap").set_value("under an hour").run()
    country = app.toggle(key="signal-new_country")
    if country.disabled:  # fixed by a rule for this login: nothing to check
        return
    country.set_value(not country.value).run()
    assert not app.exception, app.exception
    assert not app.toggle(key="signal-new_country").disabled, (
        "a rule must not lock the viewer's own choice"
    )


def test_device_type_reaches_the_model(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _open(data_dir, monkeypatch)
    device = app.radio(key="signal-device_type")
    real = device.value
    other = next(d for d in ("desktop", "mobile", "tablet") if d != real)
    device.set_value(other).run()
    assert not app.exception, app.exception
    client_inputs = (
        "Device type", "Share of all earlier logins with this", "Accounts seen with this exact user agent",
    )  # fmt: skip
    assert app.table[-1].value.iloc[0]["input"].startswith(client_inputs)
    detail = app.table[0].value  # the login in detail: real and with your changes
    row = detail[detail["field"] == "Device type"].iloc[0]
    assert (row["real login"], row["with your changes"]) == (real, other)


def test_first_login_greys_out_history_signals(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _open(data_dir, monkeypatch)
    assert _choose(app, "first")
    assert app.toggle(key="signal-new_country").disabled, "nothing to compare with yet"
    assert not app.select_slider, "no previous login, so no time since it"


def test_every_control_has_a_definition(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _open(data_dir, monkeypatch)
    assert _choose(app, "returning")
    controls = [
        *app.toggle,
        app.radio(key="signal-device_type"),
        app.select_slider(key="signal-gap"),
    ]
    assert all(c.help for c in controls), [c.label for c in controls if not c.help]


def test_demo_explains_missing_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATO_DATA_DIR", str(tmp_path))
    st.cache_resource.clear()  # the model loaded by the other test is cached per process
    st.cache_data.clear()
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert app.error, "a clear message instead of a crash"
