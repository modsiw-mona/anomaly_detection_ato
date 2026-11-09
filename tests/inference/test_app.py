"""The Streamlit demo runs end to end on a toy data directory (no real data needed)."""

import json
from pathlib import Path

import joblib
import pandas as pd
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
NETWORK_SHARE, ACCOUNTS = "in-global_share_asn", "in-asn_prior_users"
ATTACK_SHARE = "in-asn_prior_attack_share"
NUMBER_INPUTS = 6  # sliders, each with where takeovers and legitimate logins usually are


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A data directory shaped like the real one, for dataset ``rba``, built from toy events."""
    root = tmp_path_factory.mktemp("data")
    frame = build_features(_events(_random_events(1_500, seed=5)), _config()).collect()
    # takeovers = logins from an attack IP, so the "IP on an attack list" switch must move the score
    frame = frame.with_columns(pl.col("attack_ip").alias("label_ato"))
    columns = feature_columns((*DEFAULT_GROUPS, "campaign", "travel"))
    pipeline = train_supervised("random forest", frame, columns, negative_share=0.5)
    for folder in ("04_feature", "05_model_input", "06_models", "07_model_output"):
        (root / folder).mkdir()
    frame.write_parquet(root / "04_feature" / "rba_features.parquet")
    users = frame.select(pl.col("user_id").unique().sort())
    side = pl.when(pl.int_range(pl.len()) % 2 == 0).then(pl.lit("test")).otherwise(pl.lit("dev"))
    users.with_columns(side.alias("split")).write_parquet(
        root / "05_model_input" / "rba_split.parquet"
    )  # half the users on each side
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


def _table(app: AppTest) -> pd.DataFrame:
    table: pd.DataFrame = app.table[0].value  # "How the score is built"
    return table


def test_demo_scores_a_login_and_reacts_to_its_inputs(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _open(data_dir, monkeypatch)
    assert app.metric[0].label == "Takeover score"
    assert not app.metric[0].delta, "the real login, before any change, is not 'changed'"
    intro = " ".join(m.value for m in app.markdown)
    assert "How to use it" in intro, "the page explains itself"
    assert "github.com/modsiw-mona/anomaly_detection_ato" in intro, "and links its code"
    assert _choose(app, "returning")
    before = app.metric[0].value
    attack = app.toggle(key="in-attack_ip")
    attack.set_value(not attack.value).run()
    assert not app.exception, app.exception
    assert app.metric[0].value != before, "the change reaches the model"
    assert app.metric[0].delta.endswith(f"from the real login's {before}")
    app.button[-1].click().run()  # back to the real login
    assert app.metric[0].value == before
    assert not app.metric[0].delta


def test_a_number_input_is_set_directly(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A slider sets exactly the input it names; the table shows the old and the new value."""
    app = _open(data_dir, monkeypatch)
    share = app.select_slider(key=ATTACK_SHARE)
    target = 1.0 if share.value != 1.0 else 0.0
    share.set_value(target).run()
    assert not app.exception, app.exception
    table = _table(app)
    row = table[table["input"].str.startswith("Share of the network's earlier attempts")].iloc[0]
    assert row["with your changes"] == ("100%" if target == 1.0 else "0%")
    assert row["real login"] != row["with your changes"]


def test_a_never_seen_network_has_no_accounts(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _open(data_dir, monkeypatch)
    real_accounts = app.select_slider(key=ACCOUNTS).value
    app.select_slider(key=NETWORK_SHARE).set_value(0).run()
    assert not app.exception, app.exception
    accounts = app.select_slider(key=ACCOUNTS)
    assert accounts.value == 0
    assert accounts.disabled
    app.select_slider(key=NETWORK_SHARE).set_value(0.05).run()
    accounts = app.select_slider(key=ACCOUNTS)
    assert not accounts.disabled
    assert accounts.value == (real_accounts or 5), "back to the viewer's value, and never 0"
    assert 0 not in accounts.options


def test_the_table_keeps_the_order_of_the_controls(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _open(data_dir, monkeypatch)
    assert _choose(app, "returning")
    order = list(_table(app)["input"])
    assert order[0].startswith("Typical ordinary login")
    assert order[-1] == "Takeover score"
    attack = app.toggle(key="in-attack_ip")
    attack.set_value(not attack.value).run()
    assert list(_table(app)["input"]) == order, "rows do not jump when a value changes"
    total = _table(app).iloc[-1]
    assert total["effect"][0] in "↑↓", "the last row says whether the score went up or down"
    assert total["with your changes"].split()[0] == app.metric[0].value


def test_contributions_add_up_to_the_score(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _open(data_dir, monkeypatch)
    parts = _table(app)["contribution"].iloc[:-1].astype(float).sum()
    assert parts == pytest.approx(float(app.metric[0].value), abs=0.01)


def test_implied_signal_follows_its_cause(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _open(data_dir, monkeypatch)
    assert _choose(app, "returning")
    app.toggle(key="in-new_asn").set_value(True).run()
    assert not app.exception, app.exception
    new_ip = app.toggle(key="in-new_ip")
    assert new_ip.value, "a new network always means a new IP address"
    assert new_ip.disabled


def test_undoing_a_cause_restores_the_implied_signal(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _open(data_dir, monkeypatch)
    assert _choose(app, "returning")
    before = app.toggle(key="in-new_ip").value
    real_asn = app.toggle(key="in-new_asn").value
    app.toggle(key="in-new_asn").set_value(not real_asn).run()
    app.toggle(key="in-new_asn").set_value(real_asn).run()
    assert not app.exception, app.exception
    assert app.toggle(key="in-new_ip").value == before
    assert not app.metric[0].delta, "switching a signal on and off again is no change"


def test_travel_follows_the_previous_login(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _open(data_dir, monkeypatch)
    assert _choose(app, "returning")
    app.select_slider(key="in-gap").set_value("under an hour").run()
    app.toggle(key="in-country_change_1h").set_value(True).run()  # the viewer's choice
    app.select_slider(key="in-gap").set_value("over a month").run()  # a rule turns it off
    assert not app.exception, app.exception
    travel = app.toggle(key="in-country_change_1h")
    assert travel.disabled, "no country change within an hour when the previous login is old"
    assert not travel.value
    app.select_slider(key="in-gap").set_value("under an hour").run()  # the rule lets go
    assert app.toggle(key="in-country_change_1h").value, "back to what the viewer had chosen"


def test_device_type_reaches_the_model(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _open(data_dir, monkeypatch)
    device = app.radio(key="in-device_type")
    real = device.value
    other = next(d for d in ("desktop", "mobile", "tablet") if d != real)
    device.set_value(other).run()
    assert not app.exception, app.exception
    table = _table(app)
    row = table[table["input"] == "Device type"].iloc[0]
    assert (row["real login"], row["with your changes"]) == (real, other)


def test_first_login_greys_out_history_signals(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _open(data_dir, monkeypatch)
    assert _choose(app, "first")
    assert app.toggle(key="in-new_country").disabled, "nothing to compare with yet"
    assert not [s for s in app.select_slider if s.key == "in-gap"], "no previous login"


def test_technical_terms_are_explained(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _open(data_dir, monkeypatch)
    assert _choose(app, "returning")
    assert all(s.help for s in app.select_slider if s.key != "in-gap"), "every number input"
    assert "ASN (autonomous system number)" in app.select_slider(key=NETWORK_SHARE).help
    for key in ("attack_ip", "new_asn", "new_user_agent"):
        assert app.toggle(key=f"in-{key}").help, key
    assert not app.toggle(key="in-new_city").help, "a plain switch needs no definition"
    usual = [c.value for c in app.caption if c.value.startswith("Takeovers usually")]
    assert len(usual) == NUMBER_INPUTS, "each number input says where its usual values are"


def test_the_score_is_put_in_context(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _open(data_dir, monkeypatch)
    assert any(m.value.startswith("In the dataset,") for m in app.markdown)


def test_demo_explains_missing_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATO_DATA_DIR", str(tmp_path))
    st.cache_resource.clear()  # the model loaded by the other test is cached per process
    st.cache_data.clear()
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert app.error, "a clear message instead of a crash"
