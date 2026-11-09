"""The Streamlit demo runs end to end on a toy data directory (no real data needed)."""

import re
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP = Path(__file__).parents[2] / "notebooks" / "7-deploy" / "app.py"
NETWORK_SHARE, ACCOUNTS = "in-global_share_asn", "in-asn_prior_users"
ATTACK_SHARE = "in-asn_prior_attack_share"
NUMBER_INPUTS = 6  # sliders, each with where takeovers and legitimate logins usually are


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
        if f"scenario-{scenario}" in _scenario_keys(app):
            app.button(key=f"scenario-{scenario}").click().run()
            assert not app.exception, app.exception
            return True
    return False


def _scenario_keys(app: AppTest) -> list[str]:
    return [b.key for b in app.button if b.key and b.key.startswith("scenario-")]


def _log_in(app: AppTest) -> AppTest:
    """Press Log in: the response and what the model saw appear for the current attempt."""
    app.button(key="login").click().run()
    assert not app.exception, app.exception
    return app


def _table(app: AppTest) -> pd.DataFrame:
    table: pd.DataFrame = _log_in(app).table[0].value  # "How the score is built"
    return table


def _score(app: AppTest) -> str:
    return str(_log_in(app).metric[0].value)


def test_demo_scores_a_login_and_reacts_to_its_inputs(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _open(data_dir, monkeypatch)
    intro = " ".join(m.value for m in app.markdown)
    assert "How to use it" in intro, "the page explains itself"
    assert "github.com/modsiw-mona/anomaly_detection_ato" in intro, "and links its code"
    assert not app.metric, "the score stays hidden until Log in"
    _log_in(app)
    assert app.metric[0].label == "Takeover score"
    assert not app.metric[0].delta, "the real login, before any change, is not 'changed'"
    assert _choose(app, "returning")
    before = _score(app)
    attack = app.toggle(key="in-attack_ip")
    attack.set_value(not attack.value).run()
    assert not app.exception, app.exception
    assert _score(app) != before, "the change reaches the model"
    assert app.metric[0].delta.endswith(f"from the real login's {before}")
    app.button(key="back").click().run()
    assert _score(app) == before
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
    assert "0" not in accounts.options, "options are text labels"


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
    assert total["with your changes"].split()[0] == _score(app)


def test_contributions_add_up_to_the_score(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _open(data_dir, monkeypatch)
    parts = _table(app)["contribution"].iloc[:-1].astype(float).sum()
    assert parts == pytest.approx(float(_score(app)), abs=0.01)


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
    _log_in(app)
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
    assert "ASN, autonomous system number" in app.select_slider(key=NETWORK_SHARE).help
    for key in ("attack_ip", "new_asn", "new_user_agent"):
        assert app.toggle(key=f"in-{key}").help, key
    assert not app.toggle(key="in-new_city").help, "a plain switch needs no definition"
    sliders = [s for s in app.select_slider if s.key != "in-gap"]
    assert len(sliders) == NUMBER_INPUTS
    assert all("Takeovers usually:" in s.help for s in sliders), "usual values, in the tooltip"
    assert not [c for c in app.caption if "Takeovers usually" in c.value], "and not on the page"


def test_the_score_is_put_in_context(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _log_in(_open(data_dir, monkeypatch))
    assert any(m.value.startswith("In the dataset,") for m in app.markdown)
    assert any("threshold" in m.value for m in app.markdown if "Response:" in m.value), (
        "where it falls"
    )


def test_demo_explains_missing_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ATO_DEMO_OFFLINE", "1")  # no download of the published bundle
    st.cache_resource.clear()  # the model loaded by the other test is cached per process
    st.cache_data.clear()
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert app.error, "a clear message instead of a crash"


SCREENS = {"block": "Sign-in blocked.", "2fa": "Sign-in on hold.", "allow": "Welcome back."}


def _screen(app: AppTest) -> str:
    """Which response screen the page shows (the person's view, above what the model saw)."""
    shown = " ".join(m.value for m in app.markdown if "border-left" in m.value)
    found = [tier for tier, text in SCREENS.items() if text in shown]
    assert len(found) == 1, shown
    return found[0]


def test_log_in_shows_the_screen_of_each_tier(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The response the person sees always matches the tier of the score the model saw."""
    app = _open(data_dir, monkeypatch)
    seen = set()
    for key in [b.key for b in app.button if b.key and b.key.startswith("kind-")]:
        app.button(key=key).click().run()
        for scenario in _scenario_keys(app):
            app.button(key=scenario).click().run()
            assert not app.metric, "a new login starts on the sign-in card"
            assert any("press **Log in**" in i.value for i in app.info), "the next step is said"
            _log_in(app)
            tier = _table(app).iloc[-1]["with your changes"].split("(")[1].rstrip(")")
            expected = {"block": "block", "ask for 2FA": "2fa", "allow": "allow"}[tier]
            assert _screen(app) == expected, (key, scenario)
            seen.add(expected)
    assert len(seen) >= 2, "the toy data covers more than one tier"  # noqa: PLR2004


def _change(app: AppTest, control: str) -> None:
    """Move one control away from its current value."""
    if control == "attack switch":
        attack = app.toggle(key="in-attack_ip")
        attack.set_value(not attack.value).run()
    elif control == "device":
        device = app.radio(key="in-device_type")
        device.set_value(next(d for d in device.options if d != device.value)).run()
    elif control == "time gap":
        gap = app.select_slider(key="in-gap")
        gap.set_value(next(g for g in gap.options if g != gap.value)).run()
    else:
        share = app.select_slider(key=NETWORK_SHARE)
        share.set_value(next(v for v in (0.3, 0.0) if v != share.value)).run()
    assert not app.exception, app.exception


@pytest.mark.parametrize("control", ["attack switch", "device", "time gap", "network share"])
def test_a_changed_attempt_asks_to_log_in_again(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    app = _open(data_dir, monkeypatch)
    assert _choose(app, "returning")
    _log_in(app)
    assert app.metric
    _change(app, control)
    assert not app.metric, "the old response is not shown for a new attempt"
    assert any("Log in again" in i.value for i in app.info)
    _log_in(app)
    assert app.metric, "logging in again shows the new response"


def test_the_2fa_screen_takes_a_code_as_a_simulation(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _open(data_dir, monkeypatch)
    for key in [b.key for b in app.button if b.key and b.key.startswith("kind-")]:
        app.button(key=key).click().run()
        if _screen(_log_in(app)) == "2fa":
            break
    else:
        pytest.skip("no scenario asks for 2FA in the toy data")
    code = next(f for f in app.text_input if f.label == "Verification code")
    assert not code.value, "a new attempt starts with an empty code"
    code.input("123456").run()
    app.button(key="verify").click().run()
    assert not app.exception, app.exception
    boxes = " ".join(m.value for m in app.markdown if "border-left" in m.value)
    assert "Identity verified." in boxes, "the owner, who has the phone, gets in"
    attacker = next(t for t in app.toggle if t.label.startswith("Simulate an attacker"))
    attacker.set_value(True).run()
    boxes = " ".join(m.value for m in app.markdown if "border-left" in m.value)
    assert "Verification failed." in boxes, "an attacker without the phone is blocked"
    assert "Identity verified." not in boxes
    intro = " ".join(m.value for m in app.markdown)
    assert "the sign-in is only simulated" in intro, "the disclaimer is on the page"


def test_the_card_reacts_to_the_response(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused login shakes the card and colours it red or orange; a login that goes through, green."""
    app = _open(data_dir, monkeypatch)
    seen = set()
    for key in [b.key for b in app.button if b.key and b.key.startswith("kind-")]:
        app.button(key=key).click().run()
        tier = _screen(_log_in(app))
        style = " ".join(m.value for m in app.markdown if "st-key-tour-card" in m.value)
        colour = {"block": "#b42318", "2fa": "#b54708", "allow": "#1e7a34"}[tier]
        assert f"outline: 3px solid {colour}" in style, tier
        assert ("@keyframes refused-" in style) == (tier != "allow"), "only a refused login shakes"
        seen.add(tier)
    assert len(seen) >= 2  # noqa: PLR2004


def _highlighted(app: AppTest) -> str:
    styles = " ".join(m.value for m in app.markdown if "200vmax" in m.value)  # the tour's spotlight
    parts = ("kinds", "attempt", "card", "response", "system")
    return next(k for k in parts if f"st-key-tour-{k} " in styles)


def _part_is_shown(app: AppTest, part: str) -> bool:
    """The highlighted part's own content is on the page, not only in the tour's CSS."""
    keys = {b.key for b in app.button}
    if part == "kinds":
        return any(k and k.startswith("kind-") for k in keys)
    if part == "attempt":
        return bool([t for t in app.toggle if t.key == "in-attack_ip"]) and "back" in keys
    if part == "card":
        return "login" in keys
    if part == "response":
        return bool(_screen(app))
    return bool(app.metric) and bool(app.table)


def test_every_tour_target_is_a_container_of_the_page() -> None:
    source = APP.read_text()
    targets = set(re.findall(r'"(tour-(?:kinds|attempt|card|response|system))"', source))
    assert len(targets) == 5, targets  # noqa: PLR2004
    assert all(f'key="{t}"' in source for t in targets), "a step would highlight nothing"


def test_the_guided_tour_highlights_each_part(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _open(data_dir, monkeypatch)
    assert not [b for b in app.button if b.key == "tour-next"], "no tour until asked"
    app.button(key="tour-start").click().run()
    targets = []
    for _ in range(10):
        part = _highlighted(app)
        assert _part_is_shown(app, part), part
        targets.append(part)
        next_button = app.button(key="tour-next")
        if next_button.label == "Done":
            break
        next_button.click().run()
    assert targets == ["kinds", "attempt", "card", "response", "system", "attempt"]
    _change(app, "attack switch")  # the last step: "change any input and log in again"
    assert not app.metric, "the tour no longer logs in for the viewer"
    assert any("Log in again" in i.value for i in app.info)
    app.button(key="tour-next").click().run()
    assert not [b for b in app.button if b.key == "tour-next"], "Done closes the tour"
    app.button(key="tour-start").click().run()
    app.button(key="tour-next").click().run()
    app.button(key="tour-back").click().run()
    assert _highlighted(app) == "kinds"
    app.button(key="tour-skip").click().run()
    assert not [b for b in app.button if b.key == "tour-next"], "Skip closes the tour"


def test_the_selected_login_is_shown_at_a_glance(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _open(data_dir, monkeypatch)
    page = " ".join(m.value for m in app.markdown)
    assert "The selected login" in page
    assert "A takeover" in page or "The real user" in page, "what really happened"


def test_each_step_gives_feedback(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A notice confirms the login, the hint says what to do next, and a change is flagged at once."""
    app = _open(data_dir, monkeypatch)
    assert _choose(app, "returning")
    assert app.button(key="login").label == "Log in"
    _log_in(app)
    assert any("Login checked" in t.value for t in app.toast)
    assert any(i.value.startswith("Next: change anything") for i in app.info)
    _change(app, "attack switch")
    assert any("attempt changed" in t.value for t in app.toast)
    assert app.button(key="login").label == "Log in again"
