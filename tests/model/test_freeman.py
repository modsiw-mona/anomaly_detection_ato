"""Tests for the Freeman et al. (2016) RBA score: hand-computed values and no future information."""

import math
from typing import Any

import polars as pl
import pytest

from model.freeman import HIERARCHY, SCORE, freeman_scores
from tests.data.test_features import CATCH_ALL, _config, _events, _random_events, at

SAME = {k: "a" for k in ("ip", "asn", "country", "user_agent", "browser", "os", "device_type")}


def _scores(rows: list[dict[str, Any]]) -> dict[int, float | None]:
    out = freeman_scores(_events(rows), _config()).collect()
    return dict(zip(out["event_id"].to_list(), out[SCORE].to_list(), strict=True))


def test_hand_computed_scores() -> None:
    rows = [
        {"user_id": 1, "timestamp": at(0), "login_success": True, **SAME},
        {"user_id": 2, "timestamp": at(1), "login_success": True, **SAME},
        {"user_id": 1, "timestamp": at(2), "login_success": True, **SAME},
    ]
    scores = _scores(rows)
    assert scores[0] is None, "first logins are not scored (no history)"
    assert scores[1] is None
    # event 2: user 1 saw every value once (local = 1); 2 earlier logins globally, all "a"
    ip_weights, ua_weights = HIERARCHY["ip"], HIERARCHY["user_agent"]
    ip_global = ip_weights["ip"] * 2 / 3 + ip_weights["asn"] + ip_weights["country"]
    ua_global = ua_weights["user_agent"] * 2 / 3 + sum(list(ua_weights.values())[1:])
    local = 1.0
    expected = math.log(3) - math.log(2) + math.log(ip_global / local) + math.log(ua_global / local)
    assert scores[2] == pytest.approx(expected)


def test_unusual_login_scores_higher_than_a_habitual_one() -> None:
    history = [{"user_id": 1, "timestamp": at(i), "login_success": True, **SAME} for i in range(5)]
    habitual = {"user_id": 1, "timestamp": at(10), "login_success": True, **SAME}
    unusual = {**habitual, "ip": "z", "asn": "z", "country": "z", "user_agent": "z", "browser": "z"}
    unusual_score, habitual_score = (
        _scores([*history, unusual])[5],
        _scores([*history, habitual])[5],
    )
    assert unusual_score is not None
    assert habitual_score is not None
    assert unusual_score > habitual_score


def test_failed_attempts_and_catch_all_are_not_scored_or_counted() -> None:
    rows = [
        {"user_id": 1, "timestamp": at(0), "login_success": True, **SAME},
        {"user_id": 1, "timestamp": at(1), "login_success": False, **SAME},
        {"user_id": CATCH_ALL, "timestamp": at(2), "login_success": True, **SAME},
        {"user_id": 2, "timestamp": at(3), "login_success": True, **SAME},
    ]
    scores = _scores(rows)
    assert set(scores) == {0, 3}, "failed attempts and the catch-all are not scored"
    assert scores[0] is None
    assert scores[3] is None


@pytest.mark.parametrize("seed", [0, 1])
def test_score_uses_no_information_from_the_future(seed: int) -> None:
    events = _random_events(150, seed)
    config = _config()
    full = freeman_scores(_events(events), config).collect()
    for event_id in full["event_id"].gather_every(6).to_list():
        truncated = freeman_scores(_events(events[: event_id + 1]), config).collect()
        assert truncated.filter(pl.col("event_id") == event_id).equals(
            full.filter(pl.col("event_id") == event_id)
        ), f"score of event {event_id} depends on later events"
