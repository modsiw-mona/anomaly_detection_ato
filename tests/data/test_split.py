"""Tests for the user-grouped, label-stratified development / test split."""

import numpy as np
import polars as pl
import pytest

from data.split import assign_split

N_USERS = 1_000


def _features(n_users: int = N_USERS, n_takeover_users: int = 50, seed: int = 0) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)
    rows = rng.integers(1, 6, size=n_users)  # 1 to 5 logins per user
    users = np.repeat(np.arange(n_users), rows)
    label = np.zeros(users.size, dtype=bool)
    for user in range(n_takeover_users):  # one takeover login for each of the first users
        label[np.flatnonzero(users == user)[0]] = True
    order = rng.permutation(users.size)
    return pl.LazyFrame({"user_id": users[order], "label_ato": label[order]})


def test_every_user_gets_exactly_one_side() -> None:
    split = assign_split(_features())
    assert split["user_id"].is_unique().all()
    assert split.height == N_USERS
    assert set(split["split"].unique()) == {"dev", "test"}


def test_stratified_by_takeover_at_user_level() -> None:
    split = assign_split(_features())
    shares = split.group_by("has_takeover").agg((pl.col("split") == "test").mean().alias("test"))
    assert shares.filter("has_takeover")["test"].item() == pytest.approx(0.2)  # 10 of 50
    assert shares.filter(~pl.col("has_takeover"))["test"].item() == pytest.approx(0.2)


def test_deterministic_and_independent_of_row_order() -> None:
    a = assign_split(_features(seed=0))
    b = assign_split(_features(seed=0).sort("label_ato", "user_id"))
    assert a.equals(b)
    assert not a.equals(assign_split(_features(seed=0), seed=7)), "the seed matters"


@pytest.mark.parametrize("share", [0.0, 1.0, -0.1])
def test_invalid_share(share: float) -> None:
    with pytest.raises(ValueError, match="test_share"):
        assign_split(_features(), test_share=share)
