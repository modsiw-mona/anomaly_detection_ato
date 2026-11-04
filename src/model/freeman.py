"""Freeman et al. (2016) risk-based authentication score — the established heuristic baseline.

The model (Freeman et al., NDSS 2016, eq. 7 without attack data) scores a login ``x`` of user ``u`` as

    risk(x, u) = p(u is attacked) / p(u logs in) * prod over features of p(x_f) / p(x_f | u)

high when the login's values are common globally but unusual *for this user*. Features are
hierarchical: the IP feature mixes IP, ASN and country; the user-agent feature mixes user agent,
browser, OS and device type, each level weighted as in the reference implementation for this
dataset (``das-group/rba-algorithm``, read as text, not executed).

Likelihoods, as in that reference:

* **local** ``p(x_f | u)`` — weighted sum, over levels, of the share of the user's earlier
  successful logins with that value (unsmoothed: an unseen value contributes 0). If the whole
  feature is unseen, the reference sets local = global / 4, so the feature raises the risk four-fold;
* **global** ``p(x_f)`` — weighted sum of the share of all earlier successful logins with the value,
  with add-one smoothing on the top level only;
* ``p(u is attacked) = 1 / users seen so far``; ``p(u logs in) = user's logins / all logins so far``.

Documented simplifications (the reference itself warns it may differ from the unpublished original):
each level uses its plain relative frequency instead of the reference's conditional sub-likelihood
products, and the smoothing uses one unseen slot.

**First logins are not scored** (null), as in the reference implementation, which only scores a
user's second and later logins: with no history, ``p(x_f | u)`` is undefined. An earlier version gave
them the four-fold rule for both features; that put 2.2 M near-tied first logins (28% of logins) at the top
of the ranking and filled every alert budget with them. Everything is computed past-only over the
whole timeline, like the behavioural features; the score is returned as ``log(risk)``.
"""

import polars as pl

from data.datasets import DatasetConfig
from data.features import EVENT_ID, SUCCESS, TIME, USER, real_account_attempts

# Weights of the reference implementation for this dataset (das-group/rba-algorithm).
HIERARCHY: dict[str, dict[str, float]] = {
    "ip": {"ip": 0.6, "asn": 0.3, "country": 0.1},
    "user_agent": {
        "user_agent": 0.5386653840551359,
        "browser": 0.2680451498625666,
        "os": 0.18818295100109536,
        "device_type": 0.0051065150812021525,
    },
}
UNSEEN_FACTOR = 4.0  # local = global / 4 when the user never had this feature value
SCORE = "freeman_log_risk"


def _local_likelihood(feature: str) -> pl.Expr:
    n = pl.col("_n_user")
    levels = [
        weight * pl.when(n > 0).then(pl.col(f"_lc_{level}") / n).otherwise(0.0)
        for level, weight in HIERARCHY[feature].items()
    ]
    return pl.sum_horizontal(levels)


def _global_likelihood(feature: str) -> pl.Expr:
    total = pl.col("_n_global")
    parts = []
    for i, (level, weight) in enumerate(HIERARCHY[feature].items()):
        count = pl.col(f"_gc_{level}")
        if i == 0:  # add-one smoothing on the top level only
            share = pl.when(count > 0).then(count / (total + 1)).otherwise(1 / (total + 1))
        else:
            share = pl.when(total > 0).then(count / total).otherwise(0.0)
        parts.append(weight * share)
    return pl.sum_horizontal(parts)


def freeman_terms(intermediate: pl.LazyFrame, config: DatasetConfig) -> pl.LazyFrame:
    """The parts of the score, for every successful login of a real account (past-only).

    Columns: ``event_id``; ``log_prior`` (log p(u attacked) - log p(u logs in)); and, per feature
    ``f``, ``log_global_f`` = log p(x_f) and ``log_local_f`` = log p(x_f | u). Local terms and the
    prior are null on a user's first login. Exposed so that analyses can recombine the terms (e.g.
    drop the global factor) without re-implementing the counting.
    """
    levels = sorted({level for feature in HIERARCHY.values() for level in feature})
    successes = real_account_attempts(intermediate, config).filter(pl.col(SUCCESS))
    local = successes.with_columns(  # user / time order: counts over the user's earlier logins
        pl.int_range(pl.len()).over(USER).alias("_n_user"),
        *(pl.int_range(pl.len()).over(USER, level).alias(f"_lc_{level}") for level in levels),
    )
    counted = local.sort(TIME, EVENT_ID).with_columns(  # global time order: counts over all users
        pl.int_range(pl.len()).alias("_n_global"),
        pl.col(USER).is_first_distinct().cum_sum().alias("_users_seen"),
        *(pl.int_range(pl.len()).over(level).alias(f"_gc_{level}") for level in levels),
    )
    has_history = pl.col("_n_user") > 0
    terms = []
    for feature in HIERARCHY:
        glob, loc = _global_likelihood(feature), _local_likelihood(feature)
        loc = pl.when(loc > 0).then(loc).otherwise(glob / UNSEEN_FACTOR)
        terms += [
            glob.log().alias(f"log_global_{feature}"),
            pl.when(has_history).then(loc.log()).alias(f"log_local_{feature}"),
        ]
    prior = (
        (pl.col("_n_global") + 1).log()  # p(u logs in) = n_u / N, with N counting this login
        - pl.col("_users_seen").log()  # p(u attacked) = 1 / |U|
        - pl.col("_n_user").log()
    )
    return counted.select(EVENT_ID, pl.when(has_history).then(prior).alias("log_prior"), *terms)


def freeman_scores(intermediate: pl.LazyFrame, config: DatasetConfig) -> pl.LazyFrame:
    """``event_id`` and ``freeman_log_risk`` (null on a user's first login)."""
    ratio = [pl.col(f"log_global_{f}") - pl.col(f"log_local_{f}") for f in HIERARCHY]
    return freeman_terms(intermediate, config).select(
        EVENT_ID, (pl.col("log_prior") + pl.sum_horizontal(ratio)).alias(SCORE)
    )
