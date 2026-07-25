"""Shared simulation helpers for coverage tests.

Coverage tests are the backbone of this suite: an interval estimator that has not been
checked against a known ground truth is an assertion, not a result. These helpers make
the setup uniform so every estimator is judged the same way.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Trial(NamedTuple):
    """One simulated evaluation: judge labels for all examples, gold for a subset."""

    judge: np.ndarray
    gold: np.ndarray
    gold_index: np.ndarray
    true_rate: float


def simulate_trial(
    rng: np.random.Generator,
    *,
    n_total: int,
    n_gold: int,
    true_rate: float,
    sensitivity: float,
    specificity: float,
) -> Trial:
    """Draw one evaluation from a judge with known error rates.

    Gold labels are drawn iid ``Bernoulli(true_rate)``; the judge flips them according to
    ``sensitivity`` on positives and ``specificity`` on negatives. The gold-labeled subset
    is a uniform random sample, matching the assumption every estimator here declares.
    """
    truth = rng.binomial(1, true_rate, n_total)
    judge = np.where(
        truth == 1,
        rng.binomial(1, sensitivity, n_total),
        rng.binomial(1, 1.0 - specificity, n_total),
    )
    index = np.sort(rng.choice(n_total, n_gold, replace=False))
    return Trial(judge=judge, gold=truth[index], gold_index=index, true_rate=true_rate)


def coverage_bounds(
    n_replications: int, nominal: float = 0.95, n_sigma: float = 3.5
) -> tuple[float, float]:
    """Acceptance band for an observed coverage rate.

    The observed coverage of ``n_replications`` independent trials is itself binomial, so
    a correct estimator still misses the nominal rate by a predictable amount. This
    returns the band within ``n_sigma`` standard errors, which is what a test should
    assert rather than an arbitrary tolerance.
    """
    se = np.sqrt(nominal * (1.0 - nominal) / n_replications)
    return nominal - n_sigma * se, nominal + n_sigma * se
