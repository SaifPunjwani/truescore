"""Tests for truescore.contamination.

The permutation test is exact, so its false-positive rate should equal the level almost
mechanically. That makes it unusually testable: any error in the p-value arithmetic shows
up immediately as miscalibration.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from truescore.contamination import combine_shards, exchangeability_test


def test_null_false_positive_rate_matches_alpha() -> None:
    """Under exchangeability the canonical order is just another order.

    Drawing the canonical and permuted log-likelihoods from one distribution enforces the
    null exactly, so the rejection rate should land on the nominal level rather than
    merely below it.
    """
    replications, n_permutations = 4000, 99
    rng = np.random.default_rng(0)

    rejections = 0
    for _ in range(replications):
        draws = rng.normal(-1200.0, 15.0, n_permutations + 1)
        result = exchangeability_test(float(draws[0]), draws[1:])
        rejections += int(result.contaminated)

    rate = rejections / replications
    standard_error = np.sqrt(0.05 * 0.95 / replications)
    assert abs(rate - 0.05) < 4.0 * standard_error, f"observed false-positive rate {rate:.4f}"


def test_null_p_values_are_uniform_on_the_permutation_grid() -> None:
    """A sharper check than the rejection rate: the whole p-value distribution."""
    replications, n_permutations = 3000, 19
    rng = np.random.default_rng(1)
    p_values = np.array(
        [
            exchangeability_test(float(draws[0]), draws[1:]).p_value
            for draws in (rng.normal(0.0, 1.0, n_permutations + 1) for _ in range(replications))
        ]
    )
    # Exact test on a grid of 20 points: each value should appear about equally often.
    for k in range(1, n_permutations + 2):
        share = float(np.mean(np.isclose(p_values, k / (n_permutations + 1))))
        assert abs(share - 1.0 / (n_permutations + 1)) < 0.02


def test_memorized_canonical_order_is_detected() -> None:
    """A model that scores the published order far above shuffles is flagged."""
    rng = np.random.default_rng(2)
    permuted = rng.normal(-1500.0, 20.0, 199)
    canonical = -1380.0  # six standard deviations above the shuffled orders

    result = exchangeability_test(canonical, permuted)
    assert result.contaminated
    assert result.p_value == pytest.approx(1.0 / 200.0)
    assert result.z_score > 5.0
    assert "evidence of contamination" in result.summary()


def test_uncontaminated_model_is_not_flagged() -> None:
    """A canonical order sitting in the middle of the shuffles yields a large p-value."""
    rng = np.random.default_rng(3)
    permuted = rng.normal(-1500.0, 20.0, 199)
    result = exchangeability_test(float(np.median(permuted)), permuted)

    assert not result.contaminated
    assert result.p_value > 0.4
    assert "no evidence" in result.summary()


def test_resolution_is_reported_so_a_floor_is_not_read_as_a_finding() -> None:
    """With 19 permutations the smallest achievable p-value is 0.05, and the report says so."""
    rng = np.random.default_rng(4)
    permuted = rng.normal(0.0, 1.0, 19)
    result = exchangeability_test(1e6, permuted)

    assert result.p_value == pytest.approx(0.05)
    assert result.resolution == pytest.approx(0.05)
    assert "finest resolution" in result.summary()


def test_fisher_combination_is_calibrated_under_the_null() -> None:
    """Independent uniform p-values combine to a uniform p-value."""
    replications = 4000
    rng = np.random.default_rng(5)
    rejections = sum(
        int(combine_shards(rng.uniform(0.0, 1.0, 8)).contaminated) for _ in range(replications)
    )
    rate = rejections / replications
    standard_error = np.sqrt(0.05 * 0.95 / replications)
    assert abs(rate - 0.05) < 4.0 * standard_error, f"observed false-positive rate {rate:.4f}"


def test_fisher_statistic_matches_the_chi_squared_definition() -> None:
    """Cross-check the arithmetic against the closed form."""
    p_values = np.array([0.2, 0.1, 0.35, 0.5])
    combined = combine_shards(p_values)
    expected_statistic = float(-2.0 * np.sum(np.log(p_values)))

    assert combined.statistic == pytest.approx(expected_statistic)
    assert combined.p_value == pytest.approx(float(stats.chi2.sf(expected_statistic, df=8)))


def test_pooling_shards_recovers_power() -> None:
    """Weak evidence in each shard becomes decisive when pooled.

    No individual shard reaches significance at 0.05; together they do. This is why the
    sharded design exists: scoring many permutations of a whole dataset is expensive, and
    several cheap tests buy more power than one expensive one.
    """
    shard_p_values = np.array([0.12, 0.09, 0.15, 0.11, 0.08, 0.14, 0.10, 0.13])
    assert np.all(shard_p_values > 0.05)

    combined = combine_shards(shard_p_values)
    assert combined.contaminated
    assert combined.p_value < 0.01
    assert "pooled evidence of contamination" in combined.summary()


def test_combination_rejects_impossible_p_values() -> None:
    """A permutation test cannot return zero; such a value is an upstream bug."""
    with pytest.raises(ValueError, match=r"p_values must lie in \(0, 1\]"):
        combine_shards(np.array([0.0, 0.5]))
    with pytest.raises(ValueError, match=r"p_values must lie in \(0, 1\]"):
        combine_shards(np.array([0.5, 1.5]))


def test_test_rejects_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="canonical_loglik must be finite"):
        exchangeability_test(float("nan"), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="non-finite"):
        exchangeability_test(1.0, np.array([1.0, np.inf]))
    with pytest.raises(ValueError, match="must be non-empty"):
        exchangeability_test(1.0, np.array([]))
