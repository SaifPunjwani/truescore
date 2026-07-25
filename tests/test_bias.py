"""Tests for truescore.bias."""

from __future__ import annotations

import numpy as np
import pytest

from truescore.bias import judge_error_regression, length_bias, position_bias


def _hc3_reference(design: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Independent HC3 implementation written as an explicit loop.

    Deliberately follows the textbook definition step by step rather than the vectorized
    path used in the library, so agreement between them is evidence about the formula
    rather than about a shared shortcut.
    """
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ response
    residuals = response - design @ beta
    n, k = design.shape
    meat = np.zeros((k, k))
    for i in range(n):
        x_i = design[i : i + 1].T
        h_i = float(design[i] @ xtx_inv @ design[i])
        weight = (residuals[i] / (1.0 - h_i)) ** 2
        meat += weight * (x_i @ x_i.T)
    covariance = xtx_inv @ meat @ xtx_inv
    return np.sqrt(np.diag(covariance))


def test_hc3_standard_errors_match_an_independent_implementation() -> None:
    """The vectorized HC3 path agrees with a loop written from the definition."""
    rng = np.random.default_rng(50)
    n = 60
    x = rng.normal(0.0, 1.0, n)
    # Heteroscedastic on purpose: this is the case classical errors get wrong.
    y = 0.4 * x + rng.normal(0.0, 0.2 + 0.5 * np.abs(x), n)
    design = np.column_stack([np.ones(n), x])

    report = judge_error_regression(y, np.zeros(n), {"x": x})
    expected = _hc3_reference(design, y)
    assert report.effects[0].std_error == pytest.approx(float(expected[1]), rel=1e-10)


def test_regression_recovers_known_length_effect() -> None:
    """A judge whose error grows with length has that effect recovered with the right sign."""
    rng = np.random.default_rng(51)
    n = 800
    lengths = rng.uniform(50.0, 800.0, n)
    gold = rng.normal(0.0, 1.0, n)
    true_effect_per_token = 0.0012
    judge = gold + true_effect_per_token * lengths + rng.normal(0.0, 0.3, n)

    report = judge_error_regression(judge, gold, {"length": lengths})
    effect = report.effects[0]

    assert effect.effect == pytest.approx(true_effect_per_token, rel=0.15)
    assert effect.significant
    assert effect.low <= true_effect_per_token <= effect.high


def test_regression_reports_no_effect_when_the_judge_is_unbiased() -> None:
    """A judge whose errors are unrelated to length must not be flagged for length bias."""
    rng = np.random.default_rng(52)
    n = 600
    lengths = rng.uniform(50.0, 800.0, n)
    gold = rng.normal(0.0, 1.0, n)
    judge = gold + rng.normal(0.0, 0.3, n)

    effect = judge_error_regression(judge, gold, {"length": lengths}).effects[0]
    assert not effect.significant
    assert effect.low <= 0.0 <= effect.high


def test_length_bias_reports_effect_per_scaled_unit() -> None:
    """Scaling by ``per`` makes the coefficient readable without changing the inference."""
    rng = np.random.default_rng(53)
    n = 500
    lengths = rng.uniform(100.0, 900.0, n)
    gold = rng.normal(0.0, 1.0, n)
    judge = gold + 0.001 * lengths + rng.normal(0.0, 0.2, n)

    per_token = judge_error_regression(judge, gold, {"length": lengths}).effects[0]
    per_hundred = length_bias(judge, gold, lengths, per=100.0)

    assert per_hundred.effect == pytest.approx(per_token.effect * 100.0, rel=1e-9)
    assert per_hundred.p_value == pytest.approx(per_token.p_value, rel=1e-9)
    assert "100 units" in str(per_hundred)


def test_multiple_covariates_separate_confounded_effects() -> None:
    """Length and self-preference are disentangled when both are supplied.

    Only length truly drives the judge's error here; self-preference is correlated with
    length but has no effect of its own. A single-covariate regression on self-preference
    would blame it, which is the practical reason to fit them jointly.
    """
    rng = np.random.default_rng(54)
    n = 900
    lengths = rng.uniform(50.0, 600.0, n)
    self_pref = (lengths + rng.normal(0.0, 60.0, n) > 350.0).astype(float)
    gold = rng.normal(0.0, 1.0, n)
    judge = gold + 0.002 * lengths + rng.normal(0.0, 0.25, n)

    joint = judge_error_regression(judge, gold, {"length": lengths, "self_pref": self_pref})
    effects = {effect.name: effect for effect in joint.effects}

    assert effects["length"].significant
    assert not effects["self_pref"].significant


def test_position_bias_detects_always_first_judge() -> None:
    """A judge that always picks position one scores 1.0 and is flagged."""
    n = 120
    result = position_bias(np.ones(n, dtype=int), np.ones(n, dtype=int))
    assert result.first_position_rate == pytest.approx(1.0)
    assert result.p_value < 1e-9
    assert result.significant
    assert result.consistency == 0.0
    assert "biased toward the first" in result.summary()


def test_position_bias_reports_half_for_unbiased_judge() -> None:
    """A judge with stable preferences and no position effect scores exactly 0.5."""
    rng = np.random.default_rng(55)
    n = 200
    # The judge prefers the same option in both orders: chose_first flips with the order.
    prefers_a = rng.binomial(1, 0.6, n)
    original = prefers_a
    swapped = 1 - prefers_a

    result = position_bias(original, swapped)
    assert result.first_position_rate == pytest.approx(0.5)
    assert not result.significant
    assert result.consistency == pytest.approx(1.0)


def test_position_bias_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        position_bias(np.array([1, 0]), np.array([1, 0, 1]))


def test_regression_requires_covariates() -> None:
    with pytest.raises(ValueError, match="at least one covariate"):
        judge_error_regression(np.array([1.0, 0.0]), np.array([1.0, 1.0]), {})


def test_regression_rejects_a_constant_covariate() -> None:
    """A constant covariate is collinear with the intercept; fail rather than emit noise."""
    rng = np.random.default_rng(56)
    n = 40
    with pytest.raises(ValueError):
        judge_error_regression(rng.normal(size=n), rng.normal(size=n), {"constant": np.ones(n)})
