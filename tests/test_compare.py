"""Tests for truescore.compare.

The claims: McNemar's arithmetic is exact, its error rate is calibrated, the paired
bootstrap covers, the permutation test agrees with brute-force enumeration, PPI comparison
survives a judge that favors one system, and the multiplicity corrections do what their
names promise.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats

from truescore.compare import (
    benjamini_hochberg,
    holm,
    mcnemar,
    paired_bootstrap,
    paired_permutation,
    ppi_compare,
)


def _paired_binary(
    rng: np.random.Generator, n: int, discordance: float, effect: float
) -> tuple[np.ndarray, np.ndarray]:
    """Draw paired binary outcomes with a set discordance rate and pass-rate difference."""
    p10 = (discordance + effect) / 2.0
    p01 = (discordance - effect) / 2.0
    concordant = (1.0 - discordance) / 2.0
    category = rng.choice(4, size=n, p=[p10, p01, concordant, concordant])
    a = np.isin(category, [0, 2]).astype(int)
    b = np.isin(category, [1, 2]).astype(int)
    return a, b


def test_mcnemar_matches_hand_computed_exact_p() -> None:
    """10 wins for A, 2 for B, on 12 discordant pairs.

    Exact two-sided p = 2·P(X ≤ 2) with X ~ Binomial(12, ½)
                      = 2·(1 + 12 + 66)/4096 = 158/4096 = 0.038574.
    Mid-p subtracts the doubled half point mass: 0.038574 − 66/4096 = 0.022461.
    """
    a = np.array([1] * 10 + [0] * 2 + [1] * 20 + [0] * 20)
    b = np.array([0] * 10 + [1] * 2 + [1] * 20 + [0] * 20)

    exact = mcnemar(a, b, midp=False)
    assert exact.p_value == pytest.approx(0.038574, abs=1e-6)
    assert exact.n_discordant == 12

    mid = mcnemar(a, b, midp=True)
    assert mid.p_value == pytest.approx(0.022461, abs=1e-6)


def test_mcnemar_uses_only_discordant_pairs() -> None:
    """Adding examples both systems get right cannot change the evidence, only the rate."""
    a = np.array([1, 1, 1, 0, 0])
    b = np.array([0, 0, 1, 1, 0])
    base = mcnemar(a, b)
    padded = mcnemar(np.concatenate([a, np.ones(50, int)]), np.concatenate([b, np.ones(50, int)]))
    assert base.p_value == pytest.approx(padded.p_value)
    assert base.n_discordant == padded.n_discordant


def test_mcnemar_type_one_error_is_calibrated() -> None:
    """Two systems with identical accuracy are declared different at most ~5% of the time."""
    replications = 1500
    rng = np.random.default_rng(31)
    rejections = 0
    for _ in range(replications):
        a, b = _paired_binary(rng, 200, discordance=0.20, effect=0.0)
        rejections += int(mcnemar(a, b).p_value < 0.05)
    rate = rejections / replications
    assert rate <= 0.075, f"mid-p McNemar rejected {rate:.3f} of true nulls"


def test_mcnemar_flags_a_real_difference() -> None:
    """An eight-point difference on 2000 examples is detected.

    Deliberately over-powered (about 300 pairs would suffice at 80% power) so the test
    asserts the estimator's behavior rather than the luck of one draw.
    """
    rng = np.random.default_rng(32)
    a, b = _paired_binary(rng, 2000, discordance=0.25, effect=0.08)
    result = mcnemar(a, b)
    assert result.difference > 0
    assert result.significant


def test_mcnemar_does_not_flag_a_two_point_difference_on_two_hundred_examples() -> None:
    """The headline failure: a plausible-looking improvement that the data cannot support."""
    rng = np.random.default_rng(33)
    a, b = _paired_binary(rng, 200, discordance=0.20, effect=0.02)
    result = mcnemar(a, b)
    assert not result.significant
    assert "may be noise" in result.summary()


def test_paired_bootstrap_covers_at_nominal_rate() -> None:
    """Simulated coverage of the paired bootstrap interval for a difference in means."""
    replications = 400
    rng = np.random.default_rng(34)
    true_difference = 0.05
    covered = 0
    for _ in range(replications):
        a, b = _paired_binary(rng, 400, discordance=0.30, effect=true_difference)
        result = paired_bootstrap(a, b, n_bootstrap=800, seed=int(rng.integers(1 << 30)))
        covered += int(result.low <= true_difference <= result.high)
    assert 0.90 <= covered / replications <= 0.99


def test_paired_permutation_agrees_with_exact_enumeration() -> None:
    """Brute-force every sign flip on eight pairs and compare the p-value."""
    rng = np.random.default_rng(35)
    a = rng.normal(0.4, 1.0, 8)
    b = rng.normal(0.0, 1.0, 8)
    differences = a - b
    observed = abs(differences.mean())

    counts = 0
    total = 0
    for signs in itertools.product([1, -1], repeat=8):
        total += 1
        counts += int(abs(float(np.mean(differences * np.array(signs)))) >= observed - 1e-12)
    brute_force = counts / total

    result = paired_permutation(a, b, n_resamples=10000)
    assert result.p_value == pytest.approx(brute_force, abs=1e-9)


def test_ppi_compare_corrects_a_judge_biased_toward_one_system() -> None:
    """A judge that over-scores system A must not manufacture a win for A.

    The two systems are genuinely equal. The judge marks A correct 12% of the time when
    it is wrong, and never does so for B -- a length- or style-preference in caricature.
    The naive comparison sees a large fake gap; the PPI comparison, informed by gold
    labels, does not conclude A is better.
    """
    rng = np.random.default_rng(36)
    n_total, n_gold = 3000, 400
    gold_a = rng.binomial(1, 0.70, n_total)
    gold_b = rng.binomial(1, 0.70, n_total)

    judge_a = np.where(gold_a == 1, 1, rng.binomial(1, 0.12, n_total))
    judge_b = gold_b.copy()

    naive_gap = float(judge_a.mean() - judge_b.mean())
    assert naive_gap > 0.02, "the biased judge should show a fake advantage for A"

    index = np.sort(rng.choice(n_total, n_gold, replace=False))
    corrected = ppi_compare(judge_a, judge_b, gold_a[index], gold_b[index], index)

    assert abs(corrected.difference) < naive_gap
    assert corrected.low <= 0.0 <= corrected.high, "truth (no difference) must be inside"


def test_holm_controls_family_wise_error_rate() -> None:
    """Ten simultaneous null comparisons yield a false positive at most ~5% of the time."""
    replications = 2000
    rng = np.random.default_rng(37)
    families_with_a_rejection = 0
    unadjusted_rejections = 0
    for _ in range(replications):
        p_values = rng.uniform(0.0, 1.0, 10)
        families_with_a_rejection += int(np.any(holm(p_values) < 0.05))
        unadjusted_rejections += int(np.any(p_values < 0.05))
    assert families_with_a_rejection / replications <= 0.06
    assert unadjusted_rejections / replications > 0.25, (
        "unadjusted testing should fail badly here, which is why the correction exists"
    )


def test_holm_is_monotone_and_bounded() -> None:
    p_values = np.array([0.001, 0.01, 0.04, 0.2, 0.9])
    adjusted = holm(p_values)
    assert np.all(adjusted >= p_values)
    assert np.all(np.diff(adjusted) >= -1e-12)
    assert np.all(adjusted <= 1.0)


def test_benjamini_hochberg_matches_scipy() -> None:
    """Cross-check the FDR correction against scipy's implementation."""
    rng = np.random.default_rng(38)
    p_values = np.clip(rng.beta(0.5, 5.0, 25), 0.0, 1.0)
    expected = stats.false_discovery_control(p_values, method="bh")
    assert np.allclose(benjamini_hochberg(p_values), expected, atol=1e-12)


def test_comparison_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        mcnemar(np.array([1, 0, 1]), np.array([1, 0]))
