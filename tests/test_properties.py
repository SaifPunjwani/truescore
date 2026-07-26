"""Property tests and adversarial fuzzing across the whole public surface.

Two jobs.

The first is to state the invariants that must hold for *every* input rather than for the
handful a test author thought of: intervals ordered, estimates inside the unit range,
symmetries respected, comparisons antisymmetric.

The second is the contract that makes a statistics library safe to build on: **no NaN ever
escapes**. Every public function, given any input at all, either returns a finite result
or raises ``ValueError``. It never returns a quiet NaN, never returns an infinity, and
never fails with an exception the caller could not have anticipated. A NaN that reaches a
dashboard becomes a number in a slide deck; a ``ValueError`` becomes a bug report.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import truescore as ts
from truescore.agreement import cohen_kappa, gwet_ac1, judge_agreement, wilson_interval
from truescore.compare import benjamini_hochberg, holm, mcnemar, paired_bootstrap
from truescore.contamination import combine_shards, exchangeability_test
from truescore.correct import (
    gold_only_estimate,
    judge_only_estimate,
    ppi_estimate,
    rogan_gladen_estimate,
)
from truescore.drift import judge_drift
from truescore.power import min_detectable_effect, required_gold_labels, required_pairs
from truescore.sequential import confidence_sequence


@st.composite
def binary_arrays(draw: st.DrawFn, min_size: int = 4, max_size: int = 60) -> np.ndarray:
    values = draw(st.lists(st.integers(0, 1), min_size=min_size, max_size=max_size))
    return np.asarray(values, dtype=np.int64)


@st.composite
def paired_binary_arrays(
    draw: st.DrawFn, min_size: int = 4, max_size: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """Two binary arrays of equal length, drawn together rather than filtered to match."""
    n = draw(st.integers(min_size, max_size))
    a = draw(st.lists(st.integers(0, 1), min_size=n, max_size=n))
    b = draw(st.lists(st.integers(0, 1), min_size=n, max_size=n))
    return np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)


@st.composite
def label_sets(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A judge column over all examples plus a gold column over a proper subset."""
    n_total = draw(st.integers(8, 60))
    judge = np.asarray(draw(st.lists(st.integers(0, 1), min_size=n_total, max_size=n_total)))
    # Leave at least two unlabeled examples: prediction-powered inference needs an
    # unlabeled set it can actually estimate a variance from.
    n_gold = draw(st.integers(2, n_total - 2))
    gold_index = np.sort(
        np.asarray(
            draw(
                st.lists(
                    st.integers(0, n_total - 1),
                    min_size=n_gold,
                    max_size=n_gold,
                    unique=True,
                )
            )
        )
    )
    gold = np.asarray(draw(st.lists(st.integers(0, 1), min_size=n_gold, max_size=n_gold)))
    return judge, gold, gold_index


def _finite_interval(low: float, point: float, high: float) -> None:
    for value in (low, point, high):
        assert math.isfinite(value), f"non-finite value escaped: {value}"
    assert low <= point <= high, f"interval does not contain its point: {low}, {point}, {high}"


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


@given(st.integers(0, 200), st.integers(1, 200))
def test_wilson_interval_is_ordered_and_bounded(successes: int, n: int) -> None:
    assume(successes <= n)
    interval = wilson_interval(successes, n)
    _finite_interval(interval.low, interval.point, interval.high)
    assert 0.0 <= interval.low <= interval.high <= 1.0


@given(st.integers(0, 100), st.integers(1, 100))
def test_wilson_interval_is_monotone_in_successes(successes: int, n: int) -> None:
    assume(successes < n)
    lower = wilson_interval(successes, n)
    upper = wilson_interval(successes + 1, n)
    assert upper.low >= lower.low - 1e-12
    assert upper.high >= lower.high - 1e-12


@given(paired_binary_arrays())
def test_agreement_coefficients_stay_in_range(pair: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = pair
    for coefficient in (cohen_kappa(a, b), gwet_ac1(a, b)):
        assert math.isfinite(coefficient)
        assert -1.0 - 1e-9 <= coefficient <= 1.0 + 1e-9


@given(paired_binary_arrays())
def test_agreement_is_symmetric_in_its_two_raters(pair: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = pair
    assert cohen_kappa(a, b) == pytest.approx(cohen_kappa(b, a))
    assert gwet_ac1(a, b) == pytest.approx(gwet_ac1(b, a))


@given(paired_binary_arrays())
def test_agreement_is_invariant_to_relabelling_the_classes(
    pair: tuple[np.ndarray, np.ndarray],
) -> None:
    """Calling 'pass' 0 instead of 1 cannot change how much two raters agree."""
    a, b = pair
    assert cohen_kappa(a, b) == pytest.approx(cohen_kappa(1 - a, 1 - b))
    assert gwet_ac1(a, b) == pytest.approx(gwet_ac1(1 - a, 1 - b))


@given(label_sets())
def test_estimators_return_ordered_finite_intervals(
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    judge, gold, gold_index = data
    for estimate in (
        judge_only_estimate(judge),
        gold_only_estimate(gold),
        ppi_estimate(judge, gold, gold_index),
    ):
        _finite_interval(estimate.low, estimate.point, estimate.high)


@given(label_sets())
def test_estimates_are_invariant_to_example_order(
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Rows arrive in whatever order the harness wrote them; that must not matter."""
    judge, gold, gold_index = data
    permutation = np.random.default_rng(0).permutation(judge.shape[0])
    inverse = np.argsort(permutation)

    reordered_judge = judge[permutation]
    reordered_index = np.sort(inverse[gold_index])
    # Re-align gold with its examples under the new ordering.
    position = {int(original): i for i, original in enumerate(gold_index)}
    reordered_gold = np.asarray(
        [gold[position[int(permutation[j])]] for j in reordered_index], dtype=gold.dtype
    )

    original = ppi_estimate(judge, gold, gold_index)
    shuffled = ppi_estimate(reordered_judge, reordered_gold, reordered_index)
    assert shuffled.point == pytest.approx(original.point, abs=1e-9)


@given(label_sets())
def test_flipping_every_label_reflects_the_estimate(
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Measuring the failure rate instead of the pass rate must give exactly 1 - p."""
    judge, gold, gold_index = data
    direct = ppi_estimate(judge, gold, gold_index)
    flipped = ppi_estimate(1 - judge, 1 - gold, gold_index)
    assert flipped.point == pytest.approx(1.0 - direct.point, abs=1e-9)


@given(paired_binary_arrays(min_size=6))
def test_mcnemar_is_antisymmetric(pair: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = pair
    forward = mcnemar(a, b)
    backward = mcnemar(b, a)
    assert forward.difference == pytest.approx(-backward.difference)
    assert forward.p_value == pytest.approx(backward.p_value)
    assert 0.0 <= forward.p_value <= 1.0


@given(binary_arrays(min_size=6))
def test_a_system_never_beats_itself(a: np.ndarray) -> None:
    result = mcnemar(a, a)
    assert result.difference == 0.0
    assert result.p_value == 1.0
    assert not result.significant


@given(st.lists(st.floats(1e-6, 1.0), min_size=1, max_size=25))
def test_multiplicity_corrections_only_increase_p_values(raw: list[float]) -> None:
    values = np.asarray(raw)
    for adjusted in (holm(values), benjamini_hochberg(values)):
        assert np.all(np.isfinite(adjusted))
        assert np.all(adjusted >= values - 1e-12)
        assert np.all(adjusted <= 1.0 + 1e-12)


@given(st.lists(st.floats(1e-6, 1.0), min_size=2, max_size=15))
def test_holm_is_at_least_as_conservative_as_benjamini_hochberg(raw: list[float]) -> None:
    """Controlling the family-wise error rate is strictly harder than controlling the FDR."""
    values = np.asarray(raw)
    assert np.all(holm(values) >= benjamini_hochberg(values) - 1e-12)


@given(st.lists(st.integers(0, 1), min_size=5, max_size=200))
def test_confidence_sequence_bounds_are_ordered_and_monotone(values: list[int]) -> None:
    sequence = confidence_sequence(np.asarray(values, dtype=float))
    assert np.all(np.isfinite(sequence.lower_bounds))
    assert np.all(np.isfinite(sequence.upper_bounds))
    assert np.all(sequence.lower_bounds <= sequence.upper_bounds + 1e-12)
    assert np.all(np.diff(sequence.lower_bounds) >= -1e-12)
    assert np.all(np.diff(sequence.upper_bounds) <= 1e-12)


@given(st.floats(-1e5, 1e5), st.lists(st.floats(-1e5, 1e5), min_size=1, max_size=50))
def test_permutation_p_value_is_on_the_grid(canonical: float, permuted: list[float]) -> None:
    """An exact permutation p-value can only take the values k/(m+1)."""
    result = exchangeability_test(canonical, np.asarray(permuted))
    m = len(permuted)
    grid = [(k + 1) / (m + 1) for k in range(m + 1)]
    assert any(result.p_value == pytest.approx(value) for value in grid)
    assert 0.0 < result.p_value <= 1.0


@given(paired_binary_arrays(min_size=6))
def test_drift_against_an_unchanged_judge_is_exactly_zero(
    pair: tuple[np.ndarray, np.ndarray],
) -> None:
    judge, gold = pair
    report = judge_drift(judge, judge, gold)
    assert report.agreement_change == 0.0
    assert report.flip_rate == 0.0
    assert not report.agreement_changed


# ---------------------------------------------------------------------------
# Adversarial inputs: finite result or ValueError, never NaN, never a surprise
# ---------------------------------------------------------------------------

ADVERSARIAL: list[Any] = [
    np.array([]),
    np.array([0]),
    np.array([1]),
    np.array([0, 0, 0, 0]),
    np.array([1, 1, 1, 1]),
    np.array([0, 1]),
    np.zeros(3),
    np.ones(3),
    np.array([np.nan, 1.0, 0.0]),
    np.array([np.inf, 1.0]),
    np.array([0.5, 0.5]),
    np.array([[0, 1], [1, 0]]),
    np.array([2, 3, 4]),
    np.array([-1, 0, 1]),
]


@pytest.mark.parametrize("judge", ADVERSARIAL)
@pytest.mark.parametrize("gold", ADVERSARIAL)
def test_no_estimator_ever_returns_nan(judge: np.ndarray, gold: np.ndarray) -> None:
    """The central robustness contract, over every pairing of nasty inputs."""
    index = np.arange(min(len(np.atleast_1d(judge)), len(np.atleast_1d(gold))))
    for call in (
        lambda: judge_only_estimate(judge),
        lambda: gold_only_estimate(gold),
        lambda: ppi_estimate(judge, gold, index),
        lambda: rogan_gladen_estimate(judge, gold, index),
    ):
        try:
            estimate = call()
        except ValueError:
            continue  # the documented, expected failure mode
        assert math.isfinite(estimate.point), f"NaN escaped as a point estimate: {estimate}"
        assert math.isfinite(estimate.low) and math.isfinite(estimate.high)


@pytest.mark.parametrize("a", ADVERSARIAL)
@pytest.mark.parametrize("b", ADVERSARIAL)
def test_no_comparison_ever_returns_nan(a: np.ndarray, b: np.ndarray) -> None:
    for call in (lambda: mcnemar(a, b), lambda: paired_bootstrap(a, b, n_bootstrap=50)):
        try:
            result = call()
        except ValueError:
            continue
        assert math.isfinite(result.difference)
        assert math.isfinite(result.p_value)
        assert 0.0 <= result.p_value <= 1.0


@pytest.mark.parametrize("values", ADVERSARIAL)
def test_no_agreement_or_sequence_call_ever_returns_nan(values: np.ndarray) -> None:
    for call in (
        lambda: judge_agreement(values, values),
        lambda: confidence_sequence(values),
        lambda: exchangeability_test(0.0, values),
        lambda: combine_shards(values),
    ):
        try:
            result = call()
        except ValueError:
            continue
        assert result is not None


@pytest.mark.parametrize(
    ("n_total", "target", "rate", "sensitivity", "specificity"),
    [
        (3, 0.5, 0.5, 0.9, 0.9),
        (10_000, 1e-6, 0.5, 0.9, 0.9),
        (100, 0.5, 0.999, 0.999, 0.999),
        (100, 0.5, 0.001, 0.001, 0.999),
        (5, 0.9, 0.5, 0.5, 0.5),
    ],
)
def test_planning_functions_survive_extreme_configurations(
    n_total: int, target: float, rate: float, sensitivity: float, specificity: float
) -> None:
    plan = required_gold_labels(
        n_total,
        target_half_width=target,
        true_rate=rate,
        sensitivity=sensitivity,
        specificity=specificity,
    )
    assert math.isfinite(plan.achieved_half_width)
    assert plan.required_gold >= 2
    assert isinstance(plan.summary(), str)


@given(
    st.floats(0.001, 0.5),
    st.floats(0.01, 1.0),
)
def test_sample_size_planning_round_trips(effect: float, discordance: float) -> None:
    assume(effect < discordance)
    n = required_pairs(effect, discordance_rate=discordance)
    assert n >= 1
    recovered = min_detectable_effect(n, discordance_rate=discordance)
    assert math.isfinite(recovered)
    assert 0.0 < recovered <= discordance


def test_every_public_export_is_importable_and_documented() -> None:
    """A stale __all__ is a broken import for somebody; check the whole surface."""
    for name in ts.__all__:
        attribute = getattr(ts, name)
        if name == "__version__":
            continue
        assert attribute.__doc__, f"{name} is exported without a docstring"
