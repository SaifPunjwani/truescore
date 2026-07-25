"""Tests for truescore.power.

Planning functions are only useful if their promises hold, so each is checked by
simulating the study it recommends and confirming the outcome it predicted.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.simulate import simulate_trial
from truescore.compare import mcnemar
from truescore.correct import ppi_estimate
from truescore.power import min_detectable_effect, required_gold_labels, required_pairs


def test_required_gold_labels_achieves_target_in_simulation() -> None:
    """Label at the recommended budget and the interval really is about the target width."""
    plan = required_gold_labels(
        4000, target_half_width=0.03, true_rate=0.7, sensitivity=0.92, specificity=0.85
    )
    assert plan.feasible

    rng = np.random.default_rng(60)
    widths = []
    for _ in range(200):
        trial = simulate_trial(
            rng,
            n_total=4000,
            n_gold=plan.required_gold,
            true_rate=0.7,
            sensitivity=0.92,
            specificity=0.85,
        )
        widths.append(ppi_estimate(trial.judge, trial.gold, trial.gold_index).half_width)

    achieved = float(np.mean(widths))
    assert achieved == pytest.approx(0.03, abs=0.004), (
        f"planned for ±0.03 with {plan.required_gold} labels; simulation gave ±{achieved:.4f}"
    )


def test_better_judge_needs_fewer_gold_labels() -> None:
    """The whole economic argument: judge quality converts into human labels saved."""
    weak = required_gold_labels(
        5000, target_half_width=0.03, true_rate=0.6, sensitivity=0.70, specificity=0.65
    )
    strong = required_gold_labels(
        5000, target_half_width=0.03, true_rate=0.6, sensitivity=0.97, specificity=0.95
    )
    assert strong.required_gold < weak.required_gold
    assert strong.labels_saved > weak.labels_saved


def test_gold_budget_never_claims_more_than_gold_only_would_need() -> None:
    """PPI cannot need more labels than ignoring the judge entirely."""
    plan = required_gold_labels(
        3000, target_half_width=0.04, true_rate=0.5, sensitivity=0.9, specificity=0.9
    )
    assert plan.required_gold <= plan.gold_only_required


def test_unreachable_target_is_reported_as_infeasible() -> None:
    """A target beyond what the example pool supports is stated plainly, not faked."""
    plan = required_gold_labels(
        50, target_half_width=0.001, true_rate=0.5, sensitivity=0.9, specificity=0.9
    )
    assert not plan.feasible
    assert plan.achieved_half_width > 0.001
    assert "NOT reachable" in plan.summary()


def test_required_pairs_delivers_requested_power_in_simulation() -> None:
    """Run the study the function recommends; it detects the effect about 80% of the time."""
    effect, discordance = 0.06, 0.30
    n = required_pairs(effect, discordance_rate=discordance, power=0.8)

    rng = np.random.default_rng(61)
    p10 = (discordance + effect) / 2.0
    p01 = (discordance - effect) / 2.0
    concordant = (1.0 - discordance) / 2.0

    detections = 0
    replications = 500
    for _ in range(replications):
        category = rng.choice(4, size=n, p=[p10, p01, concordant, concordant])
        a = np.isin(category, [0, 2]).astype(int)
        b = np.isin(category, [1, 2]).astype(int)
        detections += int(mcnemar(a, b).p_value < 0.05)

    observed_power = detections / replications
    assert 0.74 <= observed_power <= 0.88, f"requested 0.80 power, observed {observed_power:.3f}"


def test_min_detectable_effect_inverts_required_pairs() -> None:
    """The two planning directions agree: MDE at n pairs needs about n pairs to detect."""
    discordance = 0.25
    n = 900
    effect = min_detectable_effect(n, discordance_rate=discordance)
    round_trip = required_pairs(effect, discordance_rate=discordance)
    assert round_trip == pytest.approx(n, rel=0.02)


def test_min_detectable_effect_shrinks_with_sample_size() -> None:
    small = min_detectable_effect(200, discordance_rate=0.3)
    large = min_detectable_effect(2000, discordance_rate=0.3)
    assert large < small


def test_two_hundred_examples_cannot_resolve_two_points() -> None:
    """The claim in the README, checked: a small eval cannot support small differences."""
    mde = min_detectable_effect(200, discordance_rate=0.2)
    assert mde > 0.02, f"200 examples resolve no better than {mde:.4f}"


def test_effect_cannot_exceed_discordance() -> None:
    """The pass-rate gap is bounded by how often the systems differ at all."""
    with pytest.raises(ValueError, match="cannot exceed discordance_rate"):
        required_pairs(0.3, discordance_rate=0.2)


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="target_half_width must be positive"):
        required_gold_labels(100, target_half_width=0.0)
    with pytest.raises(ValueError, match="sensitivity must lie"):
        required_gold_labels(100, target_half_width=0.05, sensitivity=1.5)
    with pytest.raises(ValueError, match="n_total must be at least 3"):
        required_gold_labels(2, target_half_width=0.05)
