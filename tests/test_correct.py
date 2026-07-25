"""Tests for the estimators in truescore.correct.

The central claims under test:
  1. PPI intervals cover at the nominal rate, whether the judge is good, mediocre, or
     actively misleading. Coverage is the whole promise; everything else is efficiency.
  2. PPI is not worse than ignoring the judge, and is meaningfully tighter when the judge
     is informative.
  3. The naive judge-only estimate is biased by a knowable amount, and its interval can
     exclude the truth entirely -- which is the failure this library exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.simulate import coverage_bounds, simulate_trial
from truescore.correct import (
    gold_only_estimate,
    judge_only_estimate,
    ppi_estimate,
    rogan_gladen_estimate,
)


def test_judge_only_ignores_judge_error() -> None:
    """A lenient judge inflates the naive estimate, and its interval misses the truth.

    Truth 0.70 with sensitivity 0.95 and specificity 0.60 implies a judge positive rate of
    0.7(0.95) + 0.3(0.40) = 0.785, so the naive estimate is biased upward by 8.5 points --
    far outside its own confidence interval at this sample size.
    """
    rng = np.random.default_rng(0)
    trial = simulate_trial(
        rng, n_total=4000, n_gold=400, true_rate=0.70, sensitivity=0.95, specificity=0.60
    )
    naive = judge_only_estimate(trial.judge)

    assert naive.point == pytest.approx(0.785, abs=0.02)
    assert naive.low > trial.true_rate, "the naive interval should exclude the truth here"


def test_ppi_reduces_to_gold_only_at_lambda_zero() -> None:
    """λ=0 discards the judge, so the PPI point estimate is exactly the gold mean."""
    rng = np.random.default_rng(1)
    trial = simulate_trial(
        rng, n_total=500, n_gold=80, true_rate=0.6, sensitivity=0.9, specificity=0.9
    )
    estimate = ppi_estimate(trial.judge, trial.gold, trial.gold_index, lambda_=0.0)
    assert estimate.point == pytest.approx(float(trial.gold.mean()), abs=1e-12)


@pytest.mark.parametrize(
    ("sensitivity", "specificity", "label"),
    [
        (0.95, 0.95, "strong judge"),
        (0.85, 0.70, "mediocre judge"),
        (0.55, 0.50, "near-useless judge"),
    ],
)
def test_ppi_covers_at_nominal_rate_under_simulation(
    sensitivity: float, specificity: float, label: str
) -> None:
    """PPI intervals cover the truth ~95% of the time regardless of judge quality.

    This is the property that makes PPI safe to adopt before you know how good your judge
    is: a bad judge costs precision, never validity.
    """
    replications = 600
    rng = np.random.default_rng(20)
    covered = 0
    for _ in range(replications):
        trial = simulate_trial(
            rng,
            n_total=1500,
            n_gold=150,
            true_rate=0.65,
            sensitivity=sensitivity,
            specificity=specificity,
        )
        estimate = ppi_estimate(trial.judge, trial.gold, trial.gold_index)
        covered += int(estimate.low <= trial.true_rate <= estimate.high)

    low, high = coverage_bounds(replications)
    observed = covered / replications
    assert low <= observed <= high, (
        f"{label}: coverage {observed:.3f} outside [{low:.3f}, {high:.3f}]"
    )


def test_ppi_is_tighter_than_gold_only_with_an_informative_judge() -> None:
    """A good judge buys precision: PPI intervals are narrower than gold-only ones.

    Averaged over trials rather than asserted per-trial, since either estimator can win a
    single draw by luck.
    """
    rng = np.random.default_rng(3)
    ppi_widths, gold_widths = [], []
    for _ in range(200):
        trial = simulate_trial(
            rng, n_total=2000, n_gold=150, true_rate=0.7, sensitivity=0.95, specificity=0.92
        )
        ppi_widths.append(ppi_estimate(trial.judge, trial.gold, trial.gold_index).half_width)
        gold_widths.append(gold_only_estimate(trial.gold).half_width)

    assert np.mean(ppi_widths) < np.mean(gold_widths), (
        "PPI should exploit an accurate judge to tighten the interval"
    )


def test_ppi_does_not_blow_up_with_an_adversarial_judge() -> None:
    """An anti-correlated judge must not widen PPI beyond the gold-only interval.

    λ is clipped at zero, so the worst case degenerates to ignoring the judge entirely.
    """
    rng = np.random.default_rng(4)
    trial = simulate_trial(
        rng, n_total=1200, n_gold=120, true_rate=0.6, sensitivity=0.9, specificity=0.9
    )
    adversarial = 1 - trial.judge  # judge says the opposite of the truth
    estimate = ppi_estimate(adversarial, trial.gold, trial.gold_index)
    gold = gold_only_estimate(trial.gold)

    assert estimate.lambda_ == 0.0
    assert estimate.half_width <= gold.half_width * 1.10


def test_gold_only_is_unbiased_and_wider_than_ppi() -> None:
    """Gold-only is centered on the truth on average, and pays for it in width."""
    rng = np.random.default_rng(5)
    points = []
    for _ in range(300):
        trial = simulate_trial(
            rng, n_total=1000, n_gold=100, true_rate=0.55, sensitivity=0.9, specificity=0.85
        )
        points.append(gold_only_estimate(trial.gold).point)
    assert float(np.mean(points)) == pytest.approx(0.55, abs=0.01)


def test_rogan_gladen_recovers_known_prevalence() -> None:
    """The misclassification correction recovers the true rate a biased judge hides."""
    rng = np.random.default_rng(6)
    trial = simulate_trial(
        rng, n_total=6000, n_gold=1500, true_rate=0.40, sensitivity=0.90, specificity=0.70
    )
    corrected = rogan_gladen_estimate(trial.judge, trial.gold, trial.gold_index)
    naive = judge_only_estimate(trial.judge)

    assert corrected.point == pytest.approx(0.40, abs=0.03)
    assert abs(corrected.point - 0.40) < abs(naive.point - 0.40)


def test_rogan_gladen_matches_hand_computed_case() -> None:
    """Hand-worked arithmetic on an exact confusion matrix.

    Gold subset: 100 positives of which the judge catches 90 (sensitivity 0.90); 100
    negatives of which the judge correctly rejects 70 (specificity 0.70). Judge positive
    rate over the whole set is fixed at 0.50 by construction, so

        p = (0.50 + 0.70 - 1) / (0.90 + 0.70 - 1) = 0.20 / 0.60 = 0.3333...
    """
    gold = np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])
    judge_on_gold = np.concatenate(
        [
            np.ones(90, dtype=int),
            np.zeros(10, dtype=int),  # 90/100 positives caught
            np.ones(30, dtype=int),
            np.zeros(70, dtype=int),
        ]  # 30/100 negatives flagged
    )
    # Extend to a full set whose judge positive rate is exactly 0.50.
    extra = np.concatenate([np.ones(80, dtype=int), np.zeros(120, dtype=int)])
    judge = np.concatenate([judge_on_gold, extra])
    assert judge.mean() == pytest.approx(0.50)

    estimate = rogan_gladen_estimate(judge, gold, np.arange(200))
    assert estimate.point == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_rogan_gladen_rejects_uninformative_judge() -> None:
    """A judge at or below chance carries no recoverable signal, so the method refuses."""
    gold = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])
    # Sensitivity 0.4, specificity 0.4: their sum is below 1.
    judge_on_gold = np.concatenate(
        [
            np.ones(20, dtype=int),
            np.zeros(30, dtype=int),
            np.ones(30, dtype=int),
            np.zeros(20, dtype=int),
        ]
    )
    judge = np.concatenate([judge_on_gold, np.zeros(100, dtype=int)])
    with pytest.raises(ValueError, match=r"sensitivity . specificity must exceed 1"):
        rogan_gladen_estimate(judge, gold, np.arange(100))


def test_ppi_requires_unlabeled_examples() -> None:
    """With every example labeled there is nothing to borrow, and the error says so."""
    judge = np.array([1, 0, 1, 1])
    gold = np.array([1, 0, 0, 1])
    with pytest.raises(ValueError, match="every example is gold-labeled"):
        ppi_estimate(judge, gold, np.arange(4))


def test_ppi_rejects_duplicate_gold_index() -> None:
    """A repeated index would double-count a label; that is a caller bug, not a warning."""
    judge = np.array([1, 0, 1, 1, 0])
    gold = np.array([1, 1])
    with pytest.raises(ValueError, match="duplicate positions"):
        ppi_estimate(judge, gold, np.array([2, 2]))


def test_estimates_record_their_assumptions() -> None:
    """Every estimate carries the conditions it depends on, for the report artifact."""
    rng = np.random.default_rng(7)
    trial = simulate_trial(
        rng, n_total=400, n_gold=60, true_rate=0.5, sensitivity=0.9, specificity=0.9
    )
    estimate = ppi_estimate(trial.judge, trial.gold, trial.gold_index)
    assert estimate.assumptions
    assert any("random sample" in a for a in estimate.assumptions)
