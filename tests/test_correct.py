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


def test_ppi_covers_a_near_deterministic_subgroup() -> None:
    """The failure mode that a high-performing slice walks straight into.

    When a subgroup is right ~99.7% of the time, a gold sample of a few dozen labels is
    entirely 1s more often than not. The sample variance is then exactly zero, and an
    asymptotic interval built from it has width zero and misses the truth. Measured before
    the fix: 7.7% coverage. The estimator now widens to the exact interval whenever one
    class has fewer than five observations.
    """
    replications, true_rate = 300, 0.997
    rng = np.random.default_rng(41)
    covered = 0
    degenerate = 0
    for _ in range(replications):
        truth = rng.binomial(1, true_rate, 2000)
        judge = np.where(truth == 1, rng.binomial(1, 0.98, 2000), rng.binomial(1, 0.2, 2000))
        index = np.sort(rng.choice(2000, 30, replace=False))
        gold = truth[index]
        degenerate += int(gold.std(ddof=1) == 0)
        estimate = ppi_estimate(judge, gold, index)
        covered += int(estimate.low <= true_rate <= estimate.high)

    assert degenerate / replications > 0.5, "this scenario should be mostly degenerate"
    assert covered / replications >= 0.93, f"coverage was {covered / replications:.3f}"


def test_the_widening_is_reported_in_the_method() -> None:
    """A widened interval says so, because a reader deserves to know which rule applied."""
    rng = np.random.default_rng(42)
    truth = np.ones(1000, dtype=int)
    truth[:3] = 0
    judge = truth.copy()
    index = np.sort(rng.choice(1000, 40, replace=False))
    estimate = ppi_estimate(judge, truth[index], index)
    assert "exact interval" in estimate.method


def test_ppi_keeps_its_advantage_when_the_approximation_is_sound() -> None:
    """The fix must be targeted: where the normal approximation is fine, PPI still wins."""
    rng = np.random.default_rng(43)
    ppi_widths, gold_widths = [], []
    for _ in range(150):
        truth = rng.binomial(1, 0.7, 4000)
        judge = np.where(truth == 1, rng.binomial(1, 0.95, 4000), rng.binomial(1, 0.08, 4000))
        index = np.sort(rng.choice(4000, 400, replace=False))
        ppi_widths.append(ppi_estimate(judge, truth[index], index).half_width)
        gold_widths.append(gold_only_estimate(truth[index]).half_width)
    assert float(np.mean(ppi_widths)) < 0.75 * float(np.mean(gold_widths))


def test_clustered_variance_reduces_to_the_iid_formula_with_singleton_clusters() -> None:
    """One observation per cluster is the independent case, and must give the same answer.

    This is what makes the cluster-robust path safe as a general replacement rather than a
    separate branch that could drift away from the estimator it is meant to generalize.
    """
    rng = np.random.default_rng(3)
    values = rng.random(80)

    plain = gold_only_estimate(values)
    singletons = gold_only_estimate(values, clusters=np.arange(80))

    assert plain.point == pytest.approx(singletons.point)
    assert plain.low == pytest.approx(singletons.low, abs=1e-12)
    assert plain.high == pytest.approx(singletons.high, abs=1e-12)


def test_clustered_data_undercovers_until_clusters_are_declared() -> None:
    """Repeated epochs of one sample are one draw looked at five times, not five draws.

    An eval run with --epochs 5 produces five correlated rows per sample. Treating them as
    independent shrinks the interval by roughly sqrt(5) more than the data supports, and a
    nominal 95% interval then covers about 86% of the time. Measured, because a coverage
    claim that is asserted rather than simulated is the claim most likely to be wrong.
    """
    rng = np.random.default_rng(0)
    samples, epochs, reps = 200, 5, 600
    naive = clustered = 0
    for _ in range(reps):
        difficulty = rng.beta(2, 2, size=samples)
        outcomes = (rng.random((samples, epochs)) < difficulty[:, None]).astype(float)
        flat = outcomes.ravel()
        groups = np.repeat(np.arange(samples), epochs)
        truth = 0.5  # the mean of Beta(2, 2)

        without = gold_only_estimate(flat)
        with_clusters = gold_only_estimate(flat, clusters=groups)
        naive += without.low <= truth <= without.high
        clustered += with_clusters.low <= truth <= with_clusters.high

    assert naive / reps < 0.90, "the bug this guards against did not reproduce"
    assert 0.93 <= clustered / reps <= 0.97


def test_ppi_covers_clustered_data_when_clusters_are_declared() -> None:
    """The same guarantee for the corrected estimate, with whole clusters labeled."""
    rng = np.random.default_rng(1)
    samples, epochs, labeled_samples, reps = 300, 4, 90, 500
    covered = 0
    for _ in range(reps):
        difficulty = rng.beta(2, 2, size=samples)
        gold = (rng.random((samples, epochs)) < difficulty[:, None]).astype(float)
        # A judge that is right most of the time, wrong in a correlated way per sample.
        flips = rng.random((samples, epochs)) < (0.12 + 0.1 * difficulty[:, None])
        judge = np.where(flips, 1.0 - gold, gold)
        groups = np.repeat(np.arange(samples), epochs)

        chosen = rng.choice(samples, size=labeled_samples, replace=False)
        index = np.flatnonzero(np.isin(groups, chosen))
        estimate = ppi_estimate(judge.ravel(), gold.ravel()[index], index, clusters=groups)
        covered += estimate.low <= 0.5 <= estimate.high

    assert 0.92 <= covered / reps <= 0.98


def test_ppi_refuses_a_cluster_split_across_the_labeled_boundary() -> None:
    """Half-labeled clusters break the independence the two terms rely on."""
    judge = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    index = np.array([0, 2, 4])  # takes one member of cluster 0, 1 and 2 each

    with pytest.raises(ValueError, match="some examples labeled and some not"):
        ppi_estimate(judge, np.array([1.0, 1.0, 0.0]), index, clusters=groups)


def test_a_single_cluster_cannot_support_an_interval() -> None:
    with pytest.raises(ValueError, match="at least 2 clusters"):
        gold_only_estimate(np.array([1.0, 0.0, 1.0, 0.0]), clusters=np.zeros(4))


def test_clusters_must_cover_every_observation() -> None:
    with pytest.raises(ValueError, match="every observation needs to say"):
        gold_only_estimate(np.array([1.0, 0.0, 1.0]), clusters=np.array([0, 1]))
