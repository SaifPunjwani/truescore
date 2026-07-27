"""Tests for truescore.agreement."""

from __future__ import annotations

import numpy as np
import pytest

from tests.simulate import coverage_bounds
from truescore.agreement import (
    cohen_kappa,
    graded_agreement,
    gwet_ac1,
    judge_agreement,
    krippendorff_alpha,
    quadratic_weighted_kappa,
    wilson_interval,
)


def test_wilson_interval_matches_hand_computed_case() -> None:
    """8 successes in 10 trials.

    z = 1.959964, denominator = 1 + z²/10 = 1.38415,
    centre = (0.8 + z²/20) / 1.38415 = 0.716737,
    half   = (z / 1.38415)·sqrt(0.8·0.2/10 + z²/400) = 0.226578,
    giving [0.4902, 0.9433] -- the value tabulated for the Wilson interval at 8/10.
    """
    interval = wilson_interval(8, 10)
    assert interval.point == pytest.approx(0.8)
    assert interval.low == pytest.approx(0.4902, abs=1e-4)
    assert interval.high == pytest.approx(0.9433, abs=1e-4)


def test_wilson_interval_stays_inside_the_unit_range_at_extremes() -> None:
    """The failure mode the normal approximation has and Wilson does not."""
    interval = wilson_interval(0, 20)
    assert interval.low == 0.0
    assert 0.0 < interval.high < 1.0
    interval = wilson_interval(20, 20)
    assert interval.high == 1.0
    assert 0.0 < interval.low < 1.0


def test_wilson_interval_covers_at_nominal_rate() -> None:
    """Simulated coverage of the Wilson interval at a small, awkward sample size."""
    replications = 2000
    rng = np.random.default_rng(11)
    p = 0.15
    draws = rng.binomial(30, p, replications)
    covered = sum(
        1 for k in draws if wilson_interval(int(k), 30).low <= p <= wilson_interval(int(k), 30).high
    )
    low, _ = coverage_bounds(replications, n_sigma=4.0)
    assert low <= covered / replications <= 1.0, "Wilson may over-cover but must not under-cover"
    assert covered / replications >= low


def test_cohen_kappa_matches_hand_computed_case() -> None:
    """40 both-positive, 40 both-negative, 10 disagreements each way.

    Observed agreement 0.80; each rater is positive half the time so chance agreement is
    0.5·0.5 + 0.5·0.5 = 0.50, giving κ = (0.80 − 0.50)/(1 − 0.50) = 0.60.
    """
    a = np.array([1] * 50 + [0] * 50)
    b = np.array([1] * 40 + [0] * 10 + [1] * 10 + [0] * 40)
    assert cohen_kappa(a, b) == pytest.approx(0.60, abs=1e-12)


def test_gwet_ac1_is_stable_under_imbalance_where_kappa_collapses() -> None:
    """The kappa paradox, and why AC1 is reported alongside it.

    95 of 100 items agree, but 97.5% of labels are positive. κ collapses to roughly zero
    (it credits nearly all agreement to chance), while AC1 stays near the observed 0.95.
    A team reading κ alone would conclude their judge is worthless when it agrees 95% of
    the time.
    """
    a = np.array([1] * 95 + [1] * 3 + [0] * 2)
    b = np.array([1] * 95 + [0] * 3 + [1] * 2)
    kappa = cohen_kappa(a, b)
    ac1 = gwet_ac1(a, b)

    assert abs(kappa) < 0.05, f"kappa should collapse under imbalance; got {kappa}"
    assert ac1 > 0.90, f"AC1 should track the observed agreement; got {ac1}"


def test_cohen_kappa_is_one_for_perfect_agreement() -> None:
    a = np.array([1, 0, 1, 0, 1])
    assert cohen_kappa(a, a) == pytest.approx(1.0)
    assert gwet_ac1(a, a) == pytest.approx(1.0)


def test_krippendorff_alpha_matches_published_example() -> None:
    """Krippendorff's canonical three-coder example with missing ratings.

    Published values for this dataset: α_nominal = 0.691, α_interval = 0.811
    (Krippendorff 2011, "Computing Krippendorff's Alpha-Reliability").
    """
    nan = np.nan
    ratings = np.array(
        [
            [nan, nan, nan, nan, nan, 3, 4, 1, 2, 1, 1, 3, 3, nan, 3],
            [1, nan, 2, 1, 3, 3, 4, 3, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, 2, 1, 3, 4, 4, nan, 2, 1, 1, 3, 3, nan, 4],
        ]
    )
    assert krippendorff_alpha(ratings, level="nominal") == pytest.approx(0.691, abs=0.002)
    assert krippendorff_alpha(ratings, level="interval") == pytest.approx(0.811, abs=0.002)


def test_krippendorff_alpha_is_one_for_identical_coders() -> None:
    ratings = np.array([[1.0, 2.0, 3.0, 1.0], [1.0, 2.0, 3.0, 1.0]])
    assert krippendorff_alpha(ratings) == pytest.approx(1.0)


def test_krippendorff_alpha_requires_pairable_units() -> None:
    ratings = np.array([[1.0, np.nan], [np.nan, 2.0]])
    with pytest.raises(ValueError, match="two or more ratings"):
        krippendorff_alpha(ratings)


def test_judge_agreement_matches_hand_computed_confusion() -> None:
    """Confusion counts, sensitivity and specificity on a constructed set.

    60 positives of which the judge catches 54 (sensitivity 0.90); 40 negatives of which
    it correctly rejects 32 (specificity 0.80); accuracy (54 + 32)/100 = 0.86.
    """
    gold = np.array([1] * 60 + [0] * 40)
    judge = np.array([1] * 54 + [0] * 6 + [1] * 8 + [0] * 32)
    report = judge_agreement(judge, gold, n_bootstrap=200)

    assert (report.true_positives, report.false_negatives) == (54, 6)
    assert (report.false_positives, report.true_negatives) == (8, 32)
    assert report.sensitivity.point == pytest.approx(0.90)
    assert report.specificity.point == pytest.approx(0.80)
    assert report.accuracy.point == pytest.approx(0.86)
    assert report.gold_prevalence == pytest.approx(0.60)


def test_judge_agreement_rejects_single_class_gold() -> None:
    """Sensitivity or specificity is undefined without both classes, so it raises."""
    with pytest.raises(ValueError, match="both classes"):
        judge_agreement(np.array([1, 1, 0]), np.array([1, 1, 1]))


def test_judge_agreement_summary_is_readable() -> None:
    gold = np.array([1] * 30 + [0] * 30)
    judge = np.array([1] * 27 + [0] * 3 + [1] * 5 + [0] * 25)
    text = judge_agreement(judge, gold, n_bootstrap=100).summary()
    for expected in ("accuracy", "sensitivity", "Cohen", "Gwet", "confusion"):
        assert expected in text


def test_binary_validation_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        judge_agreement(np.array([0, 1, 2]), np.array([0, 1, 1]))


def test_quadratic_kappa_matches_hand_computed_case() -> None:
    """A 3x3 confusion matrix worked out by hand.

    16 pairs over levels {1,2,3}: 12 agreements on the diagonal, 2 off-by-one at (1,2) and
    2 at (2,3). Building the observed and expected matrices and applying
    w_ij = (i-j)^2/(k-1)^2 gives kappa = 0.804878.
    """
    judge = np.array([1] * 4 + [2] * 4 + [3] * 4 + [1] * 2 + [2] * 2)
    gold = np.array([1] * 4 + [2] * 4 + [3] * 4 + [2] * 2 + [3] * 2)
    assert quadratic_weighted_kappa(judge, gold) == pytest.approx(0.804878, abs=1e-6)


def test_quadratic_kappa_punishes_distant_disagreements() -> None:
    """The reason to weight at all: a 1-vs-5 must cost more than a 3-vs-4.

    Both judges below disagree on every example and share the same marginals, so
    unweighted kappa cannot tell them apart. The quadratic weighting can.
    """
    gold = np.array([1, 2, 3, 4, 5] * 20)
    near = np.array([2, 3, 4, 5, 1] * 20)
    far = np.array([5, 4, 3, 2, 1] * 20)
    assert quadratic_weighted_kappa(near, gold) > quadratic_weighted_kappa(far, gold)


def test_quadratic_kappa_is_one_for_perfect_agreement() -> None:
    scores = np.array([1, 3, 5, 2, 4, 1, 5])
    assert quadratic_weighted_kappa(scores, scores) == pytest.approx(1.0)


def test_quadratic_kappa_handles_a_single_observed_level() -> None:
    """No chance disagreement is possible, so agreement is total rather than undefined."""
    assert quadratic_weighted_kappa(np.array([3, 3, 3]), np.array([3, 3, 3])) == 1.0


def test_graded_agreement_on_a_rubric() -> None:
    """A judge grading about half a level high on a 1-5 rubric."""
    rng = np.random.default_rng(0)
    gold = rng.integers(1, 6, 600)
    judge = np.clip(np.round(gold + 0.6 + rng.normal(0, 0.8, 600)), 1, 5)

    report = graded_agreement(judge, gold)

    assert report.n == 600
    assert report.mean_error > 0.3, "the judge grades high, and the report should say so"
    assert report.exact_match < report.within_one
    assert 0.7 < report.quadratic_kappa < 1.0
    assert report.spearman > 0.7
    assert "grades high" in report.summary()


def test_graded_agreement_reports_perfect_agreement_cleanly() -> None:
    scores = np.array([1, 2, 3, 4, 5] * 10)
    report = graded_agreement(scores, scores)
    assert report.exact_match == 1.0
    assert report.mean_absolute_error == 0.0
    assert report.quadratic_kappa == pytest.approx(1.0)


def test_graded_agreement_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        graded_agreement(np.array([1, 2, 3]), np.array([1, 2]))
