"""Tests for truescore.weighting."""

from __future__ import annotations

import numpy as np
import pytest

from truescore.weighting import eval_composition, post_stratified_estimate

EASY_RATE, HARD_RATE = 0.92, 0.55
PRODUCTION = {"easy": 0.80, "hard": 0.20}
PRODUCTION_TRUTH = 0.80 * EASY_RATE + 0.20 * HARD_RATE  # 0.846


def _curated_eval(
    rng: np.random.Generator, n_easy: int = 1000, n_hard: int = 3000, n_gold: int = 800
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """An evaluation set that over-samples hard questions three to one, as they do."""
    strata = np.array(["easy"] * n_easy + ["hard"] * n_hard)
    truth = np.concatenate([rng.binomial(1, EASY_RATE, n_easy), rng.binomial(1, HARD_RATE, n_hard)])
    judge = np.where(
        truth == 1,
        rng.binomial(1, 0.95, n_easy + n_hard),
        rng.binomial(1, 0.35, n_easy + n_hard),
    )
    gold_index = np.sort(rng.choice(n_easy + n_hard, n_gold, replace=False))
    return judge, truth, gold_index, strata


def test_reweighting_recovers_the_production_rate() -> None:
    """The eval set says one thing; production traffic would say another.

    The evaluation set is 75% hard questions and production is 20% hard, so the raw number
    is pessimistic by about ten points. Reweighting to the production mix recovers the rate
    customers would actually experience.
    """
    rng = np.random.default_rng(0)
    judge, truth, gold_index, strata = _curated_eval(rng)

    result = post_stratified_estimate(judge, truth[gold_index], gold_index, strata, PRODUCTION)

    assert result.low <= PRODUCTION_TRUTH <= result.high
    assert abs(result.point - PRODUCTION_TRUTH) < 0.03
    assert result.composition_effect > 0.05, "the eval set's composition was worth points"


def test_the_unweighted_number_is_the_one_that_is_wrong() -> None:
    """Stated as an assertion so the motivation cannot quietly stop being true."""
    rng = np.random.default_rng(1)
    judge, truth, gold_index, strata = _curated_eval(rng)
    result = post_stratified_estimate(judge, truth[gold_index], gold_index, strata, PRODUCTION)
    assert abs(result.unweighted.point - PRODUCTION_TRUTH) > abs(result.point - PRODUCTION_TRUTH)


def test_weighted_interval_covers_the_production_rate() -> None:
    """Coverage of the weighted estimator, simulated rather than assumed."""
    replications = 300
    rng = np.random.default_rng(2)
    covered = 0
    for _ in range(replications):
        judge, truth, gold_index, strata = _curated_eval(rng, 500, 1500, 400)
        result = post_stratified_estimate(judge, truth[gold_index], gold_index, strata, PRODUCTION)
        covered += int(result.low <= PRODUCTION_TRUTH <= result.high)
    rate = covered / replications
    assert 0.90 <= rate <= 0.99, f"observed coverage {rate:.3f}"


def test_weights_matching_the_eval_composition_change_almost_nothing() -> None:
    """If the eval set already matches production, reweighting is close to a no-op."""
    rng = np.random.default_rng(3)
    judge, truth, gold_index, strata = _curated_eval(rng)
    matching = eval_composition(strata)

    result = post_stratified_estimate(judge, truth[gold_index], gold_index, strata, dict(matching))
    unmatched = post_stratified_estimate(judge, truth[gold_index], gold_index, strata, PRODUCTION)
    assert abs(result.composition_effect) < abs(unmatched.composition_effect)


def test_weights_are_normalized_not_required_to_sum_to_one() -> None:
    """Counts from a traffic log are as acceptable as proportions."""
    rng = np.random.default_rng(4)
    judge, truth, gold_index, strata = _curated_eval(rng)

    proportions = post_stratified_estimate(
        judge, truth[gold_index], gold_index, strata, {"easy": 0.8, "hard": 0.2}
    )
    counts = post_stratified_estimate(
        judge, truth[gold_index], gold_index, strata, {"easy": 8000, "hard": 2000}
    )
    assert counts.point == pytest.approx(proportions.point)


def test_missing_stratum_weight_is_rejected() -> None:
    """Dropping a stratum silently would change what is being estimated."""
    rng = np.random.default_rng(5)
    judge, truth, gold_index, strata = _curated_eval(rng)
    with pytest.raises(ValueError, match=r"no production weight given for strata \['hard'\]"):
        post_stratified_estimate(judge, truth[gold_index], gold_index, strata, {"easy": 1.0})


def test_degenerate_weights_are_rejected() -> None:
    rng = np.random.default_rng(6)
    judge, truth, gold_index, strata = _curated_eval(rng)
    with pytest.raises(ValueError, match="non-negative"):
        post_stratified_estimate(
            judge, truth[gold_index], gold_index, strata, {"easy": -1.0, "hard": 2.0}
        )
    with pytest.raises(ValueError, match="positive value"):
        post_stratified_estimate(
            judge, truth[gold_index], gold_index, strata, {"easy": 0.0, "hard": 0.0}
        )


def test_a_stratum_with_no_human_labels_is_rejected() -> None:
    """An unlabeled stratum makes the weighted total undefined, so it raises."""
    rng = np.random.default_rng(7)
    strata = np.array(["easy"] * 500 + ["hard"] * 500)
    truth = rng.binomial(1, 0.7, 1000)
    judge = truth.copy()
    gold_index = np.arange(300)  # only the easy stratum gets labels
    with pytest.raises(ValueError, match="no human labels"):
        post_stratified_estimate(judge, truth[gold_index], gold_index, strata, PRODUCTION)


def test_thin_strata_fall_back_to_the_classical_estimator() -> None:
    """Too few labels for PPI is not a failure; it is a different, wider estimator."""
    rng = np.random.default_rng(8)
    strata = np.array(["big"] * 1800 + ["small"] * 200)
    truth = np.concatenate([rng.binomial(1, 0.8, 1800), rng.binomial(1, 0.6, 200)])
    judge = np.where(truth == 1, rng.binomial(1, 0.95, 2000), rng.binomial(1, 0.3, 2000))
    gold_index = np.sort(
        np.concatenate([rng.choice(1800, 400, replace=False), np.arange(1800, 1805)])
    )

    result = post_stratified_estimate(
        judge, truth[gold_index], gold_index, strata, {"big": 0.5, "small": 0.5}
    )
    methods = {s.name: s.method for s in result.strata}
    assert methods["big"] == "ppi++"
    assert methods["small"].startswith("gold_only")


def test_report_carries_its_assumptions_and_reads_clearly() -> None:
    rng = np.random.default_rng(9)
    judge, truth, gold_index, strata = _curated_eval(rng)
    result = post_stratified_estimate(judge, truth[gold_index], gold_index, strata, PRODUCTION)
    text = result.summary()

    assert "weighted to production mix" in text
    assert "composition effect" in text
    assert any("production weights describe current traffic" in a for a in result.assumptions)
    assert any("not a biased sample inside a stratum" in a for a in result.assumptions)


def test_eval_composition_reports_shares() -> None:
    strata = np.array(["a"] * 3 + ["b"] * 1)
    assert eval_composition(strata) == {"a": 0.75, "b": 0.25}


def test_mismatched_strata_length_is_rejected() -> None:
    rng = np.random.default_rng(10)
    judge = rng.binomial(1, 0.7, 100).astype(float)
    with pytest.raises(ValueError, match="strata must cover every example"):
        post_stratified_estimate(judge, judge[:30], np.arange(30), np.array(["a"] * 50), {"a": 1.0})
