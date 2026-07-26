"""Tests for truescore.drift."""

from __future__ import annotations

import numpy as np
import pytest

from truescore.drift import anchor_fingerprint, judge_drift, monitor_agreement


def _anchor(rng: np.random.Generator, n: int, agreement: float) -> tuple[np.ndarray, np.ndarray]:
    """Gold labels and a judge run agreeing with them at the requested rate."""
    gold = rng.binomial(1, 0.6, n)
    agrees = rng.binomial(1, agreement, n)
    judge = np.where(agrees == 1, gold, 1 - gold)
    return judge, gold


def test_no_drift_is_not_flagged() -> None:
    """Two runs of an unchanged judge on the same anchor set look unchanged."""
    rng = np.random.default_rng(0)
    judge, gold = _anchor(rng, 400, 0.88)
    report = judge_drift(judge, judge.copy(), gold)

    assert report.agreement_change == pytest.approx(0.0)
    assert not report.agreement_changed
    assert not report.behavior_changed
    assert report.flip_rate == 0.0
    assert "no detectable change" in report.summary()


def test_drift_type_one_error_is_controlled() -> None:
    """Independent re-runs of a stable stochastic judge rarely trip the alarm."""
    replications = 400
    rng = np.random.default_rng(1)
    false_alarms = 0
    for _ in range(replications):
        gold = rng.binomial(1, 0.6, 300)
        baseline = np.where(rng.binomial(1, 0.85, 300) == 1, gold, 1 - gold)
        current = np.where(rng.binomial(1, 0.85, 300) == 1, gold, 1 - gold)
        false_alarms += int(judge_drift(baseline, current, gold).agreement_changed)

    assert false_alarms / replications <= 0.075


def test_real_degradation_is_flagged() -> None:
    """A judge that drops from 90% to 75% agreement is caught."""
    rng = np.random.default_rng(2)
    gold = rng.binomial(1, 0.6, 600)
    baseline = np.where(rng.binomial(1, 0.90, 600) == 1, gold, 1 - gold)
    current = np.where(rng.binomial(1, 0.75, 600) == 1, gold, 1 - gold)

    report = judge_drift(baseline, current, gold)
    assert report.agreement_changed
    assert report.agreement_change < 0
    assert report.high < 0, "the interval should exclude 'no change'"
    assert "degraded" in report.summary()


def test_flip_rate_detects_a_rewritten_judge_at_equal_accuracy() -> None:
    """The signal an accuracy dashboard cannot show.

    Both runs agree with gold at 80%, so agreement is flat -- but they disagree with each
    other on a third of examples, because the second judge is a different model that is
    right about different things. Any downstream metric computed per-example changed
    materially while the headline number did not move.
    """
    rng = np.random.default_rng(3)
    n, n_errors = 900, 180
    gold = rng.binomial(1, 0.5, n)

    # Both runs make exactly 180 errors, so accuracy is identical to the last digit --
    # but on disjoint example sets, so every one of those 360 verdicts changed.
    shuffled = rng.permutation(n)
    baseline_errors = shuffled[:n_errors]
    current_errors = shuffled[n_errors : 2 * n_errors]

    baseline = gold.copy()
    baseline[baseline_errors] = 1 - baseline[baseline_errors]
    current = gold.copy()
    current[current_errors] = 1 - current[current_errors]

    report = judge_drift(baseline, current, gold)
    assert report.baseline_agreement == pytest.approx(report.current_agreement)

    assert not report.agreement_changed, "accuracy is unchanged by construction"
    assert report.behavior_changed, "but the verdicts moved and that must be visible"
    assert report.flip_rate > 0.25
    assert "behaves differently" in report.summary()


def test_fingerprint_is_stable_and_changes_with_the_anchor_set() -> None:
    """Comparisons are only meaningful against an unchanged anchor set."""
    gold = np.array([1, 0, 1, 1, 0])
    assert anchor_fingerprint(gold) == anchor_fingerprint(gold.copy())
    assert anchor_fingerprint(gold) != anchor_fingerprint(np.array([1, 0, 1, 1, 1]))
    assert anchor_fingerprint(gold) != anchor_fingerprint(gold[::-1])


def test_fingerprint_incorporates_example_ids() -> None:
    """Same labels on different examples is a different anchor set."""
    gold = np.array([1, 0, 1, 0])
    ids_a = np.array([10, 11, 12, 13])
    ids_b = np.array([10, 11, 12, 99])
    report_a = judge_drift(gold, gold, gold, example_ids=ids_a)
    report_b = judge_drift(gold, gold, gold, example_ids=ids_b)
    assert report_a.fingerprint != report_b.fingerprint


def test_monitor_raises_on_degradation_and_rarely_otherwise() -> None:
    """The live monitor: alarms on a real drop, quiet when nothing changed."""
    rng = np.random.default_rng(4)
    degraded = rng.binomial(1, 0.70, 2000)
    assert monitor_agreement(degraded, 0.88) is not None

    replications = 200
    false_alarms = sum(
        int(monitor_agreement(rng.binomial(1, 0.88, 600), 0.88) is not None)
        for _ in range(replications)
    )
    assert false_alarms / replications <= 0.05


def test_monitor_validates_its_baseline() -> None:
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError, match="baseline_agreement must lie"):
        monitor_agreement(rng.binomial(1, 0.5, 20), 1.5)


def test_drift_rejects_mismatched_anchor_arrays() -> None:
    with pytest.raises(ValueError, match="same length"):
        judge_drift(np.array([1, 0, 1]), np.array([1, 0]), np.array([1, 0, 1]))


def test_drift_report_records_the_sample_size_and_level() -> None:
    rng = np.random.default_rng(6)
    judge, gold = _anchor(rng, 250, 0.9)
    report = judge_drift(judge, judge, gold, alpha=0.01)
    assert report.n_anchor == 250
    assert report.level == pytest.approx(0.99)
