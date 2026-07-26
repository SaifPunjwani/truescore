"""Tests for truescore.sequential.

The claim is uniform-over-time coverage, so the tests check exactly that: not whether a
single interval covers, but whether a whole *trajectory* of intervals ever excludes the
truth. The companion test measures what a fixed-sample interval does under the same
peeking, which is the entire reason this module exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from truescore.agreement import wilson_interval
from truescore.sequential import confidence_sequence, first_exclusion, windowed_exclusion

METHODS = ("empirical_bernstein", "hoeffding")


@pytest.mark.parametrize("method", METHODS)
def test_confidence_sequence_covers_uniformly_over_time(method: str) -> None:
    """Across a whole stream, the sequence excludes the truth at most 5% of the time.

    Coverage is asserted over trajectories, not time points: a stream counts as a miss if
    the interval failed at *any* sample size. This is a strictly harder bar than
    fixed-sample coverage, and it is the property that licenses continuous monitoring.
    """
    replications, stream_length, true_rate = 300, 300, 0.30
    rng = np.random.default_rng(0)

    misses = 0
    for _ in range(replications):
        stream = rng.binomial(1, true_rate, stream_length)
        sequence = confidence_sequence(stream, method=method)  # type: ignore[arg-type]
        excluded = (sequence.lower_bounds > true_rate) | (sequence.upper_bounds < true_rate)
        misses += int(bool(np.any(excluded)))

    miss_rate = misses / replications
    assert miss_rate <= 0.05, f"{method}: uniform miss rate {miss_rate:.3f} exceeds the 5% budget"


def test_fixed_sample_intervals_fail_under_repeated_peeking() -> None:
    """The failure being corrected, measured rather than asserted.

    A 95% fixed-sample interval inspected after every observation misses the truth on a
    large fraction of streams. Any monitoring practice built on fixed-sample intervals
    inherits that false-alarm rate.
    """
    replications, stream_length, true_rate = 200, 300, 0.30
    rng = np.random.default_rng(1)

    misses = 0
    for _ in range(replications):
        stream = rng.binomial(1, true_rate, stream_length)
        successes = np.cumsum(stream)
        for t in range(1, stream_length + 1):
            interval = wilson_interval(int(successes[t - 1]), t)
            if not interval.low <= true_rate <= interval.high:
                misses += 1
                break

    assert misses / replications > 0.20, (
        "fixed-sample intervals should fail badly under peeking; if this ever passes "
        "cheaply the comparison in the docs is overstated"
    )


@pytest.mark.parametrize("method", METHODS)
def test_sequence_always_contains_the_running_mean(method: str) -> None:
    """A degenerate but load-bearing sanity check on the arithmetic."""
    rng = np.random.default_rng(2)
    stream = rng.binomial(1, 0.6, 400)
    sequence = confidence_sequence(stream, method=method)  # type: ignore[arg-type]
    assert sequence.low <= sequence.mean <= sequence.high


@pytest.mark.parametrize("method", METHODS)
def test_sequence_tightens_as_evidence_accumulates(method: str) -> None:
    """Width shrinks with sample size, and the running intersection never widens."""
    rng = np.random.default_rng(3)
    stream = rng.binomial(1, 0.5, 2000)
    sequence = confidence_sequence(stream, method=method)  # type: ignore[arg-type]

    widths = sequence.upper_bounds - sequence.lower_bounds
    assert widths[-1] < widths[49]
    assert np.all(np.diff(sequence.lower_bounds) >= -1e-12)
    assert np.all(np.diff(sequence.upper_bounds) <= 1e-12)


def test_empirical_bernstein_beats_hoeffding_on_a_low_variance_stream() -> None:
    """Adapting to variance is the point: near-certain outcomes should not pay full price."""
    rng = np.random.default_rng(4)
    stream = rng.binomial(1, 0.03, 3000)

    adaptive = confidence_sequence(stream, method="empirical_bernstein")
    worst_case = confidence_sequence(stream, method="hoeffding")
    assert adaptive.half_width < worst_case.half_width


def test_sequence_respects_custom_bounds() -> None:
    """A metric on a 1-to-5 scale is handled by rescaling, not by refusing."""
    rng = np.random.default_rng(5)
    ratings = rng.integers(1, 6, 500).astype(float)
    sequence = confidence_sequence(ratings, bounds=(1.0, 5.0))

    assert 1.0 <= sequence.low <= sequence.mean <= sequence.high <= 5.0
    assert sequence.low <= 3.0 <= sequence.high


def test_first_exclusion_detects_a_real_regression() -> None:
    """A genuine ten-point drop is caught, and caught early."""
    rng = np.random.default_rng(6)
    stream = rng.binomial(1, 0.70, 1500)
    stopped_at = first_exclusion(stream, 0.80, direction="below")

    assert stopped_at is not None
    assert stopped_at < 1500


def test_first_exclusion_false_alarm_rate_is_controlled() -> None:
    """Monitoring a metric that never changed raises an alarm at most alpha of the time.

    This is the guarantee a team actually cares about: watching the dashboard forever does
    not manufacture regressions.
    """
    replications, stream_length, rate = 300, 400, 0.75
    rng = np.random.default_rng(7)

    alarms = sum(
        int(
            first_exclusion(rng.binomial(1, rate, stream_length), rate, direction="below")
            is not None
        )
        for _ in range(replications)
    )
    assert alarms / replications <= 0.05


def test_first_exclusion_is_one_sided_when_asked() -> None:
    """An improvement must not trip a regression monitor."""
    rng = np.random.default_rng(8)
    stream = rng.binomial(1, 0.95, 1200)

    assert first_exclusion(stream, 0.70, direction="below") is None
    assert first_exclusion(stream, 0.70, direction="above") is not None


def test_invalid_arguments_are_rejected() -> None:
    rng = np.random.default_rng(9)
    stream = rng.binomial(1, 0.5, 50)

    with pytest.raises(ValueError, match="alpha must lie"):
        confidence_sequence(stream, alpha=0.0)
    with pytest.raises(ValueError, match="lambda_cap must lie"):
        confidence_sequence(stream, lambda_cap=1.0)
    with pytest.raises(ValueError, match="must lie inside bounds"):
        confidence_sequence(np.array([0.0, 2.0]))
    with pytest.raises(ValueError, match="bounds must satisfy"):
        confidence_sequence(stream, bounds=(1.0, 1.0))
    with pytest.raises(ValueError, match="non-finite"):
        confidence_sequence(np.array([0.5, np.nan]))
    with pytest.raises(ValueError, match="must be non-empty"):
        confidence_sequence(np.array([]))


def test_summary_states_the_guarantee() -> None:
    rng = np.random.default_rng(10)
    text = confidence_sequence(rng.binomial(1, 0.5, 100)).summary()
    assert "every sample size" in text


def test_windowed_exclusion_detects_a_late_regression() -> None:
    """A drop after a long healthy period is caught, which the cumulative test misses.

    The cumulative mean is held up by the healthy prefix, so asking "is the mean of
    everything below the baseline?" answers no long after the service has degraded.
    Monitoring recent windows asks the question an operator actually has.
    """
    rng = np.random.default_rng(20)
    healthy = rng.binomial(1, 0.88, 600)
    regressed = rng.binomial(1, 0.78, 600)
    stream = np.concatenate([healthy, regressed])

    assert first_exclusion(stream, 0.88, direction="below") is None, (
        "the cumulative sequence is expected to miss this; that is why windows exist"
    )

    alarm = windowed_exclusion(stream, 0.88, window=300)
    assert alarm is not None
    assert alarm > 600, "the alarm must not fire before the regression began"


def test_windowed_exclusion_false_alarm_rate_is_controlled() -> None:
    """Splitting the budget across windows keeps the whole run inside alpha."""
    replications, rate = 300, 0.85
    rng = np.random.default_rng(21)
    alarms = sum(
        int(windowed_exclusion(rng.binomial(1, rate, 900), rate, window=300) is not None)
        for _ in range(replications)
    )
    assert alarms / replications <= 0.05


def test_windowed_exclusion_handles_short_streams() -> None:
    """Fewer observations than one window is not an error, just no verdict yet."""
    rng = np.random.default_rng(22)
    assert windowed_exclusion(rng.binomial(1, 0.5, 50), 0.9, window=300) is None


def test_windowed_exclusion_validates_its_arguments() -> None:
    rng = np.random.default_rng(23)
    stream = rng.binomial(1, 0.5, 100)
    with pytest.raises(ValueError, match="window must be at least 2"):
        windowed_exclusion(stream, 0.5, window=1)
    with pytest.raises(ValueError, match="planned_windows must be at least 1"):
        windowed_exclusion(stream, 0.5, window=10, planned_windows=0)


def test_one_sided_budget_is_tighter_than_two_sided() -> None:
    """Spending the budget on one tail buys power on that tail."""
    rng = np.random.default_rng(24)
    stream = rng.binomial(1, 0.8, 500)
    two_sided = confidence_sequence(stream, one_sided=False)
    one_sided = confidence_sequence(stream, one_sided=True)
    assert one_sided.high < two_sided.high
