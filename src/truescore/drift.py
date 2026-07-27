# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Saif Punjwani
"""Did my judge change under me?

Judges are usually hosted models behind a version-less endpoint. When the provider
updates one, every downstream eval number moves, with no change in your code and nothing
in your logs to explain it. Teams discover this as an unexplained step in a dashboard,
weeks later, and often misattribute it to their own release.

The defense is an **anchor set**: a frozen sample of examples with trusted labels, re-run
through the judge on a schedule. Because it is the same examples every time, the
comparison is paired, and the question "did the judge change?" becomes a paired test
rather than an eyeball comparison of two rates.

Two signals, and they answer different questions:

- **agreement change** -- did the judge get better or worse against the gold labels?
- **label-flip rate** -- how many individual verdicts changed, in either direction?

A judge can rewrite a third of its verdicts while its accuracy stays flat, which reads as
"nothing happened" on a dashboard and is emphatically something happening. The flip rate
catches that; the agreement change does not.

References:
    tests/test_drift.py::test_flip_rate_detects_a_rewritten_judge_at_equal_accuracy
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from truescore._validation import check_alpha, check_binary, check_same_length
from truescore.compare import mcnemar
from truescore.sequential import first_exclusion

__all__ = ["DriftReport", "anchor_fingerprint", "judge_drift", "monitor_agreement"]


def anchor_fingerprint(*arrays: npt.ArrayLike) -> str:
    """Stable fingerprint of an anchor set, so comparisons are provably like-for-like.

    A drift comparison is meaningless if the anchor set silently changed between runs.
    Recording this fingerprint alongside each measurement turns that failure from an
    invisible confound into a mismatch you can see.

    Args:
        *arrays: The arrays defining the anchor set -- gold labels, and optionally example
            identifiers or prompt hashes.

    Returns:
        A 16-character hexadecimal digest.

    References:
        tests/test_drift.py::test_fingerprint_changes_when_the_anchor_set_changes
    """
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(np.asarray(array))
        digest.update(str(values.dtype).encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class DriftReport:
    """Whether a judge's behavior changed between two runs on the same anchor set.

    Attributes:
        n_anchor: Anchor examples compared.
        baseline_agreement: Agreement with gold at the baseline run.
        current_agreement: Agreement with gold at the current run.
        agreement_change: Current minus baseline.
        low: Lower confidence limit on the change.
        high: Upper confidence limit on the change.
        p_value: Paired test of no change in agreement.
        flip_rate: Fraction of anchor examples whose judge verdict changed at all.
        flip_low: Lower confidence limit on the flip rate.
        flip_high: Upper confidence limit on the flip rate.
        fingerprint: Anchor-set fingerprint, recorded so the comparison is auditable.
        level: Nominal coverage of the intervals.
    """

    n_anchor: int
    baseline_agreement: float
    current_agreement: float
    agreement_change: float
    low: float
    high: float
    p_value: float
    flip_rate: float
    flip_low: float
    flip_high: float
    fingerprint: str
    level: float

    @property
    def agreement_changed(self) -> bool:
        """Whether the change in agreement is distinguishable from zero."""
        return self.p_value < (1.0 - self.level)

    @property
    def behavior_changed(self) -> bool:
        """Whether any verdicts changed at all, beyond sampling noise."""
        return self.flip_low > 0.0

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        if self.agreement_changed:
            direction = "improved" if self.agreement_change > 0 else "degraded"
            verdict = f"agreement {direction} by {abs(self.agreement_change):.4f}"
        elif self.behavior_changed:
            verdict = (
                "agreement is unchanged, but individual verdicts moved -- the judge "
                "behaves differently on the same inputs"
            )
        else:
            verdict = "no detectable change"
        return "\n".join(
            [
                f"judge drift on anchor set {self.fingerprint} (n={self.n_anchor})",
                f"  agreement: {self.baseline_agreement:.4f} -> {self.current_agreement:.4f} "
                f"(change {self.agreement_change:+.4f} "
                f"[{self.low:+.4f}, {self.high:+.4f}], p={self.p_value:.4g})",
                f"  verdicts changed: {self.flip_rate:.4f} "
                f"[{self.flip_low:.4f}, {self.flip_high:.4f}]",
                f"  {verdict}",
            ]
        )


def judge_drift(
    baseline_judge: npt.ArrayLike,
    current_judge: npt.ArrayLike,
    gold: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    example_ids: npt.ArrayLike | None = None,
) -> DriftReport:
    """Compare two judge runs over the same gold-labeled anchor set.

    Args:
        baseline_judge: Judge verdicts from the reference run.
        current_judge: Judge verdicts from the current run, on the same examples in the
            same order.
        gold: Trusted labels for the anchor examples.
        alpha: Significance level for every interval.
        example_ids: Optional identifiers folded into the fingerprint, so a reordered or
            partially replaced anchor set is detectable.

    Returns:
        A :class:`DriftReport`.

    Raises:
        ValueError: If the three arrays are not binary and equal in length.

    References:
        tests/test_drift.py::test_no_drift_is_not_flagged
        tests/test_drift.py::test_real_degradation_is_flagged
    """
    check_alpha(alpha)
    baseline = check_binary("baseline_judge", baseline_judge)
    current = check_binary("current_judge", current_judge)
    truth = check_binary("gold", gold)
    check_same_length("baseline_judge", baseline, "current_judge", current)
    check_same_length("baseline_judge", baseline, "gold", truth)

    baseline_agrees = (baseline == truth).astype(np.int64)
    current_agrees = (current == truth).astype(np.int64)

    # Same examples in both runs, so the comparison is paired; McNemar uses exactly the
    # examples whose agreement status changed and ignores the rest.
    comparison = mcnemar(current_agrees, baseline_agrees, alpha=alpha)

    n = int(baseline.shape[0])
    flips = int(np.sum(baseline != current))
    flip_interval = stats.binomtest(flips, n).proportion_ci(
        confidence_level=1.0 - alpha, method="exact"
    )

    fingerprint = (
        anchor_fingerprint(truth) if example_ids is None else anchor_fingerprint(truth, example_ids)
    )

    return DriftReport(
        n_anchor=n,
        baseline_agreement=float(baseline_agrees.mean()),
        current_agreement=float(current_agrees.mean()),
        agreement_change=comparison.difference,
        low=comparison.low,
        high=comparison.high,
        p_value=comparison.p_value,
        flip_rate=flips / n,
        flip_low=float(flip_interval.low),
        flip_high=float(flip_interval.high),
        fingerprint=fingerprint,
        level=1.0 - alpha,
    )


def monitor_agreement(
    agreements: npt.ArrayLike,
    baseline_agreement: float,
    *,
    alpha: float = 0.05,
) -> int | None:
    """Watch judge agreement arrive and stop the first time it is provably below baseline.

    Wraps :func:`truescore.sequential.first_exclusion` in the one-sided direction that
    matters operationally: raise the alarm when the evidence says agreement has fallen,
    and do not raise it merely because it moved. Because the underlying interval is valid
    uniformly over time, checking after every anchor example is legitimate -- which is
    what makes this usable as a live monitor rather than a scheduled report.

    Args:
        agreements: Per-example agreement indicators (1 = judge matched gold) in arrival
            order.
        baseline_agreement: The agreement rate being defended, from the reference run.
        alpha: One minus the uniform coverage level; also the false-alarm budget for the
            whole monitoring run, however long it lasts.

    Returns:
        The 1-based example count at which agreement was first proven to be below
        ``baseline_agreement``, or ``None`` if it never was.

    References:
        tests/test_drift.py::test_monitor_raises_on_degradation_and_rarely_otherwise
    """
    check_alpha(alpha)
    if not 0.0 <= baseline_agreement <= 1.0:
        raise ValueError(f"baseline_agreement must lie in [0, 1]; got {baseline_agreement}")
    indicators: npt.NDArray[Any] = check_binary("agreements", agreements).astype(float)
    return first_exclusion(indicators, baseline_agreement, alpha=alpha, direction="below")
