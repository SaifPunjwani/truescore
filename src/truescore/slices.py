# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Saif Punjwani
"""Which segment actually moved?

"Is v4 better?" is never the last question. The next one is "better for whom?", because an
overall gain of nine points routinely hides a loss for Spanish-language questions, or for
the enterprise tier, or for the long tail of rare intents. Teams slice their evals for
exactly this reason, and in doing so walk into two problems at once.

The first is that every slice inherits the judge's bias, and slices differ in how much.
A slice whose answers are longer collects more of a length-biased judge's leniency, so the
uncorrected per-slice numbers are wrong by *different amounts* -- which is worse than being
wrong by the same amount, because it reorders the slices.

The second is multiplicity. Twenty slices tested at 5% produce, on average, one spurious
"significant" slice per run even when nothing differs anywhere. Teams then investigate it.
This module applies Holm or Benjamini-Hochberg across the slices, so the flagged ones are
worth the investigation.

A third problem has no statistical fix and so is reported rather than papered over: gold
labels are spread thin across slices. A slice holding eight human labels cannot support a
corrected estimate, and this module says so instead of returning a confident number built
on eight labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from truescore._validation import check_alpha, check_gold_index, to_1d_array
from truescore.compare import benjamini_hochberg, holm, ppi_compare
from truescore.correct import Estimate, judge_only_estimate, ppi_estimate

__all__ = ["SliceComparison", "SliceEstimate", "SliceReport", "compare_slices", "estimate_slices"]

MIN_GOLD_PER_SLICE = 20


@dataclass(frozen=True)
class SliceEstimate:
    """A corrected estimate for one slice.

    Attributes:
        name: Slice value, e.g. ``"es"`` for a language slice.
        n_total: Examples in the slice.
        n_gold: Human-labeled examples in the slice.
        naive: The judge-only estimate for this slice.
        corrected: The corrected estimate, or ``None`` when the slice holds too few gold
            labels to support one.
        skipped_reason: Why ``corrected`` is ``None``, when it is.
    """

    name: str
    n_total: int
    n_gold: int
    naive: Estimate
    corrected: Estimate | None
    skipped_reason: str = ""

    @property
    def naive_error(self) -> float | None:
        """How far the judge-only number sits from the corrected one, when both exist."""
        if self.corrected is None:
            return None
        return self.naive.point - self.corrected.point


@dataclass(frozen=True)
class SliceComparison:
    """A per-slice comparison of two systems, adjusted for multiple slices.

    Attributes:
        name: Slice value.
        n_total: Examples in the slice.
        n_gold: Human-labeled examples in the slice.
        difference: Corrected difference, system A minus system B.
        low: Lower confidence limit.
        high: Upper confidence limit.
        p_value: Raw p-value, before adjustment.
        adjusted_p_value: p-value after correcting for the number of slices tested.
        significant: Whether the slice survives that correction.
        skipped_reason: Why the slice was not tested, when it was not.
    """

    name: str
    n_total: int
    n_gold: int
    difference: float
    low: float
    high: float
    p_value: float
    adjusted_p_value: float
    significant: bool
    skipped_reason: str = ""


@dataclass(frozen=True)
class SliceReport:
    """Per-slice results plus the slices that could not be analyzed.

    Attributes:
        by: Name of the slicing column.
        estimates: Per-slice corrected estimates, when estimating.
        comparisons: Per-slice comparisons, when comparing.
        correction: Multiplicity correction applied across slices.
        n_tested: Slices with enough gold labels to test.
        n_skipped: Slices without enough.
    """

    by: str
    estimates: tuple[SliceEstimate, ...]
    comparisons: tuple[SliceComparison, ...]
    correction: str
    n_tested: int
    n_skipped: int

    def summary(self) -> str:
        """Human-readable multi-line summary, worst slice first."""
        lines = [
            f"slices of {self.by}: {self.n_tested} tested, {self.n_skipped} skipped for "
            "too few human labels"
        ]
        if self.comparisons:
            lines.append(f"multiplicity correction across slices: {self.correction}")
            ordered = sorted(self.comparisons, key=lambda c: (c.skipped_reason != "", c.difference))
            for comparison in ordered:
                if comparison.skipped_reason:
                    lines.append(
                        f"  {comparison.name:<16} n={comparison.n_total:<6} "
                        f"skipped ({comparison.skipped_reason})"
                    )
                    continue
                flag = "  <-- flagged" if comparison.significant else ""
                lines.append(
                    f"  {comparison.name:<16} n={comparison.n_total:<6} "
                    f"{comparison.difference:+.4f} "
                    f"[{comparison.low:+.4f}, {comparison.high:+.4f}] "
                    f"adj p={comparison.adjusted_p_value:.4g}{flag}"
                )
        for estimate in self.estimates:
            if estimate.corrected is None:
                lines.append(
                    f"  {estimate.name:<16} n={estimate.n_total:<6} "
                    f"judge says {estimate.naive.point:.4f}, "
                    f"not corrected ({estimate.skipped_reason})"
                )
                continue
            lines.append(
                f"  {estimate.name:<16} n={estimate.n_total:<6} "
                f"corrected {estimate.corrected.point:.4f} "
                f"[{estimate.corrected.low:.4f}, {estimate.corrected.high:.4f}] "
                f"(judge said {estimate.naive.point:.4f})"
            )
        return "\n".join(lines)


def _slice_positions(labels: npt.NDArray[Any]) -> dict[str, npt.NDArray[Any]]:
    """Map each slice value to the row positions belonging to it."""
    return {str(value): np.flatnonzero(labels == value) for value in np.unique(labels)}


def _adjust(p_values: list[float], correction: str) -> npt.NDArray[Any]:
    if not p_values:
        return np.asarray([], dtype=float)
    values = np.asarray(p_values, dtype=float)
    if correction == "holm":
        return holm(values)
    if correction == "bh":
        return benjamini_hochberg(values)
    if correction == "none":
        return values
    raise ValueError(f"unknown correction {correction!r}; expected 'holm', 'bh' or 'none'")


def estimate_slices(
    judge: npt.ArrayLike,
    gold: npt.ArrayLike,
    gold_index: npt.ArrayLike,
    slice_labels: npt.ArrayLike,
    *,
    by: str = "slice",
    alpha: float = 0.05,
    min_gold: int = MIN_GOLD_PER_SLICE,
) -> SliceReport:
    """Corrected pass rates per slice.

    Args:
        judge: Judge verdicts for every example.
        gold: Gold verdicts for the labeled subset.
        gold_index: Positions in ``judge`` carrying a gold verdict.
        slice_labels: Slice value for every example, aligned with ``judge``.
        by: Name of the slicing dimension, used in summaries.
        alpha: Significance level for every interval.
        min_gold: Gold labels a slice needs before a corrected estimate is attempted.
            Below this the slice reports the judge's number and says it was not corrected,
            which is more useful than a corrected number nobody should rely on.

    Returns:
        A :class:`SliceReport` whose ``estimates`` cover every slice.

    Raises:
        ValueError: If the arrays disagree in length or ``min_gold`` is below 2.

    References:
        tests/test_slices.py::test_per_slice_correction_recovers_a_hidden_regression
    """
    check_alpha(alpha)
    if min_gold < 2:
        raise ValueError(f"min_gold must be at least 2; got {min_gold}")

    judge_array = to_1d_array("judge", np.asarray(judge, dtype=float))
    gold_array = to_1d_array("gold", np.asarray(gold, dtype=float))
    index = check_gold_index(gold_index, judge_array.shape[0])
    labels = to_1d_array("slice_labels", np.asarray(slice_labels))
    if labels.shape[0] != judge_array.shape[0]:
        raise ValueError(
            f"slice_labels must cover every example; got {labels.shape[0]} labels for "
            f"{judge_array.shape[0]} examples"
        )

    gold_by_position = {int(position): gold_array[i] for i, position in enumerate(index)}

    estimates: list[SliceEstimate] = []
    tested = skipped = 0
    for name, positions in _slice_positions(labels).items():
        slice_judge = judge_array[positions]
        local_gold_positions = [p for p in positions if int(p) in gold_by_position]
        n_gold = len(local_gold_positions)

        naive = judge_only_estimate(slice_judge, alpha=alpha)
        corrected: Estimate | None = None
        reason = ""
        if n_gold < min_gold:
            reason = f"{n_gold} human labels, fewer than {min_gold}"
        elif n_gold >= len(positions) - 1:
            reason = "no unlabeled examples left in the slice to borrow strength from"
        else:
            remap = {int(p): i for i, p in enumerate(positions)}
            local_index = np.asarray([remap[int(p)] for p in local_gold_positions])
            local_gold = np.asarray([gold_by_position[int(p)] for p in local_gold_positions])
            corrected = ppi_estimate(slice_judge, local_gold, local_index, alpha=alpha)

        if corrected is None:
            skipped += 1
        else:
            tested += 1
        estimates.append(
            SliceEstimate(
                name=name,
                n_total=int(positions.shape[0]),
                n_gold=n_gold,
                naive=naive,
                corrected=corrected,
                skipped_reason=reason,
            )
        )

    return SliceReport(
        by=by,
        estimates=tuple(estimates),
        comparisons=(),
        correction="none",
        n_tested=tested,
        n_skipped=skipped,
    )


def compare_slices(
    judge_a: npt.ArrayLike,
    judge_b: npt.ArrayLike,
    gold_a: npt.ArrayLike,
    gold_b: npt.ArrayLike,
    gold_index: npt.ArrayLike,
    slice_labels: npt.ArrayLike,
    *,
    by: str = "slice",
    alpha: float = 0.05,
    correction: Literal["holm", "bh", "none"] = "holm",
    min_gold: int = MIN_GOLD_PER_SLICE,
) -> SliceReport:
    """Compare two systems within each slice, corrected for judge error and multiplicity.

    The question this answers is the one that follows every launch decision: the new
    version wins overall, but is there a segment it makes worse? Both hazards of asking it
    are handled here -- the judge's bias differing across slices, and the false positives
    that come free with testing twenty of them.

    Args:
        judge_a: Judge verdicts for system A on every example.
        judge_b: Judge verdicts for system B on the same examples.
        gold_a: Gold verdicts for system A on the labeled subset.
        gold_b: Gold verdicts for system B on the same labeled subset.
        gold_index: Positions carrying gold verdicts.
        slice_labels: Slice value for every example.
        by: Name of the slicing dimension.
        alpha: Significance level, applied after adjustment.
        correction: ``"holm"`` controls the family-wise error rate across slices and is
            the right default when a flagged slice triggers an investigation; ``"bh"``
            controls the false discovery rate and suits wide exploratory sweeps;
            ``"none"`` leaves the p-values raw and should be used only when a single slice
            was pre-registered.
        min_gold: Gold labels a slice needs before it is tested.

    Returns:
        A :class:`SliceReport` whose ``comparisons`` cover every slice, with untested
        slices carrying a ``skipped_reason``.

    References:
        tests/test_slices.py::test_multiplicity_correction_suppresses_spurious_slices
    """
    check_alpha(alpha)
    if min_gold < 2:
        raise ValueError(f"min_gold must be at least 2; got {min_gold}")

    array_a = to_1d_array("judge_a", np.asarray(judge_a, dtype=float))
    array_b = to_1d_array("judge_b", np.asarray(judge_b, dtype=float))
    gold_array_a = to_1d_array("gold_a", np.asarray(gold_a, dtype=float))
    gold_array_b = to_1d_array("gold_b", np.asarray(gold_b, dtype=float))
    index = check_gold_index(gold_index, array_a.shape[0])
    labels = to_1d_array("slice_labels", np.asarray(slice_labels))
    if array_a.shape != array_b.shape:
        raise ValueError("judge_a and judge_b must cover the same examples")
    if labels.shape[0] != array_a.shape[0]:
        raise ValueError("slice_labels must cover every example")

    position_to_gold = {
        int(position): (gold_array_a[i], gold_array_b[i]) for i, position in enumerate(index)
    }

    pending: list[tuple[str, npt.NDArray[Any], list[int]]] = []
    skipped_entries: list[SliceComparison] = []
    for name, positions in _slice_positions(labels).items():
        local_gold_positions = [int(p) for p in positions if int(p) in position_to_gold]
        n_gold = len(local_gold_positions)
        if n_gold < min_gold or n_gold >= positions.shape[0] - 1:
            reason = (
                f"{n_gold} human labels, fewer than {min_gold}"
                if n_gold < min_gold
                else "no unlabeled examples left in the slice"
            )
            skipped_entries.append(
                SliceComparison(
                    name=name,
                    n_total=int(positions.shape[0]),
                    n_gold=n_gold,
                    difference=float("nan"),
                    low=float("nan"),
                    high=float("nan"),
                    p_value=1.0,
                    adjusted_p_value=1.0,
                    significant=False,
                    skipped_reason=reason,
                )
            )
            continue
        pending.append((name, positions, local_gold_positions))

    raw: list[float] = []
    results: list[tuple[str, npt.NDArray[Any], int, Any]] = []
    for name, positions, local_gold_positions in pending:
        remap = {int(p): i for i, p in enumerate(positions)}
        local_index = np.asarray([remap[p] for p in local_gold_positions])
        local_gold_a = np.asarray([position_to_gold[p][0] for p in local_gold_positions])
        local_gold_b = np.asarray([position_to_gold[p][1] for p in local_gold_positions])
        comparison = ppi_compare(
            array_a[positions],
            array_b[positions],
            local_gold_a,
            local_gold_b,
            local_index,
            alpha=alpha,
        )
        raw.append(comparison.p_value)
        results.append((name, positions, len(local_gold_positions), comparison))

    adjusted = _adjust(raw, correction)
    comparisons = [
        SliceComparison(
            name=name,
            n_total=int(positions.shape[0]),
            n_gold=n_gold,
            difference=comparison.difference,
            low=comparison.low,
            high=comparison.high,
            p_value=comparison.p_value,
            adjusted_p_value=float(adjusted[i]),
            significant=bool(adjusted[i] < alpha),
        )
        for i, (name, positions, n_gold, comparison) in enumerate(results)
    ]

    return SliceReport(
        by=by,
        estimates=(),
        comparisons=tuple(comparisons + skipped_entries),
        correction=correction,
        n_tested=len(comparisons),
        n_skipped=len(skipped_entries),
    )


def slice_names(slice_labels: npt.ArrayLike) -> Sequence[str]:
    """Distinct slice values, in sorted order."""
    return [str(value) for value in np.unique(np.asarray(slice_labels))]


def counts_by_slice(slice_labels: npt.ArrayLike) -> Mapping[str, int]:
    """Examples per slice, for sizing a labeling plan before spending on one."""
    values, counts = np.unique(np.asarray(slice_labels), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts, strict=True)}
