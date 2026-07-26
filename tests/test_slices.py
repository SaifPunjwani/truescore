"""Tests for truescore.slices."""

from __future__ import annotations

import numpy as np
import pytest

from truescore.slices import (
    compare_slices,
    counts_by_slice,
    estimate_slices,
    slice_names,
)


def _judged(rng: np.random.Generator, truth: np.ndarray, sens: float, spec: float) -> np.ndarray:
    return np.where(
        truth == 1,
        rng.binomial(1, sens, truth.shape[0]),
        rng.binomial(1, 1.0 - spec, truth.shape[0]),
    )


def test_per_slice_correction_recovers_a_hidden_regression() -> None:
    """The case slicing exists for, and the case a judge hides.

    Two slices. The judge is lenient on both, but *more* lenient on the slice where the
    new version writes longer answers -- so the uncorrected numbers say the new version
    improved everywhere, while in truth it regressed badly on that very slice.
    """
    rng = np.random.default_rng(0)
    n_per_slice = 3000
    labels = np.array(["en"] * n_per_slice + ["es"] * n_per_slice)

    truth_b = np.concatenate(
        [rng.binomial(1, 0.80, n_per_slice), rng.binomial(1, 0.80, n_per_slice)]
    )
    # The new version improves on 'en' and regresses hard on 'es'.
    truth_a = np.concatenate(
        [rng.binomial(1, 0.88, n_per_slice), rng.binomial(1, 0.62, n_per_slice)]
    )

    judge_b = _judged(rng, truth_b, 0.95, 0.75)
    judge_a = np.concatenate(
        [
            _judged(rng, truth_a[:n_per_slice], 0.95, 0.75),
            # On 'es' the new version's answers are long, and the judge waves them through.
            _judged(rng, truth_a[n_per_slice:], 0.99, 0.15),
        ]
    )

    gold_index = np.sort(rng.choice(2 * n_per_slice, 900, replace=False))

    naive_es = judge_a[n_per_slice:].mean() - judge_b[n_per_slice:].mean()
    assert naive_es > 0, "as judged, the regressed slice looks like an improvement"

    report = compare_slices(
        judge_a,
        judge_b,
        truth_a[gold_index],
        truth_b[gold_index],
        gold_index,
        labels,
        by="language",
    )
    by_name = {c.name: c for c in report.comparisons}

    assert by_name["es"].difference < 0, "the corrected comparison recovers the regression"
    assert by_name["es"].significant
    assert by_name["en"].difference > 0
    assert "flagged" in report.summary()


def test_multiplicity_correction_suppresses_spurious_slices() -> None:
    """Twenty identical slices should not produce a 'finding' just for being twenty."""
    replications = 120
    rng = np.random.default_rng(1)
    families_with_a_flag = 0
    families_with_an_unadjusted_flag = 0

    for _ in range(replications):
        n = 4000
        labels = np.asarray([f"s{i % 20:02d}" for i in range(n)])
        truth_a = rng.binomial(1, 0.75, n)
        truth_b = rng.binomial(1, 0.75, n)
        judge_a = _judged(rng, truth_a, 0.93, 0.8)
        judge_b = _judged(rng, truth_b, 0.93, 0.8)
        gold_index = np.sort(rng.choice(n, 1200, replace=False))

        report = compare_slices(
            judge_a, judge_b, truth_a[gold_index], truth_b[gold_index], gold_index, labels
        )
        families_with_a_flag += int(any(c.significant for c in report.comparisons))
        families_with_an_unadjusted_flag += int(
            any(c.p_value < 0.05 for c in report.comparisons if not c.skipped_reason)
        )

    assert families_with_a_flag / replications <= 0.10
    assert families_with_an_unadjusted_flag > families_with_a_flag, (
        "unadjusted testing should flag more often, which is why the correction is applied"
    )


def test_thin_slices_are_reported_not_estimated() -> None:
    """A slice with a handful of human labels gets an honest refusal, not a number."""
    rng = np.random.default_rng(2)
    n = 2000
    labels = np.asarray(["big"] * 1900 + ["tiny"] * 100)
    truth = rng.binomial(1, 0.7, n)
    judge = _judged(rng, truth, 0.95, 0.6)
    # Concentrate the labels on the big slice: the tiny slice gets only five.
    gold_index = np.sort(
        np.concatenate([rng.choice(1900, 400, replace=False), np.arange(1900, 1905)])
    )

    report = estimate_slices(judge, truth[gold_index], gold_index, labels, by="tier")
    by_name = {e.name: e for e in report.estimates}

    assert by_name["big"].corrected is not None
    assert by_name["tiny"].corrected is None
    assert "fewer than" in by_name["tiny"].skipped_reason
    assert report.n_skipped == 1
    assert "not corrected" in report.summary()


def test_slice_estimates_differ_from_the_judge_by_slice_specific_amounts() -> None:
    """The reason per-slice correction is needed at all: the bias is not uniform."""
    rng = np.random.default_rng(3)
    n_per_slice = 2500
    labels = np.array(["clean"] * n_per_slice + ["messy"] * n_per_slice)
    truth = np.concatenate([rng.binomial(1, 0.75, n_per_slice), rng.binomial(1, 0.75, n_per_slice)])
    judge = np.concatenate(
        [
            _judged(rng, truth[:n_per_slice], 0.96, 0.90),
            _judged(rng, truth[n_per_slice:], 0.96, 0.30),
        ]
    )
    gold_index = np.sort(rng.choice(2 * n_per_slice, 800, replace=False))

    report = estimate_slices(judge, truth[gold_index], gold_index, labels, by="formatting")
    errors = {e.name: e.naive_error for e in report.estimates}

    assert errors["messy"] is not None and errors["clean"] is not None
    assert errors["messy"] > errors["clean"] + 0.05, (
        "the judge over-scores the messy slice much more, so one global correction "
        "would misrank the slices"
    )


def test_correction_choices_are_ordered_by_strictness() -> None:
    """Holm is at least as strict as BH, which is at least as strict as no correction."""
    rng = np.random.default_rng(4)
    n = 3000
    labels = np.asarray([f"s{i % 8}" for i in range(n)])
    truth_a = rng.binomial(1, 0.7, n)
    truth_b = rng.binomial(1, 0.66, n)
    judge_a = _judged(rng, truth_a, 0.95, 0.7)
    judge_b = _judged(rng, truth_b, 0.95, 0.7)
    gold_index = np.sort(rng.choice(n, 1500, replace=False))

    counts = {}
    for correction in ("holm", "bh", "none"):
        report = compare_slices(
            judge_a,
            judge_b,
            truth_a[gold_index],
            truth_b[gold_index],
            gold_index,
            labels,
            correction=correction,  # type: ignore[arg-type]
        )
        counts[correction] = sum(c.significant for c in report.comparisons)

    assert counts["holm"] <= counts["bh"] <= counts["none"]


def test_unknown_correction_is_rejected() -> None:
    rng = np.random.default_rng(5)
    n = 400
    labels = np.asarray(["a"] * 200 + ["b"] * 200)
    truth = rng.binomial(1, 0.7, n)
    judge = _judged(rng, truth, 0.9, 0.8)
    gold_index = np.sort(rng.choice(n, 120, replace=False))
    with pytest.raises(ValueError, match="unknown correction"):
        compare_slices(
            judge,
            judge,
            truth[gold_index],
            truth[gold_index],
            gold_index,
            labels,
            correction="bonferroni",  # type: ignore[arg-type]
        )


def test_mismatched_slice_labels_are_rejected() -> None:
    rng = np.random.default_rng(6)
    judge = rng.binomial(1, 0.7, 100).astype(float)
    gold_index = np.arange(30)
    with pytest.raises(ValueError, match="slice_labels must cover every example"):
        estimate_slices(judge, judge[gold_index], gold_index, np.asarray(["a"] * 50))


def test_helper_summaries_of_a_slicing_column() -> None:
    labels = np.asarray(["en", "es", "en", "fr", "en"])
    assert slice_names(labels) == ["en", "es", "fr"]
    assert counts_by_slice(labels) == {"en": 3, "es": 1, "fr": 1}
