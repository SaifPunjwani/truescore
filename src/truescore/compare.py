# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Saif Punjwani
"""Is system A actually better than system B?

Model comparisons are almost always run on the *same* examples, which makes them paired.
Treating them as two independent samples -- the default in most eval tooling -- throws
away the pairing and inflates the variance, so real improvements look insignificant and
noise looks real. Every test here is paired.

The practical failure this module exists to prevent: a team sees 91.5% versus 90.0% on
200 examples, ships the change, and has in fact shipped noise. On paired binary outcomes
that difference is significant only if the disagreements between the systems are lopsided
enough, which McNemar's test answers exactly.

References:
    McNemar (1947), "Note on the sampling error of the difference between correlated
        proportions or percentages", Psychometrika 12(2).
    Fagerland, Lydersen, Laake (2013), "The McNemar test for binary matched-pairs data:
        mid-p and asymptotic are better than exact conditional", BMC Medical Research
        Methodology 13(91).
    Holm (1979), "A simple sequentially rejective multiple test procedure".
    Benjamini & Hochberg (1995), "Controlling the false discovery rate".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from truescore._validation import (
    check_alpha,
    check_binary,
    check_same_length,
    to_1d_array,
)
from truescore.correct import _cluster_codes, ppi_estimate

__all__ = [
    "ComparisonResult",
    "benjamini_hochberg",
    "holm",
    "mcnemar",
    "paired_bootstrap",
    "paired_permutation",
    "ppi_compare",
]


@dataclass(frozen=True)
class ComparisonResult:
    """The result of comparing two systems on shared examples.

    Attributes:
        difference: Metric for A minus metric for B. Positive favors A.
        low: Lower confidence limit on the difference.
        high: Upper confidence limit on the difference.
        level: Nominal coverage level.
        p_value: Two-sided p-value against the null of no difference.
        method: Test name.
        n_pairs: Number of shared examples.
        n_discordant: Examples where the systems disagree, when the test is defined on
            discordant pairs. All the information in a paired binary comparison lives
            here: agreeing examples contribute nothing.
    """

    difference: float
    low: float
    high: float
    level: float
    p_value: float
    method: str
    n_pairs: int
    n_discordant: int | None = None

    @property
    def significant(self) -> bool:
        """Whether the test rejects at ``1 - level``."""
        return self.p_value < (1.0 - self.level)

    def summary(self) -> str:
        """Human-readable multi-line summary, including the plain-language verdict."""
        verdict = (
            "difference is statistically distinguishable from zero"
            if self.significant
            else "difference is NOT distinguishable from zero -- this may be noise"
        )
        lines = [
            f"{self.method} on n={self.n_pairs} paired examples",
            f"  difference (A - B): {self.difference:+.4f} "
            f"[{self.low:+.4f}, {self.high:+.4f}] at {self.level:.0%}",
            f"  p = {self.p_value:.4g}: {verdict}",
        ]
        if self.n_discordant is not None:
            lines.append(f"  discordant pairs: {self.n_discordant} (all the evidence lives here)")
        return "\n".join(lines)


def mcnemar(
    a: npt.ArrayLike, b: npt.ArrayLike, *, alpha: float = 0.05, midp: bool = True
) -> ComparisonResult:
    """McNemar's test for two systems scored on the same examples with binary outcomes.

    Only discordant pairs carry information. Writing ``n01`` for examples A got wrong and
    B got right, and ``n10`` for the reverse, the null is ``n10 ~ Binomial(n01 + n10, ½)``.

    Args:
        a: Binary outcomes for system A (1 = correct/pass).
        b: Binary outcomes for system B, aligned elementwise with ``a``.
        alpha: Significance level for the interval.
        midp: Use the mid-p correction. The exact conditional test is known to be
            conservative -- it under-rejects, so real improvements get missed -- and the
            mid-p variant has closer-to-nominal error rates (Fagerland et al. 2013).
            Set ``False`` for the strictly exact test.

    Returns:
        The comparison, with a Wald interval on the paired difference in proportions.

    Raises:
        ValueError: If inputs are not binary or differ in length.

    References:
        tests/test_compare.py::test_mcnemar_matches_hand_computed_exact_p
        tests/test_compare.py::test_mcnemar_type_one_error_is_calibrated
    """
    check_alpha(alpha)
    arr_a = check_binary("a", a)
    arr_b = check_binary("b", b)
    check_same_length("a", arr_a, "b", arr_b)

    n = arr_a.shape[0]
    n10 = int(np.sum((arr_a == 1) & (arr_b == 0)))
    n01 = int(np.sum((arr_a == 0) & (arr_b == 1)))
    discordant = n10 + n01

    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(n10, n01)
        tail = float(stats.binom.cdf(smaller, discordant, 0.5))
        p_exact = min(1.0, 2.0 * tail)
        if midp:
            point_mass = float(stats.binom.pmf(smaller, discordant, 0.5))
            p_value = max(0.0, min(1.0, p_exact - point_mass))
        else:
            p_value = p_exact

    difference = (n10 - n01) / n
    # Variance of the paired difference in proportions; agreeing pairs cancel exactly.
    variance = (discordant - (n10 - n01) ** 2 / n) / (n * n)
    half = float(stats.norm.ppf(1.0 - alpha / 2.0)) * float(np.sqrt(max(variance, 0.0)))

    return ComparisonResult(
        difference=difference,
        low=difference - half,
        high=difference + half,
        level=1.0 - alpha,
        p_value=p_value,
        method="mcnemar (mid-p)" if midp else "mcnemar (exact)",
        n_pairs=n,
        n_discordant=discordant,
    )


def paired_bootstrap(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    seed: int = 0,
    clusters: npt.ArrayLike | None = None,
) -> ComparisonResult:
    """Paired bootstrap for the difference in means of any per-example metric.

    Resamples examples (not scores) with replacement, preserving the pairing, so it
    applies to continuous scores and rubric ratings where McNemar does not.

    Args:
        a: Per-example metric values for system A.
        b: Per-example values for system B, aligned elementwise.
        alpha: Significance level.
        n_bootstrap: Number of resamples.
        seed: Seed, so a report is reproducible.
        clusters: Group label per example. Supplying it resamples whole clusters rather
            than rows, which is the resampling unit that matches the sampling design when
            observations arrive in correlated groups. Resampling rows from clustered data
            treats the correlation as if it were not there and returns an interval too
            narrow, the same way an unclustered variance does.

    Returns:
        The comparison, with a percentile interval and a bootstrap p-value obtained by
        inverting it (the smallest level at which the interval excludes zero).

    References:
        tests/test_compare.py::test_paired_bootstrap_covers_at_nominal_rate
        tests/test_compare.py::test_cluster_bootstrap_controls_false_positives
    """
    check_alpha(alpha)
    arr_a = to_1d_array("a", np.asarray(a, dtype=float))
    arr_b = to_1d_array("b", np.asarray(b, dtype=float))
    check_same_length("a", arr_a, "b", arr_b)

    diffs = arr_a - arr_b
    n = diffs.shape[0]
    rng = np.random.default_rng(seed)
    if clusters is None:
        draws = rng.integers(0, n, size=(n_bootstrap, n))
        resampled = diffs[draws].mean(axis=1)
        method = "paired bootstrap (percentile)"
    else:
        codes = _cluster_codes("clusters", clusters, n)
        order = np.argsort(codes, kind="stable")
        sorted_diffs = diffs[order]
        boundaries = np.searchsorted(codes[order], np.arange(int(codes.max()) + 2))
        groups = [
            sorted_diffs[boundaries[g] : boundaries[g + 1]] for g in range(len(boundaries) - 1)
        ]
        groups = [g for g in groups if g.size]
        if len(groups) < 2:
            raise ValueError(f"cluster bootstrap needs at least 2 clusters; got {len(groups)}")
        sums = np.array([g.sum() for g in groups])
        sizes = np.array([g.size for g in groups], dtype=float)
        picks = rng.integers(0, len(groups), size=(n_bootstrap, len(groups)))
        resampled = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
        method = f"cluster bootstrap (percentile, {len(groups)} clusters)"

    low, high = np.percentile(resampled, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided bootstrap p-value: twice the smaller tail mass on the far side of zero.
    share_below = float(np.mean(resampled <= 0.0))
    p_value = float(min(1.0, 2.0 * min(share_below, 1.0 - share_below)))

    return ComparisonResult(
        difference=float(diffs.mean()),
        low=float(low),
        high=float(high),
        level=1.0 - alpha,
        p_value=p_value,
        method=method,
        n_pairs=n,
    )


def paired_permutation(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    n_resamples: int = 10000,
    seed: int = 0,
) -> ComparisonResult:
    """Paired permutation (sign-flip) test for the difference in means.

    Exchanges the labels within each pair, which is the exact null for paired data under
    symmetry. Scipy computes the exact distribution when the number of pairs is small
    enough and samples it otherwise.

    Args:
        a: Per-example metric values for system A.
        b: Per-example values for system B, aligned elementwise.
        alpha: Significance level for the accompanying bootstrap interval.
        n_resamples: Monte Carlo resamples when exact enumeration is infeasible.
        seed: Seed, so a report is reproducible.

    Returns:
        The comparison. The p-value is permutation-based; the interval is the paired
        bootstrap interval, since a permutation test yields no interval of its own.

    References:
        tests/test_compare.py::test_paired_permutation_agrees_with_exact_enumeration
    """
    check_alpha(alpha)
    arr_a = to_1d_array("a", np.asarray(a, dtype=float))
    arr_b = to_1d_array("b", np.asarray(b, dtype=float))
    check_same_length("a", arr_a, "b", arr_b)

    result = stats.permutation_test(
        (arr_a, arr_b),
        lambda x, y: float(np.mean(x) - np.mean(y)),
        permutation_type="samples",
        alternative="two-sided",
        n_resamples=n_resamples,
        rng=np.random.default_rng(seed),
    )
    interval = paired_bootstrap(arr_a, arr_b, alpha=alpha, seed=seed)
    return ComparisonResult(
        difference=interval.difference,
        low=interval.low,
        high=interval.high,
        level=1.0 - alpha,
        p_value=float(result.pvalue),
        method="paired permutation (sign-flip)",
        n_pairs=arr_a.shape[0],
    )


def ppi_compare(
    judge_a: npt.ArrayLike,
    judge_b: npt.ArrayLike,
    gold_a: npt.ArrayLike,
    gold_b: npt.ArrayLike,
    gold_index: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    clusters: npt.ArrayLike | None = None,
) -> ComparisonResult:
    """Compare two systems on their *true* scores rather than their judge scores.

    Applies prediction-powered inference to the per-example difference, so the comparison
    inherits PPI's guarantee: valid regardless of judge quality, and tighter than a
    gold-only comparison when the judge is informative. This is the estimator to use when
    a judge's bias might differ between the two systems -- for instance when one produces
    systematically longer answers and the judge rewards length.

    Args:
        judge_a: Judge labels for system A on all examples.
        judge_b: Judge labels for system B on all examples.
        gold_a: Gold labels for system A on the labeled subset.
        gold_b: Gold labels for system B on the labeled subset.
        gold_index: Positions carrying gold labels.
        alpha: Significance level.
        clusters: Group label per example, when observations are not independent. Several
            epochs of one sample, or several turns of one conversation. Omitting it on
            clustered data produces an interval narrower than the data supports and a
            p-value smaller than it should be.

    Returns:
        The comparison of true scores.

    References:
        tests/test_compare.py::test_ppi_compare_corrects_a_judge_biased_toward_one_system
        tests/test_compare.py::test_ppi_compare_respects_clusters
    """
    check_alpha(alpha)
    ja = to_1d_array("judge_a", np.asarray(judge_a, dtype=float))
    jb = to_1d_array("judge_b", np.asarray(judge_b, dtype=float))
    ga = to_1d_array("gold_a", np.asarray(gold_a, dtype=float))
    gb = to_1d_array("gold_b", np.asarray(gold_b, dtype=float))
    check_same_length("judge_a", ja, "judge_b", jb)
    check_same_length("gold_a", ga, "gold_b", gb)

    estimate = ppi_estimate(ja - jb, ga - gb, gold_index, alpha=alpha, clusters=clusters)
    # The estimator reports its own standard error. Recovering it by dividing the half
    # width by a normal quantile was right only while every interval used one, and stopped
    # being right the moment clustered intervals started using a t quantile instead.
    standard_error = estimate.standard_error or 0.0
    p_value = (
        float(2.0 * stats.norm.sf(abs(estimate.point) / standard_error))
        if standard_error > 0.0
        else 1.0
    )
    return ComparisonResult(
        difference=estimate.point,
        low=estimate.low,
        high=estimate.high,
        level=1.0 - alpha,
        p_value=p_value,
        method="ppi++ paired difference" + ("" if clusters is None else " (cluster-robust)"),
        n_pairs=ja.shape[0],
    )


def holm(p_values: npt.ArrayLike) -> npt.NDArray[Any]:
    """Holm step-down adjusted p-values, controlling the family-wise error rate.

    Use when comparing several systems or several slices at once: at ten independent
    comparisons the chance of at least one spurious "significant" result at α=0.05 is
    about 40%, so unadjusted p-values guarantee false discoveries.

    Args:
        p_values: Raw two-sided p-values.

    Returns:
        Adjusted p-values in the input order; compare them against the original α.

    References:
        tests/test_compare.py::test_holm_controls_family_wise_error_rate
    """
    p = to_1d_array("p_values", np.asarray(p_values, dtype=float))
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    m = p.shape[0]
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, (m - rank) * p[position])
        adjusted[position] = min(1.0, running)
    return adjusted


def benjamini_hochberg(p_values: npt.ArrayLike) -> npt.NDArray[Any]:
    """Benjamini-Hochberg adjusted p-values, controlling the false discovery rate.

    Less conservative than :func:`holm`; appropriate when you are screening many slices
    and can tolerate a known proportion of false positives among the discoveries.

    Args:
        p_values: Raw two-sided p-values.

    Returns:
        Adjusted p-values (q-values) in the input order.

    References:
        tests/test_compare.py::test_benjamini_hochberg_matches_scipy
    """
    p = to_1d_array("p_values", np.asarray(p_values, dtype=float))
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    m = p.shape[0]
    order = np.argsort(p)[::-1]
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for rank, position in enumerate(order):
        factor = m / (m - rank)
        running = min(running, factor * p[position])
        adjusted[position] = min(1.0, running)
    return adjusted
