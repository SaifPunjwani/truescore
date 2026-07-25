"""How good is the judge?

Agreement statistics between an automated judge and trusted (gold) labels, each with a
confidence interval. Point estimates of agreement are routinely reported without
intervals, which hides how little a few hundred labels actually pin down.

Two deliberate choices:

- Proportions use **Wilson** intervals, not the normal approximation. The normal
  approximation fails exactly where judges live -- near 0 and 1 -- producing intervals
  that extend past the unit range and undercover badly at small ``n``.
- Chance-corrected agreement reports **Gwet's AC1** alongside Cohen's κ. κ collapses
  toward 0 when one class dominates even if raters agree almost perfectly (the "kappa
  paradox"); LLM evaluation sets are usually imbalanced, so κ alone systematically
  understates judge quality. AC1 is stable under imbalance.

References:
    Wilson (1927), "Probable inference, the law of succession, and statistical inference".
    Cohen (1960), "A coefficient of agreement for nominal scales".
    Gwet (2008), "Computing inter-rater reliability and its variance in the presence of
        high agreement", British Journal of Mathematical and Statistical Psychology 61(1).
    Krippendorff (2004), "Content Analysis: An Introduction to Its Methodology", ch. 11.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import stats

from truescore._validation import (
    check_alpha,
    check_binary,
    check_same_length,
    to_1d_array,
)

__all__ = [
    "AgreementReport",
    "Interval",
    "cohen_kappa",
    "gwet_ac1",
    "judge_agreement",
    "krippendorff_alpha",
    "wilson_interval",
]


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval.

    Attributes:
        point: The point estimate.
        low: Lower confidence limit.
        high: Upper confidence limit.
        level: Nominal coverage level (e.g. 0.95).
        method: Name of the interval construction, recorded so a report can state how
            each number was produced.
    """

    point: float
    low: float
    high: float
    level: float
    method: str

    @property
    def half_width(self) -> float:
        """Half the interval width, the usual '±' figure."""
        return (self.high - self.low) / 2.0

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}]"


def wilson_interval(successes: int, n: int, *, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a binomial proportion.

    Unlike the normal approximation ``p ± z·sqrt(p(1-p)/n)``, the Wilson interval stays
    inside [0, 1] and retains close-to-nominal coverage at small ``n`` and extreme ``p``.

    Args:
        successes: Number of successes, in ``[0, n]``.
        n: Number of trials, positive.
        alpha: Significance level; the interval has nominal coverage ``1 - alpha``.

    Returns:
        The interval, with ``point`` the plain sample proportion ``successes / n``.

    Raises:
        ValueError: If ``n`` is not positive, ``successes`` is out of range, or ``alpha``
            is outside (0, 1).

    References:
        tests/test_agreement.py::test_wilson_interval_matches_hand_computed_case
    """
    check_alpha(alpha)
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must lie in [0, {n}]; got {successes}")

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return Interval(
        point=p,
        low=float(max(0.0, center - half)),
        high=float(min(1.0, center + half)),
        level=1.0 - alpha,
        method="wilson",
    )


def _confusion_counts(judge: np.ndarray, gold: np.ndarray) -> tuple[int, int, int, int]:
    """Return (true positives, false positives, false negatives, true negatives)."""
    tp = int(np.sum((judge == 1) & (gold == 1)))
    fp = int(np.sum((judge == 1) & (gold == 0)))
    fn = int(np.sum((judge == 0) & (gold == 1)))
    tn = int(np.sum((judge == 0) & (gold == 0)))
    return tp, fp, fn, tn


def cohen_kappa(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Cohen's κ, chance-corrected agreement between two raters over nominal categories.

    ``κ = (p_o - p_e) / (1 - p_e)`` with ``p_o`` the observed agreement and ``p_e`` the
    agreement expected if the raters labeled independently with their observed marginals.

    Args:
        a: Labels from the first rater.
        b: Labels from the second rater, aligned elementwise with ``a``.

    Returns:
        κ in ``[-1, 1]``. Returns 1.0 when both raters use a single identical category
        (perfect agreement with no chance disagreement possible); this is the degenerate
        case where ``1 - p_e == 0``.

    References:
        tests/test_agreement.py::test_cohen_kappa_matches_hand_computed_case
    """
    arr_a = to_1d_array("a", a)
    arr_b = to_1d_array("b", b)
    check_same_length("a", arr_a, "b", arr_b)

    categories = np.unique(np.concatenate([arr_a, arr_b]))
    n = arr_a.shape[0]
    p_o = float(np.mean(arr_a == arr_b))
    p_e = 0.0
    for c in categories:
        p_e += float(np.mean(arr_a == c)) * float(np.mean(arr_b == c))
    if np.isclose(p_e, 1.0):
        # Both raters used one identical category throughout: agreement is perfect and
        # chance agreement is also 1, so the ratio is 0/0. Perfect agreement is the
        # meaningful reading.
        return 1.0 if np.isclose(p_o, 1.0) else 0.0
    del n
    return (p_o - p_e) / (1.0 - p_e)


def gwet_ac1(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Gwet's AC1 coefficient, chance-corrected agreement robust to class imbalance.

    AC1 replaces Cohen's chance term with ``p_e = (1/(K-1)) Σ_k π_k (1 - π_k)`` where
    ``π_k`` is the mean marginal prevalence of category ``k`` across raters. Unlike κ,
    it does not collapse toward zero when one category dominates.

    Args:
        a: Labels from the first rater.
        b: Labels from the second rater, aligned elementwise with ``a``.

    Returns:
        AC1 in ``[-1, 1]``.

    References:
        tests/test_agreement.py::test_gwet_ac1_is_stable_under_imbalance_where_kappa_collapses
    """
    arr_a = to_1d_array("a", a)
    arr_b = to_1d_array("b", b)
    check_same_length("a", arr_a, "b", arr_b)

    categories = np.unique(np.concatenate([arr_a, arr_b]))
    k = categories.size
    if k < 2:
        # A single observed category means no possible disagreement.
        return 1.0
    p_o = float(np.mean(arr_a == arr_b))
    p_e = 0.0
    for c in categories:
        pi = (float(np.mean(arr_a == c)) + float(np.mean(arr_b == c))) / 2.0
        p_e += pi * (1.0 - pi)
    p_e /= k - 1
    if np.isclose(p_e, 1.0):
        return 1.0 if np.isclose(p_o, 1.0) else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def _delta_squared(
    categories: np.ndarray, counts: np.ndarray, level: Literal["nominal", "ordinal", "interval"]
) -> np.ndarray:
    """Squared difference matrix between categories for Krippendorff's α."""
    k = categories.size
    if level == "nominal":
        return 1.0 - np.eye(k)
    if level == "interval":
        diff = categories.astype(float)[:, None] - categories.astype(float)[None, :]
        squared: np.ndarray = diff**2
        return squared
    # Ordinal: the metric depends on the observed marginal counts between the two ranks.
    cum = np.cumsum(counts)
    delta = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            lo, hi = (i, j) if i <= j else (j, i)
            interior = cum[hi] - (cum[lo] - counts[lo])
            delta[i, j] = (interior - (counts[lo] + counts[hi]) / 2.0) ** 2
    return delta


def krippendorff_alpha(
    ratings: npt.ArrayLike,
    *,
    level: Literal["nominal", "ordinal", "interval"] = "nominal",
) -> float:
    """Krippendorff's α for reliability across raters, tolerating missing ratings.

    Handles the realistic labeling setup that κ cannot: several human labelers, each
    covering an overlapping subset of examples.

    Args:
        ratings: 2-D array shaped ``(n_raters, n_units)``; use ``numpy.nan`` for a rating
            that a given rater did not provide.
        level: Difference metric. ``"nominal"`` treats categories as unordered,
            ``"ordinal"`` accounts for rank distance weighted by observed marginals, and
            ``"interval"`` uses squared numeric distance.

    Returns:
        α, where 1.0 is perfect reliability, 0.0 is chance, and negative values indicate
        systematic disagreement.

    Raises:
        ValueError: If ``ratings`` is not 2-D, or if fewer than two units have at least
            two ratings (α is undefined without pairable values).

    References:
        tests/test_agreement.py::test_krippendorff_alpha_matches_published_example
    """
    arr = np.asarray(ratings, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"ratings must be 2-D (n_raters, n_units); got shape {arr.shape}")

    observed = arr[:, ~np.all(np.isnan(arr), axis=0)]
    categories = np.unique(observed[~np.isnan(observed)])
    if categories.size == 0:
        raise ValueError("ratings contains no observed values")
    index = {float(c): i for i, c in enumerate(categories)}
    k = categories.size

    coincidence = np.zeros((k, k), dtype=float)
    pairable_units = 0
    for unit in range(observed.shape[1]):
        column = observed[:, unit]
        values = column[~np.isnan(column)]
        m = values.size
        if m < 2:
            continue
        pairable_units += 1
        for value_i in values:
            for value_j in values:
                # Ordered pairs excluding self-pairing, each weighted by 1/(m-1).
                coincidence[index[float(value_i)], index[float(value_j)]] += 1.0 / (m - 1)
        for value_i in values:
            coincidence[index[float(value_i)], index[float(value_i)]] -= 1.0 / (m - 1)

    if pairable_units < 2:
        raise ValueError("at least two units must have two or more ratings")

    marginals = coincidence.sum(axis=1)
    n_total = float(marginals.sum())
    delta = _delta_squared(categories, marginals, level)

    d_observed = float((coincidence * delta).sum()) / n_total
    expected = np.outer(marginals, marginals) - np.diag(marginals)
    d_expected = float((expected * delta).sum()) / (n_total * (n_total - 1.0))
    if np.isclose(d_expected, 0.0):
        return 1.0
    return 1.0 - d_observed / d_expected


def _bootstrap_interval(
    statistic_values: np.ndarray, point: float, alpha: float, method: str
) -> Interval:
    """Percentile bootstrap interval from resampled statistic values."""
    low, high = np.percentile(statistic_values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point=point, low=float(low), high=float(high), level=1.0 - alpha, method=method)


@dataclass(frozen=True)
class AgreementReport:
    """Judge-versus-gold agreement on a labeled subset.

    Attributes:
        n: Number of examples carrying both a judge label and a gold label.
        true_positives, false_positives, false_negatives, true_negatives: Confusion counts
            with the gold label treated as truth.
        accuracy: Overall agreement rate.
        sensitivity: ``P(judge = 1 | gold = 1)``, the true-positive rate.
        specificity: ``P(judge = 0 | gold = 0)``, the true-negative rate.
        precision: ``P(gold = 1 | judge = 1)``.
        cohen_kappa: Chance-corrected agreement, bootstrap interval.
        gwet_ac1: Imbalance-robust chance-corrected agreement, bootstrap interval.
        gold_prevalence: Rate of positive gold labels, which drives how much κ and AC1
            can diverge.
    """

    n: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    accuracy: Interval
    sensitivity: Interval
    specificity: Interval
    precision: Interval
    cohen_kappa: Interval
    gwet_ac1: Interval
    gold_prevalence: float

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        return "\n".join(
            [
                f"judge agreement on n={self.n} gold-labeled examples "
                f"(gold positive rate {self.gold_prevalence:.3f})",
                f"  accuracy    {self.accuracy}",
                f"  sensitivity {self.sensitivity}",
                f"  specificity {self.specificity}",
                f"  precision   {self.precision}",
                f"  Cohen κ     {self.cohen_kappa}",
                f"  Gwet AC1    {self.gwet_ac1}",
                f"  confusion   tp={self.true_positives} fp={self.false_positives} "
                f"fn={self.false_negatives} tn={self.true_negatives}",
            ]
        )


def judge_agreement(
    judge: npt.ArrayLike,
    gold: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> AgreementReport:
    """Measure how well a binary judge reproduces gold labels.

    Args:
        judge: Judge labels (0/1 or boolean) on the gold-labeled examples.
        gold: Trusted labels, aligned elementwise with ``judge``.
        alpha: Significance level for every reported interval.
        n_bootstrap: Resamples used for the κ and AC1 intervals, which have no simple
            closed form worth trusting at small ``n``.
        seed: Seed for the bootstrap, so a report is reproducible.

    Returns:
        An :class:`AgreementReport`.

    Raises:
        ValueError: If the inputs are not binary, differ in length, or contain no
            positive (or no negative) gold labels, which leaves sensitivity (or
            specificity) undefined.

    References:
        tests/test_agreement.py::test_judge_agreement_matches_hand_computed_confusion
    """
    check_alpha(alpha)
    judge_arr = check_binary("judge", judge)
    gold_arr = check_binary("gold", gold)
    check_same_length("judge", judge_arr, "gold", gold_arr)

    n_pos = int(np.sum(gold_arr == 1))
    n_neg = int(np.sum(gold_arr == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "gold labels must contain both classes; sensitivity and specificity are "
            f"undefined otherwise (got {n_pos} positive, {n_neg} negative)"
        )

    tp, fp, fn, tn = _confusion_counts(judge_arr, gold_arr)
    n = judge_arr.shape[0]

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    kappas = np.empty(n_bootstrap, dtype=float)
    ac1s = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        take = indices[b]
        kappas[b] = cohen_kappa(judge_arr[take], gold_arr[take])
        ac1s[b] = gwet_ac1(judge_arr[take], gold_arr[take])

    return AgreementReport(
        n=n,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        accuracy=wilson_interval(tp + tn, n, alpha=alpha),
        sensitivity=wilson_interval(tp, n_pos, alpha=alpha),
        specificity=wilson_interval(tn, n_neg, alpha=alpha),
        precision=wilson_interval(tp, tp + fp, alpha=alpha)
        if tp + fp > 0
        else Interval(float("nan"), float("nan"), float("nan"), 1 - alpha, "undefined"),
        cohen_kappa=_bootstrap_interval(
            kappas, cohen_kappa(judge_arr, gold_arr), alpha, "bootstrap-percentile"
        ),
        gwet_ac1=_bootstrap_interval(
            ac1s, gwet_ac1(judge_arr, gold_arr), alpha, "bootstrap-percentile"
        ),
        gold_prevalence=n_pos / n,
    )
