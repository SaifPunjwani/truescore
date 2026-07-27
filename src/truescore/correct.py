# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Saif Punjwani
"""What is the true score, given an imperfect judge?

A judge label is a cheap, biased measurement of the quantity you care about. Averaging
judge labels estimates *the judge's* pass rate, not the system's. This module estimates
the true rate from a large set of judge labels plus a small set of gold labels, and
reports an interval that is valid rather than optimistic.

Four estimators, all returning the same :class:`Estimate` structure so a report can put
them side by side:

- :func:`judge_only_estimate` -- the naive average of judge labels. Included so a report
  can show what the team would otherwise have published, and how far off it is.
- :func:`gold_only_estimate` -- the classical estimate on the labeled subset. Unbiased,
  but wastes every unlabeled example and is therefore wide.
- :func:`ppi_estimate` -- prediction-powered inference. Combines both: unbiased like
  gold-only, but tightened by the judge in proportion to how informative the judge is.
- :func:`rogan_gladen_estimate` -- the classical misclassification correction for binary
  labels, parameterized by measured sensitivity and specificity.

The central guarantee of PPI is worth stating plainly: coverage does not depend on the
judge being good. A useless judge makes the interval no tighter than gold-only; it does
not make it wrong. That asymmetry is what makes the method safe to adopt before you know
how good your judge is.

References:
    Angelopoulos, Bates, Fannjiang, Jordan, Zrnic (2023), "Prediction-powered inference",
        Science 382(6671).
    Angelopoulos, Bates, Zrnic (2023), "PPI++: Efficient prediction-powered inference",
        arXiv:2311.01453.
    Rogan & Gladen (1978), "Estimating prevalence from the results of a screening test",
        American Journal of Epidemiology 107(1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from truescore._validation import (
    check_alpha,
    check_binary,
    check_gold_index,
    check_same_length,
    to_1d_array,
)
from truescore.agreement import wilson_interval

__all__ = [
    "Estimate",
    "gold_only_estimate",
    "judge_only_estimate",
    "ppi_estimate",
    "rogan_gladen_estimate",
]


@dataclass(frozen=True)
class Estimate:
    """An estimate of a scalar metric with an interval and its provenance.

    Attributes:
        point: The point estimate.
        low: Lower confidence limit.
        high: Upper confidence limit.
        level: Nominal coverage level.
        method: Estimator name, recorded so a report states how the number was produced.
        n_total: Examples carrying a judge label.
        n_gold: Examples carrying a gold label.
        assumptions: Conditions the estimate relies on. Written out because an interval
            without its assumptions is not evidence.
        lambda_: PPI tuning parameter, when applicable.
    """

    point: float
    low: float
    high: float
    level: float
    method: str
    n_total: int
    n_gold: int
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    lambda_: float | None = None

    @property
    def half_width(self) -> float:
        """Half the interval width, the usual '±' figure."""
        return (self.high - self.low) / 2.0

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}] ({self.method})"


def _z(alpha: float) -> float:
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


def _cluster_codes(name: str, clusters: npt.ArrayLike, expected: int) -> npt.NDArray[Any]:
    """Turn cluster labels into integer codes, checking the length matches."""
    arr = np.asarray(clusters)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {arr.shape}")
    if arr.shape[0] != expected:
        raise ValueError(
            f"{name} has {arr.shape[0]} entries but there are {expected} observations; "
            "every observation needs to say which cluster it belongs to"
        )
    _, codes = np.unique(arr, return_inverse=True)
    return codes


def _clustered_mean_variance(values: npt.NDArray[Any], codes: npt.NDArray[Any]) -> float:
    """Variance of a sample mean when observations arrive in correlated groups.

    Five epochs of the same evaluation sample are five looks at one draw, not five draws.
    Treating them as independent divides the variance by five times too much, and the
    interval comes out narrow enough to miss the truth far more often than its nominal
    rate promises.

    The estimator is the standard cluster-robust one, summing residuals within a cluster
    before squaring::

        Var(x̄) = G/(G-1) · (1/n²) · Σ_g (Σ_{i∈g} (x_i - x̄))²

    With one observation per cluster this reduces exactly to ``var(x, ddof=1)/n``, which
    is what makes it safe as a general replacement rather than a special case.
    """
    n = values.shape[0]
    residual = values - values.mean()
    per_cluster = np.bincount(codes, weights=residual)
    groups = per_cluster.shape[0]
    if groups < 2:
        raise ValueError(
            f"cluster-robust variance needs at least 2 clusters; got {groups}. "
            "With a single cluster there is no replication to estimate variance from."
        )
    return float(groups / (groups - 1) * np.sum(per_cluster**2) / n**2)


def _is_binary(values: npt.NDArray[Any]) -> bool:
    return bool(np.all(np.isin(np.unique(values), (0, 1))))


def judge_only_estimate(judge: npt.ArrayLike, *, alpha: float = 0.05) -> Estimate:
    """Naive average of judge labels, with an interval that ignores judge error.

    This is what most evaluation reports publish. It is included so that a truescore
    report can show it next to a corrected estimate: the gap between them is the size of
    the mistake being made.

    Args:
        judge: Judge labels for every example.
        alpha: Significance level.

    Returns:
        The estimate, flagged with the assumption that makes it wrong in practice.

    References:
        tests/test_correct.py::test_judge_only_ignores_judge_error
    """
    check_alpha(alpha)
    arr = to_1d_array("judge", np.asarray(judge, dtype=float))
    n = arr.shape[0]
    if _is_binary(arr):
        interval = wilson_interval(int(arr.sum()), n, alpha=alpha)
        low, high, method = interval.low, interval.high, "judge_only (wilson)"
    else:
        se = float(np.std(arr, ddof=1) / np.sqrt(n))
        half = _z(alpha) * se
        mean = float(arr.mean())
        low, high = mean - half, mean + half
        method = "judge_only (normal)"
    return Estimate(
        point=float(arr.mean()),
        low=low,
        high=high,
        level=1.0 - alpha,
        method=method,
        n_total=n,
        n_gold=0,
        assumptions=(
            "the judge is unbiased for the target metric, which is false whenever the "
            "judge's error rates on positives and negatives differ",
        ),
    )


def gold_only_estimate(
    gold: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    n_total: int = 0,
    clusters: npt.ArrayLike | None = None,
) -> Estimate:
    """Classical estimate from gold labels alone: unbiased, and wider than it needs to be.

    Args:
        gold: Trusted labels on the labeled subset.
        alpha: Significance level.
        n_total: Total examples available, recorded for the report; does not affect the
            estimate, since this estimator discards unlabeled examples by construction.
        clusters: Group label per observation, when observations are not independent.
            Repeated epochs of one evaluation sample, or several turns of one
            conversation, are the common cases. Supplying this widens the interval to
            account for the correlation; omitting it when the data is clustered produces
            an interval that is too narrow.

    Returns:
        The estimate.

    Raises:
        ValueError: If ``clusters`` has the wrong length or names fewer than 2 groups.

    References:
        tests/test_correct.py::test_gold_only_is_unbiased_and_wider_than_ppi
        tests/test_correct.py::test_clustered_data_undercovers_until_clusters_are_declared
    """
    check_alpha(alpha)
    arr = to_1d_array("gold", np.asarray(gold, dtype=float))
    n = arr.shape[0]
    if clusters is not None:
        codes = _cluster_codes("clusters", clusters, n)
        groups = int(codes.max()) + 1
        # Wilson assumes independent Bernoulli trials, so it does not apply here however
        # binary the labels look.
        se = float(np.sqrt(_clustered_mean_variance(arr, codes)))
        half = float(stats.t.ppf(1.0 - alpha / 2.0, df=groups - 1)) * se
        return Estimate(
            point=float(arr.mean()),
            low=float(arr.mean() - half),
            high=float(arr.mean() + half),
            level=1.0 - alpha,
            method=f"gold_only (cluster-robust, {groups} clusters)",
            n_total=max(n_total, n),
            n_gold=n,
            assumptions=(
                "gold labels are a random sample of the evaluation set",
                "clusters are independent of one another; correlation within a cluster "
                "is unrestricted",
            ),
        )
    if _is_binary(arr):
        interval = wilson_interval(int(arr.sum()), n, alpha=alpha)
        low, high, method = interval.low, interval.high, "gold_only (wilson)"
    else:
        if n < 2:
            raise ValueError("gold_only_estimate needs at least 2 labels for an interval")
        se = float(np.std(arr, ddof=1) / np.sqrt(n))
        half = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)) * se
        low, high, method = float(arr.mean() - half), float(arr.mean() + half), "gold_only (t)"
    return Estimate(
        point=float(arr.mean()),
        low=low,
        high=high,
        level=1.0 - alpha,
        method=method,
        n_total=max(n_total, n),
        n_gold=n,
        assumptions=("gold labels are a random sample of the evaluation set",),
    )


def _optimal_lambda(
    gold_values: npt.NDArray[Any],
    judge_labeled: npt.NDArray[Any],
    judge_unlabeled: npt.NDArray[Any],
) -> float:
    """Variance-minimizing PPI++ tuning parameter, clipped to [0, 1].

    Minimizes ``Var(Y - λf)/n + λ²Var(f)/N_u``, giving
    ``λ* = Cov(Y, f) / (Var(f)(1 + n/N_u))``. Clipping to [0, 1] keeps the estimator
    between the classical estimate (λ=0) and vanilla PPI (λ=1); coverage holds for any
    fixed λ, and choosing λ from the same sample contributes only a higher-order term.
    """
    n = gold_values.shape[0]
    n_unlabeled = judge_unlabeled.shape[0]
    var_f = float(np.var(judge_labeled, ddof=1))
    if var_f <= 0.0:
        # A constant judge carries no information; fall back to the classical estimator.
        return 0.0
    cov = float(np.cov(gold_values, judge_labeled, ddof=1)[0, 1])
    lam = cov / (var_f * (1.0 + n / n_unlabeled))
    return float(np.clip(lam, 0.0, 1.0))


def ppi_estimate(
    judge: npt.ArrayLike,
    gold: npt.ArrayLike,
    gold_index: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    lambda_: float | None = None,
    clusters: npt.ArrayLike | None = None,
) -> Estimate:
    """Prediction-powered estimate of the true mean, using judge labels to tighten gold.

    The estimator is

    ``θ̂_λ = (λ/N_u) Σ_unlabeled f_j + (1/n) Σ_labeled (Y_i − λ f_i)``

    where ``f`` are judge labels, ``Y`` gold labels, ``n`` the labeled count and ``N_u``
    the unlabeled count. The first term borrows strength from unlabeled examples; the
    second removes the judge's bias by measuring it directly on the labeled subset. With
    ``λ = 0`` this reduces exactly to the gold-only estimate, so the method can never do
    much worse than ignoring the judge.

    Args:
        judge: Judge labels for all ``N`` examples.
        gold: Gold labels for the labeled subset, aligned with ``gold_index``.
        gold_index: Positions in ``judge`` that carry a gold label.
        alpha: Significance level.
        lambda_: Fixed tuning parameter. ``None`` selects the variance-minimizing value.
        clusters: Group label for each of the ``N`` examples, when observations are not
            independent. A cluster must fall entirely inside the labeled set or entirely
            outside it, because the estimator's two terms are only independent when the
            labeled and unlabeled sets share no cluster. Sampling whole clusters for
            labeling gives that for free, and is the design this argument is for.

    Returns:
        The estimate, with the selected ``lambda_`` recorded.

    Raises:
        ValueError: If every example is gold-labeled (no unlabeled examples remain, so
            there is nothing to borrow strength from -- use :func:`gold_only_estimate`),
            if fewer than two gold labels are supplied, or if a cluster straddles the
            labeled and unlabeled sets.

    References:
        tests/test_correct.py::test_ppi_covers_at_nominal_rate_under_simulation
        tests/test_correct.py::test_ppi_reduces_to_gold_only_at_lambda_zero
        tests/test_correct.py::test_ppi_covers_clustered_data_when_clusters_are_declared
    """
    check_alpha(alpha)
    judge_arr = to_1d_array("judge", np.asarray(judge, dtype=float))
    gold_arr = to_1d_array("gold", np.asarray(gold, dtype=float))
    idx = check_gold_index(gold_index, judge_arr.shape[0])
    check_same_length("gold", gold_arr, "gold_index", idx)

    n = gold_arr.shape[0]
    if n < 2:
        raise ValueError(f"ppi_estimate needs at least 2 gold labels; got {n}")

    unlabeled_mask = np.ones(judge_arr.shape[0], dtype=bool)
    unlabeled_mask[idx] = False
    judge_unlabeled = judge_arr[unlabeled_mask]
    if judge_unlabeled.size == 0:
        raise ValueError(
            "every example is gold-labeled, so there is no unlabeled set to borrow "
            "strength from; use gold_only_estimate instead"
        )
    if judge_unlabeled.size < 2:
        # The unlabeled variance term needs at least two observations to be estimable;
        # with one, the interval would be silently NaN.
        raise ValueError(
            f"ppi_estimate needs at least 2 unlabeled examples; got {judge_unlabeled.size}. "
            "With this little unlabeled data the judge cannot contribute precision -- use "
            "gold_only_estimate instead."
        )
    judge_labeled = judge_arr[idx]

    lam = _optimal_lambda(gold_arr, judge_labeled, judge_unlabeled) if lambda_ is None else lambda_
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lambda_ must lie in [0, 1]; got {lam}")

    n_unlabeled = judge_unlabeled.shape[0]
    rectifier = gold_arr - lam * judge_labeled
    point = lam * float(judge_unlabeled.mean()) + float(rectifier.mean())

    cluster_note = ""
    if clusters is None:
        variance = float(np.var(rectifier, ddof=1)) / n
        if lam > 0.0:
            variance += (lam**2) * float(np.var(judge_unlabeled, ddof=1)) / n_unlabeled
        half = _z(alpha) * float(np.sqrt(variance))
    else:
        codes = _cluster_codes("clusters", clusters, judge_arr.shape[0])
        labeled_codes, unlabeled_codes = codes[idx], codes[unlabeled_mask]
        straddling = np.intersect1d(labeled_codes, unlabeled_codes)
        if straddling.size:
            raise ValueError(
                f"{straddling.size} cluster(s) have some examples labeled and some not. "
                "The two terms of the estimator are only independent when the labeled and "
                "unlabeled sets share no cluster, so label whole clusters at a time, or "
                "drop the unlabeled remainder of a partly-labeled cluster."
            )
        labeled_groups = np.unique(labeled_codes, return_inverse=True)[1]
        variance = _clustered_mean_variance(rectifier, labeled_groups)
        groups_labeled = int(np.unique(labeled_codes).size)
        groups_unlabeled = int(np.unique(unlabeled_codes).size)
        if lam > 0.0:
            variance += (lam**2) * _clustered_mean_variance(
                judge_unlabeled, np.unique(unlabeled_codes, return_inverse=True)[1]
            )
        # Two variance terms with different cluster counts. Taking the smaller count is
        # conservative and avoids a Satterthwaite approximation nobody would check.
        df = min(groups_labeled, groups_unlabeled) - 1
        half = float(stats.t.ppf(1.0 - alpha / 2.0, df=df)) * float(np.sqrt(variance))
        cluster_note = f" (cluster-robust, {groups_labeled}+{groups_unlabeled} clusters)"
    method = "ppi++" + cluster_note

    # The asymptotic interval is built from the sample variance of the rectifier, and that
    # estimate is worthless when the gold labels carry almost no spread: a stratum where
    # the system is right 99% of the time will routinely draw a gold sample that is
    # entirely 1s, giving an estimated variance of exactly zero and an interval of width
    # zero that misses the truth. Binary labels have an exact alternative, so use it
    # whenever the normal approximation is out of its depth -- the usual criterion is fewer
    # than five observations in the smaller class.
    if _is_binary(gold_arr):
        successes = int(gold_arr.sum())
        if min(successes, n - successes) < 5:
            exact = wilson_interval(successes, n, alpha=alpha)
            if (exact.high - exact.low) / 2.0 > half:
                half = (exact.high - exact.low) / 2.0
                method += (
                    " (widened to the exact interval; too few of one class to trust the "
                    "normal approximation)"
                )

    return Estimate(
        point=point,
        low=point - half,
        high=point + half,
        level=1.0 - alpha,
        method=method,
        n_total=judge_arr.shape[0],
        n_gold=n,
        assumptions=(
            "gold labels are a random sample of the evaluation set",
            "judge labels are available for every example and were produced the same way "
            "for labeled and unlabeled examples",
            "the interval is asymptotic (normal); with fewer than roughly 30 gold labels "
            "prefer gold_only_estimate, whose interval is exact",
        ),
        lambda_=lam,
    )


def rogan_gladen_estimate(
    judge: npt.ArrayLike,
    gold: npt.ArrayLike,
    gold_index: npt.ArrayLike,
    *,
    alpha: float = 0.05,
) -> Estimate:
    """Misclassification-corrected rate for a binary judge with measured error rates.

    Estimates sensitivity and specificity on the gold-labeled subset, then inverts the
    relationship ``p_observed = p·sens + (1−p)·(1−spec)`` to recover the true rate:

    ``p̂ = (p_observed + spec − 1) / (sens + spec − 1)``

    The interval uses the delta method, propagating uncertainty in all three estimated
    quantities. Where the correction is large, prefer :func:`ppi_estimate`, whose validity
    does not depend on the delta-method approximation.

    Args:
        judge: Binary judge labels for all examples.
        gold: Binary gold labels for the labeled subset.
        gold_index: Positions in ``judge`` carrying a gold label.
        alpha: Significance level.

    Returns:
        The estimate, with the point clipped to [0, 1] (the raw correction can fall
        outside the unit interval when error rates are estimated noisily).

    Raises:
        ValueError: If the gold subset lacks either class, or if
            ``sensitivity + specificity <= 1``, where the correction is undefined -- a
            judge that uninformative carries no recoverable signal.

    References:
        tests/test_correct.py::test_rogan_gladen_recovers_known_prevalence
        tests/test_correct.py::test_rogan_gladen_rejects_uninformative_judge
    """
    check_alpha(alpha)
    judge_arr = check_binary("judge", judge)
    gold_arr = check_binary("gold", gold)
    idx = check_gold_index(gold_index, judge_arr.shape[0])
    check_same_length("gold", gold_arr, "gold_index", idx)

    judge_labeled = judge_arr[idx]
    n_pos = int(np.sum(gold_arr == 1))
    n_neg = int(np.sum(gold_arr == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "gold labels must contain both classes to estimate sensitivity and "
            f"specificity (got {n_pos} positive, {n_neg} negative)"
        )

    sens = float(np.mean(judge_labeled[gold_arr == 1]))
    spec = float(np.mean(1 - judge_labeled[gold_arr == 0]))
    youden = sens + spec - 1.0
    if youden <= 0.0:
        raise ValueError(
            f"sensitivity + specificity must exceed 1 for the correction to be defined; "
            f"got sensitivity={sens:.3f}, specificity={spec:.3f}. A judge at or below "
            "chance carries no recoverable signal."
        )

    n_total = judge_arr.shape[0]
    p_obs = float(judge_arr.mean())
    theta = (p_obs + spec - 1.0) / youden

    # Delta method: θ depends on (p_obs, sens, spec) with the partials below. The three
    # estimates are treated as independent, which slightly understates variance because
    # the gold subset also contributes to p_obs; the bootstrap in the report layer is the
    # check on that approximation.
    d_p = 1.0 / youden
    d_sens = -theta / youden
    d_spec = (1.0 - theta) / youden
    var = (
        d_p**2 * p_obs * (1.0 - p_obs) / n_total
        + d_sens**2 * sens * (1.0 - sens) / n_pos
        + d_spec**2 * spec * (1.0 - spec) / n_neg
    )
    half = _z(alpha) * float(np.sqrt(max(var, 0.0)))

    return Estimate(
        point=float(np.clip(theta, 0.0, 1.0)),
        low=float(np.clip(theta - half, 0.0, 1.0)),
        high=float(np.clip(theta + half, 0.0, 1.0)),
        level=1.0 - alpha,
        method="rogan-gladen (delta method)",
        n_total=n_total,
        n_gold=gold_arr.shape[0],
        assumptions=(
            "judge error rates are constant across the evaluation set, so sensitivity "
            "and specificity measured on the gold subset transfer to the whole set",
            "gold labels are a random sample of the evaluation set",
            "the delta-method interval is a first-order approximation and degrades when "
            "sensitivity + specificity is close to 1",
        ),
    )
