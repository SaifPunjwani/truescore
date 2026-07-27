# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Saif Punjwani
"""How many labels do I need?

Two planning questions, answered before spending money rather than after:

1. *How many human labels must I collect?* -- :func:`required_gold_labels` turns judge
   quality into a labeling budget under prediction-powered inference, and reports how
   many labels the same precision would have cost without the judge. That difference is
   the judge's value, in units of human hours.
2. *Is my evaluation set large enough to detect the improvement I care about?* --
   :func:`min_detectable_effect` and :func:`required_pairs` answer it for paired binary
   comparisons. Teams routinely run 200-example evals that cannot resolve anything
   smaller than a five-point change, then argue about two-point movements.

References:
    Connor (1987), "Sample size for testing differences in proportions for the paired
        sample design", Biometrics 43(1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from truescore._validation import check_alpha

__all__ = [
    "GoldBudget",
    "min_detectable_effect",
    "required_gold_labels",
    "required_pairs",
]


def _z(alpha: float) -> float:
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


@dataclass(frozen=True)
class GoldBudget:
    """A labeling plan for a target precision.

    Attributes:
        n_total: Examples available with judge labels.
        required_gold: Gold labels needed to reach ``target_half_width`` under PPI.
        achieved_half_width: Interval half-width at ``required_gold``.
        gold_only_required: Labels the same precision would need without the judge.
        labels_saved: ``gold_only_required - required_gold``; the judge's value expressed
            as human labels avoided.
        lambda_used: PPI tuning parameter at the recommended sample size.
        feasible: Whether the target is reachable at all with ``n_total`` examples. A
            judge cannot buy precision beyond what the unlabeled pool supports.
    """

    n_total: int
    required_gold: int
    achieved_half_width: float
    gold_only_required: int
    labels_saved: int
    lambda_used: float
    feasible: bool

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        if not self.feasible:
            return "\n".join(
                [
                    f"target precision is NOT reachable with {self.n_total} examples",
                    f"  best achievable half-width: ±{self.achieved_half_width:.4f} "
                    f"using {self.required_gold} gold labels",
                    "  collect more examples, or accept a wider interval",
                ]
            )
        return "\n".join(
            [
                f"labeling plan for {self.n_total} judge-labeled examples",
                f"  gold labels needed: {self.required_gold} "
                f"(half-width ±{self.achieved_half_width:.4f})",
                f"  without the judge:  {self.gold_only_required}",
                f"  the judge saves {self.labels_saved} human labels "
                f"({self.labels_saved / max(self.gold_only_required, 1):.0%})",
            ]
        )


def _ppi_half_width(
    n_gold: int,
    n_total: int,
    true_rate: float,
    sensitivity: float,
    specificity: float,
    alpha: float,
) -> tuple[float, float]:
    """Asymptotic PPI half-width and optimal λ for a binary metric under a judge model.

    Uses the population moments implied by ``(true_rate, sensitivity, specificity)``:
    the judge's positive rate is ``q = p·sens + (1-p)(1-spec)`` and its covariance with
    the truth is ``p·sens - p·q``.
    """
    n_unlabeled = n_total - n_gold
    if n_unlabeled <= 0 or n_gold < 2:
        return float("inf"), 0.0

    p = true_rate
    q = p * sensitivity + (1.0 - p) * (1.0 - specificity)
    var_y = p * (1.0 - p)
    var_f = q * (1.0 - q)
    cov = p * sensitivity - p * q

    if var_f <= 0.0:
        lam = 0.0
    else:
        lam = float(np.clip(cov / (var_f * (1.0 + n_gold / n_unlabeled)), 0.0, 1.0))

    var_rectifier = var_y - 2.0 * lam * cov + lam**2 * var_f
    variance = var_rectifier / n_gold + (lam**2) * var_f / n_unlabeled
    return _z(alpha) * float(np.sqrt(max(variance, 0.0))), lam


def required_gold_labels(
    n_total: int,
    *,
    target_half_width: float,
    true_rate: float = 0.5,
    sensitivity: float = 0.9,
    specificity: float = 0.9,
    alpha: float = 0.05,
) -> GoldBudget:
    """How many gold labels are needed to hit a target interval width, given a judge.

    Args:
        n_total: Examples carrying judge labels.
        target_half_width: Desired '±' precision on the rate, e.g. ``0.02`` for ±2 points.
        true_rate: Expected true pass rate. Precision is worst near 0.5, so leaving this
            at the default is the conservative choice.
        sensitivity: Judge's true-positive rate, from :func:`truescore.agreement`.
        specificity: Judge's true-negative rate.
        alpha: Significance level.

    Returns:
        A :class:`GoldBudget`. When the target is unreachable, ``feasible`` is ``False``
        and the fields describe the best achievable precision instead.

    Raises:
        ValueError: If ``n_total`` is below 3, the target is not positive, or any rate
            falls outside (0, 1).

    References:
        tests/test_power.py::test_required_gold_labels_achieves_target_in_simulation
        tests/test_power.py::test_better_judge_needs_fewer_gold_labels
    """
    check_alpha(alpha)
    if n_total < 3:
        raise ValueError(f"n_total must be at least 3; got {n_total}")
    if target_half_width <= 0.0:
        raise ValueError(f"target_half_width must be positive; got {target_half_width}")
    for name, value in (
        ("true_rate", true_rate),
        ("sensitivity", sensitivity),
        ("specificity", specificity),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie in (0, 1); got {value}")

    gold_only_required = int(
        np.ceil(_z(alpha) ** 2 * true_rate * (1.0 - true_rate) / target_half_width**2)
    )

    best_half = float("inf")
    best_n = n_total - 1
    best_lambda = 0.0
    for n_gold in range(2, n_total):
        half, lam = _ppi_half_width(n_gold, n_total, true_rate, sensitivity, specificity, alpha)
        if half < best_half:
            best_half, best_n, best_lambda = half, n_gold, lam
        if half <= target_half_width:
            return GoldBudget(
                n_total=n_total,
                required_gold=n_gold,
                achieved_half_width=half,
                gold_only_required=gold_only_required,
                labels_saved=max(0, gold_only_required - n_gold),
                lambda_used=lam,
                feasible=True,
            )

    return GoldBudget(
        n_total=n_total,
        required_gold=best_n,
        achieved_half_width=best_half,
        gold_only_required=gold_only_required,
        labels_saved=0,
        lambda_used=best_lambda,
        feasible=False,
    )


def required_pairs(
    effect: float,
    *,
    discordance_rate: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Paired examples needed to detect a difference between two systems.

    Args:
        effect: Difference in pass rates to detect, e.g. ``0.02`` for two points.
        discordance_rate: Fraction of examples where the two systems disagree. This, not
            the eval-set size, governs power: two near-identical systems disagree on few
            examples and therefore need many more of them. Measure it from a pilot run.
        alpha: Significance level.
        power: Desired probability of detecting a true effect of this size.

    Returns:
        Required number of paired examples.

    Raises:
        ValueError: If the effect is not positive, the discordance rate is outside (0, 1],
            or the effect exceeds the discordance rate (impossible: the difference in
            pass rates cannot exceed the rate at which the systems differ at all).

    References:
        tests/test_power.py::test_required_pairs_delivers_requested_power_in_simulation
    """
    check_alpha(alpha)
    if effect <= 0.0:
        raise ValueError(f"effect must be positive; got {effect}")
    if not 0.0 < discordance_rate <= 1.0:
        raise ValueError(f"discordance_rate must lie in (0, 1]; got {discordance_rate}")
    if effect > discordance_rate:
        raise ValueError(
            f"effect ({effect}) cannot exceed discordance_rate ({discordance_rate}): the "
            "pass-rate difference is bounded by how often the systems disagree at all"
        )
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must lie in (0, 1); got {power}")

    z_alpha = _z(alpha)
    z_beta = float(stats.norm.ppf(power))
    psi = discordance_rate
    numerator = z_alpha * np.sqrt(psi) + z_beta * np.sqrt(psi - effect**2)
    return int(np.ceil(numerator**2 / effect**2))


def min_detectable_effect(
    n_pairs: int,
    *,
    discordance_rate: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest difference an evaluation of this size can reliably detect.

    The number to check *before* declaring a winner. If the minimum detectable effect is
    five points and the observed difference is two, the evaluation cannot support the
    conclusion regardless of which system scored higher.

    Args:
        n_pairs: Paired examples available.
        discordance_rate: Fraction of examples where the two systems disagree.
        alpha: Significance level.
        power: Desired probability of detection.

    Returns:
        The minimum detectable difference in pass rates.

    Raises:
        ValueError: If arguments are out of range.

    References:
        tests/test_power.py::test_min_detectable_effect_inverts_required_pairs
    """
    check_alpha(alpha)
    if n_pairs < 2:
        raise ValueError(f"n_pairs must be at least 2; got {n_pairs}")
    if not 0.0 < discordance_rate <= 1.0:
        raise ValueError(f"discordance_rate must lie in (0, 1]; got {discordance_rate}")

    def shortfall(effect: float) -> float:
        needed = required_pairs(effect, discordance_rate=discordance_rate, alpha=alpha, power=power)
        return float(needed - n_pairs)

    upper = discordance_rate * (1.0 - 1e-9)
    if shortfall(upper) > 0.0:
        # Even the largest possible effect needs more pairs than are available.
        return upper
    lower = 1e-6
    if shortfall(lower) < 0.0:
        return lower
    return float(optimize.brentq(shortfall, lower, upper, xtol=1e-6))
