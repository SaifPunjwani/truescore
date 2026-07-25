"""What is the judge biased by?

An overall agreement rate hides *structure* in the judge's errors. A judge that agrees
with humans 88% of the time is fine if the 12% is random, and dangerous if the 12% is
concentrated on long answers -- because then any change that lengthens outputs will look
like an improvement.

This module regresses judge error on example covariates with heteroscedasticity-robust
(HC3) standard errors, and provides the standard position-bias test for pairwise judges.
HC3 is used rather than classical OLS errors because judge error variance is
systematically larger in the middle of the score range, which violates homoscedasticity
outright.

References:
    MacKinnon & White (1985), "Some heteroskedasticity-consistent covariance matrix
        estimators with improved finite sample properties" (HC3).
    Zheng et al. (2023), "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
        arXiv:2306.05685 -- position and verbosity bias in LLM judges.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from truescore._validation import check_alpha, check_binary, check_same_length, to_1d_array

__all__ = [
    "BiasEffect",
    "BiasReport",
    "PositionBiasResult",
    "judge_error_regression",
    "length_bias",
    "position_bias",
]


@dataclass(frozen=True)
class BiasEffect:
    """One covariate's estimated effect on judge error.

    Attributes:
        name: Covariate name.
        effect: Change in judge error (judge minus gold) per unit of the covariate.
            Positive means the judge over-scores as the covariate grows.
        std_error: HC3 robust standard error.
        low: Lower confidence limit.
        high: Upper confidence limit.
        p_value: Two-sided p-value against no effect.
        unit: Description of one unit of the covariate, for readable summaries.
    """

    name: str
    effect: float
    std_error: float
    low: float
    high: float
    p_value: float
    unit: str = "unit"

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero at the level it was constructed with."""
        return not (self.low <= 0.0 <= self.high)

    def __str__(self) -> str:
        flag = "" if self.significant else " (not distinguishable from zero)"
        return (
            f"{self.name}: {self.effect:+.5f} per {self.unit} "
            f"[{self.low:+.5f}, {self.high:+.5f}], p={self.p_value:.4g}{flag}"
        )


@dataclass(frozen=True)
class BiasReport:
    """Judge-error regression over one or more covariates.

    Attributes:
        n: Gold-labeled examples used.
        intercept: Mean judge error when every covariate is zero.
        mean_error: Average signed judge error; the overall over- or under-scoring.
        effects: One entry per covariate, in the order supplied.
    """

    n: int
    intercept: float
    mean_error: float
    effects: tuple[BiasEffect, ...]

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = [
            f"judge error regression on n={self.n} gold-labeled examples",
            f"  mean signed error: {self.mean_error:+.4f} "
            f"({'judge over-scores' if self.mean_error > 0 else 'judge under-scores'})",
        ]
        lines.extend(f"  {effect}" for effect in self.effects)
        return "\n".join(lines)


def _hc3_regression(
    design: npt.NDArray[Any], response: npt.NDArray[Any]
) -> tuple[np.ndarray, npt.NDArray[Any]]:
    """OLS with HC3 robust standard errors.

    Returns ``(coefficients, standard_errors)``. HC3 weights each squared residual by
    ``1/(1 - h_i)^2`` where ``h_i`` is the leverage, which corrects the downward bias of
    classical errors under heteroscedasticity in small samples.

    Rank deficiency raises rather than being absorbed by a pseudo-inverse: a collinear
    covariate has no separately identifiable effect, and reporting one anyway would be
    precisely the kind of quiet nonsense this library exists to catch.
    """
    n, k = design.shape
    if n <= k:
        raise ValueError(
            f"need more examples than parameters for a regression; got n={n}, parameters={k}"
        )
    if np.linalg.matrix_rank(design) < k:
        raise ValueError(
            "the covariates are collinear (a covariate is constant, duplicated, or an "
            "exact combination of others), so their effects are not separately "
            "identifiable; drop or combine them"
        )
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ response
    residuals = response - design @ beta
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    # Leverage of exactly 1 makes the HC3 weight infinite; guard so a duplicated design
    # row raises rather than silently producing infinite standard errors.
    if np.any(leverage >= 1.0 - 1e-12):
        raise ValueError(
            "a covariate row has leverage 1 (perfectly determined by the design); "
            "drop collinear or constant covariates"
        )
    weights = (residuals / (1.0 - leverage)) ** 2
    meat = design.T @ (design * weights[:, None])
    covariance = xtx_inv @ meat @ xtx_inv
    return beta, np.sqrt(np.diag(covariance))


def judge_error_regression(
    judge: npt.ArrayLike,
    gold: npt.ArrayLike,
    covariates: Mapping[str, npt.ArrayLike],
    *,
    alpha: float = 0.05,
    units: Mapping[str, str] | None = None,
) -> BiasReport:
    """Regress judge error on example covariates with HC3 robust standard errors.

    The response is ``judge - gold`` on the gold-labeled subset: positive where the judge
    is more generous than a human, negative where it is harsher.

    Args:
        judge: Judge labels or scores on the gold-labeled subset.
        gold: Gold labels or scores, aligned elementwise.
        covariates: Named per-example covariates, each aligned with ``judge``. Typical
            choices: response length, whether the response came from the judge's own
            model family, presence of markdown formatting, difficulty bucket.
        alpha: Significance level for the intervals.
        units: Optional per-covariate unit description used in summaries.

    Returns:
        A :class:`BiasReport` with one effect per covariate.

    Raises:
        ValueError: If no covariates are supplied, lengths disagree, or the design is
            collinear.

    References:
        tests/test_bias.py::test_regression_recovers_known_length_effect
        tests/test_bias.py::test_hc3_standard_errors_match_an_independent_implementation
    """
    check_alpha(alpha)
    judge_arr = to_1d_array("judge", np.asarray(judge, dtype=float))
    gold_arr = to_1d_array("gold", np.asarray(gold, dtype=float))
    check_same_length("judge", judge_arr, "gold", gold_arr)
    if not covariates:
        raise ValueError("supply at least one covariate to regress judge error on")

    names = list(covariates)
    columns = []
    for name in names:
        column = to_1d_array(name, np.asarray(covariates[name], dtype=float))
        check_same_length(name, column, "judge", judge_arr)
        columns.append(column)

    n = judge_arr.shape[0]
    design = np.column_stack([np.ones(n), *columns])
    error = judge_arr - gold_arr
    beta, se = _hc3_regression(design, error)

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    unit_map = dict(units or {})
    effects = []
    for i, name in enumerate(names, start=1):
        estimate, error_i = float(beta[i]), float(se[i])
        p = float(2.0 * stats.norm.sf(abs(estimate) / error_i)) if error_i > 0 else 1.0
        effects.append(
            BiasEffect(
                name=name,
                effect=estimate,
                std_error=error_i,
                low=estimate - z * error_i,
                high=estimate + z * error_i,
                p_value=p,
                unit=unit_map.get(name, "unit"),
            )
        )

    return BiasReport(
        n=n,
        intercept=float(beta[0]),
        mean_error=float(error.mean()),
        effects=tuple(effects),
    )


def length_bias(
    judge: npt.ArrayLike,
    gold: npt.ArrayLike,
    lengths: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    per: float = 100.0,
) -> BiasEffect:
    """Does the judge reward length? Reported per ``per`` units of length.

    The most consequential judge bias in practice: if the judge over-scores long answers,
    then any prompt change that lengthens outputs registers as a quality improvement.

    Args:
        judge: Judge labels or scores on the gold-labeled subset.
        gold: Gold labels or scores, aligned elementwise.
        lengths: Response length per example (tokens, characters -- your choice).
        alpha: Significance level.
        per: Scale for reporting, so the effect reads "per 100 tokens" rather than a
            number with five leading zeros.

    Returns:
        The length effect, scaled by ``per``.

    References:
        tests/test_bias.py::test_length_bias_reports_effect_per_scaled_unit
    """
    length_arr = to_1d_array("lengths", np.asarray(lengths, dtype=float))
    report = judge_error_regression(
        judge, gold, {"length": length_arr / per}, alpha=alpha, units={"length": f"{per:g} units"}
    )
    return report.effects[0]


@dataclass(frozen=True)
class PositionBiasResult:
    """Position-bias test for a pairwise judge evaluated in both presentation orders.

    Attributes:
        n_pairs: Comparisons judged in both orders.
        first_position_rate: How often the judge chose whichever option was presented
            first, pooled over both orders. 0.5 means no position bias.
        low: Lower confidence limit on that rate.
        high: Upper confidence limit.
        p_value: Two-sided exact binomial p-value against 0.5.
        consistency: Fraction of pairs where the judge chose the same option in both
            orders. A judge with no position bias and stable preferences scores 1.0.
    """

    n_pairs: int
    first_position_rate: float
    low: float
    high: float
    p_value: float
    consistency: float

    @property
    def significant(self) -> bool:
        """Whether the interval excludes 0.5."""
        return not (self.low <= 0.5 <= self.high)

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        direction = "first" if self.first_position_rate > 0.5 else "second"
        verdict = (
            f"biased toward the {direction}-presented option"
            if self.significant
            else "no detectable position bias"
        )
        return "\n".join(
            [
                f"position bias over n={self.n_pairs} pairs judged in both orders",
                f"  first-position win rate: {self.first_position_rate:.4f} "
                f"[{self.low:.4f}, {self.high:.4f}] (0.5 = unbiased)",
                f"  p = {self.p_value:.4g}: {verdict}",
                f"  order consistency: {self.consistency:.4f}",
            ]
        )


def position_bias(
    chose_first_original: npt.ArrayLike,
    chose_first_swapped: npt.ArrayLike,
    *,
    alpha: float = 0.05,
) -> PositionBiasResult:
    """Test whether a pairwise judge favors whichever option it sees first.

    Requires each comparison to be judged twice, once in each presentation order. A judge
    with no position bias picks the same *option* both times, so it picks the
    first-presented option exactly half the time; a judge that always picks position one
    scores 1.0.

    Args:
        chose_first_original: For each pair, 1 if the judge chose the option presented
            first in the original order.
        chose_first_swapped: The same indicator when the two options were swapped.
        alpha: Significance level.

    Returns:
        The position-bias result, with an exact binomial interval.

    Raises:
        ValueError: If inputs are not binary or differ in length.

    References:
        tests/test_bias.py::test_position_bias_detects_always_first_judge
        tests/test_bias.py::test_position_bias_reports_half_for_unbiased_judge
    """
    check_alpha(alpha)
    original = check_binary("chose_first_original", chose_first_original)
    swapped = check_binary("chose_first_swapped", chose_first_swapped)
    check_same_length("chose_first_original", original, "chose_first_swapped", swapped)

    n_pairs = original.shape[0]
    successes = int(original.sum() + swapped.sum())
    trials = 2 * n_pairs
    rate = successes / trials

    interval = stats.binomtest(successes, trials, 0.5).proportion_ci(
        confidence_level=1.0 - alpha, method="exact"
    )
    p_value = float(stats.binomtest(successes, trials, 0.5).pvalue)
    # Same option in both orders means the judge chose "first" in exactly one of the two
    # presentations.
    consistency = float(np.mean(original != swapped))

    return PositionBiasResult(
        n_pairs=n_pairs,
        first_position_rate=rate,
        low=float(interval.low),
        high=float(interval.high),
        p_value=p_value,
        consistency=consistency,
    )
