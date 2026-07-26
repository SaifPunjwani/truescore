"""Does this evaluation set look like production?

Almost never. Evaluation sets are curated, and curation is not random sampling: they
over-represent hard cases (because those are interesting), old cases (because that is when
the set was built), and whatever a previous incident made someone add. Every estimate in
this library is conditional on the evaluation set, and reports say so — but saying so does
not make the number match the traffic anyone cares about.

If you know roughly how production traffic splits across a few strata — question type,
customer tier, language — you can fix it. Estimate within each stratum, then recombine
using production's proportions rather than the evaluation set's:

$$\\hat\\theta = \\sum_k W_k \\hat\\theta_k, \\qquad
  \\operatorname{Var}(\\hat\\theta) = \\sum_k W_k^2 \\operatorname{Var}(\\hat\\theta_k)$$

where $W_k$ is the share of production traffic in stratum $k$. This is post-stratification.
It is old, it is simple, and it turns "our eval says 85%" into "our eval says 85%, but
weighted to how customers actually use this, 78%."

Two honest limits, stated here and in the returned report rather than buried:

- **The weights must be right.** A wrong production mix produces a confidently wrong number.
  The weights come from your traffic logs; this module cannot check them.
- **Within-stratum representativeness is still assumed.** Post-stratification fixes the
  *mix* of strata, not a biased sample inside one. If your hard questions are unrepresentative
  of production's hard questions, this does not help.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from truescore._validation import check_alpha, check_gold_index, to_1d_array
from truescore.correct import Estimate, gold_only_estimate, judge_only_estimate, ppi_estimate

__all__ = [
    "StratumEstimate",
    "WeightedEstimate",
    "eval_composition",
    "post_stratified_estimate",
]

MIN_GOLD_PER_STRATUM = 20


@dataclass(frozen=True)
class StratumEstimate:
    """One stratum's contribution to a weighted estimate.

    Attributes:
        name: Stratum label.
        weight: Share of production traffic assigned to this stratum.
        eval_share: Share of the evaluation set in this stratum. A large gap between this
            and ``weight`` is precisely why the weighted number differs from the raw one.
        n_total: Evaluation examples in the stratum.
        n_gold: Human-labeled examples in the stratum.
        estimate: The stratum's corrected estimate.
        method: Which estimator produced it.
    """

    name: str
    weight: float
    eval_share: float
    n_total: int
    n_gold: int
    estimate: Estimate
    method: str

    @property
    def over_represented(self) -> float:
        """How much more of the evaluation set this stratum holds than production does."""
        return self.eval_share - self.weight


@dataclass(frozen=True)
class WeightedEstimate:
    """An estimate reweighted to a stated production mix.

    Attributes:
        point: The weighted estimate.
        low: Lower confidence limit.
        high: Upper confidence limit.
        level: Nominal coverage.
        unweighted: The same data estimated without reweighting, for contrast.
        strata: Per-stratum detail.
        assumptions: What the number depends on, carried with it.
    """

    point: float
    low: float
    high: float
    level: float
    unweighted: Estimate
    strata: tuple[StratumEstimate, ...]
    assumptions: tuple[str, ...]

    @property
    def composition_effect(self) -> float:
        """Weighted minus unweighted: what the evaluation set's composition was worth."""
        return self.point - self.unweighted.point

    def summary(self) -> str:
        """Human-readable multi-line summary, most over-represented stratum first."""
        lines = [
            f"weighted to production mix: {self.point:.4f} "
            f"[{self.low:.4f}, {self.high:.4f}] at {self.level:.0%}",
            f"  unweighted (as the eval set is composed): {self.unweighted.point:.4f}",
            f"  composition effect: {self.composition_effect:+.4f}",
            "",
            f"  {'stratum':<16}{'prod':>8}{'eval':>8}{'n':>7}{'gold':>6}  estimate",
        ]
        for stratum in sorted(self.strata, key=lambda s: -abs(s.over_represented)):
            lines.append(
                f"  {stratum.name:<16}{stratum.weight:>8.3f}{stratum.eval_share:>8.3f}"
                f"{stratum.n_total:>7}{stratum.n_gold:>6}  "
                f"{stratum.estimate.point:.4f} "
                f"[{stratum.estimate.low:.4f}, {stratum.estimate.high:.4f}]"
            )
        return "\n".join(lines)


def post_stratified_estimate(
    judge: npt.ArrayLike,
    gold: npt.ArrayLike,
    gold_index: npt.ArrayLike,
    strata: npt.ArrayLike,
    population_weights: Mapping[str, float],
    *,
    alpha: float = 0.05,
    min_gold: int = MIN_GOLD_PER_STRATUM,
) -> WeightedEstimate:
    """Estimate the metric your production traffic would show, not your eval set.

    Each stratum is estimated with prediction-powered inference where it has enough gold
    labels and enough unlabeled examples, and with the classical gold-only estimator
    otherwise. The strata are then recombined using ``population_weights``.

    Args:
        judge: Judge verdicts for every evaluation example.
        gold: Gold verdicts for the labeled subset.
        gold_index: Positions in ``judge`` carrying a gold verdict.
        strata: Stratum label for every evaluation example.
        population_weights: Share of production traffic per stratum. Need not sum to
            exactly 1 — it is normalized, and the normalization is reported — but every
            stratum present in ``strata`` must appear, because silently dropping one would
            change the estimand without saying so.
        alpha: Significance level.
        min_gold: Gold labels a stratum needs before its correction is attempted.

    Returns:
        A :class:`WeightedEstimate`.

    Raises:
        ValueError: If a stratum has no weight, a weight is negative, the weights sum to
            zero, a stratum has no gold labels at all, or the arrays disagree in length.

    References:
        tests/test_weighting.py::test_reweighting_recovers_the_production_rate
        tests/test_weighting.py::test_missing_stratum_weight_is_rejected
    """
    check_alpha(alpha)
    if min_gold < 2:
        raise ValueError(f"min_gold must be at least 2; got {min_gold}")

    judge_array = to_1d_array("judge", np.asarray(judge, dtype=float))
    gold_array = to_1d_array("gold", np.asarray(gold, dtype=float))
    index = check_gold_index(gold_index, judge_array.shape[0])
    labels = to_1d_array("strata", np.asarray(strata))
    if labels.shape[0] != judge_array.shape[0]:
        raise ValueError(
            f"strata must cover every example; got {labels.shape[0]} labels for "
            f"{judge_array.shape[0]} examples"
        )

    present = [str(value) for value in np.unique(labels)]
    missing = [name for name in present if name not in population_weights]
    if missing:
        raise ValueError(
            f"no production weight given for strata {missing}. Every stratum in the "
            "evaluation set needs a weight; dropping one silently would change what is "
            "being estimated."
        )
    if any(population_weights[name] < 0 for name in present):
        raise ValueError("population weights must be non-negative")
    total_weight = float(sum(population_weights[name] for name in present))
    if total_weight <= 0:
        raise ValueError("population weights must sum to a positive value")

    gold_by_position = {int(position): gold_array[i] for i, position in enumerate(index)}
    n_examples = judge_array.shape[0]

    strata_estimates: list[StratumEstimate] = []
    weighted_point = 0.0
    weighted_variance = 0.0
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))

    for name in present:
        positions = np.flatnonzero(labels == name)
        local_gold_positions = [int(p) for p in positions if int(p) in gold_by_position]
        if not local_gold_positions:
            raise ValueError(
                f"stratum {name!r} has no human labels, so its rate cannot be estimated "
                "and the weighted total would be undefined"
            )

        local_gold = np.asarray([gold_by_position[p] for p in local_gold_positions])
        remap = {int(p): i for i, p in enumerate(positions)}
        local_index = np.asarray([remap[p] for p in local_gold_positions])
        slice_judge = judge_array[positions]

        enough_gold = len(local_gold_positions) >= min_gold
        enough_unlabeled = len(local_gold_positions) < positions.shape[0] - 1
        if enough_gold and enough_unlabeled:
            estimate = ppi_estimate(slice_judge, local_gold, local_index, alpha=alpha)
            method = "ppi++"
        else:
            estimate = gold_only_estimate(local_gold, alpha=alpha, n_total=int(positions.shape[0]))
            method = "gold_only (too few labels or no unlabeled examples for PPI)"

        weight = population_weights[name] / total_weight
        weighted_point += weight * estimate.point
        # Strata are disjoint, so their estimates are independent and variances add under
        # the squared weights.
        standard_error = estimate.half_width / z if z > 0 else 0.0
        weighted_variance += (weight**2) * standard_error**2

        strata_estimates.append(
            StratumEstimate(
                name=name,
                weight=weight,
                eval_share=float(positions.shape[0]) / n_examples,
                n_total=int(positions.shape[0]),
                n_gold=len(local_gold_positions),
                estimate=estimate,
                method=method,
            )
        )

    half = z * float(np.sqrt(max(weighted_variance, 0.0)))
    unweighted = judge_only_estimate(judge_array, alpha=alpha)

    return WeightedEstimate(
        point=float(np.clip(weighted_point, 0.0, 1.0)),
        low=float(np.clip(weighted_point - half, 0.0, 1.0)),
        high=float(np.clip(weighted_point + half, 0.0, 1.0)),
        level=1.0 - alpha,
        unweighted=unweighted,
        strata=tuple(strata_estimates),
        assumptions=(
            "the supplied production weights describe current traffic; a wrong mix "
            "produces a confidently wrong number, and this cannot be checked from the "
            "evaluation set",
            "within each stratum the evaluation examples are representative of production "
            "examples in that stratum -- post-stratification fixes the mix, not a biased "
            "sample inside a stratum",
            "gold labels are a random sample within each stratum",
        ),
    )


def eval_composition(strata: npt.ArrayLike) -> Mapping[str, float]:
    """Share of the evaluation set in each stratum, for comparison against traffic."""
    values, counts = np.unique(np.asarray(strata), return_counts=True)
    total = float(counts.sum())
    return {str(value): float(count) / total for value, count in zip(values, counts, strict=True)}
