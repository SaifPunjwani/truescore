"""Is my evaluation set inside my training data?

A contaminated benchmark reports skill the model does not have, and the failure is
invisible from the score alone: a memorized answer and a reasoned answer look identical.
Most contamination checks are heuristic string matching against a training corpus that,
for a hosted model, you do not have.

There is a test that needs neither the training corpus nor model internals, only
log-likelihoods you can obtain from any model that returns them. The observation is that
a benchmark is a *sequence* of examples in some canonical order, and that order carries
no information about the task. If a model never saw the dataset, the log-likelihood it
assigns to the examples concatenated in canonical order is exchangeable with the
log-likelihood under any permutation of that order. If the model was trained on the
dataset as published, the canonical order is special -- it is the one it memorized -- and
scores unusually high.

That yields an **exact permutation test**: no asymptotics, no distributional assumption,
and a false-positive rate equal to the level by construction. What you supply is the
log-likelihood of the canonical concatenation and of a set of shuffled ones; computing
those is your model's job, not this library's.

References:
    Oren, Meister, Chatterji, Ladhak, Hashimoto (2023), "Proving Test Set Contamination in
        Black-Box Language Models", arXiv:2310.17623.
    Fisher (1932), "Statistical Methods for Research Workers" (combining independent
        p-values).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from truescore._validation import check_alpha, to_1d_array

__all__ = [
    "CombinedContamination",
    "ContaminationResult",
    "combine_shards",
    "exchangeability_test",
]


@dataclass(frozen=True)
class ContaminationResult:
    """Evidence that a model saw an evaluation set in its published order.

    Attributes:
        p_value: Exact permutation p-value. Small means the canonical order scores higher
            than shuffled orders more often than chance allows.
        canonical_loglik: Log-likelihood of the examples in canonical order.
        permutation_mean: Mean log-likelihood across the shuffled orders.
        permutation_std: Standard deviation across the shuffled orders.
        z_score: Standardized position of the canonical order among the permutations,
            reported as an effect size. It is descriptive only; the p-value is exact and
            does not rely on it.
        n_permutations: Shuffled orders supplied.
        level: One minus the significance level used for the verdict.
    """

    p_value: float
    canonical_loglik: float
    permutation_mean: float
    permutation_std: float
    z_score: float
    n_permutations: int
    level: float

    @property
    def contaminated(self) -> bool:
        """Whether the null of no contamination is rejected at the chosen level."""
        return self.p_value < (1.0 - self.level)

    @property
    def resolution(self) -> float:
        """Smallest p-value the supplied number of permutations can produce.

        With ``m`` permutations no result can be more significant than ``1/(m+1)``, so a
        test run with 19 permutations cannot reach 0.01 no matter how contaminated the
        model is. Reporting this prevents reading a floor as a finding.
        """
        return 1.0 / (self.n_permutations + 1)

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        verdict = (
            "canonical order scores higher than chance allows: evidence of contamination"
            if self.contaminated
            else "no evidence that the canonical order is special"
        )
        return "\n".join(
            [
                f"contamination test on {self.n_permutations} permutations",
                f"  canonical log-likelihood: {self.canonical_loglik:.4f}",
                f"  shuffled orders: mean {self.permutation_mean:.4f}, "
                f"sd {self.permutation_std:.4f} (z = {self.z_score:+.2f})",
                f"  p = {self.p_value:.4g} (finest resolution possible here: "
                f"{self.resolution:.4g})",
                f"  {verdict}",
            ]
        )


def exchangeability_test(
    canonical_loglik: float,
    permuted_logliks: npt.ArrayLike,
    *,
    alpha: float = 0.05,
) -> ContaminationResult:
    """Exact permutation test for memorization of an evaluation set's canonical order.

    Under the null that the model never saw the dataset, the canonical ordering is just
    one of the possible orderings, so its log-likelihood is exchangeable with the shuffled
    ones. The p-value is therefore the rank of the canonical value among the permutations,

    ``p = (1 + #{shuffled ≥ canonical}) / (m + 1)``

    which is exact: it needs no assumption about the distribution of log-likelihoods,
    which is fortunate, because that distribution depends on the model, the tokenizer and
    the dataset in ways nobody can characterize.

    Args:
        canonical_loglik: Log-likelihood the model assigns to the examples concatenated in
            the dataset's published order.
        permuted_logliks: Log-likelihoods under independently shuffled orders. Use at
            least 99 for a test capable of reaching p = 0.01.
        alpha: Significance level for the verdict.

    Returns:
        A :class:`ContaminationResult`.

    Raises:
        ValueError: If no permutations are supplied or any value is non-finite.

    Note:
        The test detects memorization of the *published order*. A model trained on the
        examples in a shuffled order, or on paraphrases, can be contaminated and pass;
        a non-significant result is therefore an absence of evidence, not evidence of
        absence, and the summary says so.

    References:
        tests/test_contamination.py::test_null_false_positive_rate_matches_alpha
        tests/test_contamination.py::test_memorized_canonical_order_is_detected
    """
    check_alpha(alpha)
    if not np.isfinite(canonical_loglik):
        raise ValueError("canonical_loglik must be finite")
    permuted = to_1d_array("permuted_logliks", np.asarray(permuted_logliks, dtype=float))
    if not np.all(np.isfinite(permuted)):
        raise ValueError("permuted_logliks contains non-finite entries")

    m = int(permuted.shape[0])
    at_least_as_extreme = int(np.sum(permuted >= canonical_loglik))
    p_value = (1.0 + at_least_as_extreme) / (m + 1.0)

    mean = float(permuted.mean())
    std = float(permuted.std(ddof=1)) if m > 1 else 0.0
    z = (canonical_loglik - mean) / std if std > 0.0 else 0.0

    return ContaminationResult(
        p_value=p_value,
        canonical_loglik=float(canonical_loglik),
        permutation_mean=mean,
        permutation_std=std,
        z_score=float(z),
        n_permutations=m,
        level=1.0 - alpha,
    )


@dataclass(frozen=True)
class CombinedContamination:
    """Contamination evidence pooled across independent shards of a dataset.

    Attributes:
        p_value: Combined p-value.
        statistic: Fisher's combined statistic.
        n_shards: Shards combined.
        shard_p_values: The inputs, retained so a single driving shard is visible.
        level: One minus the significance level used for the verdict.
    """

    p_value: float
    statistic: float
    n_shards: int
    shard_p_values: npt.NDArray[Any]
    level: float

    @property
    def contaminated(self) -> bool:
        """Whether the pooled evidence rejects the null at the chosen level."""
        return self.p_value < (1.0 - self.level)

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        smallest = float(np.min(self.shard_p_values))
        return "\n".join(
            [
                f"pooled contamination evidence over {self.n_shards} shards",
                f"  Fisher statistic {self.statistic:.4f}, combined p = {self.p_value:.4g}",
                f"  smallest shard p-value: {smallest:.4g}",
                (
                    "  pooled evidence of contamination"
                    if self.contaminated
                    else "  no pooled evidence of contamination"
                ),
            ]
        )


def combine_shards(p_values: npt.ArrayLike, *, alpha: float = 0.05) -> CombinedContamination:
    """Pool per-shard contamination p-values with Fisher's method.

    A single permutation test over a large dataset is limited by how many shuffles you can
    afford to score. Splitting the dataset into disjoint shards, testing each, and pooling
    recovers power: the shards use different data, so under the null their p-values are
    independent and ``-2 Σ log p`` follows a chi-squared distribution on ``2k`` degrees of
    freedom.

    Args:
        p_values: One p-value per shard, from :func:`exchangeability_test`.
        alpha: Significance level for the verdict.

    Returns:
        A :class:`CombinedContamination`.

    Raises:
        ValueError: If any p-value lies outside (0, 1]. Exact zero is impossible from a
            permutation test and signals an upstream error rather than certainty.

    References:
        tests/test_contamination.py::test_fisher_combination_is_calibrated_under_the_null
        tests/test_contamination.py::test_pooling_shards_recovers_power
    """
    check_alpha(alpha)
    values = to_1d_array("p_values", np.asarray(p_values, dtype=float))
    if np.any(values <= 0.0) or np.any(values > 1.0):
        raise ValueError(
            "p_values must lie in (0, 1]; a permutation test cannot return exactly 0, so "
            "such a value indicates an upstream error"
        )

    statistic = float(-2.0 * np.sum(np.log(values)))
    combined = float(stats.chi2.sf(statistic, df=2 * values.shape[0]))
    return CombinedContamination(
        p_value=combined,
        statistic=statistic,
        n_shards=int(values.shape[0]),
        shard_p_values=values,
        level=1.0 - alpha,
    )
