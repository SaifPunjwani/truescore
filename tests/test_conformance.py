"""Agreement with the reference PPI implementation.

`ppi_py` is the implementation released alongside the papers this estimator comes from
(Angelopoulos, Bates, Fannjiang, Jordan, Zrnic; and PPI++ by Angelopoulos, Bates, Zrnic).
Two implementations written from the same algebra should return the same numbers, and
where they do not, one of them is wrong.

That is not a hypothetical here. This file first ran while `_optimal_lambda` estimated
Var(f) over the labeled subset alone, and the disagreement reached 5.5e-2 on the point
estimate at 50 gold labels, far too large to be arithmetic. The reference was right: judge
labels exist for every example, so estimating Var(f) from the labeled subset throws away
most of the data and leaves λ noisy in exactly the regime this library targets. Adopting
the pooled estimate cut the worst disagreement to 3.3e-3.

Two degrees-of-freedom conventions remained after that, and they are treated differently
on purpose.

The covariance inside λ divides by n in the reference and did not here. λ is a tuning
parameter rather than an estimand, so unbiasedness of that covariance buys nothing, while
agreeing to floating point removes a whole class of question about which implementation to
believe. This package adopted the reference's convention, and the point estimate now
matches to 1e-12, tight enough that any future change to λ selection fails this file.

The variance inside the interval is different. That one is an estimand, the unbiased
estimate is the conservative one, and coverage is the property this library will not
trade. So it keeps ddof=1, and its interval comes out 0.1% to 1.7% wider than the
reference across the scenarios below. The assertion is therefore directional rather than a
tolerance: never narrower, and not gratuitously wider. A tolerance would have permitted a
narrower interval, which is the one outcome that would matter.

Skipped when `ppi_py` is absent, so it is an optional check rather than a dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from truescore.correct import ppi_estimate

ppi_py = pytest.importorskip("ppi_py", reason="reference implementation not installed")

# n_gold, n_unlabeled, judge accuracy. Chosen to span the regimes this library serves:
# scarce labels against plentiful unlabeled data, an almost-perfect judge, a judge barely
# better than a coin, and a labeled set as large as the unlabeled one.
SCENARIOS = [
    (30, 2000, 0.80),
    (50, 450, 0.95),
    (100, 900, 0.90),
    (200, 1800, 0.75),
    (300, 4700, 0.85),
    (500, 500, 0.60),
]
# The point estimate is expected to agree exactly, so this is floating-point slack only.
TOLERANCE = 1e-12
# The interval is expected to be slightly wider, never narrower. See the module docstring.
MAX_RELATIVE_WIDTH = 1.05


def _draw(rng: np.random.Generator, n_gold: int, n_unlabeled: int, accuracy: float):
    """A judge that is right `accuracy` of the time, on binary gold labels."""
    truth = rng.uniform(0.3, 0.8)
    gold = (rng.random(n_gold) < truth).astype(float)
    gold_unlabeled = (rng.random(n_unlabeled) < truth).astype(float)
    judge = np.where(rng.random(n_gold) < accuracy, gold, 1.0 - gold)
    judge_unlabeled = np.where(
        rng.random(n_unlabeled) < accuracy, gold_unlabeled, 1.0 - gold_unlabeled
    )
    return gold, judge, judge_unlabeled


@pytest.mark.parametrize(("n_gold", "n_unlabeled", "accuracy"), SCENARIOS)
def test_matches_the_reference_implementation(
    n_gold: int, n_unlabeled: int, accuracy: float
) -> None:
    """The point estimate matches exactly; the interval is never narrower."""
    rng = np.random.default_rng(0)
    for _ in range(8):
        gold, judge, judge_unlabeled = _draw(rng, n_gold, n_unlabeled, accuracy)
        ours = ppi_estimate(
            np.concatenate([judge, judge_unlabeled]), gold, np.arange(n_gold), alpha=0.05
        )
        reference_point = float(
            np.atleast_1d(ppi_py.ppi_mean_pointestimate(gold, judge, judge_unlabeled))[0]
        )
        low, high = ppi_py.ppi_mean_ci(gold, judge, judge_unlabeled, alpha=0.05)
        reference_width = float(np.atleast_1d(high)[0]) - float(np.atleast_1d(low)[0])
        our_width = ours.high - ours.low

        assert ours.point == pytest.approx(reference_point, abs=TOLERANCE)
        assert our_width >= reference_width - TOLERANCE, "must never be narrower"
        assert our_width <= reference_width * MAX_RELATIVE_WIDTH


def test_lambda_uses_every_judge_label_not_only_the_labeled_ones() -> None:
    """The specific disagreement this file was written to catch.

    Judge labels exist for every example. Estimating Var(f) from the labeled subset alone
    leaves λ noisy exactly where gold labels are scarce, which is the case this library
    exists for. Constructed so the two estimates of Var(f) differ substantially: the
    labeled subset is drawn from the low-variance tail of the judge's output.
    """
    rng = np.random.default_rng(3)
    n_gold, n_unlabeled = 40, 4000
    gold = (rng.random(n_gold) < 0.9).astype(float)
    judge_labeled = gold.copy()
    judge_unlabeled = (rng.random(n_unlabeled) < 0.5).astype(float)

    ours = ppi_estimate(np.concatenate([judge_labeled, judge_unlabeled]), gold, np.arange(n_gold))
    reference = float(
        np.atleast_1d(ppi_py.ppi_mean_pointestimate(gold, judge_labeled, judge_unlabeled))[0]
    )

    assert ours.point == pytest.approx(reference, abs=TOLERANCE)


def test_a_useless_judge_lands_on_the_classical_estimate_in_both() -> None:
    """λ should collapse toward zero for both implementations when the judge is noise."""
    rng = np.random.default_rng(11)
    n_gold, n_unlabeled = 300, 3000
    gold = (rng.random(n_gold) < 0.6).astype(float)
    judge_labeled = (rng.random(n_gold) < 0.5).astype(float)
    judge_unlabeled = (rng.random(n_unlabeled) < 0.5).astype(float)

    ours = ppi_estimate(np.concatenate([judge_labeled, judge_unlabeled]), gold, np.arange(n_gold))
    reference = float(
        np.atleast_1d(ppi_py.ppi_mean_pointestimate(gold, judge_labeled, judge_unlabeled))[0]
    )

    assert ours.point == pytest.approx(reference, abs=TOLERANCE)
    assert ours.point == pytest.approx(float(gold.mean()), abs=0.05)
