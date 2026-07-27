# Methods

Derivations for the estimators in truescore. Each page states what the method assumes, what
it guarantees, what it does *not* guarantee, and which test enforces each claim.

| page | covers |
| --- | --- |
| [prediction-powered-inference.md](prediction-powered-inference.md) | Correcting a judge-scored metric with a small gold subset; the estimator, the variance-optimal λ, and the Rogan–Gladen alternative. |
| [confidence-sequences.md](confidence-sequences.md) | How fixed-sample intervals fail under continuous monitoring, the two constructions, and windowed detection. |
| [judge-bias-and-slices.md](judge-bias-and-slices.md) | HC3-robust bias regression, position bias, and the effect of one global correction on segment ordering. |
| [contamination.md](contamination.md) | The exact exchangeability permutation test, its calibration, and its blind spot. |

## Notation

| symbol | meaning |
| --- | --- |
| $N$ | examples carrying a judge label |
| $n$ | examples also carrying a gold (human) label |
| $f_i$ | the judge's label for example $i$ |
| $Y_i$ | the trusted label for example $i$, where one exists |
| $\theta$ | the quantity being estimated, e.g. the true pass rate |
| $\alpha$ | significance level; intervals have nominal coverage $1-\alpha$ |

## The conventions these pages share

**Gold labels are a random sample.** Every correction in truescore rests on this and
nothing repairs its violation. Labeling the examples that looked interesting, or the ones a
reviewer had time for, produces a biased correction that is harder to detect than no
correction at all. The `Estimate.assumptions` field repeats this text, so every estimate
carries it.

**Degenerate input raises rather than returning NaN.** Raising stops the run instead of
letting a NaN propagate into a report. Validation happens at the boundary and is
fuzz-tested across every public function
(`tests/test_properties.py::test_no_estimator_ever_returns_nan`).

**Coverage is measured by simulation.** Every interval estimator has a simulation that draws
from a known ground truth and checks that the nominal 95% interval covers about 95% of the
time. For the confidence sequence, coverage is checked over whole trajectories, matching the
guarantee it makes.

**Intervals are reported with their method.** `Estimate.method` and `Interval.method` record
which construction produced a number, so a stored report identifies the estimator behind it.
