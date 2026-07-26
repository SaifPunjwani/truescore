# Confidence sequences

The methods behind `truescore.sequential`, and why a 95% confidence interval stops being
95% the moment you look at it twice.

## The failure

A confidence interval is a promise about a procedure: *if* you fix $n$ in advance, collect
$n$ observations, and compute the interval once, it covers the truth 95% of the time.
Evaluation dashboards violate the premise in the first sentence. Data arrives continuously,
somebody watches, and the interval is inspected after every batch. Each look is a test, and
the tests are not independent — but they are numerous.

The damage is not subtle. Simulating 200 healthy streams of 300 observations at a true rate
of 0.30, and checking a 95% Wilson interval after every single observation:

| approach | streams where the interval ever excluded the truth |
| --- | --- |
| fixed-sample interval, checked continuously | **47.7%** |
| truescore confidence sequence | 0.3% |

Measured by `tests/test_sequential.py::test_fixed_sample_intervals_fail_under_repeated_peeking`.
Nearly half of healthy streams produce, at some point, an interval that excludes the true
rate — which an operator reads as a regression.

## The fix

A **confidence sequence** is a sequence of intervals $(C_t)_{t \ge 1}$ satisfying

$$\mathbb{P}\bigl(\forall t \ge 1:\; \theta \in C_t\bigr) \;\ge\; 1 - \alpha$$

Note where the quantifier sits: *inside* the probability. One error budget covers the whole
trajectory, so you may check after every observation, stop at a data-dependent time, and
act — all without inflating the error rate. The cost is width, roughly a $\sqrt{\log t}$
factor, which is the price of not having to name your sample size in advance.

Coverage is asserted over trajectories rather than time points in
`tests/test_sequential.py::test_confidence_sequence_covers_uniformly_over_time`: a stream
counts as a failure if the interval was wrong at *any* $t$.

## Construction 1: empirical-Bernstein (default)

For observations in $[0,1]$, following Waudby-Smith & Ramdas. Let $\hat\mu_t$ and
$\hat\sigma^2_t$ be running estimates with a $\tfrac12$ and $\tfrac14$ prior so that $t=1$
is defined, and let

$$\lambda_t = \min\left(c,\; \sqrt{\frac{2\log(1/\alpha)}{\hat\sigma^2_{t-1}\, t \log(1+t)}}\right)$$

$\lambda_t$ is **predictable** — it uses $\hat\sigma^2_{t-1}$, not $\hat\sigma^2_t$ — which
is what makes the underlying process a supermartingale and the guarantee time-uniform. With
$\psi_E(\lambda) = \tfrac14\bigl(-\log(1-\lambda) - \lambda\bigr)$ and
$v_i = 4(X_i - \hat\mu_{i-1})^2$, the interval at time $t$ is every $m$ satisfying

$$\left|\sum_{i \le t} \lambda_i (X_i - m)\right| \;<\; \log(2/\alpha) + \sum_{i \le t} v_i \psi_E(\lambda_i)$$

which solves to a closed-form interval centred at $\sum \lambda_i X_i / \sum \lambda_i$.
Because the bound adapts to the observed variance, a pass rate near 0 or 1 — the common
case — gets a far tighter sequence than a worst-case bound would give
(`tests/test_sequential.py::test_empirical_bernstein_beats_hoeffding_on_a_low_variance_stream`).

**One-sided budgeting.** A regression monitor only ever asks whether the rate *fell*.
Spending the whole budget on one tail replaces $\log(2/\alpha)$ with $\log(1/\alpha)$, which
is a real power gain; `first_exclusion` sets it automatically from `direction`. The untested
bound then carries no guarantee, which is why it is not the default.

## Construction 2: normal mixture (Hoeffding)

Variance-agnostic, using the sub-Gaussian parameter $\sigma = \tfrac12$ that any $[0,1]$
variable admits. With intrinsic time $V_t = t\sigma^2$ and any fixed $\rho > 0$,

$$\left|\hat\mu_t - \theta\right| \;<\; \frac{1}{t}\sqrt{2(V_t + \rho)\,\log\!\left(\frac{\sqrt{(V_t+\rho)/\rho}}{\alpha}\right)}$$

$\rho$ tunes *where* the sequence is tightest, not whether it is valid — which is why
`target_n` is documented as a planning input and must not be read off the data. Simpler and
wider than the empirical-Bernstein construction; useful as an independent check.

## Cumulative is the wrong question

`first_exclusion` asks whether the mean of *everything so far* differs from a reference.
That is the wrong question for a live monitor, and the first end-to-end example exposed it:
on a stream that ran healthy at 0.88 for 600 requests and then regressed to 0.78, the
cumulative sequence never ruled out 0.88, because the healthy prefix keeps the running mean
up long after the service has degraded.

`windowed_exclusion` asks the operational question instead — *has the recent rate departed?*
— by running an independent sequence over consecutive non-overlapping windows and splitting
$\alpha$ across them. Each window is monitored anytime-validly, so an alarm can fire
part-way through one; the Bonferroni split keeps the whole run at $\alpha$. On the same
stream it fires at request 867, 267 requests after the regression began.

The window length is the tradeoff: longer windows detect smaller departures but react more
slowly and consume more budget each. Both properties are tested —
`test_windowed_exclusion_detects_a_late_regression` and
`test_windowed_exclusion_false_alarm_rate_is_controlled`.

## What this does not give you

- **Instant detection.** Anytime validity costs data. A ten-point drop takes a few hundred
  observations to establish. If you need faster reaction, you are asking for a lower
  confidence level, and should say so explicitly rather than by peeking at a 95% interval.
- **A changepoint estimate.** The windowed monitor tells you a departure is established, not
  precisely when it began.
- **Protection against a mis-specified baseline.** If the reference rate was itself measured
  badly, the monitor faithfully defends a wrong number.

## References

- Howard, Ramdas, McAuliffe, Sekhon (2021), "Time-uniform, nonparametric, nonasymptotic
  confidence sequences", *Annals of Statistics* 49(2).
- Waudby-Smith & Ramdas (2024), "Estimating means of bounded random variables by betting",
  *JRSS-B* 86(1).
