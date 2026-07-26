"""Can I look at this number whenever I want?

A fixed-sample confidence interval is valid at *one* pre-specified sample size. Eval
dashboards are not used that way: teams watch a metric accumulate and react the moment it
moves. Every look is an implicit test, and the guarantee evaporates -- with enough looks,
a 95% interval excludes the true value almost surely, so "we caught a regression" is
frequently an artifact of having looked often enough.

A **confidence sequence** is an interval valid *uniformly over time*: with probability
1 - α, it contains the true mean at every sample size simultaneously. You may peek after
every example, stop whenever you like, and act on what you see. The price is width --
roughly a ``sqrt(log t)`` factor over a fixed-sample interval, which is the cost of not
having to decide in advance when to look.

Two constructions, both for observations bounded in a known range:

- ``"empirical_bernstein"`` (default) adapts to the observed variance, so a metric near 0
  or 1 -- the common case for pass rates -- gets a much tighter sequence.
- ``"hoeffding"`` uses a normal mixture with the worst-case sub-Gaussian constant. It is
  simpler and variance-agnostic, and therefore wider.

Validity holds for any fixed tuning parameter, so the choices below affect width only,
never coverage. That claim is checked by simulation rather than asserted; see
``tests/test_sequential.py::test_confidence_sequence_covers_uniformly_over_time``.

References:
    Howard, Ramdas, McAuliffe, Sekhon (2021), "Time-uniform, nonparametric, nonasymptotic
        confidence sequences", Annals of Statistics 49(2).
    Waudby-Smith & Ramdas (2024), "Estimating means of bounded random variables by
        betting", Journal of the Royal Statistical Society Series B 86(1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from truescore._validation import check_alpha, to_1d_array

__all__ = [
    "ConfidenceSequence",
    "confidence_sequence",
    "first_exclusion",
    "windowed_exclusion",
]

Method = Literal["empirical_bernstein", "hoeffding"]


@dataclass(frozen=True)
class ConfidenceSequence:
    """An interval valid at every sample size simultaneously.

    Attributes:
        method: Construction used.
        level: Nominal uniform coverage, ``1 - alpha``.
        n: Observations processed.
        mean: Running mean at the final observation.
        low: Lower bound at the final observation.
        high: Upper bound at the final observation.
        lower_bounds: Lower bound after each observation, length ``n``.
        upper_bounds: Upper bound after each observation, length ``n``.

    The bound arrays are monotone by construction: each step intersects the new interval
    with everything seen before, which is legitimate precisely because the guarantee is
    uniform over time.
    """

    method: str
    level: float
    n: int
    mean: float
    low: float
    high: float
    lower_bounds: npt.NDArray[Any]
    upper_bounds: npt.NDArray[Any]

    @property
    def half_width(self) -> float:
        """Half the width of the final interval."""
        return (self.high - self.low) / 2.0

    def contains(self, value: float) -> bool:
        """Whether the final interval contains ``value``."""
        return bool(self.low <= value <= self.high)

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        return "\n".join(
            [
                f"confidence sequence ({self.method}) after n={self.n} observations",
                f"  mean {self.mean:.4f}, interval [{self.low:.4f}, {self.high:.4f}] "
                f"at {self.level:.0%} uniform coverage",
                "  valid at every sample size: peek and stop whenever you like",
            ]
        )


def _rescale(values: npt.NDArray[Any], bounds: tuple[float, float]) -> npt.NDArray[Any]:
    lower, upper = bounds
    if not upper > lower:
        raise ValueError(f"bounds must satisfy upper > lower; got {bounds}")
    if float(values.min()) < lower or float(values.max()) > upper:
        raise ValueError(
            f"values must lie inside bounds {bounds}; got range "
            f"[{values.min():.6g}, {values.max():.6g}]"
        )
    scaled: npt.NDArray[Any] = (values - lower) / (upper - lower)
    return scaled


def _empirical_bernstein_bounds(
    x: npt.NDArray[Any], alpha: float, lambda_cap: float, one_sided: bool
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Predictable-plug-in empirical-Bernstein bounds for observations in [0, 1].

    Follows Waudby-Smith & Ramdas: the interval at time ``t`` is the set of means ``m``
    with ``|Σ λ_i (X_i − m)| < log(2/α) + Σ v_i ψ_E(λ_i)``, where ``λ_i`` is predictable
    (it may depend only on ``X_1..X_{i-1}``), ``v_i = 4(X_i − μ̂_{i−1})²`` and
    ``ψ_E(λ) = (−log(1 − λ) − λ)/4``. Solving the inequality for ``m`` gives the bounds
    returned here.

    A monitor that only asks "has the rate fallen?" spends its budget on one tail, so
    ``one_sided`` replaces ``log(2/α)`` with ``log(1/α)``. Both bounds are still returned;
    only the one being tested retains the guarantee.
    """
    n = x.shape[0]
    index = np.arange(1, n + 1, dtype=float)

    # Running mean and variance with a 1/2 and 1/4 prior, so that time 1 is well defined.
    running_mean = (0.5 + np.cumsum(x)) / (index + 1.0)
    mean_previous = np.concatenate([[0.5], running_mean[:-1]])
    running_var = (0.25 + np.cumsum((x - running_mean) ** 2)) / (index + 1.0)
    var_previous = np.concatenate([[0.25], running_var[:-1]])

    # λ_i must be predictable, hence the shifted variance estimate. The cap keeps
    # log(1 − λ) finite; any fixed cap below 1 preserves validity.
    lam = np.sqrt(2.0 * np.log(1.0 / alpha) / (var_previous * index * np.log1p(index)))
    lam = np.minimum(lam, lambda_cap)

    psi = (-np.log1p(-lam) - lam) / 4.0
    v = 4.0 * (x - mean_previous) ** 2

    weighted_sum = np.cumsum(lam * x)
    weight_total = np.cumsum(lam)
    tail_budget = np.log(1.0 / alpha) if one_sided else np.log(2.0 / alpha)
    budget = tail_budget + np.cumsum(v * psi)

    centre = weighted_sum / weight_total
    margin = budget / weight_total
    return centre - margin, centre + margin


def _hoeffding_bounds(
    x: npt.NDArray[Any], alpha: float, target_n: int
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Normal-mixture (Robbins) bounds using the worst-case sub-Gaussian constant.

    For observations in [0, 1] the Hoeffding sub-Gaussian parameter is ``σ = 1/2``, so the
    intrinsic time is ``V_t = t/4``. The mixture boundary
    ``sqrt(2 (V + ρ) log(sqrt((V + ρ)/ρ) / α))`` is time-uniform for any fixed ``ρ > 0``;
    ``ρ`` only tunes *where* the sequence is tightest, which is why ``target_n`` is a
    planning input rather than something read off the data.
    """
    n = x.shape[0]
    index = np.arange(1, n + 1, dtype=float)
    variance_proxy = 0.25
    intrinsic_time = index * variance_proxy
    rho = max(variance_proxy * target_n / (2.0 * np.log(1.0 / alpha)), 1e-12)

    shifted = intrinsic_time + rho
    boundary = np.sqrt(2.0 * shifted * np.log(np.sqrt(shifted / rho) / alpha))
    margin = boundary / index

    running_mean = np.cumsum(x) / index
    return running_mean - margin, running_mean + margin


def confidence_sequence(
    values: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    method: Method = "empirical_bernstein",
    bounds: tuple[float, float] = (0.0, 1.0),
    target_n: int = 1000,
    lambda_cap: float = 0.5,
    one_sided: bool = False,
) -> ConfidenceSequence:
    """Build a confidence sequence: an interval valid at every sample size at once.

    Args:
        values: Per-observation metric values in arrival order, inside ``bounds``. For a
            pass/fail metric these are 0/1; for a graded score, the score.
        alpha: One minus the uniform coverage level.
        method: ``"empirical_bernstein"`` adapts to observed variance and is tighter for
            metrics near 0 or 1; ``"hoeffding"`` is variance-agnostic and wider.
        bounds: Known range of the metric. Values are rescaled internally and the returned
            interval is expressed on the original scale.
        target_n: Sample size at which the Hoeffding sequence should be tightest. A
            planning input; it must not be chosen after seeing the data, and it has no
            effect on validity. Ignored by the empirical-Bernstein construction.
        lambda_cap: Cap on the betting fraction of the empirical-Bernstein construction.
            Any fixed value in (0, 1) is valid; larger is more aggressive early on.
        one_sided: Spend the whole error budget on one tail. Set this only when the
            decision genuinely looks in one direction -- a regression alarm, say -- since
            the untested bound then carries no guarantee. :func:`first_exclusion` sets it
            automatically from ``direction``.

    Returns:
        A :class:`ConfidenceSequence` whose bound arrays are the running intersection of
        every interval so far.

    Raises:
        ValueError: If ``values`` is empty, falls outside ``bounds``, ``bounds`` is
            degenerate, ``alpha`` is outside (0, 1), or ``lambda_cap`` is outside (0, 1).

    References:
        tests/test_sequential.py::test_confidence_sequence_covers_uniformly_over_time
        tests/test_sequential.py::test_fixed_sample_intervals_fail_under_repeated_peeking
    """
    check_alpha(alpha)
    if not 0.0 < lambda_cap < 1.0:
        raise ValueError(f"lambda_cap must lie in (0, 1); got {lambda_cap}")
    if target_n < 1:
        raise ValueError(f"target_n must be at least 1; got {target_n}")

    raw = to_1d_array("values", np.asarray(values, dtype=float))
    if not np.all(np.isfinite(raw)):
        raise ValueError("values contains non-finite entries")
    x = _rescale(raw, bounds)

    if method == "empirical_bernstein":
        low, high = _empirical_bernstein_bounds(x, alpha, lambda_cap, one_sided)
    elif method == "hoeffding":
        low, high = _hoeffding_bounds(x, alpha, target_n)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown method {method!r}")

    # The guarantee is uniform over time, so intersecting with every earlier interval is
    # free: it can only tighten a sequence that was already valid at each step.
    low = np.maximum.accumulate(np.clip(low, 0.0, 1.0))
    high = np.minimum.accumulate(np.clip(high, 0.0, 1.0))

    lower_bound, upper_bound = bounds
    span = upper_bound - lower_bound
    low_scaled = lower_bound + low * span
    high_scaled = lower_bound + high * span

    return ConfidenceSequence(
        method=method,
        level=1.0 - alpha,
        n=int(raw.shape[0]),
        mean=float(raw.mean()),
        low=float(low_scaled[-1]),
        high=float(high_scaled[-1]),
        lower_bounds=low_scaled,
        upper_bounds=high_scaled,
    )


def first_exclusion(
    values: npt.ArrayLike,
    reference: float,
    *,
    alpha: float = 0.05,
    direction: Literal["two-sided", "below", "above"] = "two-sided",
    method: Method = "empirical_bernstein",
    bounds: tuple[float, float] = (0.0, 1.0),
    target_n: int = 1000,
) -> int | None:
    """First observation at which the sequence rules out ``reference``.

    This is a valid stopping rule: because the sequence is uniform over time, acting the
    moment it excludes the reference preserves the error guarantee. It is the honest
    version of "watch the dashboard and react", and the practical entry point for
    regression monitoring -- see :mod:`truescore.drift`.

    Args:
        values: Per-observation metric values in arrival order.
        reference: The value being ruled out, e.g. last release's pass rate.
        alpha: One minus the uniform coverage level.
        direction: ``"below"`` reacts only to evidence the true mean is below the
            reference (the usual regression alarm), ``"above"`` only to evidence it is
            higher, ``"two-sided"`` to either.
        method: Construction, as in :func:`confidence_sequence`.
        bounds: Known range of the metric.
        target_n: Planning input for the Hoeffding construction.

    Returns:
        The 1-based observation count at which the reference was first excluded, or
        ``None`` if it never was.

    References:
        tests/test_sequential.py::test_first_exclusion_detects_a_real_regression
        tests/test_sequential.py::test_first_exclusion_false_alarm_rate_is_controlled
    """
    sequence = confidence_sequence(
        values,
        alpha=alpha,
        method=method,
        bounds=bounds,
        target_n=target_n,
        one_sided=direction != "two-sided",
    )
    if direction == "below":
        excluded = sequence.upper_bounds < reference
    elif direction == "above":
        excluded = sequence.lower_bounds > reference
    else:
        excluded = (sequence.upper_bounds < reference) | (sequence.lower_bounds > reference)

    hits = np.flatnonzero(excluded)
    return int(hits[0]) + 1 if hits.size else None


def windowed_exclusion(
    values: npt.ArrayLike,
    reference: float,
    *,
    window: int,
    alpha: float = 0.05,
    direction: Literal["two-sided", "below", "above"] = "below",
    method: Method = "empirical_bernstein",
    bounds: tuple[float, float] = (0.0, 1.0),
    planned_windows: int | None = None,
) -> int | None:
    """Detect a *recent* departure from ``reference``, rather than a cumulative one.

    :func:`first_exclusion` asks whether the mean of everything observed so far differs
    from the reference. That is the wrong question for a live monitor: a long healthy
    prefix keeps the cumulative mean near the baseline for a long time after a regression
    starts, so a genuine drop can go undetected for thousands of observations.

    This function asks the operational question instead -- has the *recent* rate departed?
    -- by running an independent confidence sequence over consecutive non-overlapping
    windows and splitting the error budget across them. Each window is monitored
    anytime-validly, so an alarm can fire part-way through one; the Bonferroni split keeps
    the false-alarm probability for the whole run at ``alpha``.

    The window length is the tradeoff: longer windows detect smaller departures but take
    longer to react, and consume more of the budget per window.

    Args:
        values: Per-observation metric values in arrival order.
        reference: The rate being defended.
        window: Observations per window. Must be at least 2.
        alpha: Total false-alarm budget for the entire monitoring run.
        direction: ``"below"`` for a regression alarm; see :func:`first_exclusion`.
        method: Construction, as in :func:`confidence_sequence`.
        bounds: Known range of the metric.
        planned_windows: Number of windows the budget is split across. Defaults to the
            number of complete windows in ``values``. For a live monitor with no fixed
            end, pre-specify the horizon here rather than letting it grow with the data.

    Returns:
        The 1-based observation index at which a departure was first established, or
        ``None`` if none was.

    Raises:
        ValueError: If ``window`` is below 2 or ``planned_windows`` is below 1.

    References:
        tests/test_sequential.py::test_windowed_exclusion_detects_a_late_regression
        tests/test_sequential.py::test_windowed_exclusion_false_alarm_rate_is_controlled
    """
    check_alpha(alpha)
    if window < 2:
        raise ValueError(f"window must be at least 2; got {window}")
    series = to_1d_array("values", np.asarray(values, dtype=float))

    n_windows = series.shape[0] // window
    if n_windows == 0:
        return None
    budget_windows = planned_windows if planned_windows is not None else n_windows
    if budget_windows < 1:
        raise ValueError(f"planned_windows must be at least 1; got {budget_windows}")

    per_window_alpha = alpha / budget_windows
    for w in range(n_windows):
        start = w * window
        chunk = series[start : start + window]
        hit = first_exclusion(
            chunk,
            reference,
            alpha=per_window_alpha,
            direction=direction,
            method=method,
            bounds=bounds,
        )
        if hit is not None:
            return start + hit
    return None
