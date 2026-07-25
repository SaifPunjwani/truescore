"""The defensible artifact.

A single object that records what was measured, on how many examples, with how many gold
labels, under which estimator, with which assumptions -- and, pointedly, what the naive
judge-only number would have said. In the library this is a convenience; in a regulated
setting it is the evidence that an evaluation was conducted competently.

The gap between ``naive`` and ``corrected`` is the headline: it is the size of the error
a team would have shipped on.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import numpy.typing as npt

from truescore._validation import check_alpha
from truescore.agreement import AgreementReport, judge_agreement
from truescore.bias import BiasReport, judge_error_regression
from truescore.compare import ComparisonResult
from truescore.correct import (
    Estimate,
    gold_only_estimate,
    judge_only_estimate,
    ppi_estimate,
)

__all__ = ["EvalReport", "build_report"]


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars and dataclasses to JSON-serializable values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _json_safe(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass(frozen=True)
class EvalReport:
    """A complete, auditable evaluation result.

    Attributes:
        metric_name: What was measured, e.g. "pass rate".
        system_name: What was measured on.
        n_total: Examples carrying a judge label.
        n_gold: Examples carrying a gold label.
        naive: The judge-only estimate -- what a conventional eval would have reported.
        corrected: The prediction-powered estimate; the number to act on.
        gold_only: The classical estimate from gold labels alone, for reference.
        agreement: Judge-versus-gold agreement, when computable.
        bias: Judge-error regression, when covariates were supplied.
        comparison: An accompanying system comparison, when supplied.
        created_utc: ISO-8601 timestamp.
    """

    metric_name: str
    system_name: str
    n_total: int
    n_gold: int
    naive: Estimate
    corrected: Estimate
    gold_only: Estimate
    agreement: AgreementReport | None
    bias: BiasReport | None
    comparison: ComparisonResult | None
    created_utc: str

    @property
    def naive_error(self) -> float:
        """Signed difference between the naive and corrected estimates."""
        return self.naive.point - self.corrected.point

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary of the whole report."""
        return {k: _json_safe(v) for k, v in dataclasses.asdict(self).items()}

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        """Render the report as markdown suitable for a PR comment or an audit file."""
        lines = [
            f"# Evaluation report: {self.system_name}",
            "",
            f"**Metric:** {self.metric_name}  ",
            f"**Examples:** {self.n_total} judge-labeled, {self.n_gold} human-labeled  ",
            f"**Generated:** {self.created_utc}",
            "",
            "## Result",
            "",
            "| estimate | value | 95% interval | method |",
            "| --- | --- | --- | --- |",
            f"| **Corrected (use this)** | **{self.corrected.point:.4f}** | "
            f"[{self.corrected.low:.4f}, {self.corrected.high:.4f}] | {self.corrected.method} |",
            f"| Judge-only (conventional) | {self.naive.point:.4f} | "
            f"[{self.naive.low:.4f}, {self.naive.high:.4f}] | {self.naive.method} |",
            f"| Human labels only | {self.gold_only.point:.4f} | "
            f"[{self.gold_only.low:.4f}, {self.gold_only.high:.4f}] | {self.gold_only.method} |",
            "",
            f"The conventional judge-only number is **{self.naive_error:+.4f}** away from the "
            "corrected estimate"
            + (
                ", and its interval does not contain the corrected estimate."
                if not self.naive.low <= self.corrected.point <= self.naive.high
                else "."
            ),
            "",
        ]

        if self.agreement is not None:
            lines += [
                "## Judge quality",
                "",
                "```",
                self.agreement.summary(),
                "```",
                "",
            ]
        if self.bias is not None:
            lines += [
                "## Judge bias",
                "",
                "```",
                self.bias.summary(),
                "```",
                "",
            ]
        if self.comparison is not None:
            lines += [
                "## Comparison",
                "",
                "```",
                self.comparison.summary(),
                "```",
                "",
            ]

        lines += ["## Assumptions", ""]
        lines += [f"- {assumption}" for assumption in self.corrected.assumptions]
        lines += [
            "",
            "## What this report does not establish",
            "",
            "- That the evaluation set represents production traffic.",
            "- That the gold labels are correct; they are treated as ground truth by definition.",
            "- Anything about examples outside this evaluation set.",
            "",
        ]
        return "\n".join(lines)

    def summary(self) -> str:
        """Compact human-readable summary."""
        return "\n".join(
            [
                f"{self.system_name} -- {self.metric_name}",
                f"  corrected:  {self.corrected}",
                f"  judge-only: {self.naive}  (off by {self.naive_error:+.4f})",
                f"  gold-only:  {self.gold_only}",
                f"  n={self.n_total} examples, {self.n_gold} human-labeled",
            ]
        )


def build_report(
    judge: npt.ArrayLike,
    gold: npt.ArrayLike,
    gold_index: npt.ArrayLike,
    *,
    metric_name: str = "pass rate",
    system_name: str = "system",
    alpha: float = 0.05,
    covariates: Mapping[str, npt.ArrayLike] | None = None,
    comparison: ComparisonResult | None = None,
    timestamp: str | None = None,
) -> EvalReport:
    """Assemble a full evaluation report from judge labels and a gold-labeled subset.

    Args:
        judge: Judge labels for all examples.
        gold: Gold labels for the labeled subset.
        gold_index: Positions in ``judge`` carrying a gold label.
        metric_name: What is being measured.
        system_name: What is being measured.
        alpha: Significance level for every interval in the report.
        covariates: Optional per-example covariates *on the gold subset*, used for the
            bias section.
        comparison: Optional comparison against another system, from
            :mod:`truescore.compare`.
        timestamp: ISO-8601 timestamp; defaults to now (UTC). Supply a fixed value for
            reproducible artifacts.

    Returns:
        The :class:`EvalReport`.

    References:
        tests/test_report.py::test_report_round_trips_through_json
        tests/test_report.py::test_report_markdown_contains_corrected_and_naive
    """
    check_alpha(alpha)
    judge_arr = np.asarray(judge)
    gold_arr = np.asarray(gold)
    index = np.asarray(gold_index)

    naive = judge_only_estimate(judge_arr, alpha=alpha)
    corrected = ppi_estimate(judge_arr, gold_arr, index, alpha=alpha)
    gold_view = gold_only_estimate(gold_arr, alpha=alpha, n_total=int(judge_arr.shape[0]))

    judge_on_gold = judge_arr[index]
    agreement: AgreementReport | None
    try:
        agreement = judge_agreement(judge_on_gold, gold_arr, alpha=alpha)
    except ValueError:
        # Non-binary labels, or gold labels of a single class: agreement statistics are
        # undefined, but the estimates above remain valid.
        agreement = None

    bias: BiasReport | None = None
    if covariates:
        bias = judge_error_regression(judge_on_gold, gold_arr, covariates, alpha=alpha)

    return EvalReport(
        metric_name=metric_name,
        system_name=system_name,
        n_total=int(judge_arr.shape[0]),
        n_gold=int(gold_arr.shape[0]),
        naive=naive,
        corrected=corrected,
        gold_only=gold_view,
        agreement=agreement,
        bias=bias,
        comparison=comparison,
        created_utc=timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
