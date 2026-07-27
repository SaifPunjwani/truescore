# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Saif Punjwani
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
            "- That the evaluation set represents production traffic; reweight with "
            "truescore.weighting if you know the production mix.",
            "- That the gold labels are correct; they are treated as ground truth by definition.",
            "- Anything about examples outside this evaluation set.",
            "",
        ]
        return "\n".join(lines)

    def to_html(self) -> str:
        """Render the report as a single self-contained HTML file.

        No external stylesheet, script or font, so it survives being emailed, dropped in
        a bucket, or attached to a launch review years from now. Everything user-supplied
        is escaped: a report is a document, not a template.

        References:
            tests/test_report.py::test_html_report_is_self_contained_and_escaped
        """
        outside = not self.naive.low <= self.corrected.point <= self.naive.high
        verdict_class = "bad" if outside else "good"
        verdict = (
            "The judge-only interval does not contain the corrected estimate."
            if outside
            else "The judge-only interval contains the corrected estimate."
        )

        rows = [
            ("Corrected (use this)", self.corrected, "headline"),
            ("Judge-only (conventional)", self.naive, ""),
            ("Human labels only", self.gold_only, ""),
        ]
        body = "\n".join(
            f'<tr class="{cls}"><td>{_escape(label)}</td>'
            f'<td class="n">{est.point:.4f}</td>'
            f'<td class="n">[{est.low:.4f}, {est.high:.4f}]</td>'
            f"<td>{_escape(est.method)}</td></tr>"
            for label, est, cls in rows
        )

        blocks = []
        for title, section in (
            ("Judge quality", self.agreement),
            ("Judge bias", self.bias),
            ("Comparison", self.comparison),
        ):
            if section is not None:
                blocks.append(f"<h2>{title}</h2><pre>{_escape(section.summary())}</pre>")

        assumptions = "".join(f"<li>{_escape(a)}</li>" for a in self.corrected.assumptions)

        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(self.system_name)} - evaluation report</title>
<style>{_HTML_STYLE}</style></head><body><div class="wrap">
<h1>{_escape(self.system_name)}</h1>
<p class="meta">{_escape(self.metric_name)} &middot; {self.n_total} examples,
{self.n_gold} human-labeled &middot; generated {_escape(self.created_utc)}</p>

<h2>Result</h2>
<table><tr><th>estimate</th><th>value</th><th>95% interval</th><th>method</th></tr>
{body}</table>
<p>The conventional judge-only number is
<strong class="{verdict_class}">{self.naive_error:+.4f}</strong> away from the corrected
estimate. {verdict}</p>

{"".join(blocks)}

<h2>Assumptions</h2>
<ul>{assumptions}</ul>

<h2>What this report does not establish</h2>
<ul>
<li>That the evaluation set represents production traffic; reweight with
truescore.weighting if you know the production mix.</li>
<li>That the human labels are correct; they are treated as ground truth by definition.</li>
<li>Anything about examples outside this evaluation set.</li>
</ul>

<footer>Generated by truescore. Every number here is computed from the supplied labels.</footer>
</div></body></html>"""

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


_HTML_STYLE = """
  :root { --ink:#14171a; --muted:#5b636b; --line:#e3e6e8; --bg:#fff; --code:#f6f7f8;
          --bad:#b42318; --good:#067647; }
  @media (prefers-color-scheme: dark) {
    :root { --ink:#e6e8ea; --muted:#9aa2aa; --line:#2a2f34; --bg:#101315; --code:#181c1f;
            --bad:#f97066; --good:#47cd89; } }
  body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 ui-sans-serif,
         -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  .wrap { max-width:720px; margin:0 auto; padding:40px 24px 64px; }
  h1 { font-size:26px; margin:0 0 4px; letter-spacing:-0.01em; }
  h2 { font-size:17px; margin:36px 0 10px; }
  .meta { color:var(--muted); font-size:13.5px; margin:0 0 28px; }
  table { border-collapse:collapse; width:100%; margin:0 0 14px; font-size:14px; }
  th,td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
  td.n { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  tr.headline td { font-weight:600; }
  .bad { color:var(--bad); } .good { color:var(--good); }
  pre { background:var(--code); border:1px solid var(--line); border-radius:7px;
        padding:12px 14px; overflow-x:auto; font-size:12.5px; line-height:1.5;
        font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  ul { padding-left:20px; } li { margin:4px 0; }
  .note { color:var(--muted); font-size:13px; }
  footer { margin-top:40px; padding-top:16px; border-top:1px solid var(--line);
           color:var(--muted); font-size:12.5px; }
"""


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
