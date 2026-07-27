"""Point this at your evaluation file and it will tell you what it can do.

The hardest part of adopting a statistics library is not the statistics. It is the fifteen
minutes at the start where somebody has a CSV, does not know which of their columns the
tool wants, does not know whether they have enough human labels for any of it, and gives
up. This module removes that step: it reads the file, works out what each column is, and
reports three things.

1. **What can be computed right now**, with the exact command to do it.
2. **What is blocked, and by what** -- almost always missing human labels, and it says how
   many would unblock it rather than leaving that as an exercise.
3. **What the judge appears to be biased by**, scanned across every numeric column at once
   with a multiplicity correction, because scanning twenty columns and reporting the
   smallest p-value is how teams talk themselves into imaginary findings.

Detection is heuristic and says so. It is a starting point that a human confirms, not an
authority: a column of zeros and ones might be a judge verdict or might be
`customer_is_enterprise`, and nothing in the data distinguishes them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from truescore.bias import judge_error_regression
from truescore.compare import holm
from truescore.io import get_field, read_rows
from truescore.power import required_gold_labels

__all__ = ["ColumnProfile", "Diagnosis", "diagnose"]

_MISSING = frozenset({"", "na", "n/a", "nan", "none", "null", "-", "unlabeled"})
_TRUE = frozenset({"1", "true", "t", "yes", "y", "pass", "passed", "correct", "good"})
_FALSE = frozenset({"0", "false", "f", "no", "n", "fail", "failed", "incorrect", "bad"})


@dataclass(frozen=True)
class ColumnProfile:
    """What one column of an evaluation file appears to be.

    Attributes:
        name: Column name.
        kind: One of ``verdict``, ``sparse_verdict``, ``numeric``, ``categorical``,
            ``graded``, ``sparse_graded``, ``identifier`` or ``unusable``.
        n_present: Rows carrying a value.
        n_missing: Rows where the value is blank.
        n_distinct: Distinct values observed.
        detail: A sentence a human can check the guess against.
    """

    name: str
    kind: str
    n_present: int
    n_missing: int
    n_distinct: int
    detail: str

    @property
    def coverage(self) -> float:
        """Fraction of rows carrying a value."""
        total = self.n_present + self.n_missing
        return self.n_present / total if total else 0.0


@dataclass(frozen=True)
class Diagnosis:
    """What this file supports, what it does not, and what would change that."""

    path: str
    n_rows: int
    columns: tuple[ColumnProfile, ...]
    judge_candidates: tuple[str, ...]
    gold_candidates: tuple[str, ...]
    covariate_candidates: tuple[str, ...]
    slice_candidates: tuple[str, ...]
    available: tuple[str, ...]
    blocked: tuple[tuple[str, str], ...]
    bias_findings: tuple[str, ...]
    recommendations: tuple[str, ...]

    def summary(self) -> str:
        """Human-readable multi-line report."""
        # Dotted paths from nested JSON can be long, so size the column to the data.
        width = max([len(c.name) for c in self.columns] + [len("column")]) + 2
        lines = [
            f"{self.path}: {self.n_rows} rows, {len(self.columns)} columns",
            "",
            f"  {'column':<{width}}{'kind':<18}{'coverage':>10}  detail",
        ]
        for column in self.columns:
            lines.append(
                f"  {column.name:<{width}}{column.kind:<18}{column.coverage:>9.0%}  {column.detail}"
            )

        lines += ["", "what this file supports today:"]
        lines += [f"  - {item}" for item in self.available] or ["  - nothing yet"]

        if self.blocked:
            lines += ["", "what it does not, and why:"]
            lines += [f"  - {what}: {why}" for what, why in self.blocked]

        if self.bias_findings:
            lines += ["", "what the judge appears to be biased by:"]
            lines += [f"  - {finding}" for finding in self.bias_findings]

        if self.recommendations:
            lines += ["", "suggested next steps:"]
            lines += [f"  {i}. {step}" for i, step in enumerate(self.recommendations, start=1)]
        return "\n".join(lines)


def _flat_paths(row: Mapping[str, Any], prefix: str = "", depth: int = 3) -> list[str]:
    """Leaf paths of a row, so nested JSON from an eval harness profiles like a table."""
    out: list[str] = []
    for key, value in row.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and depth > 1:
            out.extend(_flat_paths(value, f"{path}.", depth - 1))
        else:
            out.append(path)
    return out


def _classify(name: str, raw: list[Any]) -> ColumnProfile:
    """Guess what a column is from its values alone."""
    present = [v for v in raw if str(v).strip().lower() not in _MISSING and v is not None]
    n_missing = len(raw) - len(present)
    tokens = [str(v).strip().lower() for v in present]
    distinct = sorted(set(tokens))

    def profile(kind: str, detail: str) -> ColumnProfile:
        return ColumnProfile(name, kind, len(present), n_missing, len(distinct), detail)

    if not present:
        return profile("unusable", "every row is blank")

    verdict_like = set(distinct) <= (_TRUE | _FALSE)
    if verdict_like and len(distinct) <= 2:
        if n_missing > 0:
            return profile(
                "sparse_verdict",
                f"pass/fail on {len(present)} of {len(raw)} rows -- looks like human labels",
            )
        return profile("verdict", "pass/fail on every row -- looks like a judge column")

    numeric: list[float] = []
    for token in tokens:
        try:
            numeric.append(float(token))
        except ValueError:
            numeric = []
            break
    if numeric:
        if len(distinct) <= 2 and set(numeric) <= {0.0, 1.0}:
            kind = "sparse_verdict" if n_missing else "verdict"
            return profile(kind, "0/1 values")
        low, high = min(numeric), max(numeric)
        # A handful of small whole numbers is a rubric score, not a covariate. Teams grade
        # on 1-5 at least as often as pass/fail, and the graded metrics differ.
        whole = all(float(v).is_integer() for v in numeric)
        if whole and 2 < len(distinct) <= 10 and low >= 0 and high <= 10:
            kind = "sparse_graded" if n_missing else "graded"
            return profile(
                kind,
                f"{len(distinct)} levels from {low:g} to {high:g} -- looks like a rubric score",
            )
        return profile("numeric", f"range {low:g} to {high:g} -- usable as a bias covariate")

    # Near-uniqueness, by ratio rather than an absolute count, so a six-row sample is
    # classified the same way a six-thousand-row one would be.
    if len(present) >= 4 and len(distinct) / len(present) > 0.9:
        return profile("identifier", "nearly unique per row -- looks like an id")
    return profile(
        "categorical", f"{len(distinct)} values ({', '.join(distinct[:4])}...) -- sliceable"
    )


def _to_binary(values: Sequence[Any]) -> npt.NDArray[Any]:
    out = []
    for value in values:
        token = str(value).strip().lower()
        out.append(1.0 if token in _TRUE or token == "1.0" else 0.0)
    return np.asarray(out, dtype=float)


def _scan_bias(
    judge: npt.NDArray[Any],
    gold: npt.NDArray[Any],
    covariates: dict[str, npt.NDArray[Any]],
) -> tuple[str, ...]:
    """Regress judge error on each covariate separately, adjusting across the scan.

    Reported one at a time rather than jointly because the question here is a screen --
    "is anything worth looking at?" -- and because a joint fit on twenty columns of unknown
    quality is a worse first move than twenty simple fits with the p-values corrected.
    """
    if not covariates:
        return ()
    names: list[str] = []
    raw_p: list[float] = []
    slopes: list[float] = []
    for name, column in covariates.items():
        try:
            fitted = judge_error_regression(judge, gold, {name: column}).effects[0]
        except ValueError:
            continue
        names.append(name)
        raw_p.append(fitted.p_value)
        slopes.append(fitted.effect)
    if not names:
        return ()

    adjusted = holm(np.asarray(raw_p))
    findings: list[str] = []
    for name, slope, p_adj in zip(names, slopes, adjusted, strict=True):
        if p_adj < 0.05:
            direction = "more generous" if slope > 0 else "harsher"
            findings.append(
                f"{name}: the judge gets {direction} as it rises "
                f"({slope:+.5f} per unit, adjusted p={p_adj:.3g})"
            )
    if not findings:
        findings.append(
            f"nothing detectable across {len(names)} column(s) after correcting for "
            "having scanned them all"
        )
    return tuple(findings)


def diagnose(path: str | Path, *, alpha: float = 0.05) -> Diagnosis:
    """Profile an evaluation file and report what it supports.

    Args:
        path: CSV or JSON Lines file of evaluation results.
        alpha: Significance level used by the bias scan.

    Returns:
        A :class:`Diagnosis`.

    Raises:
        ValueError: If the file cannot be read or has no rows.

    References:
        tests/test_doctor.py::test_diagnose_finds_judge_and_gold_columns
        tests/test_doctor.py::test_diagnose_says_what_is_blocked_without_human_labels
    """
    rows = read_rows(path)
    names = _flat_paths(rows[0])
    columns = tuple(_classify(name, [get_field(row, name) for row in rows]) for name in names)

    judges = tuple(c.name for c in columns if c.kind == "verdict")
    golds = tuple(c.name for c in columns if c.kind == "sparse_verdict")
    graded = tuple(c.name for c in columns if c.kind == "graded")
    graded_gold = tuple(c.name for c in columns if c.kind == "sparse_graded")
    numerics = tuple(c.name for c in columns if c.kind == "numeric")
    categoricals = tuple(c.name for c in columns if c.kind == "categorical")

    available: list[str] = []
    blocked: list[tuple[str, str]] = []
    recommendations: list[str] = []
    bias_findings: tuple[str, ...] = ()

    stem = Path(path).name
    if judges:
        judge_name = judges[0]
        available.append(
            f"monitor a release for regressions, no human labels needed "
            f"(truescore monitor {stem} --metric {judge_name} --baseline <rate> --window 300)"
        )
        if len(judges) >= 2:
            available.append(
                f"compare {judges[0]} against {judges[1]} as judged "
                f"(truescore compare {stem} --judge-a {judges[0]} --judge-b {judges[1]})"
            )
    elif not graded:
        blocked.append(("everything", "no column looks like a judge verdict or a rubric score"))

    if judges and golds:
        judge_values = _to_binary([get_field(row, judges[0]) for row in rows])
        gold_positions = [
            i
            for i, row in enumerate(rows)
            if str(get_field(row, golds[0])).strip().lower() not in _MISSING
        ]
        gold_values = _to_binary([get_field(rows[i], golds[0]) for i in gold_positions])
        n_gold = len(gold_positions)

        available.append(
            f"correct the score for judge error "
            f"(truescore audit {stem} --judge {judges[0]} --gold {golds[0]})"
        )
        available.append(f"measure judge quality against those {n_gold} human labels")
        if categoricals:
            available.append(
                f"find a segment that regressed "
                f"(truescore slices {stem} --by {categoricals[0]} ...)"
            )

        covariates = {
            name: np.asarray(
                [float(get_field(rows[i], name) or 0.0) for i in gold_positions], dtype=float
            )
            for name in numerics
        }
        bias_findings = _scan_bias(judge_values[gold_positions], gold_values, covariates)

        if n_gold < 100:
            plan = required_gold_labels(
                len(rows), target_half_width=0.03, sensitivity=0.9, specificity=0.8
            )
            recommendations.append(
                f"you have {n_gold} human labels; roughly {plan.required_gold} would give "
                "a +/-0.03 interval at a typical judge quality"
            )
    elif judges:
        plan = required_gold_labels(
            len(rows), target_half_width=0.03, sensitivity=0.9, specificity=0.8
        )
        blocked.append(
            (
                "correcting the score, measuring judge quality, per-segment analysis",
                f"no human labels. About {plan.required_gold} of these {len(rows)} rows "
                "labelled by a person would unblock all three",
            )
        )
        recommendations.append(
            "label a random sample -- random, not the interesting ones, or the correction "
            "inherits your selection"
        )
        recommendations.append(
            "check whether you already have implicit human verdicts: escalations, "
            "thumbs-down, refunds, whether the customer came back"
        )

    if len(judges) >= 2 and not golds:
        recommendations.append(
            f"{len(judges)} columns look like verdicts and none is sparse. If one of them "
            "is human labels rather than a second system, pass it as --gold: nothing in "
            "the data distinguishes them"
        )

    if graded:
        if graded_gold:
            available.append(
                f"correct the mean {graded[0]} rating against {graded_gold[0]} "
                "(truescore.correct.ppi_estimate) and measure judge quality with "
                "quadratic-weighted kappa (truescore.agreement.graded_agreement)"
            )
        else:
            blocked.append(
                (
                    f"correcting the mean {graded[0]} rating and measuring judge quality",
                    "no human rubric scores. Label a random sample of the same rating "
                    "scale to unblock both",
                )
            )
            recommendations.append(
                f"{graded[0]} is a rubric score; label a random sample on the same scale "
                "to correct the mean rating and measure quadratic-weighted kappa"
            )

    if judges and not categoricals:
        recommendations.append(
            "add a column for segment, language or customer tier: the overall number can "
            "hide a segment moving the other way"
        )

    return Diagnosis(
        path=str(path),
        n_rows=len(rows),
        columns=columns,
        judge_candidates=judges,
        gold_candidates=golds,
        covariate_candidates=numerics,
        slice_candidates=categoricals,
        available=tuple(available),
        blocked=tuple(blocked),
        bias_findings=bias_findings,
        recommendations=tuple(recommendations),
    )
