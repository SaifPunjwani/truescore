# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Saif Punjwani
"""Reading the files evaluation tools actually write.

The estimators in truescore want one row per example with a judge verdict on it. Eval
tools write something else: a wrapper object with the records nested inside, scores keyed
by scorer name, metrics as a list of objects, verdicts spelled ``"C"`` or ``pass`` or
``success``. Asking people to reshape that first is the difference between trying a tool
and not.

This module reads the output of four eval tools directly and normalizes it to flat rows.
Each shape was taken from the tool's own serialization code, not inferred from a sample
file, and each is pinned by a fixture test:

============================  ============================================================
tool                          what it writes
============================  ============================================================
``inspect``                   ``EvalLog`` JSON: records at ``samples``, verdicts at
                              ``samples[].scores.<scorer>.value``
``promptfoo``                 ``--output out.json``: records at ``results.results``,
                              verdicts at ``success`` and ``gradingResult.pass``
``deepeval``                  ``.deepeval/.latest_test_run.json``: records at
                              ``testCases``, per-metric verdicts in ``metricsData``
``lm-eval``                   ``samples_<task>_<ts>.jsonl``: one record per line, metric
                              values merged in as top-level keys
============================  ============================================================

Anything unrecognized is returned as ``generic`` with its rows untouched, so an unknown
format degrades to the behaviour that existed before this module rather than failing.

Normalization never invents a verdict. Where a tool's own code defines a mapping, that
mapping is reproduced exactly and cited; where it does not, the value is passed through
and left for :func:`truescore.load_labels` to parse or reject.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from truescore.io import read_rows

__all__ = ["SUPPORTED_TOOLS", "EvalFormat", "detect_format", "read_eval"]

SUPPORTED_TOOLS = ("inspect", "promptfoo", "deepeval", "lm-eval")

# inspect-ai maps its four score literals this way in value_to_float
# (inspect_ai/scorer/_metric.py): CORRECT -> 1.0, PARTIAL -> 0.5, and both INCORRECT and
# NOANSWER -> 0.0. Reproduced rather than reinterpreted, so a corrected number matches the
# accuracy inspect itself reports.
_INSPECT_VALUES = {"C": 1.0, "P": 0.5, "I": 0.0, "N": 0.0}

_SCALAR = (str, int, float, bool)

# Keys each adapter consumes itself. Anything outside these sets is a column somebody
# added, and is carried through under its own name by _carry_extras.
_INSPECT_KNOWN = frozenset(
    {
        "id",
        "epoch",
        "input",
        "choices",
        "target",
        "sandbox",
        "files",
        "setup",
        "messages",
        "output",
        "scores",
        "metadata",
        "store",
        "events",
        "attachments",
        "error",
        "limit",
        "total_time",
        "working_time",
        "uuid",
        "model_usage",
        "working_start",
        "span_id",
    }
)
_PROMPTFOO_KNOWN = frozenset(
    {
        "id",
        "promptIdx",
        "testIdx",
        "score",
        "success",
        "namedScores",
        "gradingResult",
        "provider",
        "response",
        "testCase",
        "latencyMs",
        "cost",
        "prompt",
        "promptId",
        "vars",
        "error",
        "description",
        "failureReason",
        "metadata",
        "tokenUsage",
        "traceId",
        "evaluationId",
    }
)
_DEEPEVAL_KNOWN = frozenset(
    {
        "name",
        "input",
        "actualOutput",
        "expectedOutput",
        "context",
        "retrievalContext",
        "toolsCalled",
        "expectedTools",
        "tokenCost",
        "completionTime",
        "tags",
        "success",
        "metricsData",
        "runDuration",
        "evaluationCost",
        "order",
        "metadata",
        "comments",
        "trace",
        "imagesMapping",
    }
)
_LM_EVAL_KNOWN = frozenset(
    {
        "doc_id",
        "doc",
        "target",
        "arguments",
        "resps",
        "filtered_resps",
        "filter",
        "metrics",
        "doc_hash",
        "prompt_hash",
        "target_hash",
    }
)


@dataclass(frozen=True)
class EvalFormat:
    """One eval file, recognized and flattened.

    Attributes:
        tool: One of :data:`SUPPORTED_TOOLS`, or ``"generic"`` when nothing matched.
        rows: Flat row dictionaries, one per example.
        judge_columns: Columns holding a judge verdict, best candidate first. Empty for
            ``generic``, where the caller names the column.
        id_column: Column of example identifiers, or ``None``.
        covariates: Numeric columns worth regressing judge error on.
        segments: Categorical columns worth slicing by.
        notes: Anything a reader should know about how the values were mapped.
        source: Where the rows came from, carried into the report.
    """

    tool: str
    rows: list[dict[str, Any]]
    judge_columns: tuple[str, ...] = ()
    id_column: str | None = None
    covariates: tuple[str, ...] = ()
    segments: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    source: str = ""

    @property
    def judge_column(self) -> str | None:
        """The best judge-verdict candidate, or ``None`` if the format carries no verdict."""
        return self.judge_columns[0] if self.judge_columns else None

    def summary(self) -> str:
        """Human-readable account of what was recognized."""
        lines = [f"{self.tool}: {len(self.rows)} examples from {self.source or 'memory'}"]
        if self.judge_columns:
            lines.append(f"  judge verdict: {', '.join(self.judge_columns)}")
        else:
            lines.append("  judge verdict: not identified; name the column yourself")
        if self.id_column:
            lines.append(f"  identifier: {self.id_column}")
        if self.covariates:
            lines.append(f"  covariates: {', '.join(self.covariates)}")
        if self.segments:
            lines.append(f"  segments: {', '.join(self.segments)}")
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def _slug(name: str) -> str:
    """Lowercase a tool-supplied label into something safe to use as a column name."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return cleaned or "unnamed"


def _scalar(value: Any) -> Any | None:
    """Keep values a row can hold; drop nested structures rather than stringifying them."""
    if value is None or isinstance(value, _SCALAR):
        return value
    return None


def _put(row: dict[str, Any], key: str, value: Any) -> None:
    """Set a scalar column, skipping absent values so sparse fields stay sparse."""
    kept = _scalar(value)
    if kept is not None:
        row[key] = kept


def _spread(row: dict[str, Any], prefix: str, mapping: Any) -> list[str]:
    """Copy the scalar entries of a nested mapping onto the row under a prefix."""
    added: list[str] = []
    if not isinstance(mapping, Mapping):
        return added
    for key, value in mapping.items():
        kept = _scalar(value)
        if kept is not None:
            column = f"{prefix}{_slug(key)}"
            row[column] = kept
            added.append(column)
    return added


def _carry_extras(row: dict[str, Any], record: Mapping[str, Any], known: frozenset[str]) -> None:
    """Copy scalar keys the adapter does not recognize onto the row, under their own names.

    People add columns to eval output by hand, and a human verdict pasted into the file is
    the most valuable column in it. Dropping unrecognized keys during normalization would
    discard exactly that, with no error and no way to notice.
    """
    for key, value in record.items():
        if key in known or key in row:
            continue
        kept = _scalar(value)
        if kept is not None:
            row[key] = kept


def _text_length(value: Any) -> int | None:
    """Character count of a response, for the verbosity-bias regression."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
        return len(value[0])
    return None


@dataclass
class _Accumulator:
    """Column names discovered while walking rows, kept in first-seen order."""

    seen: dict[str, None] = field(default_factory=dict)

    def add(self, names: Sequence[str]) -> None:
        for name in names:
            self.seen.setdefault(name, None)

    def ordered(self) -> tuple[str, ...]:
        return tuple(self.seen)


# --------------------------------------------------------------------------------------
# inspect-ai
# --------------------------------------------------------------------------------------


def _is_inspect(obj: Any) -> bool:
    return (
        isinstance(obj, Mapping)
        and isinstance(obj.get("eval"), Mapping)
        and isinstance(obj.get("samples"), list)
    )


def _read_inspect(obj: Mapping[str, Any], source: str) -> EvalFormat:
    samples = obj.get("samples") or []
    if not samples:
        raise ValueError(
            f"{source}: this is an inspect log but it carries no samples. Logs written "
            "with --no-log-samples record only aggregate scores, which cannot be corrected."
        )

    rows: list[dict[str, Any]] = []
    scorers = _Accumulator()
    meta = _Accumulator()
    multi_epoch = any(int(s.get("epoch", 1) or 1) > 1 for s in samples if isinstance(s, Mapping))

    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        identifier = sample.get("id")
        epoch = sample.get("epoch", 1)
        row: dict[str, Any] = {
            "id": f"{identifier}#{epoch}" if multi_epoch else str(identifier),
        }
        _put(row, "epoch", epoch)
        _put(row, "target", sample.get("target"))

        scores = sample.get("scores")
        if isinstance(scores, Mapping):
            for scorer, score in scores.items():
                if not isinstance(score, Mapping):
                    continue
                column = f"score.{_slug(scorer)}"
                value = score.get("value")
                mapped = _INSPECT_VALUES.get(value) if isinstance(value, str) else _scalar(value)
                if mapped is None:
                    mapped = _scalar(value)
                if mapped is not None:
                    row[column] = mapped
                    scorers.add([column])
                _put(row, f"{column}.answer", score.get("answer"))

        meta.add(_spread(row, "meta.", sample.get("metadata")))

        output = sample.get("output")
        if isinstance(output, Mapping):
            completion = output.get("completion")
            length = _text_length(completion)
            if length is not None:
                row["response_chars"] = length
        _carry_extras(row, sample, _INSPECT_KNOWN)
        rows.append(row)

    covariates = ["response_chars"] if any("response_chars" in r for r in rows) else []
    covariates += [c for c in meta.ordered() if _numeric_column(rows, c)]
    return EvalFormat(
        tool="inspect",
        rows=rows,
        judge_columns=scorers.ordered(),
        id_column="id" if any(r.get("id") not in (None, "None") for r in rows) else None,
        covariates=tuple(covariates),
        segments=tuple(c for c in meta.ordered() if _categorical_column(rows, c)),
        notes=(
            'score values "C", "P", "I" and "N" were mapped to 1.0, 0.5, 0.0 and 0.0, '
            "matching value_to_float in inspect_ai/scorer/_metric.py",
        ),
        source=source,
    )


# --------------------------------------------------------------------------------------
# promptfoo
# --------------------------------------------------------------------------------------


def _promptfoo_records(obj: Any) -> list[Any] | None:
    """Locate the result list in a promptfoo output file, v2 or v3."""
    if isinstance(obj, Mapping):
        results = obj.get("results")
        if isinstance(results, Mapping):
            nested = results.get("results")
            if isinstance(nested, list):
                return nested
        if isinstance(results, list) and _looks_like_promptfoo_record(results):
            return results
    if isinstance(obj, list) and _looks_like_promptfoo_record(obj):
        return obj
    return None


# toEvaluateResult always emits promptIdx, provider and testCase alongside gradingResult,
# so a record carrying only one of these is somebody else's file that happens to share a
# key name. Requiring two markers keeps a hand-rolled JSONL on the generic path, where its
# own column names survive.
_PROMPTFOO_MARKERS = ("gradingResult", "promptIdx", "testIdx", "testCase", "namedScores")


def _looks_like_promptfoo_record(records: Sequence[Any]) -> bool:
    for record in records:
        if isinstance(record, Mapping):
            return sum(marker in record for marker in _PROMPTFOO_MARKERS) >= 2
    return False


def _read_promptfoo(records: Sequence[Any], source: str) -> EvalFormat:
    rows: list[dict[str, Any]] = []
    named = _Accumulator()
    variables = _Accumulator()
    judge = _Accumulator()

    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        row: dict[str, Any] = {}
        identifier = record.get("id")
        if identifier is None:
            prompt_index = record.get("promptIdx", 0)
            test_index = record.get("testIdx", position)
            identifier = f"{prompt_index}:{test_index}"
        row["id"] = str(identifier)

        if "success" in record:
            _put(row, "success", record.get("success"))
            judge.add(["success"])
        if "score" in record:
            _put(row, "score", record.get("score"))
            judge.add(["score"])

        grading = record.get("gradingResult")
        if isinstance(grading, Mapping):
            if "pass" in grading:
                _put(row, "grading_pass", grading.get("pass"))
                judge.add(["grading_pass"])
            _put(row, "grading_score", grading.get("score"))
            _put(row, "grading_reason", grading.get("reason"))

        named.add(_spread(row, "named.", record.get("namedScores")))

        test_case = record.get("testCase")
        if isinstance(test_case, Mapping):
            variables.add(_spread(row, "var.", test_case.get("vars")))

        provider = record.get("provider")
        if isinstance(provider, Mapping):
            _put(row, "provider", provider.get("label") or provider.get("id"))
        elif isinstance(provider, str):
            _put(row, "provider", provider)

        _put(row, "latency_ms", record.get("latencyMs"))
        _put(row, "cost", record.get("cost"))

        response = record.get("response")
        if isinstance(response, Mapping):
            length = _text_length(response.get("output"))
            if length is not None:
                row["response_chars"] = length
        _carry_extras(row, record, _PROMPTFOO_KNOWN)
        rows.append(row)

    if not rows:
        raise ValueError(f"{source}: promptfoo output with no result records")

    ordered_judge = list(judge.ordered())
    # namedScores are per-assertion verdicts. They belong after the overall verdict, since
    # a caller who wants one will name it, and a caller who does not wants the overall one.
    ordered_judge += list(named.ordered())
    covariates = [c for c in ("response_chars", "latency_ms", "cost") if _present(rows, c)]
    covariates += [c for c in variables.ordered() if _numeric_column(rows, c)]
    # provider goes through the same test as any other candidate: a single-provider run
    # has nothing to slice by, and offering it would cost the reader a wasted command.
    segments = [c for c in ("provider", *variables.ordered()) if _categorical_column(rows, c)]
    return EvalFormat(
        tool="promptfoo",
        rows=rows,
        judge_columns=tuple(ordered_judge),
        id_column="id",
        covariates=tuple(covariates),
        segments=tuple(segments),
        source=source,
    )


# --------------------------------------------------------------------------------------
# deepeval
# --------------------------------------------------------------------------------------


def _is_deepeval(obj: Any) -> bool:
    if not isinstance(obj, Mapping):
        return False
    cases = obj.get("testCases")
    if not isinstance(cases, list):
        return False
    for case in cases:
        if isinstance(case, Mapping):
            return "metricsData" in case or "success" in case
    return False


def _read_deepeval(obj: Mapping[str, Any], source: str) -> EvalFormat:
    cases = obj.get("testCases") or []
    rows: list[dict[str, Any]] = []
    metrics = _Accumulator()

    for position, case in enumerate(cases):
        if not isinstance(case, Mapping):
            continue
        row: dict[str, Any] = {"id": str(case.get("name") or case.get("order") or position)}
        if "success" in case:
            _put(row, "success", case.get("success"))

        for metric in case.get("metricsData") or []:
            if not isinstance(metric, Mapping):
                continue
            slug = _slug(metric.get("name", "metric"))
            _put(row, f"metric.{slug}.success", metric.get("success"))
            _put(row, f"metric.{slug}.score", metric.get("score"))
            _put(row, f"metric.{slug}.threshold", metric.get("threshold"))
            if metric.get("success") is not None:
                metrics.add([f"metric.{slug}.success"])
            elif metric.get("score") is not None:
                metrics.add([f"metric.{slug}.score"])

        length = _text_length(case.get("actualOutput"))
        if length is not None:
            row["response_chars"] = length
        _carry_extras(row, case, _DEEPEVAL_KNOWN)
        rows.append(row)

    if not rows:
        raise ValueError(f"{source}: deepeval run with no test cases")

    judge = (["success"] if _present(rows, "success") else []) + list(metrics.ordered())
    return EvalFormat(
        tool="deepeval",
        rows=rows,
        judge_columns=tuple(judge),
        id_column="id",
        covariates=("response_chars",) if _present(rows, "response_chars") else (),
        source=source,
    )


# --------------------------------------------------------------------------------------
# lm-evaluation-harness
# --------------------------------------------------------------------------------------


def _is_lm_eval(rows: Sequence[Any]) -> bool:
    for row in rows:
        if isinstance(row, Mapping):
            return "doc_id" in row and ("filtered_resps" in row or "resps" in row)
    return False


def _read_lm_eval(records: Sequence[Any], source: str) -> EvalFormat:
    rows: list[dict[str, Any]] = []
    metrics = _Accumulator()
    doc_fields = _Accumulator()

    for record in records:
        if not isinstance(record, Mapping):
            continue
        row: dict[str, Any] = {"id": str(record.get("doc_id"))}
        # lm-eval merges each metric value onto the record with example.update(metrics)
        # and lists the names it merged under "metrics" (lm_eval/evaluator.py).
        for name in record.get("metrics") or []:
            if isinstance(name, str) and name in record:
                column = _slug(name)
                _put(row, column, record.get(name))
                metrics.add([column])
        _put(row, "target", record.get("target"))
        doc_fields.add(_spread(row, "doc.", record.get("doc")))
        length = _text_length(record.get("filtered_resps") or record.get("resps"))
        if length is not None:
            row["response_chars"] = length
        known = _LM_EVAL_KNOWN | {str(n) for n in record.get("metrics") or []}
        _carry_extras(row, record, frozenset(known))
        rows.append(row)

    if not rows:
        raise ValueError(f"{source}: lm-eval samples file with no records")

    covariates = ["response_chars"] if _present(rows, "response_chars") else []
    covariates += [c for c in doc_fields.ordered() if _numeric_column(rows, c)]
    return EvalFormat(
        tool="lm-eval",
        rows=rows,
        judge_columns=metrics.ordered(),
        id_column="id",
        covariates=tuple(covariates),
        segments=tuple(c for c in doc_fields.ordered() if _categorical_column(rows, c)),
        source=source,
    )


# --------------------------------------------------------------------------------------
# column classification, and the entry points
# --------------------------------------------------------------------------------------


def _present(rows: Sequence[Mapping[str, Any]], column: str) -> bool:
    return any(column in row for row in rows)


def _numeric_column(rows: Sequence[Mapping[str, Any]], column: str) -> bool:
    values = [row[column] for row in rows if column in row]
    if len(values) < len(rows):
        return False
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def _categorical_column(rows: Sequence[Mapping[str, Any]], column: str) -> bool:
    values = [row[column] for row in rows if column in row]
    if len(values) < len(rows) or not all(isinstance(v, str) for v in values):
        return False
    distinct = len(set(values))
    # A column with one level cannot separate anything, and one with a level per example is
    # an identifier wearing a segment's clothes.
    return 1 < distinct <= max(2, len(values) // 2)


def detect_format(obj: Any, source: str = "memory") -> EvalFormat:
    """Recognize a parsed eval file and flatten it to rows.

    Args:
        obj: Parsed JSON: the wrapper object an eval tool writes, or a list of records.
        source: Where it came from, carried into the result for the report.

    Returns:
        An :class:`EvalFormat`. ``tool`` is ``"generic"`` when nothing matched, in which
        case ``rows`` is the input unchanged and no columns are identified.

    Raises:
        ValueError: If the format is recognized but carries no per-example records, which
            an aggregate-only log does.

    References:
        tests/test_adapters.py::test_generic_input_is_returned_untouched
    """
    if _is_inspect(obj):
        return _read_inspect(obj, source)
    if _is_deepeval(obj):
        return _read_deepeval(obj, source)
    records = _promptfoo_records(obj)
    if records is not None:
        return _read_promptfoo(records, source)
    if isinstance(obj, list) and _is_lm_eval(obj):
        return _read_lm_eval(obj, source)

    rows = [dict(item) for item in obj] if isinstance(obj, list) else []
    return EvalFormat(tool="generic", rows=rows, source=source)


def read_eval(path: str | Path) -> EvalFormat:
    """Read an eval file written by a supported tool, or fall back to plain rows.

    Handles the wrapper objects that :func:`truescore.read_rows` cannot: a promptfoo
    ``--output`` file and an inspect ``EvalLog`` are single JSON objects with the records
    nested inside, not arrays.

    Args:
        path: File to read. Any suffix ``read_rows`` accepts, plus ``.json`` holding a
            single wrapper object.

    Returns:
        An :class:`EvalFormat` naming the tool and the columns it identified.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed, or is a recognized format carrying no
            per-example records.

    References:
        tests/test_adapters.py::test_reads_a_promptfoo_output_file
        tests/test_adapters.py::test_reads_an_inspect_eval_log
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"no such file: {file_path}")

    if file_path.suffix.lower() == ".json":
        text = file_path.read_text(encoding="utf-8").strip()
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"{file_path}: {error}") from error
            found = detect_format(parsed, str(file_path))
            if found.tool == "generic":
                raise ValueError(
                    f"{file_path}: a single JSON object that matches no supported tool "
                    f"({', '.join(SUPPORTED_TOOLS)}). Supply one record per line, a "
                    "top-level array, or CSV."
                )
            return found

    return detect_format(read_rows(file_path), str(file_path))
