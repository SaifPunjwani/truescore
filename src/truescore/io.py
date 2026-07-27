# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Saif Punjwani
"""Reading the files evaluation teams actually have.

The shape of real eval output is one row per example: a judge verdict on every row, and a
human verdict on the handful of rows somebody got round to labeling. That sparse gold
column is not a defect in the data -- it is exactly the input prediction-powered inference
wants, and this module turns it into the arrays the estimators expect without asking
anyone to reshape anything.

Supports CSV and JSON Lines, with no pandas dependency. Verdicts are read leniently
(``1/0``, ``true/false``, ``yes/no``, ``pass/fail``, ``correct/incorrect``) because eval
harnesses all spell them differently, and unreadable values raise with the offending row
number rather than being coerced to something plausible.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["GoldJoin", "LabelSet", "get_field", "join_gold", "load_labels", "read_rows"]

_TRUE_TOKENS = frozenset({"1", "true", "t", "yes", "y", "pass", "passed", "correct", "good"})
_FALSE_TOKENS = frozenset({"0", "false", "f", "no", "n", "fail", "failed", "incorrect", "bad"})
_MISSING_TOKENS = frozenset({"", "na", "n/a", "nan", "none", "null", "-", "unlabeled"})


@dataclass(frozen=True)
class LabelSet:
    """Judge verdicts for every example, plus gold verdicts for the labeled subset.

    Attributes:
        judge: Judge verdict per example, length ``n_total``.
        gold: Gold verdict per labeled example, length ``n_gold``.
        gold_index: Positions in ``judge`` that carry a gold verdict.
        covariates: Per-example covariate columns, each of length ``n_total``.
        ids: Optional example identifiers, length ``n_total``.
        source: Where the data came from, recorded for the report artifact.
    """

    judge: npt.NDArray[Any]
    gold: npt.NDArray[Any]
    gold_index: npt.NDArray[Any]
    covariates: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    ids: npt.NDArray[Any] | None = None
    source: str = ""

    @property
    def n_total(self) -> int:
        """Examples carrying a judge verdict."""
        return int(self.judge.shape[0])

    @property
    def n_gold(self) -> int:
        """Examples carrying a gold verdict."""
        return int(self.gold.shape[0])

    @property
    def gold_coverage(self) -> float:
        """Fraction of examples that were human-labeled."""
        return self.n_gold / self.n_total if self.n_total else 0.0

    def covariates_on_gold(self) -> dict[str, npt.NDArray[Any]]:
        """Covariate columns restricted to the gold-labeled rows.

        The bias regression compares judge and gold verdicts, so it can only use the rows
        where both exist.
        """
        return {name: values[self.gold_index] for name, values in self.covariates.items()}

    def summary(self) -> str:
        """Human-readable multi-line summary of what was loaded."""
        lines = [
            f"loaded {self.n_total} examples from {self.source or 'memory'}",
            f"  human-labeled: {self.n_gold} ({self.gold_coverage:.1%})",
            f"  judge positive rate: {float(self.judge.mean()):.4f}",
        ]
        if self.n_gold:
            lines.append(f"  gold positive rate:  {float(self.gold.mean()):.4f}")
        if self.covariates:
            lines.append(f"  covariates: {', '.join(sorted(self.covariates))}")
        return "\n".join(lines)


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a CSV or JSON Lines file into a list of row dictionaries.

    Args:
        path: File to read. ``.csv`` is parsed as delimited text; ``.jsonl``, ``.ndjson``
            and ``.json`` are parsed as one JSON object per line (a top-level JSON array
            is also accepted).

    Returns:
        One dictionary per row.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty, the suffix is unrecognized, or a line cannot be
            parsed -- reported with the line number.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"no such file: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        with file_path.open(newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    elif suffix in {".jsonl", ".ndjson", ".json"}:
        text = file_path.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError(f"{file_path}: top-level JSON must be an array of objects")
            rows = [dict(item) for item in parsed]
        else:
            rows = []
            for number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(dict(json.loads(line)))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{file_path} line {number}: {error}") from error
    else:
        raise ValueError(
            f"unrecognized file type {suffix!r}; expected .csv, .jsonl, .ndjson or .json"
        )

    if not rows:
        raise ValueError(f"{file_path} contains no rows")
    return rows


def _is_missing(value: Any) -> bool:
    if value is None or value is _MISSING_SENTINEL:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip().lower() in _MISSING_TOKENS


def _parse_verdict(value: Any, *, column: str, row_number: int) -> float:
    """Parse one verdict cell into a number, accepting the many spellings of yes and no."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return 1.0
    if token in _FALSE_TOKENS:
        return 0.0
    try:
        return float(token)
    except ValueError as error:
        raise ValueError(
            f"row {row_number}, column {column!r}: cannot read {value!r} as a verdict. "
            f"Accepted: numbers, {sorted(_TRUE_TOKENS)[:4]}..., {sorted(_FALSE_TOKENS)[:4]}..."
        ) from error


def _parse_number(value: Any, *, column: str, row_number: int) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError as error:
        raise ValueError(
            f"row {row_number}, column {column!r}: cannot read {value!r} as a number"
        ) from error


_MISSING_SENTINEL = object()


def get_field(row: Mapping[str, Any], path: str) -> Any:
    """Read a field from a row, following dots into nested objects and lists.

    Eval harnesses write nested JSON. promptfoo puts the verdict at
    ``gradingResult.pass``; other tools nest one or two levels deeper, sometimes through a
    list. Rather than making every user flatten their output first, a column name here may
    be a dotted path: ``gradingResult.pass``, ``scores.0.value``, ``metrics.accuracy``.

    A plain name with no dots is looked up directly, so ordinary CSV columns keep working
    even if they contain dots in their header.

    Args:
        row: One row of the file.
        path: A column name, or a dotted path into nested structures.

    Returns:
        The value, or a sentinel meaning "absent" that callers treat as missing.

    References:
        tests/test_io.py::test_dotted_paths_read_nested_json
    """
    if path in row:
        return row[path]
    current: Any = row
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return _MISSING_SENTINEL
            current = current[part]
        elif isinstance(current, (list, tuple)):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return _MISSING_SENTINEL
        else:
            return _MISSING_SENTINEL
    return current


def _available_paths(row: Mapping[str, Any], prefix: str = "", depth: int = 3) -> list[str]:
    """Paths a user could plausibly mean, for the error message when one is wrong."""
    found: list[str] = []
    for key, value in row.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and depth > 1:
            found.extend(_available_paths(value, f"{path}.", depth - 1))
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], Mapping):
            if depth > 1:
                found.extend(_available_paths(value[0], f"{path}.0.", depth - 1))
        else:
            found.append(path)
    return found


def _require_column(rows: Sequence[dict[str, Any]], column: str) -> None:
    if get_field(rows[0], column) is _MISSING_SENTINEL:
        available = ", ".join(sorted(_available_paths(rows[0]))[:25]) or "(none)"
        raise ValueError(f"column {column!r} not found; available columns: {available}")


@dataclass(frozen=True)
class GoldJoin:
    """Eval rows with human labels attached, and an account of what matched.

    Attributes:
        rows: The eval rows, each matched row carrying the gold column.
        matched: Eval rows that received a human label.
        unmatched_gold: Human labels whose identifier appears in no eval row. Reported
            rather than dropped silently, because a label that does not land is either a
            wrong join key or an expensive label thrown away.
        gold_column: Name of the column the labels were written to.
    """

    rows: list[dict[str, Any]]
    matched: int
    unmatched_gold: int
    gold_column: str

    def summary(self) -> str:
        """Human-readable account of the join."""
        lines = [f"joined {self.matched} human labels onto {len(self.rows)} examples"]
        if self.unmatched_gold:
            lines.append(
                f"  warning: {self.unmatched_gold} human labels matched no example and "
                "were not used"
            )
        return "\n".join(lines)


def join_gold(
    rows: Sequence[Mapping[str, Any]],
    gold_source: str | Path | Sequence[Mapping[str, Any]],
    *,
    on: str,
    gold: str,
    gold_on: str | None = None,
    into: str = "gold",
) -> GoldJoin:
    """Attach human labels from a separate file to eval rows, matched by identifier.

    Human labels almost never live in the file the eval tool wrote. The tool writes
    verdicts; somebody labels a subset in a spreadsheet afterwards. This joins the two by
    identifier so neither file has to be edited by hand.

    Args:
        rows: Eval rows, one per example.
        gold_source: Path to a CSV or JSON Lines file of human labels, or rows in memory.
        on: Identifier column in ``rows``.
        gold: Column in ``gold_source`` holding the human verdict.
        gold_on: Identifier column in ``gold_source``. Defaults to ``on``.
        into: Column name to write the human verdict to on matched rows.

    Returns:
        A :class:`GoldJoin`. Rows that received no label simply lack the column, which
        :func:`load_labels` reads as unlabeled.

    Raises:
        ValueError: If a column is absent, if an identifier repeats in either input, or if
            no human label matches any example -- which means the join key is wrong, and
            is worth failing on rather than reporting a corrected number from zero labels.

    References:
        tests/test_io.py::test_join_gold_attaches_labels_by_identifier
        tests/test_io.py::test_join_gold_rejects_a_join_key_that_matches_nothing
    """
    eval_rows = [dict(row) for row in rows]
    if not eval_rows:
        raise ValueError("no eval rows to join onto")
    _require_column(eval_rows, on)

    if isinstance(gold_source, (str, Path)):
        gold_rows: list[dict[str, Any]] = read_rows(gold_source)
    else:
        gold_rows = [dict(row) for row in gold_source]
    if not gold_rows:
        raise ValueError("the human-label file contains no rows")

    key_column = gold_on or on
    _require_column(gold_rows, key_column)
    _require_column(gold_rows, gold)

    positions: dict[str, int] = {}
    for position, row in enumerate(eval_rows):
        identifier = str(get_field(row, on))
        if identifier in positions:
            raise ValueError(
                f"identifier {identifier!r} appears more than once in column {on!r}. "
                "A human label would attach to every copy and be counted more than once; "
                "pick a column that is unique per example."
            )
        positions[identifier] = position

    seen: set[str] = set()
    matched = 0
    unmatched = 0
    for number, row in enumerate(gold_rows, start=1):
        identifier = str(get_field(row, key_column))
        if identifier in seen:
            raise ValueError(
                f"identifier {identifier!r} appears more than once in the human-label file "
                f"(row {number}). Two humans labeling one example is a real situation, but "
                "it needs resolving into a single verdict before correction, or measuring "
                "with truescore.agreement."
            )
        seen.add(identifier)
        value = get_field(row, gold)
        if _is_missing(value):
            continue
        target = positions.get(identifier)
        if target is None:
            unmatched += 1
            continue
        eval_rows[target][into] = value
        matched += 1

    if matched == 0:
        sample = ", ".join(sorted(positions)[:3])
        raise ValueError(
            f"no human label matched any example: {unmatched} labels were read from "
            f"{key_column!r} and none appears in {on!r}. Example identifiers look like: "
            f"{sample}. Check that the two files identify examples the same way."
        )
    return GoldJoin(rows=eval_rows, matched=matched, unmatched_gold=unmatched, gold_column=into)


def load_labels(
    source: str | Path | Sequence[dict[str, Any]],
    *,
    judge: str,
    gold: str | None = None,
    id_column: str | None = None,
    covariates: Iterable[str] = (),
) -> LabelSet:
    """Load judge and gold verdicts from an evaluation file.

    Args:
        source: Path to a CSV or JSON Lines file, or an in-memory sequence of row
            dictionaries.
        judge: Column holding the judge's verdict. Must be present on every row.
        gold: Column holding the human verdict. Rows where it is blank are treated as
            unlabeled, which is the normal case -- a few hundred labels against many
            thousand examples is what prediction-powered inference is built for.
        id_column: Optional column of example identifiers, carried through so drift
            comparisons can prove they used the same examples.
        covariates: Columns to load as numeric per-example covariates for bias analysis,
            for example response length.

    Returns:
        A :class:`LabelSet`.

    Raises:
        ValueError: If a requested column is absent, a cell cannot be parsed, or the judge
            column has missing values.

    References:
        tests/test_io.py::test_load_labels_reads_a_sparse_gold_column
    """
    if isinstance(source, (str, Path)):
        rows = read_rows(source)
        origin = str(source)
    else:
        rows = [dict(row) for row in source]
        if not rows:
            raise ValueError("source contains no rows")
        origin = "memory"

    _require_column(rows, judge)
    judge_values = []
    for number, row in enumerate(rows, start=1):
        value = get_field(row, judge)
        if _is_missing(value):
            raise ValueError(
                f"row {number}, column {judge!r}: the judge verdict is missing. Every "
                "example needs a judge verdict; only the human column may be sparse."
            )
        judge_values.append(_parse_verdict(value, column=judge, row_number=number))
    judge_array = np.asarray(judge_values, dtype=float)

    gold_values: list[float] = []
    gold_positions: list[int] = []
    if gold is not None:
        _require_column(rows, gold)
        for position, (number, row) in enumerate(zip(range(1, len(rows) + 1), rows, strict=True)):
            value = get_field(row, gold)
            if _is_missing(value):
                continue
            gold_values.append(_parse_verdict(value, column=gold, row_number=number))
            gold_positions.append(position)
        if not gold_values:
            raise ValueError(
                f"column {gold!r} has no non-missing values; without at least a few human "
                "labels the judge cannot be corrected, only described"
            )

    covariate_arrays: dict[str, npt.NDArray[Any]] = {}
    for name in covariates:
        _require_column(rows, name)
        values = [
            _parse_number(get_field(row, name), column=name, row_number=number)
            for number, row in enumerate(rows, start=1)
        ]
        covariate_arrays[name] = np.asarray(values, dtype=float)

    ids = None
    if id_column is not None:
        _require_column(rows, id_column)
        ids = np.asarray([str(get_field(row, id_column)) for row in rows])

    return LabelSet(
        judge=judge_array,
        gold=np.asarray(gold_values, dtype=float),
        gold_index=np.asarray(gold_positions, dtype=np.int64),
        covariates=covariate_arrays,
        ids=ids,
        source=origin,
    )
