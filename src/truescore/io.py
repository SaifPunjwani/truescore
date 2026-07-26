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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["LabelSet", "load_labels", "read_rows"]

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
    if value is None:
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


def _require_column(rows: Sequence[dict[str, Any]], column: str) -> None:
    if column not in rows[0]:
        available = ", ".join(sorted(rows[0])) or "(none)"
        raise ValueError(f"column {column!r} not found; available columns: {available}")


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
        value = row.get(judge)
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
            value = row.get(gold)
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
            _parse_number(row.get(name), column=name, row_number=number)
            for number, row in enumerate(rows, start=1)
        ]
        covariate_arrays[name] = np.asarray(values, dtype=float)

    ids = None
    if id_column is not None:
        _require_column(rows, id_column)
        ids = np.asarray([str(row.get(id_column)) for row in rows])

    return LabelSet(
        judge=judge_array,
        gold=np.asarray(gold_values, dtype=float),
        gold_index=np.asarray(gold_positions, dtype=np.int64),
        covariates=covariate_arrays,
        ids=ids,
        source=origin,
    )
