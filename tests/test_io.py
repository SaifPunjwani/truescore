"""Tests for truescore.io.

The on-ramp gets its own tests because every user meets it first, and because a loader
that silently mis-reads a column produces confident, wrong statistics downstream.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from truescore.io import load_labels, read_rows

CSV = """example_id,judge,human,tokens
a,1,1,120
b,1,,340
c,0,0,88
d,1,0,410
e,0,,150
"""


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "eval.csv"
    path.write_text(CSV, encoding="utf-8")
    return path


def test_load_labels_reads_a_sparse_gold_column(csv_path: Path) -> None:
    """The normal shape of real data: judge on every row, human on a few."""
    labels = load_labels(csv_path, judge="judge", gold="human", covariates=["tokens"])

    assert labels.n_total == 5
    assert labels.n_gold == 3
    assert np.array_equal(labels.gold_index, [0, 2, 3])
    assert np.array_equal(labels.gold, [1.0, 0.0, 0.0])
    assert labels.gold_coverage == pytest.approx(0.6)
    assert np.array_equal(labels.covariates["tokens"], [120, 340, 88, 410, 150])
    assert np.array_equal(labels.covariates_on_gold()["tokens"], [120, 88, 410])


def test_verdict_spellings_are_all_understood(tmp_path: Path) -> None:
    """Eval harnesses spell pass and fail a dozen ways; all of them mean the same thing."""
    path = tmp_path / "spellings.csv"
    path.write_text(
        "judge,human\ntrue,PASS\nFalse,fail\nyes,Correct\nno,incorrect\n1,0\n",
        encoding="utf-8",
    )
    labels = load_labels(path, judge="judge", gold="human")
    assert np.array_equal(labels.judge, [1.0, 0.0, 1.0, 0.0, 1.0])
    assert np.array_equal(labels.gold, [1.0, 0.0, 1.0, 0.0, 0.0])


def test_missing_spellings_are_all_treated_as_unlabeled(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"
    path.write_text("judge,human\n1,\n1,NA\n0,n/a\n1,none\n0,1\n", encoding="utf-8")
    labels = load_labels(path, judge="judge", gold="human")
    assert labels.n_gold == 1
    assert np.array_equal(labels.gold_index, [4])


def test_jsonl_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"judge": True, "human": True},
                {"judge": True, "human": None},
                {"judge": False, "human": False},
            ]
        ),
        encoding="utf-8",
    )
    labels = load_labels(path, judge="judge", gold="human")
    assert labels.n_total == 3
    assert labels.n_gold == 2


def test_json_array_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([{"judge": 1, "human": 1}, {"judge": 0, "human": 0}]), "utf-8")
    assert load_labels(path, judge="judge", gold="human").n_total == 2


def test_a_missing_judge_verdict_is_an_error_not_a_gap(tmp_path: Path) -> None:
    """Only the human column may be sparse; a hole in the judge column is a broken run."""
    path = tmp_path / "holes.csv"
    path.write_text("judge,human\n1,1\n,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 2.*judge verdict is missing"):
        load_labels(path, judge="judge", gold="human")


def test_unreadable_cell_names_the_row_and_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("judge,human\n1,1\nmaybe,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 2, column 'judge'"):
        load_labels(path, judge="judge", gold="human")


def test_absent_column_lists_what_is_available(csv_path: Path) -> None:
    with pytest.raises(ValueError, match="available columns: example_id, human, judge, tokens"):
        load_labels(csv_path, judge="verdict")


def test_an_entirely_empty_gold_column_is_rejected(tmp_path: Path) -> None:
    """Without any human labels the judge can be described but not corrected."""
    path = tmp_path / "nogold.csv"
    path.write_text("judge,human\n1,\n0,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no non-missing values"):
        load_labels(path, judge="judge", gold="human")


def test_gold_column_is_optional(csv_path: Path) -> None:
    labels = load_labels(csv_path, judge="judge")
    assert labels.n_total == 5
    assert labels.n_gold == 0
    assert "judge positive rate" in labels.summary()


def test_in_memory_rows_are_accepted(csv_path: Path) -> None:
    """Loading two systems from one file should not read it twice."""
    rows = read_rows(csv_path)
    labels = load_labels(rows, judge="judge", gold="human")
    assert labels.n_total == 5
    assert labels.source == "memory"


def test_unknown_file_type_and_missing_file_are_distinct_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no such file"):
        read_rows(tmp_path / "absent.csv")
    weird = tmp_path / "data.parquet"
    weird.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="unrecognized file type"):
        read_rows(weird)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("judge,human\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no rows"):
        read_rows(path)


def test_malformed_jsonl_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"judge": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_rows(path)


def test_ids_are_carried_through(csv_path: Path) -> None:
    labels = load_labels(csv_path, judge="judge", gold="human", id_column="example_id")
    assert labels.ids is not None
    assert list(labels.ids) == ["a", "b", "c", "d", "e"]


def test_summary_reports_coverage(csv_path: Path) -> None:
    text = load_labels(csv_path, judge="judge", gold="human", covariates=["tokens"]).summary()
    assert "human-labeled: 3" in text
    assert "covariates: tokens" in text
