"""Tests for truescore.doctor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from truescore.doctor import diagnose

WITH_LABELS = """id,judge,human,tokens,segment
a,1,1,120,billing
b,1,,340,billing
c,0,0,88,technical
d,1,0,410,technical
e,0,,150,billing
f,1,1,220,technical
"""


@pytest.fixture
def labelled(tmp_path: Path) -> Path:
    path = tmp_path / "eval.csv"
    path.write_text(WITH_LABELS, encoding="utf-8")
    return path


def test_diagnose_finds_judge_and_gold_columns(labelled: Path) -> None:
    """The guess that removes the first fifteen minutes of adoption."""
    result = diagnose(labelled)
    kinds = {c.name: c.kind for c in result.columns}

    assert kinds["judge"] == "verdict"
    assert kinds["human"] == "sparse_verdict"
    assert kinds["tokens"] == "numeric"
    assert kinds["segment"] == "categorical"
    assert result.judge_candidates == ("judge",)
    assert result.gold_candidates == ("human",)


def test_diagnose_says_what_is_blocked_without_human_labels(tmp_path: Path) -> None:
    """The common case: a judge column and nothing to check it against."""
    path = tmp_path / "nogold.csv"
    path.write_text("judge,tokens\n" + "\n".join("1,100" for _ in range(200)), encoding="utf-8")

    result = diagnose(path)
    blocked = " ".join(f"{what}: {why}" for what, why in result.blocked)

    assert "no human labels" in blocked
    assert any(char.isdigit() for char in blocked), "it should say how many labels would help"
    assert any("random sample" in rec for rec in result.recommendations)
    assert any("escalations" in rec for rec in result.recommendations)


def test_monitoring_is_offered_even_with_no_human_labels(tmp_path: Path) -> None:
    """Roughly half the library needs no humans, and the report should lead with that."""
    path = tmp_path / "nogold.csv"
    path.write_text("judge\n" + "\n".join("1" for _ in range(50)), encoding="utf-8")
    result = diagnose(path)
    assert any("monitor" in item for item in result.available)


def test_bias_scan_finds_a_planted_effect_and_corrects_for_scanning(tmp_path: Path) -> None:
    """One real covariate among several null ones is found; the nulls are not reported."""
    rng = np.random.default_rng(0)
    n = 900
    tokens = rng.uniform(50, 800, n)
    truth = rng.binomial(1, 0.6, n)
    # The judge over-scores long answers, and is indifferent to the other three columns.
    flip = rng.random(n) < (0.05 + 0.0006 * tokens)
    judge = np.where((truth == 1) | flip, 1, 0)

    # Human labels are sparse, as they are in practice -- which is also what lets the
    # profiler tell a gold column from a second judge column.
    labelled = set(rng.choice(n, 700, replace=False).tolist())
    header = "judge,human,tokens,noise1,noise2,noise3\n"
    lines = [
        f"{judge[i]},{truth[i] if i in labelled else ''},{tokens[i]:.1f},"
        f"{rng.normal():.3f},{rng.normal():.3f},{rng.normal():.3f}"
        for i in range(n)
    ]
    path = tmp_path / "scan.csv"
    path.write_text(header + "\n".join(lines), encoding="utf-8")

    findings = " ".join(diagnose(path).bias_findings)
    assert "tokens" in findings
    assert "more generous" in findings
    for null_column in ("noise1", "noise2", "noise3"):
        assert null_column not in findings, "a scanned null column must not be reported"


def test_bias_scan_reports_nothing_when_there_is_nothing(tmp_path: Path) -> None:
    """An unbiased judge produces an explicit negative, not silence."""
    rng = np.random.default_rng(1)
    n = 600
    truth = rng.binomial(1, 0.7, n)
    judge = np.where(rng.random(n) < 0.9, truth, 1 - truth)
    tokens = rng.uniform(50, 500, n)
    path = tmp_path / "clean.csv"
    labelled = set(rng.choice(n, 450, replace=False).tolist())
    path.write_text(
        "judge,human,tokens\n"
        + "\n".join(
            f"{judge[i]},{truth[i] if i in labelled else ''},{tokens[i]:.1f}" for i in range(n)
        ),
        encoding="utf-8",
    )
    findings = " ".join(diagnose(path).bias_findings)
    assert "nothing detectable" in findings


def test_slices_are_offered_when_a_categorical_column_exists(labelled: Path) -> None:
    result = diagnose(labelled)
    assert result.slice_candidates == ("segment",)
    assert any("segment" in item for item in result.available)


def test_two_verdict_columns_with_no_sparse_one_is_flagged_as_ambiguous(
    tmp_path: Path,
) -> None:
    """Nothing in the data says which of two full pass/fail columns is the human."""
    path = tmp_path / "ambiguous.csv"
    path.write_text("a,b\n" + "\n".join("1,0" for _ in range(40)), encoding="utf-8")
    result = diagnose(path)
    assert any("nothing in the data distinguishes them" in rec for rec in result.recommendations)


def test_missing_segment_column_is_suggested(tmp_path: Path) -> None:
    path = tmp_path / "flat.csv"
    path.write_text(
        "judge,human\n" + "\n".join(f"1,{'1' if i % 3 else ''}" for i in range(40)),
        encoding="utf-8",
    )
    result = diagnose(path)
    assert any("segment, language or customer tier" in rec for rec in result.recommendations)


def test_a_file_with_no_judge_column_says_so(tmp_path: Path) -> None:
    path = tmp_path / "useless.csv"
    path.write_text("note,other\nhello,world\nfoo,bar\n", encoding="utf-8")
    result = diagnose(path)
    assert any("no column looks like a judge verdict" in why for _what, why in result.blocked)


def test_an_empty_column_is_marked_unusable(tmp_path: Path) -> None:
    path = tmp_path / "blank.csv"
    path.write_text("judge,empty\n1,\n0,\n1,\n", encoding="utf-8")
    kinds = {c.name: c.kind for c in diagnose(path).columns}
    assert kinds["empty"] == "unusable"


def test_summary_is_readable_and_names_commands(labelled: Path) -> None:
    text = diagnose(labelled).summary()
    assert "what this file supports today" in text
    assert "truescore audit" in text, "the report should hand over a runnable command"
    assert "coverage" in text


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        diagnose(tmp_path / "absent.csv")


def test_doctor_profiles_nested_json_paths(tmp_path: Path) -> None:
    """A harness's nested output should profile like a table, with usable path names."""
    path = tmp_path / "nested.jsonl"
    rows = [
        {
            "gradingResult": {"pass": i % 4 != 0},
            "response": {"tokenUsage": {"completion": 100 + i}},
            "human": (i % 4 != 0) if i % 2 == 0 else None,
        }
        for i in range(40)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    result = diagnose(path)
    kinds = {c.name: c.kind for c in result.columns}

    assert kinds["gradingResult.pass"] == "verdict"
    assert kinds["response.tokenUsage.completion"] == "numeric"
    assert kinds["human"] == "sparse_verdict"
    # The suggested commands must carry the dotted path, or they won't run.
    assert any("--judge gradingResult.pass" in item for item in result.available)


def test_doctor_recognises_a_rubric_score(tmp_path: Path) -> None:
    """Teams grade on 1-5 at least as often as pass/fail, and the metrics differ."""
    path = tmp_path / "rubric.csv"
    rows = ["id,judge_score,human_score"]
    for i in range(200):
        gold = (i % 5) + 1
        rows.append(f"q{i},{min(5, gold + (i % 2))},{gold if i % 4 == 0 else ''}")
    path.write_text("\n".join(rows), encoding="utf-8")

    result = diagnose(path)
    kinds = {c.name: c.kind for c in result.columns}

    assert kinds["judge_score"] == "graded"
    assert kinds["human_score"] == "sparse_graded"
    assert any("quadratic-weighted kappa" in item for item in result.available)


def test_a_rubric_without_human_scores_says_what_is_blocked(tmp_path: Path) -> None:
    path = tmp_path / "rubric_nogold.csv"
    rows = ["id,judge_score"] + [f"q{i},{(i % 5) + 1}" for i in range(120)]
    path.write_text("\n".join(rows), encoding="utf-8")

    result = diagnose(path)
    blocked = " ".join(f"{what}: {why}" for what, why in result.blocked)
    assert "no human rubric scores" in blocked
    assert "everything" not in blocked, "a rubric column is not nothing"


def test_a_binary_column_is_still_a_verdict_not_a_rubric(tmp_path: Path) -> None:
    """The graded heuristic must not swallow pass/fail columns."""
    path = tmp_path / "binary.csv"
    path.write_text("judge\n" + "\n".join(str(i % 2) for i in range(60)), encoding="utf-8")
    assert {c.name: c.kind for c in diagnose(path).columns}["judge"] == "verdict"


def test_a_wide_numeric_column_is_a_covariate_not_a_rubric(tmp_path: Path) -> None:
    path = tmp_path / "tokens.csv"
    rows = ["judge,tokens"] + [f"1,{100 + i}" for i in range(60)]
    path.write_text("\n".join(rows), encoding="utf-8")
    assert {c.name: c.kind for c in diagnose(path).columns}["tokens"] == "numeric"
