"""Tests for the command line interface.

Exit codes carry meaning here -- a pipeline gates on them -- so they are asserted as
carefully as the numbers. The distinction that matters most is between a *finding* (2) and
a *failure to run* (1): a monitor that conflates them cannot be trusted in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from truescore.cli import EXIT_ERROR, EXIT_FINDING, EXIT_OK, main

DATA = Path(__file__).parent.parent / "examples" / "data"
pytestmark = pytest.mark.skipif(
    not (DATA / "support_eval.csv").exists(),
    reason="sample data absent; run examples/generate_sample_data.py",
)


def test_audit_reports_the_correction_and_flags_the_finding(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "audit",
            str(DATA / "support_eval.csv"),
            "--judge",
            "judge_passed",
            "--gold",
            "human_passed",
            "--covariate",
            "response_tokens",
            "--system-name",
            "support-v4",
        ]
    )
    out = capsys.readouterr().out
    assert code == EXIT_FINDING, "the judge-only score lies outside the corrected interval"
    assert "corrected:" in out
    assert "judge-only:" in out
    assert "judge error regression" in out
    assert "FINDING" in out


def test_audit_writes_the_artifacts(tmp_path: Path) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    main(
        [
            "audit",
            str(DATA / "support_eval.csv"),
            "--judge",
            "judge_passed",
            "--gold",
            "human_passed",
            "--json",
            str(json_path),
            "--markdown",
            str(markdown_path),
        ]
    )
    assert "corrected" in json_path.read_text(encoding="utf-8")
    assert "# Evaluation report" in markdown_path.read_text(encoding="utf-8")


def test_compare_separates_the_judged_and_corrected_verdicts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "compare",
            str(DATA / "support_compare.csv"),
            "--judge-a",
            "v4_judge_passed",
            "--judge-b",
            "v3_judge_passed",
            "--gold-a",
            "v4_human_passed",
            "--gold-b",
            "v3_human_passed",
        ]
    )
    out = capsys.readouterr().out
    assert code == EXIT_FINDING
    assert "as judged (uncorrected)" in out
    assert "corrected for judge error" in out


def test_drift_exits_with_a_finding_when_the_judge_changed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "drift",
            str(DATA / "judge_anchor.csv"),
            "--baseline",
            "judge_may",
            "--current",
            "judge_june",
            "--gold",
            "human_passed",
        ]
    )
    assert code == EXIT_FINDING
    assert "judge drift on anchor set" in capsys.readouterr().out


def test_monitor_windowed_catches_the_regression(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "monitor",
            str(DATA / "release_stream.csv"),
            "--metric",
            "passed",
            "--baseline",
            "0.88",
            "--window",
            "300",
        ]
    )
    out = capsys.readouterr().out
    assert code == EXIT_FINDING
    assert "was ruled out at observation" in out


def test_monitor_is_quiet_on_a_healthy_stream(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rng = np.random.default_rng(0)
    path = tmp_path / "healthy.csv"
    path.write_text(
        "passed\n" + "\n".join(str(int(v)) for v in rng.binomial(1, 0.88, 900)), "utf-8"
    )
    code = main(
        ["monitor", str(path), "--metric", "passed", "--baseline", "0.88", "--window", "300"]
    )
    assert code == EXIT_OK
    assert "no evidence against the baseline" in capsys.readouterr().out


def test_contamination_reports_a_clean_model(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["contamination", str(DATA / "contamination_logliks.csv")])
    assert code == EXIT_OK
    assert "contamination test on 199 permutations" in capsys.readouterr().out


def test_plan_prints_a_budget(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "plan",
            "--n-total",
            "4000",
            "--target",
            "0.03",
            "--rate",
            "0.7",
            "--sensitivity",
            "0.95",
            "--specificity",
            "0.85",
        ]
    )
    assert code == EXIT_OK
    assert "gold labels needed" in capsys.readouterr().out


def test_agreement_prints_judge_quality(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "agreement",
            str(DATA / "support_eval.csv"),
            "--judge",
            "judge_passed",
            "--gold",
            "human_passed",
        ]
    )
    assert code == EXIT_OK
    assert "sensitivity" in capsys.readouterr().out


def test_a_bad_column_exits_one_not_two(capsys: pytest.CaptureFixture[str]) -> None:
    """The distinction a CI gate depends on: broken input is not a statistical finding."""
    code = main(
        ["audit", str(DATA / "support_eval.csv"), "--judge", "nope", "--gold", "human_passed"]
    )
    assert code == EXIT_ERROR
    assert "available columns" in capsys.readouterr().err


def test_a_missing_file_exits_one(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["audit", "definitely_absent.csv", "--judge", "j", "--gold", "g"])
    assert code == EXIT_ERROR
    assert "no such file" in capsys.readouterr().err


def test_compare_rejects_misaligned_gold_columns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Paired correction requires both systems labeled on the same rows."""
    path = tmp_path / "misaligned.csv"
    path.write_text("a_j,b_j,a_g,b_g\n1,1,1,\n0,1,,0\n1,0,1,1\n", encoding="utf-8")
    code = main(
        [
            "compare",
            str(path),
            "--judge-a",
            "a_j",
            "--judge-b",
            "b_j",
            "--gold-a",
            "a_g",
            "--gold-b",
            "b_g",
        ]
    )
    assert code == EXIT_ERROR
    assert "same rows" in capsys.readouterr().err
