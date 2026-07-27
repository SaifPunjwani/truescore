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


def test_slices_flags_the_regressed_segment(capsys: pytest.CaptureFixture[str]) -> None:
    """The headline slice case, through the command line a CI job would run."""
    code = main(
        [
            "slices",
            str(DATA / "support_segments.csv"),
            "--by",
            "segment",
            "--judge-a",
            "v4_judge_passed",
            "--gold-a",
            "v4_human_passed",
            "--judge-b",
            "v3_judge_passed",
            "--gold-b",
            "v3_human_passed",
        ]
    )
    out = capsys.readouterr().out
    assert code == EXIT_FINDING
    assert "technical" in out
    assert "regressed on technical" in out


def test_slices_estimates_when_only_one_system_is_given(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "slices",
            str(DATA / "support_segments.csv"),
            "--by",
            "segment",
            "--judge-a",
            "v4_judge_passed",
            "--gold-a",
            "v4_human_passed",
        ]
    )
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "corrected" in out
    assert "judge said" in out


def test_audit_writes_an_html_report(tmp_path: Path) -> None:
    html_path = tmp_path / "report.html"
    main(
        [
            "audit",
            str(DATA / "support_eval.csv"),
            "--judge",
            "judge_passed",
            "--gold",
            "human_passed",
            "--html",
            str(html_path),
        ]
    )
    html = html_path.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "Corrected (use this)" in html


def test_audit_reads_promptfoo_output_and_a_separate_label_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The flow this tool has to support: the eval file the tool wrote, plus a spreadsheet.

    Human labels never arrive in the same file as judge verdicts. The eval tool writes one
    and a person fills in the other, so requiring a hand merge first is the step that stops
    people trying anything at all.
    """
    import json

    rng = np.random.default_rng(11)
    records = []
    for i in range(400):
        passed = bool(rng.random() < 0.8)
        records.append(
            {
                "id": f"case-{i}",
                "promptIdx": 0,
                "testIdx": i,
                "success": passed,
                "score": float(passed),
                "gradingResult": {"pass": passed, "score": float(passed)},
                "provider": {"id": "openai:gpt-4o", "label": "gpt-4o"},
                "response": {"output": "x" * int(rng.integers(20, 400))},
                "testCase": {"vars": {"topic": "billing" if i % 2 else "account"}},
            }
        )
    eval_path = tmp_path / "promptfoo-output.json"
    eval_path.write_text(
        json.dumps({"evalId": "e", "results": {"version": 3, "results": records}}),
        encoding="utf-8",
    )

    # A hundred of them get a human verdict; the judge is lenient, passing some the human
    # fails, which is the whole reason to correct the score.
    label_lines = ["example_id,human"]
    for i in range(100):
        judged = records[i]["success"]
        human = judged and not (rng.random() < 0.25)
        label_lines.append(f"case-{i},{int(human)}")
    label_path = tmp_path / "labels.csv"
    label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    code = main(
        [
            "audit",
            str(eval_path),
            "--gold-file",
            str(label_path),
            "--gold",
            "human",
            "--gold-id",
            "example_id",
            "--covariate",
            "response_chars",
        ]
    )
    out = capsys.readouterr().out

    assert code in (EXIT_OK, EXIT_FINDING)
    assert "promptfoo: 400 examples" in out
    assert "joined 100 human labels onto 400 examples" in out
    # --judge was never passed: the column came from recognizing the format.
    assert "judge verdict: success" in out
    assert "loaded 400 examples" in out
    assert "human-labeled: 100" in out


def test_audit_without_a_judge_column_on_an_unrecognized_file_says_what_to_do(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Inference is a convenience, so its failure has to point somewhere useful."""
    path = tmp_path / "plain.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    code = main(["audit", str(path), "--gold", "b"])

    assert code == EXIT_ERROR
    assert "truescore doctor" in capsys.readouterr().err


def test_gold_file_without_an_id_column_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "plain.csv"
    path.write_text("judge,x\n1,2\n", encoding="utf-8")
    labels = tmp_path / "l.csv"
    labels.write_text("id,human\n1,1\n", encoding="utf-8")

    code = main(
        ["audit", str(path), "--judge", "judge", "--gold", "human", "--gold-file", str(labels)]
    )

    assert code == EXIT_ERROR
    assert "needs --id-column" in capsys.readouterr().err
