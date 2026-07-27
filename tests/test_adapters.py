"""Tests for truescore.adapters.

The fixtures below are not invented. Each field is taken from the tool's own
serialization code, read from the published package:

- inspect: ``EvalSample`` and ``Score`` in ``inspect_ai/log/_log.py`` and
  ``inspect_ai/scorer/_metric.py``; the "C"/"P"/"I"/"N" mapping is ``value_to_float``.
- promptfoo: ``createOutputData`` and ``toEvaluateSummary`` produce
  ``{evalId, results: {version: 3, results: [...]}}``; per-record fields come from
  ``EvalResult.toEvaluateResult``.
- deepeval: ``TestRun`` and ``LLMApiTestCase`` dumped with ``by_alias=True``, so
  ``testCases`` and ``metricsData``; ``MetricData`` supplies name/threshold/success/score.
- lm-eval: the ``example`` dict in ``lm_eval/evaluator.py``, followed by
  ``example.update(metrics)``.

A test that passes against a fixture the implementation also authored proves nothing, so
the value of these lives in that provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from truescore.adapters import detect_format, read_eval
from truescore.correct import ppi_estimate
from truescore.io import join_gold, load_labels

INSPECT_LOG = {
    "version": 2,
    "status": "success",
    "eval": {"task": "support_qa", "model": "openai/gpt-4o", "run_id": "abc"},
    "plan": {"name": "plan"},
    "samples": [
        {
            "id": "q1",
            "epoch": 1,
            "input": "How do I get a refund?",
            "target": "Refunds take 5 days.",
            "output": {"completion": "You can request a refund in settings."},
            "scores": {
                "model_graded_qa": {
                    "value": "C",
                    "answer": "correct",
                    "explanation": "matches the target",
                }
            },
            "metadata": {"segment": "billing", "turns": 2},
        },
        {
            "id": "q2",
            "epoch": 1,
            "input": "Reset my password?",
            "target": "Use the reset link.",
            "output": {"completion": "Click forgot password."},
            "scores": {"model_graded_qa": {"value": "I", "answer": "wrong"}},
            "metadata": {"segment": "account", "turns": 1},
        },
        {
            "id": "q3",
            "epoch": 1,
            "input": "Cancel subscription?",
            "target": "Cancel in billing.",
            "output": {"completion": "Go to billing, then cancel, then confirm."},
            "scores": {"model_graded_qa": {"value": "P"}},
            "metadata": {"segment": "billing", "turns": 3},
        },
        {
            "id": "q4",
            "epoch": 1,
            "input": "Where is my invoice?",
            "target": "In billing.",
            "output": {"completion": ""},
            "scores": {"model_graded_qa": {"value": "N"}},
            "metadata": {"segment": "billing", "turns": 1},
        },
    ],
}

PROMPTFOO_OUTPUT = {
    "evalId": "eval-2026-07-27",
    "results": {
        "version": 3,
        "timestamp": "2026-07-27T00:00:00.000Z",
        "prompts": [{"raw": "Answer: {{topic}}", "label": "p1"}],
        "results": [
            {
                "id": "r1",
                "promptIdx": 0,
                "testIdx": 0,
                "score": 1,
                "success": True,
                "namedScores": {"llm-rubric": 1},
                "gradingResult": {"pass": True, "score": 1, "reason": "meets the rubric"},
                "provider": {"id": "openai:gpt-4o", "label": "gpt-4o"},
                "response": {"output": "A refund takes five days."},
                "testCase": {"vars": {"topic": "billing"}},
                "latencyMs": 812,
                "cost": 0.0012,
            },
            {
                "id": "r2",
                "promptIdx": 0,
                "testIdx": 1,
                "score": 0,
                "success": False,
                "namedScores": {"llm-rubric": 0},
                "gradingResult": {"pass": False, "score": 0, "reason": "off topic"},
                "provider": {"id": "openai:gpt-4o", "label": "gpt-4o"},
                "response": {"output": "I cannot help."},
                "testCase": {"vars": {"topic": "account"}},
                "latencyMs": 640,
                "cost": 0.0009,
            },
        ],
        "stats": {"successes": 1, "failures": 1},
    },
    "config": {"providers": ["openai:gpt-4o"]},
    "shareableUrl": None,
}

DEEPEVAL_RUN = {
    "testFile": "test_support.py",
    "testCases": [
        {
            "name": "test_case_0",
            "input": "How do I get a refund?",
            "actualOutput": "Refunds take five days.",
            "success": True,
            "order": 0,
            "metricsData": [
                {
                    "name": "Answer Relevancy",
                    "threshold": 0.7,
                    "success": True,
                    "score": 0.91,
                    "reason": "on topic",
                    "evaluationModel": "gpt-4o",
                }
            ],
        },
        {
            "name": "test_case_1",
            "input": "Reset my password?",
            "actualOutput": "No.",
            "success": False,
            "order": 1,
            "metricsData": [
                {
                    "name": "Answer Relevancy",
                    "threshold": 0.7,
                    "success": False,
                    "score": 0.22,
                    "reason": "unhelpful",
                    "evaluationModel": "gpt-4o",
                }
            ],
        },
    ],
    "metricsScores": [{"metric": "Answer Relevancy", "score": 0.565}],
    "runDuration": 4.2,
}

LM_EVAL_SAMPLES = [
    {
        "doc_id": 0,
        "doc": {"question": "2+2?", "category": "arithmetic"},
        "target": "4",
        "arguments": [["Q: 2+2?\nA:", {}]],
        "resps": [["4"]],
        "filtered_resps": ["4"],
        "filter": "none",
        "metrics": ["exact_match"],
        "doc_hash": "d0",
        "prompt_hash": "p0",
        "target_hash": "t0",
        "exact_match": 1.0,
    },
    {
        "doc_id": 1,
        "doc": {"question": "3*3?", "category": "arithmetic"},
        "target": "9",
        "arguments": [["Q: 3*3?\nA:", {}]],
        "resps": [["6"]],
        "filtered_resps": ["6"],
        "filter": "none",
        "metrics": ["exact_match"],
        "doc_hash": "d1",
        "prompt_hash": "p1",
        "target_hash": "t1",
        "exact_match": 0.0,
    },
]


def _write(path: Path, obj: object) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_reads_an_inspect_eval_log(tmp_path: Path) -> None:
    found = read_eval(_write(tmp_path / "log.json", INSPECT_LOG))

    assert found.tool == "inspect"
    assert len(found.rows) == 4
    assert found.judge_columns == ("score.model_graded_qa",)
    assert [row["id"] for row in found.rows] == ["q1", "q2", "q3", "q4"]


def test_inspect_score_letters_map_the_way_inspect_maps_them(tmp_path: Path) -> None:
    """C, P, I and N become 1.0, 0.5, 0.0 and 0.0, as value_to_float defines them.

    Reproducing the tool's own mapping is what makes a corrected number comparable to the
    accuracy the tool reported. Inventing a different one for "P" would silently shift
    every score built on partial credit.
    """
    found = read_eval(_write(tmp_path / "log.json", INSPECT_LOG))

    values = [row["score.model_graded_qa"] for row in found.rows]
    assert values == [1.0, 0.0, 0.5, 0.0]


def test_inspect_metadata_becomes_covariates_and_segments(tmp_path: Path) -> None:
    found = read_eval(_write(tmp_path / "log.json", INSPECT_LOG))

    assert "meta.segment" in found.segments
    assert "meta.turns" in found.covariates
    assert "response_chars" in found.covariates
    assert found.rows[0]["meta.segment"] == "billing"


def test_inspect_log_without_samples_is_rejected(tmp_path: Path) -> None:
    """An aggregate-only log cannot be corrected, and says so instead of loading empty."""
    log = {**INSPECT_LOG, "samples": []}

    with pytest.raises(ValueError, match="no samples"):
        read_eval(_write(tmp_path / "log.json", log))


def test_reads_a_promptfoo_output_file(tmp_path: Path) -> None:
    found = read_eval(_write(tmp_path / "out.json", PROMPTFOO_OUTPUT))

    assert found.tool == "promptfoo"
    assert len(found.rows) == 2
    assert found.judge_column == "success"
    assert found.rows[0]["success"] is True
    assert found.rows[0]["grading_pass"] is True
    assert found.rows[0]["var.topic"] == "billing"
    assert found.rows[0]["response_chars"] == len("A refund takes five days.")


def test_promptfoo_named_scores_are_offered_after_the_overall_verdict(
    tmp_path: Path,
) -> None:
    """A per-assertion score is a judge verdict too, but not the one a caller means first."""
    found = read_eval(_write(tmp_path / "out.json", PROMPTFOO_OUTPUT))

    assert found.judge_columns[0] == "success"
    assert "named.llm_rubric" in found.judge_columns


def test_reads_promptfoo_records_without_the_wrapper(tmp_path: Path) -> None:
    """--output out.jsonl writes bare records, with no results wrapper around them."""
    path = tmp_path / "out.jsonl"
    records = PROMPTFOO_OUTPUT["results"]["results"]  # type: ignore[index]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    found = read_eval(path)

    assert found.tool == "promptfoo"
    assert len(found.rows) == 2


def test_reads_a_deepeval_test_run(tmp_path: Path) -> None:
    found = read_eval(_write(tmp_path / "run.json", DEEPEVAL_RUN))

    assert found.tool == "deepeval"
    assert found.judge_column == "success"
    assert "metric.answer_relevancy.success" in found.judge_columns
    assert found.rows[0]["metric.answer_relevancy.score"] == 0.91


def test_reads_lm_eval_samples(tmp_path: Path) -> None:
    path = tmp_path / "samples_arith.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in LM_EVAL_SAMPLES), encoding="utf-8")

    found = read_eval(path)

    assert found.tool == "lm-eval"
    assert found.judge_columns == ("exact_match",)
    assert [row["exact_match"] for row in found.rows] == [1.0, 0.0]
    assert found.rows[0]["doc.category"] == "arithmetic"


def test_generic_input_is_returned_untouched(tmp_path: Path) -> None:
    """Unrecognized rows keep working exactly as they did before adapters existed."""
    rows = [{"judge": 1, "human": 1}, {"judge": 0, "human": ""}]
    path = tmp_path / "plain.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    found = read_eval(path)

    assert found.tool == "generic"
    assert found.rows == rows
    assert found.judge_columns == ()


def test_a_file_that_borrows_one_key_name_stays_generic(tmp_path: Path) -> None:
    """One shared key is a coincidence, not a format.

    A hand-rolled JSONL with a ``gradingResult`` object must keep its own column names,
    because claiming it would rename the columns out from under whoever wrote it.
    """
    rows = [{"gradingResult": {"pass": True}, "human": 1} for _ in range(3)]
    path = tmp_path / "mine.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    found = read_eval(path)

    assert found.tool == "generic"
    assert found.rows[0]["gradingResult"] == {"pass": True}


def test_hand_added_columns_survive_normalization(tmp_path: Path) -> None:
    """A human verdict pasted into the eval file is the most valuable column in it.

    Normalization drops nothing it does not recognize, so pasting a ``human`` column into
    a promptfoo export works without a separate label file.
    """
    output = json.loads(json.dumps(PROMPTFOO_OUTPUT))
    output["results"]["results"][0]["human"] = 1
    output["results"]["results"][1]["human"] = 1

    found = read_eval(_write(tmp_path / "out.json", output))

    assert [row["human"] for row in found.rows] == [1, 1]
    labels = load_labels(found.rows, judge="success", gold="human")
    assert labels.n_gold == 2


def test_an_unrecognized_json_object_names_the_tools_it_looked_for(tmp_path: Path) -> None:
    path = _write(tmp_path / "mystery.json", {"summary": {"score": 0.8}})

    with pytest.raises(ValueError, match="promptfoo"):
        read_eval(path)


def test_no_adapter_puts_a_nested_value_in_a_row(tmp_path: Path) -> None:
    """Rows must stay flat and scalar, since every downstream reader assumes that.

    A dict or list leaking into a row would reach the estimators as an unparseable
    verdict, and the failure would surface far from its cause.
    """
    fixtures = [
        _write(tmp_path / "a.json", INSPECT_LOG),
        _write(tmp_path / "b.json", PROMPTFOO_OUTPUT),
        _write(tmp_path / "c.json", DEEPEVAL_RUN),
    ]
    for path in fixtures:
        for row in read_eval(path).rows:
            for name, value in row.items():
                assert isinstance(value, (str, int, float, bool)), f"{path.name}:{name}"


def test_summary_reports_the_tool_and_the_columns(tmp_path: Path) -> None:
    found = read_eval(_write(tmp_path / "out.json", PROMPTFOO_OUTPUT))

    text = found.summary()
    assert "promptfoo" in text
    assert "success" in text


def test_detect_format_accepts_already_parsed_json() -> None:
    found = detect_format(INSPECT_LOG, source="in memory")

    assert found.tool == "inspect"
    assert found.source == "in memory"


# ------------------------------------------------------------------------------------
# joining human labels, which live in a different file than the judge verdicts
# ------------------------------------------------------------------------------------


def test_join_gold_attaches_labels_by_identifier() -> None:
    rows = [{"id": "a", "judge": 1}, {"id": "b", "judge": 0}, {"id": "c", "judge": 1}]
    labels = [{"id": "a", "human": 1}, {"id": "c", "human": 0}]

    joined = join_gold(rows, labels, on="id", gold="human")

    assert joined.matched == 2
    assert joined.unmatched_gold == 0
    assert joined.rows[0]["gold"] == 1
    assert "gold" not in joined.rows[1]


def test_join_gold_rejects_a_join_key_that_matches_nothing() -> None:
    """A wrong key would otherwise produce a corrected number from zero human labels."""
    rows = [{"id": "a", "judge": 1}, {"id": "b", "judge": 0}]
    labels = [{"id": "row-1", "human": 1}]

    with pytest.raises(ValueError, match="no human label matched"):
        join_gold(rows, labels, on="id", gold="human")


def test_join_gold_rejects_duplicate_identifiers_in_the_eval_file() -> None:
    """One human label attaching to two rows would be counted twice."""
    rows = [{"id": "a", "judge": 1}, {"id": "a", "judge": 0}]
    labels = [{"id": "a", "human": 1}]

    with pytest.raises(ValueError, match="more than once"):
        join_gold(rows, labels, on="id", gold="human")


def test_join_gold_rejects_two_labels_for_one_example() -> None:
    rows = [{"id": "a", "judge": 1}]
    labels = [{"id": "a", "human": 1}, {"id": "a", "human": 0}]

    with pytest.raises(ValueError, match="more than once"):
        join_gold(rows, labels, on="id", gold="human")


def test_join_gold_counts_labels_that_landed_nowhere() -> None:
    """Labels for examples outside this run are reported, not silently discarded."""
    rows = [{"id": "a", "judge": 1}]
    labels = [{"id": "a", "human": 1}, {"id": "z", "human": 0}]

    joined = join_gold(rows, labels, on="id", gold="human")

    assert joined.matched == 1
    assert joined.unmatched_gold == 1
    assert "1 human labels matched no example" in joined.summary()


def test_join_gold_reads_labels_from_a_csv(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("example,verdict\na,1\nc,0\n", encoding="utf-8")
    rows = [{"id": "a", "judge": 1}, {"id": "b", "judge": 0}, {"id": "c", "judge": 1}]

    joined = join_gold(rows, path, on="id", gold="verdict", gold_on="example")

    assert joined.matched == 2


def test_promptfoo_output_and_a_label_csv_reach_an_estimate(tmp_path: Path) -> None:
    """The path this module exists for, end to end and with nothing reshaped by hand.

    A promptfoo run of 40 examples, 12 of them labeled in a separate spreadsheet, from two
    files to a corrected interval.
    """
    records = [
        {
            "id": f"r{i}",
            "promptIdx": 0,
            "testIdx": i,
            "score": float(i % 4 != 0),
            "success": i % 4 != 0,
            "gradingResult": {"pass": i % 4 != 0, "score": float(i % 4 != 0)},
            "provider": {"id": "openai:gpt-4o", "label": "gpt-4o"},
            "response": {"output": "x" * (20 + i)},
            "testCase": {"vars": {"topic": "billing" if i % 2 else "account"}},
        }
        for i in range(40)
    ]
    output = {"evalId": "e1", "results": {"version": 3, "results": records}}
    eval_path = _write(tmp_path / "promptfoo.json", output)

    label_path = tmp_path / "labels.csv"
    label_rows = ["example_id,human"]
    # The judge is right on ten of the twelve labeled examples and wrong on two, i=0 and
    # i=7, where the human verdict is flipped against it.
    label_rows += [f"r{i},{int((i % 4 != 0) != (i % 7 == 0))}" for i in range(12)]
    label_path.write_text("\n".join(label_rows) + "\n", encoding="utf-8")

    found = read_eval(eval_path)
    joined = join_gold(found.rows, label_path, on="id", gold="human", gold_on="example_id")
    labels = load_labels(
        joined.rows,
        judge=found.judge_column or "success",
        gold="gold",
        id_column="id",
        covariates=["response_chars"],
    )

    assert labels.n_total == 40
    assert labels.n_gold == 12

    estimate = ppi_estimate(labels.judge, labels.gold, labels.gold_index)
    assert 0.0 <= estimate.low <= estimate.point <= estimate.high <= 1.0
    assert estimate.n_total == 40
    assert estimate.n_gold == 12


def test_a_single_provider_run_offers_no_provider_segment(tmp_path: Path) -> None:
    """Every segment candidate goes through the same test, including provider.

    A run against one model has nothing to slice by, and doctor reports that column as
    constant. The two layers must agree, or one of them sends the reader to a command that
    cannot work.
    """
    found = read_eval(_write(tmp_path / "out.json", PROMPTFOO_OUTPUT))

    assert "provider" not in found.segments
    assert "var.topic" in found.segments


def test_multi_epoch_inspect_logs_collapse_to_one_row_per_sample(tmp_path: Path) -> None:
    """Five epochs of a sample are one draw seen five times, not five draws.

    Emitting a row per (sample, epoch) makes every downstream estimator divide the variance
    by five times more than the data supports. Measured at 86% coverage on a nominal 95%
    interval in tests/test_correct.py::test_clustered_data_undercovers_until_clusters_are_declared,
    which is why the adapter averages them here rather than leaving it to the caller.
    """
    samples = []
    for sample_id in range(20):
        for epoch in range(1, 6):
            samples.append(
                {
                    "id": f"q{sample_id}",
                    "epoch": epoch,
                    "target": "yes",
                    "output": {"completion": "x" * (10 + epoch)},
                    # 3 of 5 epochs correct, so the sample's mean score is 0.6
                    "scores": {"grader": {"value": "C" if epoch <= 3 else "I"}},
                    "metadata": {"segment": "billing"},
                }
            )
    log = {"eval": {"task": "t"}, "samples": samples}

    found = read_eval(_write(tmp_path / "log.json", log))

    assert len(found.rows) == 20, "one row per sample, not per sample-epoch"
    assert {row["id"] for row in found.rows} == {f"q{i}" for i in range(20)}
    assert all(row["epochs"] == 5 for row in found.rows)
    assert all(row["score.grader"] == pytest.approx(0.6) for row in found.rows)
    assert any("epochs per sample were averaged" in note for note in found.notes)


def test_single_epoch_logs_are_left_alone(tmp_path: Path) -> None:
    """The averaging must not fire on the ordinary case, or ids would gain a stray column."""
    found = read_eval(_write(tmp_path / "log.json", INSPECT_LOG))

    assert len(found.rows) == 4
    assert all("epochs" not in row for row in found.rows)
    assert not any("averaged" in note for note in found.notes)
