"""Tests for truescore.report."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.simulate import simulate_trial
from truescore.compare import mcnemar
from truescore.report import build_report

FIXED_TIMESTAMP = "2026-07-25T00:00:00+00:00"


def _report(seed: int = 70, **kwargs: object) -> object:
    rng = np.random.default_rng(seed)
    trial = simulate_trial(
        rng, n_total=2000, n_gold=200, true_rate=0.70, sensitivity=0.95, specificity=0.60
    )
    return build_report(
        trial.judge,
        trial.gold,
        trial.gold_index,
        system_name="assistant-v4",
        timestamp=FIXED_TIMESTAMP,
        **kwargs,  # type: ignore[arg-type]
    )


def test_report_records_all_three_estimates() -> None:
    """The naive, corrected, and gold-only numbers appear together, which is the point."""
    report = _report()
    assert report.n_total == 2000  # type: ignore[attr-defined]
    assert report.n_gold == 200  # type: ignore[attr-defined]
    assert report.corrected.method == "ppi++"  # type: ignore[attr-defined]
    assert report.naive.point > report.corrected.point  # type: ignore[attr-defined]


def test_report_quantifies_the_naive_error() -> None:
    """A lenient judge (sens 0.95, spec 0.60) inflates the naive number by several points."""
    report = _report()
    assert report.naive_error > 0.05  # type: ignore[attr-defined]


def test_report_round_trips_through_json() -> None:
    """The artifact serializes losslessly enough to be stored and re-read."""
    report = _report()
    payload = json.loads(report.to_json())  # type: ignore[attr-defined]

    assert payload["system_name"] == "assistant-v4"
    assert payload["created_utc"] == FIXED_TIMESTAMP
    assert payload["corrected"]["method"] == "ppi++"
    assert payload["corrected"]["assumptions"], "assumptions must survive serialization"
    assert isinstance(payload["naive"]["point"], float)


def test_report_markdown_contains_corrected_and_naive() -> None:
    """The rendered artifact states the corrected number, the naive one, and the caveats."""
    text = _report().to_markdown()  # type: ignore[attr-defined]
    for expected in (
        "# Evaluation report: assistant-v4",
        "Corrected (use this)",
        "Judge-only (conventional)",
        "## Assumptions",
        "## What this report does not establish",
    ):
        assert expected in text


def test_report_includes_agreement_and_bias_sections_when_available() -> None:
    """Supplying covariates adds the bias section; binary labels add the judge-quality one."""
    rng = np.random.default_rng(71)
    trial = simulate_trial(
        rng, n_total=1500, n_gold=250, true_rate=0.6, sensitivity=0.9, specificity=0.7
    )
    lengths = rng.uniform(50.0, 500.0, 250)
    report = build_report(
        trial.judge,
        trial.gold,
        trial.gold_index,
        covariates={"length": lengths},
        timestamp=FIXED_TIMESTAMP,
    )
    assert report.agreement is not None
    assert report.bias is not None
    text = report.to_markdown()
    assert "## Judge quality" in text
    assert "## Judge bias" in text


def test_report_carries_an_attached_comparison() -> None:
    """A comparison against another system is rendered in its own section."""
    rng = np.random.default_rng(72)
    trial = simulate_trial(
        rng, n_total=800, n_gold=120, true_rate=0.6, sensitivity=0.9, specificity=0.8
    )
    other = rng.binomial(1, 0.55, 800)
    comparison = mcnemar(trial.judge, other)
    report = build_report(
        trial.judge,
        trial.gold,
        trial.gold_index,
        comparison=comparison,
        timestamp=FIXED_TIMESTAMP,
    )
    assert report.comparison is not None
    assert "## Comparison" in report.to_markdown()


def test_report_omits_agreement_for_continuous_scores() -> None:
    """Continuous judge scores have no confusion matrix; the report says less, not wrong."""
    rng = np.random.default_rng(73)
    judge = rng.uniform(0.0, 1.0, 500)
    index = np.sort(rng.choice(500, 80, replace=False))
    gold = np.clip(judge[index] + rng.normal(0.0, 0.1, 80), 0.0, 1.0)

    report = build_report(judge, gold, index, timestamp=FIXED_TIMESTAMP, metric_name="mean rating")
    assert report.agreement is None
    assert report.corrected.method == "ppi++"
    assert "mean rating" in report.to_markdown()


def test_report_summary_is_readable() -> None:
    text = _report().summary()  # type: ignore[attr-defined]
    assert "corrected:" in text
    assert "judge-only:" in text
    assert "off by" in text


def test_report_timestamp_defaults_to_now() -> None:
    """Omitting the timestamp stamps the artifact rather than leaving it blank."""
    rng = np.random.default_rng(74)
    trial = simulate_trial(
        rng, n_total=400, n_gold=60, true_rate=0.5, sensitivity=0.9, specificity=0.9
    )
    report = build_report(trial.judge, trial.gold, trial.gold_index)
    assert report.created_utc.startswith("20")
    assert report.created_utc != ""


def test_json_is_free_of_numpy_scalars() -> None:
    """numpy types would break json.dumps for a consumer; the converter must catch them."""
    report = _report()
    text = report.to_json()  # type: ignore[attr-defined]
    assert "np.float" not in text
    parsed = json.loads(text)
    assert isinstance(parsed["corrected"]["low"], float)


def test_report_requires_a_valid_alpha() -> None:
    rng = np.random.default_rng(75)
    trial = simulate_trial(
        rng, n_total=300, n_gold=50, true_rate=0.5, sensitivity=0.9, specificity=0.9
    )
    with pytest.raises(ValueError, match="alpha must lie"):
        build_report(trial.judge, trial.gold, trial.gold_index, alpha=1.5)


def test_html_report_is_self_contained_and_escaped() -> None:
    """A report gets emailed and opened years later, so it must not fetch anything.

    It also has to escape whatever the caller passed as a name: a report is a document,
    not a template.
    """
    rng = np.random.default_rng(80)
    trial = simulate_trial(
        rng, n_total=800, n_gold=120, true_rate=0.7, sensitivity=0.95, specificity=0.6
    )
    report = build_report(
        trial.judge,
        trial.gold,
        trial.gold_index,
        system_name='<script>alert("x")</script>',
        timestamp=FIXED_TIMESTAMP,
    )
    html = report.to_html()

    assert html.startswith("<!DOCTYPE html>")
    for external in ("http://", "https://", "<script", "src="):
        assert external not in html, f"report should not reference {external!r}"
    assert "&lt;script&gt;" in html, "the system name must be escaped"
    assert "Corrected (use this)" in html
    assert "What this report does not establish" in html


def test_html_report_includes_optional_sections() -> None:
    rng = np.random.default_rng(81)
    trial = simulate_trial(
        rng, n_total=900, n_gold=200, true_rate=0.6, sensitivity=0.9, specificity=0.7
    )
    lengths = rng.uniform(50.0, 500.0, 200)
    html = build_report(
        trial.judge,
        trial.gold,
        trial.gold_index,
        covariates={"length": lengths},
        timestamp=FIXED_TIMESTAMP,
    ).to_html()
    assert "Judge quality" in html
    assert "Judge bias" in html
