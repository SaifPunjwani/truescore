"""Audit a judge-scored evaluation: what is the pass rate, really?

Reads ``data/support_eval.csv`` -- 4000 support questions, every one scored by an LLM
judge, 400 of them also labeled by a human -- and produces the corrected pass rate, the
judge's error profile, and what the judge is biased by.

Command-line equivalent:

    truescore audit examples/data/support_eval.csv \\
        --judge judge_passed --gold human_passed \\
        --covariate response_tokens --system-name support-assistant-v4

Run: python examples/01_audit_an_eval.py
"""

from __future__ import annotations

from pathlib import Path

import truescore as ts
from truescore.io import load_labels

DATA = Path(__file__).parent / "data" / "support_eval.csv"
TRUE_PASS_RATE = 0.7140  # known because generate_sample_data.py made the data


def main() -> None:
    labels = load_labels(
        DATA,
        judge="judge_passed",
        gold="human_passed",
        id_column="example_id",
        covariates=["response_tokens"],
    )
    print(labels.summary())
    print()

    # Scale length to hundreds of tokens so the reported effect is readable: "per 100
    # tokens" rather than a coefficient with four leading zeros.
    covariates = {"tokens_per_100": labels.covariates_on_gold()["response_tokens"] / 100.0}
    report = ts.build_report(
        labels.judge,
        labels.gold,
        labels.gold_index,
        system_name="support-assistant-v4",
        metric_name="pass rate",
        covariates=covariates,
    )
    print(report.summary())
    print()

    print(f"(the data was generated with a true pass rate of {TRUE_PASS_RATE:.4f})")
    naive_covers = report.naive.low <= TRUE_PASS_RATE <= report.naive.high
    corrected_covers = report.corrected.low <= TRUE_PASS_RATE <= report.corrected.high
    print(f"  judge-only interval contains the truth: {naive_covers}")
    print(f"  corrected interval contains the truth:  {corrected_covers}")
    print()

    assert report.agreement is not None
    print(report.agreement.summary())
    print()

    assert report.bias is not None
    print(report.bias.summary())
    print()
    print(
        "The judge is not merely noisy: its errors are concentrated on long answers, so\n"
        "any change that makes answers longer will read as a quality improvement."
    )

    output = Path(__file__).parent / "audit_report.md"
    output.write_text(report.to_markdown(), encoding="utf-8")
    print(f"\nwrote a shareable report to {output}")


if __name__ == "__main__":
    main()
