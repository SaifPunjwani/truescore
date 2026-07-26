"""The judge changed and nobody shipped anything.

Judges are hosted models behind a version-less endpoint. When the provider updates one,
every eval number moves with no change in your code, and the shift gets attributed to
whatever release happened to be nearby.

The defense is an anchor set: a frozen sample of examples with trusted labels, re-run
through the judge on a schedule. ``data/judge_anchor.csv`` holds 600 such examples scored
by the judge in May and again in June.

Command-line equivalent, suitable for a scheduled CI job:

    truescore drift examples/data/judge_anchor.csv \\
        --baseline judge_may --current judge_june --gold human_passed

It exits 2 when the judge has changed, so a pipeline can fail the build on it.

Run: python examples/04_detect_judge_drift.py
"""

from __future__ import annotations

from pathlib import Path

from truescore.drift import judge_drift
from truescore.io import read_rows


def main() -> None:
    rows = read_rows(Path(__file__).parent / "data" / "judge_anchor.csv")
    baseline = [int(row["judge_may"]) for row in rows]
    current = [int(row["judge_june"]) for row in rows]
    gold = [int(row["human_passed"]) for row in rows]
    ids = [row["example_id"] for row in rows]

    report = judge_drift(baseline, current, gold, example_ids=ids)
    print(report.summary())
    print()

    if report.agreement_changed:
        print(
            "The judge is measuring differently than it was in May. Every metric computed\n"
            "with it since then is on a different scale, and comparisons that straddle the\n"
            "change are comparing two instruments rather than two models."
        )
    elif report.behavior_changed:
        print(
            "Accuracy is unchanged, so a dashboard would show nothing -- but the judge\n"
            "rewrote individual verdicts, so any per-example analysis has shifted under you."
        )
    else:
        print("No detectable change: the judge is the same instrument it was.")

    print()
    print(
        "The anchor fingerprint above pins which examples were compared. Without it, a\n"
        "drift result is unfalsifiable: a changed anchor set produces exactly the same\n"
        "symptom as a changed judge."
    )


if __name__ == "__main__":
    main()
