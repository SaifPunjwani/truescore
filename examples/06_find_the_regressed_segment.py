"""v4 is better overall. Is it better for everyone?

The launch review approves v4 because it wins overall. Six weeks later, technical-support
tickets are escalating and nobody can say why.

This is the failure mode: the new version improved on two segments and regressed on the
third, and the judge could not see it, because the segment it regressed on is the one where
its answers got longest -- and the judge rewards length. On that segment the judge reports a
seventeen-point *improvement* where the truth is a nine-point *regression*. The sign is
inverted, not merely the magnitude.

Two things are needed to recover it, and both are here: correcting each segment separately
(the judge's bias differs by segment, so one global correction would misrank them) and
adjusting for having asked about several segments at once (twenty slices tested at 5%
produce a spurious winner roughly once per run).

Command-line equivalent:

    truescore slices examples/data/support_segments.csv --by segment \\
        --judge-a v4_judge_passed --judge-b v3_judge_passed \\
        --gold-a v4_human_passed --gold-b v3_human_passed

Run: python examples/06_find_the_regressed_segment.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from truescore.compare import mcnemar, ppi_compare
from truescore.io import load_labels, read_rows
from truescore.slices import compare_slices, counts_by_slice

DATA = Path(__file__).parent / "data" / "support_segments.csv"


def main() -> None:
    rows = read_rows(DATA)
    v4 = load_labels(rows, judge="v4_judge_passed", gold="v4_human_passed")
    v3 = load_labels(rows, judge="v3_judge_passed", gold="v3_human_passed")
    segments = np.asarray([row["segment"] for row in rows])

    print(
        f"{v4.n_total} questions across {len(counts_by_slice(segments))} segments, "
        f"{v4.n_gold} human-labeled"
    )
    for name, count in sorted(counts_by_slice(segments).items()):
        print(f"  {name:<12} {count}")
    print()

    print("=" * 78)
    print("STEP 1 -- the overall verdict, which is what the launch review sees")
    print("=" * 78)
    overall_judged = mcnemar(v4.judge.astype(int), v3.judge.astype(int))
    overall_corrected = ppi_compare(v4.judge, v3.judge, v4.gold, v3.gold, v4.gold_index)
    print(f"  as judged:  {overall_judged.difference:+.4f}")
    print(
        f"  corrected:  {overall_corrected.difference:+.4f} "
        f"[{overall_corrected.low:+.4f}, {overall_corrected.high:+.4f}]"
    )
    print("  v4 wins overall on both readings. Ship it.")
    print()

    print("=" * 78)
    print("STEP 2 -- per segment, as the judge sees it")
    print("=" * 78)
    for name in sorted(set(segments)):
        mask = segments == name
        gap = v4.judge[mask].mean() - v3.judge[mask].mean()
        print(f"  {name:<12} {gap:+.4f}")
    print("  Every segment improved. There is nothing here to investigate.")
    print()

    print("=" * 78)
    print("STEP 3 -- per segment, corrected for judge error")
    print("=" * 78)
    report = compare_slices(
        v4.judge,
        v3.judge,
        v4.gold,
        v3.gold,
        v4.gold_index,
        segments,
        by="segment",
        correction="holm",
    )
    print(report.summary())
    print()

    regressions = [c for c in report.comparisons if c.significant and c.difference < 0]
    if regressions:
        for regression in regressions:
            print(
                f"FINDING: v4 regressed on '{regression.name}' by "
                f"{abs(regression.difference):.4f} "
                f"[{abs(regression.high):.4f}, {abs(regression.low):.4f}], "
                f"adjusted p = {regression.adjusted_p_value:.4g}."
            )
        print(
            "\nThe judge reported an improvement on that segment. It reported an\n"
            "improvement because the answers got longer, and it rewards length. The\n"
            "overall number was real and the decision to ship may still be right --\n"
            "but shipping it without a mitigation for that segment is a choice the\n"
            "launch review never knew it was making."
        )
    else:
        print("No segment regressed at the adjusted level.")


if __name__ == "__main__":
    main()
