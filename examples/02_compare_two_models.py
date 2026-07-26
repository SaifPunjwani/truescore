"""Did v4 actually beat v3, or did it just write longer answers?

This is the example that matters. A team runs both versions over the same 4000 questions,
the judge scores everything, and the dashboard shows v4 ahead by seventeen points. That is the
number that goes in the launch review.

The truth is a ten-point improvement -- a good release, worth shipping. The other seven
points are the judge rewarding v4 for writing longer answers, a preference that has
nothing to do with whether the answer helped the customer. Correcting for judge error
recovers the real difference, and the difference between those two numbers is the
difference between learning something about your model and learning something about your
judge.

Command-line equivalent:

    truescore compare examples/data/support_compare.csv \\
        --judge-a v4_judge_passed --judge-b v3_judge_passed \\
        --gold-a v4_human_passed --gold-b v3_human_passed

Run: python examples/02_compare_two_models.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import truescore as ts
from truescore.io import load_labels, read_rows

DATA = Path(__file__).parent / "data" / "support_compare.csv"
TRUE_IMPROVEMENT = 0.0900  # known because generate_sample_data.py made the data


def main() -> None:
    rows = read_rows(DATA)
    v4 = load_labels(rows, judge="v4_judge_passed", gold="v4_human_passed")
    v3 = load_labels(rows, judge="v3_judge_passed", gold="v3_human_passed")

    print(f"{v4.n_total} shared questions, {v4.n_gold} of them human-labeled for both versions")
    print(f"  judge says v3: {v3.judge.mean():.4f}")
    print(f"  judge says v4: {v4.judge.mean():.4f}")
    print()

    print("=" * 78)
    print("STEP 1 -- the comparison a normal eval pipeline reports")
    print("=" * 78)
    naive = ts.mcnemar(v4.judge.astype(int), v3.judge.astype(int))
    print(naive.summary())
    print(
        "\nPaired and statistically sound, and still wrong: it compares what the judge\n"
        "said, and the judge is not measuring the same thing for both versions."
    )
    print()

    print("=" * 78)
    print("STEP 2 -- the same comparison, corrected for judge error")
    print("=" * 78)
    corrected = ts.ppi_compare(v4.judge, v3.judge, v4.gold, v3.gold, v4.gold_index)
    print(corrected.summary())
    print()

    print("=" * 78)
    print("STEP 3 -- the verdict")
    print("=" * 78)
    inflation = naive.difference - corrected.difference
    print(f"  judge-reported improvement: {naive.difference:+.4f}")
    print(f"  corrected improvement:      {corrected.difference:+.4f}")
    print(
        f"  overstatement:              {inflation:+.4f}  "
        f"(the judge's number is {naive.difference / corrected.difference:.1f}x the real gain)"
    )
    print(f"  (true improvement in the generated data: {TRUE_IMPROVEMENT:+.4f})")
    print()

    print("Why: the judge rewards length, and v4 writes longer answers.")
    lengths_v4 = np.array([float(row["v4_tokens"]) for row in rows])
    lengths_v3 = np.array([float(row["v3_tokens"]) for row in rows])
    print(f"  median response length v3: {np.median(lengths_v3):.0f} tokens")
    print(f"  median response length v4: {np.median(lengths_v4):.0f} tokens")

    length_effect = ts.length_bias(
        v4.judge[v4.gold_index], v4.gold, lengths_v4[v4.gold_index], per=100.0
    )
    print(f"  judge length bias: {length_effect}")
    print()
    print(
        "v4 is genuinely better and worth shipping. But a launch review that believed the\n"
        "judge's number would be crediting the model for verbosity as much as for quality,\n"
        "and would keep paying for length on every release after this one."
    )


if __name__ == "__main__":
    main()
