"""The core demonstration: a lenient judge, and what correcting for it changes.

Simulates an evaluation of 2000 examples where a judge marks answers pass/fail and 200 of
them were also labeled by a human. The judge is realistic rather than adversarial: it
rarely misses a genuinely good answer (sensitivity 0.95) but frequently waves through a
bad one (specificity 0.60). That combination -- lenient, not random -- is the common case
in practice, and it is the one that inflates reported scores.

Run: python examples/judge_correction.py
"""

from __future__ import annotations

import numpy as np

import truescore as ts

TRUE_RATE = 0.70
SENSITIVITY = 0.95
SPECIFICITY = 0.60
N_TOTAL = 2000
N_GOLD = 200


def main() -> None:
    rng = np.random.default_rng(0)

    truth = rng.binomial(1, TRUE_RATE, N_TOTAL)
    judge = np.where(
        truth == 1,
        rng.binomial(1, SENSITIVITY, N_TOTAL),
        rng.binomial(1, 1.0 - SPECIFICITY, N_TOTAL),
    )
    gold_index = np.sort(rng.choice(N_TOTAL, N_GOLD, replace=False))
    gold = truth[gold_index]

    print(f"ground truth pass rate: {truth.mean():.4f}  (known only because this is a simulation)")
    print(
        f"judge: sensitivity {SENSITIVITY}, specificity {SPECIFICITY} -- "
        "lenient, so it over-reports"
    )
    print()

    report = ts.build_report(
        judge,
        gold,
        gold_index,
        system_name="assistant-v4",
        metric_name="pass rate",
        covariates=None,
    )
    print(report.summary())
    print()

    naive_covers = report.naive.low <= truth.mean() <= report.naive.high
    corrected_covers = report.corrected.low <= truth.mean() <= report.corrected.high
    print(f"naive interval contains the truth:     {naive_covers}")
    print(f"corrected interval contains the truth: {corrected_covers}")
    print()

    print("judge quality on the 200 human-labeled examples:")
    print(ts.judge_agreement(judge[gold_index], gold).summary())
    print()

    print("planning the next evaluation:")
    for target in (0.05, 0.02):
        plan = ts.required_gold_labels(
            N_TOTAL,
            target_half_width=target,
            true_rate=float(report.corrected.point),
            sensitivity=SENSITIVITY,
            specificity=SPECIFICITY,
        )
        print(f"\n  target ±{target:.2f}:")
        print("\n".join("  " + line for line in plan.summary().splitlines()))


if __name__ == "__main__":
    main()
