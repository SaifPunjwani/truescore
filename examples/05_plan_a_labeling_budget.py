"""How many human labels do we actually need?

The question every eval program eventually reaches, usually after someone has already
spent the money. Human labeling is the dominant cost of trustworthy evaluation, and the
right amount to buy depends on how good the judge is: a strong judge does most of the work
and needs only a few labels to anchor it, while a weak one needs many.

This example turns that into a number, and prices the alternative -- labeling without a
judge at all -- so the judge's contribution is expressed in human labels avoided.

Command-line equivalent:

    truescore plan --n-total 4000 --target 0.02 --rate 0.71 \\
        --sensitivity 0.98 --specificity 0.56

Run: python examples/05_plan_a_labeling_budget.py
"""

from __future__ import annotations

from pathlib import Path

from truescore.agreement import judge_agreement
from truescore.io import load_labels
from truescore.power import min_detectable_effect, required_gold_labels, required_pairs

DATA = Path(__file__).parent / "data" / "support_eval.csv"


def main() -> None:
    labels = load_labels(DATA, judge="judge_passed", gold="human_passed")
    measured = judge_agreement(labels.judge[labels.gold_index].astype(int), labels.gold.astype(int))
    sensitivity = measured.sensitivity.point
    specificity = measured.specificity.point

    print("measured from the labels already collected:")
    print(f"  judge sensitivity {sensitivity:.4f}, specificity {specificity:.4f}")
    print(
        f"  (a lenient judge: it almost never fails a correct answer, and passes "
        f"{1 - specificity:.0%} of wrong ones)"
    )
    print()

    print("=" * 78)
    print("how many labels for a given precision, at this judge quality")
    print("=" * 78)
    for target in (0.05, 0.03, 0.02):
        plan = required_gold_labels(
            labels.n_total,
            target_half_width=target,
            true_rate=0.71,
            sensitivity=sensitivity,
            specificity=specificity,
        )
        if plan.feasible:
            print(
                f"  +/-{target:.2f}: {plan.required_gold:>5} labels "
                f"(without the judge: {plan.gold_only_required:>5}, "
                f"saving {plan.labels_saved})"
            )
        else:
            print(
                f"  +/-{target:.2f}: not reachable with {labels.n_total} examples; "
                f"best is +/-{plan.achieved_half_width:.4f}"
            )
    print()

    print("=" * 78)
    print("what a better judge would be worth")
    print("=" * 78)
    for name, sens, spec in (
        ("today's judge", sensitivity, specificity),
        ("a stricter judge", 0.95, 0.85),
        ("a near-perfect judge", 0.98, 0.97),
    ):
        plan = required_gold_labels(
            labels.n_total,
            target_half_width=0.03,
            true_rate=0.71,
            sensitivity=sens,
            specificity=spec,
        )
        detail = f"{plan.required_gold:>5} labels" if plan.feasible else "not reachable"
        print(f"  {name:<22} {detail}")
    print(
        "\n  Improving the judge is not a cosmetic exercise: it converts directly into\n"
        "  human hours you do not have to buy for every future evaluation."
    )
    print()

    print("=" * 78)
    print("is the evaluation set big enough to detect the change we care about?")
    print("=" * 78)
    for n_examples in (200, 1000, 4000):
        mde = min_detectable_effect(n_examples, discordance_rate=0.25)
        print(f"  {n_examples:>5} examples resolve differences down to {mde:.4f}")
    needed = required_pairs(0.02, discordance_rate=0.25)
    print(f"\n  detecting a two-point change would take {needed} examples.")
    print(
        "  A 200-example eval cannot support a claim about a two-point difference, no\n"
        "  matter which system scored higher on the day."
    )


if __name__ == "__main__":
    main()
