# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Saif Punjwani
"""Does GPT-4, used as a judge, favour its own answers? Measured on public data.

MT-Bench published both GPT-4's pairwise judgments and human judgments over the same
comparisons, which makes it one of the few public datasets where a judge's error can be
measured rather than assumed. This script joins the two, reports how far the judge-reported
win rate sits from the human one for each model, separates two competing explanations for
the gap, and checks that truescore recovers the human number from a fraction of the labels.

The data is downloaded at run time from lmsys/mt_bench_human_judgments (CC-BY-4.0). No
model is called and nothing is cached outside the working directory.

    pip install truescore pandas pyarrow
    python run.py

Every number in FINDINGS.md is printed by this script.
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np
import pandas as pd

from truescore.agreement import judge_agreement
from truescore.bias import judge_error_regression
from truescore.correct import gold_only_estimate, judge_only_estimate, ppi_estimate

BASE = (
    "https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/"
    "resolve/refs%2Fconvert%2Fparquet/default"
)
SCORE = {"model_a": 1.0, "tie": 0.5, "model_b": 0.0}
KEY = ["qid", "turn", "ma", "mb"]


def load() -> pd.DataFrame:
    """Join the human and GPT-4 splits into one row per comparison.

    The two splits list the same comparison with the models in either order, so each row is
    put in a canonical order and its verdict flipped to match. MT-Bench's
    "tie (inconsistent)" means the judge was run in both presentation orders and disagreed
    with itself, which is a tie for our purposes and is folded in.
    """

    def normalize(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for row in frame.itertuples():
            a, b, winner = row.model_a, row.model_b, row.winner
            if a > b:
                a, b = b, a
                winner = {"model_a": "model_b", "model_b": "model_a"}.get(winner, winner)
            rows.append((row.question_id, row.turn, a, b, "tie" if "tie" in winner else winner))
        return pd.DataFrame(rows, columns=[*KEY, "w"])

    human = normalize(pd.read_parquet(f"{BASE}/human/0000.parquet"))
    judge = normalize(pd.read_parquet(f"{BASE}/gpt4_pair/0000.parquet"))

    def majority(votes: pd.Series) -> str:
        counts = Counter(votes).most_common()
        if len(counts) > 1 and counts[0][1] == counts[1][1]:
            return "tie"  # a split panel is not evidence for either side
        return str(counts[0][0])

    resolved = human.groupby(KEY)["w"].apply(majority).rename("human").reset_index()
    single = judge.drop_duplicates(KEY).set_index(KEY)["w"].rename("judge").reset_index()
    return resolved.merge(single, on=KEY, how="inner")


def oriented(data: pd.DataFrame, model: str) -> tuple[np.ndarray, np.ndarray]:
    """Judge and human outcomes for one model, oriented so 1.0 means that model won."""
    subset = data[(data.ma == model) | (data.mb == model)]
    flip = (subset.mb == model).values
    judge = subset.judge.map(SCORE).values
    human = subset.human.map(SCORE).values
    return np.where(flip, 1 - judge, judge), np.where(flip, 1 - human, human)


def leave_one_out_strength(data: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each model's human win rate, computed without the comparison being explained.

    The naive version computes a model's strength from every comparison including the one
    whose error is the outcome, so the covariate contains the outcome. The induced
    correlation runs opposite to the effect reported here, meaning it would understate
    rather than manufacture it, but leaving the row out removes the question entirely.
    """
    strength: dict[str, np.ndarray] = {}
    for model in sorted(set(data.ma) | set(data.mb)):
        involved = ((data.ma == model) | (data.mb == model)).values
        _, human = oriented(data, model)
        total, count = human.sum(), len(human)
        # For rows involving this model, drop that row's own contribution.
        per_row = np.full(len(data), total / count)
        per_row[involved] = (total - human) / (count - 1)
        strength[model] = per_row
    return strength


def main() -> int:
    data = load()
    models = sorted(set(data.ma) | set(data.mb))
    print(f"comparisons carrying both a GPT-4 and a human verdict: {len(data)}")

    judge_all = data.judge.map(SCORE).values
    human_all = data.human.map(SCORE).values
    decisive = (data.judge != "tie") & (data.human != "tie")
    print(f"three-way agreement including ties:  {(data.judge == data.human).mean():.4f}")
    print(
        f"agreement where neither called a tie: "
        f"{(data.judge[decisive] == data.human[decisive]).mean():.4f} "
        f"(n={int(decisive.sum())})"
    )

    print("\n1. Reported win rate against the human one, per model")
    print(f"{'model':<17}{'n':>5}{'GPT-4 judge':>13}{'human':>8}{'gap':>9}  95% CI on the gap")
    gaps = {}
    for model in models:
        judge, human = oriented(data, model)
        difference = judge - human
        half = 1.96 * difference.std(ddof=1) / np.sqrt(len(difference))
        gaps[model] = (difference.mean(), difference.mean() - half, difference.mean() + half)
        print(
            f"{model:<17}{len(judge):>5}{judge.mean():>13.4f}{human.mean():>8.4f}"
            f"{difference.mean():>+9.4f}  [{difference.mean() - half:+.4f}, "
            f"{difference.mean() + half:+.4f}]"
        )

    print("\n2. Two explanations for that gap, fitted together")
    strength = leave_one_out_strength(data)
    spread = np.array(
        [
            strength[a][i] - strength[b][i]
            for i, (a, b) in enumerate(zip(data.ma, data.mb, strict=True))
        ]
    )
    gpt4_side = np.array([a == "gpt-4" for a in data.ma], float) - np.array(
        [b == "gpt-4" for b in data.mb], float
    )
    regression = judge_error_regression(
        judge_all,
        human_all,
        {"stronger_model_margin": spread, "gpt4_is_this_side": gpt4_side},
    )
    print(regression.summary())
    print(
        "  stronger_model_margin absorbs 'the judge exaggerates the quality spread'.\n"
        "  gpt4_is_this_side is what survives it: the extra credit GPT-4 gives its own\n"
        "  answers, over and above being the strongest model in the set."
    )

    print("\n3. Judge quality against the human verdict, decisive comparisons only")
    binary = data[decisive]
    print(judge_agreement(binary.judge.map(SCORE).values, binary.human.map(SCORE).values).summary())

    print("\n4. Recovering the human number from a fraction of the labels")
    judge, human = oriented(data, "gpt-4")
    truth, n = human.mean(), len(human)
    print(f"   gpt-4's win rate: judge says {judge.mean():.4f}, all human labels say {truth:.4f}")
    print(f"   {'labels':>7}{'judge-only':>13}{'gold-only':>12}{'truescore':>12}{'width':>9}")
    rng = np.random.default_rng(0)
    reps = 600
    for budget in (60, 100, 150):
        hits = {"judge": 0, "gold": 0, "ppi": 0}
        widths = {"gold": [], "ppi": []}
        for _ in range(reps):
            index = np.sort(rng.choice(n, size=budget, replace=False))
            ppi = ppi_estimate(judge, human[index], index)
            gold = gold_only_estimate(human[index])
            naive = judge_only_estimate(judge)
            hits["ppi"] += ppi.low <= truth <= ppi.high
            hits["gold"] += gold.low <= truth <= gold.high
            hits["judge"] += naive.low <= truth <= naive.high
            widths["ppi"].append(ppi.high - ppi.low)
            widths["gold"].append(gold.high - gold.low)
        print(
            f"   {budget:>7}{hits['judge'] / reps:>12.1%}{hits['gold'] / reps:>12.1%}"
            f"{hits['ppi'] / reps:>12.1%}{np.mean(widths['ppi']):>9.4f}"
        )
    print(
        "   Coverage is the fraction of 600 random label subsets whose 95% interval\n"
        "   contained the full-human rate. The full human labels are never given to the\n"
        "   estimator; they are only used to score it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
