# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Saif Punjwani
"""Does a judge leaderboard's ordering survive its own error bars?

RewardBench scores LLM judges on 2985 preference pairs with human-verified answers, and
publishes a rank-ordered table. Picking a judge off that table is a decision people make.
This script asks two questions the table does not answer.

First, which adjacent ranks are distinguishable? Every judge is scored on the same 2985
examples, so the comparisons are paired, which is where the power is. With a multiplicity
correction across all pairs, the ordering collapses into a handful of tiers.

Second, does a tied headline number mean two judges are interchangeable? It does not. The
two judges at the top of this table have almost identical overall accuracy and opposite
strengths, and which one is right for a given workload is not visible in the number the
leaderboard sorts by.

Data is downloaded at run time from allenai/reward-bench-results, which publishes each
judge's per-example verdict. No model is called.

    pip install truescore
    python run.py

Every number in FINDINGS.md is printed below.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import urllib.request

import numpy as np

from truescore.agreement import wilson_interval
from truescore.compare import holm, paired_bootstrap

REPO = "allenai/reward-bench-results"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
API = f"https://huggingface.co/api/datasets/{REPO}"

# Judges to include: the recognisable general-purpose models of the 2024 generation plus
# the specialised critic models that beat them, which is the comparison that matters.
WANTED = (
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini-2024-07-18",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-exp-0801",
    "Meta-Llama-3.1-405B-Instruct-Turbo",
    "Meta-Llama-3.1-70B-Instruct.json",
    "Skywork-Critic-Llama-3.1-8B",
    "Hermes-3-Llama-3.1-70B",
    "prometheus-8x7b-v2.0",
)

# A judge whose harness recorded a tie on most examples is not a judge scoring 0.5, it is a
# parse failure. Reporting that as accuracy would be a false claim about a real product.
MAX_TIE_RATE = 0.5


def _fetch(url: str) -> object:
    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=headers)))


def load() -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Per-example verdicts for each judge, plus the subset label of each example."""
    tree = _fetch(f"{API}/tree/main/eval-set-scores?recursive=1&limit=1000")
    paths = [
        entry["path"]  # type: ignore[index]
        for entry in tree  # type: ignore[union-attr]
        if entry["path"].endswith(".json")  # type: ignore[index]
        and any(name in entry["path"] for name in WANTED)  # type: ignore[index]
    ]

    verdicts: dict[str, np.ndarray] = {}
    subsets: np.ndarray | None = None
    ids: tuple[int, ...] | None = None
    excluded: list[str] = []

    for path in sorted(paths):
        record = _fetch(f"{BASE}/{path}")
        name = record["model"]  # type: ignore[index]
        results = np.asarray(record["results"], dtype=float)  # type: ignore[index]
        tie_rate = float((results == 0.5).mean())
        if tie_rate > MAX_TIE_RATE:
            excluded.append(f"{name} ({tie_rate:.1%} of entries are ties)")
            continue

        this_ids = tuple(record["id"])  # type: ignore[index]
        if ids is None:
            ids, subsets = this_ids, np.asarray(record["subset"])  # type: ignore[index]
        elif this_ids != ids:
            raise SystemExit(f"{name} was scored on a different example set")
        verdicts[name] = results

    assert subsets is not None
    return verdicts, subsets, excluded


def main() -> int:
    verdicts, subsets, excluded = load()
    ranked = sorted(verdicts.items(), key=lambda kv: -kv[1].mean())
    names = [name for name, _ in ranked]
    n = len(subsets)
    print(f"{len(names)} judges, {n} paired examples, {len(set(subsets))} subsets")
    for note in excluded:
        print(f"  excluded: {note}")

    print("\n1. The leaderboard, with the interval it is usually printed without")
    print(f"   {'judge':<46}{'accuracy':>10}  95% interval")
    for name, results in ranked:
        # Ties are half credit, so the count is not integral; round for the exact interval.
        interval = wilson_interval(round(float(results.sum())), n)
        print(f"   {name:<46}{results.mean():>10.4f}  [{interval.low:.4f}, {interval.high:.4f}]")

    print("\n2. Which differences survive a paired test and a multiplicity correction")
    pairs = list(itertools.combinations(names, 2))
    comparisons = [
        paired_bootstrap(verdicts[a], verdicts[b], n_bootstrap=20000, seed=0) for a, b in pairs
    ]
    adjusted = holm([c.p_value for c in comparisons])
    distinguishable = {
        frozenset(pair): bool(p < 0.05) for pair, p in zip(pairs, adjusted, strict=True)
    }

    print(f"   all {len(pairs)} pairs, Holm-adjusted at 0.05")
    header = "".join(f"{i + 1:>4}" for i in range(len(names)))
    print(f"   {'':<34}{header}")
    for i, a in enumerate(names):
        row = "".join(
            "   ." if i == j else ("   x" if distinguishable[frozenset((a, b))] else "   ~")
            for j, b in enumerate(names)
        )
        print(f"   {i + 1}. {names[i].split('/')[-1][:29]:<31}{row}")
    print("   x = distinguishable   ~ = not distinguishable")

    print("\n   adjacent ranks:")
    indistinct = 0
    for (a, _), (b, _) in itertools.pairwise(ranked):
        real = distinguishable[frozenset((a, b))]
        indistinct += not real
        gap = verdicts[a].mean() - verdicts[b].mean()
        label = f"{a.split('/')[-1][:28]} vs {b.split('/')[-1][:28]}"
        print(f"     {label:<62}{gap:>+8.4f}  {'yes' if real else 'NO'}")
    print(f"   {indistinct} of {len(ranked) - 1} adjacent gaps are not distinguishable at n={n}")

    print("\n3. Two judges tied on the headline number, and where they differ")
    first, second = names[0], names[1]
    print(f"   {first} vs {second}")
    print(f"   {'subset':<24}{'n':>5}{'first':>9}{'second':>9}{'gap':>9}{'p adj':>9}")
    labels = sorted(set(subsets))
    per_subset = []
    for label in labels:
        mask = subsets == label
        per_subset.append(
            paired_bootstrap(
                verdicts[first][mask], verdicts[second][mask], n_bootstrap=20000, seed=0
            )
        )
    subset_adjusted = holm([c.p_value for c in per_subset])
    flagged = 0
    for label, p in zip(labels, subset_adjusted, strict=True):
        mask = subsets == label
        a, b = verdicts[first][mask].mean(), verdicts[second][mask].mean()
        marker = ""
        if p < 0.05:
            flagged += 1
            marker = "  <-"
        print(
            f"   {label:<24}{int(mask.sum()):>5}{a:>9.3f}{b:>9.3f}{a - b:>+9.3f}{p:>9.3f}{marker}"
        )
    print(
        f"   {flagged} of {len(labels)} subsets differ after correction, in both "
        "directions, between two judges whose overall scores are "
        f"{verdicts[first].mean():.4f} and {verdicts[second].mean():.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
