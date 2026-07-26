"""Generate the sample evaluation data the other examples read.

The scenario is a customer-support assistant. Two versions are evaluated on the same 4000
support questions: ``v3`` in production and ``v4`` proposed to replace it. An LLM judge
marks every answer pass or fail, and a contractor has hand-labeled 400 of the questions
for both versions.

Three properties are baked in deliberately, because all three are ordinary and all three
break a naive evaluation:

1. **The judge is lenient.** It rarely fails a genuinely correct answer but waves through
   a lot of wrong ones, so every reported pass rate is too high.
2. **The judge rewards length.** Its verdict depends partly on how long the answer is,
   independently of whether the answer is right.
3. **v4 writes longer answers than v3.** Which means the judge flatters v4 for a reason
   that has nothing to do with quality.

The true improvement from v3 to v4 is ten points, which is a genuinely good release. The
judge will report roughly twice that, and a launch review reading the judge's number would
credit the model for verbosity. Everything the examples demonstrate follows from those
three lines.

Run: python examples/generate_sample_data.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
SEED = 20260725

N_EXAMPLES = 4000
N_HUMAN_LABELS = 600

TRUE_PASS_V3 = 0.62
TRUE_PASS_V4 = 0.72

JUDGE_SENSITIVITY = 0.94
JUDGE_SPECIFICITY = 0.62
# Change in the judge's pass probability per 100 tokens of response length, applied on top
# of its base error rates. Small per token, decisive in aggregate.
# The length effect lands mostly on the false-positive rate: a long *wrong* answer is what
# fools a judge, whereas a long right answer is still right. Expressed per 100 tokens.
JUDGE_LENGTH_EFFECT = 0.300
# Length is measured against a fixed reference, not against each version's own mean --
# otherwise the bias would be centred separately per version and could not inflate one
# version relative to the other, which is the entire failure being demonstrated.
REFERENCE_TOKENS = 150.0

MEAN_LOG_LENGTH_V3 = 4.60
MEAN_LOG_LENGTH_V4 = 5.30
SD_LOG_LENGTH = 0.45


def _judge_verdicts(rng: np.random.Generator, truth: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Judge a set of answers: lenient overall, and biased toward long answers."""
    centred_length = (lengths - REFERENCE_TOKENS) / 100.0
    pass_given_wrong = np.clip(
        (1.0 - JUDGE_SPECIFICITY) + JUDGE_LENGTH_EFFECT * centred_length, 0.02, 0.90
    )
    pass_given_right = np.clip(
        JUDGE_SENSITIVITY + 0.2 * JUDGE_LENGTH_EFFECT * centred_length, 0.50, 0.99
    )
    probability = np.where(truth == 1, pass_given_right, pass_given_wrong)
    return rng.binomial(1, probability)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Questions differ in difficulty, and both versions find the same ones hard.
    difficulty = rng.beta(2.0, 2.0, N_EXAMPLES)
    truth_v3 = rng.binomial(1, np.clip(TRUE_PASS_V3 + 0.25 * (0.5 - difficulty), 0.02, 0.98))
    truth_v4 = rng.binomial(1, np.clip(TRUE_PASS_V4 + 0.25 * (0.5 - difficulty), 0.02, 0.98))

    length_v3 = np.round(rng.lognormal(MEAN_LOG_LENGTH_V3, SD_LOG_LENGTH, N_EXAMPLES))
    length_v4 = np.round(rng.lognormal(MEAN_LOG_LENGTH_V4, SD_LOG_LENGTH, N_EXAMPLES))

    judge_v3 = _judge_verdicts(rng, truth_v3, length_v3)
    judge_v4 = _judge_verdicts(rng, truth_v4, length_v4)

    labeled = np.sort(rng.choice(N_EXAMPLES, N_HUMAN_LABELS, replace=False))
    is_labeled = np.zeros(N_EXAMPLES, dtype=bool)
    is_labeled[labeled] = True

    def blank(values: np.ndarray, index: int) -> str:
        return str(int(values[index])) if is_labeled[index] else ""

    # 1. The main audit file: one row per example, judge always, human sometimes.
    with (DATA_DIR / "support_eval.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["example_id", "difficulty", "response_tokens", "judge_passed", "human_passed"]
        )
        for i in range(N_EXAMPLES):
            writer.writerow(
                [
                    f"q{i:05d}",
                    f"{difficulty[i]:.4f}",
                    int(length_v4[i]),
                    int(judge_v4[i]),
                    blank(truth_v4, i),
                ]
            )

    # 2. The comparison file: both versions on the same questions.
    with (DATA_DIR / "support_compare.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "example_id",
                "v3_tokens",
                "v4_tokens",
                "v3_judge_passed",
                "v4_judge_passed",
                "v3_human_passed",
                "v4_human_passed",
            ]
        )
        for i in range(N_EXAMPLES):
            writer.writerow(
                [
                    f"q{i:05d}",
                    int(length_v3[i]),
                    int(length_v4[i]),
                    int(judge_v3[i]),
                    int(judge_v4[i]),
                    blank(truth_v3, i),
                    blank(truth_v4, i),
                ]
            )

    # 3. A frozen anchor set, judged twice: the provider updated the judge in between.
    anchor_size = 600
    anchor = np.sort(rng.choice(N_EXAMPLES, anchor_size, replace=False))
    anchor_truth = truth_v4[anchor]
    judge_may = np.where(rng.binomial(1, 0.88, anchor_size) == 1, anchor_truth, 1 - anchor_truth)
    # The June judge is stricter: it keeps most verdicts but fails more borderline answers.
    keep = rng.binomial(1, 0.82, anchor_size)
    judge_june = np.where(keep == 1, judge_may, 0)

    with (DATA_DIR / "judge_anchor.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["example_id", "human_passed", "judge_may", "judge_june"])
        for position, index in enumerate(anchor):
            writer.writerow(
                [
                    f"q{index:05d}",
                    int(anchor_truth[position]),
                    int(judge_may[position]),
                    int(judge_june[position]),
                ]
            )

    # 4. A live stream from a release that genuinely regressed partway through.
    stream_length = 1200
    healthy = rng.binomial(1, 0.88, stream_length // 2)
    regressed = rng.binomial(1, 0.79, stream_length - stream_length // 2)
    stream = np.concatenate([healthy, regressed])
    with (DATA_DIR / "release_stream.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["request_id", "passed"])
        for i, value in enumerate(stream):
            writer.writerow([f"r{i:05d}", int(value)])

    # 5. A segmented dataset: v4 improves overall but regresses on one segment, and the
    #    judge is most lenient on exactly that segment because its answers are longest.
    segment_names = np.asarray(["billing", "account", "technical"])
    segment_share = [0.40, 0.35, 0.25]
    v4_by_segment = {"billing": 0.78, "account": 0.76, "technical": 0.46}
    segment_length = {"billing": 4.6, "account": 4.7, "technical": 5.6}

    seg_index = rng.choice(3, N_EXAMPLES, p=segment_share)
    segments = segment_names[seg_index]
    seg_truth_v3 = rng.binomial(1, 0.62, N_EXAMPLES)
    seg_truth_v4 = rng.binomial(1, np.asarray([v4_by_segment[s] for s in segments]))
    seg_len_v3 = np.round(rng.lognormal(4.6, SD_LOG_LENGTH, N_EXAMPLES))
    seg_len_v4 = np.round(
        rng.lognormal(np.asarray([segment_length[s] for s in segments]), SD_LOG_LENGTH)
    )
    seg_judge_v3 = _judge_verdicts(rng, seg_truth_v3, seg_len_v3)
    seg_judge_v4 = _judge_verdicts(rng, seg_truth_v4, seg_len_v4)
    seg_labeled = np.sort(rng.choice(N_EXAMPLES, 1500, replace=False))
    seg_is_labeled = np.zeros(N_EXAMPLES, dtype=bool)
    seg_is_labeled[seg_labeled] = True

    with (DATA_DIR / "support_segments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "example_id",
                "segment",
                "v3_judge_passed",
                "v4_judge_passed",
                "v3_human_passed",
                "v4_human_passed",
            ]
        )
        for i in range(N_EXAMPLES):
            writer.writerow(
                [
                    f"q{i:05d}",
                    segments[i],
                    int(seg_judge_v3[i]),
                    int(seg_judge_v4[i]),
                    str(int(seg_truth_v3[i])) if seg_is_labeled[i] else "",
                    str(int(seg_truth_v4[i])) if seg_is_labeled[i] else "",
                ]
            )

    # 6. Log-likelihoods for a contamination check on a public benchmark: one canonical
    #    ordering and 199 shuffles, for a model that never saw the data.
    with (DATA_DIR / "contamination_logliks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "loglik"])
        permuted = rng.normal(-18432.0, 41.0, 199)
        writer.writerow(["canonical", f"{rng.normal(-18432.0, 41.0):.4f}"])
        for value in permuted:
            writer.writerow(["permuted", f"{value:.4f}"])

    print(f"wrote sample data to {DATA_DIR}")
    print(f"  true pass rate v3: {truth_v3.mean():.4f}")
    print(f"  true pass rate v4: {truth_v4.mean():.4f}")
    print(f"  true improvement:  {truth_v4.mean() - truth_v3.mean():+.4f}")
    print(f"  judge says v3:     {judge_v3.mean():.4f}")
    print(f"  judge says v4:     {judge_v4.mean():.4f}")
    print(f"  judge's version:   {judge_v4.mean() - judge_v3.mean():+.4f}")
    print("  segmented dataset (support_segments.csv):")
    for name in segment_names:
        mask = segments == name
        true_gap = seg_truth_v4[mask].mean() - seg_truth_v3[mask].mean()
        judged_gap = seg_judge_v4[mask].mean() - seg_judge_v3[mask].mean()
        print(
            f"    {name:<10} true {true_gap:+.4f}   as judged {judged_gap:+.4f}"
            f"   ({mask.sum()} examples)"
        )


if __name__ == "__main__":
    main()
