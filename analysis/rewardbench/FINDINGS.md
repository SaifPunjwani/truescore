# A judge leaderboard, with the error bars it is printed without

The top two judges on RewardBench are 0.37 points apart and not distinguishable on 2985
paired examples. They also disagree on 10 of 23 subsets, in both directions, by up to 22
points.

Both facts come from the same data. Reproduce them with:

```sh
pip install truescore
python run.py
```

Per-example verdicts come from
[allenai/reward-bench-results](https://huggingface.co/datasets/allenai/reward-bench-results),
which publishes each judge's decision on all 2985 preference pairs. Ground truth is
RewardBench's human-verified answer for each pair. No model is called.

## The leaderboard

Nine judges of the 2024 generation, scored on the same 2985 examples.

| | judge | accuracy | 95% interval |
| --- | --- | --- | --- |
| 1 | Skywork-Critic-Llama-3.1-8B | 0.8898 | `[0.8781, 0.9005]` |
| 2 | gpt-4o-2024-08-06 | 0.8861 | `[0.8742, 0.8970]` |
| 3 | gpt-4o-2024-05-13 | 0.8670 | `[0.8543, 0.8787]` |
| 4 | Meta-Llama-3.1-405B-Instruct-Turbo | 0.8613 | `[0.8484, 0.8732]` |
| 5 | Meta-Llama-3.1-70B-Instruct | 0.8586 | `[0.8457, 0.8707]` |
| 6 | gemini-1.5-flash-001 | 0.8382 | `[0.8245, 0.8510]` |
| 7 | gpt-4o-mini-2024-07-18 | 0.8271 | `[0.8132, 0.8403]` |
| 8 | Hermes-3-Llama-3.1-70B | 0.8064 | `[0.7918, 0.8201]` |
| 9 | prometheus-8x7b-v2.0 | 0.7680 | `[0.7524, 0.7826]` |

Every judge sees the same examples, so comparisons are paired and far more powerful than
those intervals suggest. Comparing all 36 pairs with a paired bootstrap and a Holm
correction:

```
                                     1   2   3   4   5   6   7   8   9
1. Skywork-Critic-Llama-3.1-8B       .   ~   x   x   x   x   x   x   x
2. gpt-4o-2024-08-06                 ~   .   x   x   x   x   x   x   x
3. gpt-4o-2024-05-13                 x   x   .   ~   ~   x   x   x   x
4. Meta-Llama-3.1-405B-Instruct      x   x   ~   .   ~   x   x   x   x
5. Meta-Llama-3.1-70B-Instruct       x   x   ~   ~   .   x   x   x   x
6. gemini-1.5-flash-001              x   x   x   x   x   .   ~   x   x
7. gpt-4o-mini-2024-07-18            x   x   x   x   x   ~   .   x   x
8. Hermes-3-Llama-3.1-70B            x   x   x   x   x   x   x   .   x
9. prometheus-8x7b-v2.0              x   x   x   x   x   x   x   x   .

x = distinguishable    ~ = not distinguishable
```

Four of the eight adjacent gaps do not survive. The nine ranks are really four tiers, and
inside a tier the ordering is noise:

- **1-2**: an 8B open critic model and `gpt-4o-2024-08-06`
- **3-5**: `gpt-4o-2024-05-13`, Llama-3.1-405B and Llama-3.1-70B. The 405B model is not
  distinguishable from the 70B one at this sample size.
- **6-7**: `gemini-1.5-flash-001` and `gpt-4o-mini`
- **8**, **9**: separated from everything

## A tied number is not a tie

Ranks 1 and 2 have overall accuracies of 0.8898 and 0.8861. Broken out by subset, with a
Holm correction across all 23:

| subset | n | Skywork-Critic-8B | gpt-4o-2024-08-06 | gap | p adj |
| --- | --- | --- | --- | --- | --- |
| llmbar-adver-neighbor | 134 | 0.806 | **0.582** | +0.224 | 0.000 |
| math-prm | 447 | 0.915 | 0.749 | +0.166 | 0.000 |
| llmbar-adver-GPTInst | 92 | 0.924 | 0.761 | +0.163 | 0.005 |
| refusals-dangerous | 100 | 0.960 | 0.850 | +0.110 | 0.000 |
| hep-js | 164 | 0.872 | 0.994 | -0.122 | 0.000 |
| hep-go | 164 | 0.872 | 0.988 | -0.116 | 0.000 |
| hep-rust | 164 | 0.872 | 0.988 | -0.116 | 0.000 |
| hep-java | 164 | 0.884 | 0.982 | -0.098 | 0.000 |
| hep-python | 164 | 0.896 | 0.982 | -0.085 | 0.017 |
| hep-cpp | 164 | 0.890 | 0.963 | -0.073 | 0.039 |

Ten of 23 subsets differ after correction, and they differ in both directions. The pattern
is not subtle. `gpt-4o-2024-08-06` wins all six code subsets by 7 to 12 points.
Skywork-Critic wins adversarial preference, process-supervised maths and dangerous-refusal
detection by 11 to 22 points. On `llmbar-adver-neighbor`, which pairs a good response with
a plausible near-miss, `gpt-4o-2024-08-06` scores 0.582 against a chance rate of 0.5.

Two judges with the same headline number, and the right choice between them depends
entirely on what you are judging. That is not visible in the number the table sorts by.

## Why the pairing matters

The interval on rank 1 is `[0.8781, 0.9005]` and on rank 2 is `[0.8742, 0.8970]`. They
overlap heavily, and a reader might conclude nothing here is distinguishable. That
inference is wrong in the other direction: overlapping marginal intervals do not imply a
non-significant difference when the measurements are paired. Ranks 2 and 3 also overlap
(`[0.8742, 0.8970]` against `[0.8543, 0.8787]`) and *are* distinguishable, because the same
2985 examples were scored by both and the paired test uses that.

Every comparison here is paired for the same reason, which is what makes a 1.9-point gap
detectable while a 0.4-point gap is not.

## What this does not show

- **Not distinguishable is not identical.** Ranks 1 and 2 differ by 0.37 points and 2985
  examples cannot resolve that. A larger benchmark might. The claim is that the published
  ordering is not evidence, not that the judges are equivalent.
- **These are the maintainers' runs.** Prompt, parsing and tie handling are RewardBench's
  choices, and a different harness would move these numbers. That cuts both ways: it is
  also true of the leaderboard itself.
- **`gemini-1.5-pro-exp-0801` is excluded.** Its results file records a tie on 96.3% of
  examples, which is a parsing failure rather than a judge scoring 0.5. Reporting the
  resulting 0.5151 as its accuracy would be a false claim about a real product.
- **Ground truth is human-verified preference**, assembled from several sources, and
  imperfect like any human labelling. Subsets are also unequal in size, so power varies:
  `llmbar-adver-GPTOut` shows a 19-point gap that does not survive correction at n=47.
- **This is RewardBench v1.** A v2 exists with a different construction.

## A note on the arithmetic

Running this study found a bug in truescore. On `mt-bench-easy` both judges score 1.000, so
every paired difference is exactly zero and every bootstrap resample is exactly zero.
`paired_bootstrap` was measuring one tail as the complement of the other, which put all of
that mass in a single tail and returned p = 0.000 beside a gap of +0.000. It then survived
the Holm correction and was reported as a real difference. Both tails now count the mass
sitting at zero, so perfect agreement returns p = 1. Fixed in 0.7.4, with
`tests/test_compare.py::test_identical_systems_are_not_a_significant_difference`.

Worth stating plainly rather than quietly correcting: the bug produced a wrong row in this
table, and the reason it was caught is that a subset where two judges agree perfectly is
obvious to a reader in a way it is not to a test suite.
