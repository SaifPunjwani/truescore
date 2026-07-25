# truescore

Statistically valid evaluation for LLM-judged benchmarks: bias-corrected scores, honest
intervals, and comparisons that survive scrutiny.

**Private and unreleased.** See `LICENSE` — all rights reserved pending a decision on
commercial direction. Design and go-to-market notes live in
[`docs/superpowers/specs/2026-07-25-truescore-design.md`](docs/superpowers/specs/2026-07-25-truescore-design.md).

## The problem

A team runs 200 examples through an LLM judge and reports "92% pass rate." Three things
are wrong with that number, and each has a known remedy that almost nobody applies:

1. **The judge is a biased instrument.** It agrees with careful human labels maybe 85% of
   the time, and its errors are asymmetric — judges favor longer answers, first-presented
   answers, and answers from their own model family. Averaging judge labels estimates the
   *judge's* pass rate, not the system's.
2. **Nobody reports intervals.** At n=200, a 92% score carries roughly a ±3.8-point
   interval. Teams ship on 1.5-point "improvements" that are indistinguishable from noise.
3. **Comparisons ignore pairing.** Two systems are scored on the same examples, then
   compared with tests that assume independent samples, throwing away most of the power.

truescore takes labels you already have — judge labels on many examples, human labels on
a few — and returns numbers you can defend.

```python
import truescore as ts

# 2000 examples judged by an LLM; 200 of them also labeled by humans.
report = ts.build_report(judge_labels, human_labels, human_label_index,
                         system_name="assistant-v4", metric_name="pass rate")
print(report.summary())
```

```
assistant-v4 -- pass rate
  corrected:  0.6942 [0.6362, 0.7523] (ppi++)
  judge-only: 0.7935 [0.7752, 0.8107] (judge_only (wilson))  (off by +0.0993)
  gold-only:  0.7050 [0.6384, 0.7639] (gold_only (wilson))
  n=2000 examples, 200 human-labeled
```

That output is real — `examples/judge_correction.py` produces it, and because the example
is a simulation we know the truth is 0.6970. The judge-only number is **9.9 points too
high, and its interval does not contain the truth at all**. The corrected estimate does,
and is still tighter than using the 200 human labels alone: that is the point of
prediction-powered inference — the judge contributes precision without contributing its
bias.

The same run answers the budgeting question:

```
target ±0.05:
  gold labels needed: 217 (half-width ±0.0500)
  without the judge:  327
  the judge saves 110 human labels (34%)

target ±0.02:
  target precision is NOT reachable with 2000 examples
  best achievable half-width: ±0.0202 using 1999 gold labels
```

## What it does

| module | question it answers |
| --- | --- |
| `truescore.agreement` | How good is my judge? Accuracy, sensitivity/specificity, Cohen's κ, Gwet's AC1, Krippendorff's α — all with intervals. |
| `truescore.correct` | What is the *true* score? Prediction-powered inference and Rogan–Gladen misclassification correction. |
| `truescore.bias` | What is my judge biased by? HC3-robust regression of judge error on length, self-preference, formatting; position-bias test for pairwise judges. |
| `truescore.compare` | Is A actually better than B? McNemar (mid-p), paired bootstrap, sign-flip permutation, PPI-corrected comparison, multiplicity control. |
| `truescore.power` | How many human labels do I need? Gold-label budgets, minimum detectable effect, required sample size. |
| `truescore.report` | The artifact: JSON and markdown recording estimator, assumptions, and what the naive number would have said. |

## What it is not

Not an eval runner, not a judge, not a prompt framework, not a dataset, not an
observability platform. It never calls a model. It consumes the output of whatever
harness you already use — internal, lm-eval-harness, or a vendor — which is what makes it
composable rather than competitive.

## Why the intervals are trustworthy

Every interval estimator is verified by simulation in CI: draw from a known ground truth
a thousand times, and assert the nominal 95% interval actually covers about 95% of the
time. A statistics library that does not verify its own coverage is asserting exactly what
it ought to be proving.

## Install (development)

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest
```

Dependencies: numpy and scipy. Runs anywhere; no GPU, no network, no model calls.
