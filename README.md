# truescore

Statistically valid evaluation for LLM-judged benchmarks: bias-corrected scores, honest
intervals, and comparisons that survive scrutiny.

**Private and unreleased.** See `LICENSE` — all rights reserved pending a decision on
commercial direction. Design and go-to-market notes are in
[`docs/superpowers/specs/2026-07-25-truescore-design.md`](docs/superpowers/specs/2026-07-25-truescore-design.md).

## The problem, in one table

A support assistant is evaluated on 4000 questions. An LLM judge scores every answer; a
contractor hand-labeled 600 of them. Here is what the two versions of that number look
like, from `examples/01_audit_an_eval.py`:

```
support-assistant-v4 -- pass rate
  corrected:  0.7150 [0.6836, 0.7463] (ppi++)
  judge-only: 0.8383 [0.8265, 0.8493] (judge_only (wilson))  (off by +0.1233)
  gold-only:  0.7183 [0.6810, 0.7528] (gold_only (wilson))
  n=4000 examples, 600 human-labeled
```

The example data is simulated, so the truth is known: **0.7140**. The judge-only number is
12 points too high and its interval does not contain the truth at all. The corrected
estimate does — and is *tighter* than the 600 human labels alone, because prediction-powered
inference takes the judge's precision without taking its bias.

It gets worse when you compare two systems. From `examples/02_compare_two_models.py`:

| | improvement of v4 over v3 |
| --- | --- |
| what the judge reports | **+0.1700** |
| corrected for judge error | **+0.0950** `[+0.0505, +0.1394]` |
| the truth | +0.0900 |

v4 really is better, and worth shipping. But 7 of those 17 points are the judge rewarding
v4 for writing longer answers — the median v4 answer is 197 tokens against v3's 99, and the
judge's length bias measures at `+0.069` per 100 tokens (p = 1e-05). A launch review that
believed the judge's number would be paying for verbosity, and would keep paying on every
release after this one.

And the overall number is not the end of it. Split the same evaluation by support
segment, and the judge reports an improvement on all three — while v4 has in fact
regressed on one of them by 17 points, because that is the segment where its answers got
longest. Correcting each segment separately recovers it (`-0.1659` against a planted truth
of `-0.1663`) and flags it at an adjusted p of 2e-07.

## And you cannot fix it by watching the dashboard

A 95% confidence interval is valid at *one* pre-specified sample size. Checked repeatedly
as data arrives, it is not valid at all. Measured over 300 simulated healthy streams:

| approach | false-alarm rate under continuous monitoring |
| --- | --- |
| fixed-sample interval, checked every observation | **47.7%** |
| truescore confidence sequence | 0.3% (budget: 5%) |

Nearly half of continuously-watched evals will show a regression that is not there.
`truescore.sequential` gives intervals valid at every sample size simultaneously, so you
can look whenever you like and act the moment they move.

## Install and use

```sh
pip install -e '.[dev]'
python examples/generate_sample_data.py   # writes the sample files used below
```

As a library:

```python
import truescore as ts
from truescore.io import load_labels

labels = load_labels("results.csv", judge="judge_passed", gold="human_passed")
report = ts.build_report(labels.judge, labels.gold, labels.gold_index)
print(report.summary())
```

As a command-line tool, over the CSV or JSONL your harness already writes:

```sh
truescore audit   results.csv --judge judge_passed --gold human_passed --markdown report.md
truescore compare results.csv --judge-a v4_pass --judge-b v3_pass --gold-a v4_human --gold-b v3_human
truescore drift   anchor.csv  --baseline judge_may --current judge_june --gold human_passed
truescore monitor stream.csv  --metric passed --baseline 0.88 --window 300
truescore plan    --n-total 4000 --target 0.03 --sensitivity 0.97 --specificity 0.47
```

Exit codes are built for CI: **0** ran and found nothing, **2** found something (drift, a
regression, contamination), **1** could not run. A gate that cannot tell "your judge
changed" from "your file has a typo" gets ignored, so those are different codes.

## What it does

| module | question it answers |
| --- | --- |
| `agreement` | How good is my judge? Accuracy, sensitivity/specificity, Cohen's κ, Gwet's AC1, Krippendorff's α — with intervals. |
| `correct` | What is the *true* score? Prediction-powered inference and Rogan–Gladen misclassification correction. |
| `bias` | What is my judge biased by? HC3-robust regression of judge error on length, self-preference, formatting; position-bias test for pairwise judges. |
| `compare` | Is A actually better than B? McNemar (mid-p), paired bootstrap, sign-flip permutation, PPI-corrected comparison, Holm and BH. |
| `slices` | Better for *whom*? Per-segment corrected estimates and comparisons, with multiplicity control across segments and an honest refusal for segments too thin to support a number. |
| `sequential` | Can I watch this continuously? Confidence sequences valid at every sample size; windowed detection for regressions that start late. |
| `drift` | Did my judge change under me? Paired anchor-set comparison, plus a label-flip rate that catches a rewritten judge whose accuracy is unchanged. |
| `contamination` | Is my eval set in the training data? The exact exchangeability permutation test, with Fisher pooling across shards. |
| `power` | How many human labels do I need? Gold-label budgets, minimum detectable effect, required sample size. |
| `report` | The artifact: JSON and markdown recording estimator, assumptions, and what the naive number would have said. |
| `io` / `cli` | The on-ramp: sparse gold columns, many spellings of pass/fail, and a CI-gateable command line. |

Human labeling is the real cost of trustworthy evaluation, so `power` prices it directly.
Against the judge measured in the example data:

```
  +/-0.05:   232 labels (without the judge:   317, saving 85)
  +/-0.03:   670 labels (without the judge:   879, saving 209)
  +/-0.02:  1644 labels (without the judge:  1978, saving 334)
```

## What it is not

Not an eval runner, not a judge, not a prompt framework, not a dataset, not an
observability platform. It never calls a model. It consumes the output of whatever harness
you already use — internal, lm-eval-harness, or a vendor — which is what makes it
composable rather than competitive.

## Why the numbers are trustworthy

Every interval estimator is verified by simulation in CI: draw from a known ground truth
hundreds of times and assert the nominal 95% interval covers about 95% of the time. That
includes the hard case — the confidence sequence is checked for coverage *over whole
trajectories*, not at single time points. A statistics library that does not verify its own
coverage is asserting exactly what it ought to be proving.

Beyond coverage, the suite fuzzes every public function against adversarial inputs and
enforces one contract: **no NaN ever escapes.** Any input either produces a finite result or
raises `ValueError`. That pass found three real defects on its first run, all fixed.

575 tests, `mypy --strict`, numpy and scipy as the only dependencies, no GPU, no network,
no model calls.

## Examples

| script | what it shows |
| --- | --- |
| `01_audit_an_eval.py` | Correct a judge-scored eval; judge quality and bias profile. |
| `02_compare_two_models.py` | The launch-review trap: +17 points reported, +9 real. |
| `03_monitor_a_release.py` | Anytime-valid monitoring, and why cumulative is the wrong question. |
| `04_detect_judge_drift.py` | An anchor set catching a judge that changed. |
| `05_plan_a_labeling_budget.py` | What precision costs, and what a better judge is worth. |
| `06_find_the_regressed_segment.py` | The judge says all three segments improved; one regressed by 17 points. |
| `judge_correction.py` | The core correction in isolation, no files needed. |

Every number those scripts print is computed at run time. `generate_sample_data.py` builds
the inputs and states the ground truth it planted, so each claim can be checked.
