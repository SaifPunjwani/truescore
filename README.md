# truescore

Statistics for LLM-judged evaluations: bias-corrected scores, real confidence intervals,
and comparisons that hold up.

[![CI](https://github.com/SaifPunjwani/truescore/actions/workflows/ci.yml/badge.svg)](https://github.com/SaifPunjwani/truescore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/truescore)](https://pypi.org/project/truescore/)
[![Python](https://img.shields.io/pypi/pyversions/truescore)](https://pypi.org/project/truescore/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

```sh
pip install truescore
truescore doctor your_eval_results.csv
```

## The problem

You grade your model with an LLM judge. The judge is wrong in ways you haven't measured,
so the number it gives you is wrong too.

Here's a run from `examples/01_audit_an_eval.py`. A support assistant, 4000 questions
graded by a judge, 600 of them also graded by a person:

```
support-assistant-v4 -- pass rate
  corrected:  0.7150 [0.6836, 0.7463] (ppi++)
  judge-only: 0.8383 [0.8265, 0.8493] (judge_only (wilson))  (off by +0.1233)
  gold-only:  0.7183 [0.6810, 0.7528] (gold_only (wilson))
  n=4000 examples, 600 human-labeled
```

The data is simulated, so we know the real pass rate is 0.7140. The judge reports 0.8383.
Its confidence interval doesn't contain the true value. The corrected estimate does, and
it's narrower than using the 600 human labels on their own.

Comparisons are worse. From `examples/02_compare_two_models.py`:

| | v4 minus v3 |
| --- | --- |
| judge says | +0.1700 |
| corrected | +0.0950 `[+0.0505, +0.1394]` |
| actual | +0.0900 |

v4 is genuinely better. But 7 of those 17 points come from the judge preferring longer
answers, and v4's median answer is 197 tokens against v3's 99. Measured length bias:
+0.069 per 100 tokens, p = 1e-05.

Slice the same data by support segment and the judge says all three improved. One of them
regressed by 17 points. It's the segment where answers got longest. Per-segment correction
gives -0.1659 where the planted truth was -0.1663.

## Watching a dashboard doesn't help

A 95% confidence interval is valid at one sample size you picked in advance. Check it
repeatedly as data arrives and it isn't 95% anymore. Over 300 simulated healthy streams:

| checked after every observation | false alarms |
| --- | --- |
| fixed-sample interval | 47.7% |
| truescore confidence sequence | 0.3% |

`truescore.sequential` gives intervals valid at every sample size at once, so you can
check whenever you want.

## Usage

Start with `doctor`. It reads your file, works out what your columns are, and tells you
what it can compute:

```sh
truescore doctor results.csv
```

```
results.csv: 4000 rows, 5 columns

  column                  kind                coverage  detail
  judge_passed            verdict                 100%  pass/fail on every row
  human_passed            sparse_verdict           15%  pass/fail on 600 of 4000 rows
  response_tokens         numeric                 100%  range 35 to 1016

what this file supports today:
  - correct the score for judge error (truescore audit results.csv --judge judge_passed --gold human_passed)
  - monitor a release for regressions, no human labels needed

what the judge appears to be biased by:
  - response_tokens: the judge gets more generous as it rises (+0.00069 per unit, adjusted p=2.23e-05)
```

The rest of the commands:

```sh
truescore audit   results.csv --judge judge_passed --gold human_passed --markdown report.md
truescore compare results.csv --judge-a v4 --judge-b v3 --gold-a v4_human --gold-b v3_human
truescore slices  results.csv --by segment --judge-a v4 --gold-a v4_human --judge-b v3 --gold-b v3_human
truescore drift   anchor.csv  --baseline judge_may --current judge_june --gold human_passed
truescore monitor stream.csv  --metric passed --baseline 0.88 --window 300
truescore plan    --n-total 4000 --target 0.03 --sensitivity 0.97 --specificity 0.47
```

Exit codes: 0 means nothing found, 2 means a finding, 1 means it couldn't run. They're
different so a CI job can fail on drift without also failing on a typo in a filename.

From Python:

```python
import truescore as ts
from truescore.io import load_labels

labels = load_labels("results.csv", judge="judge_passed", gold="human_passed")
report = ts.build_report(labels.judge, labels.gold, labels.gold_index)
print(report.summary())
```

### In CI

```yaml
- uses: SaifPunjwani/truescore@v0.5.0
  with:
    command: drift
    args: anchor.csv --baseline judge_pinned --current judge_today --gold human_passed
```

Set `fail-on-finding: false` to report without blocking.

## What's in it

| module | what it answers |
| --- | --- |
| `doctor` | What can this file support? Profiles columns, lists runnable commands, says what's blocked and how many labels would unblock it, scans numeric columns for judge bias. |
| `correct` | What's the true score? Prediction-powered inference, Rogan–Gladen correction. |
| `agreement` | How good is the judge? Accuracy, sensitivity, specificity, Cohen's κ, Gwet's AC1, Krippendorff's α, all with intervals. |
| `bias` | What's the judge biased by? HC3 regression on length, self-preference, formatting. Position-bias test for pairwise judges. |
| `compare` | Is A better than B? McNemar (mid-p), paired bootstrap, permutation, PPI-corrected comparison, Holm and BH. |
| `slices` | Better for whom? Per-segment correction with multiplicity control. |
| `sequential` | Can I watch this live? Confidence sequences, windowed detection for late regressions. |
| `drift` | Did the judge change? Anchor-set comparison plus a label-flip rate. |
| `weighting` | Does this eval set look like production? Post-stratified reweighting. |
| `contamination` | Is the eval set in the training data? Exact permutation test. |
| `power` | How many labels do I need? Budgets, minimum detectable effect, sample size. |
| `report` | JSON and markdown artifact recording estimator, assumptions, and the naive number. |
| `io` / `cli` | CSV and JSONL loading, sparse gold columns, the command line. |

Labeling is the expensive part, so `power` prices it. Against the judge in the sample data:

```
  +/-0.05:   232 labels (without the judge:   317, saving 85)
  +/-0.03:   670 labels (without the judge:   879, saving 209)
  +/-0.02:  1644 labels (without the judge:  1978, saving 334)
```

## How many human labels do you actually need?

Some, for the correction. You can't calibrate an instrument without something to calibrate
it against.

About half the library needs none: regression monitoring, contamination testing,
position-bias detection, and label planning all run on judge output alone.

For the rest, three things make it cheaper than it sounds. It's a few hundred labels, not
thousands. You label once and reuse across releases. And you may already have human
verdicts sitting in your database: escalations, thumbs-down, refunds, whether the customer
came back.

You can substitute a stronger model for the human. The math still works, but your
guarantee is then relative to that model being right, which nobody has checked.

## What it isn't

Not an eval runner, not a judge, not a prompt framework, not a dataset. It never calls a
model. It reads the output of whatever harness you already use.

## Why trust the numbers

Every interval estimator has a coverage test: simulate from a known truth hundreds of
times, check that the 95% interval covers about 95% of the time. The confidence sequence
is checked over whole trajectories, which is the stronger claim it makes.

The suite also fuzzes every public function against adversarial inputs and requires that
nothing ever returns a NaN. Any input either gives a finite result or raises `ValueError`.
That found three real bugs before release, including one where a near-deterministic slice
got an interval with 7.7% coverage.

602 tests. `mypy --strict`. numpy and scipy are the only dependencies.

Derivations, assumptions, and known limits are in [`docs/methods/`](docs/methods/).

## Examples

```sh
python examples/generate_sample_data.py
for f in examples/0*.py; do python "$f"; done
```

| script | what it shows |
| --- | --- |
| `01_audit_an_eval.py` | Correcting a judge-scored eval. |
| `02_compare_two_models.py` | +17 points reported, +9 real. |
| `03_monitor_a_release.py` | Live monitoring, and why cumulative is the wrong question. |
| `04_detect_judge_drift.py` | An anchor set catching a changed judge. |
| `05_plan_a_labeling_budget.py` | What precision costs. |
| `06_find_the_regressed_segment.py` | All three segments "improved". One regressed 17 points. |

Every number those scripts print is computed when you run them.

## License

Apache-2.0.
