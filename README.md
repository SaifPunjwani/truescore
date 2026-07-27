# truescore

Statistics for LLM-judged evaluations: bias-corrected scores, valid confidence intervals,
and model comparisons that survive review.

[![CI](https://github.com/SaifPunjwani/truescore/actions/workflows/ci.yml/badge.svg)](https://github.com/SaifPunjwani/truescore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/truescore)](https://pypi.org/project/truescore/)
[![Python](https://img.shields.io/pypi/pyversions/truescore)](https://pypi.org/project/truescore/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## Installation

```sh
pip install truescore
```

Requires Python 3.10 or later. numpy and scipy are the only dependencies. truescore never
calls a model, so there is no API key and no data leaves the machine.

## Quick start

```sh
truescore doctor results.csv
```

`doctor` reads the file, identifies the columns, and reports what it can compute:

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

Then run the correction it suggests:

```sh
truescore audit results.csv --judge judge_passed --gold human_passed --html report.html
```

## Background

An LLM judge grades your model. The judge has its own error rates, which most teams never
measure, so the reported score carries an unknown bias.

Output from `examples/01_audit_an_eval.py`, a support assistant scored on 4000 questions
with 600 of them also graded by a person:

```
support-assistant-v4 -- pass rate
  corrected:  0.7150 [0.6836, 0.7463] (ppi++)
  judge-only: 0.8383 [0.8265, 0.8493] (judge_only (wilson))  (off by +0.1233)
  gold-only:  0.7183 [0.6810, 0.7528] (gold_only (wilson))
  n=4000 examples, 600 human-labeled
```

The example data is simulated, so the real pass rate is known to be 0.7140. The judge
reports 0.8383 and its confidence interval stops at 0.8265, above the true value. The
corrected interval contains it, and is narrower than the one from the 600 human labels
alone.

Comparisons carry more error than single scores. From `examples/02_compare_two_models.py`:

| | v4 minus v3 |
| --- | --- |
| judge says | +0.1700 |
| corrected | +0.0950 `[+0.0505, +0.1394]` |
| actual | +0.0900 |

v4 is better, by about half what the judge reports. The gap traces to answer length: v4's
median answer runs 197 tokens against v3's 99, and the judge's measured length bias is
+0.069 per 100 tokens (p = 1e-05).

Splitting the same evaluation by support segment, the judge reports an improvement on all
three. One segment regressed by 17 points, the one where answers grew longest. Per-segment
correction returns -0.1659 against a planted truth of -0.1663.

### Repeated checks invalidate a fixed-sample interval

A 95% interval is valid at a sample size chosen in advance. Inspected repeatedly as data
arrives, it is not. Measured over 300 simulated healthy streams:

| checked after every observation | false alarms |
| --- | --- |
| fixed-sample interval | 47.7% |
| truescore confidence sequence | 0.3% |

`truescore.sequential` provides intervals valid at every sample size simultaneously.

## Usage

### Command line

```sh
truescore audit   results.csv --judge judge_passed --gold human_passed --html report.html
truescore compare results.csv --judge-a v4 --judge-b v3 --gold-a v4_human --gold-b v3_human
truescore slices  results.csv --by segment --judge-a v4 --gold-a v4_human --judge-b v3 --gold-b v3_human
truescore drift   anchor.csv  --baseline judge_may --current judge_june --gold human_passed
truescore monitor stream.csv  --metric passed --baseline 0.88 --window 300
truescore plan    --n-total 4000 --target 0.03 --sensitivity 0.97 --specificity 0.47
```

Exit codes are 0 for no finding, 2 for a finding, and 1 for a failure to run. A CI job can
therefore fail on judge drift without also failing on a malformed file.

### Eval tool output

Point any command at the file your eval tool already wrote. These formats are recognized
and flattened, and the judge column is identified for you:

| tool | file |
| --- | --- |
| [Inspect AI](https://inspect.aisi.org.uk/) | `logs/*.json` written with `--log-format=json` |
| [promptfoo](https://promptfoo.dev/) | `promptfoo eval -o results.json` (also `.jsonl`, `.csv`) |
| [DeepEval](https://deepeval.com/) | `.deepeval/.latest_test_run.json` |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | `samples_<task>_<timestamp>.jsonl` |

```sh
truescore doctor logs/2026-07-27_support-qa.json
```

Each shape is taken from that tool's own serialization code rather than from a sample
file, and each is pinned by a fixture test in `tests/test_adapters.py`. Score values are
mapped the way the tool maps them: Inspect's `C`, `P`, `I` and `N` become 1.0, 0.5, 0.0
and 0.0, matching its `value_to_float`, so a corrected number stays comparable to the
accuracy Inspect itself reported. Columns you added by hand are carried through under
their own names. Anything unrecognized is read as plain rows, and column names accept
dotted paths (`gradingResult.pass`, `scores.0.value`) for in-house formats.

### Human labels in a separate file

Human labels rarely live in the file the eval tool wrote: the tool writes verdicts, and
somebody labels a subset in a spreadsheet afterwards. `--gold-file` joins the two by
identifier, so neither file needs editing:

```sh
truescore audit results.json \
    --gold-file labels.csv --gold human --gold-id example_id \
    --covariate response_chars
```

`--judge` is optional here because the format was recognized. The join reports how many
labels landed, and refuses rather than guessing when the key matches nothing, when an
identifier repeats, or when one example carries two human verdicts.

`response_chars` is derived from the response text for every supported format, which makes
the standard verbosity-bias check available on any eval output without adding a column.

### Python

```python
import truescore as ts
from truescore.io import load_labels

labels = load_labels("results.csv", judge="judge_passed", gold="human_passed")
report = ts.build_report(labels.judge, labels.gold, labels.gold_index)
print(report.summary())
```

### GitHub Actions

```yaml
- uses: SaifPunjwani/truescore@v0.7.0
  with:
    command: drift
    args: anchor.csv --baseline judge_pinned --current judge_today --gold human_passed
```

Set `fail-on-finding: false` to report without blocking the build.

## Modules

| module | purpose |
| --- | --- |
| `doctor` | Profiles an evaluation file, lists runnable commands, reports what is blocked and how many labels would unblock it, scans numeric columns for judge bias. |
| `correct` | Bias-corrected scores by prediction-powered inference and the Rogan–Gladen correction. |
| `agreement` | Judge quality: accuracy, sensitivity, specificity, Cohen's κ, Gwet's AC1, Krippendorff's α, all with intervals. Quadratic-weighted kappa for graded rubrics. |
| `bias` | HC3-robust regression of judge error on length, self-preference and formatting. Position-bias test for pairwise judges. |
| `compare` | Paired comparison: McNemar (mid-p), bootstrap, permutation, PPI-corrected difference, Holm and Benjamini–Hochberg. |
| `slices` | Per-segment correction with multiplicity control across segments. |
| `sequential` | Confidence sequences valid at every sample size, with windowed detection for late regressions. |
| `drift` | Anchor-set comparison of two judge runs, reporting agreement change and label-flip rate. |
| `weighting` | Post-stratified reweighting to a stated production traffic mix. |
| `contamination` | Exact permutation test for a memorised evaluation set. |
| `power` | Labeling budgets, minimum detectable effect, required sample size. |
| `report` | JSON, markdown and self-contained HTML artifacts recording estimator, assumptions and the uncorrected number. |
| `adapters` | Reads Inspect AI, promptfoo, DeepEval and lm-evaluation-harness output directly, and identifies the judge column. |
| `io` / `cli` | CSV and JSON Lines loading with sparse gold columns, joining human labels from a separate file; the command line. |

## Human labels

The correction requires trusted labels on a subset of examples. Calibrating a measurement
instrument requires a reference to calibrate against.

Roughly half the library runs without any: regression monitoring, contamination testing,
position-bias detection and label planning all use judge output alone.

Where labels are needed, `power` prices them before any are collected. Against the judge in
the sample data:

```
  +/-0.05:   232 labels (without the judge:   317, saving 85)
  +/-0.03:   670 labels (without the judge:   879, saving 209)
  +/-0.02:  1644 labels (without the judge:  1978, saving 334)
```

The same few hundred labels serve every future release against that evaluation set. Many
teams also hold implicit human verdicts already: escalations, thumbs-down, refunds, repeat
contacts.

A stronger model can stand in for the human labeler. The arithmetic is unchanged, but the
resulting guarantee holds only relative to that model being correct.

## Scope

truescore consumes the output of an evaluation harness. It does not run evaluations, act as
a judge, provide prompts or datasets, or call any model.

## Validation

Every interval estimator carries a coverage test: simulate from a known ground truth
several hundred times and confirm the 95% interval covers about 95% of the time.
Confidence sequences are checked over whole trajectories, which is the stronger property
they claim.

The suite fuzzes every public function against adversarial inputs under one rule: any
input produces a finite result or raises `ValueError`, never a silent NaN. That pass found
three defects before the first release, including a near-deterministic slice whose interval
covered 7.7% of the time.

650 tests, `mypy --strict`, CI on Linux and macOS across Python 3.10 and 3.13.

## Findings

[88% agreement, 13 points of error](analysis/mt_bench/FINDINGS.md). MT-Bench published
GPT-4's pairwise judgments and human judgments over the same 1814 comparisons. GPT-4 agrees
with the humans 88% of the time and still reports its own win rate 12.7 points above what
they give it. A 9.3-point self-preference survives controlling for the judge exaggerating
the quality spread. `analysis/mt_bench/run.py` downloads the data and prints every number
in the write-up.

## Documentation

Derivations, assumptions and known limits: [`docs/methods/`](docs/methods/).

| page | contents |
| --- | --- |
| [prediction-powered-inference.md](docs/methods/prediction-powered-inference.md) | The estimator, the variance-optimal λ, and the Rogan–Gladen alternative. |
| [confidence-sequences.md](docs/methods/confidence-sequences.md) | Why fixed-sample intervals fail under monitoring; both constructions. |
| [judge-bias-and-slices.md](docs/methods/judge-bias-and-slices.md) | HC3 regression, position bias, per-segment correction. |
| [contamination.md](docs/methods/contamination.md) | The exchangeability permutation test and its blind spot. |

## Examples

```sh
python examples/generate_sample_data.py
for f in examples/0*.py; do python "$f"; done
```

| script | subject |
| --- | --- |
| `01_audit_an_eval.py` | Correcting a judge-scored evaluation. |
| `02_compare_two_models.py` | A 17-point reported improvement against a 9-point real one. |
| `03_monitor_a_release.py` | Live monitoring, and why the cumulative test misses a late regression. |
| `04_detect_judge_drift.py` | An anchor set detecting a changed judge. |
| `05_plan_a_labeling_budget.py` | The cost of a given precision. |
| `06_find_the_regressed_segment.py` | Three segments reported as improved, one regressed 17 points. |

Every figure these scripts print is computed at run time.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New estimators ship with a derivation and a
coverage simulation; new diagnostics ship with a measured false-positive rate.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
