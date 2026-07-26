# Examples

A single scenario runs through all of these: a customer-support assistant, two versions
(`v3` in production, `v4` proposed), 4000 shared questions, an LLM judge on every answer,
and 600 questions hand-labeled by a contractor.

Three things are true of the judge, and all three are ordinary:

1. It is **lenient** — it rarely fails a correct answer but passes many wrong ones.
2. It **rewards length**, independently of whether the answer is right.
3. **v4 writes longer answers than v3**, so the judge flatters v4 for a reason that has
   nothing to do with quality.

Start by generating the data. It writes to `examples/data/` and prints the ground truth it
planted, so every claim below can be checked against it:

```sh
python examples/generate_sample_data.py
```

## The scripts

| script | question | what it demonstrates |
| --- | --- | --- |
| `01_audit_an_eval.py` | What is the pass rate, really? | The judge says 0.8383; the truth is 0.7140. The judge's interval does not contain the truth; the corrected one does. |
| `02_compare_two_models.py` | Did v4 beat v3? | Reported +17 points, real +9. The other 7 are length bias, measured and attributed. |
| `03_monitor_a_release.py` | Should we roll back? | A regression at request 600, caught at 867 — with an error budget covering the entire run, however often it was checked. |
| `04_detect_judge_drift.py` | Did the judge change? | Anchor-set agreement falls 0.855 → 0.747 between two runs, with a fingerprint proving the same examples were used. |
| `05_plan_a_labeling_budget.py` | How many labels do we buy? | What each precision costs, what a better judge would save, and what an eval that size can resolve. |
| `06_find_the_regressed_segment.py` | Better for *everyone*? | The judge says all three segments improved. One regressed by 17 points; the corrected analysis recovers it at p = 2e-07. |
| `judge_correction.py` | Why does any of this work? | The correction in isolation, on generated data, no files required. |

Run them in order; each is self-contained and takes a few seconds.

```sh
for f in examples/0*.py; do python "$f"; done
```

## The same thing from the command line

Every example has a CLI equivalent, and the CLI is the form you would put in CI:

```sh
truescore audit examples/data/support_eval.csv \
    --judge judge_passed --gold human_passed \
    --covariate response_tokens --system-name support-v4 --markdown report.md

truescore compare examples/data/support_compare.csv \
    --judge-a v4_judge_passed --judge-b v3_judge_passed \
    --gold-a v4_human_passed --gold-b v3_human_passed

truescore drift examples/data/judge_anchor.csv \
    --baseline judge_may --current judge_june --gold human_passed

truescore monitor examples/data/release_stream.csv \
    --metric passed --baseline 0.88 --window 300

truescore contamination examples/data/contamination_logliks.csv

truescore plan --n-total 4000 --target 0.03 --rate 0.71 \
    --sensitivity 0.97 --specificity 0.47

truescore slices examples/data/support_segments.csv --by segment \
    --judge-a v4_judge_passed --judge-b v3_judge_passed \
    --gold-a v4_human_passed --gold-b v3_human_passed
```

`drift` and `monitor` exit **2** when they find something, so a nightly job can fail the
build on a judge that changed underneath you:

```yaml
- name: judge did not drift
  run: truescore drift anchor.csv --baseline judge_pinned --current judge_today --gold human
```

## The data files

| file | contents |
| --- | --- |
| `support_eval.csv` | 4000 questions: judge verdict on every row, human verdict on 600, response length, difficulty. |
| `support_compare.csv` | Both versions on the same questions, with human labels for both. |
| `judge_anchor.csv` | 600 frozen anchor examples judged in May and again in June. |
| `release_stream.csv` | 1200 requests from a release that regressed at the halfway point. |
| `support_segments.csv` | The same two versions split across three support segments, where v4 regresses on one of them. |
| `contamination_logliks.csv` | One canonical ordering and 199 shuffles, for a model that never saw the data. |

The sparse `human_passed` column — filled on a few rows, blank on the rest — is not a
defect in the data. It is exactly the shape prediction-powered inference is built for, and
`truescore.io` reads it directly.
