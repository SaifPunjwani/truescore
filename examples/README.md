# Examples

All of these use one scenario: a customer-support assistant, two versions (`v3` in
production, `v4` proposed), 4000 shared questions, an LLM judge on every answer, 600
questions labeled by a person.

Three things are true of the judge, and all three are common in practice:

1. It's lenient. It rarely fails a correct answer but passes a lot of wrong ones.
2. It rewards length, regardless of whether the answer is right.
3. v4 writes longer answers than v3, so the judge flatters v4 for the wrong reason.

Generate the data first. It prints the ground truth it planted, so you can check any claim
below against it:

```sh
python examples/generate_sample_data.py
```

## Scripts

| script | question | result |
| --- | --- | --- |
| `01_audit_an_eval.py` | What's the real pass rate? | Judge says 0.8383, truth is 0.7140. The judge's interval misses; the corrected one doesn't. |
| `02_compare_two_models.py` | Did v4 beat v3? | Reported +17 points, real +9. The other 7 are length bias. |
| `03_monitor_a_release.py` | Should we roll back? | Regression at request 600, caught at 867, with the error budget covering the whole run. |
| `04_detect_judge_drift.py` | Did the judge change? | Anchor agreement drops 0.855 to 0.747, with a fingerprint proving the same examples were used. |
| `05_plan_a_labeling_budget.py` | How many labels? | What each precision costs and what a better judge would save. |
| `06_find_the_regressed_segment.py` | Better for everyone? | The judge says all three segments improved. One regressed 17 points. |
| `judge_correction.py` | Why does this work? | The correction on its own, no files needed. |

```sh
for f in examples/0*.py; do python "$f"; done
```

## The same thing from the command line

```sh
truescore doctor examples/data/support_eval.csv

truescore audit examples/data/support_eval.csv \
    --judge judge_passed --gold human_passed \
    --covariate response_tokens --system-name support-v4 --markdown report.md

truescore compare examples/data/support_compare.csv \
    --judge-a v4_judge_passed --judge-b v3_judge_passed \
    --gold-a v4_human_passed --gold-b v3_human_passed

truescore slices examples/data/support_segments.csv --by segment \
    --judge-a v4_judge_passed --judge-b v3_judge_passed \
    --gold-a v4_human_passed --gold-b v3_human_passed

truescore drift examples/data/judge_anchor.csv \
    --baseline judge_may --current judge_june --gold human_passed

truescore monitor examples/data/release_stream.csv \
    --metric passed --baseline 0.88 --window 300

truescore contamination examples/data/contamination_logliks.csv

truescore plan --n-total 4000 --target 0.03 --rate 0.71 \
    --sensitivity 0.97 --specificity 0.47
```

`drift`, `monitor` and `slices` exit 2 when they find something, so a nightly job can fail
the build:

```yaml
- name: judge did not drift
  run: truescore drift anchor.csv --baseline judge_pinned --current judge_today --gold human
```

## Data files

| file | contents |
| --- | --- |
| `support_eval.csv` | 4000 questions: judge verdict on every row, human verdict on 600, response length, difficulty. |
| `support_compare.csv` | Both versions on the same questions, human labels for both. |
| `support_segments.csv` | Both versions split across three segments. v4 regresses on one. |
| `judge_anchor.csv` | 600 frozen anchor examples judged in May and again in June. |
| `release_stream.csv` | 1200 requests from a release that regressed halfway through. |
| `contamination_logliks.csv` | One canonical ordering and 199 shuffles, for a clean model. |

The sparse `human_passed` column, filled on some rows and blank on the rest, is the normal
shape of real eval data. `truescore.io` reads it directly.
