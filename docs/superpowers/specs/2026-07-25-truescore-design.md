# truescore — design

**Status:** approved by delegation (Saif asked me to decide in his absence, 2026-07-25).
**Repository:** private until Saif reviews. No package publication, no announcements.

## 1. The problem

A team ships an LLM feature. To measure it they run 200–2000 examples through an
LLM judge and report one number: "92% pass rate." Every decision — ship, roll back,
pick model A over model B — hangs on that number. The number is not defensible, for
three separate reasons, each with a known statistical remedy that almost nobody applies:

1. **The judge is a biased, noisy instrument.** It agrees with careful human labels
   perhaps 80–90% of the time, and its errors are not symmetric: judges systematically
   favor longer answers, answers in the first position, and answers from their own model
   family. The judge's 92% is a measurement of the judge, not of the system.
2. **n is small, and nobody reports intervals.** At n=200 a 92% score carries roughly a
   ±3.8-point 95% interval. A "1.5-point improvement" between two runs is, in the
   overwhelming majority of cases, indistinguishable from noise. Teams ship on it anyway.
3. **Peeking inflates error.** Eval dashboards are watched continuously; each look is an
   implicit test. Fixed-sample confidence intervals are invalid under repeated peeking,
   so "we saw a regression" is frequently a false alarm.

The correction for (1) and (2) jointly is well-established outside ML: when you have a
large set of cheap, imperfect labels and a small set of expensive, trusted labels, you
can produce an estimate of the *true* quantity with valid coverage — tighter than using
the trusted labels alone, and unbiased unlike using the cheap labels alone. That is
**prediction-powered inference** (Angelopoulos et al., *Science* 2023) and, in the binary
special case, the **Rogan–Gladen** correction long used for imperfect diagnostic tests in
epidemiology. Neither is standard practice in LLM evaluation.

truescore packages these methods for LLM evaluation, and — critically — proves its own
guarantees by simulation in CI.

## 2. What truescore is and is not

**Is:** a statistics library that takes labels you already have — judge labels on many
examples, human labels on a few — and returns defensible estimates: bias-corrected point
estimates, valid confidence intervals, honest comparisons between systems, and a report
artifact recording every assumption.

**Is not:** an eval runner, a prompt framework, a judge, a dataset, or an observability
platform. It never calls a model. It has no opinion about how you obtained your labels.
This boundary is what keeps it composable with Braintrust, LangSmith, Arize, lm-eval-harness,
or an internal harness — truescore consumes their outputs rather than competing with them.

**Non-goals for v0.1:** contamination detection, sequential/anytime-valid monitoring, and
judge-drift changepoint detection are deferred to v0.2 (§8). Rank aggregation across many
systems (leaderboard-style) is out of scope entirely.

## 3. Architecture

Pure functions over numpy arrays, organized so each module answers one question. No
global state, no I/O except the report writer, no network. Dependencies: `numpy` and
`scipy` only.

```
truescore/
├── _validation.py     shared shape/dtype/label validation, one error-message style
├── agreement.py       how good is my judge?
├── correct.py         what is the true score, given an imperfect judge?
├── bias.py            what is my judge biased by?
├── compare.py         is system A actually better than system B?
├── power.py           how many labels do I need?
└── report.py          the defensible artifact
```

Data model, used everywhere: labels are 1-D integer or boolean arrays over a common
example index. `judge` is the cheap label available for all `n` examples; `gold` is the
trusted label available for a subset, identified by an index array. Binary is the primary
case (pass/fail); ordinal and continuous scores are supported where the estimator admits
them, and rejected with a clear error where they do not.

### 3.1 `agreement.py` — judge quality

Estimates of how well the judge reproduces gold labels, with intervals:

- accuracy, sensitivity, specificity, precision, recall, and the full confusion matrix,
  each with Wilson intervals (not normal-approximation, which fails near 0 and 1)
- Cohen's κ, and **Gwet's AC1** — κ collapses under prevalence imbalance (the "kappa
  paradox"), and LLM eval sets are usually imbalanced, so reporting κ alone is misleading
- Krippendorff's α for the multi-rater / ordinal case
- bootstrap intervals for every agreement coefficient (BCa where feasible)

### 3.2 `correct.py` — the true score

The core value. Three estimators of a target metric (a proportion in v0.1):

- `judge_only` — the naive estimate, computed only so the report can show what it
  would have said, and by how much it is wrong
- `gold_only` — the unbiased but wide classical estimate on the labeled subset
- `ppi` — prediction-powered estimate combining both: the gold-only estimate corrected by
  the judge's measured error on the labeled subset, with a tuning parameter λ (PPI++)
  chosen to minimize variance. Provably valid coverage regardless of judge quality; never
  meaningfully wider than gold-only, and much tighter when the judge is good.
- `rogan_gladen` — the classical misclassification correction
  `p_true = (p_obs + spec − 1) / (sens + spec − 1)`, with a delta-method interval that
  propagates uncertainty in the estimated sensitivity and specificity, plus the
  documented failure mode (`sens + spec ≤ 1` ⇒ undefined, raises)

Every estimator returns the same `Estimate` structure: point, interval, method name,
n_total, n_gold, and the assumptions it relies on, so the report can be assembled
mechanically.

### 3.3 `bias.py` — what the judge is biased by

Regression of judge error (`judge − gold` on the labeled subset) on example covariates,
with HC3 heteroscedasticity-robust standard errors:

- length bias: does the judge favor longer responses, and by how much per 100 tokens
- position bias: for pairwise judges, the standard test — the same pair scored in both
  orders; the asymmetry rate is an unbiased estimate of position bias with an exact
  binomial interval
- arbitrary user-supplied covariates (self-preference, formatting, language, difficulty
  bucket), each returning effect size, robust SE, interval, and a p-value

### 3.4 `compare.py` — is A better than B

Paired comparison on shared examples, which is the only correct framing (unpaired tests
throw away the pairing and lose most of the power):

- McNemar's exact and mid-p tests for paired binary outcomes
- paired bootstrap for arbitrary metrics
- exact permutation test for small n
- `ppi_compare` — paired comparison using the corrected estimate, so the comparison is
  between true scores rather than judge scores
- multiplicity control (Holm and Benjamini–Hochberg) for the realistic case of comparing
  many systems or many slices at once

### 3.5 `power.py` — how many labels

The question every team actually asks, answered in both directions:

- minimum detectable effect at given n, α, power
- required n for a target effect
- **required gold-label budget**: given judge accuracy and a labeling cost, how many human
  labels buy an interval of a target width under PPI — the practical planning tool, and the
  thing that makes the value of a better judge concrete in dollars

### 3.6 `report.py` — the artifact

An `EvalReport` frozen dataclass assembled from the above, serializable to JSON and
rendered to markdown, recording: what was measured, on how many examples, with how many
gold labels, under which estimator, with which assumptions, and what the naive number
would have been. This is a developer convenience in the OSS core and the compliance
deliverable in the commercial tier — the same object, different packaging.

## 4. Verification strategy

The library's claims are statistical, so the tests must be statistical. This is the
differentiator and it is CI-cheap (numpy, no GPU, no network).

1. **Analytic golden tests.** Hand-computed cases for every estimator: a 2×2 confusion
   matrix whose κ, AC1, Wilson interval, and Rogan–Gladen correction are worked out by
   hand in the derivation docs.
2. **Coverage simulations — the central test class.** For each interval estimator:
   simulate a known ground truth, draw 1000+ replications, and assert that the nominal
   95% interval covers the truth within a binomial tolerance of 95%. An interval library
   that does not verify its own coverage is asserting exactly what it should be proving.
3. **Efficiency assertions.** PPI intervals must be no wider than gold-only intervals
   when the judge is informative, and must not break coverage when the judge is
   adversarially bad (the property that makes PPI safe to adopt).
4. **Property tests (Hypothesis).** Invariances: label permutation, class relabeling,
   monotonicity of interval width in n, degenerate inputs raising rather than returning
   NaN.
5. **Cross-implementation checks.** Against `scipy.stats` exact binomial and McNemar
   where they exist, and against closed forms elsewhere.

## 5. Why this is defensible

The crowded market is *running* evals. truescore does not run evals; it audits the
numbers evals produce, which none of the platforms do because it requires a statistical
posture they have not taken. The moat is not the code — PPI is published — it is being
the reference implementation with verified coverage, the vocabulary ("what is your
gold-label budget?"), and the report artifact that regulators and internal risk teams
learn to expect.

## 6. Commercial ladder

Recorded for planning; no action taken without Saif.

1. **OSS core, free.** Adoption and credibility. Launch evidence: re-analyze public
   judge-based leaderboards with intervals and paired tests, and publish how many reported
   rankings do not survive. A finding, not a marketing claim.
2. **Design partners.** AI product teams at Series B–D companies: budget, no in-house
   statistician, no procurement friction.
3. **Hosted tier ($200–2k/mo).** Continuous monitoring, recalibration when a provider
   silently updates a judge model, history, team dashboards — the things a library cannot do.
4. **Enterprise ($25–100k/yr).** Audit-grade eval evidence mapped to EU AI Act accuracy
   and robustness documentation obligations, SSO, on-prem.

**Gate before any revenue:** written clearance from NVIDIA on outside activity. Publishing
open source and selling software are different things under most employment agreements.

## 7. Risks, honestly

- **Adoption friction:** PPI requires *some* human labels. Teams with zero gold labels get
  only the agreement and bias modules. Mitigation: `power.gold_budget` makes the ask
  concrete and small ("120 labels buys you a ±2.5-point interval").
- **Category confusion:** if it reads as "another eval tool," it dies. Positioning must
  stay "audit layer for eval numbers," and the library must never grow an eval runner.
- **Statistical rigor is a hard sell** to teams that prefer a single number. The counter is
  the demo: show a real "improvement" evaporating under a paired test.

## 8. Deferred to v0.2

Anytime-valid confidence sequences for continuous monitoring (e-value / betting
martingales), judge-drift changepoint detection against a frozen anchor set, and
contamination testing. Each is independently valuable and none blocks v0.1.
