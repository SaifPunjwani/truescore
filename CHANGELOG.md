# Changelog

## 0.2.0 — 2026-07-25

Everything needed to run this against a real evaluation, rather than only from a notebook.

### Added

- **`truescore.sequential`** — confidence sequences valid at every sample size at once
  (empirical-Bernstein and normal-mixture Hoeffding), so a metric can be watched
  continuously and acted on immediately. `windowed_exclusion` detects a regression that
  starts after a healthy period, which a cumulative sequence cannot: the healthy prefix
  holds the running mean up long after the service degraded. One-sided budgeting for
  monitors that only look in one direction.
- **`truescore.drift`** — paired comparison of two judge runs over a fingerprinted anchor
  set, reporting both the agreement change and the label-flip rate. The flip rate catches
  a judge that rewrote a third of its verdicts while its accuracy stayed flat.
- **`truescore.contamination`** — the exchangeability permutation test (Oren et al.) for
  whether a model memorized an evaluation set in its published order, with Fisher pooling
  across shards. Exact, and verified calibrated to the nominal level rather than merely
  conservative.
- **`truescore.io`** — CSV and JSON Lines ingestion for the shape real eval output has: a
  judge verdict on every row and a human verdict on the few that were labeled.
- **`truescore.cli`** — `audit`, `compare`, `drift`, `monitor`, `plan`, `contamination`
  and `agreement`, with exit codes that separate a finding (2) from a failure to run (1),
  so a pipeline can gate on drift without also failing on a typo.
- **Examples on realistic data** — a support-assistant scenario where the judge is lenient
  and rewards length, and v4 writes longer answers than v3. The judge reports a 17-point
  improvement; the truth is 9. Every printed number is computed at run time.

### Fixed

Found by the adversarial fuzz pass, which pairs every nasty input with every other:

- NaN and infinity flowed through the continuous estimators and the paired bootstrap into
  reported numbers. Finiteness is now enforced at the boundary in `to_1d_array`, so no
  quantity in the library can carry a NaN past validation.
- The Wilson interval could exclude its own point estimate at `successes == n`, where
  rounding put the upper limit a hair below 1.
- `ppi_estimate` returned a NaN interval when exactly one unlabeled example remained, a
  case where the unlabeled variance term is not estimable. It now raises and points at
  `gold_only_estimate`.

## 0.1.0 — 2026-07-25

Initial release: the estimator suite.

- `agreement` — Wilson intervals, Cohen's κ, Gwet's AC1 (stable where κ collapses under
  class imbalance), Krippendorff's α matching the published canonical example.
- `correct` — prediction-powered inference (PPI++) with variance-minimizing λ, and the
  Rogan–Gladen misclassification correction with a delta-method interval.
- `bias` — HC3-robust regression of judge error on covariates, and the two-order
  position-bias test. Collinear designs raise rather than being absorbed by a
  pseudo-inverse.
- `compare` — McNemar exact and mid-p, paired bootstrap, sign-flip permutation,
  PPI-corrected paired comparison, Holm and Benjamini–Hochberg.
- `power` — gold-label budgets under PPI, minimum detectable effect, required sample size.
- `report` — the JSON and markdown artifact recording estimator, assumptions, and what the
  naive number would have said.
