# Changelog

## 0.5.0 — 2026-07-26

First public release.

### Added

- **`truescore doctor`** — point it at an evaluation file and it profiles every column,
  lists the commands you can run today, says what is blocked and how many human labels
  would unblock it, and scans every numeric column for judge bias with a Holm correction.
  On the sample data it finds both planted biases without being told to look.
- Landing page, contributing and security policies, issue and PR templates.

### Changed

- Licensed Apache-2.0 and published to PyPI.
- README, contributing guide and examples guide rewritten in plainer prose. No technical
  claim or number changed.

## 0.4.1 — 2026-07-26

### Fixed

- **`ppi_estimate` returned an invalidly narrow interval for a near-deterministic
  subgroup.** When a slice is right ~99.7% of the time, a gold sample of a few dozen
  labels comes back entirely 1s more often than not; the sample variance is then exactly
  zero and the asymptotic interval collapses to width zero. Measured coverage in that
  regime was **7.7%** against a nominal 95%. The estimator now widens to the exact
  interval whenever one class has fewer than five observations, and says so in `method`.
  Coverage in the same scenario is now above 0.93, while PPI keeps its full advantage
  (42% tighter than gold-only) where the normal approximation is sound.
- `WeightedEstimate` gained the `half_width` property every other result type already had.

### Investigated and rejected

- **Stratified label allocation.** An initial measurement suggested that concentrating
  human labels in the judge's uncertain band bought the same precision from 55% fewer
  labels. That result was an artifact of the interval bug above: the "saving" came from
  invalidly narrow intervals in the confident strata. Re-measured with correct intervals,
  Neyman allocation delivers +1% to +3% in favourable scenarios and is materially *worse*
  in the near-deterministic case that motivated it, because starving a high-weight stratum
  of labels leaves genuine uncertainty the exact interval then has to report. The module
  was removed rather than shipped with a claim its own measurements do not support.

## 0.4.0 — 2026-07-26

### Added

- **`truescore.weighting`** — post-stratified estimation, closing the one assumption every
  report could previously only disclaim: that the evaluation set looks like production.
  Evaluation sets are curated, and curation over-samples hard cases. Given the production
  mix across a few strata, each stratum is estimated with PPI and recombined with
  production's weights. On a set that over-samples hard questions three to one, the raw
  number reads 0.745 where production would see 0.846; the weighted estimate recovers
  0.854 with the truth inside its interval. Coverage is simulated, and the two limits that
  have no statistical fix -- wrong weights, and bias *within* a stratum -- are carried in
  the report's `assumptions` rather than left to the reader.
- A composite **GitHub Action**, so a nightly job can fail on evidence: exit 2 becomes a
  build failure unless `fail-on-finding` is false.
- **`docs/methods/`** — derivations for PPI, confidence sequences, judge bias and slicing,
  and the contamination test, each stating what it does *not* guarantee and naming the test
  that enforces every claim.

## 0.3.0 — 2026-07-26

### Added

- **`truescore.slices`** — the question that follows every launch decision: v4 wins
  overall, but is there a segment it makes worse? Per-segment corrected estimates and
  comparisons, with multiplicity control across segments (Holm by default) and an honest
  refusal for segments holding too few human labels to support a number. In the sample
  data the judge reports an improvement on all three support segments while v4 has in
  fact regressed on one by 17 points; per-segment correction recovers `-0.1659` against a
  planted truth of `-0.1663`.
- `truescore slices` on the command line, exiting 2 when a segment regressed.
- `examples/06_find_the_regressed_segment.py` and a segmented sample dataset.

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
