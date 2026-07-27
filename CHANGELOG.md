# Changelog

## 0.7.4 - 2026-07-27

### Fixed

- **`paired_bootstrap` reported p = 0 when two systems agreed on every example.** The
  two-sided p-value measured one tail as the complement of the other. When every paired
  difference is exactly zero, every resample is exactly zero, and all of that mass landed
  in one tail: perfect agreement came back as the strongest possible evidence of a
  difference. Both tails now count the mass sitting at zero, so identical systems return
  p = 1. Found by the RewardBench study, where a subset both judges scored 1.000 on
  produced `p=0.000` beside a gap of `+0.000` and survived a Holm correction.

### Added

- **A second study, `analysis/rewardbench`.** RewardBench ranks LLM judges by accuracy
  against human-verified answers on 2985 preference pairs, all judges scored on the same
  examples. Four of the eight adjacent gaps do not survive a paired test with a Holm
  correction, so nine published ranks are really four tiers. The top two are 0.37 points
  apart and not distinguishable, and disagree on 10 of 23 subsets in both directions by up
  to 22 points. Uses only truescore and the standard library.

## 0.7.3 - 2026-07-27

### Fixed

- **λ was chosen from too little data.** The PPI tuning parameter estimated `Var(f)` over
  the gold-labeled subset alone. Judge labels exist for every example, so that discarded
  most of the available data and left λ noisy in exactly the regime this library is for: a
  few dozen human labels against thousands of examples. Pooling both sets moved the point
  estimate by up to 5.5e-2 at 50 gold labels. Coverage was never at risk, since validity
  holds for any fixed λ, but the intervals were wider than they needed to be.

### Added

- **Conformance against the reference implementation.** `tests/test_conformance.py` checks
  the corrected estimate against `ppi_py`, released with the papers this estimator comes
  from, across six regimes. Writing it is what found the λ defect above. The point estimate
  now agrees to 1e-12; the interval is asserted to be never narrower than the reference and
  no more than 5% wider, which is a directional claim rather than a tolerance because a
  tolerance would have permitted the one outcome that matters. Installed via the new
  `conformance` extra, kept out of `dev` because it pulls in numba, pandas and
  scikit-learn; the test skips itself when absent.
- `Estimate.standard_error`, populated by `ppi_estimate`.

### Changed

- The covariance inside λ now divides by `n` rather than `n-1`, matching the reference
  exactly. λ is a tuning parameter and not an estimand, so unbiasedness there buys nothing
  while exact agreement removes a question about which implementation to believe. The
  variance inside the *interval* keeps `n-1`, because that one is an estimand and the
  unbiased estimate is the conservative one.
- CONTRIBUTING records that mypy has to be run against Python 3.10 as well, since numpy's
  stubs differ enough between the two matrix legs that it can pass on one and fail on the
  other.

## 0.7.2 - 2026-07-27

Completes the clustering work in 0.7.1, which fixed `audit` and left the other commands
with the problem it had just named.

### Added

- **`clusters=` on `ppi_compare` and `paired_bootstrap`, `--cluster-column` on `compare`.**
  A paired comparison over correlated rows understates its variance exactly as a single
  score does. The bootstrap resamples whole clusters rather than rows, which is the unit
  the sampling design actually draws; on singleton clusters it reduces to the row bootstrap.
- **`Estimate.standard_error`.** Callers were recovering it by dividing the half width by a
  normal quantile, which stopped being right the moment clustered intervals began using a t
  quantile. `ppi_compare` now takes the standard error from the estimator that computed it,
  so its p-value and its interval cannot disagree.
- **`compare` swaps McNemar for the cluster bootstrap** when a grouping is declared.
  McNemar conditions on discordant pairs assumed independent, which they are not once
  several rows describe one example, so the uncorrected line would otherwise have been the
  only wrong number left in the output.

### Changed

- **Every command that opens a file now shares one reader**, so format detection and the
  repeated-identifier warning reach `compare`, `slices` and `drift` rather than only
  `audit`. A warning that depends on which subcommand you picked is worse than none,
  because its silence means nothing.
- The straddling-cluster error names the three ways out instead of two, since averaging
  each cluster to one row is what most people want and it was the one left unsaid.

## 0.7.1 - 2026-07-27

### Fixed

- **Multi-epoch Inspect logs produced intervals that were too narrow.** 0.7.0 emitted one
  row per (sample, epoch), so a run with `--epochs 5` handed every estimator five
  correlated rows per sample. Scored as independent draws they shrink the interval by about
  sqrt(5) more than the data supports, and a nominal 95% interval covers 86% of the time.
  Measured, not estimated: `test_clustered_data_undercovers_until_clusters_are_declared`.
  The adapter now averages epochs to one row per sample, which is the unit carrying one
  independent observation, and says so in its notes.
- **A column that never varies is no longer offered as a covariate or a segment.** Inspect
  logs carry an `epochs` column holding the same integer on every row; suggesting it as a
  bias covariate sent the reader to a command that raises on a collinear design. A verdict
  column keeps its reading, since a judge that passed everything still has a pass rate
  worth monitoring.

### Added

- **`clusters=` on `ppi_estimate` and `gold_only_estimate`, `--cluster-column` on `audit`.**
  Cluster-robust variance for the general case: several turns of one conversation, several
  questions from one document, repeated runs. Residuals are summed within a cluster before
  squaring, so within-cluster correlation is unrestricted. With one observation per cluster
  it reduces exactly to the independent formula, which is pinned by a test rather than
  asserted. For PPI a cluster must fall wholly inside or wholly outside the labeled set,
  because the estimator's two terms are only independent when they share no cluster; a
  straddling cluster raises with an explanation instead of returning a confident number.
- **`audit` warns when the identifier column repeats** and no grouping was declared, since
  the resulting interval looks entirely healthy while being wrong.

## 0.7.0 - 2026-07-27

### Added

- **Native eval-tool formats.** `truescore.adapters` reads the output of Inspect AI,
  promptfoo, DeepEval and lm-evaluation-harness directly, identifies the judge column, and
  flattens the records. Two of these formats a plain row reader cannot open at all: an
  Inspect log and a promptfoo `--output` file are single JSON objects with the records
  nested inside. Every shape was taken from the tool's own serialization code, so the
  fixtures in `tests/test_adapters.py` describe what the tools write rather than what
  seemed likely. Inspect's `C`, `P`, `I` and `N` map to 1.0, 0.5, 0.0 and 0.0, reproducing
  its `value_to_float` so a corrected number stays comparable to the accuracy Inspect
  reported.
- **`--gold-file`.** Human labels can live in their own file and be joined to the
  evaluation by identifier, which is where they actually live: the eval tool writes
  verdicts, and somebody labels a subset in a spreadsheet afterwards. Requiring a hand
  merge first was the step that stopped people getting as far as a number. The join
  refuses rather than guessing when the key matches nothing, when an identifier repeats in
  the evaluation, or when one example carries two human verdicts, and it reports labels
  that landed nowhere instead of dropping them.
- **`--judge` is now optional** when the format is recognized.
- **`response_chars`** is derived from the response text for every supported format, so
  the verbosity-bias regression runs on any eval output without adding a column.
- **`doctor` opens eval-tool output**, and names the format it recognized.

### Changed

- Format detection requires two independent markers before claiming a file. One shared key
  name is a coincidence, and claiming a hand-rolled JSONL would rename its columns out from
  under whoever wrote it.
- Normalization carries through keys it does not recognize, under their own names. A human
  verdict pasted into an eval export is the most valuable column in the file, and dropping
  it silently would have been the worst kind of bug this change could introduce.

## 0.6.0 - 2026-07-26

### Added

- **Nested field paths.** Column names may be dotted paths, so output from promptfoo,
  lm-eval-harness or an in-house harness can be read where it lies:
  `--judge gradingResult.pass`, `--covariate response.tokenUsage.completion`. `doctor`
  flattens nested JSON when profiling and hands back commands carrying the paths. A flat
  CSV header containing a dot is still looked up directly.
- **`EvalReport.to_html` and `audit --html`.** A single self-contained file with no
  external stylesheet, script or font, so a report survives being emailed and opened
  later. Caller-supplied names are escaped.
- **Graded rubric support.** `graded_agreement` and `quadratic_weighted_kappa` for judges
  that emit a 1-5 score rather than pass/fail, where binary metrics do not apply and
  unweighted agreement treats a 1-vs-5 disagreement the same as a 3-vs-4. Kappa is
  verified against a hand-computed 3x3 confusion matrix. `doctor` now recognises a rubric
  column and reports what a graded file supports.

## 0.5.0 - 2026-07-26

First public release.

### Added

- **`truescore doctor`**. Point it at an evaluation file and it profiles every column,
  lists the commands you can run today, says what is blocked and how many human labels
  would unblock it, and scans every numeric column for judge bias with a Holm correction.
  On the sample data it finds both planted biases with no configuration.
- Landing page, contributing and security policies, issue and PR templates.

### Changed

- Licensed Apache-2.0 and published to PyPI.
- README, contributing guide and examples guide rewritten in plainer prose. No technical
  claim or number changed.

## 0.4.1 - 2026-07-26

### Fixed

- **`ppi_estimate` returned an invalidly narrow interval for a near-deterministic
  subgroup.** When a slice is right ~99.7% of the time, a gold sample of a few dozen
  labels comes back entirely 1s more often than not; the sample variance is then exactly
  zero and the asymptotic interval collapses to width zero. Measured coverage in that
  regime was 7.7% against a nominal 95%. The estimator now widens to the exact
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
  of labels leaves real uncertainty that the exact interval then has to report. The module
  was removed instead of shipping with a claim its own measurements do not support.

## 0.4.0 - 2026-07-26

### Added

- **`truescore.weighting`**. Post-stratified estimation, closing the one assumption every
  report could previously only disclaim: that the evaluation set looks like production.
  Evaluation sets are curated, and curation over-samples hard cases. Given the production
  mix across a few strata, each stratum is estimated with PPI and recombined with
  production's weights. On a set that over-samples hard questions three to one, the raw
  number reads 0.745 where production would see 0.846; the weighted estimate recovers
  0.854 with the truth inside its interval. Coverage is simulated. The two limits that
  have no statistical fix (wrong weights, and bias *within* a stratum) are carried in the
  report's `assumptions`.
- A composite **GitHub Action**, so a nightly job can fail on a finding: exit 2 becomes a
  build failure unless `fail-on-finding` is false.
- **`docs/methods/`**: derivations for PPI, confidence sequences, judge bias and slicing,
  and the contamination test, each stating what it does *not* guarantee and naming the test
  that enforces every claim.

## 0.3.0 - 2026-07-26

### Added

- **`truescore.slices`**, for the case that follows every launch decision: v4 wins overall
  but may be worse on some segment. Per-segment corrected estimates and
  comparisons, with multiplicity control across segments (Holm by default) and an honest
  refusal for segments holding too few human labels to support a number. In the sample
  data the judge reports an improvement on all three support segments while v4 has in
  fact regressed on one by 17 points; per-segment correction recovers `-0.1659` against a
  planted truth of `-0.1663`.
- `truescore slices` on the command line, exiting 2 when a segment regressed.
- `examples/06_find_the_regressed_segment.py` and a segmented sample dataset.

## 0.2.0 - 2026-07-25

Everything needed to run this against a real evaluation, rather than only from a notebook.

### Added

- **`truescore.sequential`**: confidence sequences valid at every sample size at once
  (empirical-Bernstein and normal-mixture Hoeffding), so a metric can be watched
  continuously and acted on immediately. `windowed_exclusion` detects a regression that
  starts after a healthy period, which a cumulative sequence cannot: the healthy prefix
  holds the running mean up long after the service degraded. One-sided budgeting for
  monitors that only look in one direction.
- **`truescore.drift`**: paired comparison of two judge runs over a fingerprinted anchor
  set, reporting both the agreement change and the label-flip rate. The flip rate catches
  a judge that rewrote a third of its verdicts while its accuracy stayed flat.
- **`truescore.contamination`**: the exchangeability permutation test (Oren et al.) for
  whether a model memorized an evaluation set in its published order, with Fisher pooling
  across shards. Exact, and verified calibrated to the nominal level rather than merely
  conservative.
- **`truescore.io`**. CSV and JSON Lines ingestion for the shape real eval output has: a
  judge verdict on every row and a human verdict on the few that were labeled.
- **`truescore.cli`**: `audit`, `compare`, `drift`, `monitor`, `plan`, `contamination`
  and `agreement`, with exit codes that separate a finding (2) from a failure to run (1),
  so a pipeline can gate on drift without also failing on a typo.
- **Examples on realistic data**: a support-assistant scenario where the judge is lenient
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

## 0.1.0 - 2026-07-25

Initial release: the estimator suite.

- `agreement`: Wilson intervals, Cohen's κ, Gwet's AC1 (stable where κ collapses under
  class imbalance), Krippendorff's α matching the published canonical example.
- `correct`: prediction-powered inference (PPI++) with variance-minimizing λ, and the
  Rogan–Gladen misclassification correction with a delta-method interval.
- `bias`: HC3-robust regression of judge error on covariates, and the two-order
  position-bias test. Collinear designs raise rather than being absorbed by a
  pseudo-inverse.
- `compare`: McNemar exact and mid-p, paired bootstrap, sign-flip permutation,
  PPI-corrected paired comparison, Holm and Benjamini–Hochberg.
- `power`: gold-label budgets under PPI, minimum detectable effect, required sample size.
- `report`: the JSON and markdown artifact recording estimator, assumptions, and what the
  naive number would have said.
