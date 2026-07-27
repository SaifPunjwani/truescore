# Judge bias, and why it must be corrected per segment

The methods behind `truescore.bias` and `truescore.slices`.

## Where judge error lands

An agreement rate of 88% is fine if the 12% is scattered at random. It is dangerous if the
12% is concentrated somewhere, because then a change that moves examples *into* that region
registers as a quality change when nothing about quality moved.

`judge_error_regression` regresses the signed judge error $e_i = f_i - Y_i$ on per-example
covariates. Positive coefficients mean the judge grows more generous as the covariate grows.

Standard errors are HC3 rather than classical, because judge error is heteroscedastic by
construction: $e_i$ is bounded in $\{-1, 0, 1\}$ for binary labels, so its variance depends
on where the judge sits in its own range. HC3 weights each squared residual by
$1/(1-h_i)^2$, where $h_i$ is the leverage:

$$\widehat{\operatorname{Var}}(\hat\beta) = (X^\top X)^{-1}\left(\sum_i \frac{\hat e_i^2}{(1-h_i)^2} x_i x_i^\top\right)(X^\top X)^{-1}$$

The vectorized implementation is cross-checked against an explicit loop written from that
definition in
`tests/test_bias.py::test_hc3_standard_errors_match_an_independent_implementation`. The two
implementations are independent, so agreement between them is evidence about the algebra.

**Collinear designs raise.** A constant or duplicated covariate has no separately
identifiable effect, and a pseudo-inverse would return a meaningless coefficient for it.
`_hc3_regression` checks the rank first and refuses. This was found by a test that expected
a raise and did not get one.

## Position bias

For pairwise judges there is a cleaner test that needs no covariates. Judge each comparison
twice, once in each presentation order. A judge with no position preference picks the same
*option* both times, and therefore picks the first-presented option exactly half the time. A
judge that always picks position one scores 1.0.

The estimate is a pooled proportion over $2n$ judgements with an exact binomial interval
against $\tfrac12$. Both ends are pinned by tests:
`test_position_bias_detects_always_first_judge` and
`test_position_bias_reports_half_for_unbiased_judge`.

## Why one global correction is not enough

The judge's bias is not constant across segments, so a single global correction reorders
them. That is the argument for `truescore.slices`.

The sample data in `examples/data/support_segments.csv` shows it. A support assistant is
evaluated across three segments. The new version writes longer answers on the technical
segment than elsewhere, and the judge rewards length:

| segment | true change | as judged |
| --- | --- | --- |
| billing | +0.2051 | +0.1223 |
| account | +0.1522 | +0.1106 |
| technical | **−0.1663** | **+0.1525** |

On the technical segment the judge inverts the sign of the change. It reports a fifteen-point
improvement where there is a seventeen-point regression. A global correction
estimated across all segments would subtract roughly the same amount everywhere and leave
the ordering intact, so the regression would survive the correction untouched. Correcting
each segment separately recovers −0.1659, against a planted truth of −0.1663.

## Multiplicity

Slicing invites a second error. Twenty segments tested at $\alpha = 0.05$ produce, on
average, one spurious "significant" segment per run even when nothing differs anywhere. That
segment then gets investigated, and a story gets constructed for it.

`compare_slices` applies a correction across the segments tested in one call:

- **Holm** (default) controls the family-wise error rate: the probability of *any* false
  flag stays at $\alpha$. The right default when a flagged segment triggers work.
- **Benjamini–Hochberg** controls the false discovery rate: a known proportion of the
  flagged segments are false. Appropriate for wide exploratory sweeps.
- **None**, for a single pre-registered segment.

`tests/test_slices.py::test_multiplicity_correction_suppresses_spurious_slices` runs 120
replications of twenty null segments and confirms the corrected flag rate stays low while
the unadjusted rate does not.

## Thin segments

Gold labels spread thin across segments. A segment holding eight human labels cannot support
a corrected estimate. Below `min_gold` (default 20), `slices` reports the judge's number,
marks the segment as not corrected, and gives the reason;
`test_thin_slices_are_reported_not_estimated` pins that behavior. Reporting the judge's
number with the reason attached is preferable to publishing a corrected number that eight
labels cannot support.

## References

- MacKinnon & White (1985), "Some heteroskedasticity-consistent covariance matrix estimators
  with improved finite sample properties".
- Zheng et al. (2023), "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
  arXiv:2306.05685. Position and verbosity bias in LLM judges.
- Holm (1979); Benjamini & Hochberg (1995).
