# Contamination testing

The method behind `truescore.contamination`. Its false-positive rate equals the level *by
construction*, with no distributional assumption at all.

## The problem with the usual approaches

A contaminated benchmark reports skill the model does not have, and the score alone cannot
reveal it. A memorized answer and a reasoned answer look identical. The common defences both
have a hole:

- **String matching against the training corpus** requires the training corpus, which for
  any hosted model you do not have.
- **Perplexity thresholds** require a calibration set that is known-clean, which is the
  thing in question, and a threshold nobody can justify.

## The observation

A benchmark is a *sequence* of examples, published in some order. That order carries no
information about the task: the questions would be equally valid shuffled. So under the null
that a model never saw the dataset, the log-likelihood it assigns to the examples
concatenated in canonical order is exchangeable with the log-likelihood under any
permutation of that order.

If the model trained on the dataset as published, the canonical order is not exchangeable
with the others: it is the one it memorized, and it scores unusually high.

## The test

Let $\ell_{\text{canonical}}$ be the log-likelihood of the canonical concatenation and
$\ell_1,\dots,\ell_m$ those of $m$ independently shuffled orders. The p-value is the rank of
the canonical value among them:

$$p = \frac{1 + \#\{j : \ell_j \ge \ell_{\text{canonical}}\}}{m + 1}$$

Under the null all $m+1$ orderings are exchangeable, so the canonical value's rank is
uniform on $\{1, \dots, m+1\}$ and $p$ is uniform on the grid $\{\tfrac{1}{m+1}, \dots, 1\}$.
There is no asymptotics here and no assumption about the distribution of log-likelihoods.
That distribution depends on the model, the tokenizer and the dataset in ways nobody can
characterize.

Two tests measure the calibration. `test_null_false_positive_rate_matches_alpha` confirms
the rejection rate lands *on* 0.05 rather than merely below it, and
`test_null_p_values_are_uniform_on_the_permutation_grid` checks the whole p-value
distribution point by point.

## Resolution, and reading a floor as a finding

With $m$ permutations the smallest achievable p-value is $\tfrac{1}{m+1}$. A test run with
19 permutations cannot reach $p < 0.05$ no matter how contaminated the model is, and cannot
reach 0.01 at all. `ContaminationResult.resolution` reports this and `summary()` prints it.
A p-value sitting exactly at the floor is a statement about the permutation budget, not
about the model.

## Pooling shards

Scoring many permutations of a large dataset is expensive. Splitting the dataset into
disjoint shards, testing each, and pooling recovers power: the shards use different data, so
under the null their p-values are independent and Fisher's statistic

$$-2\sum_{k} \log p_k \;\sim\; \chi^2_{2K}$$

`test_pooling_shards_recovers_power` shows eight shards, none individually significant at
0.05, combining to $p < 0.01$; `test_fisher_combination_is_calibrated_under_the_null`
confirms the combination is itself calibrated.

## What a negative result does not mean

The test detects memorization of the published order. A model trained on the same examples
in a shuffled order, or on paraphrases, or on a derived dataset, can be thoroughly
contaminated and pass this test cleanly. A non-significant result is an absence of evidence,
not evidence of absence. The docstring states this.

The test also requires log-likelihoods, so it applies only to models that expose them. No
amount of statistics removes that limit. Computing the log-likelihoods is the caller's job;
`truescore` never calls a model.

## References

- Oren, Meister, Chatterji, Ladhak, Hashimoto (2023), "Proving Test Set Contamination in
  Black-Box Language Models", arXiv:2310.17623.
- Fisher (1932), *Statistical Methods for Research Workers*.
