# Prediction-powered inference

The estimator behind `truescore.correct.ppi_estimate`, and the reason a judge can make an
evaluation *more* precise without making it wrong.

## The setting

An evaluation has $N$ examples. A judge produces a cheap label $f_i$ for every one of
them. A human produces a trusted label $Y_i$ for a subset $L$ of size $n \ll N$. We want
$\theta = \mathbb{E}[Y]$, the rate a careful human would report if they labeled everything.

Two obvious estimators, both unsatisfying:

$$\hat\theta_{\text{judge}} = \frac{1}{N}\sum_{i=1}^{N} f_i
\qquad\qquad
\hat\theta_{\text{gold}} = \frac{1}{n}\sum_{i \in L} Y_i$$

The first is precise and biased: it converges to $\mathbb{E}[f]$, the judge's rate, not the
system's. Extra unlabeled data does not fix that. More examples buy a tighter interval
around the wrong number. The second is unbiased and wide, because it throws away every
unlabeled example.

## The estimator

Write $L$ for the labeled set and $U$ for the unlabeled set, $|U| = N_u$. For a tuning
parameter $\lambda$,

$$\hat\theta_\lambda \;=\; \underbrace{\frac{\lambda}{N_u}\sum_{i \in U} f_i}_{\text{borrow from unlabeled data}} \;+\; \underbrace{\frac{1}{n}\sum_{i \in L}\bigl(Y_i - \lambda f_i\bigr)}_{\text{measure and remove the judge's bias}}$$

The second term does the work. It is an unbiased estimate of $\mathbb{E}[Y - \lambda f]$,
measured on the examples where both labels exist. Adding it back to $\lambda\,\mathbb{E}[f]$
recovers $\mathbb{E}[Y]$ exactly, whatever the judge does:

$$\mathbb{E}[\hat\theta_\lambda] = \lambda\,\mathbb{E}[f] + \mathbb{E}[Y] - \lambda\,\mathbb{E}[f] = \theta$$

**Unbiasedness does not depend on the judge being good.** It depends only on the gold
subset being a random sample. A judge that is wrong half the time, or systematically
lenient, or anti-correlated with the truth, still leaves the estimator centred on $\theta$.
Judge quality affects only how *wide* the interval is. So the method can be adopted before
the judge's quality is known.
`tests/test_correct.py::test_ppi_covers_at_nominal_rate_under_simulation` runs the coverage
simulation at three judge qualities, down to one barely better than chance.

## Variance and the choice of λ

Since $L$ and $U$ are disjoint, the two terms are independent:

$$\operatorname{Var}(\hat\theta_\lambda) = \frac{\operatorname{Var}(Y - \lambda f)}{n} + \frac{\lambda^{2}\operatorname{Var}(f)}{N_u}$$

Differentiating in $\lambda$ and solving:

$$\frac{\partial}{\partial\lambda}\left[\frac{\operatorname{Var}(Y) - 2\lambda\operatorname{Cov}(Y,f) + \lambda^{2}\operatorname{Var}(f)}{n} + \frac{\lambda^{2}\operatorname{Var}(f)}{N_u}\right] = 0$$

$$\lambda^{\star} = \frac{\operatorname{Cov}(Y, f)}{\operatorname{Var}(f)\left(1 + n/N_u\right)}$$

Three cases worth reading off that expression:

| judge | $\operatorname{Cov}(Y,f)$ | $\lambda^\star$ | consequence |
| --- | --- | --- | --- |
| informative | large | near 1 | interval much tighter than gold-only |
| uninformative | ≈ 0 | ≈ 0 | reduces to the classical estimator |
| anti-correlated | negative | clipped to 0 | reduces to the classical estimator |

At $\lambda = 0$ the estimator *is* $\hat\theta_{\text{gold}}$, exactly
(`tests/test_correct.py::test_ppi_reduces_to_gold_only_at_lambda_zero`). That sets the
floor: PPI cannot do meaningfully worse than ignoring the judge.
`tests/test_correct.py::test_ppi_does_not_blow_up_with_an_adversarial_judge` passes a judge
that says the opposite of the truth and still gets an interval no wider than gold-only.

Validity holds for any *fixed* $\lambda$. Estimating it from the same sample contributes
only a higher-order term; this is the standard PPI++ argument. The coverage simulations
use the estimated $\lambda$, so they test the estimator as it ships.

## Assumptions

1. **The gold subset is a random sample of the evaluation set.** This is the one that
   breaks in practice: labeling the examples that looked interesting, or the ones a
   reviewer had time for, violates it and no amount of statistics repairs it. The
   `Estimate.assumptions` field carries this text into every report.
2. **Judge labels were produced the same way for labeled and unlabeled examples.** Scoring
   the labeled subset with a more careful prompt breaks the comparison.
3. **The interval is asymptotic.** With fewer than roughly 30 gold labels, prefer
   `gold_only_estimate`, whose Wilson interval is exact.
4. **One row is one independent observation.** Evaluations violate this routinely. A run
   with `--epochs 5` scores every sample five times, and those five outcomes share the
   sample's difficulty, prompt and answer. Treated as five draws they shrink the interval
   by about $\sqrt{5}$ more than the data supports: a nominal 95% interval covers 86% of
   the time, measured in
   `tests/test_correct.py::test_clustered_data_undercovers_until_clusters_are_declared`.
   Pass `clusters=` to `ppi_estimate` and `gold_only_estimate`, or `--cluster-column` on
   the command line, and the variance becomes cluster-robust:

   $$\operatorname{Var}(\bar{x}) = \frac{G}{G-1}\cdot\frac{1}{n^{2}}\sum_{g}\Bigl(\sum_{i \in g}(x_i - \bar{x})\Bigr)^{2}$$

   Residuals are summed within a cluster before squaring, so within-cluster correlation is
   unrestricted. With one observation per cluster this reduces exactly to
   $\operatorname{var}(x)/n$, pinned by
   `test_clustered_variance_reduces_to_the_iid_formula_with_singleton_clusters`, which is
   what makes it safe as a general replacement rather than a separate code path.

   For PPI a cluster must lie wholly inside the labeled set or wholly outside it, since
   the estimator's two terms are only independent when they share no cluster. Sampling
   whole clusters for labeling gives that for free; a straddling cluster raises. The
   Inspect adapter averages epochs to one row per sample before any of this, so the common
   case needs nothing from the caller. The docstring says so.

## Agreement with the reference implementation

`ppi_py` is the implementation released with the papers above. Two implementations written
from the same algebra should return the same numbers, and `tests/test_conformance.py`
checks that they do across six regimes, from 30 gold labels against 2000 unlabeled
examples to a labeled set as large as the unlabeled one.

Writing that test found a real defect. λ was being chosen with $\operatorname{Var}(f)$
estimated over the labeled subset alone, which disagreed with the reference by 5.5e-2 on
the point estimate at 50 gold labels, far too large to be arithmetic. The reference was
right: judge labels exist for *every* example, so estimating $\operatorname{Var}(f)$ from
the labeled subset throws away most of the available data and leaves λ noisy in exactly
the regime this library targets. Pooling both sets fixed it.

Two degrees-of-freedom conventions remained, and they are treated differently on purpose.

The covariance inside λ now divides by $n$, matching the reference, and the point estimate
agrees to 1e-12. λ is a tuning parameter rather than an estimand, so unbiasedness of that
covariance buys nothing, while exact agreement removes a whole class of question about
which implementation to believe.

The variance inside the interval keeps $n-1$. That one *is* an estimand, the unbiased
estimate is the conservative one, and coverage is the property this library will not
trade. The interval therefore comes out 0.1% to 1.7% wider than the reference. The test
asserts that direction rather than a tolerance: never narrower, and not gratuitously
wider. A tolerance would have permitted a narrower interval, which is the one outcome that
would matter.

## Rogan–Gladen, and when to prefer it

For binary labels there is a classical alternative. If the judge has sensitivity
$\mathrm{se}$ and specificity $\mathrm{sp}$, its observed positive rate relates to the
truth by

$$p_{\text{obs}} = p\,\mathrm{se} + (1-p)(1 - \mathrm{sp})$$

which inverts to

$$\hat p = \frac{p_{\text{obs}} + \mathrm{sp} - 1}{\mathrm{se} + \mathrm{sp} - 1}$$

verified against hand-computed arithmetic in
`tests/test_correct.py::test_rogan_gladen_matches_hand_computed_case`. The denominator is
Youden's $J$; at $\mathrm{se} + \mathrm{sp} \le 1$ the judge carries no recoverable signal
and `rogan_gladen_estimate` raises rather than dividing by something near zero.

The interval uses the delta method, propagating uncertainty in all three estimated
quantities:

$$\operatorname{Var}(\hat p) \approx \frac{1}{J^{2}}\operatorname{Var}(p_{\text{obs}}) + \frac{\hat p^{2}}{J^{2}}\operatorname{Var}(\mathrm{se}) + \frac{(1-\hat p)^{2}}{J^{2}}\operatorname{Var}(\mathrm{sp})$$

Prefer PPI in general: its validity does not rest on a first-order approximation, and it
extends to continuous scores. Rogan–Gladen is here for two reasons. It is the estimator a
statistician will ask about, and it gives an independent check on the PPI number in terms
of quantities teams already measure.

## References

- Angelopoulos, Bates, Fannjiang, Jordan, Zrnic (2023), "Prediction-powered inference",
  *Science* 382(6671).
- Angelopoulos, Bates, Zrnic (2023), "PPI++: Efficient prediction-powered inference",
  arXiv:2311.01453.
- Rogan & Gladen (1978), "Estimating prevalence from the results of a screening test",
  *American Journal of Epidemiology* 107(1).
