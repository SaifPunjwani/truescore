# 88% agreement, 13 points of error

GPT-4 agrees with human judges on 88% of MT-Bench comparisons. It still reports its own
win rate 12.7 points above what the humans give it.

Both numbers come from the same 1814 comparisons. Reproduce them with:

```sh
pip install truescore pandas pyarrow
python run.py
```

The data is [lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
(CC-BY-4.0), which published GPT-4's pairwise judgments and human judgments over the same
comparisons. That makes it one of the few public datasets where a judge's error can be
measured rather than assumed. No model is called; the script downloads two parquet files
and does arithmetic.

## Agreement is high

On the 1078 comparisons where neither GPT-4 nor the humans called a tie:

| | |
| --- | --- |
| accuracy | 0.8840 `[0.8636, 0.9018]` |
| sensitivity | 0.8891 `[0.8608, 0.9122]` |
| specificity | 0.8782 `[0.8467, 0.9040]` |
| Cohen's κ | 0.7670 `[0.7287, 0.8046]` |

This replicates the agreement rate MT-Bench reports for itself, and it is the number teams
quote when they justify shipping an LLM judge.

## The reported win rates are still wrong

Win rate for each model against all opponents, ties counting a half. The judge column is
what you would publish from GPT-4's verdicts alone; the human column is what the human
verdicts give.

| model | n | GPT-4 judge | human | gap | 95% CI on the gap |
| --- | --- | --- | --- | --- | --- |
| alpaca-13b | 582 | 0.2388 | 0.2938 | -0.0550 | `[-0.0817, -0.0283]` |
| claude-v1 | 594 | 0.7146 | 0.6776 | +0.0370 | `[+0.0063, +0.0678]` |
| gpt-3.5-turbo | 678 | 0.6158 | 0.6431 | -0.0273 | `[-0.0560, +0.0015]` |
| **gpt-4** | 578 | **0.8521** | **0.7249** | **+0.1272** | `[+0.0953, +0.1591]` |
| llama-13b | 611 | 0.1154 | 0.1776 | -0.0622 | `[-0.0847, -0.0397]` |
| vicuna-13b-v1.2 | 585 | 0.4615 | 0.4735 | -0.0120 | `[-0.0428, +0.0188]` |

Five of the six gaps sit within 6 points of zero. GPT-4's own is +12.7, and the bottom of
its interval is above the top of every other model's.

Agreeing 88% of the time does not bound the error in an aggregate. Agreement counts how
often two labellers match; a win rate is a mean. Errors that fall mostly one way move the
mean while barely moving the agreement rate, and a judge's errors are rarely symmetric.

## Two explanations, separated

The gap tracks model strength: the judge is harsh on the weak models and generous to the
strong ones, so "the judge exaggerates the spread" explains part of it without any
self-preference. Fitting both at once on all 1814 comparisons, with HC3-robust standard
errors:

| term | coefficient | 95% CI | p |
| --- | --- | --- | --- |
| stronger_model_margin | +0.12401 | `[+0.07366, +0.17436]` | 1.4e-06 |
| gpt4_is_this_side | +0.09342 | `[+0.05641, +0.13042]` | 7.5e-07 |

The spread effect is real, and a 9.3-point self-preference survives it. Each model's
strength is computed leaving out the comparison being explained, so the covariate cannot
contain its own outcome.

## Correcting it

Treat GPT-4's verdicts as a cheap label on all 578 comparisons involving GPT-4, and human
verdicts as gold on a random subset. Over 600 random subsets at each budget, the fraction
whose 95% interval contained the full-human rate of 0.7249:

| human labels | judge-only | gold-only | truescore |
| --- | --- | --- | --- |
| 60 | 0.0% | 96.2% | 96.2% |
| 100 | 0.0% | 96.3% | 96.5% |
| 150 | 0.0% | 98.0% | 97.8% |

The judge-only interval missed on every one of 1800 draws. It is both narrow and wrong,
which is worse than being wide and wrong: a tight interval around 0.8521 reads as a settled
result. The corrected interval covers at its nominal rate from 60 labels.

**What the correction does not buy here is width.** At 100 labels truescore's interval is
5% narrower than using those 100 human labels alone, which is close to nothing. Prediction-
powered inference converts unlabeled examples into precision, and there are only about five
unlabeled comparisons per labeled one in this set. The gain scales with that ratio, so an
eval with 4000 examples and 200 labels sees a real narrowing and this one does not. What it
buys here is the ability to use the judge at all without inheriting its bias.

## What this does not show

- **The human labels are not ground truth.** They are a majority over 1 to 7 judges, and
  47% of the comparisons carry a single judge. Human disagreement is folded into the
  "gap" as noise, so the gaps are attenuated toward zero rather than inflated.
- **These are 2023 models**, including the GPT-4 snapshot doing the judging. Nothing here
  says a current judge behaves the same way, only that this one did and that the effect was
  large enough to change a leaderboard.
- **Self-preference is not fully identified.** `gpt4_is_this_side` marks the model whose
  answers GPT-4 wrote, but that model is also the newest and strongest in the set.
  `stronger_model_margin` absorbs strength as measured by humans; anything about GPT-4
  answers that humans undervalue and GPT-4 correctly values would land in the same term.
- **Ties are collapsed to 0.5**, and "tie (inconsistent)", MT-Bench's marker for a judge
  that disagreed with itself across presentation orders, is treated as a tie.

## Prior work

Self-preference in LLM judges is established, not new here. The MT-Bench paper itself
(Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", arXiv:2306.05685)
names self-enhancement bias and reports GPT-4 favouring its own answers. Panickssery,
Bowman and Feng ("LLM Evaluators Recognize and Favor Their Own Generations",
arXiv:2404.13076) show evaluators recognise their own text and that the preference tracks
that recognition.

What is added here is a measurement rather than a mechanism: the size of the resulting
error in a published win rate, an interval on it, a separation from the competing
spread-exaggeration explanation, and the demonstration that a judge agreeing 88% of the
time still moves a leaderboard number by 12.7 points.

## The general point

An agreement rate is a property of a labeller. A leaderboard is a property of an aggregate.
Measuring the first tells you less about the second than it appears to, and the only way to
know the size of the gap is to hold out human labels and look. That is what
[truescore](https://github.com/SaifPunjwani/truescore) is for.
