# Evaluation report: support-assistant-v4

**Metric:** pass rate  
**Examples:** 4000 judge-labeled, 400 human-labeled  
**Generated:** 2026-07-26T02:21:35+00:00

## Result

| estimate | value | 95% interval | method |
| --- | --- | --- | --- |
| **Corrected (use this)** | **0.6930** | [0.6558, 0.7301] | ppi++ |
| Judge-only (conventional) | 0.8383 | [0.8265, 0.8493] | judge_only (wilson) |
| Human labels only | 0.6775 | [0.6302, 0.7214] | gold_only (wilson) |

The conventional judge-only number is **+0.1453** away from the corrected estimate, and its interval does not contain the corrected estimate.

## Judge quality

```
judge agreement on n=400 gold-labeled examples (gold positive rate 0.677)
  accuracy    0.8350 [0.7955, 0.8682]
  sensitivity 0.9815 [0.9575, 0.9921]
  specificity 0.5271 [0.4414, 0.6113]
  precision   0.8135 [0.7677, 0.8519]
  Cohen κ     0.5740 [0.4858, 0.6540]
  Gwet AC1    0.7349 [0.6664, 0.7953]
  confusion   tp=266 fp=61 fn=5 tn=68
```

## Judge bias

```
judge error regression on n=400 gold-labeled examples
  mean signed error: +0.1400 (judge over-scores)
  tokens_per_100: +0.03686 per unit [+0.00361, +0.07010], p=0.02978
```

## Assumptions

- gold labels are a random sample of the evaluation set
- judge labels are available for every example and were produced the same way for labeled and unlabeled examples
- the interval is asymptotic (normal); with fewer than roughly 30 gold labels prefer gold_only_estimate, whose interval is exact

## What this report does not establish

- That the evaluation set represents production traffic.
- That the gold labels are correct; they are treated as ground truth by definition.
- Anything about examples outside this evaluation set.
