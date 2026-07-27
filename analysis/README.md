# Analysis

Studies run with truescore on public data. Each directory holds one script that downloads
its own data and prints every number in the write-up beside it, so a reader can check any
claim without taking it on trust.

These are not part of the package. `mt_bench` additionally needs pandas and pyarrow to read
parquet; `rewardbench` needs nothing beyond truescore itself.

| study | finding |
| --- | --- |
| [rewardbench](rewardbench/FINDINGS.md) | The top two judges on RewardBench are 0.37 points apart and not distinguishable on 2985 paired examples. They disagree on 10 of 23 subsets, in both directions, by up to 22 points: one wins every code subset, the other wins adversarial preference and maths. |
| [mt_bench](mt_bench/FINDINGS.md) | GPT-4 agrees with human judges on 88% of MT-Bench comparisons and still reports its own win rate 12.7 points above theirs. A 9.3-point self-preference survives controlling for the judge exaggerating the quality spread. |
