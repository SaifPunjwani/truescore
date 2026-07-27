# Analysis

Studies run with truescore on public data. Each directory holds one script that downloads
its own data and prints every number in the write-up beside it, so a reader can check any
claim without taking it on trust.

These are not part of the package. They need pandas and pyarrow, which truescore does not
depend on.

| study | finding |
| --- | --- |
| [mt_bench](mt_bench/FINDINGS.md) | GPT-4 agrees with human judges on 88% of MT-Bench comparisons and still reports its own win rate 12.7 points above theirs. A 9.3-point self-preference survives controlling for the judge exaggerating the quality spread. |
