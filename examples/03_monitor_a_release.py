"""Watch a release without inventing regressions.

A release goes out and the team watches the pass rate accumulate request by request. The
moment it dips they want to know whether to roll back. Checking a fixed-sample confidence
interval after every request is the obvious approach, and it is wrong: a 95% interval
inspected repeatedly excludes the truth on roughly half of all healthy streams, so the
practice generates rollbacks that were never warranted.

A confidence sequence is valid at every sample size at once, so it can be checked
continuously and acted on immediately.

``data/release_stream.csv`` holds 1200 requests from a release that genuinely regressed at
the halfway point: the first 600 pass at 88%, the rest at 79%.

Command-line equivalent:

    truescore monitor examples/data/release_stream.csv \\
        --metric passed --baseline 0.88 --direction below

Run: python examples/03_monitor_a_release.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from truescore.agreement import wilson_interval
from truescore.io import load_labels
from truescore.sequential import confidence_sequence, first_exclusion, windowed_exclusion

DATA = Path(__file__).parent / "data" / "release_stream.csv"
BASELINE = 0.88
REGRESSION_AT = 600


def main() -> None:
    labels = load_labels(DATA, judge="passed", id_column="request_id")
    stream = labels.judge
    print(f"{stream.shape[0]} requests; the release regressed at request {REGRESSION_AT}")
    print(f"  observed pass rate before: {stream[:REGRESSION_AT].mean():.4f}")
    print(f"  observed pass rate after:  {stream[REGRESSION_AT:].mean():.4f}")
    print()

    sequence = confidence_sequence(stream)
    print(sequence.summary())
    print()

    cumulative = first_exclusion(stream, BASELINE, direction="below")
    print("asking 'is the mean of everything below baseline?':")
    print(f"  {'no verdict' if cumulative is None else f'alarm at {cumulative}'}")
    print(
        "  The healthy first half holds the cumulative mean up, so this question stays\n"
        "  unanswered long after the service degraded. It is the wrong question."
    )
    print()

    print("asking 'has the recent rate departed?' (windows of 300):")
    alarm = windowed_exclusion(stream, BASELINE, window=300, direction="below")
    if alarm is None:
        print(f"  the monitor never ruled out the {BASELINE:.2f} baseline")
    else:
        print(
            f"  ALARM at request {alarm}: the pass rate is provably below the "
            f"{BASELINE:.2f} baseline,"
        )
        print(f"  {alarm - REGRESSION_AT} requests after the regression began.")
        print(
            "  The alarm is trustworthy: the 5% error budget covers the entire run,\n"
            "  however many times it was checked and whenever it was stopped."
        )
    print()

    print("=" * 78)
    print("what the naive approach would have done on the healthy half of this stream")
    print("=" * 78)
    healthy = stream[:REGRESSION_AT]
    successes = np.cumsum(healthy)
    naive_alarms = [
        t
        for t in range(30, REGRESSION_AT + 1)
        if wilson_interval(int(successes[t - 1]), t).high < BASELINE
    ]
    if naive_alarms:
        print(
            f"  a fixed-sample 95% interval, checked after every request, first fires at\n"
            f"  request {naive_alarms[0]} -- during the healthy period, before anything "
            "had gone wrong."
        )
    else:
        print("  on this particular stream the naive approach happened not to false-alarm.")
    print(
        "\n  Whether it fires on any one stream is luck. Across streams it fires on\n"
        "  roughly half of them, which is why continuous monitoring needs a sequence\n"
        "  rather than an interval."
    )


if __name__ == "__main__":
    main()
