"""Command line interface.

Five commands over the files a team already produces:

    truescore doctor   results.csv
    truescore audit    results.csv --judge passed --gold human_passed
    truescore compare  results.csv --judge-a v3_passed --judge-b v4_passed --gold-a ... --gold-b ...
    truescore drift    anchor.csv  --baseline judge_may --current judge_june --gold human
    truescore monitor  stream.csv  --metric passed --baseline 0.88
    truescore plan     --n-total 5000 --target 0.02 --sensitivity 0.92 --specificity 0.85
    truescore slices   results.csv --by segment --judge-a v4 --gold-a v4_human \
                                   --judge-b v3 --gold-b v3_human

Exit codes are chosen so the tool can gate a pipeline:

==== ===========================================================================
0    ran, and found nothing that should stop a release
2    ran, and found something: drift, a regression, or contamination
1    could not run -- bad arguments, unreadable file, unusable data
==== ===========================================================================

That distinction matters more than it looks. A monitor that exits 1 on both "your judge
changed" and "your file has a typo" cannot be wired into CI, because the only safe
response to an ambiguous failure is to ignore it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from truescore import __version__
from truescore.agreement import judge_agreement
from truescore.compare import mcnemar, ppi_compare
from truescore.contamination import combine_shards, exchangeability_test
from truescore.doctor import diagnose
from truescore.drift import judge_drift
from truescore.io import LabelSet, load_labels, read_rows
from truescore.power import required_gold_labels
from truescore.report import build_report
from truescore.sequential import confidence_sequence, first_exclusion, windowed_exclusion
from truescore.slices import compare_slices, estimate_slices

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_FINDING = 2

__all__ = ["main"]


def _write(path: str | None, text: str, label: str) -> None:
    if path:
        Path(path).write_text(text, encoding="utf-8")
        print(f"\nwrote {label} to {path}")


def _load(args: argparse.Namespace) -> LabelSet:
    """Load the label set an argparse namespace describes."""
    judge: str = args.judge
    gold: str | None = getattr(args, "gold", None)
    return load_labels(
        args.file,
        judge=judge,
        gold=gold,
        id_column=getattr(args, "id_column", None),
        covariates=getattr(args, "covariate", None) or (),
    )


def _cmd_audit(args: argparse.Namespace) -> int:
    labels = _load(args)
    print(labels.summary())
    print()

    report = build_report(
        labels.judge,
        labels.gold,
        labels.gold_index,
        metric_name=args.metric_name,
        system_name=args.system_name,
        alpha=args.alpha,
        covariates=labels.covariates_on_gold() or None,
    )
    print(report.summary())

    if report.agreement is not None:
        print()
        print(report.agreement.summary())
    if report.bias is not None:
        print()
        print(report.bias.summary())

    _write(args.json, report.to_json(), "JSON report")
    _write(args.markdown, report.to_markdown(), "markdown report")

    # A judge-only number that falls outside the corrected interval is the finding this
    # command exists to surface: the conventional score is not merely imprecise, it is
    # outside the range the evidence supports.
    naive_outside = not (report.corrected.low <= report.naive.point <= report.corrected.high)
    if naive_outside:
        print(
            f"\nFINDING: the judge-only score {report.naive.point:.4f} lies outside the "
            f"corrected interval [{report.corrected.low:.4f}, {report.corrected.high:.4f}]."
        )
        return EXIT_FINDING
    return EXIT_OK


def _cmd_compare(args: argparse.Namespace) -> int:
    rows = read_rows(args.file)
    a = load_labels(rows, judge=args.judge_a, gold=args.gold_a)
    b = load_labels(rows, judge=args.judge_b, gold=args.gold_b)

    print(f"comparing {args.judge_a} against {args.judge_b} on {a.n_total} shared examples")
    print()

    naive = mcnemar(a.judge.astype(int), b.judge.astype(int), alpha=args.alpha)
    print("as judged (uncorrected):")
    print(naive.summary())

    corrected = None
    if args.gold_a and args.gold_b:
        if not np.array_equal(a.gold_index, b.gold_index):
            raise ValueError(
                "the two gold columns must be labeled on the same rows for a paired "
                "comparison of true scores"
            )
        corrected = ppi_compare(a.judge, b.judge, a.gold, b.gold, a.gold_index, alpha=args.alpha)
        print()
        print("corrected for judge error:")
        print(corrected.summary())

        if naive.significant and not corrected.significant:
            print(
                "\nFINDING: the difference is significant as judged but not after "
                "correcting for judge error. The apparent win may be judge bias."
            )
            return EXIT_FINDING

    result = corrected or naive
    if result.significant:
        print(f"\nFINDING: a real difference of {result.difference:+.4f}.")
        return EXIT_FINDING
    return EXIT_OK


def _cmd_drift(args: argparse.Namespace) -> int:
    rows = read_rows(args.file)
    baseline = load_labels(rows, judge=args.baseline)
    current = load_labels(rows, judge=args.current)
    gold = load_labels(rows, judge=args.gold)

    report = judge_drift(
        baseline.judge.astype(int),
        current.judge.astype(int),
        gold.judge.astype(int),
        alpha=args.alpha,
    )
    print(report.summary())

    if report.agreement_changed or report.behavior_changed:
        print("\nFINDING: the judge is not the instrument it was at baseline.")
        return EXIT_FINDING
    return EXIT_OK


def _cmd_monitor(args: argparse.Namespace) -> int:
    labels = load_labels(args.file, judge=args.metric)
    sequence = confidence_sequence(labels.judge, alpha=args.alpha, bounds=tuple(args.bounds))
    print(sequence.summary())

    if args.window:
        # Windowed monitoring asks whether the *recent* rate departed, which is the
        # operational question; the cumulative test is held up by a healthy prefix.
        stopped = windowed_exclusion(
            labels.judge,
            args.baseline,
            window=args.window,
            alpha=args.alpha,
            direction=args.direction,
            bounds=tuple(args.bounds),
        )
    else:
        stopped = first_exclusion(
            labels.judge,
            args.baseline,
            alpha=args.alpha,
            direction=args.direction,
            bounds=tuple(args.bounds),
        )
    print()
    if stopped is None:
        print(
            f"no evidence against the baseline {args.baseline:.4f} after {sequence.n} observations"
        )
        return EXIT_OK

    print(
        f"FINDING: the baseline {args.baseline:.4f} was ruled out at observation "
        f"{stopped} of {sequence.n}."
    )
    return EXIT_FINDING


def _cmd_plan(args: argparse.Namespace) -> int:
    plan = required_gold_labels(
        args.n_total,
        target_half_width=args.target,
        true_rate=args.rate,
        sensitivity=args.sensitivity,
        specificity=args.specificity,
        alpha=args.alpha,
    )
    print(plan.summary())
    return EXIT_OK if plan.feasible else EXIT_FINDING


def _cmd_contamination(args: argparse.Namespace) -> int:
    rows = read_rows(args.file)
    if args.shard_column:
        shards: dict[str, list[float]] = {}
        canonical: dict[str, float] = {}
        for row in rows:
            shard = str(row[args.shard_column])
            value = float(row[args.loglik_column])
            if str(row.get(args.kind_column, "")).strip().lower() == "canonical":
                canonical[shard] = value
            else:
                shards.setdefault(shard, []).append(value)
        p_values = [
            exchangeability_test(canonical[name], values, alpha=args.alpha).p_value
            for name, values in sorted(shards.items())
            if name in canonical
        ]
        if not p_values:
            raise ValueError("no shard had both a canonical row and permutation rows")
        pooled = combine_shards(np.asarray(p_values), alpha=args.alpha)
        print(pooled.summary())
        return EXIT_FINDING if pooled.contaminated else EXIT_OK

    canonical_values = [
        float(row[args.loglik_column])
        for row in rows
        if str(row.get(args.kind_column, "")).strip().lower() == "canonical"
    ]
    permuted = [
        float(row[args.loglik_column])
        for row in rows
        if str(row.get(args.kind_column, "")).strip().lower() != "canonical"
    ]
    if len(canonical_values) != 1:
        raise ValueError(
            f"expected exactly one row with {args.kind_column}='canonical'; "
            f"found {len(canonical_values)}"
        )
    result = exchangeability_test(canonical_values[0], permuted, alpha=args.alpha)
    print(result.summary())
    return EXIT_FINDING if result.contaminated else EXIT_OK


def _cmd_doctor(args: argparse.Namespace) -> int:
    print(diagnose(args.file, alpha=args.alpha).summary())
    return EXIT_OK


def _cmd_slices(args: argparse.Namespace) -> int:
    rows = read_rows(args.file)
    segments = np.asarray([str(row[args.by]) for row in rows])

    if args.judge_b:
        a = load_labels(rows, judge=args.judge_a, gold=args.gold_a)
        b = load_labels(rows, judge=args.judge_b, gold=args.gold_b)
        if not np.array_equal(a.gold_index, b.gold_index):
            raise ValueError("both gold columns must be labeled on the same rows")
        report = compare_slices(
            a.judge,
            b.judge,
            a.gold,
            b.gold,
            a.gold_index,
            segments,
            by=args.by,
            alpha=args.alpha,
            correction=args.correction,
            min_gold=args.min_gold,
        )
        print(report.summary())
        regressed = [c for c in report.comparisons if c.significant and c.difference < 0]
        if regressed:
            names = ", ".join(c.name for c in regressed)
            print(f"\nFINDING: regressed on {names} after correcting for judge error.")
            return EXIT_FINDING
        return EXIT_OK

    labels = load_labels(rows, judge=args.judge_a, gold=args.gold_a)
    report = estimate_slices(
        labels.judge,
        labels.gold,
        labels.gold_index,
        segments,
        by=args.by,
        alpha=args.alpha,
        min_gold=args.min_gold,
    )
    print(report.summary())
    return EXIT_OK


def _cmd_agreement(args: argparse.Namespace) -> int:
    labels = _load(args)
    report = judge_agreement(
        labels.judge[labels.gold_index].astype(int), labels.gold.astype(int), alpha=args.alpha
    )
    print(report.summary())
    return EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="truescore",
        description="Statistically valid evaluation for LLM-judged benchmarks.",
    )
    parser.add_argument("--version", action="version", version=f"truescore {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--alpha", type=float, default=0.05, help="significance level")

    audit = sub.add_parser("audit", help="correct a judge-scored evaluation and report it")
    audit.add_argument("file", help="CSV or JSONL of evaluation results")
    audit.add_argument("--judge", required=True, help="column holding the judge verdict")
    audit.add_argument("--gold", required=True, help="column holding the human verdict")
    audit.add_argument("--covariate", action="append", help="numeric column for bias analysis")
    audit.add_argument("--id-column", help="column of example identifiers")
    audit.add_argument("--metric-name", default="pass rate")
    audit.add_argument("--system-name", default="system")
    audit.add_argument("--json", help="write the JSON report here")
    audit.add_argument("--markdown", help="write the markdown report here")
    add_common(audit)
    audit.set_defaults(func=_cmd_audit)

    compare = sub.add_parser("compare", help="compare two systems on shared examples")
    compare.add_argument("file")
    compare.add_argument("--judge-a", required=True)
    compare.add_argument("--judge-b", required=True)
    compare.add_argument("--gold-a", help="human verdicts for system A")
    compare.add_argument("--gold-b", help="human verdicts for system B")
    add_common(compare)
    compare.set_defaults(func=_cmd_compare)

    drift = sub.add_parser("drift", help="detect a judge that changed between runs")
    drift.add_argument("file")
    drift.add_argument("--baseline", required=True, help="column of baseline judge verdicts")
    drift.add_argument("--current", required=True, help="column of current judge verdicts")
    drift.add_argument("--gold", required=True, help="column of anchor gold labels")
    add_common(drift)
    drift.set_defaults(func=_cmd_drift)

    monitor = sub.add_parser("monitor", help="anytime-valid monitoring of a metric stream")
    monitor.add_argument("file")
    monitor.add_argument("--metric", required=True, help="column of per-example metric values")
    monitor.add_argument("--baseline", type=float, required=True, help="value being defended")
    monitor.add_argument("--direction", choices=("two-sided", "below", "above"), default="below")
    monitor.add_argument("--bounds", type=float, nargs=2, default=[0.0, 1.0])
    monitor.add_argument(
        "--window",
        type=int,
        help="monitor consecutive windows of this size instead of the cumulative mean; "
        "use this to catch a regression that starts after a healthy period",
    )
    add_common(monitor)
    monitor.set_defaults(func=_cmd_monitor)

    plan = sub.add_parser("plan", help="how many human labels a target precision needs")
    plan.add_argument("--n-total", type=int, required=True)
    plan.add_argument("--target", type=float, required=True, help="desired interval half-width")
    plan.add_argument("--rate", type=float, default=0.5, help="expected true pass rate")
    plan.add_argument("--sensitivity", type=float, default=0.9)
    plan.add_argument("--specificity", type=float, default=0.9)
    add_common(plan)
    plan.set_defaults(func=_cmd_plan)

    contamination = sub.add_parser(
        "contamination", help="exact test for a memorized evaluation set"
    )
    contamination.add_argument("file")
    contamination.add_argument("--loglik-column", default="loglik")
    contamination.add_argument("--kind-column", default="kind")
    contamination.add_argument("--shard-column", help="pool independent shards if present")
    add_common(contamination)
    contamination.set_defaults(func=_cmd_contamination)

    doctor = sub.add_parser("doctor", help="point at an eval file and see what it supports")
    doctor.add_argument("file")
    add_common(doctor)
    doctor.set_defaults(func=_cmd_doctor)

    slices = sub.add_parser("slices", help="per-segment estimates or comparisons")
    slices.add_argument("file")
    slices.add_argument("--by", required=True, help="column to slice on")
    slices.add_argument("--judge-a", required=True)
    slices.add_argument("--gold-a", required=True)
    slices.add_argument("--judge-b", help="second system; omit to estimate rather than compare")
    slices.add_argument("--gold-b")
    slices.add_argument("--correction", choices=("holm", "bh", "none"), default="holm")
    slices.add_argument("--min-gold", type=int, default=20)
    add_common(slices)
    slices.set_defaults(func=_cmd_slices)

    agreement = sub.add_parser("agreement", help="judge quality against human labels")
    agreement.add_argument("file")
    agreement.add_argument("--judge", required=True)
    agreement.add_argument("--gold", required=True)
    add_common(agreement)
    agreement.set_defaults(func=_cmd_agreement)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit code rather than calling ``sys.exit``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        code: int = args.func(args)
        return code
    except (ValueError, FileNotFoundError, KeyError) as error:
        # Data and usage problems exit 1, distinct from a statistical finding, so a CI
        # gate can tell "your judge drifted" from "your file is malformed".
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
