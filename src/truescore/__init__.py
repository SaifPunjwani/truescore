# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Saif Punjwani
"""Statistically valid evaluation for LLM-judged benchmarks.

An LLM judge is a cheap, biased measurement instrument. Averaging its labels estimates
the judge's pass rate, not the system's. truescore takes labels you already have -- judge
labels on many examples, human labels on a few -- and returns estimates you can defend:
bias-corrected points, intervals with verified coverage, and comparisons that survive
scrutiny.

truescore never calls a model and never runs an evaluation. It consumes the outputs of
whatever harness you already use.
"""

from __future__ import annotations

from truescore.agreement import (
    AgreementReport,
    GradedAgreementReport,
    Interval,
    cohen_kappa,
    graded_agreement,
    gwet_ac1,
    judge_agreement,
    krippendorff_alpha,
    quadratic_weighted_kappa,
    wilson_interval,
)
from truescore.bias import (
    BiasEffect,
    BiasReport,
    PositionBiasResult,
    judge_error_regression,
    length_bias,
    position_bias,
)
from truescore.compare import (
    ComparisonResult,
    benjamini_hochberg,
    holm,
    mcnemar,
    paired_bootstrap,
    paired_permutation,
    ppi_compare,
)
from truescore.contamination import (
    CombinedContamination,
    ContaminationResult,
    combine_shards,
    exchangeability_test,
)
from truescore.correct import (
    Estimate,
    gold_only_estimate,
    judge_only_estimate,
    ppi_estimate,
    rogan_gladen_estimate,
)
from truescore.doctor import ColumnProfile, Diagnosis, diagnose
from truescore.drift import (
    DriftReport,
    anchor_fingerprint,
    judge_drift,
    monitor_agreement,
)
from truescore.power import (
    GoldBudget,
    min_detectable_effect,
    required_gold_labels,
    required_pairs,
)
from truescore.report import EvalReport, build_report
from truescore.sequential import (
    ConfidenceSequence,
    confidence_sequence,
    first_exclusion,
    windowed_exclusion,
)
from truescore.slices import (
    SliceComparison,
    SliceEstimate,
    SliceReport,
    compare_slices,
    counts_by_slice,
    estimate_slices,
)
from truescore.weighting import (
    StratumEstimate,
    WeightedEstimate,
    eval_composition,
    post_stratified_estimate,
)

__version__ = "0.7.4"

__all__ = [
    "AgreementReport",
    "BiasEffect",
    "BiasReport",
    "ColumnProfile",
    "CombinedContamination",
    "ComparisonResult",
    "ConfidenceSequence",
    "ContaminationResult",
    "Diagnosis",
    "DriftReport",
    "Estimate",
    "EvalReport",
    "GoldBudget",
    "GradedAgreementReport",
    "Interval",
    "PositionBiasResult",
    "SliceComparison",
    "SliceEstimate",
    "SliceReport",
    "StratumEstimate",
    "WeightedEstimate",
    "__version__",
    "anchor_fingerprint",
    "benjamini_hochberg",
    "build_report",
    "cohen_kappa",
    "combine_shards",
    "compare_slices",
    "confidence_sequence",
    "counts_by_slice",
    "diagnose",
    "estimate_slices",
    "eval_composition",
    "exchangeability_test",
    "first_exclusion",
    "gold_only_estimate",
    "graded_agreement",
    "gwet_ac1",
    "holm",
    "judge_agreement",
    "judge_drift",
    "judge_error_regression",
    "judge_only_estimate",
    "krippendorff_alpha",
    "length_bias",
    "mcnemar",
    "min_detectable_effect",
    "monitor_agreement",
    "paired_bootstrap",
    "paired_permutation",
    "position_bias",
    "post_stratified_estimate",
    "ppi_compare",
    "ppi_estimate",
    "quadratic_weighted_kappa",
    "required_gold_labels",
    "required_pairs",
    "rogan_gladen_estimate",
    "wilson_interval",
    "windowed_exclusion",
]
