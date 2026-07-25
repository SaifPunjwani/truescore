"""Input validation shared across truescore.

Single source of error-message style. Every public function validates before computing,
and degenerate input raises ``ValueError`` naming the argument rather than returning NaN:
a silent NaN in an evaluation report is worse than a crash.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "check_alpha",
    "check_binary",
    "check_gold_index",
    "check_same_length",
    "to_1d_array",
]


def to_1d_array(name: str, values: npt.ArrayLike) -> np.ndarray:
    """Return ``values`` as a 1-D array, raising ``ValueError`` if it is not 1-D."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def check_binary(name: str, values: npt.ArrayLike) -> np.ndarray:
    """Return ``values`` as a 1-D array of 0/1 integers.

    Accepts booleans and integer arrays holding only 0 and 1. Anything else raises,
    because silently coercing (say) a 1-5 Likert score to binary would corrupt every
    downstream estimate.
    """
    arr = to_1d_array(name, values)
    if arr.dtype == bool:
        return arr.astype(np.int64)
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be boolean or numeric; got dtype {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    unique = np.unique(arr)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError(f"{name} must contain only 0 and 1; got values {unique.tolist()}")
    return arr.astype(np.int64)


def check_same_length(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> None:
    """Raise ``ValueError`` unless ``a`` and ``b`` have the same length."""
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"{name_a} and {name_b} must have the same length; got {a.shape[0]} and {b.shape[0]}"
        )


def check_gold_index(gold_index: npt.ArrayLike, n_total: int) -> np.ndarray:
    """Validate an index array selecting the gold-labeled subset of ``n_total`` examples.

    Raises:
        ValueError: If the index is not 1-D, is empty, contains duplicates, or addresses
            positions outside ``[0, n_total)``.
    """
    idx = to_1d_array("gold_index", gold_index)
    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError(f"gold_index must hold integers; got dtype {idx.dtype}")
    if idx.min() < 0 or idx.max() >= n_total:
        raise ValueError(
            f"gold_index values must lie in [0, {n_total}); got range [{idx.min()}, {idx.max()}]"
        )
    if np.unique(idx).size != idx.size:
        raise ValueError("gold_index must not contain duplicate positions")
    return idx.astype(np.int64)


def check_alpha(alpha: float) -> float:
    """Validate a significance level, returning it unchanged."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}")
    return alpha
