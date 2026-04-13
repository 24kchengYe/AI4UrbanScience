"""Distribution-divergence metrics used in Experiment 06 (Fig. 4).

The three metrics are:

* Mean Absolute Error (MAE) of bin-level relative frequencies.
* Overlap Ratio (OR) between binned quantile intervals.
* Jensen–Shannon Divergence (JSD) between two discrete distributions.

All functions accept 1-D arrays and produce plain floats so they compose
easily with :mod:`pandas` groupby operations.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Bin construction
# ---------------------------------------------------------------------------

def equal_width_bins(*arrays: np.ndarray, n_bins: int = 30) -> np.ndarray:
    """Return ``n_bins + 1`` equal-width bin edges covering every input."""
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    lo = min(np.nanmin(a) for a in arrays if a.size)
    hi = max(np.nanmax(a) for a in arrays if a.size)
    if lo == hi:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_bins + 1)


def relative_frequencies(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return the per-bin relative frequencies (sums to 1 when non-empty)."""
    values = np.asarray(values, dtype=float)
    counts, _ = np.histogram(values[np.isfinite(values)], bins=edges)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / float(total)


# ---------------------------------------------------------------------------
# Distribution metrics
# ---------------------------------------------------------------------------

def mae_bin(real: np.ndarray, generated: np.ndarray, n_bins: int = 30) -> float:
    """Mean Absolute Error of bin-level relative frequencies.

    Low MAE means the generated distribution matches the real-world
    distribution's shape at this binning resolution.
    """
    edges = equal_width_bins(real, generated, n_bins=n_bins)
    p = relative_frequencies(real, edges)
    q = relative_frequencies(generated, edges)
    return float(np.mean(np.abs(p - q)))


def overlap_ratio(real: np.ndarray, generated: np.ndarray, n_bins: int = 30) -> float:
    """Per-bin overlap ratio (``sum min / sum max``).

    Ranges in ``[0, 1]``: 0 = disjoint distributions, 1 = identical.
    """
    edges = equal_width_bins(real, generated, n_bins=n_bins)
    p = relative_frequencies(real, edges)
    q = relative_frequencies(generated, edges)
    num = np.minimum(p, q).sum()
    den = np.maximum(p, q).sum()
    if den == 0:
        return 0.0
    return float(num / den)


def jsd(real: np.ndarray, generated: np.ndarray, n_bins: int = 30,
        base: float = 2.0) -> float:
    """Jensen-Shannon divergence between the binned distributions.

    Uses ``base=2`` so the result is in bits and ``JSD <= 1``.
    """
    edges = equal_width_bins(real, generated, n_bins=n_bins)
    p = relative_frequencies(real, edges)
    q = relative_frequencies(generated, edges)

    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    m = 0.5 * (p + q)
    kl_pm = _kl(p, m)
    kl_qm = _kl(q, m)
    return float(0.5 * (kl_pm + kl_qm) / np.log(base))
