"""Mathematical forms of the urban theories tested by the paper.

Each theory is implemented as a plain function plus, where helpful, a small
container for empirical reference values (e.g. the Bettencourt-2013 scaling
exponents). Experiment scripts import these instead of re-deriving them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Urban scaling laws (Bettencourt, 2007 / 2013)
# ---------------------------------------------------------------------------

def power_law(x: np.ndarray, a: float, beta: float) -> np.ndarray:
    """Classic ``Y = a * X^beta`` scaling relation."""
    return a * np.power(x, beta)


def zipf_rank_size(rank: np.ndarray, a: float, beta: float) -> np.ndarray:
    """Zipf's rank-size law in log-log space: ``population = a * rank^beta``."""
    return a * np.power(rank, beta)


@dataclass(frozen=True)
class ScalingBenchmark:
    """Empirical reference value for an intra-urban scaling exponent.

    Defaults come from Bettencourt (2013) *The Origins of Scaling in Cities*,
    Science 340 (6139): 1438-1441.
    """
    metric: str
    beta: float
    beta_sd: float

    def label(self) -> str:
        return rf"$\beta_{{emp}}={self.beta:.3f}\pm{self.beta_sd:.3f}$"


BETTENCOURT_2013 = {
    "infrastructure": ScalingBenchmark("Infrastructure", 0.849, 0.038),
    "gdp":            ScalingBenchmark("GDP",            1.126, 0.023),
}


# ---------------------------------------------------------------------------
# Distance decay (meso-scale)
# ---------------------------------------------------------------------------

def inverse_s(r: np.ndarray, r0: float, alpha: float, beta: float) -> np.ndarray:
    """Inverse-S density decay model.

    Parameters
    ----------
    r : array-like
        Distance from the city center.
    r0 : float
        Scaling parameter (characteristic radius of the city).
    alpha : float
        Growth rate controlling steepness of the decay.
    beta : float
        Minimum (asymptotic) density at large distance.

    Notes
    -----
    Functional form::

        D(r) = beta + (1 - beta) / (1 + exp(alpha * (r / r0 - 1)))

    so ``D(0) → 1`` near the center and ``D(r → ∞) → beta``.
    """
    return beta + (1.0 - beta) / (1.0 + np.exp(alpha * (r / r0 - 1.0)))


# ---------------------------------------------------------------------------
# Jacobs' urban vitality (micro-scale)
# ---------------------------------------------------------------------------

VITALITY_ATTRIBUTES: tuple[str, ...] = (
    "Population Density",
    "Building Mix Index",
    "Short Block",
    "Aged Building",
    "Tall Building",
)

# Theoretical sign (positive/negative) of the effect of each attribute on
# livability according to Jacobs' hypothesis. Used to check whether
# GenAI-produced regression coefficients have the theoretically-expected sign.
VITALITY_EXPECTED_SIGN: dict[str, int] = {
    "Population Density": +1,
    "Building Mix Index": +1,
    "Short Block":        +1,
    "Aged Building":      +1,   # Jacobs argued aged buildings support diversity
    "Tall Building":      -1,   # debated; Jacobs was skeptical of super-tall
}
