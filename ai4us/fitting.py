"""Curve fitting helpers used across the paper's analysis scripts.

Every experiment's analysis step ultimately boils down to fitting one of
three model families:

* log-log linear regression (for power-law / Zipf),
* non-linear inverse-S fit (for distance decay),
* multiple OLS regression (for Jacobs' vitality).

These used to be re-implemented in every analysis script with slightly
different defaults. Centralizing them here makes Figures 1-3 numerically
reproducible from a single code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress

from ai4us import theories


# ---------------------------------------------------------------------------
# Log-log / power-law fit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LogLogFit:
    """Result of a log-log linear regression ``log Y = log a + beta * log X``."""

    a: float
    beta: float
    r_squared: float
    p_value: float
    n: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.a * np.power(x, self.beta)


def fit_power_law(x: np.ndarray, y: np.ndarray) -> LogLogFit:
    """Fit ``y = a * x^beta`` via OLS on log-transformed values.

    Non-positive entries are removed before fitting. The function is
    deterministic and numerically robust, matching the convention used in
    every scaling-law script in the paper.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x_f, y_f = x[mask], y[mask]
    if len(x_f) < 3:
        raise ValueError(f"Not enough positive points to fit (got {len(x_f)}).")

    slope, intercept, r, p, _se = linregress(np.log(x_f), np.log(y_f))
    return LogLogFit(
        a=float(np.exp(intercept)),
        beta=float(slope),
        r_squared=float(r * r),
        p_value=float(p),
        n=len(x_f),
    )


def fit_zipf(populations: np.ndarray) -> LogLogFit:
    """Fit ``population = a * rank^beta``. Assumes sorted population input."""
    sorted_pop = np.sort(np.asarray(populations, dtype=float))[::-1]
    ranks = np.arange(1, len(sorted_pop) + 1, dtype=float)
    return fit_power_law(ranks, sorted_pop)


# ---------------------------------------------------------------------------
# Inverse-S (distance decay)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InverseSFit:
    """Result of fitting the inverse-S density profile."""

    r0: float
    alpha: float
    beta: float
    r_squared: float
    n: int

    def predict(self, r: np.ndarray) -> np.ndarray:
        return theories.inverse_s(r, self.r0, self.alpha, self.beta)


def fit_inverse_s(r: np.ndarray, density: np.ndarray,
                  *, initial: tuple[float, float, float] = (1.0, 5.0, 0.0)
                  ) -> InverseSFit:
    """Non-linear fit of the inverse-S density model."""
    r = np.asarray(r, dtype=float)
    density = np.asarray(density, dtype=float)
    mask = np.isfinite(r) & np.isfinite(density)
    r_f, d_f = r[mask], density[mask]
    if len(r_f) < 4:
        raise ValueError(f"Not enough points for inverse-S fit (got {len(r_f)}).")

    popt, _pcov = curve_fit(
        theories.inverse_s, r_f, d_f, p0=initial, maxfev=10_000,
    )
    pred = theories.inverse_s(r_f, *popt)
    ss_res = float(np.sum((d_f - pred) ** 2))
    ss_tot = float(np.sum((d_f - d_f.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return InverseSFit(
        r0=float(popt[0]),
        alpha=float(popt[1]),
        beta=float(popt[2]),
        r_squared=float(r_squared),
        n=len(r_f),
    )


# ---------------------------------------------------------------------------
# Multiple OLS (Jacobs' vitality)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OLSFit:
    """Result of a multiple OLS regression with per-coefficient statistics."""

    intercept: float
    coefficients: dict[str, float]
    p_values: dict[str, float]
    r_squared: float
    n: int


def fit_ols(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> OLSFit:
    """Fit ``y = a + sum(b_i * x_i)`` with statsmodels OLS.

    We use statsmodels here (rather than ``linregress`` or ``numpy.polyfit``)
    because the paper reports per-coefficient p-values and standardized
    betas which statsmodels produces natively.
    """
    import statsmodels.api as sm

    df = df[[y_col, *x_cols]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < len(x_cols) + 2:
        raise ValueError(
            f"Not enough rows for OLS ({len(df)} rows, {len(x_cols)} regressors)."
        )

    X = sm.add_constant(df[x_cols])
    y = df[y_col]
    result = sm.OLS(y, X).fit()

    return OLSFit(
        intercept=float(result.params["const"]),
        coefficients={k: float(result.params[k]) for k in x_cols},
        p_values={k: float(result.pvalues[k]) for k in x_cols},
        r_squared=float(result.rsquared),
        n=int(result.nobs),
    )
