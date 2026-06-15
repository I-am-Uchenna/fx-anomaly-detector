"""Value-at-Risk and Conditional VaR estimation.

Three estimators are provided: parametric (normal), historical (empirical
quantile) and Monte Carlo (simulated from a fitted normal or Student-t). CVaR
(expected shortfall) is the mean loss beyond the VaR threshold and is the more
coherent tail measure. All figures are returned as positive loss fractions for
a one-day horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class VarEstimate:
    """A VaR/CVaR pair at one confidence level.

    Args:
        confidence: Confidence level, e.g. 0.95.
        var: Value-at-Risk as a positive loss fraction.
        cvar: Conditional VaR (expected shortfall) as a positive loss fraction.
    """

    confidence: float
    var: float
    cvar: float


def _clean(returns: pd.Series) -> np.ndarray:
    """Return a NaN-free numpy view of a return series.

    Args:
        returns: Return series.

    Returns:
        1-D array of finite returns.
    """
    arr = np.asarray(returns, dtype="float64")
    return arr[np.isfinite(arr)]


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> VarEstimate:
    """Normal-distribution parametric VaR and CVaR.

    Args:
        returns: Return series.
        confidence: Confidence level in (0, 1).

    Returns:
        A VarEstimate.

    Raises:
        ValueError: If confidence is not in (0, 1) or there are no returns.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")
    arr = _clean(returns)
    if arr.size == 0:
        raise ValueError("No finite returns provided.")
    mu, sigma = arr.mean(), arr.std(ddof=1)
    z = stats.norm.ppf(confidence)
    var = z * sigma - mu
    # Closed-form normal expected shortfall.
    es = sigma * stats.norm.pdf(z) / (1.0 - confidence) - mu
    return VarEstimate(confidence, max(float(var), 0.0), max(float(es), 0.0))


def historical_var(returns: pd.Series, confidence: float = 0.95) -> VarEstimate:
    """Empirical (historical-simulation) VaR and CVaR.

    Args:
        returns: Return series.
        confidence: Confidence level in (0, 1).

    Returns:
        A VarEstimate.

    Raises:
        ValueError: If there are no returns.
    """
    arr = _clean(returns)
    if arr.size == 0:
        raise ValueError("No finite returns provided.")
    quantile = np.quantile(arr, 1.0 - confidence)
    var = -quantile
    tail = arr[arr <= quantile]
    cvar = -tail.mean() if tail.size > 0 else var
    return VarEstimate(confidence, max(float(var), 0.0), max(float(cvar), 0.0))


def monte_carlo_var(
    returns: pd.Series,
    confidence: float = 0.95,
    n_sims: int = 10_000,
    distribution: str = "t",
    seed: int = 0,
) -> VarEstimate:
    """Monte Carlo VaR and CVaR from a fitted normal or Student-t.

    Args:
        returns: Return series.
        confidence: Confidence level in (0, 1).
        n_sims: Number of simulated returns.
        distribution: "normal" or "t".
        seed: RNG seed for reproducibility.

    Returns:
        A VarEstimate.

    Raises:
        ValueError: If distribution is unknown or there are no returns.
    """
    arr = _clean(returns)
    if arr.size == 0:
        raise ValueError("No finite returns provided.")
    rng = np.random.default_rng(seed)
    if distribution == "normal":
        sims = rng.normal(arr.mean(), arr.std(ddof=1), n_sims)
    elif distribution == "t":
        df, loc, scale = stats.t.fit(arr)
        sims = stats.t.rvs(df, loc=loc, scale=scale, size=n_sims, random_state=rng)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    quantile = np.quantile(sims, 1.0 - confidence)
    var = -quantile
    tail = sims[sims <= quantile]
    cvar = -tail.mean() if tail.size > 0 else var
    return VarEstimate(confidence, max(float(var), 0.0), max(float(cvar), 0.0))


def compute_var_cvar(returns: pd.Series) -> dict[str, VarEstimate]:
    """Compute VaR/CVaR by all three methods at 95% and 99%.

    Args:
        returns: Return series.

    Returns:
        Mapping of "<method>_<confidence>" to its VarEstimate, for method in
        {parametric, historical, monte_carlo} and confidence in {95, 99}.
    """
    out: dict[str, VarEstimate] = {}
    for confidence in (0.95, 0.99):
        tag = int(confidence * 100)
        out[f"parametric_{tag}"] = parametric_var(returns, confidence)
        out[f"historical_{tag}"] = historical_var(returns, confidence)
        out[f"monte_carlo_{tag}"] = monte_carlo_var(returns, confidence)
    return out


def rolling_var(
    returns: pd.Series,
    window: int,
    confidence: float = 0.95,
    method: str = "historical",
) -> pd.Series:
    """Rolling VaR series for use as a time-varying risk estimate.

    Args:
        returns: Return series.
        window: Rolling window length.
        confidence: Confidence level in (0, 1).
        method: "historical" or "parametric".

    Returns:
        Series of VaR loss fractions aligned to the input.

    Raises:
        ValueError: If method is unknown.
    """
    if method == "historical":
        return -returns.rolling(window).quantile(1.0 - confidence)
    if method == "parametric":
        z = stats.norm.ppf(confidence)
        roll = returns.rolling(window)
        return z * roll.std(ddof=1) - roll.mean()
    raise ValueError(f"Unknown method: {method}")
