"""Return-distribution features that capture departures from normality.

For each pair and each rolling window we compute the first four moments, a
current-bar z-score, a Jarque-Bera normality p-value, and the Hurst exponent
via the rescaled-range (R/S) method. The Hurst exponent (Mandelbrot and Wallis,
1969) distinguishes mean-reverting (H < 0.5), random-walk (H = 0.5) and trending
(H > 0.5) behaviour and is implemented from scratch as required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from config import settings

# Minimum window length for which an R/S Hurst estimate is meaningful. Below
# this we cannot fit at least two distinct chunk sizes, so we return NaN.
_MIN_HURST_WINDOW = 16


def hurst_exponent(series: np.ndarray, min_chunk: int = 8) -> float:
    """Estimate the Hurst exponent of a 1-D series via rescaled range (R/S).

    The series is split into non-overlapping chunks of several sizes. For each
    chunk the rescaled range R/S is computed (R is the range of cumulative
    mean-deviations, S the chunk standard deviation). The slope of
    log(mean R/S) regressed on log(chunk size) estimates H.

    Args:
        series: 1-D array of observations (typically log returns).
        min_chunk: Smallest chunk size to use.

    Returns:
        The estimated Hurst exponent, or NaN if the series is too short or
        degenerate (constant) to support an estimate.
    """
    clean = np.asarray(series, dtype="float64")
    clean = clean[~np.isnan(clean)]
    n = clean.size
    if n < _MIN_HURST_WINDOW:
        return float("nan")

    # Build a geometric ladder of chunk sizes from min_chunk to n // 2.
    max_chunk = n // 2
    if max_chunk < min_chunk:
        return float("nan")
    chunk_sizes: list[int] = []
    size = min_chunk
    while size <= max_chunk:
        chunk_sizes.append(size)
        size *= 2
    if len(chunk_sizes) < 2:
        return float("nan")

    log_sizes: list[float] = []
    log_rs: list[float] = []
    for chunk in chunk_sizes:
        n_chunks = n // chunk
        rs_values: list[float] = []
        for i in range(n_chunks):
            segment = clean[i * chunk : (i + 1) * chunk]
            deviations = segment - segment.mean()
            cumulative = np.cumsum(deviations)
            r = cumulative.max() - cumulative.min()
            s = segment.std(ddof=0)
            if s > 0:
                rs_values.append(r / s)
        if rs_values:
            log_sizes.append(np.log(chunk))
            log_rs.append(np.log(np.mean(rs_values)))

    if len(log_sizes) < 2:
        return float("nan")
    slope, _intercept = np.polyfit(log_sizes, log_rs, 1)
    return float(slope)


def _rolling_hurst(returns: pd.Series, window: int) -> pd.Series:
    """Compute a rolling Hurst exponent over a return series.

    Args:
        returns: Log-return series.
        window: Rolling window length.

    Returns:
        A series of Hurst estimates aligned to the right edge of each window.
        Windows shorter than the minimum usable length yield NaN.
    """
    if window < _MIN_HURST_WINDOW:
        return pd.Series(np.nan, index=returns.index)
    return returns.rolling(window).apply(lambda arr: hurst_exponent(arr), raw=True)


def _rolling_jarque_bera_pval(returns: pd.Series, window: int) -> pd.Series:
    """Compute a rolling Jarque-Bera normality p-value.

    Args:
        returns: Log-return series.
        window: Rolling window length.

    Returns:
        Series of JB test p-values; small values indicate non-normality.
    """

    def _jb(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if valid.size < 8 or valid.std(ddof=0) == 0:
            return float("nan")
        return float(stats.jarque_bera(valid).pvalue)

    return returns.rolling(window).apply(_jb, raw=True)


def compute(master: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Compute return-distribution features for every pair in the master frame.

    Args:
        master: Multi-level (pair, feature) frame from the preprocessor; must
            contain a "log_return" feature per pair.
        windows: Rolling windows to use. Defaults to short/medium/long.

    Returns:
        A multi-level (pair, feature) DataFrame of return features.

    Raises:
        KeyError: If a pair has no log_return column.
    """
    if windows is None:
        windows = settings.WINDOWS.as_list()

    pairs = master.columns.get_level_values(0).unique()
    blocks: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        if (pair, "log_return") not in master.columns:
            raise KeyError(f"Pair {pair} has no log_return column.")
        ret = master[(pair, "log_return")]
        feats = pd.DataFrame(index=master.index)
        for w in windows:
            roll = ret.rolling(w)
            feats[f"ret_mean_{w}"] = roll.mean()
            feats[f"ret_std_{w}"] = roll.std(ddof=0)
            feats[f"ret_skew_{w}"] = roll.skew()
            feats[f"ret_kurt_{w}"] = roll.kurt()
            feats[f"ret_zscore_{w}"] = (ret - roll.mean()) / roll.std(ddof=0)
            feats[f"ret_jarque_bera_pval_{w}"] = _rolling_jarque_bera_pval(ret, w)
            feats[f"ret_hurst_{w}"] = _rolling_hurst(ret, w)
        blocks[pair] = feats

    result = pd.concat(blocks, axis=1)
    result.columns = result.columns.set_names(["pair", "feature"])
    return result
