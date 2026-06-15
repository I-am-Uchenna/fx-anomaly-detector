"""Volatility features using several estimators for cross-validation.

Close-to-close, Parkinson (1980) and Garman-Klass (1980) estimators each use a
different slice of the OHLC bar, so disagreement between them is itself a signal
(for example Parkinson rising faster than close-to-close points to elevated
intraday range). A GARCH(1,1) conditional volatility (Bollerslev, 1986) and its
standardised residuals are also produced.

The GARCH fit is in-sample over the supplied series. In the walk-forward
backtest the feature builder is invoked separately on each training fold, so no
information leaks across folds; within a fold the fit is descriptive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

from config import settings
from config.logging_config import get_logger

logger = get_logger()

_LN2 = np.log(2.0)
_GARCH_MIN_OBS = 252
# arch expects returns on a percentage-like scale for stable optimisation.
_GARCH_SCALE = 100.0


def _parkinson_daily_var(high: pd.Series, low: pd.Series) -> pd.Series:
    """Per-bar Parkinson variance estimate from the high-low range.

    Args:
        high: Bar highs.
        low: Bar lows.

    Returns:
        Per-bar variance estimates; NaN where high or low is missing.
    """
    log_hl = np.log(high / low)
    return (log_hl**2) / (4.0 * _LN2)


def _garman_klass_daily_var(
    high: pd.Series, low: pd.Series, close: pd.Series, open_: pd.Series
) -> pd.Series:
    """Per-bar Garman-Klass variance estimate from the full OHLC bar.

    Args:
        high: Bar highs.
        low: Bar lows.
        close: Bar closes.
        open_: Bar opens.

    Returns:
        Per-bar variance estimates; NaN where any input is missing.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    return 0.5 * log_hl**2 - (2.0 * _LN2 - 1.0) * log_co**2


def _garch_conditional_vol(returns: pd.Series) -> pd.Series:
    """Fit GARCH(1,1) and return the conditional volatility series.

    Args:
        returns: Log-return series.

    Returns:
        Conditional volatility aligned to the input index, on the original
        return scale. Returns NaN everywhere if there are too few observations
        or the optimiser fails.
    """
    clean = returns.dropna()
    if clean.size < _GARCH_MIN_OBS:
        logger.debug("GARCH skipped: only {} observations (<{})", clean.size, _GARCH_MIN_OBS)
        return pd.Series(np.nan, index=returns.index)
    try:
        model = arch_model(clean * _GARCH_SCALE, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
        res = model.fit(disp="off")
    except Exception as exc:  # optimiser failure on pathological data
        logger.warning("GARCH fit failed: {}", exc)
        return pd.Series(np.nan, index=returns.index)
    cond_vol = pd.Series(np.asarray(res.conditional_volatility), index=clean.index) / _GARCH_SCALE
    logger.debug(
        "GARCH params omega={:.3e} alpha={:.3f} beta={:.3f}",
        res.params.get("omega", np.nan),
        res.params.get("alpha[1]", np.nan),
        res.params.get("beta[1]", np.nan),
    )
    return cond_vol.reindex(returns.index)


def _garch_standardized_resid(returns: pd.Series) -> pd.Series:
    """Magnitude of GARCH standardised residuals as an anomaly signal.

    Args:
        returns: Log-return series.

    Returns:
        Absolute standardised residuals aligned to the input index.
    """
    clean = returns.dropna()
    if clean.size < _GARCH_MIN_OBS:
        return pd.Series(np.nan, index=returns.index)
    try:
        model = arch_model(clean * _GARCH_SCALE, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
        res = model.fit(disp="off")
    except Exception:
        return pd.Series(np.nan, index=returns.index)
    std_resid = pd.Series(np.asarray(res.std_resid), index=clean.index)
    return std_resid.abs().reindex(returns.index)


def compute(master: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Compute volatility features for every pair in the master frame.

    Args:
        master: Multi-level (pair, feature) frame containing per pair the
            columns open, high, low, close and log_return.
        windows: Rolling windows. Defaults to short/medium/long.

    Returns:
        A multi-level (pair, feature) DataFrame of volatility features.
    """
    if windows is None:
        windows = settings.WINDOWS.as_list()
    long_window = settings.WINDOWS.long
    extended = settings.WINDOWS.extended

    pairs = master.columns.get_level_values(0).unique()
    blocks: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        high = master[(pair, "high")]
        low = master[(pair, "low")]
        close = master[(pair, "close")]
        open_ = master[(pair, "open")]
        ret = master[(pair, "log_return")]

        park_var = _parkinson_daily_var(high, low)
        gk_var = _garman_klass_daily_var(high, low, close, open_)

        feats = pd.DataFrame(index=master.index)
        for w in windows:
            vol_cc = ret.rolling(w).std(ddof=0)
            vol_park = np.sqrt(park_var.rolling(w).mean())
            vol_gk = np.sqrt(gk_var.rolling(w).mean().clip(lower=0))
            feats[f"vol_close_{w}"] = vol_cc
            feats[f"vol_parkinson_{w}"] = vol_park
            feats[f"vol_garman_klass_{w}"] = vol_gk
            feats[f"vol_ratio_park_cc_{w}"] = vol_park / vol_cc
            feats[f"vol_of_vol_{w}"] = vol_cc.rolling(w).std(ddof=0)
            ext_roll = vol_cc.rolling(extended)
            feats[f"vol_zscore_{w}"] = (vol_cc - ext_roll.mean()) / ext_roll.std(ddof=0)

        short_vol = ret.rolling(settings.WINDOWS.short).std(ddof=0)
        long_vol = ret.rolling(long_window).std(ddof=0)
        feats["vol_term_structure"] = short_vol / long_vol

        feats["garch_conditional_vol"] = _garch_conditional_vol(ret)
        feats["garch_std_resid"] = _garch_standardized_resid(ret)
        blocks[pair] = feats

    result = pd.concat(blocks, axis=1)
    result.columns = result.columns.set_names(["pair", "feature"])
    return result
