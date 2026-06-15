"""Microstructure proxies derived from daily OHLC (no tick data available).

The Corwin-Schultz (2012) estimator recovers an effective bid-ask spread from
daily high-low ranges, exploiting that highs are typically buyer-initiated and
lows seller-initiated. Amihud illiquidity and a Kyle-lambda price-impact proxy
use the daily range as a stand-in for volume, since Yahoo FX volume is
unreliable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings

# Constant 3 - 2*sqrt(2) appears throughout the Corwin-Schultz derivation.
_CS_CONST = 3.0 - 2.0 * np.sqrt(2.0)


def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """Corwin-Schultz high-low effective spread estimator.

    Args:
        high: Bar highs.
        low: Bar lows.

    Returns:
        Estimated proportional spread per bar, floored at zero (negative
        estimates are treated as noise).
    """
    log_hl_sq = np.log(high / low) ** 2
    # Beta uses the two single-day squared log ranges (t-1 and t).
    beta = log_hl_sq + log_hl_sq.shift(1)

    two_day_high = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    two_day_low = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = np.log(two_day_high / two_day_low) ** 2

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_CONST - np.sqrt(gamma / _CS_CONST)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.clip(lower=0.0)


def _rolling_kyle_lambda(abs_ret: pd.Series, volume_proxy: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope of |return| on a volume proxy (Kyle's lambda).

    Args:
        abs_ret: Absolute log returns (dependent variable).
        volume_proxy: Activity proxy (independent variable).
        window: Rolling window length.

    Returns:
        Rolling slope estimates; NaN where the proxy variance is zero.
    """
    cov = abs_ret.rolling(window).cov(volume_proxy)
    var = volume_proxy.rolling(window).var(ddof=0)
    return cov / var.replace(0.0, np.nan)


def compute(master: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Compute microstructure proxy features for every pair.

    Args:
        master: Multi-level (pair, feature) frame containing per pair high,
            low, close and log_return.
        windows: Rolling windows. Defaults to short/medium/long.

    Returns:
        A multi-level (pair, feature) DataFrame of microstructure features.
    """
    if windows is None:
        windows = settings.WINDOWS.as_list()

    pairs = master.columns.get_level_values(0).unique()
    blocks: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        high = master[(pair, "high")]
        low = master[(pair, "low")]
        close = master[(pair, "close")]
        ret = master[(pair, "log_return")]

        abs_ret = ret.abs()
        # Normalised daily range used as a volume / activity proxy.
        volume_proxy = (high - low) / close
        spread = corwin_schultz_spread(high, low)

        feats = pd.DataFrame(index=master.index)
        feats["spread_proxy_hl"] = spread
        for w in windows:
            roll_spread = spread.rolling(w)
            feats[f"spread_zscore_{w}"] = (spread - roll_spread.mean()) / roll_spread.std(ddof=0)
            feats[f"amihud_illiquidity_{w}"] = (
                (abs_ret / volume_proxy.replace(0.0, np.nan)).rolling(w).mean()
            )
            feats[f"kyle_lambda_{w}"] = _rolling_kyle_lambda(abs_ret, volume_proxy, w)
            roll_range = volume_proxy.rolling(w)
            feats[f"range_ratio_{w}"] = (volume_proxy - roll_range.mean()) / roll_range.std(ddof=0)
        blocks[pair] = feats

    result = pd.concat(blocks, axis=1)
    result.columns = result.columns.set_names(["pair", "feature"])
    return result
