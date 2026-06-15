"""Macro-financial features approximated from freely available market data.

VIX is the global risk-aversion gauge, the dollar index (DXY) captures broad
USD strength, and gold-versus-dollar correlation flags safe-haven regime
breaks. Interest-rate and covered-interest-parity inputs that require a paid
forward-rate feed are approximated or flagged as missing rather than
fabricated. Macro signals are broadcast to every pair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from config.logging_config import get_logger

logger = get_logger()


def _aligned_close(
    raw_frames: dict[str, pd.DataFrame], ticker: str, index: pd.Index
) -> pd.Series | None:
    """Return a ticker's close reindexed to the master index, if present.

    Args:
        raw_frames: Mapping of symbol to OHLCV frame.
        ticker: Auxiliary ticker to extract.
        index: Target datetime index.

    Returns:
        The reindexed close series, or None if the ticker is unavailable.
    """
    if ticker not in raw_frames:
        logger.warning("Macro ticker {} unavailable; related features will be NaN", ticker)
        return None
    return raw_frames[ticker]["close"].reindex(index)


def compute(
    master: pd.DataFrame,
    raw_frames: dict[str, pd.DataFrame] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute macro features and broadcast them to every pair.

    Args:
        master: Multi-level (pair, feature) frame with close and log_return.
        raw_frames: Fetcher output, used to read VIX, DXY and gold closes. If
            None, macro market features are filled with NaN.
        windows: Rolling windows. Defaults to short/medium/long.

    Returns:
        A multi-level (pair, feature) DataFrame of macro features.
    """
    if windows is None:
        windows = settings.WINDOWS.as_list()
    raw_frames = raw_frames or {}
    index = master.index

    vix = _aligned_close(raw_frames, settings.VIX_TICKER, index)
    dxy = _aligned_close(raw_frames, settings.DXY_TICKER, index)
    gold = _aligned_close(raw_frames, settings.GOLD_TICKER, index)

    dxy_return = np.log(dxy / dxy.shift(1)) if dxy is not None else None
    gold_return = np.log(gold / gold.shift(1)) if gold is not None else None

    gold_fx_divergence = None
    if dxy_return is not None and gold_return is not None:
        gold_fx_divergence = gold_return.rolling(settings.WINDOWS.medium).corr(dxy_return)

    pairs = list(master.columns.get_level_values(0).unique())
    fx_pairs = [p for p in pairs if p in {pc.symbol for pc in settings.FX_PAIRS}]

    blocks: dict[str, pd.DataFrame] = {}
    for pair in fx_pairs:
        ret = master[(pair, "log_return")]
        feats = pd.DataFrame(index=index)

        # Crude carry proxy: annualised trailing drift of the pair. Documented
        # as a fallback because per-currency 2y yields are not freely reliable.
        feats["rate_differential_proxy"] = (
            ret.rolling(settings.WINDOWS.long).mean() * settings.BACKTEST.trading_days_per_year
        )
        # CIP deviation needs forward points we do not have; flagged missing.
        feats["cip_deviation"] = np.nan

        if vix is not None:
            feats["vix_level"] = vix
            for w in windows:
                roll = vix.rolling(w)
                feats[f"vix_zscore_{w}"] = (vix - roll.mean()) / roll.std(ddof=0)
        if dxy_return is not None:
            for w in windows:
                feats[f"dxy_return_{w}"] = dxy_return.rolling(w).sum()
        if gold_fx_divergence is not None:
            feats["gold_fx_divergence"] = gold_fx_divergence

        blocks[pair] = feats

    result = pd.concat(blocks, axis=1)
    result.columns = result.columns.set_names(["pair", "feature"])
    return result
