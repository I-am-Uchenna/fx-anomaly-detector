"""Transform raw OHLCV into analysis-ready returns and normalised series.

Produces a master DataFrame with a two-level column index (pair, feature) so
that downstream feature modules can address columns by pair. The preprocessor
is deliberately limited to return computation and basic normalisation; richer
features live in src/features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
from config.logging_config import get_logger

logger = get_logger()

_PROCESSED_DIR = Path("data/processed")


def _winsorize(series: pd.Series, lower_pct: float, upper_pct: float) -> pd.Series:
    """Clip a series to its lower and upper empirical percentiles.

    Args:
        series: Input values, may contain NaN.
        lower_pct: Lower quantile in [0, 1].
        upper_pct: Upper quantile in [0, 1].

    Returns:
        The clipped series. NaNs are preserved.

    Raises:
        ValueError: If percentile bounds are not ordered within [0, 1].
    """
    if not 0.0 <= lower_pct < upper_pct <= 1.0:
        raise ValueError("Require 0 <= lower_pct < upper_pct <= 1.")
    valid = series.dropna()
    if valid.empty:
        return series
    low = valid.quantile(lower_pct)
    high = valid.quantile(upper_pct)
    return series.clip(lower=low, upper=high)


def compute_pair_features(
    ohlcv: pd.DataFrame,
    zscore_window: int = settings.WINDOWS.medium,
) -> pd.DataFrame:
    """Compute return-based columns for a single pair.

    Args:
        ohlcv: Aligned OHLCV frame for one pair.
        zscore_window: Rolling window used for the return z-score.

    Returns:
        A DataFrame indexed like the input with columns: close, log_return,
        simple_return, overnight_return, intraday_return, log_return_winsor,
        ret_zscore. Leading NaNs from differencing are left in place; the
        builder drops them once all pairs are aligned.

    Raises:
        KeyError: If required OHLC columns are absent.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise KeyError(f"OHLCV frame is missing columns: {sorted(missing)}")

    close = ohlcv["close"]
    out = pd.DataFrame(index=ohlcv.index)
    out["close"] = close
    out["high"] = ohlcv["high"]
    out["low"] = ohlcv["low"]
    out["open"] = ohlcv["open"]

    out["log_return"] = np.log(close / close.shift(1))
    out["simple_return"] = close.pct_change()
    # Overnight: previous close to today's open. Intraday: open to close.
    out["overnight_return"] = np.log(ohlcv["open"] / close.shift(1))
    out["intraday_return"] = np.log(close / ohlcv["open"])

    out["log_return_winsor"] = _winsorize(
        out["log_return"],
        settings.WINSORIZE_LOWER_PCT,
        settings.WINSORIZE_UPPER_PCT,
    )

    roll = out["log_return"].rolling(zscore_window)
    out["ret_zscore"] = (out["log_return"] - roll.mean()) / roll.std(ddof=0)
    return out


def _log_outliers(features: pd.DataFrame, symbol: str, sigma: float) -> None:
    """Log bars whose log return exceeds a multiple of its full-sample sigma.

    Args:
        features: Per-pair feature frame containing log_return.
        symbol: Pair symbol, for the log message.
        sigma: Number of standard deviations defining an outlier.

    Returns:
        None.
    """
    ret = features["log_return"].dropna()
    if ret.empty:
        return
    std = ret.std(ddof=0)
    if std == 0 or np.isnan(std):
        return
    extreme = ret[np.abs(ret - ret.mean()) > sigma * std]
    if not extreme.empty:
        logger.warning(
            "{}: {} bars exceed {} sigma (max |z|={:.1f})",
            symbol,
            len(extreme),
            sigma,
            float(np.abs((extreme - ret.mean()) / std).max()),
        )


def preprocess(
    raw_frames: dict[str, pd.DataFrame],
    pair_symbols: list[str] | None = None,
    persist: bool = True,
    processed_dir: Path = _PROCESSED_DIR,
) -> pd.DataFrame:
    """Build the master multi-level DataFrame from aligned raw frames.

    Args:
        raw_frames: Mapping of symbol to aligned OHLCV frame (from the fetcher).
        pair_symbols: Restrict processing to these symbols. Defaults to every
            configured FX pair present in raw_frames.
        persist: If True, write the result to processed_dir as parquet.
        processed_dir: Output directory for the processed parquet file.

    Returns:
        A DataFrame with a MultiIndex column of (pair, feature). The datetime
        index is the union business-day index from the fetcher.

    Raises:
        ValueError: If none of the requested pairs are present in raw_frames.
    """
    if pair_symbols is None:
        pair_symbols = [p.symbol for p in settings.FX_PAIRS]
    available = [s for s in pair_symbols if s in raw_frames]
    if not available:
        raise ValueError("None of the requested pairs are present in raw_frames.")

    per_pair: dict[str, pd.DataFrame] = {}
    for symbol in available:
        features = compute_pair_features(raw_frames[symbol])
        _log_outliers(features, symbol, settings.OUTLIER_RETURN_SIGMA)
        per_pair[symbol] = features

    master = pd.concat(per_pair, axis=1)
    master.columns = master.columns.set_names(["pair", "feature"])

    if persist:
        processed_dir.mkdir(parents=True, exist_ok=True)
        path = processed_dir / "master_returns.parquet"
        master.to_parquet(path)
        logger.info("Persisted processed master ({} rows) to {}", len(master), path)

    return master
