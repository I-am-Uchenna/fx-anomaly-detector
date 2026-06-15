"""Shared pytest fixtures.

Generates deterministic synthetic FX data with and without injected anomalies
(return spikes, volatility jumps, correlation breaks). All randomness is seeded
so tests are reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

_N_BARS = 900
_SEED = 12345

# Dates where anomalies are injected (positional indices into the series).
SPIKE_INDEX = 600
VOL_JUMP_START = 700
VOL_JUMP_LEN = 25


def _build_ohlcv(returns: np.ndarray, index: pd.DatetimeIndex, start_price: float) -> pd.DataFrame:
    """Construct an OHLCV frame from a return path.

    Args:
        returns: Daily log returns.
        index: Datetime index.
        start_price: Initial price level.

    Returns:
        An OHLCV DataFrame with a plausible intraday range around each close.
    """
    close = start_price * np.exp(np.cumsum(returns))
    rng = np.random.default_rng(int(start_price * 1000) % 2**32)
    # Intraday range scales with the size of the day's move (as in real data),
    # so high-low based estimators track close-to-close volatility.
    span = np.abs(returns) + np.abs(rng.normal(0, 0.0008, len(close))) + 0.0003
    high = close * (1 + span)
    low = close * (1 - span)
    open_ = close * (1 + rng.normal(0, 0.0008, len(close)))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 0.0}, index=index
    )


def _make_raw(inject: bool) -> dict[str, pd.DataFrame]:
    """Build a dictionary of synthetic OHLCV frames.

    Args:
        inject: If True, inject a return spike and a volatility jump.

    Returns:
        Mapping of symbol to OHLCV frame for four pairs plus macro tickers.
    """
    index = pd.bdate_range("2014-01-01", periods=_N_BARS)
    symbols: list[str] = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
    raw: dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(symbols):
        rng = np.random.default_rng(_SEED + i)
        returns = rng.normal(0, 0.005, _N_BARS)
        if inject:
            returns[SPIKE_INDEX] += 0.08  # large one-day move
            returns[VOL_JUMP_START : VOL_JUMP_START + VOL_JUMP_LEN] *= 6.0  # volatility jump
        start = 110.0 if "JPY" in symbol else 1.2
        raw[symbol] = _build_ohlcv(returns, index, start)

    for j, macro in enumerate(["^VIX", "DX-Y.NYB", "GC=F"]):
        rng = np.random.default_rng(_SEED + 100 + j)
        returns = rng.normal(0, 0.01, _N_BARS)
        start = {"^VIX": 18.0, "DX-Y.NYB": 96.0, "GC=F": 1800.0}[macro]
        raw[macro] = _build_ohlcv(returns, index, start)
    return raw


@pytest.fixture(scope="session")
def synthetic_raw() -> dict[str, pd.DataFrame]:
    """Synthetic raw frames with injected anomalies."""
    return _make_raw(inject=True)


@pytest.fixture(scope="session")
def clean_raw() -> dict[str, pd.DataFrame]:
    """Synthetic raw frames without injected anomalies."""
    return _make_raw(inject=False)


@pytest.fixture(scope="session")
def feature_frame(synthetic_raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Feature frame built from the anomalous synthetic data."""
    from src.features.builder import build_features

    pairs = [s for s in synthetic_raw if s.endswith("=X")]
    return build_features(synthetic_raw, pair_symbols=pairs, persist=False, validate=False)


@pytest.fixture(scope="session")
def clean_feature_frame(clean_raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Feature frame built from the clean synthetic data."""
    from src.features.builder import build_features

    pairs = [s for s in clean_raw if s.endswith("=X")]
    return build_features(clean_raw, pair_symbols=pairs, persist=False, validate=False)
