"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import cross_pair, returns, volatility
from src.features.returns import hurst_exponent


def test_hurst_random_walk_near_half() -> None:
    rng = np.random.default_rng(0)
    series = rng.normal(0, 1, 4096)
    h = hurst_exponent(series)
    assert abs(h - 0.5) < 0.12


def test_hurst_trending_above_half() -> None:
    # AR(1) with positive coefficient produces persistence (trending).
    rng = np.random.default_rng(1)
    n = 4096
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + rng.normal(0, 1)
    h = hurst_exponent(x)
    assert h > 0.5


def test_hurst_mean_reverting_below_half() -> None:
    # AR(1) with negative coefficient produces anti-persistence.
    rng = np.random.default_rng(2)
    n = 4096
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = -0.6 * x[t - 1] + rng.normal(0, 1)
    h = hurst_exponent(x)
    assert h < 0.5


def test_hurst_too_short_returns_nan() -> None:
    assert np.isnan(hurst_exponent(np.arange(5.0)))


def test_parkinson_correlates_with_close_to_close() -> None:
    # Build data with genuine time-varying volatility so both estimators have
    # something to track; intraday range scales with the day's move.
    index = pd.bdate_range("2015-01-01", periods=600)
    rng = np.random.default_rng(11)
    daily_vol = 0.003 + 0.02 * (np.sin(np.linspace(0, 12 * np.pi, len(index))) + 1) / 2
    rets = rng.normal(0, 1, len(index)) * daily_vol
    close = 1.2 * np.exp(np.cumsum(rets))
    span = np.abs(rets) + 0.0003
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * (1 + span),
            "low": close * (1 - span),
            "close": close,
            "log_return": pd.Series(close, index=index).pipe(np.log).diff(),
        },
        index=index,
    )
    master = pd.concat({"EURUSD=X": frame}, axis=1)
    master.columns = master.columns.set_names(["pair", "feature"])
    vol = volatility.compute(master, windows=[21])
    cc = vol[("EURUSD=X", "vol_close_21")].dropna()
    park = vol[("EURUSD=X", "vol_parkinson_21")].dropna()
    common = cc.index.intersection(park.index)
    corr = np.corrcoef(cc.loc[common], park.loc[common])[0, 1]
    assert corr > 0.3


def test_triangular_residual_near_zero_for_consistent_data() -> None:
    # Build EURJPY exactly as EURUSD * USDJPY so the residual must be ~0.
    index = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(3)
    eurusd = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.004, len(index))))
    usdjpy = 110.0 * np.exp(np.cumsum(rng.normal(0, 0.004, len(index))))
    eurjpy = eurusd * usdjpy

    def ohlcv(close: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 0.0,
            },
            index=index,
        )

    from src.data.preprocessor import preprocess

    raw = {"EURUSD=X": ohlcv(eurusd), "USDJPY=X": ohlcv(usdjpy), "EURJPY=X": ohlcv(eurjpy)}
    master = preprocess(raw, pair_symbols=list(raw), persist=False)
    cp = cross_pair.compute(master, windows=[21])
    residual = cp[("EURJPY=X", "triangular_arb_residual")].dropna()
    assert residual.abs().max() < 1e-6


def test_returns_features_present(feature_frame: pd.DataFrame) -> None:
    cols = feature_frame["EURUSD=X"].columns
    for name in ["ret_skew_21", "ret_kurt_21", "ret_zscore_21", "ret_jarque_bera_pval_63"]:
        assert name in cols


def test_microstructure_spread_non_negative(feature_frame: pd.DataFrame) -> None:
    spread = feature_frame[("EURUSD=X", "spread_proxy_hl")].dropna()
    assert (spread >= 0).all()


def test_returns_compute_skew_matches_pandas() -> None:
    index = pd.bdate_range("2015-01-01", periods=120)
    rng = np.random.default_rng(7)
    close = 1.2 * np.exp(np.cumsum(rng.normal(0, 0.005, len(index))))
    master = pd.concat(
        {
            "EURUSD=X": pd.DataFrame(
                {
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "log_return": pd.Series(close, index=index).pipe(np.log).diff(),
                },
                index=index,
            )
        },
        axis=1,
    )
    master.columns = master.columns.set_names(["pair", "feature"])
    feats = returns.compute(master, windows=[21])
    expected = master[("EURUSD=X", "log_return")].rolling(21).skew()
    pd.testing.assert_series_equal(feats[("EURUSD=X", "ret_skew_21")], expected, check_names=False)
