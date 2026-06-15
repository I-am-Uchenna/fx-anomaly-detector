"""Tests for the walk-forward backtester and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import metrics
from src.backtest.engine import _extract_trades, walk_forward_folds


def test_walk_forward_folds_sizes_and_purge() -> None:
    folds = walk_forward_folds(n_rows=1000, train_days=500, test_days=60, purge_days=5)
    assert len(folds) > 0
    for fold in folds:
        assert fold.train_end - fold.train_start == 500
        assert fold.test_end - fold.test_start == 60
        # The purge gap separates train and test.
        assert fold.test_start - fold.train_end == 5


def test_walk_forward_test_windows_do_not_overlap() -> None:
    folds = walk_forward_folds(n_rows=1000, train_days=400, test_days=63, purge_days=5)
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert later.test_start >= earlier.test_end


def test_walk_forward_raises_when_too_short() -> None:
    with pytest.raises(ValueError):
        walk_forward_folds(n_rows=100, train_days=500, test_days=60, purge_days=5)


def test_extract_trades_counts_round_trips() -> None:
    index = pd.bdate_range("2020-01-01", periods=6)
    positions = pd.DataFrame({"EURUSD=X": [0, -1, -1, 0, -1, 0]}, index=index)
    returns = pd.DataFrame({"EURUSD=X": [0.0, -0.01, -0.02, 0.0, 0.03, 0.0]}, index=index)
    trades = _extract_trades(positions, returns)
    assert len(trades) == 2
    # First trade: short over two down days -> positive return.
    first = trades.iloc[0]
    assert first["trade_return"] == pytest.approx(0.03)


def test_metrics_on_known_equity_curve() -> None:
    # Constant positive daily return -> positive Sharpe, zero drawdown.
    daily = pd.Series([0.001] * 252)
    equity = 1_000_000.0 * (1.0 + daily).cumprod()
    trades = pd.DataFrame({"trade_return": [0.02, -0.01, 0.03], "bars": [3, 2, 4]})
    result = metrics.compute_metrics(daily, equity, trades, risk_free_rate=0.0)
    assert result.max_drawdown == pytest.approx(0.0, abs=1e-9)
    assert result.sharpe_ratio > 0
    assert result.n_trades == 3
    assert result.win_rate == pytest.approx(2 / 3)


def test_metrics_profit_factor() -> None:
    daily = pd.Series(np.zeros(10))
    equity = pd.Series(np.ones(10))
    trades = pd.DataFrame({"trade_return": [0.04, -0.02], "bars": [2, 2]})
    result = metrics.compute_metrics(daily, equity, trades)
    assert result.profit_factor == pytest.approx(2.0)


def _small_features() -> pd.DataFrame:
    """Build a small two-pair feature frame for backtest/reporter tests."""
    from src.features.builder import build_features

    index = pd.bdate_range("2014-01-01", periods=820)

    def ohlcv(seed: int, start: float) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rets = rng.normal(0, 0.005, len(index))
        close = start * np.exp(np.cumsum(rets))
        span = np.abs(rets) + 0.0004
        return pd.DataFrame(
            {
                "open": close,
                "high": close * (1 + span),
                "low": close * (1 - span),
                "close": close,
                "volume": 0.0,
            },
            index=index,
        )

    raw = {"EURUSD=X": ohlcv(1, 1.2), "GBPUSD=X": ohlcv(2, 1.3)}
    raw["^VIX"] = ohlcv(3, 18.0)
    raw["DX-Y.NYB"] = ohlcv(4, 96.0)
    raw["GC=F"] = ohlcv(5, 1800.0)
    return build_features(raw, pair_symbols=["EURUSD=X", "GBPUSD=X"], persist=False, validate=False)


def test_run_backtest_produces_equity_and_report(tmp_path) -> None:
    from dataclasses import replace

    from config import settings
    from src.backtest.engine import run_backtest
    from src.backtest.reporter import generate_report

    features = _small_features()
    cfg = replace(settings.BACKTEST, walk_forward_train_days=300, walk_forward_test_days=63)
    result = run_backtest(features, config=cfg, include_autoencoder=False)
    assert len(result.equity_curve) > 0
    assert result.strategy_returns.notna().any()

    report = generate_report(
        result,
        metrics.compute_metrics(result.strategy_returns, result.equity_curve, result.trades),
        tmp_path / "report.html",
    )
    assert report.exists()
    assert report.stat().st_size > 1000
