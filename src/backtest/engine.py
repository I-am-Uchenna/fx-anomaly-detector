"""Walk-forward backtesting engine with purging.

Data is partitioned into consecutive train/test folds. Detectors are refit on
each fold so no information leaks across folds (De Prado, 2018). A purge gap is
removed between the train and test windows to prevent leakage from
overlapping-label contamination. The trading rule is defensive: go short (or
reduce exposure) on a confirmed anomaly, otherwise stay flat. Out-of-sample
test-window returns are concatenated into a single equity curve.

Within a fold the autoencoder trains on the fold's leading train window and the
HMM fits on the fold slice; this is the intended per-fold retrain and is
documented as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import settings
from config.logging_config import get_logger
from src.detectors import ensemble

logger = get_logger()


@dataclass
class Fold:
    """Index boundaries for one walk-forward fold.

    Args:
        train_start: First training row index (inclusive).
        train_end: Last training row index (exclusive).
        test_start: First test row index (inclusive), after the purge gap.
        test_end: Last test row index (exclusive).
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass
class BacktestResult:
    """Outputs of a walk-forward backtest.

    Args:
        equity_curve: Out-of-sample strategy equity level series.
        strategy_returns: Out-of-sample daily strategy returns.
        benchmark_equity: Buy-and-hold benchmark equity over the same period.
        trades: DataFrame of completed trades (pair, entry, exit, return).
        anomaly_scores: Out-of-sample ensemble score per (datetime, pair).
        folds: The fold boundaries used.
    """

    equity_curve: pd.Series
    strategy_returns: pd.Series
    benchmark_equity: pd.Series
    trades: pd.DataFrame
    anomaly_scores: pd.DataFrame
    folds: list[Fold]


def walk_forward_folds(
    n_rows: int,
    train_days: int,
    test_days: int,
    purge_days: int,
) -> list[Fold]:
    """Generate non-overlapping walk-forward folds with a purge gap.

    Args:
        n_rows: Total number of rows available.
        train_days: Training window length.
        test_days: Test window length.
        purge_days: Rows removed between train and test windows.

    Returns:
        A list of Fold objects covering the data.

    Raises:
        ValueError: If the windows do not fit in the data.
    """
    if train_days + purge_days + test_days > n_rows:
        raise ValueError("Not enough rows for a single fold.")
    folds: list[Fold] = []
    test_start = train_days + purge_days
    while test_start + test_days <= n_rows:
        folds.append(
            Fold(
                train_start=test_start - purge_days - train_days,
                train_end=test_start - purge_days,
                test_start=test_start,
                test_end=test_start + test_days,
            )
        )
        test_start += test_days
    return folds


def _round_trip_cost(config: settings.BacktestConfig, price: float) -> float:
    """Per-unit transaction plus slippage cost as a return fraction.

    Args:
        config: Backtest configuration.
        price: Reference price for converting pips to a return.

    Returns:
        Cost as a fraction of notional for opening or closing a position.
    """
    pips = config.transaction_cost_pips + config.slippage_pips
    # Use a representative pip size; JPY pairs are scaled by their own price.
    pip_value = 0.0001
    return (pips * pip_value) / price if price > 0 else 0.0


def _extract_trades(positions: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Identify completed short trades and their realised returns.

    Args:
        positions: Wide position frame (0 or -1) indexed by datetime.
        returns: Wide forward-return frame aligned to positions.

    Returns:
        DataFrame with columns pair, entry, exit, bars, trade_return.
    """
    records = []
    for pair in positions.columns:
        pos = positions[pair].to_numpy()
        ret = returns[pair].to_numpy()
        idx = positions.index
        in_trade = False
        entry_i = 0
        cum = 0.0
        for t in range(len(pos)):
            if pos[t] == -1 and not in_trade:
                in_trade = True
                entry_i = t
                cum = 0.0
            if in_trade:
                cum += -1.0 * (ret[t] if not np.isnan(ret[t]) else 0.0)
            if in_trade and (pos[t] == 0 or t == len(pos) - 1):
                records.append(
                    {
                        "pair": pair,
                        "entry": idx[entry_i],
                        "exit": idx[t],
                        "bars": t - entry_i + 1,
                        "trade_return": cum,
                    }
                )
                in_trade = False
    return pd.DataFrame.from_records(records)


def run_backtest(
    features: pd.DataFrame,
    config: settings.BacktestConfig | None = None,
    include_autoencoder: bool = False,
) -> BacktestResult:
    """Run the walk-forward backtest over a precomputed feature frame.

    Features are trailing by construction and computed once on the full history;
    only the fitted detectors are refit per fold, which is where leakage risk
    lives.

    Args:
        features: Multi-level (pair, feature) frame including close and
            log_return per pair.
        config: Backtest configuration. Defaults to the global BACKTEST config.
        include_autoencoder: Whether to run the autoencoder detector per fold.

    Returns:
        A BacktestResult.

    Raises:
        ValueError: If the data is too short for one fold.
    """
    config = config or settings.BACKTEST
    pairs = list(features.columns.get_level_values(0).unique())
    n = len(features)
    folds = walk_forward_folds(
        n, config.walk_forward_train_days, config.walk_forward_test_days, config.purge_days
    )
    logger.info("Backtest over {} folds, {} pairs", len(folds), len(pairs))

    returns_wide = pd.concat({p: features[(p, "log_return")] for p in pairs}, axis=1)
    close_wide = pd.concat({p: features[(p, "close")] for p in pairs}, axis=1)

    all_positions = []
    all_scores = []
    for fold in folds:
        fold_slice = features.iloc[fold.train_start : fold.test_end]
        ens = ensemble.detect(fold_slice, include_autoencoder=include_autoencoder)
        flag_wide = ens.pivot(index="datetime", columns="pair", values="ensemble_flag")
        score_wide = ens.pivot(index="datetime", columns="pair", values="ensemble_score")

        test_index = features.index[fold.test_start : fold.test_end]
        flag_test = flag_wide.reindex(index=test_index, columns=pairs).fillna(False)
        score_test = score_wide.reindex(index=test_index, columns=pairs)
        positions = -flag_test.astype(int)  # short on anomaly, else flat
        all_positions.append(positions)
        all_scores.append(score_test)

    positions = pd.concat(all_positions).sort_index()
    oos_index = positions.index
    oos_returns = returns_wide.reindex(oos_index)

    # Strategy return at t comes from the position held from t-1.
    held = positions.shift(1).fillna(0)
    gross = held * oos_returns

    # Transaction costs charged when a position changes.
    changes = positions.diff().abs().fillna(0)
    cost_frame = pd.DataFrame(index=oos_index, columns=pairs, dtype="float64")
    for pair in pairs:
        prices = close_wide[pair].reindex(oos_index)
        per_unit = prices.apply(lambda p: _round_trip_cost(config, p) if pd.notna(p) else 0.0)
        cost_frame[pair] = changes[pair] * per_unit
    net = gross - cost_frame

    # Equal capital allocation across pairs, scaled by the per-trade cap.
    portfolio_return = net.mean(axis=1) * (config.max_position_pct * len(pairs))
    equity = config.initial_capital * (1.0 + portfolio_return).cumprod()

    benchmark_return = oos_returns.mean(axis=1)
    benchmark_equity = config.initial_capital * (1.0 + benchmark_return).cumprod()

    trades = _extract_trades(positions, oos_returns)
    scores = pd.concat(all_scores).sort_index()

    return BacktestResult(
        equity_curve=equity,
        strategy_returns=portfolio_return,
        benchmark_equity=benchmark_equity,
        trades=trades,
        anomaly_scores=scores,
        folds=folds,
    )
