"""Performance metrics for a backtest equity curve and trade list."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from config import settings
from src.risk.drawdown import summarize_drawdown


@dataclass
class PerformanceMetrics:
    """Standard performance statistics."""

    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    avg_win_loss_ratio: float
    n_trades: int
    avg_holding_period: float
    annualized_turnover: float

    def as_dict(self) -> dict:
        """Return the metrics as a plain dictionary.

        Returns:
            Mapping of metric name to value.
        """
        return asdict(self)


def compute_metrics(
    strategy_returns: pd.Series,
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    risk_free_rate: float = settings.BACKTEST.risk_free_rate,
    trading_days: int = settings.BACKTEST.trading_days_per_year,
) -> PerformanceMetrics:
    """Compute performance metrics from returns, equity and trades.

    Args:
        strategy_returns: Daily strategy returns.
        equity_curve: Strategy equity level series.
        trades: DataFrame with columns trade_return and bars.
        risk_free_rate: Annual risk-free rate for excess-return metrics.
        trading_days: Trading days per year for annualisation.

    Returns:
        A PerformanceMetrics instance.

    Raises:
        ValueError: If inputs are empty.
    """
    if equity_curve.empty or strategy_returns.empty:
        raise ValueError("equity_curve and strategy_returns must be non-empty.")

    n = len(equity_curve)
    total_growth = float(equity_curve.iloc[-1] / equity_curve.iloc[0])
    years = n / trading_days
    annualized_return = total_growth ** (1.0 / years) - 1.0 if years > 0 else 0.0

    daily = strategy_returns.dropna()
    annualized_vol = float(daily.std(ddof=1) * np.sqrt(trading_days))
    sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0.0

    downside = daily[daily < 0]
    downside_dev = float(downside.std(ddof=1) * np.sqrt(trading_days)) if len(downside) > 1 else 0.0
    sortino = (annualized_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0

    dd = summarize_drawdown(equity_curve)
    calmar = annualized_return / dd.max_drawdown if dd.max_drawdown > 0 else 0.0

    if trades.empty:
        win_rate = profit_factor = avg_win_loss = avg_hold = 0.0
        n_trades = 0
    else:
        tr = trades["trade_return"]
        wins = tr[tr > 0]
        losses = tr[tr < 0]
        n_trades = int(len(tr))
        win_rate = float(len(wins) / n_trades)
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = -losses.mean() if len(losses) > 0 else 0.0
        avg_win_loss = float(avg_win / avg_loss) if avg_loss > 0 else 0.0
        avg_hold = float(trades["bars"].mean())

    annualized_turnover = float(n_trades / years) if years > 0 else 0.0

    return PerformanceMetrics(
        annualized_return=annualized_return,
        annualized_volatility=annualized_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=dd.max_drawdown,
        max_drawdown_duration=dd.max_drawdown_duration,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win_loss_ratio=avg_win_loss,
        n_trades=n_trades,
        avg_holding_period=avg_hold,
        annualized_turnover=annualized_turnover,
    )
