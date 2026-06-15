"""HTML backtest report generation using Plotly.

Assembles the equity curve, drawdown, monthly-returns heatmap, rolling Sharpe,
anomaly-score timeline, optional regime timeline, a summary statistics table
and a per-pair breakdown into a single self-contained HTML file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import settings
from config.logging_config import get_logger
from src.backtest.engine import BacktestResult
from src.backtest.metrics import PerformanceMetrics
from src.risk.drawdown import underwater_curve

logger = get_logger()


def _fig_html(fig: go.Figure, include_js: bool) -> str:
    """Render a figure to an embeddable HTML fragment.

    Args:
        fig: The Plotly figure.
        include_js: Whether to inline the Plotly.js bundle (once per report).

    Returns:
        HTML fragment string.
    """
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if include_js else False)


def _equity_figure(result: BacktestResult) -> go.Figure:
    """Build the strategy-versus-benchmark equity figure."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=result.equity_curve.index, y=result.equity_curve.values, name="Strategy")
    )
    fig.add_trace(
        go.Scatter(
            x=result.benchmark_equity.index,
            y=result.benchmark_equity.values,
            name="Buy and hold",
        )
    )
    fig.update_layout(title="Out-of-sample equity curve", xaxis_title="Date", yaxis_title="Equity")
    return fig


def _drawdown_figure(result: BacktestResult) -> go.Figure:
    """Build the underwater drawdown figure."""
    underwater = underwater_curve(result.equity_curve)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=underwater.index, y=underwater.values, fill="tozeroy", name="Drawdown")
    )
    fig.update_layout(title="Drawdown (underwater)", xaxis_title="Date", yaxis_title="Drawdown")
    return fig


def _monthly_heatmap(result: BacktestResult) -> go.Figure:
    """Build a year-by-month returns heatmap."""
    rets = result.strategy_returns.dropna()
    if rets.empty:
        return go.Figure()
    monthly = (1.0 + rets).resample("ME").prod() - 1.0
    table = monthly.to_frame("ret")
    table["year"] = table.index.year
    table["month"] = table.index.month
    pivot = table.pivot_table(index="year", columns="month", values="ret")
    fig = go.Figure(
        go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="RdYlGn", zmid=0)
    )
    fig.update_layout(title="Monthly returns", xaxis_title="Month", yaxis_title="Year")
    return fig


def _rolling_sharpe_figure(result: BacktestResult, window: int = 63) -> go.Figure:
    """Build the rolling annualised Sharpe figure."""
    rets = result.strategy_returns.dropna()
    roll = rets.rolling(window)
    ann = settings.BACKTEST.trading_days_per_year
    sharpe = (roll.mean() * ann - settings.BACKTEST.risk_free_rate) / (
        roll.std(ddof=1) * np.sqrt(ann)
    )
    fig = go.Figure(go.Scatter(x=sharpe.index, y=sharpe.values, name="Rolling Sharpe"))
    fig.update_layout(
        title=f"Rolling Sharpe ({window}-day)", xaxis_title="Date", yaxis_title="Sharpe"
    )
    return fig


def _anomaly_timeline(result: BacktestResult) -> go.Figure:
    """Build the anomaly-score timeline averaged across pairs."""
    scores = result.anomaly_scores
    fig = go.Figure()
    if not scores.empty:
        mean_score = scores.mean(axis=1)
        fig.add_trace(
            go.Scatter(x=mean_score.index, y=mean_score.values, name="Mean ensemble score")
        )
    fig.update_layout(
        title="Anomaly score timeline", xaxis_title="Date", yaxis_title="Ensemble score"
    )
    return fig


def _regime_timeline(regime_states: dict[str, pd.Series]) -> go.Figure:
    """Build a regime-state timeline figure for the provided pairs."""
    fig = go.Figure()
    for pair, series in regime_states.items():
        clean = series.dropna()
        if not clean.empty:
            fig.add_trace(go.Scatter(x=clean.index, y=clean.values, name=pair, mode="lines"))
    fig.update_layout(title="Regime state timeline", xaxis_title="Date", yaxis_title="Regime")
    return fig


def _metrics_table_html(metrics: PerformanceMetrics) -> str:
    """Render the metrics dataclass as an HTML table."""
    rows = "".join(
        (
            f"<tr><td>{key}</td><td>{value:.4f}</td></tr>"
            if isinstance(value, float)
            else f"<tr><td>{key}</td><td>{value}</td></tr>"
        )
        for key, value in metrics.as_dict().items()
    )
    return f"<table border='1' cellpadding='6'><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"


def _per_pair_html(result: BacktestResult) -> str:
    """Render a per-pair trade breakdown as an HTML table."""
    if result.trades.empty:
        return "<p>No trades.</p>"
    grouped = result.trades.groupby("pair").agg(
        n_trades=("trade_return", "size"),
        total_return=("trade_return", "sum"),
        win_rate=("trade_return", lambda s: float((s > 0).mean())),
    )
    return grouped.round(4).to_html(border=1)


def generate_report(
    result: BacktestResult,
    metrics: PerformanceMetrics,
    output_path: Path,
    regime_states: dict[str, pd.Series] | None = None,
) -> Path:
    """Generate the full HTML backtest report.

    Args:
        result: The backtest result.
        metrics: Computed performance metrics.
        output_path: Destination HTML file path.
        regime_states: Optional map of pair to regime-state series.

    Returns:
        The path written.
    """
    figures = [
        _equity_figure(result),
        _drawdown_figure(result),
        _monthly_heatmap(result),
        _rolling_sharpe_figure(result),
        _anomaly_timeline(result),
    ]
    if regime_states:
        figures.append(_regime_timeline(regime_states))

    fragments = [_fig_html(fig, include_js=(i == 0)) for i, fig in enumerate(figures)]
    body = "\n".join(fragments)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>FX Anomaly Detector Backtest Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
h1, h2 {{ color: #1f2937; }}
table {{ border-collapse: collapse; margin: 12px 0; }}
</style>
</head>
<body>
<h1>FX Anomaly Detector Backtest Report</h1>
<h2>Summary statistics</h2>
{_metrics_table_html(metrics)}
<h2>Charts</h2>
{body}
<h2>Per-pair breakdown</h2>
{_per_pair_html(result)}
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Wrote backtest report to {}", output_path)
    return output_path
