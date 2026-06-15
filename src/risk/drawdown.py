"""Drawdown analysis for an equity curve.

Computes the running peak, the underwater (drawdown) series, the maximum
drawdown and its duration, the current drawdown depth, and a simple recovery
time estimate based on the average historical recovery rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DrawdownSummary:
    """Summary drawdown statistics for an equity curve.

    Args:
        max_drawdown: Largest peak-to-trough decline as a positive fraction.
        max_drawdown_duration: Longest run of consecutive underwater bars.
        current_drawdown: Current drawdown depth as a positive fraction.
        current_duration: Bars since the last equity peak.
        estimated_recovery_bars: Estimated bars to recover the current
            drawdown, or NaN if not underwater or not estimable.
    """

    max_drawdown: float
    max_drawdown_duration: int
    current_drawdown: float
    current_duration: int
    estimated_recovery_bars: float


def underwater_curve(equity: pd.Series) -> pd.Series:
    """Compute the underwater (drawdown) series of an equity curve.

    Args:
        equity: Equity level series (must be positive).

    Returns:
        Series of drawdowns as non-positive fractions (0 at new highs).

    Raises:
        ValueError: If the equity series is empty.
    """
    if equity.empty:
        raise ValueError("equity series is empty.")
    running_peak = equity.cummax()
    return equity / running_peak - 1.0


def _max_underwater_duration(drawdown: pd.Series) -> int:
    """Longest consecutive run of strictly negative drawdown values.

    Args:
        drawdown: Underwater series (<= 0).

    Returns:
        Longest underwater run length in bars.
    """
    underwater = (drawdown < 0).to_numpy()
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def _current_duration(drawdown: pd.Series) -> int:
    """Number of trailing bars since the last new equity peak.

    Args:
        drawdown: Underwater series (<= 0).

    Returns:
        Count of trailing consecutive underwater bars.
    """
    underwater = (drawdown < 0).to_numpy()
    count = 0
    for flag in reversed(underwater):
        if not flag:
            break
        count += 1
    return count


def summarize_drawdown(equity: pd.Series) -> DrawdownSummary:
    """Compute the full drawdown summary for an equity curve.

    Args:
        equity: Equity level series.

    Returns:
        A DrawdownSummary.
    """
    drawdown = underwater_curve(equity)
    max_dd = float(-drawdown.min())
    max_duration = _max_underwater_duration(drawdown)
    current_dd = float(-drawdown.iloc[-1])
    current_duration = _current_duration(drawdown)

    # Estimate recovery time from the average per-bar gain during prior
    # recoveries (climbs back toward the running peak).
    gains = equity.diff().clip(lower=0)
    avg_gain = gains[gains > 0].mean()
    peak = equity.cummax().iloc[-1]
    shortfall = peak - equity.iloc[-1]
    if current_dd > 0 and avg_gain and avg_gain > 0:
        estimated_recovery = float(shortfall / avg_gain)
    else:
        estimated_recovery = float("nan")

    return DrawdownSummary(
        max_drawdown=max_dd,
        max_drawdown_duration=max_duration,
        current_drawdown=current_dd,
        current_duration=current_duration,
        estimated_recovery_bars=estimated_recovery,
    )
