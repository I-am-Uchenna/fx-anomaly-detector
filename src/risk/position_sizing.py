"""Kelly-criterion position sizing.

The Kelly fraction maximises the long-run growth rate of capital. Full Kelly is
aggressive and sensitive to estimation error in the win probability, so half
Kelly is the standard risk-adjusted choice. The output is always capped at the
configured per-trade maximum regardless of the Kelly recommendation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from config import settings


@dataclass(frozen=True)
class KellySizing:
    """Kelly sizing result.

    Args:
        full_kelly: Unconstrained full Kelly fraction (may be negative).
        half_kelly: Half of the full Kelly fraction.
        capped_fraction: Fraction after flooring at zero and capping at the
            configured per-trade maximum.
    """

    full_kelly: float
    half_kelly: float
    capped_fraction: float


def kelly_fraction(win_probability: float, win_loss_ratio: float) -> float:
    """Full Kelly fraction for a bet with given edge.

    Args:
        win_probability: Probability of a winning trade, in [0, 1].
        win_loss_ratio: Ratio of average win size to average loss size (b > 0).

    Returns:
        The full Kelly fraction f* = (p * b - q) / b.

    Raises:
        ValueError: If inputs are out of range.
    """
    if not 0.0 <= win_probability <= 1.0:
        raise ValueError("win_probability must be in [0, 1].")
    if win_loss_ratio <= 0.0:
        raise ValueError("win_loss_ratio must be positive.")
    q = 1.0 - win_probability
    return (win_probability * win_loss_ratio - q) / win_loss_ratio


def estimate_from_outcomes(trade_returns: Sequence[float]) -> tuple[float, float]:
    """Estimate win probability and win/loss ratio from trade outcomes.

    Args:
        trade_returns: Realised per-trade returns.

    Returns:
        A tuple of (win_probability, win_loss_ratio). The ratio defaults to 1.0
        when there are no losses (or no wins) to avoid division by zero.
    """
    arr = np.asarray(trade_returns, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_prob = wins.size / arr.size
    avg_win = wins.mean() if wins.size > 0 else 0.0
    avg_loss = -losses.mean() if losses.size > 0 else 0.0
    win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 1.0
    return float(win_prob), float(win_loss_ratio)


def size_position(
    win_probability: float,
    win_loss_ratio: float,
    max_fraction: float = settings.BACKTEST.max_position_pct,
) -> KellySizing:
    """Compute capped half-Kelly position size.

    Args:
        win_probability: Probability of a winning trade.
        win_loss_ratio: Average win over average loss.
        max_fraction: Hard cap on the position fraction.

    Returns:
        A KellySizing. The capped fraction uses half Kelly, floored at zero and
        capped at max_fraction.
    """
    full = kelly_fraction(win_probability, win_loss_ratio)
    half = full / 2.0
    capped = float(np.clip(half, 0.0, max_fraction))
    return KellySizing(full_kelly=full, half_kelly=half, capped_fraction=capped)
