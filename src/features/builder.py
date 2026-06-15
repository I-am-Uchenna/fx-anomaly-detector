"""Feature pipeline orchestrator.

Runs every feature module against the preprocessed master frame, concatenates
the results into a single (pair, feature) DataFrame, drops the leading warm-up
rows, logs per-feature completeness, runs a lookahead-bias spot check, and
optionally persists the result to the feature store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from config.logging_config import get_logger
from src.data import feature_store, preprocessor
from src.features import cross_pair, macro, microstructure, returns, volatility

logger = get_logger()

_LOOKAHEAD_TOLERANCE = 1e-9


def _validate_no_lookahead(master_returns: pd.DataFrame) -> None:
    """Spot check that trailing-window features do not use future data.

    Recomputes a representative rolling feature on a truncated copy of the
    input and confirms the value at the truncation point matches the value from
    the full-history computation.

    Args:
        master_returns: Preprocessed (pair, feature) frame with log_return.

    Returns:
        None.

    Raises:
        ValueError: If a recomputed past value differs from the full-history
            value, which would indicate lookahead leakage.
    """
    pair = master_returns.columns.get_level_values(0)[0]
    cutoff = int(len(master_returns) * 0.7)
    if cutoff < settings.WINDOWS.medium + 5:
        return  # too short to check meaningfully

    full = returns.compute(master_returns, windows=[settings.WINDOWS.medium])
    truncated = returns.compute(
        master_returns.iloc[: cutoff + 1], windows=[settings.WINDOWS.medium]
    )
    feature = f"ret_zscore_{settings.WINDOWS.medium}"
    full_value = full[(pair, feature)].iloc[cutoff]
    trunc_value = truncated[(pair, feature)].iloc[cutoff]
    if np.isnan(full_value) and np.isnan(trunc_value):
        return
    if abs(full_value - trunc_value) > _LOOKAHEAD_TOLERANCE:
        raise ValueError(
            "Lookahead check failed: feature value at a past bar changed when "
            "future data was added."
        )


def _log_completeness(frame: pd.DataFrame) -> None:
    """Log the fraction of non-null values for the least complete features.

    Args:
        frame: The assembled feature frame.

    Returns:
        None.
    """
    completeness = frame.notna().mean().sort_values()
    worst = completeness.head(5)
    for col, ratio in worst.items():
        logger.info("Feature completeness {}: {:.3f}", col, ratio)


def build_features(
    raw_frames: dict[str, pd.DataFrame],
    pair_symbols: list[str] | None = None,
    persist: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """Build the full feature matrix from aligned raw frames.

    Args:
        raw_frames: Fetcher output (symbol to OHLCV frame), including macro
            tickers.
        pair_symbols: Pairs to build features for. Defaults to all FX pairs.
        persist: If True, save the result to the feature store.
        validate: If True, run the lookahead-bias spot check.

    Returns:
        A (pair, feature) DataFrame with leading warm-up rows removed.

    Raises:
        ValueError: If the lookahead check fails.
    """
    master = preprocessor.preprocess(raw_frames, pair_symbols=pair_symbols, persist=False)

    if validate:
        _validate_no_lookahead(master)

    blocks = [
        master,
        returns.compute(master),
        volatility.compute(master),
        microstructure.compute(master),
        cross_pair.compute(master),
        macro.compute(master, raw_frames=raw_frames),
    ]
    combined = pd.concat(blocks, axis=1)
    combined = combined.sort_index(axis=1)
    combined.columns = combined.columns.set_names(["pair", "feature"])

    # Drop the leading warm-up region where the longest window is still filling.
    warmup = settings.WINDOWS.extended
    if len(combined) > warmup:
        combined = combined.iloc[warmup:]

    _log_completeness(combined)

    if persist:
        feature_store.save_features(combined, name="features")

    return combined
