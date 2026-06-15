"""Anomaly detector package.

Every detector returns a long-format DataFrame with a fixed schema so the
ensemble can align them by (datetime, pair). Shared helpers for building that
schema and for normalising raw scores into the unit interval live here to avoid
duplication across the detector modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STANDARD_COLUMNS = ["datetime", "pair", "anomaly_score", "anomaly_flag", "detector_name"]

# Raw price columns are non-stationary and excluded from detector feature
# vectors; engineered features are used instead.
_PRICE_COLUMNS = {"open", "high", "low", "close"}


def normalize_unit_interval(values: pd.Series) -> pd.Series:
    """Rank-normalise a score series into [0, 1].

    Rank normalisation is robust to outliers and unaffected by the raw scale of
    the input, which differs across detectors.

    Args:
        values: Raw anomaly scores (higher means more anomalous).

    Returns:
        Scores mapped to [0, 1]; NaNs are preserved.
    """
    ranks = values.rank(method="average", na_option="keep")
    count = ranks.notna().sum()
    if count <= 1:
        return pd.Series(np.where(values.notna(), 0.0, np.nan), index=values.index)
    return (ranks - 1.0) / (count - 1.0)


def pair_feature_matrix(features: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Extract a single pair's engineered feature matrix.

    Args:
        features: Multi-level (pair, feature) feature frame.
        pair: Pair symbol to extract.

    Returns:
        A DataFrame indexed by datetime with one column per engineered feature.
        Raw price columns and all-NaN columns are dropped.

    Raises:
        KeyError: If the pair is absent from the feature frame.
    """
    if pair not in features.columns.get_level_values(0):
        raise KeyError(f"Pair {pair} not present in feature frame.")
    block = features[pair]
    keep = [c for c in block.columns if c not in _PRICE_COLUMNS]
    block = block[keep].dropna(axis=1, how="all")
    return block


def to_long_format(
    score_wide: pd.DataFrame,
    flag_wide: pd.DataFrame,
    detector_name: str,
) -> pd.DataFrame:
    """Convert wide per-pair score and flag frames to the standard long format.

    Args:
        score_wide: DataFrame indexed by datetime with one column per pair of
            anomaly scores in [0, 1].
        flag_wide: DataFrame with the same shape of boolean anomaly flags.
        detector_name: Name recorded in the detector_name column.

    Returns:
        A long DataFrame with columns STANDARD_COLUMNS.

    Raises:
        ValueError: If the score and flag frames are not aligned.
    """
    if not score_wide.index.equals(flag_wide.index) or list(score_wide.columns) != list(
        flag_wide.columns
    ):
        raise ValueError("score_wide and flag_wide must share index and columns.")

    scores = score_wide.stack(future_stack=True).rename("anomaly_score")
    flags = flag_wide.stack(future_stack=True).rename("anomaly_flag")
    merged = pd.concat([scores, flags], axis=1).reset_index()
    merged.columns = ["datetime", "pair", "anomaly_score", "anomaly_flag"]
    merged["anomaly_flag"] = merged["anomaly_flag"].fillna(False).astype(bool)
    merged["detector_name"] = detector_name
    return merged[STANDARD_COLUMNS]
