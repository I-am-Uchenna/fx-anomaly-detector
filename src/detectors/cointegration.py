"""Cointegration-based anomaly detection for pair relationships.

Engle-Granger (1987) two-step monitoring estimates a long-run hedge ratio,
confirms cointegration with an ADF test on the residual, then flags large
spread z-scores as mean-reversion opportunities or relationship breaks. A
Johansen (1991) rank monitor watches a group of related pairs; a drop in the
cointegration rank signals a structural break across the group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from config import settings
from config.logging_config import get_logger
from src.detectors import normalize_unit_interval, to_long_format

logger = get_logger()

# Engle-Granger monitored relationships (dependent, independent).
ENGLE_GRANGER_PAIRS: list[tuple[str, str]] = [
    ("EURUSD=X", "GBPUSD=X"),
    ("AUDUSD=X", "NZDUSD=X"),
    ("EURGBP=X", "EURUSD=X"),
]

# Johansen rank-monitored group.
JOHANSEN_GROUP: list[str] = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X"]

_ADF_COINTEGRATION_PVALUE = 0.05


def engle_granger_monitor(
    dependent: pd.Series,
    independent: pd.Series,
    window: int,
    recalibration_days: int,
) -> tuple[pd.Series, pd.Series]:
    """Stepwise Engle-Granger spread z-score and cointegration indicator.

    The hedge ratio is re-estimated every recalibration_days on the trailing
    window and held constant between recalibrations. An ADF test on the window
    residuals determines whether the relationship is currently cointegrated.

    Args:
        dependent: Dependent price series.
        independent: Independent price series.
        window: Estimation window length.
        recalibration_days: Bars between hedge-ratio re-estimations.

    Returns:
        A tuple of (spread z-score series, boolean cointegrated series).
    """
    paired = pd.concat([dependent, independent], axis=1).dropna()
    paired.columns = ["y", "x"]
    n = len(paired)
    zscore = pd.Series(np.nan, index=paired.index)
    cointegrated = pd.Series(False, index=paired.index)
    if n <= window:
        return zscore.reindex(dependent.index), cointegrated.reindex(dependent.index)

    y = paired["y"].to_numpy()
    x = paired["x"].to_numpy()
    beta = alpha = None
    is_coint = False
    for t in range(window, n):
        if (t - window) % recalibration_days == 0 or beta is None:
            train_x = x[t - window : t]
            train_y = y[t - window : t]
            beta, alpha = np.polyfit(train_x, train_y, 1)
            resid_train = train_y - (alpha + beta * train_x)
            try:
                pvalue = adfuller(resid_train, maxlag=1, autolag=None)[1]
            except Exception:
                pvalue = 1.0
            is_coint = pvalue < _ADF_COINTEGRATION_PVALUE
        window_x = x[t - window : t + 1]
        window_y = y[t - window : t + 1]
        resid = window_y - (alpha + beta * window_x)
        std = resid.std(ddof=0)
        if std > 0:
            zscore.iloc[t] = (resid[-1] - resid.mean()) / std
        cointegrated.iloc[t] = is_coint

    return zscore.reindex(dependent.index), cointegrated.reindex(dependent.index)


def johansen_rank_series(prices: pd.DataFrame, window: int, recalibration_days: int) -> pd.Series:
    """Rolling Johansen cointegration rank for a group of price series.

    Args:
        prices: Frame of aligned price series (columns are pairs).
        window: Rolling window length.
        recalibration_days: Bars between Johansen re-estimations.

    Returns:
        Series of integer cointegration ranks (0..k) aligned to the input.
    """
    clean = prices.dropna()
    n = len(clean)
    rank = pd.Series(np.nan, index=clean.index)
    if n <= window or clean.shape[1] < 2:
        return rank.reindex(prices.index)

    current_rank = np.nan
    for t in range(window, n):
        if (t - window) % recalibration_days == 0 or np.isnan(current_rank):
            block = clean.iloc[t - window : t].to_numpy()
            try:
                result = coint_johansen(block, det_order=0, k_ar_diff=1)
                trace_stat = result.lr1
                crit_95 = result.cvt[:, 1]
                current_rank = int(np.sum(trace_stat > crit_95))
            except Exception:
                current_rank = np.nan
        rank.iloc[t] = current_rank
    return rank.reindex(prices.index)


def cointegration_detector(features: pd.DataFrame) -> pd.DataFrame:
    """Cointegration detector producing standard long-format output.

    Args:
        features: Multi-level (pair, feature) frame containing close prices.

    Returns:
        Long-format detector output named "cointegration".
    """
    pairs = list(features.columns.get_level_values(0).unique())
    window = settings.WINDOWS.extended
    recalib = settings.DETECTOR.cointegration_recalibration_days
    z_threshold = settings.DETECTOR.cointegration_zscore_threshold

    score_wide = pd.DataFrame(0.0, index=features.index, columns=pairs)
    flag_wide = pd.DataFrame(False, index=features.index, columns=pairs)

    for dependent, independent in ENGLE_GRANGER_PAIRS:
        if (dependent, "close") not in features.columns:
            continue
        if (independent, "close") not in features.columns:
            continue
        zscore, cointegrated = engle_granger_monitor(
            features[(dependent, "close")],
            features[(independent, "close")],
            window,
            recalib,
        )
        score_wide[dependent] = normalize_unit_interval(zscore.abs()).fillna(0.0)
        flag_wide[dependent] = (zscore.abs() > z_threshold) & cointegrated

    # Johansen rank drop flags every member of the monitored group.
    group_cols = [(p, "close") for p in JOHANSEN_GROUP if (p, "close") in features.columns]
    if len(group_cols) >= 2:
        prices = features[group_cols]
        prices.columns = [c[0] for c in group_cols]
        rank = johansen_rank_series(prices, window, recalib)
        rank_drop = rank.diff() < 0
        for pair in JOHANSEN_GROUP:
            if pair in flag_wide.columns:
                flag_wide[pair] = flag_wide[pair] | rank_drop.fillna(False)

    return to_long_format(score_wide, flag_wide, "cointegration")
