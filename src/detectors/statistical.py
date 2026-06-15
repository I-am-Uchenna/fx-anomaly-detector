"""Classical statistical anomaly detectors (the interpretable baseline).

Three detectors with different strengths: a univariate rolling z-score, a
multivariate Mahalanobis distance with Ledoit-Wolf shrinkage (catches joint
outliers that no single feature flags), and a Grubbs test that frames the most
extreme observation as a formal hypothesis test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf

from config import settings
from config.logging_config import get_logger
from src.detectors import normalize_unit_interval, pair_feature_matrix, to_long_format

logger = get_logger()


def _rolling_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing z-score of each column relative to its rolling window.

    Args:
        frame: Feature matrix indexed by datetime.
        window: Rolling window length.

    Returns:
        DataFrame of z-scores with the same shape as the input.
    """
    roll = frame.rolling(window)
    return (frame - roll.mean()) / roll.std(ddof=0)


def zscore_detector(features: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Rolling z-score detector.

    Args:
        features: Multi-level (pair, feature) frame.
        window: Rolling window for the z-score. Defaults to the long window.

    Returns:
        Long-format detector output named "zscore".
    """
    window = window or settings.WINDOWS.long
    pairs = list(features.columns.get_level_values(0).unique())
    # The score is the max absolute z across many features, so a fixed per
    # feature threshold inflates the type-I rate through multiple comparisons.
    # Apply a Bonferroni correction: hold the family-wise per-bar tail
    # probability at the level implied by the configured single-feature
    # threshold.
    per_feature_alpha = 2.0 * stats.norm.sf(settings.DETECTOR.zscore_threshold)
    score_wide = pd.DataFrame(index=features.index)
    flag_wide = pd.DataFrame(index=features.index)
    for pair in pairs:
        block = pair_feature_matrix(features, pair)
        n_features = max(block.shape[1], 1)
        critical = float(stats.norm.isf(per_feature_alpha / (2.0 * n_features)))
        zed = _rolling_zscore(block, window).abs()
        max_abs = zed.max(axis=1)
        score_wide[pair] = normalize_unit_interval(max_abs)
        flag_wide[pair] = max_abs > critical
    return to_long_format(score_wide, flag_wide.fillna(False), "zscore")


def _mahalanobis_series(block: pd.DataFrame, window: int, refit_step: int) -> tuple[pd.Series, int]:
    """Rolling Mahalanobis distance using periodically refit shrunk covariance.

    The covariance and centroid are estimated on the trailing window and reused
    for refit_step bars to keep the computation tractable.

    Args:
        block: Single-pair feature matrix.
        window: Trailing window used to estimate covariance and centroid.
        refit_step: Number of bars between covariance refits.

    Returns:
        A tuple of (distance series aligned to the input index, feature
        dimensionality used for the chi-squared threshold).
    """
    clean = block.dropna(axis=1, how="any")
    n, dim = clean.shape
    out = np.full(n, np.nan)
    if dim == 0 or n <= window:
        return pd.Series(out, index=block.index), max(dim, 1)

    values = clean.to_numpy()
    inv_cov = None
    centroid = None
    for t in range(window, n):
        if (t - window) % refit_step == 0 or inv_cov is None:
            train = values[t - window : t]
            estimator = LedoitWolf().fit(train)
            centroid = estimator.location_
            inv_cov = np.linalg.pinv(estimator.covariance_)
        delta = values[t] - centroid
        out[t] = float(np.sqrt(delta @ inv_cov @ delta))
    return pd.Series(out, index=block.index), dim


def mahalanobis_detector(
    features: pd.DataFrame,
    window: int | None = None,
    refit_step: int | None = None,
) -> pd.DataFrame:
    """Mahalanobis distance detector with a chi-squared flag threshold.

    Args:
        features: Multi-level (pair, feature) frame.
        window: Covariance estimation window. Defaults to the extended window.
        refit_step: Bars between covariance refits. Defaults to the backtest
            test window length.

    Returns:
        Long-format detector output named "mahalanobis".
    """
    window = window or settings.WINDOWS.extended
    refit_step = refit_step or settings.BACKTEST.walk_forward_test_days
    pairs = list(features.columns.get_level_values(0).unique())
    score_wide = pd.DataFrame(index=features.index)
    flag_wide = pd.DataFrame(index=features.index)
    for pair in pairs:
        block = pair_feature_matrix(features, pair)
        distance, dim = _mahalanobis_series(block, window, refit_step)
        # Mahalanobis distance squared is chi-squared with dim dof under
        # multivariate normality; flag the upper 1% tail.
        critical = float(np.sqrt(stats.chi2.ppf(0.99, df=max(dim, 1))))
        score_wide[pair] = normalize_unit_interval(distance)
        flag_wide[pair] = distance > critical
    return to_long_format(score_wide, flag_wide.fillna(False), "mahalanobis")


def _grubbs_pvalue_last(arr: np.ndarray) -> float:
    """Two-sided Grubbs test p-value for the most recent observation.

    Args:
        arr: Window of observations; the test targets the last element.

    Returns:
        The p-value that the last observation is not an outlier, or NaN if the
        window is degenerate.
    """
    valid = arr[~np.isnan(arr)]
    n = valid.size
    if n < 7:
        return float("nan")
    std = valid.std(ddof=1)
    if std == 0:
        return float("nan")
    g = abs(valid[-1] - valid.mean()) / std
    # Convert the Grubbs statistic to a t value and then a two-sided p-value.
    denom = (n - 1) ** 2 - n * g**2
    if denom <= 0:
        return 0.0
    t_sq = n * (n - 2) * g**2 / denom
    t_val = np.sqrt(max(t_sq, 0.0))
    p_single = stats.t.sf(t_val, df=n - 2)
    return float(min(1.0, n * p_single))


def grubbs_detector(features: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Grubbs outlier-test detector.

    Args:
        features: Multi-level (pair, feature) frame.
        window: Rolling window for the test. Defaults to the long window.

    Returns:
        Long-format detector output named "grubbs".
    """
    window = window or settings.WINDOWS.long
    pairs = list(features.columns.get_level_values(0).unique())
    score_wide = pd.DataFrame(index=features.index)
    flag_wide = pd.DataFrame(index=features.index)
    for pair in pairs:
        block = pair_feature_matrix(features, pair)
        pval_min = pd.Series(1.0, index=block.index)
        for col in block.columns:
            pvals = block[col].rolling(window).apply(_grubbs_pvalue_last, raw=True)
            pval_min = np.minimum(pval_min, pvals.fillna(1.0))
        score_wide[pair] = normalize_unit_interval(1.0 - pval_min)
        flag_wide[pair] = pval_min < 0.01
    return to_long_format(score_wide, flag_wide.fillna(False), "grubbs")
