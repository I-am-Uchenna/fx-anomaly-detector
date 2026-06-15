"""Unsupervised machine-learning anomaly detectors.

Isolation Forest (Liu, Ting and Zhou, 2008) isolates anomalies with few random
splits and scales well; Local Outlier Factor flags points sitting in unusually
sparse neighbourhoods. Both pool every pair's feature vector into a single
training set and are refit periodically on a trailing window so the models do
not go stale. DBSCAN is intentionally omitted: it adds a third density method
with no clear marginal value over LOF for this feature space.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from config import settings
from config.logging_config import get_logger
from src.detectors import normalize_unit_interval, pair_feature_matrix, to_long_format

logger = get_logger()

# A fitted model that exposes score_samples-like behaviour and a predict.
_ScoreFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def _pooled_matrices(
    features: pd.DataFrame,
) -> tuple[list[str], list[str], dict[str, pd.DataFrame]]:
    """Build per-pair feature matrices restricted to common columns.

    Args:
        features: Multi-level (pair, feature) frame.

    Returns:
        A tuple of (pairs, common_columns, per_pair_matrices).
    """
    pairs = list(features.columns.get_level_values(0).unique())
    blocks = {p: pair_feature_matrix(features, p) for p in pairs}
    common = set.intersection(*(set(b.columns) for b in blocks.values()))
    common_cols = sorted(common)
    blocks = {p: b[common_cols] for p, b in blocks.items()}
    return pairs, common_cols, blocks


def _blockwise_detect(
    features: pd.DataFrame,
    fit_score: _ScoreFn,
    detector_name: str,
    window: int,
    refit_step: int,
) -> pd.DataFrame:
    """Run a refit-on-a-schedule outlier model pooled across pairs.

    Args:
        features: Multi-level (pair, feature) frame.
        fit_score: Callable that, given (train, test) standardised arrays,
            returns (anomaly_scores, outlier_flags) for the test rows.
        detector_name: Name for the output and logs.
        window: Trailing training window length in bars.
        refit_step: Bars between refits.

    Returns:
        Long-format detector output.
    """
    pairs, common_cols, blocks = _pooled_matrices(features)
    index = features.index
    n = len(index)
    score_wide = pd.DataFrame(np.nan, index=index, columns=pairs)
    flag_wide = pd.DataFrame(False, index=index, columns=pairs)

    if not common_cols or n <= window:
        return to_long_format(score_wide, flag_wide, detector_name)

    arrays = {p: blocks[p].to_numpy() for p in pairs}
    for start in range(window, n, refit_step):
        train_rows = [arrays[p][start - window : start] for p in pairs]
        train = np.vstack(train_rows)
        medians = np.nanmedian(train, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        train = np.where(np.isnan(train), medians, train)
        scaler = StandardScaler().fit(train)
        train_scaled = scaler.transform(train)

        end = min(start + refit_step, n)
        test_raw = np.vstack([arrays[p][start:end] for p in pairs])
        test_raw = np.where(np.isnan(test_raw), medians, test_raw)
        test_scaled = scaler.transform(test_raw)

        try:
            scores, flags = fit_score(train_scaled, test_scaled)
        except Exception as exc:  # degenerate training block
            logger.warning("{} fit failed in block at {}: {}", detector_name, start, exc)
            continue

        block_len = end - start
        n_pairs = len(pairs)
        scores = scores.reshape(n_pairs, block_len)
        flags = flags.reshape(n_pairs, block_len)
        for i, pair in enumerate(pairs):
            score_wide.iloc[start:end, score_wide.columns.get_loc(pair)] = scores[i]
            flag_wide.iloc[start:end, flag_wide.columns.get_loc(pair)] = flags[i]

    for pair in pairs:
        score_wide[pair] = normalize_unit_interval(score_wide[pair])
    return to_long_format(score_wide, flag_wide, detector_name)


def isolation_forest_detector(
    features: pd.DataFrame,
    window: int | None = None,
    refit_step: int | None = None,
) -> pd.DataFrame:
    """Isolation Forest detector.

    Args:
        features: Multi-level (pair, feature) frame.
        window: Trailing training window. Defaults to the extended window.
        refit_step: Bars between refits. Defaults to the backtest test window.

    Returns:
        Long-format detector output named "isolation_forest".
    """
    window = window or settings.WINDOWS.extended
    refit_step = refit_step or settings.BACKTEST.walk_forward_test_days

    def fit_score(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model = IsolationForest(
            contamination=settings.DETECTOR.isolation_forest_contamination,
            random_state=0,
            n_estimators=200,
        ).fit(train)
        # decision_function is positive for inliers; negate so higher = anomaly.
        scores = -model.decision_function(test)
        flags = model.predict(test) == -1
        return scores, flags

    return _blockwise_detect(features, fit_score, "isolation_forest", window, refit_step)


def lof_detector(
    features: pd.DataFrame,
    window: int | None = None,
    refit_step: int | None = None,
    n_neighbors: int | None = None,
) -> pd.DataFrame:
    """Local Outlier Factor detector (novelty mode).

    Args:
        features: Multi-level (pair, feature) frame.
        window: Trailing training window. Defaults to the extended window.
        refit_step: Bars between refits. Defaults to the backtest test window.
        n_neighbors: Neighbourhood size. Defaults to the configured value.

    Returns:
        Long-format detector output named "lof".
    """
    window = window or settings.WINDOWS.extended
    refit_step = refit_step or settings.BACKTEST.walk_forward_test_days
    k = n_neighbors or settings.DETECTOR.lof_n_neighbors

    def fit_score(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model = LocalOutlierFactor(n_neighbors=min(k, len(train) - 1), novelty=True).fit(train)
        scores = -model.decision_function(test)
        flags = model.predict(test) == -1
        return scores, flags

    return _blockwise_detect(features, fit_score, "lof", window, refit_step)
