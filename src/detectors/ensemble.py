"""Ensemble combination of all anomaly detectors.

Individual detector scores are combined by a weighted average. Weights for
detectors that produced no score at a given bar are dropped and the remainder
renormalised, so a missing detector (for example the autoencoder when
TensorFlow is absent) does not bias the result. A bar is flagged only when the
ensemble score clears the threshold and at least two individual detectors agree,
and the flag must persist for a minimum number of consecutive bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from config.logging_config import get_logger
from src.detectors import statistical
from src.detectors.autoencoder import autoencoder_detector
from src.detectors.cointegration import cointegration_detector
from src.detectors.ml_unsupervised import isolation_forest_detector, lof_detector
from src.detectors.regime import regime_detector

logger = get_logger()


def run_all_detectors(
    features: pd.DataFrame, include_autoencoder: bool = True
) -> dict[str, pd.DataFrame]:
    """Run every detector and collect their long-format outputs.

    Args:
        features: Multi-level (pair, feature) frame.
        include_autoencoder: If False, the autoencoder detector is skipped
            (useful when TensorFlow is unavailable or for fast runs).

    Returns:
        Mapping of detector_name to its long-format output.
    """
    outputs: dict[str, pd.DataFrame] = {
        "zscore": statistical.zscore_detector(features),
        "mahalanobis": statistical.mahalanobis_detector(features),
        "grubbs": statistical.grubbs_detector(features),
        "isolation_forest": isolation_forest_detector(features),
        "lof": lof_detector(features),
        "regime": regime_detector(features),
        "cointegration": cointegration_detector(features),
    }
    if include_autoencoder:
        outputs["autoencoder"] = autoencoder_detector(features)
    return outputs


def _apply_persistence(flag: pd.Series, persistence: int) -> pd.Series:
    """Require a flag to hold for a number of consecutive bars.

    Args:
        flag: Boolean series ordered in time (single pair).
        persistence: Minimum consecutive True bars to confirm an anomaly.

    Returns:
        Confirmed boolean series.
    """
    if persistence <= 1:
        return flag
    rolling_sum = flag.astype(int).rolling(persistence).sum()
    return rolling_sum >= persistence


def build_ensemble(
    detector_outputs: dict[str, pd.DataFrame],
    include_individual_scores: bool = True,
) -> pd.DataFrame:
    """Combine detector outputs into an ensemble score and flag.

    Args:
        detector_outputs: Mapping of detector_name to long-format output.
        include_individual_scores: If True, retain per-detector score columns
            in the result (used by the alert engine).

    Returns:
        A long DataFrame indexed implicitly by (datetime, pair) with columns:
        datetime, pair, ensemble_score, n_flags, ensemble_flag, and optionally
        score_<detector> columns.

    Raises:
        ValueError: If detector_outputs is empty.
    """
    if not detector_outputs:
        raise ValueError("detector_outputs is empty.")

    weight_map = settings.ENSEMBLE.weight_map()
    score_frames = []
    flag_frames = []
    for name, df in detector_outputs.items():
        indexed = df.set_index(["datetime", "pair"])
        score_frames.append(indexed["anomaly_score"].rename(f"score_{name}"))
        flag_frames.append(indexed["anomaly_flag"].rename(f"flag_{name}"))

    scores = pd.concat(score_frames, axis=1)
    flags = pd.concat(flag_frames, axis=1).fillna(False)

    weighted_sum = pd.Series(0.0, index=scores.index)
    weight_total = pd.Series(0.0, index=scores.index)
    for name in detector_outputs:
        col = scores[f"score_{name}"]
        weight = weight_map.get(name, 0.0)
        present = col.notna()
        weighted_sum = weighted_sum.add((col.fillna(0.0) * weight).where(present, 0.0))
        weight_total = weight_total.add(np.where(present, weight, 0.0))

    ensemble_score = weighted_sum / weight_total.replace(0.0, np.nan)
    n_flags = flags.sum(axis=1)

    raw_flag = (ensemble_score > settings.ENSEMBLE.score_threshold) & (
        n_flags >= settings.ENSEMBLE.min_detectors_flagged
    )

    result = pd.DataFrame(
        {
            "ensemble_score": ensemble_score,
            "n_flags": n_flags.astype(int),
            "raw_flag": raw_flag.fillna(False),
        }
    )
    if include_individual_scores:
        result = result.join(scores)

    result = result.reset_index()
    # Apply the persistence filter per pair.
    persistence = settings.DETECTOR.min_anomaly_persistence
    confirmed = []
    for _pair, group in result.sort_values("datetime").groupby("pair"):
        group = group.copy()
        group["ensemble_flag"] = _apply_persistence(group["raw_flag"], persistence).values
        confirmed.append(group)
    result = pd.concat(confirmed).sort_values(["datetime", "pair"]).reset_index(drop=True)
    result = result.drop(columns=["raw_flag"])
    return result


def detect(features: pd.DataFrame, include_autoencoder: bool = True) -> pd.DataFrame:
    """Convenience wrapper: run all detectors and build the ensemble.

    Args:
        features: Multi-level (pair, feature) frame.
        include_autoencoder: Whether to include the autoencoder detector.

    Returns:
        The ensemble long-format DataFrame from build_ensemble.
    """
    outputs = run_all_detectors(features, include_autoencoder=include_autoencoder)
    return build_ensemble(outputs)
