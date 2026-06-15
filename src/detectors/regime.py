"""Hidden Markov Model regime detection.

A three-state Gaussian HMM (Hamilton, 1989) classifies each bar into a
low-volatility, elevated-volatility or crisis regime. HMMs are used rather than
a static classifier because the transition matrix models regime persistence
explicitly. The crisis state is identified after fitting as the state with the
highest mean realised volatility, so the labelling is data driven rather than
assumed.

The model is fit in-sample on the supplied observation sequence. The backtest
engine invokes the detector per fold, which prevents cross-fold leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from config import settings
from config.logging_config import get_logger
from src.detectors import to_long_format

logger = get_logger()

# Feature columns used as HMM observations, in preference order.
_OBSERVATION_FEATURES = ["vol_close_21", "abs_return", "spread_proxy_hl", "vix_level"]


@dataclass
class RegimeResult:
    """Decoded regime information for one pair.

    Args:
        state: Ordered regime label per bar (0 calm, n-1 crisis).
        crisis_probability: Posterior probability of the crisis state per bar.
        transition_flag: True where the decoded state changed from the prior bar.
        transition_matrix: Estimated state transition matrix (ordered).
        index: The datetime index the arrays align to.
    """

    state: pd.Series
    crisis_probability: pd.Series
    transition_flag: pd.Series
    transition_matrix: np.ndarray
    index: pd.Index


def _build_observations(features: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Assemble the HMM observation matrix for one pair.

    Args:
        features: Multi-level (pair, feature) frame.
        pair: Pair symbol.

    Returns:
        A DataFrame of available observation features (NaN rows dropped).
    """
    block = features[pair]
    obs = pd.DataFrame(index=block.index)
    if "vol_close_21" in block.columns:
        obs["vol_close_21"] = block["vol_close_21"]
    if "log_return" in block.columns:
        obs["abs_return"] = block["log_return"].abs()
    if "spread_proxy_hl" in block.columns:
        obs["spread_proxy_hl"] = block["spread_proxy_hl"]
    if "vix_level" in block.columns:
        obs["vix_level"] = block["vix_level"]
    return obs.dropna()


def compute_regimes(features: pd.DataFrame, pair: str) -> RegimeResult:
    """Fit a Gaussian HMM for one pair and decode its regime sequence.

    Args:
        features: Multi-level (pair, feature) frame.
        pair: Pair symbol.

    Returns:
        A RegimeResult. If there are too few observations the result contains
        empty/NaN series and an empty transition matrix.

    Raises:
        KeyError: If the pair is absent.
    """
    if pair not in features.columns.get_level_values(0):
        raise KeyError(f"Pair {pair} not present in feature frame.")

    obs = _build_observations(features, pair)
    full_index = features.index
    n_regimes = settings.DETECTOR.hmm_n_regimes

    if len(obs) < settings.DETECTOR.hmm_min_observations or obs.shape[1] == 0:
        nan = pd.Series(np.nan, index=full_index)
        return RegimeResult(
            nan, nan, pd.Series(False, index=full_index), np.empty((0, 0)), full_index
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(obs.to_numpy())
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=200,
        random_state=0,
    )
    try:
        model.fit(scaled)
    except Exception as exc:
        logger.warning("HMM fit failed for {}: {}", pair, exc)
        nan = pd.Series(np.nan, index=full_index)
        return RegimeResult(
            nan, nan, pd.Series(False, index=full_index), np.empty((0, 0)), full_index
        )

    raw_states = model.predict(scaled)
    posteriors = model.predict_proba(scaled)

    # Order states by mean of the first observation feature (realised vol):
    # the highest-volatility state is the crisis regime.
    vol_means = model.means_[:, 0]
    order = np.argsort(vol_means)
    rank = {old: new for new, old in enumerate(order)}
    ordered_states = np.array([rank[s] for s in raw_states])
    crisis_state_raw = order[-1]

    ordered_transmat = model.transmat_[np.ix_(order, order)]
    logger.info(
        "HMM {} fitted: ordered transition matrix diag={}",
        pair,
        np.round(np.diag(ordered_transmat), 3).tolist(),
    )

    state_series = pd.Series(ordered_states, index=obs.index).reindex(full_index)
    crisis_prob = pd.Series(posteriors[:, crisis_state_raw], index=obs.index).reindex(full_index)
    transition = state_series.ne(state_series.shift(1)) & state_series.notna()

    return RegimeResult(state_series, crisis_prob, transition, ordered_transmat, full_index)


def regime_detector(features: pd.DataFrame) -> pd.DataFrame:
    """Regime detector producing standard long-format output.

    Anomaly score is the crisis-state probability; a bar is flagged when the
    crisis probability exceeds the configured threshold or the regime has just
    transitioned into the crisis state.

    Args:
        features: Multi-level (pair, feature) frame.

    Returns:
        Long-format detector output named "regime".
    """
    pairs = list(features.columns.get_level_values(0).unique())
    crisis_threshold = settings.DETECTOR.crisis_probability_threshold
    n_regimes = settings.DETECTOR.hmm_n_regimes

    score_wide = pd.DataFrame(0.0, index=features.index, columns=pairs)
    flag_wide = pd.DataFrame(False, index=features.index, columns=pairs)
    for pair in pairs:
        result = compute_regimes(features, pair)
        crisis_prob = result.crisis_probability.fillna(0.0)
        is_crisis = result.state == (n_regimes - 1)
        just_entered = is_crisis & result.transition_flag.fillna(False)
        score_wide[pair] = crisis_prob
        flag_wide[pair] = (crisis_prob > crisis_threshold) | just_entered
    return to_long_format(score_wide, flag_wide, "regime")
