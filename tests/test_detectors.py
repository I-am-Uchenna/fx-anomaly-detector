"""Tests for the anomaly detectors and the ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors import ensemble, statistical, to_long_format
from src.detectors.ensemble import _apply_persistence, build_ensemble


def _spike_date(feature_frame: pd.DataFrame) -> pd.Timestamp:
    """Return the feature-frame date closest to the injected return spike."""
    from tests.conftest import SPIKE_INDEX

    full_index = pd.bdate_range("2014-01-01", periods=900)
    return full_index[SPIKE_INDEX]


def test_zscore_detector_flags_injected_spike(feature_frame: pd.DataFrame) -> None:
    out = statistical.zscore_detector(feature_frame)
    spike = _spike_date(feature_frame)
    nearby = out[
        (out["datetime"] >= spike - pd.Timedelta(days=4))
        & (out["datetime"] <= spike + pd.Timedelta(days=4))
    ]
    assert nearby["anomaly_flag"].any()


def test_ensemble_flags_near_spike(feature_frame: pd.DataFrame) -> None:
    ens = ensemble.detect(feature_frame, include_autoencoder=False)
    spike = _spike_date(feature_frame)
    nearby = ens[
        (ens["datetime"] >= spike - pd.Timedelta(days=5))
        & (ens["datetime"] <= spike + pd.Timedelta(days=7))
    ]
    assert nearby["ensemble_flag"].any()


def test_false_positive_rate_low_on_clean_data(clean_feature_frame: pd.DataFrame) -> None:
    ens = ensemble.detect(clean_feature_frame, include_autoencoder=False)
    false_positive_rate = float(ens["ensemble_flag"].mean())
    assert false_positive_rate < 0.05


def test_detector_scores_in_unit_interval(feature_frame: pd.DataFrame) -> None:
    out = statistical.zscore_detector(feature_frame)
    scores = out["anomaly_score"].dropna()
    assert scores.between(0.0, 1.0).all()


def test_persistence_filter_drops_single_bar_spike() -> None:
    flag = pd.Series([False, True, False, True, True, False])
    confirmed = _apply_persistence(flag, persistence=2)
    # The isolated True at index 1 must be dropped; the 3-4 run survives at 4.
    assert not confirmed.iloc[1]
    assert confirmed.iloc[4]


def test_ensemble_voting_requires_two_detectors() -> None:
    index = pd.bdate_range("2020-01-01", periods=3)
    pairs = ["EURUSD=X"]

    def make(name: str, scores: list[float], flags: list[bool]) -> pd.DataFrame:
        score_wide = pd.DataFrame({pairs[0]: scores}, index=index)
        flag_wide = pd.DataFrame({pairs[0]: flags}, index=index)
        return to_long_format(score_wide, flag_wide, name)

    # Bar 0: one detector flags -> below the 2-detector minimum.
    # Bar 1 and 2: two detectors flag and score is high -> eligible.
    outputs = {
        "zscore": make("zscore", [0.9, 0.9, 0.9], [True, True, True]),
        "mahalanobis": make("mahalanobis", [0.1, 0.9, 0.9], [False, True, True]),
    }
    result = build_ensemble(outputs).set_index("datetime")
    # With persistence=2, only the sustained two-detector run is confirmed.
    assert not result["ensemble_flag"].iloc[0]
    assert result["ensemble_flag"].iloc[2]


def test_ensemble_handles_missing_detector_weights() -> None:
    index = pd.bdate_range("2020-01-01", periods=2)
    score_wide = pd.DataFrame({"EURUSD=X": [0.6, 0.7]}, index=index)
    flag_wide = pd.DataFrame({"EURUSD=X": [True, True]}, index=index)
    outputs = {"zscore": to_long_format(score_wide, flag_wide, "zscore")}
    result = build_ensemble(outputs)
    # Single detector present: ensemble score equals that detector's score.
    assert np.allclose(result["ensemble_score"].values, [0.6, 0.7])


def test_autoencoder_noop_when_tf_absent(feature_frame: pd.DataFrame, mocker) -> None:
    from src.detectors import autoencoder

    mocker.patch.object(autoencoder, "_import_tf", return_value=None)
    out = autoencoder.autoencoder_detector(feature_frame)
    assert (out["anomaly_score"] == 0.0).all()
    assert not out["anomaly_flag"].any()
    assert set(out["pair"].unique()) == set(feature_frame.columns.get_level_values(0).unique())
