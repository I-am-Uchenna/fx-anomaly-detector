"""End-to-end integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.alerts.engine import Alert, generate_alerts
from src.detectors import ensemble
from src.features.builder import build_features


def test_fetch_ticker_parses_mocked_download(mocker) -> None:
    from src.data import fetcher

    index = pd.date_range("2020-01-01", periods=5, freq="D")
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["EURUSD=X"]]
    )
    data = np.tile(np.array([1.10, 1.12, 1.09, 1.11, 1.11, 0.0]), (5, 1))
    raw = pd.DataFrame(data, index=index, columns=columns)
    mocker.patch.object(fetcher.yf, "download", return_value=raw)

    frame = fetcher.fetch_ticker("EURUSD=X", start="2020-01-01", end=None, interval="1d")
    assert frame is not None
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert len(frame) == 5


def test_full_pipeline_raw_to_alerts(synthetic_raw: dict[str, pd.DataFrame]) -> None:
    pairs = ["EURUSD=X", "GBPUSD=X"]
    features = build_features(synthetic_raw, pair_symbols=pairs, persist=False, validate=False)

    ens = ensemble.detect(features, include_autoencoder=False)
    expected_cols = {"datetime", "pair", "ensemble_score", "n_flags", "ensemble_flag"}
    assert expected_cols.issubset(set(ens.columns))
    assert set(ens["pair"].unique()) == set(pairs)

    pair_returns = {p: features[(p, "log_return")] for p in pairs}
    alerts = generate_alerts(ens, pair_returns=pair_returns)
    assert isinstance(alerts, list)
    for alert in alerts:
        assert isinstance(alert, Alert)
        assert 0.0 <= alert.ensemble_score <= 1.0
        assert alert.severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_pipeline_output_index_alignment(synthetic_raw: dict[str, pd.DataFrame]) -> None:
    features = build_features(
        synthetic_raw, pair_symbols=["EURUSD=X", "GBPUSD=X"], persist=False, validate=False
    )
    ens = ensemble.detect(features, include_autoencoder=False)
    # Every ensemble datetime must exist in the feature index.
    assert set(ens["datetime"].unique()).issubset(set(features.index))


def test_feature_store_roundtrip(tmp_path) -> None:
    from src.data import feature_store

    index = pd.bdate_range("2020-01-01", periods=10)
    frame = pd.concat(
        {"EURUSD=X": pd.DataFrame({"f1": np.arange(10.0), "f2": np.arange(10.0)}, index=index)},
        axis=1,
    )
    frame.columns = frame.columns.set_names(["pair", "feature"])
    for _ in range(7):
        feature_store.save_features(frame, name="t", store_dir=tmp_path, keep_versions=3)
    versions = list(tmp_path.glob("t_*.parquet"))
    assert len(versions) == 3  # pruned to keep_versions
    loaded = feature_store.load_features(name="t", store_dir=tmp_path)
    assert loaded.shape == frame.shape


def test_notifier_dispatch_and_webhook(mocker) -> None:
    from src.alerts import notifier
    from src.alerts.engine import Alert

    alert = Alert(
        timestamp="2024-01-01T00:00:00",
        pair="EURUSD=X",
        ensemble_score=0.92,
        severity="CRITICAL",
        n_detectors_flagged=4,
        detector_scores={"zscore": 0.9},
        regime_state=2.0,
        var_99=0.01,
        suggested_action="Escalate.",
    )
    notifier.notify_console([alert])
    notifier.notify_log([alert])
    assert notifier.notify_webhook([alert], url="") == 0
    post = mocker.patch.object(notifier.requests, "post")
    post.return_value.raise_for_status.return_value = None
    assert notifier.notify_webhook([alert], url="http://example.com/hook") == 1


def test_fetcher_aligns_multiple_tickers(mocker) -> None:
    from src.data import fetcher

    index = pd.date_range("2020-01-01", periods=8, freq="D")

    def fake_download(symbol, **kwargs):
        base = 1.1 if "EUR" in symbol else 1.3
        data = {
            "Open": base,
            "High": base * 1.01,
            "Low": base * 0.99,
            "Close": base,
            "Adj Close": base,
            "Volume": 0.0,
        }
        return pd.DataFrame(data, index=index)

    mocker.patch.object(fetcher.yf, "download", side_effect=fake_download)
    out = fetcher.fetch_fx_data(symbols=["EURUSD=X", "GBPUSD=X"], include_macro=False, cache=False)
    assert set(out) == {"EURUSD=X", "GBPUSD=X"}
    assert out["EURUSD=X"].index.equals(out["GBPUSD=X"].index)
