"""Streamlit dashboard for the FX anomaly detector.

Run from the project root:
    streamlit run src/dashboard/app.py

Four pages: a live monitor, a per-pair deep dive, regime analysis, and backtest
results. Expensive pipeline steps are cached so navigation stays responsive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import settings  # noqa: E402
from src.backtest.engine import run_backtest  # noqa: E402
from src.backtest.metrics import compute_metrics  # noqa: E402
from src.data.fetcher import fetch_fx_data  # noqa: E402
from src.detectors import ensemble  # noqa: E402
from src.detectors.regime import compute_regimes  # noqa: E402
from src.features.builder import build_features  # noqa: E402

_TRAFFIC = {0: "🟢", 1: "🟡", 2: "🔴"}


@st.cache_data(show_spinner="Fetching live data and building features...")
def load_features(start: str) -> pd.DataFrame:
    """Fetch live data and build the feature frame (cached).

    Args:
        start: Start date for the data download.

    Returns:
        The feature DataFrame.
    """
    raw = fetch_fx_data(start=start)
    return build_features(raw, persist=False, validate=False)


@st.cache_data(show_spinner="Running detectors...")
def load_detections(start: str, include_autoencoder: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the ensemble and return (features, ensemble_long) (cached).

    Args:
        start: Start date for the data download.
        include_autoencoder: Whether to include the autoencoder.

    Returns:
        A tuple of the feature frame and the ensemble long DataFrame.
    """
    features = load_features(start)
    ens = ensemble.detect(features, include_autoencoder=include_autoencoder)
    return features, ens


def _page_live_monitor(features: pd.DataFrame, ens: pd.DataFrame) -> None:
    """Render the live monitor page."""
    st.header("Live monitor")
    latest_date = ens["datetime"].max()
    st.caption(f"Last update: {latest_date}")

    latest = ens[ens["datetime"] == latest_date]
    score_pivot = latest.pivot_table(index="pair", values="ensemble_score")
    fig = go.Figure(
        go.Heatmap(
            z=[score_pivot["ensemble_score"].values],
            x=score_pivot.index,
            y=["score"],
            colorscale="OrRd",
            zmin=0,
            zmax=1,
        )
    )
    fig.update_layout(title="Current ensemble anomaly scores")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Active anomalies")
    active = latest[latest["ensemble_flag"]][["pair", "ensemble_score", "n_flags"]]
    st.dataframe(active, use_container_width=True)

    st.subheader("Regime indicators")
    cols = st.columns(len(settings.FX_PAIRS))
    for col, pair_cfg in zip(cols, settings.FX_PAIRS, strict=False):
        if pair_cfg.symbol in features.columns.get_level_values(0):
            state = compute_regimes(features, pair_cfg.symbol).state.dropna()
            light = _TRAFFIC.get(int(state.iloc[-1]), "⚪") if not state.empty else "⚪"
            col.metric(pair_cfg.name, light)


def _page_pair_deep_dive(features: pd.DataFrame, ens: pd.DataFrame) -> None:
    """Render the per-pair deep-dive page."""
    st.header("Pair deep dive")
    pairs = list(features.columns.get_level_values(0).unique())
    pair = st.selectbox("Pair", pairs)

    close = features[(pair, "close")]
    pair_ens = ens[ens["pair"] == pair].set_index("datetime")
    flagged = pair_ens[pair_ens["ensemble_flag"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, name="Close"))
    fig.add_trace(
        go.Scatter(
            x=flagged.index,
            y=close.reindex(flagged.index).values,
            mode="markers",
            marker={"color": "red", "size": 8},
            name="Anomaly",
        )
    )
    fig.update_layout(title=f"{pair} price with anomaly overlays")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature time series")
    feature_names = [c for c in features[pair].columns if c not in {"open", "high", "low", "close"}]
    chosen = st.selectbox("Feature", feature_names)
    st.line_chart(features[(pair, chosen)])

    st.subheader("Detector score breakdown")
    score_cols = [c for c in pair_ens.columns if c.startswith("score_")]
    st.line_chart(pair_ens[score_cols])


def _page_regime_analysis(features: pd.DataFrame) -> None:
    """Render the regime-analysis page."""
    st.header("Regime analysis")
    pairs = list(features.columns.get_level_values(0).unique())
    pair = st.selectbox("Pair", pairs, key="regime_pair")
    result = compute_regimes(features, pair)

    st.subheader("Regime state timeline")
    st.line_chart(result.state)

    if result.transition_matrix.size > 0:
        st.subheader("Transition probability matrix")
        fig = go.Figure(go.Heatmap(z=result.transition_matrix, colorscale="Blues"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Regime-conditional returns")
        rets = features[(pair, "log_return")]
        frame = pd.DataFrame({"state": result.state, "ret": rets}).dropna()
        violin = go.Figure()
        for state in sorted(frame["state"].unique()):
            violin.add_trace(
                go.Violin(y=frame[frame["state"] == state]["ret"], name=f"State {int(state)}")
            )
        st.plotly_chart(violin, use_container_width=True)

        current = result.state.dropna()
        if not current.empty:
            st.metric("Current regime", f"State {int(current.iloc[-1])}")


def _page_backtest(features: pd.DataFrame) -> None:
    """Render the backtest-results page."""
    st.header("Backtest results")
    if st.button("Run backtest"):
        with st.spinner("Running walk-forward backtest..."):
            result = run_backtest(features, include_autoencoder=False)
            metrics = compute_metrics(result.strategy_returns, result.equity_curve, result.trades)

        st.subheader("Summary metrics")
        st.dataframe(pd.Series(metrics.as_dict()).to_frame("value"), use_container_width=True)

        st.subheader("Equity curve")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=result.equity_curve.index, y=result.equity_curve.values, name="Strategy")
        )
        fig.add_trace(
            go.Scatter(
                x=result.benchmark_equity.index, y=result.benchmark_equity.values, name="Benchmark"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Per-pair performance")
        if not result.trades.empty:
            grouped = result.trades.groupby("pair")["trade_return"].agg(["size", "sum", "mean"])
            st.dataframe(grouped, use_container_width=True)


def main() -> None:
    """Dashboard entry point."""
    st.set_page_config(page_title="FX Anomaly Detector", layout="wide")
    st.title("FX Anomaly Detector")

    start = st.sidebar.text_input("Data start date", settings.DATA_START_DATE)
    include_ae = st.sidebar.checkbox("Include autoencoder", value=False)
    page = st.sidebar.radio(
        "Page", ["Live monitor", "Pair deep dive", "Regime analysis", "Backtest results"]
    )

    features, ens = load_detections(start, include_ae)

    if page == "Live monitor":
        _page_live_monitor(features, ens)
    elif page == "Pair deep dive":
        _page_pair_deep_dive(features, ens)
    elif page == "Regime analysis":
        _page_regime_analysis(features)
    elif page == "Backtest results":
        _page_backtest(features)


if __name__ == "__main__":
    main()
