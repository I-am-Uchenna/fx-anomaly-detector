"""FX Anomaly Radar - a fast, focused anomaly monitor for G10 majors.

Single-page Streamlit app. The engine is fully vectorised (no per-row apply,
no GARCH/HMM/autoencoder), so a cold start renders in seconds. Anomaly scores
blend a robust multivariate z-score with an Isolation Forest, both computed on
a tight, economically motivated feature set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import IsolationForest

PAIRS: dict[str, str] = {
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD",
    "USDCAD=X": "USD/CAD",
    "USDCHF=X": "USD/CHF",
}
ROLL = 63  # ~1 quarter, the lookback for rolling statistics

st.set_page_config(page_title="FX Anomaly Radar", page_icon="📡", layout="wide")

_CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1250px;}
#MainMenu, footer {visibility: hidden;}
.hero h1 {font-size: 2.1rem; margin-bottom: .1rem; font-weight: 700;}
.hero p {color: #9aa0aa; margin-top: 0; font-size: .95rem;}
div[data-testid="stMetric"] {
  background: #161B26; border: 1px solid #232a38; border-radius: 14px;
  padding: 16px 18px;
}
div[data-testid="stMetricLabel"] p {color:#9aa0aa; font-size:.78rem; letter-spacing:.04em;}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(years: int) -> dict[str, pd.DataFrame]:
    """Download recent daily OHLC for each pair from Yahoo Finance.

    Args:
        years: Number of years of history to fetch.

    Returns:
        Mapping of symbol to a tz-naive OHLC DataFrame.
    """
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    out: dict[str, pd.DataFrame] = {}
    for symbol in PAIRS:
        raw = yf.download(symbol, start=start, interval="1d", auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        frame = raw[["Open", "High", "Low", "Close"]].copy()
        frame.columns = ["open", "high", "low", "close"]
        frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
        out[symbol] = frame.dropna()
    return out


def _zscore(series: pd.Series, window: int = ROLL) -> pd.Series:
    """Trailing rolling z-score of a series."""
    roll = series.rolling(window)
    return (series - roll.mean()) / roll.std(ddof=0)


@st.cache_data(ttl=3600, show_spinner=False)
def build_scores(years: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compute anomaly scores for every pair (fully vectorised).

    Args:
        years: History length to load.

    Returns:
        A tuple of (scores wide-frame indexed by date with one column per pair,
        per-pair feature frames including close and the components).
    """
    prices = load_prices(years)
    returns = pd.DataFrame({s: np.log(df["close"]).diff() for s, df in prices.items()})
    basket = returns.mean(axis=1)

    feats: dict[str, pd.DataFrame] = {}
    score_cols: dict[str, pd.Series] = {}
    for symbol, df in prices.items():
        r = np.log(df["close"]).diff()
        vol = r.rolling(21).std(ddof=0)
        rng = (df["high"] - df["low"]) / df["close"]
        corr = r.rolling(ROLL).corr(basket)

        z = pd.DataFrame(
            {
                "return": _zscore(r),
                "volatility": _zscore(vol),
                "range": _zscore(rng),
                "correlation": _zscore(corr),
            }
        )
        # Robust multivariate deviation: RMS of the component z-scores.
        rms = np.sqrt((z**2).mean(axis=1))
        stat_score = 1.0 - np.exp(-rms / 2.5)

        block = z.copy()
        block["close"] = df["close"]
        block["stat_score"] = stat_score
        feats[symbol] = block
        score_cols[symbol] = stat_score

    scores = pd.DataFrame(score_cols)

    # Isolation Forest as a second opinion on the pooled, recent feature space.
    pooled = pd.concat(
        [feats[s][["return", "volatility", "range", "correlation"]].assign(pair=s) for s in feats]
    ).dropna()
    if len(pooled) > 50:
        x = pooled[["return", "volatility", "range", "correlation"]].to_numpy()
        model = IsolationForest(contamination=0.05, random_state=0, n_estimators=150).fit(x)
        iso = pd.Series(-model.decision_function(x), index=pooled.index)
        iso = (iso - iso.min()) / (iso.max() - iso.min() + 1e-9)
        pooled["iso"] = iso.values
        for symbol in feats:
            pair_iso = pooled[pooled["pair"] == symbol]["iso"]
            blended = 0.6 * scores[symbol] + 0.4 * pair_iso.reindex(scores.index)
            scores[symbol] = blended.fillna(scores[symbol])

    return scores, feats


SEVERITY = [(0.85, "Critical", "#ef4444"), (0.7, "High", "#f59e0b"), (0.55, "Elevated", "#eab308")]


def severity(score: float) -> tuple[str, str]:
    """Map a score to a (label, colour) severity band."""
    for threshold, label, colour in SEVERITY:
        if score >= threshold:
            return label, colour
    return "Normal", "#22c55e"


def main() -> None:
    """Render the single-page dashboard."""
    years = st.sidebar.slider("Years of history", 1, 5, 2)
    pair_label = st.sidebar.selectbox("Focus pair", list(PAIRS.values()), index=0)
    focus = {v: k for k, v in PAIRS.items()}[pair_label]

    with st.spinner("Scanning the FX market..."):
        scores, feats = build_scores(years)

    latest_date = scores.dropna(how="all").index.max()
    latest = scores.loc[latest_date]
    mean_stress = float(latest.mean())
    flagged = latest[latest >= 0.55]
    top_pair = latest.idxmax()
    top_label, top_colour = severity(float(latest.max()))

    st.markdown(
        "<div class='hero'><h1>📡 FX Anomaly Radar</h1>"
        "<p>Fast, multivariate anomaly detection across the G10 majors.</p></div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Live Yahoo Finance data · as of {latest_date.date()}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market stress", f"{mean_stress * 100:.0f}%")
    c2.metric("Pairs flagged", f"{len(flagged)} / {len(PAIRS)}")
    c3.metric("Most anomalous", PAIRS[top_pair], top_label)
    c4.metric("Top score", f"{float(latest.max()) * 100:.0f}%")

    left, right = st.columns([3, 2])

    with left:
        st.subheader(f"{pair_label} · price & anomalies")
        block = feats[focus]
        score_series = scores[focus]
        flags = score_series[score_series >= 0.7]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=block.index, y=block["close"], name="Close",
                line={"color": "#6C5CE7", "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=flags.index, y=block["close"].reindex(flags.index),
                mode="markers", name="Anomaly",
                marker={"color": "#ef4444", "size": 9, "line": {"color": "#fff", "width": 1}},
            )
        )
        fig.update_layout(
            template="plotly_dark", height=360, margin={"l": 0, "r": 0, "t": 10, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend={"orientation": "h", "y": 1.1},
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Anomaly score trend")
        recent = scores[[focus]].tail(120).rename(columns={focus: pair_label})
        trend = go.Figure()
        trend.add_trace(
            go.Scatter(
                x=recent.index, y=recent[pair_label], fill="tozeroy",
                line={"color": "#00D1B2", "width": 2}, name=pair_label,
            )
        )
        trend.add_hline(y=0.7, line={"color": "#f59e0b", "dash": "dot"})
        trend.update_layout(
            template="plotly_dark", height=360, margin={"l": 0, "r": 0, "t": 10, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis={"range": [0, 1]},
            showlegend=False,
        )
        st.plotly_chart(trend, use_container_width=True)

    st.subheader("Current standings")
    table = pd.DataFrame(
        {
            "Pair": [PAIRS[s] for s in scores.columns],
            "Score": [float(latest[s]) for s in scores.columns],
            "Status": [severity(float(latest[s]))[0] for s in scores.columns],
        }
    ).sort_values("Score", ascending=False)
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Anomaly score", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )


if __name__ == "__main__":
    main()
