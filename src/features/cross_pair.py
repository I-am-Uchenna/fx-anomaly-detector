"""Cross-pair features that detect structural breaks in FX relationships.

Includes triangular no-arbitrage residuals, rolling correlation against the FX
cross-section, the share of variance explained by the first principal component
(a risk-on/risk-off concentration gauge), cross-pair betas against synthetic
constituent rates, and an Engle-Granger style cointegration residual that is
shared with the cointegration detector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings

# Pairs treated as cointegrated for spread monitoring (dependent, independent).
COINTEGRATION_PAIRS: list[tuple[str, str]] = [
    ("EURUSD=X", "GBPUSD=X"),
    ("AUDUSD=X", "NZDUSD=X"),
]


def parse_symbol(symbol: str) -> tuple[str, str]:
    """Split a Yahoo FX ticker into base and quote currency codes.

    Args:
        symbol: Ticker like "EURUSD=X".

    Returns:
        A (base, quote) tuple, e.g. ("EUR", "USD").

    Raises:
        ValueError: If the symbol is not a six-letter FX ticker.
    """
    core = symbol.replace("=X", "")
    if len(core) != 6:
        raise ValueError(f"Cannot parse FX symbol: {symbol}")
    return core[:3], core[3:]


def implied_cross_return(leg_a: pd.Series, sym_a: str, leg_b: pd.Series, sym_b: str) -> pd.Series:
    """Compute the implied cross log return from two constituent legs.

    Log returns add when chaining or dividing rates, so the implied return is a
    signed sum of the two leg returns depending on the currency that cancels.

    Args:
        leg_a: Log returns of the first leg.
        sym_a: Symbol of the first leg.
        leg_b: Log returns of the second leg.
        sym_b: Symbol of the second leg.

    Returns:
        The implied cross log return series.
    """
    _, quote_a = parse_symbol(sym_a)
    base_b, quote_b = parse_symbol(sym_b)
    if quote_a == base_b:
        # base_a/quote_a * base_b/quote_b -> base_a/quote_b (chain).
        return leg_a + leg_b
    if quote_a == quote_b:
        # (base_a/quote) / (base_b/quote) -> base_a/base_b (division).
        return leg_a - leg_b
    # Fallback: chain.
    return leg_a + leg_b


def _triangular_residuals(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute triangular arbitrage residuals attached to each cross leg.

    Args:
        master: Multi-level (pair, feature) frame with close prices.

    Returns:
        Mapping of cross symbol to a frame with triangular_arb_residual and its
        z-score.
    """
    out: dict[str, pd.DataFrame] = {}
    window = settings.WINDOWS.medium
    for sym_a, sym_b, sym_c in settings.TRIANGULAR_TRIPLETS:
        cols = {(s, "close") for s in (sym_a, sym_b, sym_c)}
        if not cols.issubset(set(master.columns)):
            continue
        price_a = master[(sym_a, "close")]
        price_b = master[(sym_b, "close")]
        price_c = master[(sym_c, "close")]

        _, quote_a = parse_symbol(sym_a)
        base_b, quote_b = parse_symbol(sym_b)
        implied = price_a * price_b if quote_a == base_b else price_a / price_b
        residual = price_c - implied
        roll = residual.rolling(window)
        frame = pd.DataFrame(index=master.index)
        frame["triangular_arb_residual"] = residual
        frame["triangular_arb_zscore"] = (residual - roll.mean()) / roll.std(ddof=0)
        out[sym_c] = frame
    return out


def _rolling_pca_first_ratio(returns_matrix: pd.DataFrame, window: int) -> pd.Series:
    """Rolling share of variance explained by the first principal component.

    Args:
        returns_matrix: Frame of pair log returns (columns are pairs).
        window: Rolling window length.

    Returns:
        Series of first-component explained-variance ratios in [0, 1].
    """
    values = returns_matrix.to_numpy()
    n = len(returns_matrix)
    out = np.full(n, np.nan)
    for end in range(window, n + 1):
        block = values[end - window : end]
        if np.isnan(block).any():
            continue
        cov = np.cov(block, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)
        total = eigenvalues.sum()
        if total > 0:
            out[end - 1] = float(eigenvalues[-1] / total)
    return pd.Series(out, index=returns_matrix.index)


def rolling_ols_residual_zscore(
    dependent: pd.Series, independent: pd.Series, window: int
) -> pd.Series:
    """Z-score of the residual from a rolling OLS of dependent on independent.

    Shared with the cointegration detector. Beta and intercept are estimated on
    the trailing window; the residual z-score measures deviation from the
    fitted long-run relationship.

    Args:
        dependent: Dependent price series.
        independent: Independent price series.
        window: Rolling estimation window.

    Returns:
        Residual z-score series; NaN where the window is incomplete.
    """
    mean_x = independent.rolling(window).mean()
    mean_y = dependent.rolling(window).mean()
    cov = independent.rolling(window).cov(dependent)
    var_x = independent.rolling(window).var(ddof=0)
    beta = cov / var_x.replace(0.0, np.nan)
    alpha = mean_y - beta * mean_x
    residual = dependent - (alpha + beta * independent)
    roll_res = residual.rolling(window)
    return (residual - roll_res.mean()) / roll_res.std(ddof=0)


def compute(master: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Compute cross-pair features for every pair in the master frame.

    Market-wide signals (PCA concentration) are broadcast to every pair so the
    per-pair feature vector carries the global context.

    Args:
        master: Multi-level (pair, feature) frame with close and log_return.
        windows: Rolling windows. Defaults to short/medium/long.

    Returns:
        A multi-level (pair, feature) DataFrame of cross-pair features.
    """
    if windows is None:
        windows = settings.WINDOWS.as_list()

    pairs = list(master.columns.get_level_values(0).unique())
    fx_pairs = [p for p in pairs if p in {pc.symbol for pc in settings.FX_PAIRS}]

    returns_matrix = pd.concat({p: master[(p, "log_return")] for p in fx_pairs}, axis=1)
    market_return = returns_matrix.mean(axis=1)
    pca_ratio = _rolling_pca_first_ratio(returns_matrix, settings.WINDOWS.long)

    triangular = _triangular_residuals(master)

    blocks: dict[str, pd.DataFrame] = {}
    for pair in fx_pairs:
        ret = master[(pair, "log_return")]
        feats = pd.DataFrame(index=master.index)
        feats["pca_explained_ratio"] = pca_ratio
        for w in windows:
            corr = ret.rolling(w).corr(market_return)
            feats[f"pair_correlation_{w}"] = corr
            ext = corr.rolling(settings.WINDOWS.extended)
            feats[f"correlation_zscore_{w}"] = (corr - ext.mean()) / ext.std(ddof=0)

        # Cross-pair beta against a synthetic constituent return.
        base, quote = parse_symbol(pair)
        if base != "USD" and quote != "USD":
            leg_a_sym = f"{base}USD=X"
            leg_b_sym = f"USD{quote}=X" if f"USD{quote}=X" in fx_pairs else f"{quote}USD=X"
            if leg_a_sym in fx_pairs and leg_b_sym in fx_pairs:
                leg_a = master[(leg_a_sym, "log_return")]
                leg_b = master[(leg_b_sym, "log_return")]
                implied = implied_cross_return(leg_a, leg_a_sym, leg_b, leg_b_sym)
                for w in windows:
                    cov = ret.rolling(w).cov(implied)
                    var = implied.rolling(w).var(ddof=0)
                    feats[f"pair_beta_{w}"] = cov / var.replace(0.0, np.nan)

        if pair in triangular:
            feats = feats.join(triangular[pair])

        blocks[pair] = feats

    # Engle-Granger cointegration residual on the dependent leg of each pair.
    for dependent, independent in COINTEGRATION_PAIRS:
        if dependent in blocks and (independent, "close") in master.columns:
            resid_z = rolling_ols_residual_zscore(
                master[(dependent, "close")],
                master[(independent, "close")],
                settings.WINDOWS.long,
            )
            blocks[dependent]["cointegration_residual"] = resid_z

    result = pd.concat(blocks, axis=1)
    result.columns = result.columns.set_names(["pair", "feature"])
    return result
