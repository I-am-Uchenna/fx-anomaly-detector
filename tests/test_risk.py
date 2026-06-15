"""Tests for risk metrics and position sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk import drawdown, position_sizing, var


def test_parametric_var_matches_normal_theory() -> None:
    rng = np.random.default_rng(0)
    sigma = 0.02
    sample = pd.Series(rng.normal(0.0, sigma, 200_000))
    estimate = var.parametric_var(sample, confidence=0.95)
    # 95% normal VaR is about 1.645 * sigma when the mean is ~0.
    assert estimate.var == pytest.approx(1.645 * sigma, rel=0.05)
    assert estimate.cvar > estimate.var


def test_historical_var_close_to_quantile() -> None:
    rng = np.random.default_rng(1)
    sample = pd.Series(rng.normal(0.0, 0.02, 200_000))
    estimate = var.historical_var(sample, confidence=0.99)
    assert estimate.var == pytest.approx(2.326 * 0.02, rel=0.1)


def test_var_rejects_bad_confidence() -> None:
    with pytest.raises(ValueError):
        var.parametric_var(pd.Series([0.01, -0.01]), confidence=1.5)


def test_kelly_zero_win_probability_is_negative() -> None:
    assert position_sizing.kelly_fraction(0.0, 2.0) == pytest.approx(-0.5)


def test_kelly_certain_win_equals_one() -> None:
    assert position_sizing.kelly_fraction(1.0, 2.0) == pytest.approx(1.0)


def test_kelly_rejects_nonpositive_ratio() -> None:
    with pytest.raises(ValueError):
        position_sizing.kelly_fraction(0.5, 0.0)


def test_size_position_is_capped() -> None:
    sizing = position_sizing.size_position(0.99, 5.0, max_fraction=0.02)
    assert sizing.capped_fraction == pytest.approx(0.02)
    assert sizing.full_kelly > sizing.capped_fraction


def test_estimate_from_outcomes() -> None:
    outcomes = [0.02, -0.01, 0.03, -0.02, 0.01]
    win_prob, ratio = position_sizing.estimate_from_outcomes(outcomes)
    assert win_prob == pytest.approx(0.6)
    assert ratio > 0


def test_drawdown_on_monotonic_curve_is_zero() -> None:
    equity = pd.Series(np.linspace(1.0, 2.0, 100))
    summary = drawdown.summarize_drawdown(equity)
    assert summary.max_drawdown == pytest.approx(0.0)


def test_drawdown_detects_known_decline() -> None:
    equity = pd.Series([1.0, 1.2, 0.9, 1.0, 1.3])
    summary = drawdown.summarize_drawdown(equity)
    # Peak 1.2 down to 0.9 is a 25% drawdown.
    assert summary.max_drawdown == pytest.approx(0.25)


def test_monte_carlo_var_reasonable() -> None:
    rng = np.random.default_rng(2)
    sample = pd.Series(rng.normal(0.0, 0.02, 5000))
    estimate = var.monte_carlo_var(sample, confidence=0.95, n_sims=20000, distribution="t")
    assert 0.02 < estimate.var < 0.06
    assert estimate.cvar >= estimate.var


def test_monte_carlo_rejects_unknown_distribution() -> None:
    with pytest.raises(ValueError):
        var.monte_carlo_var(pd.Series([0.01, -0.01, 0.02]), distribution="cauchy")


def test_compute_var_cvar_has_all_keys() -> None:
    rng = np.random.default_rng(3)
    out = var.compute_var_cvar(pd.Series(rng.normal(0, 0.01, 2000)))
    for method in ("parametric", "historical", "monte_carlo"):
        for conf in (95, 99):
            assert f"{method}_{conf}" in out


def test_rolling_var_shapes_and_method() -> None:
    rng = np.random.default_rng(4)
    rets = pd.Series(rng.normal(0, 0.01, 300))
    hist = var.rolling_var(rets, window=63, method="historical")
    param = var.rolling_var(rets, window=63, method="parametric")
    assert len(hist) == len(rets)
    assert hist.iloc[:62].isna().all()
    assert param.notna().sum() > 0
    with pytest.raises(ValueError):
        var.rolling_var(rets, window=63, method="bogus")
