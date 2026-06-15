"""Central configuration for the FX anomaly detector.

All magic numbers (pairs, window sizes, detector thresholds, backtest
parameters) live here so they can be tuned in one place. Every other module
imports from this file rather than hard coding constants.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PairConfig:
    """Configuration for a single currency pair.

    Args:
        symbol: Yahoo Finance ticker, e.g. "EURUSD=X".
        name: Human readable name, e.g. "EUR/USD".
        pip_size: Price increment of one pip (0.0001 for most, 0.01 for JPY).
        avg_spread_pips: Typical institutional spread in pips.
    """

    symbol: str
    name: str
    pip_size: float
    avg_spread_pips: float


# G10 major and cross pairs.
FX_PAIRS: list[PairConfig] = [
    PairConfig("EURUSD=X", "EUR/USD", 0.0001, 0.8),
    PairConfig("GBPUSD=X", "GBP/USD", 0.0001, 1.0),
    PairConfig("USDJPY=X", "USD/JPY", 0.01, 0.9),
    PairConfig("USDCHF=X", "USD/CHF", 0.0001, 1.2),
    PairConfig("AUDUSD=X", "AUD/USD", 0.0001, 1.1),
    PairConfig("NZDUSD=X", "NZD/USD", 0.0001, 1.5),
    PairConfig("USDCAD=X", "USD/CAD", 0.0001, 1.3),
    PairConfig("EURGBP=X", "EUR/GBP", 0.0001, 1.0),
    PairConfig("EURJPY=X", "EUR/JPY", 0.01, 1.5),
    PairConfig("GBPJPY=X", "GBP/JPY", 0.01, 2.0),
]

# Triangular arbitrage triplets (base_quote, quote_cross, base_cross).
# The cross rate of the third leg should equal the product of the first two.
TRIANGULAR_TRIPLETS: list[tuple[str, str, str]] = [
    ("EURUSD=X", "USDJPY=X", "EURJPY=X"),
    ("GBPUSD=X", "USDJPY=X", "GBPJPY=X"),
    ("EURUSD=X", "GBPUSD=X", "EURGBP=X"),
]

# Auxiliary market tickers used by the macro feature module.
VIX_TICKER = "^VIX"
DXY_TICKER = "DX-Y.NYB"
GOLD_TICKER = "GC=F"
MACRO_TICKERS: list[str] = [VIX_TICKER, DXY_TICKER, GOLD_TICKER]


@dataclass(frozen=True)
class WindowConfig:
    """Rolling window sizes in trading days."""

    short: int = 5  # 1 week
    medium: int = 21  # 1 month
    long: int = 63  # 1 quarter
    extended: int = 252  # 1 year

    def as_list(self) -> list[int]:
        """Return the short, medium and long windows as a list.

        Returns:
            The three primary feature windows in ascending order.
        """
        return [self.short, self.medium, self.long]


@dataclass(frozen=True)
class DetectorConfig:
    """Anomaly detection thresholds."""

    zscore_threshold: float = 3.0
    mahalanobis_threshold: float = 3.5
    isolation_forest_contamination: float = 0.01
    lof_n_neighbors: int = 20
    autoencoder_reconstruction_percentile: float = 99.0
    hmm_n_regimes: int = 3  # normal, stress, crisis
    hmm_min_observations: int = 504
    crisis_probability_threshold: float = 0.7
    cointegration_zscore_threshold: float = 2.5
    cointegration_recalibration_days: int = 63
    min_anomaly_persistence: int = 2  # minimum consecutive bars
    alert_cooldown_minutes: int = 60


@dataclass(frozen=True)
class EnsembleConfig:
    """Ensemble voting weights and decision thresholds.

    Weights are keyed by detector_name and must sum to 1.0.
    """

    weights: tuple[tuple[str, float], ...] = (
        ("zscore", 0.15),
        ("mahalanobis", 0.15),
        ("isolation_forest", 0.15),
        ("lof", 0.10),
        ("autoencoder", 0.20),
        ("regime", 0.15),
        ("cointegration", 0.10),
    )
    score_threshold: float = 0.5
    min_detectors_flagged: int = 2

    def weight_map(self) -> dict:
        """Return the weights as a plain dictionary.

        Returns:
            Mapping of detector_name to its ensemble weight.
        """
        return {name: weight for name, weight in self.weights}


@dataclass(frozen=True)
class BacktestConfig:
    """Backtesting parameters."""

    initial_capital: float = 1_000_000.0
    max_position_pct: float = 0.02  # 2% of capital per trade
    transaction_cost_pips: float = 1.0
    slippage_pips: float = 0.5
    walk_forward_train_days: int = 504  # 2 years
    walk_forward_test_days: int = 63  # 1 quarter
    purge_days: int = 5
    risk_free_rate: float = 0.04
    trading_days_per_year: int = 252


WINDOWS = WindowConfig()
DETECTOR = DetectorConfig()
ENSEMBLE = EnsembleConfig()
BACKTEST = BacktestConfig()

# Data parameters.
DATA_START_DATE = "2015-01-01"
DATA_FREQUENCY = "1d"
MAX_FORWARD_FILL_DAYS = 3
OUTLIER_RETURN_SIGMA = 5.0
WINSORIZE_LOWER_PCT = 0.01
WINSORIZE_UPPER_PCT = 0.99

# Webhook notifier (disabled by default; set a URL to enable).
WEBHOOK_URL = ""
