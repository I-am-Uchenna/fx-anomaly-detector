"""Command-line entry point for the FX anomaly detector.

Examples:
    python scripts/run_detector.py --mode live --pairs all
    python scripts/run_detector.py --mode backtest --start 2018-01-01 --end 2024-12-31
    python scripts/run_detector.py --mode calibrate --pairs "EURUSD=X,GBPUSD=X"
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

# Ensure the project root is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from config.logging_config import configure_logging, get_logger  # noqa: E402
from src.alerts.engine import generate_alerts, persist_alerts  # noqa: E402
from src.alerts.notifier import dispatch  # noqa: E402
from src.backtest.engine import run_backtest  # noqa: E402
from src.backtest.metrics import compute_metrics  # noqa: E402
from src.backtest.reporter import generate_report  # noqa: E402
from src.data.fetcher import fetch_fx_data  # noqa: E402
from src.detectors import ensemble  # noqa: E402
from src.features.builder import build_features  # noqa: E402

logger = get_logger()


def _parse_pairs(pairs: str) -> list[str] | None:
    """Parse the --pairs option into a list of symbols or None for all.

    Args:
        pairs: Either "all" or a comma/space separated list of symbols.

    Returns:
        A list of symbols, or None to mean every configured pair.
    """
    if pairs.strip().lower() == "all":
        return None
    tokens = [t for t in pairs.replace(",", " ").split() if t]
    return tokens or None


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["live", "backtest", "calibrate"]),
    required=True,
    help="Run mode.",
)
@click.option("--pairs", default="all", help="'all' or comma/space separated symbols.")
@click.option("--start", default=settings.DATA_START_DATE, help="Start date YYYY-MM-DD.")
@click.option("--end", default=None, help="End date YYYY-MM-DD.")
@click.option("--no-autoencoder", is_flag=True, help="Skip the autoencoder detector.")
@click.option("--log-level", default="INFO", help="Logging level.")
def main(
    mode: str,
    pairs: str,
    start: str,
    end: str | None,
    no_autoencoder: bool,
    log_level: str,
) -> None:
    """Fetch live data and run the detector in the requested mode."""
    configure_logging(level=log_level)
    symbols = _parse_pairs(pairs)
    include_ae = not no_autoencoder

    logger.info("Fetching live data from yfinance (start={}, end={})", start, end)
    raw = fetch_fx_data(symbols=symbols, start=start, end=end)
    features = build_features(raw, pair_symbols=symbols, persist=True)

    if mode == "live":
        result = ensemble.detect(features, include_autoencoder=include_ae)
        pair_returns = {
            p: features[(p, "log_return")] for p in features.columns.get_level_values(0).unique()
        }
        alerts = generate_alerts(result, pair_returns=pair_returns)
        persist_alerts(alerts)
        dispatch(alerts)
        click.echo(f"Generated {len(alerts)} alerts.")

    elif mode == "backtest":
        result = run_backtest(features, include_autoencoder=include_ae)
        metrics = compute_metrics(result.strategy_returns, result.equity_curve, result.trades)
        report_path = generate_report(result, metrics, Path("data/signals/backtest_report.html"))
        click.echo(f"Backtest complete. Sharpe={metrics.sharpe_ratio:.3f}. Report: {report_path}")

    elif mode == "calibrate":
        from scripts.calibrate import calibrate_thresholds

        summary = calibrate_thresholds(features, include_autoencoder=include_ae)
        click.echo(summary.to_string())


if __name__ == "__main__":
    main()
