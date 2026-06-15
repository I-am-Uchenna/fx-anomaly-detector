"""Standalone backtest report generator.

Fetches live data, builds features, runs the walk-forward backtest and writes
the HTML report. Useful for producing a report without the full live detection
loop.

Example:
    python scripts/generate_report.py --start 2018-01-01 --end 2024-12-31 --output report.html
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from config.logging_config import configure_logging, get_logger  # noqa: E402
from src.backtest.engine import run_backtest  # noqa: E402
from src.backtest.metrics import compute_metrics  # noqa: E402
from src.backtest.reporter import generate_report  # noqa: E402
from src.data.fetcher import fetch_fx_data  # noqa: E402
from src.detectors.regime import compute_regimes  # noqa: E402
from src.features.builder import build_features  # noqa: E402

logger = get_logger()


@click.command()
@click.option("--start", default=settings.DATA_START_DATE, help="Start date YYYY-MM-DD.")
@click.option("--end", default=None, help="End date YYYY-MM-DD.")
@click.option("--output", default="data/signals/backtest_report.html", help="Output HTML path.")
@click.option("--no-autoencoder", is_flag=True, help="Skip the autoencoder detector.")
def main(start: str, end: str | None, output: str, no_autoencoder: bool) -> None:
    """Generate a standalone HTML backtest report from live data."""
    configure_logging()
    raw = fetch_fx_data(start=start, end=end)
    features = build_features(raw, persist=True)

    result = run_backtest(features, include_autoencoder=not no_autoencoder)
    metrics = compute_metrics(result.strategy_returns, result.equity_curve, result.trades)

    regimes = {
        pair: compute_regimes(features, pair).state
        for pair in features.columns.get_level_values(0).unique()
    }
    path = generate_report(result, metrics, Path(output), regime_states=regimes)
    click.echo(f"Report written to {path}. Sharpe={metrics.sharpe_ratio:.3f}")


if __name__ == "__main__":
    main()
