"""Threshold calibration and false-positive analysis.

Runs every detector over a feature frame and summarises, per detector, the flag
rate (a proxy for the false-positive rate on assumed-normal data) and the score
distribution. Use this to sanity check that thresholds are not flagging an
implausibly large share of bars.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import get_logger  # noqa: E402
from src.detectors import ensemble  # noqa: E402

logger = get_logger()


def calibrate_thresholds(features: pd.DataFrame, include_autoencoder: bool = False) -> pd.DataFrame:
    """Summarise detector flag rates and score distributions.

    Args:
        features: Multi-level (pair, feature) frame.
        include_autoencoder: Whether to include the autoencoder detector.

    Returns:
        A DataFrame indexed by detector_name with columns flag_rate,
        mean_score, p95_score and p99_score.
    """
    outputs = ensemble.run_all_detectors(features, include_autoencoder=include_autoencoder)
    rows = []
    for name, df in outputs.items():
        scores = df["anomaly_score"].dropna()
        rows.append(
            {
                "detector_name": name,
                "flag_rate": float(df["anomaly_flag"].mean()),
                "mean_score": float(scores.mean()) if not scores.empty else float("nan"),
                "p95_score": float(scores.quantile(0.95)) if not scores.empty else float("nan"),
                "p99_score": float(scores.quantile(0.99)) if not scores.empty else float("nan"),
            }
        )
    summary = (
        pd.DataFrame(rows).set_index("detector_name").sort_values("flag_rate", ascending=False)
    )
    logger.info("Calibration summary:\n{}", summary.round(4).to_string())
    return summary
