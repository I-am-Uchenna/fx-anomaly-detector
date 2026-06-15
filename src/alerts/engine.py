"""Alert generation with rising-edge detection, cooldown and persistence.

An alert fires when a pair's ensemble flag transitions from False to True. The
rising-edge rule prevents a sustained anomaly from generating an alert every
bar; an additional cooldown suppresses repeat alerts for the same pair within a
configurable window. Alerts are appended to data/signals/alerts.jsonl.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from config import settings
from config.logging_config import get_logger
from src.alerts.severity import classify, suggested_action
from src.risk import var as var_module

logger = get_logger()

_ALERTS_PATH = Path("data/signals/alerts.jsonl")


@dataclass
class Alert:
    """A single anomaly alert.

    Args:
        timestamp: ISO timestamp of the flagged bar.
        pair: Pair symbol.
        ensemble_score: Ensemble anomaly score.
        severity: Classified severity level.
        n_detectors_flagged: Count of individual detectors that flagged.
        detector_scores: Per-detector score map.
        regime_state: Regime label at the bar, if available.
        var_99: Trailing 99% historical VaR estimate, if available.
        suggested_action: Recommended operator action.
    """

    timestamp: str
    pair: str
    ensemble_score: float
    severity: str
    n_detectors_flagged: int
    detector_scores: dict[str, float]
    regime_state: float | None
    var_99: float | None
    suggested_action: str = field(default="")


def _detector_score_map(row: pd.Series) -> dict[str, float]:
    """Extract per-detector scores from an ensemble row.

    Args:
        row: One row of the ensemble DataFrame.

    Returns:
        Mapping of detector name to its score, for finite scores only.
    """
    out: dict[str, float] = {}
    for col in row.index:
        if isinstance(col, str) and col.startswith("score_"):
            value = row[col]
            if pd.notna(value):
                out[col[len("score_") :]] = float(value)
    return out


def generate_alerts(
    ensemble_df: pd.DataFrame,
    pair_returns: dict[str, pd.Series] | None = None,
    regime_states: dict[str, pd.Series] | None = None,
    cooldown_minutes: int = settings.DETECTOR.alert_cooldown_minutes,
    var_window: int = settings.WINDOWS.long,
) -> list[Alert]:
    """Generate alerts from the ensemble output.

    Args:
        ensemble_df: Output of detectors.ensemble.build_ensemble.
        pair_returns: Optional map of pair to return series for the VaR
            estimate attached to each alert.
        regime_states: Optional map of pair to regime-state series.
        cooldown_minutes: Suppress repeat alerts for a pair within this window.
        var_window: Trailing window for the historical VaR estimate.

    Returns:
        A list of Alert objects in chronological order.
    """
    cooldown = pd.Timedelta(minutes=cooldown_minutes)
    alerts: list[Alert] = []

    for pair, group in ensemble_df.sort_values("datetime").groupby("pair"):
        group = group.reset_index(drop=True)
        flag = group["ensemble_flag"].astype(bool)
        rising_edge = flag & ~flag.shift(1, fill_value=False)
        last_alert_time: pd.Timestamp | None = None

        returns = pair_returns.get(pair) if pair_returns else None
        regimes = regime_states.get(pair) if regime_states else None

        for idx in group.index[rising_edge.to_numpy()]:
            row = group.loc[idx]
            timestamp = pd.Timestamp(row["datetime"])
            if last_alert_time is not None and timestamp - last_alert_time < cooldown:
                continue
            last_alert_time = timestamp

            severity = classify(float(row["ensemble_score"]))
            var_99 = None
            if returns is not None:
                trailing = returns.loc[:timestamp].tail(var_window)
                if len(trailing) >= 2:
                    var_99 = var_module.historical_var(trailing, 0.99).var
            regime_state = None
            if regimes is not None and timestamp in regimes.index:
                value = regimes.loc[timestamp]
                regime_state = None if pd.isna(value) else float(value)

            alerts.append(
                Alert(
                    timestamp=timestamp.isoformat(),
                    pair=str(pair),
                    ensemble_score=float(row["ensemble_score"]),
                    severity=severity.value,
                    n_detectors_flagged=int(row["n_flags"]),
                    detector_scores=_detector_score_map(row),
                    regime_state=regime_state,
                    var_99=var_99,
                    suggested_action=suggested_action(severity),
                )
            )

    logger.info("Generated {} alerts", len(alerts))
    return alerts


def persist_alerts(alerts: list[Alert], path: Path = _ALERTS_PATH) -> None:
    """Append alerts to the JSONL alert log.

    Args:
        alerts: Alerts to write.
        path: Destination JSONL file; parent directories are created.

    Returns:
        None.
    """
    if not alerts:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for alert in alerts:
            handle.write(json.dumps(asdict(alert)) + "\n")
    logger.info("Persisted {} alerts to {}", len(alerts), path)
