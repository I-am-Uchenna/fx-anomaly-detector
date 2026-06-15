"""Alert dispatch to console, structured log and an optional webhook.

Console output is colour coded by severity. The webhook is disabled unless a
URL is configured in settings; failures to POST are logged but never raise, so
a notification outage cannot break the detection pipeline.
"""

from __future__ import annotations

import requests

from config import settings
from config.logging_config import get_logger
from src.alerts.engine import Alert
from src.alerts.severity import Severity

logger = get_logger()

# ANSI colour codes by severity for console output.
_COLORS = {
    Severity.LOW.value: "\033[36m",  # cyan
    Severity.MEDIUM.value: "\033[33m",  # yellow
    Severity.HIGH.value: "\033[35m",  # magenta
    Severity.CRITICAL.value: "\033[31m",  # red
}
_RESET = "\033[0m"


def _format_console(alert: Alert) -> str:
    """Format a single alert for coloured console output.

    Args:
        alert: The alert to format.

    Returns:
        A colourised one-line string.
    """
    color = _COLORS.get(alert.severity, "")
    return (
        f"{color}[{alert.severity}] {alert.timestamp} {alert.pair} "
        f"score={alert.ensemble_score:.3f} detectors={alert.n_detectors_flagged} "
        f"-> {alert.suggested_action}{_RESET}"
    )


def notify_console(alerts: list[Alert]) -> None:
    """Print alerts to the console with severity colours.

    Args:
        alerts: Alerts to display.

    Returns:
        None.
    """
    for alert in alerts:
        print(_format_console(alert))


def notify_log(alerts: list[Alert]) -> None:
    """Emit each alert as a structured log entry.

    Args:
        alerts: Alerts to log.

    Returns:
        None.
    """
    for alert in alerts:
        logger.bind(alert=True).info(
            "ALERT {} {} score={:.3f} severity={}",
            alert.timestamp,
            alert.pair,
            alert.ensemble_score,
            alert.severity,
        )


def notify_webhook(
    alerts: list[Alert], url: str = settings.WEBHOOK_URL, timeout: float = 5.0
) -> int:
    """POST alerts to a webhook if a URL is configured.

    Args:
        alerts: Alerts to send.
        url: Webhook URL; if empty, dispatch is skipped.
        timeout: Per-request timeout in seconds.

    Returns:
        The number of alerts successfully delivered.
    """
    if not url:
        return 0
    delivered = 0
    for alert in alerts:
        try:
            response = requests.post(url, json=alert.__dict__, timeout=timeout)
            response.raise_for_status()
            delivered += 1
        except requests.RequestException as exc:
            logger.warning("Webhook delivery failed for {}: {}", alert.pair, exc)
    return delivered


def dispatch(alerts: list[Alert]) -> None:
    """Dispatch alerts to all enabled channels.

    Args:
        alerts: Alerts to dispatch.

    Returns:
        None.
    """
    if not alerts:
        return
    notify_console(alerts)
    notify_log(alerts)
    notify_webhook(alerts)
