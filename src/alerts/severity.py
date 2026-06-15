"""Alert severity classification by ensemble score band."""

from __future__ import annotations

from enum import Enum


class Severity(str, Enum):
    """Ordered alert severity levels."""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


def classify(score: float) -> Severity:
    """Map an ensemble anomaly score to a severity level.

    Bands: LOW [0.5, 0.65), MEDIUM [0.65, 0.8), HIGH [0.8, 0.9),
    CRITICAL [0.9, 1.0]. Scores below 0.5 are NONE.

    Args:
        score: Ensemble anomaly score in [0, 1].

    Returns:
        The corresponding Severity.
    """
    if score >= 0.9:
        return Severity.CRITICAL
    if score >= 0.8:
        return Severity.HIGH
    if score >= 0.65:
        return Severity.MEDIUM
    if score >= 0.5:
        return Severity.LOW
    return Severity.NONE


def suggested_action(severity: Severity) -> str:
    """Return a suggested operator action for a severity level.

    Args:
        severity: The classified severity.

    Returns:
        A short, non-prescriptive action string.
    """
    actions = {
        Severity.NONE: "No action.",
        Severity.LOW: "Monitor; no immediate action required.",
        Severity.MEDIUM: "Review the pair and related crosses.",
        Severity.HIGH: "Reduce exposure and investigate the driver.",
        Severity.CRITICAL: "Consider hedging or flattening exposure; escalate.",
    }
    return actions[severity]
