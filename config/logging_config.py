"""Structured logging setup using loguru.

Provides a single configure_logging entry point so every script and module
shares a consistent, structured log format. Console output is human readable;
file output (optional) is JSON for downstream parsing.
"""

import sys
from pathlib import Path

from loguru import logger

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """Configure the global loguru logger.

    Removes loguru's default handler and installs a console handler plus an
    optional JSON file handler. Safe to call multiple times; each call resets
    the handler set.

    Args:
        level: Minimum log level for the console handler, e.g. "INFO".
        log_file: Optional path for a JSON-serialised log sink. Parent
            directories are created if needed.

    Returns:
        None.

    Raises:
        ValueError: If level is not a recognised loguru level name.
    """
    valid_levels = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
    if level.upper() not in valid_levels:
        raise ValueError(f"Unknown log level: {level}. Expected one of {sorted(valid_levels)}.")

    logger.remove()
    logger.add(sys.stderr, level=level.upper(), format=_CONSOLE_FORMAT, colorize=True)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level="DEBUG",
            serialize=True,
            rotation="10 MB",
            retention=5,
        )


def get_logger():
    """Return the shared loguru logger instance.

    Returns:
        The configured loguru logger.
    """
    return logger
