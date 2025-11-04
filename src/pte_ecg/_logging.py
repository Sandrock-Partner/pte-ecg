"""Logging configuration for the PTE-ECG package.

This module provides a configured logger instance and utility functions for setting log levels
and file handlers. The logger is configured to output to stdout by default with a standardized
format that includes the logger name, log level, and message.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

_formatter = logging.Formatter("%(name)s | %(levelname)s | %(message)s")


def _configure_default_logging() -> logging.Logger:
    """Configure default logging for the root logger.

    This sets up basic console logging with INFO level and proper formatting.
    Called automatically when this module is imported.
    """
    logger = logging.getLogger("cad_dataset_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setLevel(logging.WARNING)
    _console_handler.setFormatter(_formatter)
    logger.addHandler(_console_handler)
    return logger


logger = _configure_default_logging()


def set_log_level(
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
) -> None:
    """Configure logging level.

    Args:
        log_level: Log level for console output (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Example:
        >>> from pathlib import Path
        >>> logger = set_log_level(log_level="WARNING")
        >>> logger.info("This won't show (below WARNING)")
        >>> logger.warning("This will show on console and in file")
    """
    numeric_level: int = int(getattr(logging, log_level))
    old_level = logger.level
    if old_level == numeric_level:
        return
    logger.setLevel(numeric_level)


def set_log_file(
    log_file: Path,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG",
) -> None:
    """Configure logging for the root logger (all modules inherit from it).

    This function configures the root logger non-destructively, preserving existing
    file handlers while updating console handlers. This ensures that when subprocesses
    call set_log_level(), they don't remove the main scheduler's file handler.

    Args:
        log_file: Path to log file.
        log_level: Log level for console output (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        The root logger instance

    Example:
        >>> from pathlib import Path
        >>> logger = set_log_file(
        ...     log_level="WARNING",
        ...     log_file=Path("logs/pipeline.log"),
        ... )
    """
    numeric_level: int = int(getattr(logging, log_level))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)

    file_handler = RotatingFileHandler(log_file, maxBytes=100_000_000, backupCount=3)
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(_formatter)
    logger.addHandler(file_handler)
