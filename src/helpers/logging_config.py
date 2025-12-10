"""Centralized logging configuration for FineFoundry.

This module provides a unified logging setup with:
- Database logging (SQLite)
- Optional console output
- Configurable log levels
"""

import logging
import os
from typing import Optional

from db.logs import DatabaseHandler


# Log format for console output
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    log_file: Optional[str] = None,  # Deprecated, ignored
    level: int = logging.INFO,
    console_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # Deprecated, ignored
    backup_count: int = 5,  # Deprecated, ignored
) -> logging.Logger:
    """Set up a logger with database and optional console handlers.

    Args:
        name: Logger name (typically __name__ from calling module)
        log_file: Deprecated, ignored (logs go to database)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to also log to console
        max_bytes: Deprecated, ignored
        backup_count: Deprecated, ignored

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Database handler
    db_handler = DatabaseHandler(level)
    logger.addHandler(db_handler)

    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with default configuration.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Logger instance
    """
    # Check if DEBUG mode is enabled via environment variable
    debug_mode = os.getenv("FINEFOUNDRY_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    level = logging.DEBUG if debug_mode else logging.INFO

    return setup_logger(name, level=level)


# Create main application logger
app_logger = get_logger("finefoundry")


def set_global_log_level(level: int):
    """Set log level for all existing loggers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith("finefoundry"):
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)
