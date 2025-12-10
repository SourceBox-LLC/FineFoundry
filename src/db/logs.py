"""Database logging handler and utilities.

Provides a logging handler that writes to SQLite database.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime
from typing import List, Optional

from db.core import get_connection, init_db


class DatabaseHandler(logging.Handler):
    """Logging handler that writes log records to SQLite database."""

    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)
        self._initialized = False

    def _ensure_db(self) -> None:
        """Ensure database is initialized."""
        if not self._initialized:
            try:
                init_db()
                self._initialized = True
            except Exception:
                pass

    def emit(self, record: logging.LogRecord) -> None:
        """Write a log record to the database."""
        try:
            self._ensure_db()
            conn = get_connection()
            cursor = conn.cursor()

            # Format exception info if present
            exc_info = None
            if record.exc_info:
                exc_info = "".join(traceback.format_exception(*record.exc_info))

            cursor.execute(
                """
                INSERT INTO app_logs (timestamp, level, logger, message, module, func_name, line_no, exc_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.fromtimestamp(record.created).isoformat(),
                    record.levelname,
                    record.name,
                    record.getMessage(),
                    record.module,
                    record.funcName,
                    record.lineno,
                    exc_info,
                ),
            )
            conn.commit()
        except Exception:
            # Don't raise exceptions from logging
            self.handleError(record)


def get_logs(
    level: Optional[str] = None,
    logger: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[dict]:
    """Retrieve logs from the database.

    Args:
        level: Filter by log level (e.g., "ERROR", "WARNING")
        logger: Filter by logger name
        limit: Maximum number of logs to return
        offset: Offset for pagination

    Returns:
        List of log dictionaries
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM app_logs WHERE 1=1"
    params: list = []

    if level:
        query += " AND level = ?"
        params.append(level)

    if logger:
        query += " AND logger LIKE ?"
        params.append(f"%{logger}%")

    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()

    return [dict(row) for row in rows]


def clear_logs(before_date: Optional[str] = None) -> int:
    """Clear logs from the database.

    Args:
        before_date: Optional ISO date string. If provided, only clears logs before this date.

    Returns:
        Number of logs deleted
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    if before_date:
        cursor.execute("DELETE FROM app_logs WHERE timestamp < ?", (before_date,))
    else:
        cursor.execute("DELETE FROM app_logs")

    deleted = cursor.rowcount
    conn.commit()
    return deleted


def get_log_count(level: Optional[str] = None) -> int:
    """Get the count of logs in the database.

    Args:
        level: Optional filter by log level

    Returns:
        Number of logs
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    if level:
        cursor.execute("SELECT COUNT(*) FROM app_logs WHERE level = ?", (level,))
    else:
        cursor.execute("SELECT COUNT(*) FROM app_logs")

    return cursor.fetchone()[0]


def setup_database_logging(
    level: int = logging.INFO,
    logger_name: Optional[str] = None,
) -> DatabaseHandler:
    """Set up database logging for the application.

    Args:
        level: Minimum log level to capture
        logger_name: Optional specific logger name. If None, configures root logger.

    Returns:
        The DatabaseHandler instance
    """
    handler = DatabaseHandler(level)
    handler.setFormatter(logging.Formatter("%(message)s"))

    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()

    # Remove any existing file handlers
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)

    logger.addHandler(handler)
    return handler
