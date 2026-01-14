"""Core database initialization and connection management."""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Optional

# Thread-local storage for connections
_local = threading.local()

# Default database path (project root)
_DB_NAME = "finefoundry.db"

# Override for testing - set _DB_PATH_OVERRIDE["path"] to use a different db
_DB_PATH_OVERRIDE: dict = {}


def get_db_path(project_root: Optional[str] = None) -> str:
    """Get the path to the SQLite database file."""
    # Check for test override first
    if "path" in _DB_PATH_OVERRIDE:
        return _DB_PATH_OVERRIDE["path"]
    if project_root:
        return os.path.join(project_root, _DB_NAME)
    # Default: project root (parent of src/)
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(src_dir, "..", _DB_NAME)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Get a thread-local database connection.

    Creates the connection if it doesn't exist for this thread.
    Connections are reused within the same thread.
    """
    if db_path is None:
        db_path = get_db_path()

    # Normalize path for consistent caching
    db_path = os.path.abspath(db_path)

    # Check if we have a connection for this path in this thread
    if not hasattr(_local, "connections"):
        _local.connections = {}

    if db_path not in _local.connections:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        _local.connections[db_path] = conn

    return _local.connections[db_path]


def close_all_connections() -> None:
    """Close all thread-local connections."""
    if hasattr(_local, "connections"):
        for conn in _local.connections.values():
            try:
                conn.close()
            except Exception:
                pass
        _local.connections.clear()


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize the database schema.

    Creates all tables if they don't exist. Safe to call multiple times.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Settings table - key/value store for app settings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # Training configs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            config_json TEXT NOT NULL,
            train_target TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # Last used config tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # Scrape sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scrape_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            source TEXT NOT NULL,
            source_details TEXT,
            dataset_format TEXT DEFAULT 'standard',
            pair_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            metadata_json TEXT
        )
    """)

    # Migration: Add 'name' column to existing scrape_sessions table
    try:
        cursor.execute("ALTER TABLE scrape_sessions ADD COLUMN name TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        # Column already exists
        pass

    # Scraped pairs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scraped_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            input_text TEXT NOT NULL,
            output_text TEXT NOT NULL,
            source_url TEXT,
            metadata_json TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES scrape_sessions(id) ON DELETE CASCADE
        )
    """)

    # Training runs table - managed storage for training outputs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            base_model TEXT,
            dataset_source TEXT,
            dataset_id TEXT,
            storage_path TEXT NOT NULL,
            output_dir TEXT,
            adapter_path TEXT,
            checkpoint_path TEXT,
            hp_json TEXT,
            logs_json TEXT,
            started_at TEXT,
            completed_at TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            metadata_json TEXT
        )
    """)

    # Evaluation runs table - store benchmark results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            training_run_id INTEGER,
            base_model TEXT NOT NULL,
            adapter_path TEXT,
            benchmark TEXT NOT NULL,
            num_samples INTEGER,
            batch_size INTEGER,
            metrics_json TEXT NOT NULL,
            is_base_eval INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (training_run_id) REFERENCES training_runs(id) ON DELETE SET NULL
        )
    """)

    # Application logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS app_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (datetime('now')),
            level TEXT NOT NULL,
            logger TEXT,
            message TEXT NOT NULL,
            module TEXT,
            func_name TEXT,
            line_no INTEGER,
            exc_info TEXT
        )
    """)

    # Create indexes for common queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_scraped_pairs_session 
        ON scraped_pairs(session_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_app_logs_timestamp 
        ON app_logs(timestamp)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_app_logs_level 
        ON app_logs(level)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_configs_target 
        ON training_configs(train_target)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_runs_status 
        ON training_runs(status)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_evaluation_runs_training 
        ON evaluation_runs(training_run_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_evaluation_runs_benchmark 
        ON evaluation_runs(benchmark)
    """)

    conn.commit()


def get_schema_version(db_path: Optional[str] = None) -> int:
    """Get the current schema version for migrations."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT value FROM app_state WHERE key = 'schema_version'")
        row = cursor.fetchone()
        return int(row["value"]) if row else 0
    except Exception:
        return 0


def set_schema_version(version: int, db_path: Optional[str] = None) -> None:
    """Set the schema version."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO app_state (key, value) VALUES ('schema_version', ?)", (str(version),))
    conn.commit()
