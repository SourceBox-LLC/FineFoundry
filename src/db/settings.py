"""Settings storage using SQLite.

Replaces ff_settings.json with database storage.
Settings are stored as key-value pairs with JSON values.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .core import get_connection, init_db


def get_setting(key: str, default: Any = None, db_path: Optional[str] = None) -> Any:
    """Get a setting value by key.
    
    Args:
        key: Setting key (e.g., "huggingface.token", "runpod.api_key")
        default: Default value if key doesn't exist
        db_path: Optional database path
        
    Returns:
        The setting value (parsed from JSON) or default
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    
    if row is None:
        return default
    
    try:
        return json.loads(row["value"])
    except (json.JSONDecodeError, TypeError):
        return row["value"]


def set_setting(key: str, value: Any, db_path: Optional[str] = None) -> None:
    """Set a setting value.
    
    Args:
        key: Setting key
        value: Value to store (will be JSON-encoded)
        db_path: Optional database path
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # JSON-encode the value
    json_value = json.dumps(value, ensure_ascii=False)
    
    cursor.execute("""
        INSERT INTO settings (key, value, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = datetime('now')
    """, (key, json_value))
    
    conn.commit()


def get_all_settings(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Get all settings as a dictionary.
    
    Returns:
        Dictionary of all settings with parsed values
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT key, value FROM settings")
    rows = cursor.fetchall()
    
    result = {}
    for row in rows:
        try:
            result[row["key"]] = json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            result[row["key"]] = row["value"]
    
    return result


def delete_setting(key: str, db_path: Optional[str] = None) -> bool:
    """Delete a setting.
    
    Args:
        key: Setting key to delete
        db_path: Optional database path
        
    Returns:
        True if setting was deleted, False if it didn't exist
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM settings WHERE key = ?", (key,))
    conn.commit()
    
    return cursor.rowcount > 0


# Convenience functions for common settings

def get_hf_token(db_path: Optional[str] = None) -> str:
    """Get Hugging Face token."""
    return get_setting("huggingface.token", "", db_path) or ""


def set_hf_token(token: str, db_path: Optional[str] = None) -> None:
    """Set Hugging Face token."""
    set_setting("huggingface.token", token.strip(), db_path)


def get_runpod_api_key(db_path: Optional[str] = None) -> str:
    """Get RunPod API key."""
    return get_setting("runpod.api_key", "", db_path) or ""


def set_runpod_api_key(api_key: str, db_path: Optional[str] = None) -> None:
    """Set RunPod API key."""
    set_setting("runpod.api_key", api_key.strip(), db_path)


def get_ollama_config(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Get Ollama configuration."""
    return get_setting("ollama", {}, db_path) or {}


def set_ollama_config(config: Dict[str, Any], db_path: Optional[str] = None) -> None:
    """Set Ollama configuration."""
    set_setting("ollama", config, db_path)


def get_proxy_config(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Get proxy configuration."""
    return get_setting("proxy", {}, db_path) or {}


def set_proxy_config(config: Dict[str, Any], db_path: Optional[str] = None) -> None:
    """Set proxy configuration."""
    set_setting("proxy", config, db_path)
