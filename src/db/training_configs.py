"""Training configuration storage using SQLite.

Replaces saved_configs/*.json with database storage.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .core import get_connection, init_db


def list_training_configs(train_target: Optional[str] = None, db_path: Optional[str] = None) -> List[str]:
    """List all saved training config names.

    Args:
        train_target: Optional filter by training target (e.g., "Runpod - Pod", "Local")
        db_path: Optional database path

    Returns:
        List of config names, sorted alphabetically
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    if train_target:
        # Filter by target, but also include configs without a target (backward compat)
        target_lower = train_target.strip().lower()
        cursor.execute(
            """
            SELECT name FROM training_configs
            WHERE train_target IS NULL 
               OR train_target = ''
               OR LOWER(train_target) LIKE ?
            ORDER BY name
        """,
            (f"{target_lower}%",),
        )
    else:
        cursor.execute("SELECT name FROM training_configs ORDER BY name")

    return [row["name"] for row in cursor.fetchall()]


def get_training_config(name: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get a training config by name.

    Args:
        name: Config name
        db_path: Optional database path

    Returns:
        Config dictionary or None if not found
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT config_json FROM training_configs WHERE name = ?", (name,))
    row = cursor.fetchone()

    if row is None:
        return None

    try:
        return json.loads(row["config_json"])
    except (json.JSONDecodeError, TypeError):
        return None


def save_training_config(name: str, config: Dict[str, Any], db_path: Optional[str] = None) -> None:
    """Save a training config.

    Args:
        name: Config name (will be used as unique identifier)
        config: Config dictionary with 'hp', 'meta', etc.
        db_path: Optional database path
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Extract train_target from meta for filtering
    train_target = None
    try:
        meta = config.get("meta") or {}
        train_target = meta.get("train_target", "")
    except Exception:
        pass

    config_json = json.dumps(config, ensure_ascii=False, indent=2)

    cursor.execute(
        """
        INSERT INTO training_configs (name, config_json, train_target, updated_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(name) DO UPDATE SET
            config_json = excluded.config_json,
            train_target = excluded.train_target,
            updated_at = datetime('now')
    """,
        (name, config_json, train_target),
    )

    conn.commit()


def delete_training_config(name: str, db_path: Optional[str] = None) -> bool:
    """Delete a training config.

    Args:
        name: Config name to delete
        db_path: Optional database path

    Returns:
        True if config was deleted, False if it didn't exist
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM training_configs WHERE name = ?", (name,))
    conn.commit()

    return cursor.rowcount > 0


def rename_training_config(old_name: str, new_name: str, db_path: Optional[str] = None) -> Tuple[bool, str]:
    """Rename a training config.

    Args:
        old_name: Current config name
        new_name: New config name
        db_path: Optional database path

    Returns:
        Tuple of (success, message)
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Check if old config exists
    cursor.execute("SELECT id FROM training_configs WHERE name = ?", (old_name,))
    if cursor.fetchone() is None:
        return False, f"Config '{old_name}' not found"

    # Check if new name already exists
    cursor.execute("SELECT id FROM training_configs WHERE name = ?", (new_name,))
    if cursor.fetchone() is not None:
        return False, f"Config '{new_name}' already exists"

    cursor.execute(
        "UPDATE training_configs SET name = ?, updated_at = datetime('now') WHERE name = ?", (new_name, old_name)
    )
    conn.commit()

    return True, "OK"


def get_last_used_config(db_path: Optional[str] = None) -> Optional[str]:
    """Get the name of the last used training config.

    Returns:
        Config name or None
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT value FROM app_state WHERE key = 'last_used_config'")
    row = cursor.fetchone()

    return row["value"] if row else None


def set_last_used_config(name: str, db_path: Optional[str] = None) -> None:
    """Set the last used training config name.

    Args:
        name: Config name
        db_path: Optional database path
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO app_state (key, value)
        VALUES ('last_used_config', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
    """,
        (name,),
    )

    conn.commit()


def validate_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a training config structure.

    Args:
        config: Config dictionary to validate

    Returns:
        Tuple of (is_valid, message)
    """
    if not isinstance(config, dict):
        return False, "Config is not a dictionary"

    hp = config.get("hp")
    if not isinstance(hp, dict):
        return False, "Missing or invalid 'hp' section"

    required = ["base_model", "epochs", "lr", "bsz", "grad_accum", "max_steps", "output_dir"]
    missing = [k for k in required if not str(hp.get(k) or "").strip()]

    if missing:
        return False, f"Missing required hp fields: {', '.join(missing)}"

    if not (hp.get("hf_dataset_id") or hp.get("json_path")):
        return True, "Note: no dataset specified (hf_dataset_id/json_path)"

    return True, "OK"
