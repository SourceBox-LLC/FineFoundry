"""Training configuration helpers.

Uses SQLite database for storage. No filesystem fallback.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# Import database functions
from db import (
    list_training_configs as db_list_configs,
    get_training_config as db_get_config,
    save_training_config as db_save_config,
    delete_training_config as db_delete_config,
    rename_training_config as db_rename_config,
    get_last_used_config as db_get_last_used,
    set_last_used_config as db_set_last_used,
    validate_config as db_validate_config,
    init_db,
)


def saved_configs_dir(project_root: Optional[str] = None) -> str:
    """Legacy function - returns empty string. Configs are stored in database."""
    return ""


def list_saved_configs(dir_path: Optional[str] = None) -> List[str]:
    """List all saved training config names from database."""
    init_db()
    return db_list_configs()


def read_json_file(name: str) -> Optional[dict]:
    """Read a training config from database by name."""
    init_db()
    return db_get_config(name)


def save_config(name: str, config: dict) -> None:
    """Save a training config to the database."""
    init_db()
    db_save_config(name, config)


def delete_config(name: str) -> bool:
    """Delete a training config from database."""
    init_db()
    return db_delete_config(name)


def rename_config(old_name: str, new_name: str) -> Tuple[bool, str]:
    """Rename a training config in database."""
    init_db()
    return db_rename_config(old_name, new_name)


def validate_config(conf: dict) -> Tuple[bool, str]:
    """Validate a training config schema."""
    return db_validate_config(conf)


def get_last_used_config_name(project_root: Optional[str] = None) -> Optional[str]:
    """Get the name of the last used training config from database."""
    init_db()
    return db_get_last_used()


def set_last_used_config_name(name: str, project_root: Optional[str] = None) -> None:
    """Set the last used training config name in database."""
    init_db()
    db_set_last_used(name)
