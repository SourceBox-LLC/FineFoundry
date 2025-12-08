"""Training configuration helpers.

Now uses SQLite database for storage. Maintains backward-compatible API.
"""

from __future__ import annotations

import json
import os
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
    """Return path to the saved_configs directory.

    Note: This is kept for backward compatibility but configs are now stored in SQLite.
    """
    root = project_root or os.path.dirname(os.path.dirname(__file__))
    d = os.path.join(root, "saved_configs")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def list_saved_configs(dir_path: Optional[str] = None) -> List[str]:
    """List all saved training config names.

    Now reads from SQLite database instead of filesystem.
    """
    try:
        init_db()
        return db_list_configs()
    except Exception:
        # Fallback to filesystem for backward compatibility
        d = dir_path or saved_configs_dir()
        try:
            files = [f for f in os.listdir(d) if f.lower().endswith(".json")]
        except Exception:
            files = []
        return sorted(files)


def read_json_file(path: str) -> Optional[dict]:
    """Read a training config.

    If path looks like a config name (ends with .json), tries database first.
    Otherwise falls back to filesystem.
    """
    # Extract config name from path
    name = os.path.basename(path)

    try:
        init_db()
        config = db_get_config(name)
        if config is not None:
            return config
    except Exception:
        pass

    # Fallback to filesystem
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_config(name: str, config: dict) -> None:
    """Save a training config to the database."""
    try:
        init_db()
        db_save_config(name, config)
    except Exception:
        # Fallback to filesystem
        d = saved_configs_dir()
        path = os.path.join(d, name if name.endswith(".json") else f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


def delete_config(name: str) -> bool:
    """Delete a training config."""
    try:
        init_db()
        return db_delete_config(name)
    except Exception:
        return False


def rename_config(old_name: str, new_name: str) -> Tuple[bool, str]:
    """Rename a training config."""
    try:
        init_db()
        return db_rename_config(old_name, new_name)
    except Exception as e:
        return False, str(e)


def validate_config(conf: dict) -> Tuple[bool, str]:
    """Lightweight schema validation for saved training configs.
    Returns (ok, message). On ok=False, message explains the issue.
    """
    try:
        return db_validate_config(conf)
    except Exception:
        # Inline validation as fallback
        if not isinstance(conf, dict):
            return False, "Config is not a JSON object."
        hp = conf.get("hp")
        if not isinstance(hp, dict):
            return False, "Missing or invalid 'hp' section."
        required = ["base_model", "epochs", "lr", "bsz", "grad_accum", "max_steps", "output_dir"]
        missing = [k for k in required if not str(hp.get(k) or "").strip()]
        if missing:
            return False, f"Missing required hp fields: {', '.join(missing)}"
        if not (hp.get("hf_dataset_id") or hp.get("json_path")):
            return True, "Note: no dataset specified (hf_dataset_id/json_path)."
        return True, "OK"


def _last_used_config_path(project_root: Optional[str] = None) -> str:
    """Legacy path for last used config marker."""
    d = saved_configs_dir(project_root)
    return os.path.join(d, ".last_used_config")


def get_last_used_config_name(project_root: Optional[str] = None) -> Optional[str]:
    """Get the name of the last used training config."""
    try:
        init_db()
        return db_get_last_used()
    except Exception:
        # Fallback to filesystem
        path = _last_used_config_path(project_root)
        try:
            with open(path, "r", encoding="utf-8") as f:
                name = f.read().strip()
                return name or None
        except Exception:
            return None


def set_last_used_config_name(name: str, project_root: Optional[str] = None) -> None:
    """Set the last used training config name."""
    try:
        init_db()
        db_set_last_used(name)
    except Exception:
        # Fallback to filesystem
        path = _last_used_config_path(project_root)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(name or "").strip())
        except Exception:
            pass
