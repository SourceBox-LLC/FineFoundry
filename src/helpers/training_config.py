from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple


def saved_configs_dir(project_root: Optional[str] = None) -> str:
    """Return path to the saved_configs directory; create it if missing."""
    root = project_root or os.path.dirname(os.path.dirname(__file__))
    d = os.path.join(root, "saved_configs")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def list_saved_configs(dir_path: Optional[str] = None) -> List[str]:
    d = dir_path or saved_configs_dir()
    try:
        files = [f for f in os.listdir(d) if f.lower().endswith(".json")]
    except Exception:
        files = []
    return sorted(files)


def read_json_file(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def validate_config(conf: dict) -> Tuple[bool, str]:
    """Lightweight schema validation for saved training configs.
    Returns (ok, message). On ok=False, message explains the issue.
    """
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
    d = saved_configs_dir(project_root)
    return os.path.join(d, ".last_used_config")


def get_last_used_config_name(project_root: Optional[str] = None) -> Optional[str]:
    path = _last_used_config_path(project_root)
    try:
        with open(path, "r", encoding="utf-8") as f:
            name = f.read().strip()
            return name or None
    except Exception:
        return None


def set_last_used_config_name(name: str, project_root: Optional[str] = None) -> None:
    path = _last_used_config_path(project_root)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(name or "").strip())
    except Exception:
        pass
