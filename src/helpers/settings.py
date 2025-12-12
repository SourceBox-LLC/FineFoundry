"""Application settings helpers.

Provides a unified interface for loading and saving settings.
Uses SQLite database for storage.
"""

from __future__ import annotations

from typing import Any, Dict

from db import init_db
from db.settings import (
    get_setting,
    set_setting,
    get_all_settings,
    get_hf_token,
    set_hf_token,
    get_runpod_api_key,
    set_runpod_api_key,
    get_ollama_config,
    set_ollama_config,
    get_proxy_config,
    set_proxy_config,
)


def load_settings() -> Dict[str, Any]:
    """Load all settings as a nested dictionary.

    Returns settings in the format:
    {
        "huggingface": {"token": "..."},
        "runpod": {"api_key": "..."},
        "ollama": {...},
        "proxy": {...}
    }
    """
    init_db()

    # Get all settings and reconstruct nested structure
    all_settings = get_all_settings()

    result: Dict[str, Any] = {}
    for key, value in all_settings.items():
        parts = key.split(".", 1)
        if len(parts) == 2:
            section, subkey = parts
            if section not in result:
                result[section] = {}
            result[section][subkey] = value
        else:
            result[key] = value

    return result


def save_settings(data: Dict[str, Any]) -> None:
    """Save settings from a nested dictionary.

    Accepts settings in the format:
    {
        "huggingface": {"token": "..."},
        "runpod": {"api_key": "..."},
        ...
    }
    """
    init_db()

    for section, values in data.items():
        if isinstance(values, dict):
            for key, value in values.items():
                set_setting(f"{section}.{key}", value)
        else:
            set_setting(section, values)


# Convenience functions for common settings


def load_hf_config() -> Dict[str, str]:
    """Load Hugging Face configuration."""
    init_db()
    return {"token": get_hf_token()}


def save_hf_config(cfg: Dict[str, str]) -> None:
    """Save Hugging Face configuration."""
    init_db()
    token = (cfg.get("token") or "").strip()
    set_hf_token(token)


def load_runpod_config() -> Dict[str, str]:
    """Load RunPod configuration."""
    init_db()
    return {"api_key": get_runpod_api_key()}


def save_runpod_config(cfg: Dict[str, str]) -> None:
    """Save RunPod configuration."""
    init_db()
    api_key = (cfg.get("api_key") or "").strip()
    set_runpod_api_key(api_key)


def load_ollama_config() -> Dict[str, Any]:
    """Load Ollama configuration."""
    init_db()
    return get_ollama_config()


def save_ollama_config(cfg: Dict[str, Any]) -> None:
    """Save Ollama configuration."""
    init_db()
    set_ollama_config(cfg)


def load_proxy_config() -> Dict[str, Any]:
    """Load proxy configuration."""
    init_db()
    return get_proxy_config()


def save_proxy_config(cfg: Dict[str, Any]) -> None:
    """Save proxy configuration."""
    init_db()
    set_proxy_config(cfg)


def load_offline_mode() -> bool:
    init_db()
    value = get_setting("app.offline_mode", True)
    try:
        return bool(value)
    except Exception:
        return True


def save_offline_mode(enabled: bool) -> None:
    init_db()
    set_setting("app.offline_mode", bool(enabled))
