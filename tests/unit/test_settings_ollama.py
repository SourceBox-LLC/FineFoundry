"""Tests for Ollama settings helper.

Now uses SQLite database for storage.
"""

import os
import tempfile

from helpers import settings_ollama as so
from db import init_db
from db.core import _DB_PATH_OVERRIDE


def test_load_config_returns_dict():
    """load_config should return a dict (empty if no config set)."""
    # Use a temp database
    with tempfile.TemporaryDirectory() as tmpdir:
        _DB_PATH_OVERRIDE["path"] = os.path.join(tmpdir, "test.db")
        try:
            init_db()
            result = so.load_config()
            assert isinstance(result, dict)
        finally:
            _DB_PATH_OVERRIDE.pop("path", None)


def test_save_and_load_roundtrip():
    """save_config and load_config should roundtrip correctly."""
    cfg = {"enabled": True, "base_url": "http://127.0.0.1:11434", "default_model": "llama2"}

    with tempfile.TemporaryDirectory() as tmpdir:
        _DB_PATH_OVERRIDE["path"] = os.path.join(tmpdir, "test.db")
        try:
            init_db()
            so.save_config(cfg)
            loaded = so.load_config()

            assert loaded.get("enabled") is True
            assert loaded.get("base_url") == "http://127.0.0.1:11434"
            assert loaded.get("default_model") == "llama2"
        finally:
            _DB_PATH_OVERRIDE.pop("path", None)
