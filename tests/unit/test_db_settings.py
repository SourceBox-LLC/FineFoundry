"""Unit tests for db/settings.py.

Tests cover:
- get_setting() / set_setting() - basic CRUD
- get_all_settings() - bulk retrieval
- delete_setting() - deletion
- Convenience functions (HF token, RunPod key, Ollama, proxy)
- Edge cases
"""

import pytest

from db.core import init_db, close_all_connections, _DB_PATH_OVERRIDE
from db.settings import (
    get_setting,
    set_setting,
    get_all_settings,
    delete_setting,
    get_hf_token,
    set_hf_token,
    get_runpod_api_key,
    set_runpod_api_key,
    get_ollama_config,
    set_ollama_config,
    get_proxy_config,
    set_proxy_config,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = str(tmp_path / "test.db")
    _DB_PATH_OVERRIDE["path"] = db_path
    init_db(db_path)
    yield db_path
    close_all_connections()
    _DB_PATH_OVERRIDE.clear()


# =============================================================================
# get_setting() / set_setting() tests
# =============================================================================


class TestGetSetSetting:
    """Tests for basic get/set operations."""

    def test_set_and_get_string(self, temp_db):
        """Test setting and getting a string value."""
        set_setting("test.key", "test_value", db_path=temp_db)
        result = get_setting("test.key", db_path=temp_db)
        assert result == "test_value"

    def test_set_and_get_int(self, temp_db):
        """Test setting and getting an integer value."""
        set_setting("test.number", 42, db_path=temp_db)
        result = get_setting("test.number", db_path=temp_db)
        assert result == 42

    def test_set_and_get_float(self, temp_db):
        """Test setting and getting a float value."""
        set_setting("test.float", 3.14, db_path=temp_db)
        result = get_setting("test.float", db_path=temp_db)
        assert result == 3.14

    def test_set_and_get_bool(self, temp_db):
        """Test setting and getting a boolean value."""
        set_setting("test.enabled", True, db_path=temp_db)
        result = get_setting("test.enabled", db_path=temp_db)
        assert result is True

    def test_set_and_get_list(self, temp_db):
        """Test setting and getting a list value."""
        set_setting("test.items", [1, 2, 3], db_path=temp_db)
        result = get_setting("test.items", db_path=temp_db)
        assert result == [1, 2, 3]

    def test_set_and_get_dict(self, temp_db):
        """Test setting and getting a dict value."""
        data = {"key": "value", "nested": {"a": 1}}
        set_setting("test.config", data, db_path=temp_db)
        result = get_setting("test.config", db_path=temp_db)
        assert result == data

    def test_get_nonexistent_returns_default(self, temp_db):
        """Test getting nonexistent key returns default."""
        result = get_setting("nonexistent", default="fallback", db_path=temp_db)
        assert result == "fallback"

    def test_get_nonexistent_returns_none(self, temp_db):
        """Test getting nonexistent key returns None by default."""
        result = get_setting("nonexistent", db_path=temp_db)
        assert result is None

    def test_update_existing_setting(self, temp_db):
        """Test updating an existing setting."""
        set_setting("test.key", "original", db_path=temp_db)
        set_setting("test.key", "updated", db_path=temp_db)
        result = get_setting("test.key", db_path=temp_db)
        assert result == "updated"

    def test_set_null_value(self, temp_db):
        """Test setting null value."""
        set_setting("test.null", None, db_path=temp_db)
        result = get_setting("test.null", db_path=temp_db)
        assert result is None


# =============================================================================
# get_all_settings() tests
# =============================================================================


class TestGetAllSettings:
    """Tests for bulk retrieval."""

    def test_get_all_empty(self, temp_db):
        """Test getting all settings when empty."""
        result = get_all_settings(db_path=temp_db)
        assert result == {}

    def test_get_all_multiple(self, temp_db):
        """Test getting all settings with multiple values."""
        set_setting("key1", "value1", db_path=temp_db)
        set_setting("key2", 42, db_path=temp_db)
        set_setting("key3", True, db_path=temp_db)

        result = get_all_settings(db_path=temp_db)

        assert len(result) == 3
        assert result["key1"] == "value1"
        assert result["key2"] == 42
        assert result["key3"] is True


# =============================================================================
# delete_setting() tests
# =============================================================================


class TestDeleteSetting:
    """Tests for deletion."""

    def test_delete_existing(self, temp_db):
        """Test deleting an existing setting."""
        set_setting("to_delete", "value", db_path=temp_db)
        result = delete_setting("to_delete", db_path=temp_db)

        assert result is True
        assert get_setting("to_delete", db_path=temp_db) is None

    def test_delete_nonexistent(self, temp_db):
        """Test deleting a nonexistent setting."""
        result = delete_setting("nonexistent", db_path=temp_db)
        assert result is False


# =============================================================================
# Convenience function tests
# =============================================================================


class TestHfToken:
    """Tests for HF token functions."""

    def test_set_and_get_hf_token(self, temp_db):
        """Test setting and getting HF token."""
        set_hf_token("hf_test_token_123", db_path=temp_db)
        result = get_hf_token(db_path=temp_db)
        assert result == "hf_test_token_123"

    def test_get_hf_token_empty(self, temp_db):
        """Test getting HF token when not set."""
        result = get_hf_token(db_path=temp_db)
        assert result == ""

    def test_set_hf_token_strips_whitespace(self, temp_db):
        """Test that HF token is stripped."""
        set_hf_token("  hf_token  ", db_path=temp_db)
        result = get_hf_token(db_path=temp_db)
        assert result == "hf_token"


class TestRunpodApiKey:
    """Tests for RunPod API key functions."""

    def test_set_and_get_runpod_key(self, temp_db):
        """Test setting and getting RunPod API key."""
        set_runpod_api_key("rp_test_key_456", db_path=temp_db)
        result = get_runpod_api_key(db_path=temp_db)
        assert result == "rp_test_key_456"

    def test_get_runpod_key_empty(self, temp_db):
        """Test getting RunPod key when not set."""
        result = get_runpod_api_key(db_path=temp_db)
        assert result == ""

    def test_set_runpod_key_strips_whitespace(self, temp_db):
        """Test that RunPod key is stripped."""
        set_runpod_api_key("  rp_key  ", db_path=temp_db)
        result = get_runpod_api_key(db_path=temp_db)
        assert result == "rp_key"


class TestOllamaConfig:
    """Tests for Ollama config functions."""

    def test_set_and_get_ollama_config(self, temp_db):
        """Test setting and getting Ollama config."""
        config = {"host": "localhost", "port": 11434}
        set_ollama_config(config, db_path=temp_db)
        result = get_ollama_config(db_path=temp_db)
        assert result == config

    def test_get_ollama_config_empty(self, temp_db):
        """Test getting Ollama config when not set."""
        result = get_ollama_config(db_path=temp_db)
        assert result == {}


class TestProxyConfig:
    """Tests for proxy config functions."""

    def test_set_and_get_proxy_config(self, temp_db):
        """Test setting and getting proxy config."""
        config = {"http": "http://proxy:8080", "https": "https://proxy:8080"}
        set_proxy_config(config, db_path=temp_db)
        result = get_proxy_config(db_path=temp_db)
        assert result == config

    def test_get_proxy_config_empty(self, temp_db):
        """Test getting proxy config when not set."""
        result = get_proxy_config(db_path=temp_db)
        assert result == {}


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_unicode_key_and_value(self, temp_db):
        """Test unicode in key and value."""
        set_setting("日本語.キー", "日本語の値", db_path=temp_db)
        result = get_setting("日本語.キー", db_path=temp_db)
        assert result == "日本語の値"

    def test_special_characters_in_value(self, temp_db):
        """Test special characters in value."""
        value = 'String with "quotes" and\nnewlines'
        set_setting("special", value, db_path=temp_db)
        result = get_setting("special", db_path=temp_db)
        assert result == value

    def test_empty_string_value(self, temp_db):
        """Test empty string value."""
        set_setting("empty", "", db_path=temp_db)
        result = get_setting("empty", db_path=temp_db)
        assert result == ""

    def test_nested_key_format(self, temp_db):
        """Test nested key format (dot notation)."""
        set_setting("section.subsection.key", "value", db_path=temp_db)
        result = get_setting("section.subsection.key", db_path=temp_db)
        assert result == "value"

    def test_very_long_value(self, temp_db):
        """Test very long value."""
        long_value = "x" * 100000
        set_setting("long", long_value, db_path=temp_db)
        result = get_setting("long", db_path=temp_db)
        assert result == long_value

    def test_complex_nested_dict(self, temp_db):
        """Test complex nested dictionary."""
        data = {"level1": {"level2": {"level3": {"value": 42, "list": [1, 2, 3]}}}}
        set_setting("complex", data, db_path=temp_db)
        result = get_setting("complex", db_path=temp_db)
        assert result == data
