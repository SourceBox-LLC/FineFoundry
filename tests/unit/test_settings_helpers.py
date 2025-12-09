"""Unit tests for helpers/settings.py.

Tests cover:
- Loading and saving settings
- Convenience functions for HF, RunPod, Ollama, Proxy configs
"""

import sys
from pathlib import Path
from unittest.mock import patch


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from helpers.settings import (
    load_settings,
    save_settings,
    load_hf_config,
    save_hf_config,
    load_runpod_config,
    save_runpod_config,
    load_ollama_config,
    save_ollama_config,
    load_proxy_config,
    save_proxy_config,
)


class TestLoadSettings:
    """Tests for load_settings function."""

    def test_load_settings_returns_dict(self):
        """Test load_settings returns a dictionary."""
        with patch("helpers.settings.init_db"):
            with patch("helpers.settings.get_all_settings", return_value={}):
                result = load_settings()
                assert isinstance(result, dict)

    def test_load_settings_nested_structure(self):
        """Test load_settings creates nested structure from dot notation."""
        with patch("helpers.settings.init_db"):
            with patch("helpers.settings.get_all_settings", return_value={
                "huggingface.token": "hf_token",
                "runpod.api_key": "rp_key",
            }):
                result = load_settings()
                assert result["huggingface"]["token"] == "hf_token"
                assert result["runpod"]["api_key"] == "rp_key"

    def test_load_settings_flat_keys(self):
        """Test load_settings handles flat keys."""
        with patch("helpers.settings.init_db"):
            with patch("helpers.settings.get_all_settings", return_value={
                "simple_key": "value",
            }):
                result = load_settings()
                assert result["simple_key"] == "value"


class TestSaveSettings:
    """Tests for save_settings function."""

    def test_save_settings_nested(self):
        """Test save_settings with nested dictionary."""
        with patch("helpers.settings.init_db"):
            with patch("helpers.settings.set_setting") as mock_set:
                save_settings({
                    "huggingface": {"token": "test_token"},
                    "runpod": {"api_key": "test_key"},
                })
                mock_set.assert_any_call("huggingface.token", "test_token")
                mock_set.assert_any_call("runpod.api_key", "test_key")

    def test_save_settings_flat(self):
        """Test save_settings with flat values."""
        with patch("helpers.settings.init_db"):
            with patch("helpers.settings.set_setting") as mock_set:
                save_settings({"flat_key": "flat_value"})
                mock_set.assert_called_with("flat_key", "flat_value")


class TestHFConfig:
    """Tests for HuggingFace config functions."""

    def test_load_hf_config(self):
        """Test loading HF config."""
        with patch("helpers.settings.init_db"):
            with patch("helpers.settings.get_hf_token", return_value="test_token"):
                result = load_hf_config()
                assert result == {"token": "test_token"}

    def test_save_hf_config(self):
        """Test saving HF config."""
        with patch("helpers.settings.init_db"):
            with patch("helpers.settings.set_hf_token") as mock_set:
                save_hf_config({"token": "  new_token  "})
                mock_set.assert_called_with("new_token")

    def test_save_hf_config_empty(self):
        """Test saving empty HF config."""
        with patch("helpers.settings.init_db"):
            with patch("helpers.settings.set_hf_token") as mock_set:
                save_hf_config({})
                mock_set.assert_called_with("")


class TestRunPodConfig:
    """Tests for RunPod config functions."""

    def test_load_runpod_config(self):
        """Test loading RunPod config."""
        with patch("helpers.settings.init_db"):
            with patch("helpers.settings.get_runpod_api_key", return_value="api_key"):
                result = load_runpod_config()
                assert result == {"api_key": "api_key"}

    def test_save_runpod_config(self):
        """Test saving RunPod config."""
        with patch("helpers.settings.init_db"):
            with patch("helpers.settings.set_runpod_api_key") as mock_set:
                save_runpod_config({"api_key": "  key123  "})
                mock_set.assert_called_with("key123")


class TestOllamaConfig:
    """Tests for Ollama config functions."""

    def test_load_ollama_config(self):
        """Test loading Ollama config."""
        expected = {"host": "localhost", "port": 11434}
        with patch("helpers.settings.init_db"):
            with patch("helpers.settings.get_ollama_config", return_value=expected):
                result = load_ollama_config()
                assert result == expected

    def test_save_ollama_config(self):
        """Test saving Ollama config."""
        config = {"host": "remote", "port": 8080}
        with patch("helpers.settings.init_db"):
            with patch("helpers.settings.set_ollama_config") as mock_set:
                save_ollama_config(config)
                mock_set.assert_called_with(config)


class TestProxyConfig:
    """Tests for Proxy config functions."""

    def test_load_proxy_config(self):
        """Test loading Proxy config."""
        expected = {"enabled": True, "host": "proxy.example.com"}
        with patch("helpers.settings.init_db"):
            with patch("helpers.settings.get_proxy_config", return_value=expected):
                result = load_proxy_config()
                assert result == expected

    def test_save_proxy_config(self):
        """Test saving Proxy config."""
        config = {"enabled": False}
        with patch("helpers.settings.init_db"):
            with patch("helpers.settings.set_proxy_config") as mock_set:
                save_proxy_config(config)
                mock_set.assert_called_with(config)
