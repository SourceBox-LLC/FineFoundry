from pathlib import Path

from helpers import settings_ollama as so


def test_config_path_uses_project_root():
    root = "/tmp/myproj"
    path = so.config_path(root)
    assert path.endswith("ff_settings.json")
    assert str(Path(path).parent) == root


def test_save_and_load_roundtrip(tmp_path):
    cfg = {"enabled": True, "base_url": "http://127.0.0.1:11434", "default_model": "llama2"}
    p = tmp_path / "ff_settings.json"

    so.save_config(cfg, path=str(p))
    loaded = so.load_config(path=str(p))

    assert loaded.get("enabled") is True
    assert loaded.get("base_url") == "http://127.0.0.1:11434"
    assert loaded.get("default_model") == "llama2"
