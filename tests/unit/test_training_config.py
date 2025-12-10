"""Tests for helpers.training_config.

These ensure the helpers call into the DB layer correctly and init_db()
gets invoked where expected.
"""

from __future__ import annotations

from typing import Any, List

import helpers.training_config as tc


class DummyDB:
    def __init__(self) -> None:
        self.calls: List[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.last_used: str | None = None

    def _rec(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    # db API mirrors
    def init_db(self) -> None:  # pragma: no cover - trivial
        self._rec("init_db")

    def list_training_configs(self) -> list[str]:
        self._rec("list_training_configs")
        return ["cfg-a", "cfg-b"]

    def get_training_config(self, name: str) -> dict | None:
        self._rec("get_training_config", name)
        if name == "exists":
            return {"name": name}
        return None

    def save_training_config(self, name: str, conf: dict) -> None:
        self._rec("save_training_config", name, conf)

    def delete_training_config(self, name: str) -> bool:
        self._rec("delete_training_config", name)
        return name == "exists"

    def rename_training_config(self, old: str, new: str):
        self._rec("rename_training_config", old, new)
        return True, "ok"

    def validate_config(self, conf: dict):
        self._rec("validate_config", conf)
        return True, "ok"

    def get_last_used_config(self) -> str | None:
        self._rec("get_last_used_config")
        return self.last_used

    def set_last_used_config(self, name: str) -> None:
        self._rec("set_last_used_config", name)
        self.last_used = name


def _patch_db(monkeypatch) -> DummyDB:
    db = DummyDB()
    monkeypatch.setattr(tc, "init_db", db.init_db)
    monkeypatch.setattr(tc, "db_list_configs", db.list_training_configs)
    monkeypatch.setattr(tc, "db_get_config", db.get_training_config)
    monkeypatch.setattr(tc, "db_save_config", db.save_training_config)
    monkeypatch.setattr(tc, "db_delete_config", db.delete_training_config)
    monkeypatch.setattr(tc, "db_rename_config", db.rename_training_config)
    monkeypatch.setattr(tc, "db_get_last_used", db.get_last_used_config)
    monkeypatch.setattr(tc, "db_set_last_used", db.set_last_used_config)
    monkeypatch.setattr(tc, "db_validate_config", db.validate_config)
    return db


def test_list_saved_configs_calls_db(monkeypatch):
    db = _patch_db(monkeypatch)
    names = tc.list_saved_configs()
    assert names == ["cfg-a", "cfg-b"]
    # init_db + list_training_configs recorded
    assert any(name == "init_db" for name, *_ in db.calls)
    assert any(name == "list_training_configs" for name, *_ in db.calls)


def test_read_and_save_config(monkeypatch):
    db = _patch_db(monkeypatch)
    conf = tc.read_json_file("exists")
    assert conf == {"name": "exists"}

    tc.save_config("new", {"x": 1})
    names = [name for name, *_ in db.calls]
    assert "get_training_config" in names
    assert "save_training_config" in names


def test_delete_and_rename_config(monkeypatch):
    db = _patch_db(monkeypatch)
    ok = tc.delete_config("exists")
    assert ok is True

    renamed, msg = tc.rename_config("old", "new")
    assert renamed is True
    assert msg == "ok"
    names = [name for name, *_ in db.calls]
    assert "delete_training_config" in names
    assert "rename_training_config" in names


def test_validate_and_last_used(monkeypatch):
    db = _patch_db(monkeypatch)

    valid, msg = tc.validate_config({"a": 1})
    assert valid is True
    assert msg == "ok"

    assert tc.get_last_used_config_name() is None
    tc.set_last_used_config_name("cfg-a")
    assert tc.get_last_used_config_name() == "cfg-a"
    names = [name for name, *_ in db.calls]
    assert "validate_config" in names
    assert "get_last_used_config" in names
    assert "set_last_used_config" in names
