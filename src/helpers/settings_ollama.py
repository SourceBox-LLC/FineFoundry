from __future__ import annotations

import json
import os
from typing import List

import httpx


def config_path(project_root: str | None = None) -> str:
    root = project_root or os.path.dirname(os.path.dirname(__file__))
    # Unified app settings file shared by Hugging Face, Runpod, and Ollama
    return os.path.join(root, "ff_settings.json")


def _load_all(path: str | None = None) -> dict:
    p = path or config_path()
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        # Fallback: support legacy standalone ollama_config.json if present
        root = os.path.dirname(os.path.dirname(__file__))
        legacy = os.path.join(root, "ollama_config.json")
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                legacy_cfg = json.load(f) or {}
            return {"ollama": legacy_cfg}
        except Exception:
            return {}


def _save_all(cfg: dict, path: str | None = None) -> None:
    p = path or config_path()
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_config(path: str | None = None) -> dict:
    data = _load_all(path)
    section = data.get("ollama") or {}
    if not isinstance(section, dict):
        return {}
    return section


def save_config(cfg: dict, path: str | None = None) -> None:
    data = _load_all(path)
    if not isinstance(data, dict):
        data = {}
    data["ollama"] = cfg or {}
    _save_all(data, path)


async def fetch_tags(base_url: str) -> dict:
    url = f"{(base_url or '').rstrip('/')}/api/tags"
    async with httpx.AsyncClient(timeout=6.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()


async def chat(base_url: str, model: str, messages: List[dict]) -> str:
    url = f"{(base_url or '').rstrip('/')}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message") or {}
        content = msg.get("content") or data.get("content") or ""
        return str(content)
