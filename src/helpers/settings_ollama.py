from __future__ import annotations

import json
import os
from typing import List

import httpx


def config_path(project_root: str | None = None) -> str:
    root = project_root or os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, "ollama_config.json")


def load_config(path: str | None = None) -> dict:
    p = path or config_path()
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def save_config(cfg: dict, path: str | None = None) -> None:
    p = path or config_path()
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


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
