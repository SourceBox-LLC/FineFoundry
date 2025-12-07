"""Ollama settings and API helpers.

Now uses SQLite database for storage. Maintains backward-compatible API.
"""

from __future__ import annotations

from typing import List

import httpx

# Import database functions
from db import init_db
from db.settings import get_ollama_config, set_ollama_config


def load_config(path: str | None = None) -> dict:
    """Load Ollama configuration from database."""
    try:
        init_db()
        return get_ollama_config() or {}
    except Exception:
        return {}


def save_config(cfg: dict, path: str | None = None) -> None:
    """Save Ollama configuration to database."""
    try:
        init_db()
        set_ollama_config(cfg or {})
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
