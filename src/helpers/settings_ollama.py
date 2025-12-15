"""Ollama settings and API helpers.

Uses SQLite database for storage.
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
    base = (base_url or "").rstrip("/")
    if not base:
        raise RuntimeError("Missing Ollama base URL")
    if not (model or "").strip():
        raise RuntimeError("Missing Ollama model")

    timeout = httpx.Timeout(120.0, connect=10.0)

    def _messages_to_prompt(msgs: List[dict]) -> str:
        lines: list[str] = []
        for m in msgs or []:
            role = str((m or {}).get("role") or "").strip().lower() or "user"
            content = str((m or {}).get("content") or "")
            if not content.strip():
                continue
            if role == "system":
                lines.append(f"System:\n{content}")
            elif role == "assistant":
                lines.append(f"Assistant:\n{content}")
            else:
                lines.append(f"User:\n{content}")
        lines.append("Assistant:\n")
        return "\n\n".join(lines).strip()

    async def _post_json(url: str, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = ""
                try:
                    body = (r.text or "").strip()
                except Exception:
                    body = ""
                raise RuntimeError(f"Ollama HTTP {r.status_code} calling {url}: {body or str(e)}") from e
            try:
                return r.json()
            except Exception as e:
                txt = ""
                try:
                    txt = (r.text or "").strip()
                except Exception:
                    txt = ""
                raise RuntimeError(f"Invalid JSON response from Ollama ({url}): {txt}") from e

    # Prefer /api/chat, but fall back to /api/generate for older servers/models.
    chat_url = f"{base}/api/chat"
    chat_payload = {"model": model, "messages": messages, "stream": False}
    try:
        data = await _post_json(chat_url, chat_payload)
        msg = data.get("message")
        if isinstance(msg, dict):
            content = msg.get("content") or ""
        else:
            content = data.get("content") or ""
        content = str(content or "").strip()
        if content:
            return content
        raise RuntimeError("Empty response from Ollama /api/chat")
    except Exception as chat_exc:
        gen_url = f"{base}/api/generate"
        gen_payload = {"model": model, "prompt": _messages_to_prompt(messages), "stream": False}
        try:
            data = await _post_json(gen_url, gen_payload)
            content = str(data.get("response") or data.get("content") or "").strip()
            if content:
                return content
            raise RuntimeError("Empty response from Ollama /api/generate")
        except Exception as gen_exc:
            raise RuntimeError(f"Ollama request failed: {gen_exc}") from chat_exc
