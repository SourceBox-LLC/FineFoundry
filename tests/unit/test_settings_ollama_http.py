import pytest

from helpers import settings_ollama as so


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.mark.anyio
async def test_fetch_tags_calls_correct_endpoint(monkeypatch):
    seen = {}

    class DummyClient:
        def __init__(self, *_, **__):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            seen["url"] = url
            return _DummyResponse({"models": []})

    monkeypatch.setattr(so.httpx, "AsyncClient", DummyClient)

    out = await so.fetch_tags("http://127.0.0.1:11434")

    assert seen["url"].endswith("/api/tags")
    assert isinstance(out, dict)


@pytest.mark.anyio
async def test_chat_sends_model_and_messages(monkeypatch):
    seen = {}

    class DummyClient:
        def __init__(self, *_, **__):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json=None):  # type: ignore[override]
            seen["url"] = url
            seen["payload"] = json
            return _DummyResponse({"message": {"content": "ok"}})

    monkeypatch.setattr(so.httpx, "AsyncClient", DummyClient)

    msg = await so.chat(
        base_url="http://localhost:11434",
        model="my-model",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert msg == "ok"
    assert seen["url"].endswith("/api/chat")
    assert seen["payload"]["model"] == "my-model"
    assert seen["payload"]["messages"][0]["content"] == "hi"
