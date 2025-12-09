import asyncio

from helpers import common


class DummyPage:
    def __init__(self):
        self.updated = False

    async def update_async(self):  # safe_update prefers update_async when available
        self.updated = True


async def _call_safe_update(page):
    await common.safe_update(page)


def test_safe_update_calls_update_event_loop():
    page = DummyPage()
    # Run the async helper in a fresh event loop
    asyncio.run(_call_safe_update(page))
    assert page.updated is True


def test_set_terminal_title_does_not_raise():
    # We can't easily assert the terminal title in a test runner, but we can
    # at least ensure the function is callable and does not throw.
    common.set_terminal_title("TEST-TITLE")


def test_set_terminal_title_with_special_chars():
    """Test terminal title with special characters."""
    common.set_terminal_title("Title with 'quotes' and \"double\"")


def test_set_terminal_title_with_unicode():
    """Test terminal title with unicode."""
    common.set_terminal_title("日本語タイトル")


def test_set_terminal_title_empty():
    """Test terminal title with empty string."""
    common.set_terminal_title("")


class DummyPageSync:
    """Page without update_async (older Flet versions)."""

    def __init__(self):
        self.updated = False

    def update(self):
        self.updated = True


async def _call_safe_update_sync(page):
    await common.safe_update(page)


def test_safe_update_fallback_to_sync():
    """Test safe_update falls back to sync update."""
    page = DummyPageSync()
    asyncio.run(_call_safe_update_sync(page))
    assert page.updated is True


def test_module_exports():
    """Test that __all__ exports are correct."""
    assert "safe_update" in common.__all__
    assert "set_terminal_title" in common.__all__
