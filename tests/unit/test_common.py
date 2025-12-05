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
