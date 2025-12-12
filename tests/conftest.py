import pytest

import flet as ft


@pytest.fixture(params=["asyncio"])  # Use only asyncio backend for anyio-based tests
def anyio_backend(request):
    return request.param


@pytest.fixture
def offline_mode_sw() -> ft.Switch:
    return ft.Switch(value=False)
