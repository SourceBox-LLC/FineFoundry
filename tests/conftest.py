import pytest


@pytest.fixture(params=["asyncio"])  # Use only asyncio backend for anyio-based tests
def anyio_backend(request):
    return request.param
