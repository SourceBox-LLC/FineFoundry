"""Tests for helpers.local_docker.on_docker_pull.

These tests focus on early-return branches and command orchestration.
They mock out system dependencies (docker CLI, subprocess, httpx) and
use simple dummy Flet controls.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List

import helpers.local_docker as ld


# ---------------------------------------------------------------------------
# Dummy UI controls
# ---------------------------------------------------------------------------


@dataclass
class DummyTextField:
    value: str = ""


@dataclass
class DummyText:
    value: str = ""
    color: Any | None = None


class DummyListView:
    def __init__(self) -> None:
        self.controls: List[Any] = []


@dataclass
class DummyPlaceholder:
    visible: bool = True


class DummyPage:
    def __init__(self) -> None:
        self.snack_bar: Any | None = None
        self.open_calls: List[Any] = []

    def open(self, snack_bar):  # type: ignore[override]
        self.open_calls.append(snack_bar)


# Minimal COLORS / ICONS stand-ins -----------------------------------------


class DummyCOLORS:
    RED_400 = "red"
    RED = "red"
    GREEN_400 = "green"
    GREEN = "green"
    WHITE = "white"


class DummyICONS:
    ERROR_OUTLINE = "error_outline"
    ERROR = "error"
    WARNING = "warning"


async def _noop_safe_update(page: Any) -> None:  # noqa: ARG001
    """Async no-op to replace safe_update in tests."""
    return None


def _make_basic_args() -> dict[str, Any]:
    page = DummyPage()
    docker_image_tf = DummyTextField(value="my-image:latest")
    docker_status = DummyText()
    docker_log_timeline = DummyListView()
    docker_log_placeholder = DummyPlaceholder()
    args = dict(
        page=page,
        ICONS=DummyICONS,
        COLORS=DummyCOLORS,
        docker_image_tf=docker_image_tf,
        docker_status=docker_status,
        DEFAULT_DOCKER_IMAGE="default-image:latest",
        docker_log_timeline=docker_log_timeline,
        docker_log_placeholder=docker_log_placeholder,
    )
    return args


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_on_docker_pull_missing_docker_cli(monkeypatch):
    """If docker CLI is missing, status is set and function returns early."""

    args = _make_basic_args()

    # Patch safe_update and shutil.which
    monkeypatch.setattr(ld, "safe_update", _noop_safe_update)
    monkeypatch.setattr(ld.shutil, "which", lambda _: None)

    asyncio.run(ld.on_docker_pull(**args))

    # Should show a clear error and not attempt to pull
    assert "Docker CLI not found" in args["docker_status"].value
    assert args["docker_status"].color in (DummyCOLORS.RED_400, DummyCOLORS.RED)


def test_on_docker_pull_daemon_not_running(monkeypatch):
    """If docker daemon is not running, show a helpful error and snackbar."""

    args = _make_basic_args()

    monkeypatch.setattr(ld, "safe_update", _noop_safe_update)
    monkeypatch.setattr(ld.shutil, "which", lambda _: "/usr/bin/docker")

    # docker info fails (non-zero return code)
    class InfoResult:
        def __init__(self):
            self.returncode = 1
            self.stderr = "Docker daemon not responding"
            self.stdout = ""

    def fake_run(cmd, capture_output=False, text=False, timeout=None):  # noqa: ARG002
        assert cmd[:2] == ["docker", "info"]
        return InfoResult()

    monkeypatch.setattr(ld.subprocess, "run", fake_run)

    asyncio.run(ld.on_docker_pull(**args))

    # Status message about Docker not running
    assert "Docker is not running" in args["docker_status"].value
    # A snackbar should have been opened at least once
    assert args["page"].open_calls


def test_on_docker_pull_image_already_present(monkeypatch):
    """If image is already present locally, do not perform a pull."""

    args = _make_basic_args()

    monkeypatch.setattr(ld, "safe_update", _noop_safe_update)
    monkeypatch.setattr(ld.shutil, "which", lambda _: "/usr/bin/docker")

    # First docker info succeeds, then image inspect says image exists
    class Result:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, capture_output=False, text=False, timeout=None):  # noqa: ARG002
        if cmd[:2] == ["docker", "info"]:
            return Result(returncode=0)
        if cmd[:3] == ["docker", "image", "inspect"]:
            payload = [
                {
                    "RepoTags": ["my-image:latest"],
                    "Id": "sha256:1234567890abcdef",
                    "Created": "2024-01-01T12:00:00Z",
                }
            ]
            return Result(returncode=0, stdout=json_dumps(payload))
        raise AssertionError(f"Unexpected docker command: {cmd}")

    from json import dumps as json_dumps

    monkeypatch.setattr(ld.subprocess, "run", fake_run)

    # Prevent actual pull subprocess from being spawned
    async def fake_create_subprocess_exec(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("docker pull should not be executed when image exists")

    monkeypatch.setattr(ld.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    asyncio.run(ld.on_docker_pull(**args))

    assert "Image already present locally" in args["docker_status"].value
    # No docker pull should have been attempted, as our fake would raise


def test_on_docker_pull_successful_pull(monkeypatch):
    """Happy path where docker pull completes successfully (rc=0)."""

    args = _make_basic_args()

    monkeypatch.setattr(ld, "safe_update", _noop_safe_update)
    monkeypatch.setattr(ld.shutil, "which", lambda _: "/usr/bin/docker")

    # docker info passes, image inspect reports missing image
    class Result:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, capture_output=False, text=False, timeout=None):  # noqa: ARG002
        if cmd[:2] == ["docker", "info"]:
            return Result(returncode=0)
        if cmd[:3] == ["docker", "image", "inspect"]:
            return Result(returncode=1, stderr="No such image")
        # Any other docker run (e.g., image ls) should not be called in success path
        return Result(returncode=0)

    monkeypatch.setattr(ld.subprocess, "run", fake_run)

    # Fake docker pull subprocess
    class DummyProc:
        def __init__(self, rc: int) -> None:
            self._rc = rc
            self.stdout = None
            self.stderr = None

        async def wait(self) -> int:  # pragma: no cover - trivial
            return self._rc

    async def fake_create_subprocess_exec(*args, **kwargs):  # noqa: ARG001
        return DummyProc(rc=0)

    monkeypatch.setattr(ld.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    asyncio.run(ld.on_docker_pull(**args))

    assert "Pulled successfully" in args["docker_status"].value
    # A success snackbar should have been opened
    assert args["page"].open_calls
