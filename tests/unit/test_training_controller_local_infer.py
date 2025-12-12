"""UI-level tests for Quick Local Inference wiring in training_controller.

We mock the local_infer_generate_text_helper to ensure:
- Prompt and slider values are forwarded correctly
- Empty prompts short-circuit before calling the helper
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

import flet as ft

import ui.tabs.training_controller as tc


class DummyPage:
    """Minimal stand-in for ft.Page.

    Provides only the attributes used by training_controller: run_task,
    snack_bar, open, and update.
    """

    def __init__(self) -> None:  # pragma: no cover - simple init
        self.snack_bar = None
        self.open_calls: List[Any] = []
        self.run_task_calls: List[Any] = []
        # training_controller expects page.overlay to be a list it can append FilePickers to
        self.overlay: List[Any] = []

    def run_task(self, coro):  # type: ignore[override]
        """Synchronously execute an async handler for tests."""
        self.run_task_calls.append(coro)
        # training_controller passes the async function itself
        if asyncio.iscoroutinefunction(coro):
            return asyncio.run(coro())
        # Fallback if a coroutine object is passed
        return asyncio.run(coro)  # type: ignore[arg-type]

    def open(self, snack_bar):  # type: ignore[override]
        self.snack_bar = snack_bar
        self.open_calls.append(snack_bar)

    def update(self):  # pragma: no cover
        return None


async def _noop_safe_update(page: Any) -> None:  # noqa: ARG001
    """Async no-op to replace helpers.common.safe_update in tests."""
    return None


def _build_training_tab(monkeypatch) -> Tuple[DummyPage, ft.Control, Dict[str, Any]]:
    page = DummyPage()

    # Avoid real UI updates inside tests
    monkeypatch.setattr(tc, "safe_update", _noop_safe_update)

    def section_title(*args, **kwargs):  # noqa: ARG001
        return ft.Text("section")

    def _mk_help_handler(msg: str):  # noqa: ARG001
        def _h(e=None):  # pragma: no cover
            return None

        return _h

    tab, train_state = tc.build_training_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
        offline_mode_sw=ft.Switch(value=False),
        _hf_cfg={},
        _runpod_cfg={},
        hf_token_tf=ft.TextField(),
        proxy_enable_cb=ft.Switch(),
        use_env_cb=ft.Switch(),
        proxy_url_tf=ft.TextField(),
    )
    return page, tab, train_state


def _find_control_by_label(root: ft.Control, label: str) -> ft.Control | None:
    # Depth-first search over control tree by .label attribute
    stack = [root]
    while stack:
        ctl = stack.pop()
        if hasattr(ctl, "label") and getattr(ctl, "label") == label:
            return ctl
        for child_attr in ("controls", "content"):
            child = getattr(ctl, child_attr, None)
            if isinstance(child, list):
                stack.extend(child)
            elif isinstance(child, ft.Control):
                stack.append(child)
    return None


def test_local_infer_empty_prompt_short_circuits(monkeypatch):
    """Empty prompt should not call local_infer_generate_text_helper."""

    page, tab, train_state = _build_training_tab(monkeypatch)

    # Ensure local_infer helper is patched
    calls: List[tuple] = []

    def fake_generate(*args, **kwargs):  # type: ignore[override]
        calls.append((args, kwargs))
        return "should not be called"

    monkeypatch.setattr(tc, "local_infer_generate_text_helper", fake_generate)

    # Find prompt field and clear it
    prompt_tf = _find_control_by_label(tab, "Quick local inference prompt")
    assert isinstance(prompt_tf, ft.TextField)
    prompt_tf.value = ""  # empty

    # Trigger on_local_infer_generate via button's on_click
    # Find the "Run Inference" button
    run_btn = None
    stack = [tab]
    while stack and run_btn is None:
        ctl = stack.pop()
        if isinstance(ctl, ft.ElevatedButton) and getattr(ctl, "text", "") == "Run Inference":
            run_btn = ctl
            break
        for child_attr in ("controls", "content"):
            child = getattr(ctl, child_attr, None)
            if isinstance(child, list):
                stack.extend(child)
            elif isinstance(child, ft.Control):
                stack.append(child)

    assert run_btn is not None
    assert callable(run_btn.on_click)

    # Call handler (which uses page.run_task internally)
    run_btn.on_click(None)

    # Helper should not have been called
    assert calls == []


def test_local_infer_forwards_prompt_and_sliders(monkeypatch):
    """Prompt and slider values should be forwarded to generate_text helper."""

    page, tab, train_state = _build_training_tab(monkeypatch)

    # Set up train_state so local_infer is considered ready
    train_state["local_infer"] = {
        "adapter_path": "/tmp/adapter",
        "base_model": "base-model",
        "model_loaded": True,
    }

    # Avoid early-return on adapter path existence check
    monkeypatch.setattr(tc.os.path, "isdir", lambda p: True)

    calls: List[tuple] = []

    def fake_generate(base_model, adapter_path, prompt, max_new_tokens, temperature, repetition_penalty):  # type: ignore[override]
        calls.append((base_model, adapter_path, prompt, max_new_tokens, temperature, repetition_penalty))
        return "ok"

    monkeypatch.setattr(tc, "local_infer_generate_text_helper", fake_generate)

    # Prompt
    prompt_tf = _find_control_by_label(tab, "Quick local inference prompt")
    assert isinstance(prompt_tf, ft.TextField)
    prompt_tf.value = "Hello world"

    # Find sliders by scanning the tree; identify them by their default
    # values from training_controller (0.7 temp, 256 max tokens, 1.15 rep).
    sliders: List[ft.Slider] = []
    stack = [tab]
    while stack:
        ctl = stack.pop()
        if isinstance(ctl, ft.Slider):
            sliders.append(ctl)
        for child_attr in ("controls", "content"):
            child = getattr(ctl, child_attr, None)
            if isinstance(child, list):
                stack.extend(child)
            elif isinstance(child, ft.Control):
                stack.append(child)

    temp_slider = max_tokens_slider = rep_slider = None
    for s in sliders:
        if s.value == 0.7:
            temp_slider = s
        elif s.value == 256:
            max_tokens_slider = s
        elif s.value == 1.15:
            rep_slider = s

    assert temp_slider is not None and max_tokens_slider is not None and rep_slider is not None

    temp_slider.value = 0.5
    max_tokens_slider.value = 300
    rep_slider.value = 1.25

    # Trigger handler via Run Inference button
    run_btn = None
    stack = [tab]
    while stack and run_btn is None:
        ctl = stack.pop()
        if isinstance(ctl, ft.ElevatedButton) and getattr(ctl, "text", "") == "Run Inference":
            run_btn = ctl
            break
        for child_attr in ("controls", "content"):
            child = getattr(ctl, child_attr, None)
            if isinstance(child, list):
                stack.extend(child)
            elif isinstance(child, ft.Control):
                stack.append(child)

    assert run_btn is not None
    assert callable(run_btn.on_click)
    run_btn.on_click(None)

    # Helper should have been called once with expected args
    assert len(calls) == 1
    base_model, adapter_path, prompt, max_new_tokens, temperature, repetition_penalty = calls[0]
    assert base_model == "base-model"
    assert adapter_path == "/tmp/adapter"
    assert prompt == "Hello world"
    assert max_new_tokens == 300
    assert temperature == 0.5
    assert repetition_penalty == 1.25
