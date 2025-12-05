"""Inference tab controller for FineFoundry.

This module builds the Inference tab controls and wires up all inference
and chat handlers, keeping `src/main.py` smaller.

Layout composition still lives in `tab_inference.py`; this module focuses
on behavior and state wiring.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict

import flet as ft

from helpers.common import safe_update
from helpers.logging_config import get_logger
from helpers.theme import ACCENT_COLOR, BORDER_BASE, COLORS, ICONS
from helpers.ui import WITH_OPACITY
from helpers.local_inference import generate_text as local_infer_generate_text_helper
from ui.tabs.tab_inference import build_inference_tab


logger = get_logger(__name__)


def build_inference_tab_with_logic(
    page: ft.Page,
    *,
    section_title,
    _mk_help_handler,
    train_state: Dict[str, Any],
) -> ft.Control:
    """Build the Inference tab and attach all related handlers.

    This function mirrors the previous inline Inference tab setup from
    `main.py`, but keeps the behavior localized to this module.
    """

    # Status + meta
    infer_status = ft.Text(
        "Pick a base model and adapter directory, or import the latest local training run.",
        color=WITH_OPACITY(0.6, BORDER_BASE),
    )
    infer_meta = ft.Text(
        "",
        size=12,
        color=WITH_OPACITY(0.65, BORDER_BASE),
    )

    # Model + adapter inputs
    infer_base_model_tf = ft.TextField(
        label="Base model",
        value="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        width=520,
        dense=True,
    )
    infer_adapter_dir_tf = ft.TextField(
        label="Adapter directory",
        width=520,
        dense=True,
        hint_text="Folder containing adapter (e.g. /path/to/outputs/run/adapter)",
    )

    # Directory picker for adapter dir
    async def on_infer_validate_adapter(e=None):  # noqa: ARG001
        """Validate adapter directory immediately when selected.

        Shows a loading indicator and snackbar, and unlocks the prompt section
        only when the adapter path looks valid.
        """

        def _looks_like_adapter_dir(adapter_path: str) -> bool:
            """Heuristic check that a directory looks like a LoRA adapter folder."""
            try:
                cfg_path = os.path.join(adapter_path, "adapter_config.json")
                if os.path.isfile(cfg_path):
                    return True
                for name in os.listdir(adapter_path):
                    if name.endswith(".safetensors") or name.endswith(".bin"):
                        return True
            except Exception:  # pragma: no cover - defensive
                return False
            return False

        adapter_path = (infer_adapter_dir_tf.value or "").strip()
        base_model_name = (
            infer_base_model_tf.value or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
        ).strip()
        try:
            infer_busy_ring.visible = True
            await safe_update(page)
        except Exception:  # pragma: no cover - UI best-effort
            pass
        if (not adapter_path) or (not os.path.isdir(adapter_path)):
            infer_status.value = (
                "Adapter directory is missing or invalid. Pick a valid folder containing an adapter."
            )
            try:
                infer_status.color = getattr(
                    COLORS,
                    "RED_400",
                    getattr(COLORS, "RED", None),
                )
            except Exception:
                pass
            _set_infer_controls_enabled(False)
            try:
                page.snack_bar = ft.SnackBar(
                    ft.Text("Invalid adapter directory. Please choose a valid adapter folder."),
                )
                page.open(page.snack_bar)
            except Exception:
                pass
            await safe_update(page)
            try:
                infer_busy_ring.visible = False
                await safe_update(page)
            except Exception:
                pass
            return
        if not _looks_like_adapter_dir(adapter_path):
            try:
                logger.warning(
                    "Inference adapter directory %r does not contain adapter_config.json or weight files; refusing to validate.",
                    adapter_path,
                )
            except Exception:
                pass
            infer_status.value = (
                "Adapter directory doesn't look like a valid LoRA adapter. "
                "Select the adapter folder from a completed fine-tuning run."
            )
            try:
                infer_status.color = getattr(
                    COLORS,
                    "RED_400",
                    getattr(COLORS, "RED", None),
                )
            except Exception:
                pass
            _set_infer_controls_enabled(False)
            try:
                page.snack_bar = ft.SnackBar(
                    ft.Text("Adapter directory doesn't look like a valid LoRA adapter."),
                )
                page.open(page.snack_bar)
            except Exception:
                pass
            await safe_update(page)
            try:
                infer_busy_ring.visible = False
                await safe_update(page)
            except Exception:
                pass
            return
        # Looks valid – unlock controls and surface success
        infer_status.value = "Adapter directory validated. Ready for inference."
        try:
            infer_status.color = WITH_OPACITY(0.6, BORDER_BASE)
        except Exception:
            pass
        try:
            infer_meta.value = f"Adapter: {adapter_path} • Base model: {base_model_name}"
        except Exception:
            pass
        _set_infer_controls_enabled(True)
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Adapter directory validated."))
            page.open(page.snack_bar)
        except Exception:
            pass
        try:
            infer_busy_ring.visible = False
            await safe_update(page)
        except Exception:
            pass

    def _on_infer_adapter_picked(e: ft.FilePickerResultEvent) -> None:
        try:
            sel = getattr(e, "path", None) or ""
            if sel:
                infer_adapter_dir_tf.value = sel
                try:
                    if hasattr(page, "run_task"):
                        page.run_task(on_infer_validate_adapter)
                except Exception:
                    pass
            page.update()
        except Exception:
            pass

    infer_dir_picker = ft.FilePicker(on_result=_on_infer_adapter_picked)
    try:
        page.overlay.append(infer_dir_picker)
    except Exception:
        pass
    infer_browse_btn = ft.OutlinedButton(
        "Browse…",
        icon=getattr(ICONS, "FOLDER_OPEN", getattr(ICONS, "FOLDER", ICONS.SEARCH)),
        on_click=lambda e: infer_dir_picker.get_directory_path(
            dialog_title="Select adapter directory (folder containing adapter)",
        ),
    )

    # Button to import latest local training adapter
    def on_infer_use_latest(e=None):  # noqa: ARG001
        try:
            info = train_state.get("local_infer") or {}
        except Exception:
            info = {}
        try:
            adapter_path = (info.get("adapter_path") or "").strip()
            base_model_name = (
                info.get("base_model")
                or infer_base_model_tf.value
                or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
            ).strip()
            if adapter_path and os.path.isdir(adapter_path):
                infer_adapter_dir_tf.value = adapter_path
                infer_base_model_tf.value = base_model_name
                try:
                    train_state.setdefault("inference", {})
                    train_state["inference"]["adapter_path"] = adapter_path
                    train_state["inference"]["base_model"] = base_model_name
                    train_state["inference"]["model_loaded"] = False
                except Exception:
                    pass
                infer_status.value = "Using adapter from latest local training run."
                try:
                    infer_status.color = WITH_OPACITY(0.6, BORDER_BASE)
                except Exception:
                    pass
                infer_meta.value = f"Adapter: {adapter_path} • Base model: {base_model_name}"
                try:
                    if hasattr(page, "run_task"):
                        page.run_task(on_infer_validate_adapter)
                except Exception:
                    pass
            else:
                infer_status.value = (
                    "Latest local training adapter not found. Run a local training job first."
                )
                try:
                    infer_status.color = getattr(
                        COLORS,
                        "RED_400",
                        getattr(COLORS, "RED", None),
                    )
                except Exception:
                    pass
                infer_meta.value = ""
            page.update()
        except Exception:
            pass

    infer_use_latest_btn = ft.TextButton(
        "Use latest local training",
        icon=getattr(ICONS, "HISTORY", getattr(ICONS, "UPDATE", ICONS.CACHED)),
        on_click=on_infer_use_latest,
    )

    # Generation controls
    infer_preset_dd = ft.Dropdown(
        label="Preset",
        options=[
            ft.dropdown.Option("Deterministic"),
            ft.dropdown.Option("Balanced"),
            ft.dropdown.Option("Creative"),
        ],
        value="Balanced",
        width=220,
    )
    infer_temp_slider = ft.Slider(
        label="Temperature: {value}",
        min=0.1,
        max=1.2,
        divisions=11,
        value=0.7,
        width=320,
    )
    infer_max_tokens_slider = ft.Slider(
        label="Max new tokens: {value}",
        min=64,
        max=512,
        divisions=14,
        value=256,
        width=320,
    )
    infer_prompt_tf = ft.TextField(
        label="Prompt",
        multiline=True,
        min_lines=3,
        max_lines=8,
        width=1000,
        dense=True,
    )
    infer_output = ft.ListView(expand=True, spacing=4, auto_scroll=True)
    infer_output_placeholder = ft.Text(
        "Responses will appear here after running inference.",
        color=WITH_OPACITY(0.5, BORDER_BASE),
    )
    infer_generate_btn = ft.ElevatedButton(
        "Generate",
        icon=getattr(ICONS, "PLAY_CIRCLE", ICONS.PLAY_ARROW),
    )
    infer_clear_btn = ft.TextButton(
        "Clear history",
        icon=getattr(ICONS, "DELETE_SWEEP", getattr(ICONS, "DELETE", ICONS.CLOSE)),
    )
    infer_busy_ring = ft.ProgressRing(visible=False, width=24, height=24, stroke_width=3)

    infer_full_chat_btn = ft.OutlinedButton(
        "Full Chat View",
        icon=getattr(ICONS, "CHAT", getattr(ICONS, "PSYCHOLOGY", ICONS.PLAY_CIRCLE)),
    )
    infer_chat_messages = ft.ListView(expand=True, spacing=8, auto_scroll=True)
    infer_chat_placeholder = ft.Text(
        "Start chatting with your fine-tuned model.",
        color=WITH_OPACITY(0.5, BORDER_BASE),
    )
    infer_chat_input = ft.TextField(
        hint_text="Type your message...",
        multiline=True,
        min_lines=1,
        max_lines=3,
        expand=True,
        dense=True,
    )
    infer_chat_send_btn = ft.FilledTonalButton(
        "Send",
        icon=getattr(ICONS, "SEND", getattr(ICONS, "PLAY_ARROW", ICONS.CHEVRON_RIGHT)),
    )
    infer_chat_busy_ring = ft.ProgressRing(visible=False, width=22, height=22, stroke_width=3)

    def _reset_infer_history() -> None:
        """Clear shared inference conversation state and both history UIs."""
        try:
            try:
                infer_output.controls.clear()
            except Exception:
                pass
            try:
                infer_output_placeholder.visible = True
            except Exception:
                pass
            try:
                infer_chat_messages.controls.clear()
            except Exception:
                pass
            try:
                infer_chat_placeholder.visible = True
            except Exception:
                pass
            try:
                state = train_state.get("inference") or {}
                if isinstance(state.get("chat_history"), list):
                    state["chat_history"].clear()
            except Exception:
                pass
            try:
                infer_status.value = "Idle — history cleared."
            except Exception:
                pass
            try:
                infer_status.color = WITH_OPACITY(0.6, BORDER_BASE)
            except Exception:
                pass
            try:
                infer_chat_input.value = ""
            except Exception:
                pass
            try:
                page.update()
            except Exception:
                pass
        except Exception:
            pass

    def _close_infer_chat_dialog(e=None):  # noqa: ARG001
        try:
            try:
                infer_chat_dialog.open = False
            except Exception:
                pass
            try:
                if hasattr(page, "close") and callable(getattr(page, "close")):
                    page.close(infer_chat_dialog)
            except Exception:
                pass
            try:
                page.update()
            except Exception:
                pass
        except Exception:
            pass

    infer_chat_dialog = ft.AlertDialog(
        modal=True,
        content=ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Row(
                                [
                                    ft.Icon(
                                        getattr(ICONS, "CHAT", getattr(ICONS, "PSYCHOLOGY", ICONS.PLAY_CIRCLE)),
                                        size=22,
                                        color=ACCENT_COLOR,
                                    ),
                                    ft.Text(
                                        "Full Chat View",
                                        size=16,
                                        weight=getattr(ft.FontWeight, "BOLD", None),
                                    ),
                                ],
                                spacing=8,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Container(
                        ft.Stack([infer_chat_messages, infer_chat_placeholder], expand=True),
                        expand=True,
                        height=420,
                        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                        border_radius=8,
                        padding=10,
                    ),
                    ft.Row(
                        [infer_chat_input, infer_chat_send_btn, infer_chat_busy_ring],
                        alignment=ft.MainAxisAlignment.END,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ],
                spacing=12,
                expand=True,
            ),
            width=900,
            height=560,
            padding=16,
        ),
        actions=[
            ft.TextButton(
                "Clear history",
                icon=getattr(ICONS, "DELETE_SWEEP", getattr(ICONS, "DELETE", ICONS.CLOSE)),
                on_click=lambda e: _reset_infer_history(),
            ),
            ft.TextButton(
                "Close",
                icon=getattr(ICONS, "CLOSE", getattr(ICONS, "CANCEL", ICONS.CLOSE)),
                on_click=_close_infer_chat_dialog,
            ),
        ],
    )

    # Inference state
    train_state.setdefault("inference", {})

    def _set_infer_controls_enabled(enabled: bool) -> None:
        """Enable/disable prompt & response controls until a valid adapter is selected."""
        flag = bool(enabled)
        try:
            infer_preset_dd.disabled = not flag
        except Exception:
            pass
        try:
            infer_temp_slider.disabled = not flag
        except Exception:
            pass
        try:
            infer_max_tokens_slider.disabled = not flag
        except Exception:
            pass
        try:
            infer_prompt_tf.disabled = not flag
        except Exception:
            pass
        try:
            infer_generate_btn.disabled = not flag
        except Exception:
            pass
        try:
            infer_clear_btn.disabled = not flag
        except Exception:
            pass
        try:
            infer_full_chat_btn.disabled = not flag
        except Exception:
            pass

    # Lock prompt section until we validate an adapter
    _set_infer_controls_enabled(False)

    def on_infer_clear(e=None):  # noqa: ARG001
        try:
            _reset_infer_history()
        except Exception:
            pass

    def on_infer_preset_change(e=None):  # noqa: ARG001
        try:
            name = (infer_preset_dd.value or "Balanced").lower()
        except Exception:
            name = "balanced"
        if name.startswith("deterministic"):
            t = 0.2
            n = 128
        elif name.startswith("creative"):
            t = 1.0
            n = 512
        else:
            t = 0.7
            n = 256
        try:
            infer_temp_slider.value = t
            infer_max_tokens_slider.value = n
            page.update()
        except Exception:
            pass

    async def on_infer_generate(e=None):  # noqa: ARG001
        prompt = (infer_prompt_tf.value or "").strip()
        if not prompt:
            infer_status.value = "Enter a prompt to run inference."
            await safe_update(page)
            return
        base_model_name = (
            infer_base_model_tf.value or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
        ).strip()
        adapter_path = (infer_adapter_dir_tf.value or "").strip()
        if (not adapter_path) or (not os.path.isdir(adapter_path)):
            infer_status.value = (
                "Adapter directory is missing or invalid. Pick a valid folder containing an adapter."
            )
            try:
                infer_status.color = getattr(
                    COLORS,
                    "RED_400",
                    getattr(COLORS, "RED", None),
                )
            except Exception:
                pass
            await safe_update(page)
            return

        def _looks_like_adapter_dir(adapter_path: str) -> bool:
            try:
                cfg_path = os.path.join(adapter_path, "adapter_config.json")
                if os.path.isfile(cfg_path):
                    return True
                for name in os.listdir(adapter_path):
                    if name.endswith(".safetensors") or name.endswith(".bin"):
                        return True
            except Exception:
                return False
            return False

        # Adapter sanity check: require obvious adapter artifacts
        if not _looks_like_adapter_dir(adapter_path):
            try:
                logger.warning(
                    "Inference adapter directory %r does not contain adapter_config.json or weight files; refusing to run.",
                    adapter_path,
                )
            except Exception:
                pass
            infer_status.value = (
                "Adapter directory doesn't look like a valid LoRA adapter. "
                "Select the adapter folder from a completed fine-tuning run."
            )
            try:
                infer_status.color = getattr(
                    COLORS,
                    "RED_400",
                    getattr(COLORS, "RED", None),
                )
            except Exception:
                pass
            await safe_update(page)
            return
        try:
            max_tokens = int(getattr(infer_max_tokens_slider, "value", 256) or 256)
        except Exception:
            max_tokens = 256
        try:
            temperature = float(getattr(infer_temp_slider, "value", 0.7) or 0.7)
        except Exception:
            temperature = 0.7
        if max_tokens <= 0:
            max_tokens = 1
        if temperature <= 0:
            temperature = 0.1
        info = train_state.get("inference") or {}
        loaded = bool(info.get("model_loaded")) and (
            (info.get("adapter_path") or "").strip() == adapter_path
        ) and ((info.get("base_model") or "").strip() == base_model_name)
        infer_generate_btn.disabled = True
        try:
            infer_busy_ring.visible = True
        except Exception:
            pass
        if not loaded:
            infer_status.value = "Loading fine-tuned model and generating response..."
        else:
            infer_status.value = "Generating response from fine-tuned model..."
        await safe_update(page)
        try:
            text = await asyncio.to_thread(
                local_infer_generate_text_helper,
                base_model_name,
                adapter_path,
                prompt,
                max_tokens,
                temperature,
            )
            try:
                infer_output_placeholder.visible = False
            except Exception:
                pass
            infer_output.controls.append(
                ft.Column(
                    [
                        ft.Text("Prompt", weight=getattr(ft.FontWeight, "BOLD", None)),
                        ft.Text(prompt),
                        ft.Text("Response", weight=getattr(ft.FontWeight, "BOLD", None)),
                        ft.Text(text),
                    ],
                    spacing=4,
                ),
            )
            # Keep shared chat history in sync with the main Prompt & responses view
            try:
                state = train_state.setdefault("inference", {})
                history = state.setdefault("chat_history", [])
                history.append({"role": "user", "content": prompt})
                history.append({"role": "assistant", "content": text})
            except Exception:
                pass
            try:
                infer_chat_placeholder.visible = False
            except Exception:
                pass
            try:
                infer_chat_messages.controls.append(
                    ft.Column(
                        [
                            ft.Row(
                                [
                                    ft.Container(
                                        content=ft.Text(prompt),
                                        padding=8,
                                        border_radius=12,
                                        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                                    ),
                                ],
                                alignment=ft.MainAxisAlignment.END,
                            ),
                            ft.Row(
                                [
                                    ft.Container(
                                        content=ft.Text(text),
                                        padding=8,
                                        border_radius=12,
                                        bgcolor=WITH_OPACITY(0.03, BORDER_BASE),
                                    ),
                                ],
                                alignment=ft.MainAxisAlignment.START,
                            ),
                        ],
                        spacing=6,
                    ),
                )
            except Exception:
                pass
            try:
                train_state.setdefault("inference", {})
                train_state["inference"]["model_loaded"] = True
                train_state["inference"]["adapter_path"] = adapter_path
                train_state["inference"]["base_model"] = base_model_name
            except Exception:
                pass
            infer_status.value = "Idle — last inference complete."
            try:
                infer_status.color = WITH_OPACITY(0.6, BORDER_BASE)
            except Exception:
                pass
        except Exception as ex:  # pragma: no cover - runtime error path
            try:
                logger.exception(
                    "Inference failed for base_model=%r adapter=%r: %s",
                    base_model_name,
                    adapter_path,
                    ex,
                )
            except Exception:
                pass
            infer_status.value = f"Inference failed: {ex}"
            try:
                infer_status.color = getattr(
                    COLORS,
                    "RED_400",
                    getattr(COLORS, "RED", None),
                )
            except Exception:
                pass
        finally:
            infer_generate_btn.disabled = False
            try:
                infer_busy_ring.visible = False
            except Exception:
                pass
            await safe_update(page)

    def on_infer_open_full_chat(e=None):  # noqa: ARG001
        """Open the Full Chat View dialog in a way that works across Flet versions."""
        try:
            opened = False
            # Prefer the newer page.open(dlg) API if available
            try:
                if hasattr(page, "open") and callable(getattr(page, "open")):
                    page.open(infer_chat_dialog)
                    opened = True
            except Exception:
                opened = False
            if not opened:
                try:
                    page.dialog = infer_chat_dialog
                    infer_chat_dialog.open = True
                except Exception:
                    pass
            # Rebuild chat UI from shared chat_history so it matches the main history
            try:
                state = train_state.get("inference") or {}
                history = state.get("chat_history") or []
            except Exception:
                history = []
            try:
                infer_chat_messages.controls.clear()
            except Exception:
                pass
            if history:
                try:
                    i = 0
                    n = len(history)
                    while i < n:
                        user_msg = ""
                        assistant_msg = ""
                        try:
                            user_item = history[i]
                            if isinstance(user_item, dict):
                                user_msg = user_item.get("content") or ""
                            else:
                                user_msg = str(user_item or "")
                        except Exception:
                            user_msg = ""
                        if i + 1 < n:
                            try:
                                asst_item = history[i + 1]
                                if isinstance(asst_item, dict):
                                    assistant_msg = asst_item.get("content") or ""
                                else:
                                    assistant_msg = str(asst_item or "")
                            except Exception:
                                assistant_msg = ""
                        infer_chat_messages.controls.append(
                            ft.Column(
                                [
                                    ft.Row(
                                        [
                                            ft.Container(
                                                content=ft.Text(user_msg),
                                                padding=8,
                                                border_radius=12,
                                                border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                                            ),
                                        ],
                                        alignment=ft.MainAxisAlignment.END,
                                    ),
                                    ft.Row(
                                        [
                                            ft.Container(
                                                content=ft.Text(assistant_msg),
                                                padding=8,
                                                border_radius=12,
                                                bgcolor=WITH_OPACITY(0.03, BORDER_BASE),
                                            ),
                                        ],
                                        alignment=ft.MainAxisAlignment.START,
                                    ),
                                ],
                                spacing=6,
                            ),
                        )
                        i += 2
                    infer_chat_placeholder.visible = False
                except Exception:
                    pass
            else:
                try:
                    infer_chat_placeholder.visible = True
                except Exception:
                    pass
            try:
                infer_chat_input.value = ""
            except Exception:
                pass
            try:
                page.update()
            except Exception:
                pass
        except Exception:
            pass

    async def on_infer_chat_send(e=None):  # noqa: ARG001
        msg = (infer_chat_input.value or "").strip()
        if not msg:
            return
        base_model_name = (
            infer_base_model_tf.value or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
        ).strip()
        adapter_path = (infer_adapter_dir_tf.value or "").strip()
        if (not adapter_path) or (not os.path.isdir(adapter_path)):
            infer_status.value = (
                "Adapter directory is missing or invalid. Pick a valid folder containing an adapter."
            )
            try:
                infer_status.color = getattr(
                    COLORS,
                    "RED_400",
                    getattr(COLORS, "RED", None),
                )
            except Exception:
                pass
            await safe_update(page)
            return

        def _looks_like_adapter_dir(adapter_path: str) -> bool:
            try:
                cfg_path = os.path.join(adapter_path, "adapter_config.json")
                if os.path.isfile(cfg_path):
                    return True
                for name in os.listdir(adapter_path):
                    if name.endswith(".safetensors") or name.endswith(".bin"):
                        return True
            except Exception:
                return False
            return False

        if not _looks_like_adapter_dir(adapter_path):
            try:
                logger.warning(
                    "Chat adapter directory %r does not contain adapter_config.json or weight files; refusing to run.",
                    adapter_path,
                )
            except Exception:
                pass
            infer_status.value = (
                "Adapter directory doesn't look like a valid LoRA adapter. "
                "Select the adapter folder from a completed fine-tuning run."
            )
            try:
                infer_status.color = getattr(
                    COLORS,
                    "RED_400",
                    getattr(COLORS, "RED", None),
                )
            except Exception:
                pass
            await safe_update(page)
            return
        try:
            max_tokens = int(getattr(infer_max_tokens_slider, "value", 256) or 256)
        except Exception:
            max_tokens = 256
        try:
            temperature = float(getattr(infer_temp_slider, "value", 0.7) or 0.7)
        except Exception:
            temperature = 0.7
        if max_tokens <= 0:
            max_tokens = 1
        if temperature <= 0:
            temperature = 0.1
        info = train_state.get("inference") or {}
        loaded = bool(info.get("model_loaded")) and (
            (info.get("adapter_path") or "").strip() == adapter_path
        ) and ((info.get("base_model") or "").strip() == base_model_name)
        infer_chat_send_btn.disabled = True
        try:
            infer_chat_busy_ring.visible = True
        except Exception:
            pass
        if not loaded:
            infer_status.value = "Loading fine-tuned model and generating chat response..."
        else:
            infer_status.value = "Generating chat response from fine-tuned model..."
        await safe_update(page)
        state = train_state.setdefault("inference", {})
        history = state.setdefault("chat_history", [])
        history.append({"role": "user", "content": msg})
        prompt_parts = []
        for m in history:
            role = "User" if (m.get("role") or "user") == "user" else "Assistant"
            prompt_parts.append(f"{role}: {m.get('content') or ''}")
        prompt_text = "\n".join(prompt_parts) + "\nAssistant:"
        try:
            text = await asyncio.to_thread(
                local_infer_generate_text_helper,
                base_model_name,
                adapter_path,
                prompt_text,
                max_tokens,
                temperature,
            )
            history.append({"role": "assistant", "content": text})
            try:
                infer_chat_input.value = ""
            except Exception:
                pass
            try:
                infer_chat_placeholder.visible = False
            except Exception:
                pass
            infer_chat_messages.controls.append(
                ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Container(
                                    content=ft.Text(msg),
                                    padding=8,
                                    border_radius=12,
                                    border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.END,
                        ),
                        ft.Row(
                            [
                                ft.Container(
                                    content=ft.Text(text),
                                    padding=8,
                                    border_radius=12,
                                    bgcolor=WITH_OPACITY(0.03, BORDER_BASE),
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.START,
                        ),
                    ],
                    spacing=6,
                ),
            )
            # Mirror chat turns into the main Prompt & responses history list
            try:
                infer_output_placeholder.visible = False
            except Exception:
                pass
            try:
                infer_output.controls.append(
                    ft.Column(
                        [
                            ft.Text("Prompt", weight=getattr(ft.FontWeight, "BOLD", None)),
                            ft.Text(msg),
                            ft.Text("Response", weight=getattr(ft.FontWeight, "BOLD", None)),
                            ft.Text(text),
                        ],
                        spacing=4,
                    ),
                )
            except Exception:
                pass
            try:
                state["model_loaded"] = True
                state["adapter_path"] = adapter_path
                state["base_model"] = base_model_name
            except Exception:
                pass
            infer_status.value = "Idle — last chat response complete."
            try:
                infer_status.color = WITH_OPACITY(0.6, BORDER_BASE)
            except Exception:
                pass
        except Exception as ex:  # pragma: no cover - runtime error path
            try:
                logger.exception(
                    "Chat inference failed for base_model=%r adapter=%r: %s",
                    base_model_name,
                    adapter_path,
                    ex,
                )
            except Exception:
                pass
            infer_status.value = f"Inference failed: {ex}"
            try:
                infer_status.color = getattr(
                    COLORS,
                    "RED_400",
                    getattr(COLORS, "RED", None),
                )
            except Exception:
                pass
        finally:
            infer_chat_send_btn.disabled = False
            try:
                infer_chat_busy_ring.visible = False
            except Exception:
                pass
            await safe_update(page)

    try:
        infer_preset_dd.on_change = on_infer_preset_change
    except Exception:
        pass
    infer_generate_btn.on_click = lambda e: page.run_task(on_infer_generate)
    infer_clear_btn.on_click = on_infer_clear
    infer_full_chat_btn.on_click = on_infer_open_full_chat
    infer_chat_send_btn.on_click = lambda e: page.run_task(on_infer_chat_send)

    inference_tab = build_inference_tab(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        infer_status=infer_status,
        infer_meta=infer_meta,
        infer_base_model_tf=infer_base_model_tf,
        infer_adapter_dir_tf=infer_adapter_dir_tf,
        infer_browse_btn=infer_browse_btn,
        infer_use_latest_btn=infer_use_latest_btn,
        infer_preset_dd=infer_preset_dd,
        infer_temp_slider=infer_temp_slider,
        infer_max_tokens_slider=infer_max_tokens_slider,
        infer_prompt_tf=infer_prompt_tf,
        infer_generate_btn=infer_generate_btn,
        infer_clear_btn=infer_clear_btn,
        infer_busy_ring=infer_busy_ring,
        infer_output=infer_output,
        infer_output_placeholder=infer_output_placeholder,
        infer_full_chat_btn=infer_full_chat_btn,
    )

    return inference_tab
