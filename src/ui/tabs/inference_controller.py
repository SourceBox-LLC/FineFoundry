"""Inference tab controller for FineFoundry.

This module builds the Inference tab controls and wires up all inference
and chat handlers, keeping `src/main.py` smaller.

Layout composition still lives in `tab_inference.py`; this module focuses
on behavior and state wiring.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List

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
        "Select a completed training run to load for inference.",
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

    # Training run selector (replaces adapter directory text field)
    infer_training_run_dd = ft.Dropdown(
        label="Training run",
        options=[],
        width=520,
        tooltip="Select a completed training run to load for inference",
    )
    infer_training_run_refresh_btn = ft.IconButton(
        icon=ft.Icons.REFRESH,
        tooltip="Refresh training runs",
    )

    # Hidden field for adapter path (populated from selected run)
    infer_adapter_dir_tf = ft.TextField(
        label="Adapter directory",
        width=520,
        dense=True,
        visible=False,  # Hidden - managed internally
    )

    def _refresh_inference_runs(_=None):
        """Refresh the training runs dropdown with completed runs."""
        try:
            from db.training_runs import list_training_runs

            runs = list_training_runs(status="completed", limit=50)
            options = []
            for r in runs:
                # Only show runs with valid adapter paths
                adapter_path = r.get("adapter_path", "")
                if adapter_path and os.path.isdir(adapter_path):
                    label = f"✅ {r['name']} - {r['base_model'].split('/')[-1] if r.get('base_model') else 'unknown'} ({r['created_at'][:10]})"
                    options.append(ft.dropdown.Option(key=str(r["id"]), text=label))

            # Also add pending/running runs (grayed out info)
            other_runs = list_training_runs(limit=20)
            for r in other_runs:
                if r["status"] != "completed":
                    status_icon = {"pending": "⏳", "running": "🔄", "failed": "❌"}.get(r["status"], "")
                    label = f"{status_icon} {r['name']} ({r['status']})"
                    options.append(
                        ft.dropdown.Option(key=str(r["id"]), text=label, disabled=(r["status"] != "completed"))
                    )

            infer_training_run_dd.options = options
            if not options:
                infer_training_run_dd.options = [ft.dropdown.Option(key="", text="No completed training runs found")]
        except Exception as e:
            infer_training_run_dd.options = [ft.dropdown.Option(key="", text=f"Error: {e}")]
        try:
            page.update()
        except Exception:
            pass

    def _on_training_run_selected(_=None):
        """Handle training run selection - load adapter path and base model."""
        try:
            run_id = infer_training_run_dd.value
            if not run_id:
                return

            from db.training_runs import get_training_run

            run = get_training_run(int(run_id))
            if not run:
                infer_status.value = "Training run not found."
                infer_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
                page.update()
                return

            if run["status"] != "completed":
                infer_status.value = f"Training run is {run['status']}. Select a completed run."
                infer_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
                page.update()
                return

            adapter_path = run.get("adapter_path", "")
            base_model = run.get("base_model", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")

            if not adapter_path or not os.path.isdir(adapter_path):
                infer_status.value = f"Adapter path not found: {adapter_path}"
                infer_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
                page.update()
                return

            # Update UI fields
            infer_adapter_dir_tf.value = adapter_path
            infer_base_model_tf.value = base_model

            # Update train_state for inference
            train_state.setdefault("inference", {})
            train_state["inference"]["adapter_path"] = adapter_path
            train_state["inference"]["base_model"] = base_model
            train_state["inference"]["model_loaded"] = False
            train_state["inference"]["training_run_id"] = run_id

            infer_status.value = f"Loaded: {run['name']}"
            infer_status.color = WITH_OPACITY(0.6, BORDER_BASE)
            infer_meta.value = f"Adapter: {adapter_path} • Base model: {base_model}"

            _set_infer_controls_enabled(True)
            page.update()

            # Validate adapter
            if hasattr(page, "run_task"):
                page.run_task(on_infer_validate_adapter)
        except Exception as e:
            infer_status.value = f"Error loading run: {e}"
            infer_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
            page.update()

    infer_training_run_dd.on_change = _on_training_run_selected
    infer_training_run_refresh_btn.on_click = _refresh_inference_runs

    # Initialize training runs on load
    try:
        _refresh_inference_runs()
    except Exception:
        pass

    # Validate adapter directory
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
        base_model_name = (infer_base_model_tf.value or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit").strip()
        try:
            infer_busy_ring.visible = True
            await safe_update(page)
        except Exception:  # pragma: no cover - UI best-effort
            pass
        if (not adapter_path) or (not os.path.isdir(adapter_path)):
            infer_status.value = "Adapter directory is missing or invalid. Select a completed training run."
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
                    ft.Text("Invalid adapter directory. Please select a completed training run."),
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
                "The training run may not have completed successfully."
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
        infer_status.value = "Adapter validated. Ready for inference."
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
            page.snack_bar = ft.SnackBar(ft.Text("Adapter validated. Ready for inference."))
            page.open(page.snack_bar)
        except Exception:
            pass
        try:
            infer_busy_ring.visible = False
            await safe_update(page)
        except Exception:
            pass

    # Button to use latest training run
    def on_infer_use_latest(e=None):  # noqa: ARG001
        """Select the most recent completed training run."""
        try:
            from db.training_runs import list_training_runs

            # Get latest completed run
            runs = list_training_runs(status="completed", limit=1)
            if not runs:
                infer_status.value = "No completed training runs found. Complete a training run first."
                infer_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
                infer_meta.value = ""
                page.update()
                return

            run = runs[0]
            infer_training_run_dd.value = str(run["id"])
            _on_training_run_selected()
        except Exception as e:
            infer_status.value = f"Error loading latest run: {e}"
            infer_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
            page.update()

    infer_use_latest_btn = ft.TextButton(
        "Use latest completed run",
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
        min=0.1,
        max=1.2,
        divisions=11,
        value=0.7,
        width=280,
    )
    infer_temp_label = ft.Text("Temperature: 0.7", size=12)
    infer_max_tokens_slider = ft.Slider(
        min=64,
        max=512,
        divisions=14,
        value=256,
        width=280,
    )
    infer_max_tokens_label = ft.Text("Max tokens: 256", size=12)
    infer_rep_penalty_slider = ft.Slider(
        min=1.0,
        max=1.5,
        divisions=10,
        value=1.15,
        width=280,
    )
    infer_rep_penalty_label = ft.Text("Rep. penalty: 1.15", size=12)

    def _update_infer_slider_labels(e=None):
        try:
            infer_temp_label.value = f"Temperature: {infer_temp_slider.value:.1f}"
            infer_max_tokens_label.value = f"Max tokens: {int(infer_max_tokens_slider.value)}"
            infer_rep_penalty_label.value = f"Rep. penalty: {infer_rep_penalty_slider.value:.2f}"
            page.update()
        except Exception:
            pass

    infer_temp_slider.on_change = _update_infer_slider_labels
    infer_max_tokens_slider.on_change = _update_infer_slider_labels
    infer_rep_penalty_slider.on_change = _update_infer_slider_labels
    infer_prompt_tf = ft.TextField(
        label="Prompt",
        multiline=True,
        min_lines=3,
        max_lines=8,
        width=1000,
        dense=True,
    )
    # Dataset selector for sample prompts (any saved dataset)
    infer_dataset_dd = ft.Dropdown(
        label="Dataset for sample prompts",
        options=[],
        width=400,
        hint_text="Select a dataset to sample prompts from",
    )
    infer_dataset_refresh_btn = ft.IconButton(
        icon=getattr(ICONS, "REFRESH", ft.Icons.REFRESH),
        tooltip="Refresh datasets",
    )
    infer_sample_prompts_dd = ft.Dropdown(
        label="Sample prompts (optional)",
        options=[],
        width=550,
        hint_text="Select a sample prompt or enter your own above",
    )
    infer_sample_refresh_btn = ft.IconButton(
        icon=getattr(ICONS, "REFRESH", ft.Icons.REFRESH),
        tooltip="Get new random samples",
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
    infer_export_btn = ft.TextButton(
        "Export chats",
        icon=getattr(ICONS, "DOWNLOAD", getattr(ICONS, "SAVE_ALT", ICONS.SAVE)),
    )
    infer_busy_ring = ft.ProgressRing(visible=False, width=24, height=24, stroke_width=3)
    # Buffer for storing chat history (prompt/response pairs) for export
    infer_chat_buffer: List[Dict[str, str]] = []

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
                infer_chat_buffer.clear()
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

    def _on_save_infer_chats(e):
        """Save inference chat history to a file."""
        try:
            path = getattr(e, "path", None)
            if not path:
                return
            if not infer_chat_buffer:
                infer_status.value = "No chats to export."
                page.update()
                return
            # Format as readable text
            lines = []
            for i, chat in enumerate(infer_chat_buffer, 1):
                lines.append(f"=== Chat {i} ===")
                lines.append(f"Prompt:\n{chat.get('prompt', '')}")
                lines.append(f"\nResponse:\n{chat.get('response', '')}")
                lines.append("")
            txt = "\n".join(lines)
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            infer_status.value = f"Exported {len(infer_chat_buffer)} chats to: {path}"
            page.update()
        except Exception as ex:
            infer_status.value = f"Failed to export chats: {ex}"
            try:
                page.update()
            except Exception:
                pass

    infer_chats_picker = ft.FilePicker(on_result=_on_save_infer_chats)
    try:
        page.overlay.append(infer_chats_picker)
    except Exception:
        pass

    infer_export_btn.on_click = lambda e: infer_chats_picker.save_file(
        dialog_title="Export chat history",
        file_name=f"inference-chats-{int(time.time())}.txt",
        allowed_extensions=["txt", "md"],
    )

    def _refresh_infer_datasets(e=None):
        """Refresh the list of available datasets for sample prompts."""
        try:
            from db.scraped_data import list_scrape_sessions

            sessions = list_scrape_sessions()
            if sessions:
                options = []
                for s in sessions:
                    session_id = s.get("id")
                    source = s.get("source", "unknown")
                    details = s.get("source_details", "")
                    pair_count = s.get("pair_count", 0)
                    label = f"{source}"
                    if details:
                        label += f" - {details[:30]}"
                    label += f" ({pair_count} pairs)"
                    options.append(ft.dropdown.Option(key=str(session_id), text=label))
                infer_dataset_dd.options = options
            else:
                infer_dataset_dd.options = []
            infer_dataset_dd.value = None
            infer_sample_prompts_dd.options = []
            infer_sample_prompts_dd.value = None
            page.update()
        except Exception:
            pass

    def _refresh_infer_sample_prompts(e=None):
        """Refresh sample prompts from the selected dataset."""
        try:
            session_id = infer_dataset_dd.value
            if not session_id:
                infer_sample_prompts_dd.options = []
                infer_sample_prompts_dd.value = None
                page.update()
                return

            from db.scraped_data import get_random_prompts_for_session

            prompts = get_random_prompts_for_session(int(session_id), count=5)
            if prompts:
                options = []
                for i, prompt in enumerate(prompts):
                    display_text = prompt[:80] + "..." if len(prompt) > 80 else prompt
                    display_text = display_text.replace("\n", " ")
                    options.append(ft.dropdown.Option(key=prompt, text=f"{i + 1}. {display_text}"))
                infer_sample_prompts_dd.options = options
                infer_sample_prompts_dd.value = None
            else:
                infer_sample_prompts_dd.options = []
                infer_sample_prompts_dd.value = None
            page.update()
        except Exception:
            pass

    def _on_infer_sample_prompt_selected(e=None):
        """When a sample prompt is selected, populate the prompt text field."""
        try:
            selected = infer_sample_prompts_dd.value
            if selected:
                infer_prompt_tf.value = selected
                page.update()
        except Exception:
            pass

    def _on_infer_dataset_changed(e=None):
        """When dataset changes, refresh sample prompts."""
        _refresh_infer_sample_prompts()

    # Wire up dataset and sample prompts handlers
    try:
        infer_dataset_dd.on_change = _on_infer_dataset_changed
        infer_dataset_refresh_btn.on_click = _refresh_infer_datasets
        infer_sample_prompts_dd.on_change = _on_infer_sample_prompt_selected
        infer_sample_refresh_btn.on_click = _refresh_infer_sample_prompts
    except Exception:
        pass

    # Initial load of datasets
    _refresh_infer_datasets()

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
            infer_rep_penalty_slider.disabled = not flag
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
            r = 1.2  # Higher penalty for more focused output
        elif name.startswith("creative"):
            t = 1.0
            n = 512
            r = 1.1  # Lower penalty for more variety
        else:
            t = 0.7
            n = 256
            r = 1.15  # Balanced default
        try:
            infer_temp_slider.value = t
            infer_max_tokens_slider.value = n
            infer_rep_penalty_slider.value = r
            # Update labels
            infer_temp_label.value = f"Temperature: {t:.1f}"
            infer_max_tokens_label.value = f"Max tokens: {int(n)}"
            infer_rep_penalty_label.value = f"Rep. penalty: {r:.2f}"
            page.update()
        except Exception:
            pass

    async def on_infer_generate(e=None):  # noqa: ARG001
        prompt = (infer_prompt_tf.value or "").strip()
        if not prompt:
            infer_status.value = "Enter a prompt to run inference."
            await safe_update(page)
            return
        base_model_name = (infer_base_model_tf.value or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit").strip()
        adapter_path = (infer_adapter_dir_tf.value or "").strip()
        if (not adapter_path) or (not os.path.isdir(adapter_path)):
            infer_status.value = "Adapter directory is missing or invalid. Pick a valid folder containing an adapter."
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
        try:
            rep_penalty = float(getattr(infer_rep_penalty_slider, "value", 1.15) or 1.15)
        except Exception:
            rep_penalty = 1.15
        if max_tokens <= 0:
            max_tokens = 1
        if temperature <= 0:
            temperature = 0.1
        if rep_penalty < 1.0:
            rep_penalty = 1.0
        info = train_state.get("inference") or {}
        loaded = (
            bool(info.get("model_loaded"))
            and ((info.get("adapter_path") or "").strip() == adapter_path)
            and ((info.get("base_model") or "").strip() == base_model_name)
        )
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
                rep_penalty,
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
            # Store in chat buffer for export
            infer_chat_buffer.append({"prompt": prompt, "response": text})
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
        base_model_name = (infer_base_model_tf.value or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit").strip()
        adapter_path = (infer_adapter_dir_tf.value or "").strip()
        if (not adapter_path) or (not os.path.isdir(adapter_path)):
            infer_status.value = "Adapter directory is missing or invalid. Pick a valid folder containing an adapter."
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
        loaded = (
            bool(info.get("model_loaded"))
            and ((info.get("adapter_path") or "").strip() == adapter_path)
            and ((info.get("base_model") or "").strip() == base_model_name)
        )
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
        try:
            text = await asyncio.to_thread(
                local_infer_generate_text_helper,
                base_model_name,
                adapter_path,
                "",  # prompt ignored when chat_history is provided
                max_tokens,
                temperature,
                1.15,  # repetition_penalty
                list(history),  # pass chat history for proper template formatting
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
            # Store in chat buffer for export
            infer_chat_buffer.append({"prompt": msg, "response": text})
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
        infer_training_run_dd=infer_training_run_dd,
        infer_training_run_refresh_btn=infer_training_run_refresh_btn,
        infer_use_latest_btn=infer_use_latest_btn,
        infer_preset_dd=infer_preset_dd,
        infer_temp_slider=infer_temp_slider,
        infer_temp_label=infer_temp_label,
        infer_max_tokens_slider=infer_max_tokens_slider,
        infer_max_tokens_label=infer_max_tokens_label,
        infer_rep_penalty_slider=infer_rep_penalty_slider,
        infer_rep_penalty_label=infer_rep_penalty_label,
        infer_prompt_tf=infer_prompt_tf,
        infer_dataset_dd=infer_dataset_dd,
        infer_dataset_refresh_btn=infer_dataset_refresh_btn,
        infer_sample_prompts_dd=infer_sample_prompts_dd,
        infer_sample_refresh_btn=infer_sample_refresh_btn,
        infer_generate_btn=infer_generate_btn,
        infer_clear_btn=infer_clear_btn,
        infer_export_btn=infer_export_btn,
        infer_busy_ring=infer_busy_ring,
        infer_output=infer_output,
        infer_output_placeholder=infer_output_placeholder,
        infer_full_chat_btn=infer_full_chat_btn,
    )

    return inference_tab
