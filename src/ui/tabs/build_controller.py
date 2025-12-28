"""Publish tab controller for FineFoundry.

This module builds the Publish tab controls and wires up all build
and push handlers, keeping `src/main.py` slimmer. Layout composition
still lives in `tab_build.py` and its section builders.
"""

from __future__ import annotations

from typing import Any, Dict

import asyncio
import io
import json
import os
import random

import flet as ft

from helpers.common import safe_update
from helpers.theme import BORDER_BASE, COLORS, ICONS, REFRESH_ICON
from helpers.ui import WITH_OPACITY, make_empty_placeholder, pill, build_offline_banner, offline_reason_text
from helpers.build import (
    run_build as run_build_helper,
    run_push_async as run_push_async_helper,
)
from helpers.settings_ollama import (
    load_config as load_ollama_config_helper,
    chat as ollama_chat_helper,
)

from huggingface_hub import HfApi, HfFolder, create_repo

# save_dataset utilities (local, with PYTHONPATH pointing to project src)
try:  # pragma: no cover - normal path
    import save_dataset as sd
except Exception:  # pragma: no cover - fallback for alternate runtimes
    import sys as _sys

    _sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    import save_dataset as sd

from ui.tabs.tab_build import build_build_tab


def _schedule_task(page: ft.Page, coro):
    """Robust scheduler helper for async tasks.

    Mirrors the pattern used in other controllers, preferring
    ``page.run_task`` when available and falling back to
    ``asyncio.create_task``.
    """

    try:
        if hasattr(page, "run_task") and callable(getattr(page, "run_task")):
            return page.run_task(coro)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        return asyncio.create_task(coro())
    except Exception:  # pragma: no cover - defensive
        return None


def build_build_tab_with_logic(
    page: ft.Page,
    *,
    section_title,
    _mk_help_handler,
    _hf_cfg: Dict[str, Any],
    ollama_enable_cb: ft.Control,
    ollama_models_dd: ft.Dropdown,
    offline_mode_sw: ft.Switch,
) -> ft.Control:
    """Build the Publish tab UI and attach all related handlers.

    This mirrors the previous inline Build tab setup from ``main.py``, but
    keeps the behavior localized to this module.
    """

    # Source selector for dataset preview/processing (Database only; merged
    # dataset mode has been removed in favor of database-backed workflow)
    source_mode = ft.Dropdown(
        options=[
            ft.dropdown.Option("Database"),
        ],
        value="Database",
        width=180,
    )

    # Data source - Database only (JSON removed as user option)
    data_source_dd = ft.Dropdown(
        label="Data source",
        options=[
            ft.dropdown.Option("Database"),
        ],
        value="Database",
        width=160,
        visible=False,  # Hidden since only one option
    )
    db_session_dd = ft.Dropdown(
        label="Scrape session",
        options=[],
        width=360,
        visible=True,
        tooltip="Select a scrape session from the database",
    )
    db_refresh_btn = ft.IconButton(
        icon=ft.Icons.REFRESH,
        tooltip="Refresh sessions",
        visible=True,
    )
    # Data file kept for internal use only (not user-facing)
    data_file = ft.TextField(
        label="Data file (JSON)",
        value="",
        width=360,
        visible=False,
    )

    def _refresh_db_sessions(_=None):
        """Refresh the database session dropdown."""
        try:
            from db.scraped_data import list_scrape_sessions

            sessions = list_scrape_sessions(limit=50)
            options = []
            for s in sessions:
                # Prefer custom name if set, otherwise fallback to auto-generated label
                if s.get("name"):
                    label = f"{s['name']} ({s['pair_count']} pairs)"
                elif s.get("source_details"):
                    label = f"{s['source']}: {s['source_details'][:30]} - {s['pair_count']} pairs"
                else:
                    label = f"{s['source']} - {s['pair_count']} pairs ({s['created_at'][:10]})"
                options.append(ft.dropdown.Option(key=str(s["id"]), text=label))
            db_session_dd.options = options
            if options and not db_session_dd.value:
                db_session_dd.value = options[0].key
        except Exception as e:
            db_session_dd.options = [ft.dropdown.Option(key="", text=f"Error: {e}")]
        try:
            page.update()
        except Exception:
            pass

    db_refresh_btn.on_click = _refresh_db_sessions

    def _update_data_source(_=None):
        # Always use database - JSON file option removed
        db_session_dd.visible = True
        db_refresh_btn.visible = True
        data_file.visible = False
        _refresh_db_sessions()
        try:
            page.update()
        except Exception:
            pass

    data_source_dd.on_change = _update_data_source
    # Initialize sessions on load
    try:
        _refresh_db_sessions()
    except Exception:
        pass
    merged_dir = ft.TextField(
        label="Merged dataset dir",
        value="merged_dataset",
        width=240,
    )
    seed = ft.TextField(
        label="Seed",
        value="42",
        width=120,
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    shuffle = ft.Switch(label="Shuffle", value=True)
    val_slider = ft.Slider(
        min=0,
        max=0.2,
        value=0.01,
        divisions=20,
        label="{value}",
    )
    test_slider = ft.Slider(
        min=0,
        max=0.2,
        value=0.0,
        divisions=20,
        label="{value}",
    )
    min_len_b = ft.TextField(
        label="Min Length",
        value="1",
        width=120,
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    save_dir = ft.TextField(
        label="Save dir",
        value="hf_dataset",
        width=240,
    )

    push_toggle = ft.Switch(label="Push to Hub", value=False)
    repo_id = ft.TextField(
        label="Repo ID",
        value="username/my-dataset",
        width=280,
    )
    private = ft.Switch(label="Private", value=True)
    token_val_ui = ft.TextField(
        label="HF Token",
        password=True,
        can_reveal_password=True,
        width=320,
    )
    try:
        token_val_ui.visible = False
    except Exception:
        pass

    model_run_dd = ft.Dropdown(
        label="Training run",
        options=[],
        width=520,
        tooltip="Select a completed training run (adapter) to publish",
    )
    model_run_refresh_btn = ft.IconButton(
        icon=ft.Icons.REFRESH,
        tooltip="Refresh training runs",
    )
    model_repo_id_tf = ft.TextField(
        label="Model repo id",
        value="",
        width=360,
        hint_text="username/my-adapter-model",
    )
    model_private_sw = ft.Switch(label="Private", value=True)
    model_publish_ring = ft.ProgressRing(width=18, height=18, value=None, visible=False)
    model_publish_state: Dict[str, Any] = {"inflight": False}

    # Validation chip for splits
    split_error = ft.Text("", color=COLORS.RED)

    def on_split_change(_):
        total = (val_slider.value or 0) + (test_slider.value or 0)
        if total >= 0.9:  # generous limit
            split_error.value = f"Warning: val+test too large ({total:.2f})"
        else:
            split_error.value = ""
        page.update()

    val_slider.on_change = on_split_change
    test_slider.on_change = on_split_change

    # Toggle UI fields (Database is the only mode; merged dataset support
    # has been removed)
    def on_source_change(_):
        try:
            # Always show database controls
            db_session_dd.visible = True
            db_refresh_btn.visible = True
            data_file.visible = False  # Always hidden - JSON removed as user option
            merged_dir.visible = False
            # Enable processing params in Database mode
            for ctl in [seed, shuffle, min_len_b, val_slider, test_slider]:
                try:
                    ctl.disabled = False
                except Exception:
                    pass
            _refresh_db_sessions()
        except Exception:
            pass
        page.update()

    source_mode.on_change = on_source_change
    data_source_dd.on_change = on_source_change
    # Initialize visibility/disabled state
    on_source_change(None)

    # Split badges (values updated during build)
    split_badges = {
        "train": pill("Train: 0", COLORS.BLUE, ICONS.STACKED_LINE_CHART),
        "val": pill("Val: 0", COLORS.ORANGE, ICONS.SIGNAL_CELLULAR_ALT),
        "test": pill("Test: 0", COLORS.PURPLE, ICONS.SSID_CHART),
    }
    split_meta = {
        "train": (COLORS.BLUE, ICONS.STACKED_LINE_CHART),
        "val": (COLORS.ORANGE, ICONS.SIGNAL_CELLULAR_ALT),
        "test": (COLORS.PURPLE, ICONS.SSID_CHART),
    }

    # Timeline (scrollable)
    timeline = ft.ListView(expand=1, auto_scroll=True, spacing=6)
    timeline_placeholder = make_empty_placeholder("No status yet", ICONS.TASK)
    status_section_ref: Dict[str, Any] = {}

    # Offline banner (tab-level constraints)
    offline_banner = build_offline_banner(
        [
            "Publishing to Hugging Face Hub is disabled.",
            "Local-only actions (e.g., Ollama) remain available.",
        ]
    )

    push_offline_reason = offline_reason_text("Offline Mode: Hugging Face Hub actions are disabled.")
    offline_click_state: Dict[str, Any] = {"shown": False}

    def _show_offline_snack_once(msg: str) -> None:
        try:
            if bool(offline_click_state.get("shown")):
                return
            offline_click_state["shown"] = True
        except Exception:
            return
        try:
            page.snack_bar = ft.SnackBar(ft.Text(msg))
            page.open(page.snack_bar)
            page.update()
        except Exception:
            pass

    def on_push_offline_click(_=None):
        try:
            if not bool(getattr(offline_mode_sw, "value", False)):
                return
        except Exception:
            return
        _show_offline_snack_once("Offline Mode: Hugging Face Hub actions are disabled.")

    def on_model_publish_offline_click(_=None):
        try:
            if not bool(getattr(offline_mode_sw, "value", False)):
                return
        except Exception:
            return
        _show_offline_snack_once("Offline Mode: Hugging Face Hub actions are disabled.")

    def _refresh_model_runs(_=None):
        try:
            from db.training_runs import list_training_runs

            runs = list_training_runs(limit=50)
            options = []
            for r in runs:
                status = r.get("status")
                rid = r.get("id")
                name = r.get("name") or "(unnamed)"
                base_model = r.get("base_model") or ""
                adapter_path = r.get("adapter_path") or ""
                created = (r.get("created_at") or "")[:10]

                ok = status == "completed" and adapter_path and os.path.isdir(adapter_path)
                if ok:
                    label = f"✅ {name} - {base_model.split('/')[-1] if base_model else 'unknown'} ({created})"
                    options.append(ft.dropdown.Option(key=str(rid), text=label))
                else:
                    status_icon = {"pending": "⏳", "running": "🔄", "failed": "❌"}.get(status, "")
                    label = f"{status_icon} {name} ({status})"
                    options.append(ft.dropdown.Option(key=str(rid), text=label, disabled=True))

            model_run_dd.options = options
            if options and not model_run_dd.value:
                for opt in options:
                    if not getattr(opt, "disabled", False):
                        model_run_dd.value = opt.key
                        break
        except Exception as e:
            model_run_dd.options = [ft.dropdown.Option(key="", text=f"Error: {e}")]
        try:
            page.update()
        except Exception:
            pass

    model_run_refresh_btn.on_click = _refresh_model_runs
    try:
        _refresh_model_runs()
    except Exception:
        pass

    async def on_publish_adapter():
        try:
            if model_publish_state.get("inflight"):
                return
        except Exception:
            return

        try:
            if bool(getattr(offline_mode_sw, "value", False)):
                page.snack_bar = ft.SnackBar(
                    ft.Text("Offline mode is enabled; publishing to Hugging Face Hub is disabled.")
                )
                page.open(page.snack_bar)
                await safe_update(page)
                return
        except Exception:
            pass

        run_id = (model_run_dd.value or "").strip()
        repo = (model_repo_id_tf.value or "").strip()
        try:
            tok = (_hf_cfg.get("token") or "").strip() if isinstance(_hf_cfg, dict) else ""
        except Exception:
            tok = ""
        if not tok:
            try:
                tok = os.environ.get("HF_TOKEN") or getattr(HfFolder, "get_token", lambda: "")()
            except Exception:
                tok = ""

        if not run_id:
            page.snack_bar = ft.SnackBar(ft.Text("Select a completed training run first."))
            page.open(page.snack_bar)
            await safe_update(page)
            return
        if not tok:
            page.snack_bar = ft.SnackBar(ft.Text("Set your Hugging Face token in Settings → Hugging Face Access."))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        from db.training_runs import get_training_run

        run = get_training_run(int(run_id))
        if not run:
            page.snack_bar = ft.SnackBar(ft.Text("Training run not found."))
            page.open(page.snack_bar)
            await safe_update(page)
            return
        if run.get("status") != "completed":
            page.snack_bar = ft.SnackBar(ft.Text("Select a completed training run."))
            page.open(page.snack_bar)
            await safe_update(page)
            return
        adapter_path = (run.get("adapter_path") or "").strip()
        if not adapter_path or not os.path.isdir(adapter_path):
            page.snack_bar = ft.SnackBar(ft.Text(f"Adapter folder not found: {adapter_path}"))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        api = HfApi(token=tok)
        if not repo:
            try:
                user = api.whoami()["name"]
            except Exception:
                user = "username"
            safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in (run.get("name") or "finefoundry"))
            repo = f"{user}/{safe_name}-adapter"
            try:
                model_repo_id_tf.value = repo
            except Exception:
                pass

        model_publish_state["inflight"] = True
        model_publish_ring.visible = True
        timeline.controls.append(
            ft.Row([ft.Icon(ICONS.CLOUD_UPLOAD, color=COLORS.BLUE), ft.Text(f"Publishing adapter to Hub: {repo}")])
        )
        update_status_placeholder()
        await safe_update(page)

        try:
            await asyncio.to_thread(
                create_repo,
                repo_id=repo,
                token=tok,
                private=bool(getattr(model_private_sw, "value", True)),
                repo_type="model",
                exist_ok=True,
            )

            try:
                wants_card = bool(getattr(model_use_custom_card, "value", False))
            except Exception:
                wants_card = False
            try:
                card_md = (getattr(model_card_editor, "value", "") or "").strip()
            except Exception:
                card_md = ""
            if wants_card and not card_md:
                raise RuntimeError("Model card is enabled but empty. Add content or disable the model card toggle.")

            await asyncio.to_thread(
                api.upload_folder,
                folder_path=adapter_path,
                repo_id=repo,
                repo_type="model",
            )

            if wants_card and card_md:
                await asyncio.to_thread(
                    api.upload_file,
                    path_or_fileobj=io.BytesIO(card_md.encode("utf-8")),
                    path_in_repo="README.md",
                    repo_id=repo,
                    repo_type="model",
                    commit_message="Add model card",
                )
            timeline.controls.append(
                ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text("Adapter published")])
            )
            _url = f"https://huggingface.co/{repo}"
            timeline.controls.append(
                ft.Row(
                    [
                        ft.Icon(ICONS.OPEN_IN_NEW, color=COLORS.BLUE),
                        ft.TextButton("Open on Hugging Face", on_click=lambda e, u=_url: page.launch_url(u)),
                    ]
                )
            )
            page.snack_bar = ft.SnackBar(ft.Text("Published adapter to Hub"))
            page.open(page.snack_bar)
            await safe_update(page)
        except Exception as e:
            timeline.controls.append(
                ft.Row([ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED), ft.Text(f"Publish failed: {e}")])
            )
            page.snack_bar = ft.SnackBar(ft.Text(f"Publish failed: {e}"))
            page.open(page.snack_bar)
            await safe_update(page)
        finally:
            try:
                model_publish_state["inflight"] = False
            except Exception:
                pass
            try:
                model_publish_ring.visible = False
            except Exception:
                pass
            update_status_placeholder()
            await safe_update(page)

    cancel_build: Dict[str, Any] = {"cancelled": False}
    dd_ref: Dict[str, Any] = {"dd": None}
    push_state: Dict[str, Any] = {"inflight": False}
    push_ring = ft.ProgressRing(width=18, height=18, value=None, visible=False)
    card_preview_container: ft.Container | None = None
    model_card_preview_container: ft.Container | None = None

    # --- Model Card Creator controls/state ---
    # Switch to enable custom model card instead of autogenerated
    use_custom_card = ft.Switch(
        label="Use custom model card (README.md)",
        value=False,
    )

    # Helper to build a simple default template (used when user wants a starting point)
    def _default_card_template(repo: str) -> str:
        rid = (repo or "username/dataset").strip()
        return f"""---
 tags:
   - text-generation
 language:
   - en
 license: other
 pretty_name: {rid}
 ---

 # Dataset Card: {rid}

 ## Dataset Summary
 Provide a concise description of the dataset, its source, and intended purpose.

 ## Data Fields
 - input: description
 - output: description

 ## Source and Collection
 Describe how data was collected and any preprocessing steps.

 ## Splits
 - Train: <num>
 - Validation: <num>
 - Test: <num>

 ## Usage
 ```python
 from datasets import load_dataset
 ds = load_dataset("{rid}")
 print(ds)
 ```

 ## Ethical Considerations and Warnings
 - Content may include offensive or unsafe material depending on source. Use responsibly.

 ## Licensing
 Specify license and any restrictions.

 ## Changelog
 - v1.0: Initial release.
 """

    # Editor and preview
    card_editor = ft.TextField(
        label="Model Card Markdown",
        multiline=True,
        min_lines=12,
        max_lines=32,
        value="",
        width=960,
        disabled=True,
    )

    # The dataset card editor is wrapped in a bordered container (for scrolling/height).
    # Disable the TextField's own outline border to avoid a messy double-border effect.
    try:
        _InputBorder = getattr(ft, "InputBorder", None)
        if _InputBorder is not None and hasattr(_InputBorder, "NONE"):
            card_editor.border = _InputBorder.NONE
    except Exception:
        pass

    # Safe Markdown factory for wider Flet compatibility
    def _make_md(value: str):
        try:
            return ft.Markdown(value, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB)
        except Exception:
            try:
                return ft.Markdown(value)
            except Exception:
                # Fallback: plain text if Markdown is unavailable
                return ft.Text(value)

    model_use_custom_card = ft.Switch(
        label="Use custom model card (README.md)",
        value=False,
    )

    def _default_model_card_template(repo: str) -> str:
        rid = (repo or "username/model").strip()
        return f"""---
tags:
  - fine-tuning
  - lora
  - adapter
library_name: transformers
license: other
---

# Model Card: {rid}

## Model Description
Describe what this adapter does and how it changes the base model.

## Base Model
- Base model: <base-model>

## Training Data
Describe the dataset(s) used for fine-tuning.

## Training Procedure
- Method: LoRA adapter fine-tuning

## How to Use
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "<base-model>"
adapter = "{rid}"

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
```

## Limitations and Bias
Add known limitations and risks.

## License
Specify license and any restrictions.
"""

    model_card_editor = ft.TextField(
        label="Model Card Markdown",
        multiline=True,
        min_lines=12,
        max_lines=32,
        value="",
        width=960,
        disabled=True,
    )

    model_card_preview_switch = ft.Switch(
        label="Live preview",
        value=False,
        disabled=True,
    )
    model_card_preview_md = _make_md("")
    try:
        model_card_preview_md.visible = False
    except Exception:
        pass

    model_card_preview_container = ft.Container(
        ft.Column(
            [model_card_preview_md],
            scroll=ft.ScrollMode.AUTO,
            spacing=0,
        ),
        height=300,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        padding=8,
        visible=False,
    )

    def _model_has_card_content() -> bool:
        try:
            return bool((model_card_editor.value or "").strip())
        except Exception:
            return False

    def _model_apply_preview_visibility() -> None:
        try:
            show = (
                bool(model_use_custom_card.value)
                and bool(model_card_preview_switch.value)
                and _model_has_card_content()
            )
            try:
                if hasattr(model_card_preview_md, "visible"):
                    model_card_preview_md.visible = show
            except Exception:
                pass
            try:
                if model_card_preview_container is not None:
                    model_card_preview_container.visible = show
            except Exception:
                pass
        except Exception:
            pass

    def _model_update_preview() -> None:
        try:
            if hasattr(model_card_preview_md, "value"):
                model_card_preview_md.value = model_card_editor.value or ""
        except Exception:
            try:
                model_card_preview_md.value = model_card_editor.value or ""
            except Exception:
                pass
        _model_apply_preview_visibility()

    def _on_toggle_model_custom_card(_):
        enabled = bool(model_use_custom_card.value)
        try:
            model_card_editor.disabled = not enabled
            model_card_preview_switch.disabled = not enabled
            if hasattr(model_card_preview_md, "visible"):
                model_card_preview_md.visible = enabled and bool(model_card_preview_switch.value)
        except Exception:
            pass
        _model_apply_preview_visibility()
        page.update()

    model_use_custom_card.on_change = _on_toggle_model_custom_card

    def _on_model_editor_change(_):
        if bool(model_card_preview_switch.value):
            _model_update_preview()
            page.update()

    try:
        model_card_editor.on_change = _on_model_editor_change
    except Exception:
        pass

    def _on_model_preview_toggle(_):
        try:
            if hasattr(model_card_preview_md, "visible"):
                model_card_preview_md.visible = bool(model_card_preview_switch.value) and bool(
                    model_use_custom_card.value
                )
        except Exception:
            pass
        _model_update_preview()
        page.update()

    model_card_preview_switch.on_change = _on_model_preview_toggle

    def _on_model_load_simple_template(_):
        model_use_custom_card.value = True
        _on_toggle_model_custom_card(None)
        model_card_editor.value = _default_model_card_template((model_repo_id_tf.value or "username/model").strip())
        _model_update_preview()
        page.update()

    model_load_template_btn = ft.TextButton(
        "Load simple template",
        icon=ICONS.ARTICLE,
        on_click=_on_model_load_simple_template,
    )

    model_clear_card_btn = ft.TextButton(
        "Clear",
        icon=ICONS.BACKSPACE,
        on_click=lambda e: (
            setattr(model_card_editor, "value", ""),
            _model_update_preview(),
            page.update(),
        ),
    )

    model_ollama_spinner = ft.ProgressRing(width=16, height=16, stroke_width=2)
    try:
        model_ollama_spinner.visible = False
    except Exception:
        pass
    model_ollama_status_text = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))
    model_ollama_gen_status = ft.Row(
        [model_ollama_spinner, model_ollama_status_text],
        spacing=8,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    async def _on_model_generate_with_ollama():
        try:
            model_gen_with_ollama_btn.disabled = True
        except Exception:
            pass
        try:
            model_ollama_spinner.visible = True
        except Exception:
            pass
        try:
            model_ollama_status_text.value = "Generating with Ollama…"
        except Exception:
            pass
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Generating model card with Ollama…"))
            page.open(page.snack_bar)
        except Exception:
            pass
        await safe_update(page)

        try:
            if not bool(ollama_enable_cb.value):
                page.snack_bar = ft.SnackBar(ft.Text("Enable Ollama in Settings first."))
                page.open(page.snack_bar)
                await safe_update(page)
                return
        except Exception:
            pass

        cfg = load_ollama_config_helper()
        base_url = (cfg.get("base_url") or "http://127.0.0.1:11434").strip()
        model_name = (
            (ollama_models_dd.value or "") or (cfg.get("selected_model") or "") or (cfg.get("default_model") or "")
        ).strip()
        if not model_name:
            page.snack_bar = ft.SnackBar(ft.Text("Select an Ollama model in Settings."))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        run_id = (model_run_dd.value or "").strip()
        if not run_id:
            page.snack_bar = ft.SnackBar(ft.Text("Select a completed training run first."))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        try:
            from db.training_runs import get_training_run

            run = await asyncio.to_thread(lambda: get_training_run(int(run_id)))
        except Exception as e:
            page.snack_bar = ft.SnackBar(ft.Text(f"Failed to load training run: {e}"))
            page.open(page.snack_bar)
            await safe_update(page)
            return
        if not isinstance(run, dict):
            page.snack_bar = ft.SnackBar(ft.Text("Training run not found."))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        rid = (model_repo_id_tf.value or "username/model").strip()
        base = (run.get("base_model") or "").strip()
        ds_src = (run.get("dataset_source") or "").strip()
        ds_id = (run.get("dataset_id") or "").strip()
        name = (run.get("name") or "").strip()
        hp = run.get("hp") if isinstance(run.get("hp"), dict) else {}

        try:
            model_ollama_status_text.value = f"Generating with Ollama model '{model_name}'…"
        except Exception:
            pass
        await safe_update(page)

        system_prompt = (
            "You write concise, high-quality Hugging Face model cards for fine-tuned adapters. "
            "Output ONLY valid Markdown starting with YAML frontmatter."
        )
        hp_json = "{}"
        try:
            hp_json = json.dumps(hp or {}, ensure_ascii=False, indent=2)
        except Exception:
            hp_json = "{}"

        user_prompt = (
            "Create a professional Hugging Face model card (README.md) in Markdown for a LoRA adapter.\n"
            f"Repo: {rid}\n"
            f"Run name: {name}\n"
            f"Base model: {base}\n"
            f"Dataset source: {ds_src}\n"
            f"Dataset id: {ds_id}\n\n"
            "Include YAML frontmatter (tags, license, language if known), then sections: "
            "Model Description, Base Model, Training Data, Training Procedure, Intended Use, "
            "How to Use (code snippet), Limitations & Bias, and Citation.\n\n"
            f"Training hyperparameters (JSON):\n```json\n{hp_json}\n```\n"
            "If unknown, state assumptions transparently."
        )

        try:
            md = await ollama_chat_helper(
                base_url,
                model_name,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            model_use_custom_card.value = True
            _on_toggle_model_custom_card(None)
            model_card_editor.value = md
            _model_update_preview()
            try:
                model_ollama_status_text.value = "Generated with Ollama ✓"
            except Exception:
                pass
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Model card generated with Ollama"))
                page.open(page.snack_bar)
            except Exception:
                pass
            await safe_update(page)
        except Exception as e:
            msg = f"Ollama generation failed: {e}"
            try:
                model_ollama_status_text.value = msg
            except Exception:
                pass
            try:
                page.snack_bar = ft.SnackBar(ft.Text(msg))
                page.open(page.snack_bar)
            except Exception:
                pass
            await safe_update(page)
        finally:
            try:
                model_ollama_spinner.visible = False
            except Exception:
                pass
            try:
                model_gen_with_ollama_btn.disabled = False
            except Exception:
                pass
            await safe_update(page)

    model_gen_with_ollama_btn = ft.ElevatedButton(
        "Generate with Ollama",
        icon=getattr(ICONS, "SMART_TOY", ICONS.HUB),
        on_click=lambda e: _schedule_task(page, _on_model_generate_with_ollama),
    )

    card_preview_switch = ft.Switch(
        label="Live preview",
        value=False,
        disabled=True,
    )
    card_preview_md = _make_md("")
    try:
        # Some Flet controls don't have 'visible'; guard accordingly
        card_preview_md.visible = False
    except Exception:
        pass

    # Dedicated preview container (hidden until we have content + preview enabled)
    card_preview_container = ft.Container(
        ft.Column(
            [card_preview_md],
            scroll=ft.ScrollMode.AUTO,
            spacing=0,
        ),
        height=300,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        padding=8,
        visible=False,
    )

    def _has_card_content() -> bool:
        try:
            return bool((card_editor.value or "").strip())
        except Exception:
            return False

    def _apply_preview_visibility() -> None:
        # Show preview only when: custom mode enabled + live preview on + content non-empty
        try:
            show = bool(use_custom_card.value) and bool(card_preview_switch.value) and _has_card_content()
            try:
                if hasattr(card_preview_md, "visible"):
                    card_preview_md.visible = show
            except Exception:
                pass
            try:
                if card_preview_container is not None:
                    card_preview_container.visible = show
            except Exception:
                pass
        except Exception:
            pass

    def _update_preview() -> None:
        try:
            if hasattr(card_preview_md, "value"):
                card_preview_md.value = card_editor.value or ""
        except Exception:
            # If preview control is Text (fallback), set .value via content replacement
            try:
                card_preview_md.value = card_editor.value or ""
            except Exception:
                pass
        # Re-evaluate visibility whenever content changes
        _apply_preview_visibility()

    def _on_toggle_custom_card(_):
        enabled = bool(use_custom_card.value)
        try:
            card_editor.disabled = not enabled
            card_preview_switch.disabled = not enabled
            if hasattr(card_preview_md, "visible"):
                card_preview_md.visible = enabled and bool(card_preview_switch.value)
        except Exception:
            pass
        _apply_preview_visibility()
        page.update()

    use_custom_card.on_change = _on_toggle_custom_card

    def _on_editor_change(_):
        if bool(card_preview_switch.value):
            _update_preview()
            page.update()

    try:
        card_editor.on_change = _on_editor_change
    except Exception:
        pass

    def _on_preview_toggle(_):
        try:
            if hasattr(card_preview_md, "visible"):
                card_preview_md.visible = bool(card_preview_switch.value) and bool(use_custom_card.value)
        except Exception:
            pass
        _update_preview()
        page.update()

    card_preview_switch.on_change = _on_preview_toggle

    def _on_load_simple_template(_):
        # Turn on custom mode and load a simple template scaffold
        use_custom_card.value = True
        _on_toggle_custom_card(None)
        card_editor.value = _default_card_template((repo_id.value or "username/dataset").strip())
        _update_preview()
        page.update()

    ollama_spinner = ft.ProgressRing(width=16, height=16, stroke_width=2)
    try:
        ollama_spinner.visible = False
    except Exception:
        pass
    ollama_status_text = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))
    ollama_gen_status = ft.Row(
        [ollama_spinner, ollama_status_text],
        spacing=8,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    async def _on_generate_with_ollama():
        # Generate using Ollama from the selected data file (JSON list of {input,output})
        try:
            gen_with_ollama_btn.disabled = True
        except Exception:
            pass
        try:
            ollama_spinner.visible = True
        except Exception:
            pass
        try:
            ollama_status_text.value = "Generating with Ollama…"
        except Exception:
            pass
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Generating dataset card with Ollama…"))
            page.open(page.snack_bar)
        except Exception:
            pass
        await safe_update(page)

        try:
            if not bool(ollama_enable_cb.value):
                page.snack_bar = ft.SnackBar(
                    ft.Text("Enable Ollama in Settings first."),
                )
                page.open(page.snack_bar)
                await safe_update(page)
                return
        except Exception:
            pass

        cfg = load_ollama_config_helper()
        base_url = (cfg.get("base_url") or "http://127.0.0.1:11434").strip()
        model_name = (
            (ollama_models_dd.value or "") or (cfg.get("selected_model") or "") or (cfg.get("default_model") or "")
        ).strip()
        if not model_name:
            page.snack_bar = ft.SnackBar(
                ft.Text("Select an Ollama model in Settings."),
            )
            page.open(page.snack_bar)
            await safe_update(page)
            return

        # Load records from database
        try:
            db_session_id = (db_session_dd.value or "").strip()
            if not db_session_id:
                raise RuntimeError("No database session selected")
            from db.scraped_data import get_pairs_for_session

            records = await asyncio.to_thread(lambda: get_pairs_for_session(int(db_session_id)))
            if not records:
                raise RuntimeError(f"No pairs in session {db_session_id}")
        except Exception as e:  # pragma: no cover - runtime error path
            page.snack_bar = ft.SnackBar(
                ft.Text(f"Failed to load data: {e}"),
            )
            page.open(page.snack_bar)
            await safe_update(page)
            return

        if not isinstance(records, list) or len(records) == 0:
            page.snack_bar = ft.SnackBar(
                ft.Text(
                    "Data file is empty or invalid (expected list of records).",
                ),
            )
            page.open(page.snack_bar)
            await safe_update(page)
            return

        total_n = len(records)
        # Sample a small subset for context
        k = min(8, total_n)
        idxs = random.sample(range(total_n), k) if total_n >= k else list(range(total_n))
        samples: list[dict[str, str]] = []
        for i in idxs:
            rec = records[i] if isinstance(records[i], dict) else {}
            inp = str(rec.get("input", ""))
            outp = str(rec.get("output", ""))
            try:
                inp = sd._truncate(inp, 400)  # type: ignore[attr-defined]
                outp = sd._truncate(outp, 400)  # type: ignore[attr-defined]
            except Exception:
                if len(inp) > 400:
                    inp = inp[:399] + "…"
                if len(outp) > 400:
                    outp = outp[:399] + "…"
            samples.append({"input": inp, "output": outp})

        # Size category helper
        try:
            size_cat = sd._size_category(total_n)  # type: ignore[attr-defined]
        except Exception:
            size_cat = (
                "n<1K"
                if total_n < 1_000
                else (
                    "1K<n<10K"
                    if total_n < 10_000
                    else ("10K<n<100K" if total_n < 100_000 else ("100K<n<1M" if total_n < 1_000_000 else "n>1M"))
                )
            )

        rid = (repo_id.value or "username/dataset").strip()
        user_prompt = (
            f"You are an expert data curator. Create a professional Hugging Face dataset card (README.md) "
            f"in Markdown for the dataset '{rid}'.\n"
            f"Use the provided random samples to infer characteristics. Include a YAML frontmatter header with "
            f"tags, task_categories=text-generation, language=en, license=other, size_categories=[{size_cat}].\n"
            "Then include sections: Dataset Summary, Data Fields, Source and Collection, Splits (estimate if "
            "needed), Usage (datasets code snippet), Ethical Considerations and Warnings, Licensing, Example "
            "Records (re-embed the samples), How to Cite, Changelog.\n"
            "Keep the tone clear and factual. If unsure, state assumptions transparently."
        )
        samples_json = json.dumps(samples, ensure_ascii=False, indent=2)
        user_prompt += f"\n\nSamples (JSON):\n```json\n{samples_json}\n```\nTotal records (approx): {total_n}"

        system_prompt = (
            "You write concise, high-quality dataset cards for Hugging Face. "
            "Output ONLY valid Markdown starting with YAML frontmatter."
        )

        try:
            ollama_status_text.value = f"Generating with Ollama model '{model_name}'…"
        except Exception:
            pass
        await safe_update(page)
        try:
            md = await ollama_chat_helper(
                base_url,
                model_name,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            use_custom_card.value = True
            _on_toggle_custom_card(None)
            card_editor.value = md
            _update_preview()
            try:
                ollama_status_text.value = "Generated with Ollama ✓"
            except Exception:
                pass
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Dataset card generated with Ollama"))
                page.open(page.snack_bar)
            except Exception:
                pass
            await safe_update(page)
        except Exception as e:  # pragma: no cover - runtime error path
            msg = f"Ollama generation failed: {e}"
            try:
                ollama_status_text.value = msg
            except Exception:
                pass
            try:
                page.snack_bar = ft.SnackBar(ft.Text(msg))
                page.open(page.snack_bar)
            except Exception:
                pass
            await safe_update(page)
        finally:
            try:
                ollama_spinner.visible = False
            except Exception:
                pass
            try:
                gen_with_ollama_btn.disabled = False
            except Exception:
                pass
            await safe_update(page)

    load_template_btn = ft.TextButton(
        "Load simple template",
        icon=ICONS.ARTICLE,
        on_click=_on_load_simple_template,
    )
    gen_from_ds_btn = ft.Container(visible=False)
    gen_with_ollama_btn = ft.ElevatedButton(
        "Generate with Ollama",
        icon=getattr(ICONS, "SMART_TOY", ICONS.HUB),
        on_click=lambda e: _schedule_task(page, _on_generate_with_ollama),
    )
    clear_card_btn = ft.TextButton(
        "Clear",
        icon=ICONS.BACKSPACE,
        on_click=lambda e: (
            setattr(card_editor, "value", ""),
            _update_preview(),
            page.update(),
        ),
    )

    def update_status_placeholder() -> None:
        try:
            has_entries = len(getattr(timeline, "controls", []) or []) > 0
            timeline_placeholder.visible = not has_entries
            try:
                ctl = status_section_ref.get("control")
                if ctl is not None:
                    ctl.visible = has_entries
            except Exception:
                pass
        except Exception:
            pass
        page.update()

    def on_refresh_build(_):
        cancel_build["cancelled"] = False
        timeline.controls.clear()
        for k in split_badges:
            label = {"train": "Train", "val": "Val", "test": "Test"}[k]
            split_badges[k].content = pill(
                f"{label}: 0",
                split_meta[k][0],
                split_meta[k][1],
            ).content
        push_state["inflight"] = False
        push_ring.visible = False
        # Re-enable push button if it was disabled
        for ctl in build_actions.controls:
            if isinstance(ctl, ft.TextButton) and "Push + Upload README" in getattr(ctl, "text", ""):
                ctl.disabled = False
        update_status_placeholder()

    async def on_build():
        # Respect Offline Mode: block dataset build/publish workflow when
        # offline, even if the button is somehow still clickable.
        try:
            if bool(getattr(offline_mode_sw, "value", False)):
                page.snack_bar = ft.SnackBar(
                    ft.Text("Offline mode is enabled; dataset build/publish is disabled."),
                )
                page.open(page.snack_bar)
                await safe_update(page)
                return
        except Exception:
            pass

        # Delegate to helper to keep controller slim
        hf_cfg_token = (_hf_cfg.get("token") or "").strip() if isinstance(_hf_cfg, dict) else ""
        return await run_build_helper(
            page=page,
            source_mode=source_mode,
            data_source_dd=data_source_dd,
            db_session_dd=db_session_dd,
            data_file=data_file,
            merged_dir=merged_dir,
            seed=seed,
            shuffle=shuffle,
            val_slider=val_slider,
            test_slider=test_slider,
            min_len_b=min_len_b,
            save_dir=save_dir,
            push_toggle=push_toggle,
            repo_id=repo_id,
            private=private,
            token_val_ui=token_val_ui,
            timeline=timeline,
            timeline_placeholder=timeline_placeholder,
            split_badges=split_badges,
            split_meta=split_meta,
            dd_ref=dd_ref,
            cancel_build=cancel_build,
            use_custom_card=use_custom_card,
            card_editor=card_editor,
            hf_cfg_token=hf_cfg_token,
            update_status_placeholder=update_status_placeholder,
        )

    async def on_push_async():
        # Respect Offline Mode: block remote Hub push when offline
        try:
            if bool(getattr(offline_mode_sw, "value", False)):
                page.snack_bar = ft.SnackBar(
                    ft.Text("Offline mode is enabled; pushing to Hugging Face Hub is disabled."),
                )
                page.open(page.snack_bar)
                await safe_update(page)
                return
        except Exception:
            pass

        # Delegate to helper to keep controller slim
        hf_cfg_token = (_hf_cfg.get("token") or "").strip() if isinstance(_hf_cfg, dict) else ""
        return await run_push_async_helper(
            page=page,
            repo_id=repo_id,
            token_val_ui=token_val_ui,
            private=private,
            dd_ref=dd_ref,
            push_state=push_state,
            push_ring=push_ring,
            build_actions=build_actions,
            timeline=timeline,
            timeline_placeholder=timeline_placeholder,
            update_status_placeholder=update_status_placeholder,
            use_custom_card=use_custom_card,
            card_editor=card_editor,
            hf_cfg_token=hf_cfg_token,
        )

    def on_cancel_build(_):
        cancel_build["cancelled"] = True
        # Surface immediate feedback in the timeline
        try:
            timeline.controls.append(
                ft.Row(
                    [
                        ft.Icon(ICONS.CANCEL, color=COLORS.RED),
                        ft.Text("Cancel requested — will stop ASAP"),
                    ]
                )
            )
            update_status_placeholder()
        except Exception:
            pass

    push_upload_btn = ft.TextButton(
        "Push + Upload README",
        icon=ICONS.CLOUD_UPLOAD,
        on_click=lambda e: _schedule_task(page, on_push_async),
        disabled=not bool(getattr(push_toggle, "value", False)),
    )

    build_actions = ft.Row(
        [
            ft.ElevatedButton(
                "Build Dataset",
                icon=ICONS.BUILD,
                on_click=lambda e: _schedule_task(page, on_build),
            ),
            ft.OutlinedButton(
                "Cancel",
                icon=ICONS.CANCEL,
                on_click=on_cancel_build,
            ),
            ft.TextButton(
                "Refresh",
                icon=REFRESH_ICON,
                on_click=on_refresh_build,
            ),
            push_upload_btn,
            push_ring,
        ],
        spacing=10,
    )

    publish_adapter_btn = ft.ElevatedButton(
        "Publish adapter",
        icon=ICONS.CLOUD_UPLOAD,
        on_click=lambda e: _schedule_task(page, on_publish_adapter),
        disabled=True,
    )

    model_publish_actions = ft.Row(
        [
            publish_adapter_btn,
            model_publish_ring,
        ],
        spacing=10,
    )

    def _update_publish_action_enabled(_=None):
        try:
            is_offline = bool(getattr(offline_mode_sw, "value", False))
        except Exception:
            is_offline = False

        try:
            push_upload_btn.disabled = (not bool(getattr(push_toggle, "value", False))) or is_offline
        except Exception:
            pass

        try:
            has_run = bool((getattr(model_run_dd, "value", None) or "").strip())
            has_repo = bool((getattr(model_repo_id_tf, "value", None) or "").strip())
            try:
                needs_card = bool(getattr(model_use_custom_card, "value", False))
            except Exception:
                needs_card = False
            try:
                has_card = bool((getattr(model_card_editor, "value", "") or "").strip())
            except Exception:
                has_card = False
            publish_adapter_btn.disabled = (not (has_run and has_repo and ((not needs_card) or has_card))) or is_offline
        except Exception:
            pass

        try:
            page.update()
        except Exception:
            pass

    try:
        push_toggle.on_change = _update_publish_action_enabled
    except Exception:
        pass
    try:
        model_run_dd.on_change = _update_publish_action_enabled
    except Exception:
        pass
    try:
        model_repo_id_tf.on_change = _update_publish_action_enabled
    except Exception:
        pass
    try:
        model_use_custom_card.on_change = lambda e=None: (
            _on_toggle_model_custom_card(e),
            _update_publish_action_enabled(e),
        )
    except Exception:
        pass
    try:
        model_card_editor.on_change = lambda e=None: (_on_model_editor_change(e), _update_publish_action_enabled(e))
    except Exception:
        pass

    model_offline_reason = offline_reason_text("Offline Mode: Hugging Face Hub actions are disabled.")

    model_publish_section = ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Publish model (adapter)",
                    getattr(ICONS, "HUB", getattr(ICONS, "PUBLIC", ICONS.CLOUD_UPLOAD)),
                    "Publish a LoRA adapter from a completed training run.",
                    on_help_click=_mk_help_handler(
                        "Publish model (adapter): Upload the LoRA adapter folder from a completed training run to a Hugging Face model repository."
                    ),
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Row([model_run_dd, model_run_refresh_btn], wrap=True),
                            ft.Row([model_repo_id_tf, model_private_sw], wrap=True),
                            ft.Divider(),
                            ft.Row([model_use_custom_card, model_card_preview_switch], wrap=True),
                            ft.Row([model_load_template_btn, model_clear_card_btn], wrap=True),
                            ft.Row([model_gen_with_ollama_btn], wrap=True),
                            model_ollama_gen_status,
                            model_card_editor,
                            model_card_preview_container,
                            model_publish_actions,
                        ],
                        spacing=10,
                    ),
                    width=1000,
                    border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                    border_radius=8,
                    padding=10,
                ),
                ft.Container(content=model_offline_reason, on_click=on_model_publish_offline_click),
            ],
            spacing=12,
        ),
        width=1000,
    )

    build_tab = build_build_tab(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        offline_banner=offline_banner,
        push_offline_reason=push_offline_reason,
        push_offline_click_handler=on_push_offline_click,
        source_mode=source_mode,
        data_source_dd=data_source_dd,
        db_session_dd=db_session_dd,
        db_refresh_btn=db_refresh_btn,
        data_file=data_file,
        merged_dir=merged_dir,
        seed=seed,
        shuffle=shuffle,
        min_len_b=min_len_b,
        save_dir=save_dir,
        val_slider=val_slider,
        test_slider=test_slider,
        split_error=split_error,
        split_badges=split_badges,
        push_toggle=push_toggle,
        repo_id=repo_id,
        private=private,
        token_val_ui=token_val_ui,
        build_actions=build_actions,
        use_custom_card=use_custom_card,
        card_preview_switch=card_preview_switch,
        load_template_btn=load_template_btn,
        gen_from_ds_btn=gen_from_ds_btn,
        gen_with_ollama_btn=gen_with_ollama_btn,
        clear_card_btn=clear_card_btn,
        ollama_gen_status=ollama_gen_status,
        card_editor=card_editor,
        card_preview_container=card_preview_container,
        model_publish_section=model_publish_section,
        timeline=timeline,
        timeline_placeholder=timeline_placeholder,
        status_section_ref=status_section_ref,
    )

    _update_publish_action_enabled()

    # Hook: respond to Offline Mode changes (disable Build/Publish while
    # offline, but keep Ollama-based generation available)
    def apply_offline_mode_to_build(_=None):
        try:
            is_offline = bool(getattr(offline_mode_sw, "value", False))
        except Exception:
            is_offline = False

        try:
            offline_banner.visible = is_offline
        except Exception:
            pass

        try:
            push_offline_reason.visible = is_offline
        except Exception:
            pass

        try:
            model_offline_reason.visible = is_offline
        except Exception:
            pass

        # Dataset source & parameters
        for ctl in [
            source_mode,
            data_source_dd,
            db_session_dd,
            db_refresh_btn,
            merged_dir,
            seed,
            shuffle,
            val_slider,
            test_slider,
            min_len_b,
            save_dir,
        ]:
            try:
                ctl.disabled = is_offline
            except Exception:
                pass

        # Push controls
        for ctl in [push_toggle, repo_id, private, token_val_ui]:
            try:
                ctl.disabled = is_offline
            except Exception:
                pass

        # Build / push / refresh / cancel buttons row
        try:
            for ctl in getattr(build_actions, "controls", []) or []:
                try:
                    if isinstance(ctl, (ft.ElevatedButton, ft.OutlinedButton, ft.TextButton)):
                        ctl.disabled = is_offline
                except Exception:
                    pass
        except Exception:
            pass

        # Model card helpers (HF/DB driven). Keep Ollama generator enabled.
        for ctl in [use_custom_card, card_preview_switch, load_template_btn, gen_from_ds_btn, clear_card_btn]:
            try:
                ctl.disabled = is_offline
            except Exception:
                pass

        # Model publish controls
        for ctl in [
            model_run_dd,
            model_run_refresh_btn,
            model_repo_id_tf,
            model_private_sw,
        ]:
            try:
                ctl.disabled = is_offline
            except Exception:
                pass
        try:
            for ctl in getattr(model_publish_actions, "controls", []) or []:
                try:
                    if isinstance(ctl, (ft.ElevatedButton, ft.OutlinedButton, ft.TextButton)):
                        ctl.disabled = is_offline
                except Exception:
                    pass
        except Exception:
            pass

        # Do not touch gen_with_ollama_btn, ollama_enable_cb, ollama_models_dd,
        # or card_editor so Ollama-based editing still works offline.

        try:
            page.update()
        except Exception:
            pass

    # Register with the shared Offline Mode switch
    try:
        hooks = getattr(offline_mode_sw, "data", None)
        if hooks is None:
            hooks = {}
        hooks["build_tab_offline"] = lambda e=None: apply_offline_mode_to_build(e)
        offline_mode_sw.data = hooks
    except Exception:
        pass

    # Apply current offline state on first load
    apply_offline_mode_to_build()
    update_status_placeholder()

    return build_tab
