import asyncio
import random
import time
import os
from typing import List, Optional, Tuple
import json
import httpx
import re
from collections import Counter
from datetime import datetime


import flet as ft

# Runpod modules (pod lifecycle and infra helpers)
try:
    from runpod import runpod_pod as rp_pod
    from runpod import ensure_infra as rp_infra
except Exception:
    import sys as __sys2
    __sys2.path.append(os.path.dirname(__file__))
    from runpod import runpod_pod as rp_pod
    try:
        from runpod import ensure_infra as rp_infra
    except Exception:
        rp_infra = None

# Hugging Face datasets (used in training preview and helpers)
try:
    from datasets import load_dataset, get_dataset_config_names
except Exception:
    load_dataset = None
    get_dataset_config_names = None

# Hugging Face Hub client (used in Settings HF token test)
try:  # pragma: no cover - optional dependency
    from huggingface_hub import HfApi
except Exception:  # pragma: no cover
    HfApi = None  # type: ignore

from helpers.common import safe_update, set_terminal_title
from helpers.logging_config import get_logger
from helpers.theme import (
    COLORS,
    ICONS,
    ACCENT_COLOR,
    BORDER_BASE,
    REFRESH_ICON,
)

# Initialize logger for main module
logger = get_logger(__name__)
try:
    import save_dataset as sd
except Exception:
    import sys as _sys
    _sys.path.append(os.path.dirname(__file__))
    import save_dataset as sd
from helpers.boards import load_4chan_boards
from helpers.ui import (
    WITH_OPACITY,
    pill,
    section_title,
    make_wrap,
    make_selectable_pill,
    make_empty_placeholder,
    compute_two_col_flex,
    two_col_header,
    two_col_row,
)
from helpers.training import (
    run_local_training as run_local_training_helper,
    stop_local_training as stop_local_training_helper,
    build_hp_from_controls as build_hp_from_controls_helper,
)
from helpers.local_inference import generate_text as local_infer_generate_text_helper
from helpers.training_pod import (
    run_pod_training as run_pod_training_helper,
    restart_pod_container as restart_pod_container_helper,
    open_runpod as open_runpod_helper,
    open_web_terminal as open_web_terminal_helper,
    copy_ssh_command as copy_ssh_command_helper,
    ensure_infrastructure as ensure_infrastructure_helper,
    refresh_teardown_ui as refresh_teardown_ui_helper,
    do_teardown as do_teardown_helper,
    confirm_teardown_selected as confirm_teardown_selected_helper,
    confirm_teardown_all as confirm_teardown_all_helper,
    refresh_expert_gpus as refresh_expert_gpus_helper,
)
from helpers.datasets import guess_input_output_columns
from ui.tabs.tab_settings import build_settings_tab
from ui.tabs.tab_scrape import build_scrape_tab
from ui.tabs.tab_build import build_build_tab
from ui.tabs.tab_training import build_training_tab
from ui.tabs.tab_merge import build_merge_tab
from ui.tabs.tab_analysis import build_analysis_tab
from ui.tabs.training.sections.local_specs_section import build_local_specs_container
from ui.tabs.training.sections.logs_section import build_pod_logs_section
from ui.tabs.training.sections.pod_content import build_pod_content_container
from ui.tabs.training.sections.config_section import build_config_section
from ui.tabs.training.sections.rp_infra_section import build_rp_infra_panel
from ui.tabs.training.sections.dataset_section import build_dataset_section
from ui.tabs.training.sections.train_params_section import build_train_params_section
from ui.tabs.training.sections.teardown_section import build_teardown_section
from helpers.scrape import (
    run_reddit_scrape as run_reddit_scrape_helper,
    run_real_scrape as run_real_scrape_helper,
    run_stackexchange_scrape as run_stackexchange_scrape_helper,
)
from helpers.build import (
    run_build as run_build_helper,
    run_push_async as run_push_async_helper,
)
from helpers.merge import (
    run_merge as run_merge_helper,
    preview_merged as preview_merged_helper,
)
from helpers.local_docker import (
    on_docker_pull as on_docker_pull_helper,
)
from helpers.local_specs import (
    gather_local_specs as gather_local_specs_helper,
    refresh_local_gpus as refresh_local_gpus_helper,
)
from helpers.training_config import (
    saved_configs_dir as saved_configs_dir_helper,
    list_saved_configs as list_saved_configs_helper,
    read_json_file as read_json_file_helper,
    validate_config as validate_config_helper,
    get_last_used_config_name as get_last_used_config_name_helper,
    set_last_used_config_name as set_last_used_config_name_helper,
)
from helpers.settings_ollama import (
    load_config as load_ollama_config_helper,
    save_config as save_ollama_config_helper,
    fetch_tags as fetch_ollama_tags_helper,
    chat as ollama_chat_helper,
)

# Set terminal title to uppercase for the current session
set_terminal_title("PYTHON: MAIN")

APP_TITLE = "FineFoundry"






def main(page: ft.Page):
    page.title = APP_TITLE
    page.theme_mode = ft.ThemeMode.LIGHT
    page.theme = ft.Theme(color_scheme_seed=ACCENT_COLOR)
    page.window_min_width = 980
    page.window_min_height = 700

    # --- AppBar ---
    about_dialog = ft.AlertDialog(
        title=ft.Text("About"),
        content=ft.Text(
            "FineFoundry Core Application\n\n"
            "Scrape sources (4chan, Reddit, StackExchange), build/merge datasets, train locally or on Runpod, and analyze data quality.\n"
            "Built with Flet."
        ),
        actions=[ft.TextButton("Close", on_click=lambda e: setattr(about_dialog, "open", False))],
        on_dismiss=lambda e: page.update(),
    )

    def open_about(_):
        about_dialog.open = True
        page.dialog = about_dialog
        page.update()

    def toggle_theme(_):
        page.theme_mode = ft.ThemeMode.DARK if page.theme_mode == ft.ThemeMode.LIGHT else ft.ThemeMode.LIGHT
        page.update()

    def refresh_app(_):
        # Soft-refresh the app: clear overlays/controls and rebuild the UI
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Refreshing app..."))
            page.open(page.snack_bar)
        except Exception:
            pass

    async def on_save_current_config():
        try:
            payload = _build_config_payload_from_ui()
        except Exception as ex:
            try:
                page.snack_bar = ft.SnackBar(ft.Text(f"Failed to build config from UI: {ex}"))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return
        try:
            default_name = f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(payload.get('hp', {}).get('base_model','model')).replace('/', '_')}.json"
        except Exception:
            default_name = f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        name_tf = ft.TextField(label="Save as", value=default_name, width=420)

        def _do_save(_=None):
            try:
                name = (name_tf.value or default_name).strip()
                if not name:
                    return
                d = _saved_configs_dir()
                path = os.path.join(d, name if name.endswith('.json') else f"{name}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                    f.write("\n")
                try:
                    set_last_used_config_name_helper(os.path.basename(path))
                except Exception:
                    pass
                try:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Saved config: {os.path.basename(path)}"))
                    page.snack_bar.open = True
                except Exception:
                    pass
                try:
                    _refresh_config_list()
                except Exception:
                    pass
                try:
                    dlg.open = False
                    page.update()
                except Exception:
                    pass
            except Exception as ex:
                try:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Failed to save config: {ex}"))
                    page.snack_bar.open = True
                except Exception:
                    pass

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(getattr(ICONS, "SAVE_ALT", ICONS.SAVE), color=ACCENT_COLOR),
                ft.Text("Save current training setup"),
            ], alignment=ft.MainAxisAlignment.START),
            content=ft.Column([
                ft.Text("This will snapshot the current dataset, hyperparameters, target, and local/Runpod settings."),
                name_tf,
            ], tight=True, spacing=6),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: (setattr(dlg, "open", False), page.update())),
                ft.ElevatedButton("Save", icon=getattr(ICONS, "SAVE", ICONS.CHECK), on_click=_do_save),
            ],
        )
        try:
            if hasattr(page, "open") and callable(getattr(page, "open")):
                page.open(dlg)
            else:
                page.dialog = dlg
                dlg.open = True
        except Exception:
            try:
                page.dialog = dlg
                dlg.open = True
            except Exception:
                pass
        await safe_update(page)

    # In-app User Guide (opens a detailed, scrollable modal)
    def open_user_guide(_):
        try:
            # Immediate feedback to verify click wiring
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Opening user guide..."))
                page.open(page.snack_bar)
            except Exception:
                pass

            guide_md = """# FineFoundry — In‑App User Guide

## Overview
FineFoundry is a desktop studio to scrape, merge, analyze, build/publish, and train LLM datasets.

Tabs:
- Scrape
- Build / Publish
- Training
- Merge Datasets
- Dataset Analysis
- Settings

## Quick Navigation
- Top‑right icons: Refresh • Theme toggle
- User guide: press F1 or click the bottom‑right Help FAB.
- Use the tabs to switch workflows.

## Scrape
- Choose source (4chan, Reddit, StackExchange).
- Configure parameters (max pairs, threads, delays, length filters).
- Start scraping and preview results in a two‑column grid (input/output).
- Proxy support per scraper; system env proxies optional.

## Build / Publish
- Create train/val/test splits with Hugging Face `datasets`.
- Save locally and optionally push to the Hugging Face Hub.
- Provide `HF_TOKEN` in Settings or via environment/CLI login.

## Merge Datasets
- Combine JSON files and/or HF datasets.
- Auto‑map `input`/`output` columns; filter empty rows.
- Save merged JSON or build a `datasets.DatasetDict`.

## Training
- Training target: choose **Runpod - Pod** (remote GPU pod) or **local** (Docker on this machine).
- Hyperparameters: base model, epochs, learning rate, batch size, grad accumulation, max steps, packing, resume.
- Outputs & checkpoints: saved under **Output dir**. For containers, use paths under `/data/...` so they map back into your mounted host folder.
- Hugging Face auth: save a token in **Settings → Hugging Face** or export `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`. Required for pushing models or using private datasets.

### Local Training (Docker)
- Requires Docker Desktop / `docker` CLI available on your machine.
- Set **Host data dir** to a folder on your machine; it will be mounted at `/data` inside the container.
- Configure image, container name, GPU usage, and whether to **Pass HF token to container** (for private datasets / `--push`).
- Click **Start Local Training** to run the same `train.py` command used for Runpod, but inside a local container.
- After a successful local run, a **Quick Local Inference** panel appears so you can test the trained adapter immediately.

### Quick Local Inference
- Loads the base model and LoRA adapter from your last successful local training run.
- Controls: prompt box, **Generate** button, temperature slider, max tokens slider, presets (Deterministic / Balanced / Creative), **Clear history**, and inline model info.
- Useful for quick sanity checks of a new run without leaving the app.

### Configuration (save / load setups)
- Mode:
  - **Normal**: edit dataset + hyperparameters directly.
  - **Configuration**: pick a saved config and run with minimal inputs.
- Use **Save current setup** (in the Configuration section or near the training controls) to snapshot:
  - Dataset + hyperparameters
  - Training target (Runpod or local)
  - Runpod infrastructure or local Docker settings
- Saved configs are simple JSON files under `src/saved_configs/`. The last used config auto‑loads on startup so you can continue where you left off.

## Dataset Analysis
- Select dataset source (HF or JSON) and click **Analyze dataset**.
- Use **Select all** to toggle modules. Only enabled modules are computed and shown.
- Modules: Basic Stats • Duplicates • Sentiment • Class balance (length) •
  Extra proxies (Coverage overlap, Data leakage, Conversation depth, Speaker balance,
  Question vs Statement, Readability, NER proxy, Toxicity, Politeness, Dialogue Acts, Topics, Alignment).
- Summary lists active modules after each run; sample rows are shown for quick checks.

## Settings
- Hugging Face: save access token.
- Proxies: per‑scraper defaults and/or system env proxies.
- Runpod: save API key; standard mount path is `/data`.
- Ollama: enable connection, set base URL, list/select models; used for dataset cards.

## Runpod Notes
- Use `/data` as the network volume mount to avoid path issues with container assets.

## Troubleshooting
- No data found: verify source/boards and `Max Pairs` > 0; check network access.
- Push fails (401/403): ensure token has write scope and is provided.
- Slow previews: use the paginated preview or open the saved dataset with Python.
- SSL/cert errors: update certificates (e.g., `certifi`).

## About
- App: FineFoundry — desktop studio for dataset curation and fine‑tuning.
"""

            # Build Markdown content with robust fallbacks (some Flet builds differ)
            try:
                content = _make_md(guide_md)
            except Exception:
                try:
                    content = ft.Markdown(guide_md, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB)
                except Exception:
                    try:
                        content = ft.Markdown(guide_md)
                    except Exception:
                        content = ft.Text(guide_md)

            def _close_dialog():
                try:
                    # Preferred in Flet 0.28+: use page.close(dlg)
                    page.close(dlg)
                except Exception:
                    try:
                        # Fallback to old pattern
                        dlg.open = False
                        page.update()
                    except Exception:
                        pass

            dlg = ft.AlertDialog(
                title=ft.Text("FineFoundry User Guide"),
                modal=True,
                on_dismiss=lambda e: page.update(),
            )
            dlg.content = ft.Container(
                width=1000,
                height=640,
                content=ft.Column([content], scroll=ft.ScrollMode.AUTO),
            )
            dlg.actions = [ft.TextButton("Close", on_click=lambda e: _close_dialog())]

            # Show dialog (Flet 0.28+ pattern with backward-compatible fallback)
            try:
                page.open(dlg)
                try:
                    page.update()
                except Exception:
                    pass
            except Exception:
                try:
                    page.dialog = dlg
                    dlg.open = True
                    page.update()
                except Exception:
                    try:
                        page.overlay.append(dlg)
                        dlg.open = True
                        page.update()
                    except Exception:
                        pass
        except Exception as e:
            try:
                page.snack_bar = ft.SnackBar(ft.Text(f"Failed to open guide: {e}"))
                page.open(page.snack_bar)
            except Exception:
                pass

    # Removed unused open_user_guide_async wrapper; direct click handlers call open_user_guide

    # Helper: create an AppBar action that falls back to a TextButton if icon isn't available
    def _appbar_action(icon_const, tooltip: str, on_click_cb, text_fallback: Optional[str] = None):
        try:
            if icon_const is not None:
                # Use page.run_task if target is async; otherwise call directly
                if getattr(on_click_cb, "__name__", "").endswith("_async"):
                    return ft.IconButton(icon=icon_const, tooltip=tooltip, on_click=lambda e: page.run_task(on_click_cb))
                return ft.IconButton(icon=icon_const, tooltip=tooltip, on_click=lambda e: on_click_cb(e))
        except Exception:
            pass
        if getattr(on_click_cb, "__name__", "").endswith("_async"):
            return ft.TextButton(text_fallback or tooltip, on_click=lambda e: page.run_task(on_click_cb))
        return ft.TextButton(text_fallback or tooltip, on_click=lambda e: on_click_cb(e))

    page.appbar = ft.AppBar(
        leading=ft.Icon(getattr(ICONS, "DATASET_LINKED_OUTLINED", getattr(ICONS, "DATASET", getattr(ICONS, "DESCRIPTION", None)))) ,
        title=ft.Text(APP_TITLE, weight=ft.FontWeight.BOLD),
        center_title=False,
        bgcolor=WITH_OPACITY(0.03, COLORS.AMBER),
        actions=[
            _appbar_action(REFRESH_ICON or getattr(ICONS, "SYNC", getattr(ICONS, "CACHED", None)), "Refresh app", refresh_app, text_fallback="Refresh"),
            _appbar_action(
                getattr(ICONS, "DARK_MODE", getattr(ICONS, "BRIGHTNESS_4", getattr(ICONS, "NIGHTS_STAY", None))),
                "Toggle theme",
                toggle_theme,
                text_fallback="Theme",
            ),
        ],
    )
    try:
        page.update()
    except Exception:
        pass

    # Keyboard shortcut: F1 opens the user guide
    def _kb(e):
        try:
            key = str(getattr(e, "key", "")).upper()
            et = str(getattr(e, "type", "")).lower()
            if key == "F1" and et == "keydown":
                open_user_guide(None)
        except Exception:
            pass
    try:
        page.on_keyboard_event = _kb
    except Exception:
        pass

    # Guide FAB is placed in root Stack (see page.add(root_stack) above). No overlay used.

    # Reusable: build a click handler that opens a small dialog with the given help text
    def _mk_help_handler(text: str):
        def _handler(e):
            try:
                dlg = ft.AlertDialog(title=ft.Text("Info"), content=ft.Text(text))
                page.dialog = dlg
                dlg.open = True
                page.update()
            except Exception:
                try:
                    page.snack_bar = ft.SnackBar(ft.Text(text))
                    page.open(page.snack_bar)
                except Exception:
                    pass
        return _handler

    # ---------- SCRAPE TAB ----------
    # Boards (dynamic from API with fallback) and multi-select pills
    boards = load_4chan_boards()
    default_sel = {"pol", "b", "x"}
    board_pills: List[ft.Container] = [make_selectable_pill(b, selected=b in default_sel, base_color=COLORS.AMBER) for b in boards]
    boards_wrap = make_wrap(board_pills, spacing=6, run_spacing=6)
    board_warning = ft.Text("", color=COLORS.RED)

    def select_all_boards(_):
        for pill in board_pills:
            pill.data["selected"] = True
            pill.bgcolor = WITH_OPACITY(0.15, pill.data["base_color"]) 
        page.update()
        update_board_validation()

    def clear_all_boards(_):
        for pill in board_pills:
            pill.data["selected"] = False
            pill.bgcolor = None
        page.update()
        update_board_validation()

    board_actions = ft.Row([
        ft.TextButton("Select All", on_click=select_all_boards),
        ft.TextButton("Clear", on_click=clear_all_boards),
    ], spacing=8)

    # Inputs
    source_dd = ft.Dropdown(
        label="Source",
        value="4chan",
        options=[ft.dropdown.Option("4chan"), ft.dropdown.Option("reddit"), ft.dropdown.Option("stackexchange")],
        width=180,
    )
    reddit_url = ft.TextField(label="Reddit URL (subreddit or post)", value="https://www.reddit.com/r/LocalLLaMA/", width=420)
    reddit_max_posts = ft.TextField(label="Max Posts (Reddit)", value="30", width=180, keyboard_type=ft.KeyboardType.NUMBER)
    se_site = ft.TextField(label="StackExchange Site", value="stackoverflow", width=260)
    max_threads = ft.TextField(label="Max Threads", value="50", width=160, keyboard_type=ft.KeyboardType.NUMBER)
    max_pairs = ft.TextField(label="Max Pairs", value="5000", width=160, keyboard_type=ft.KeyboardType.NUMBER)
    delay = ft.TextField(label="Delay (s)", value="1.0", width=160, keyboard_type=ft.KeyboardType.NUMBER)
    min_len = ft.TextField(label="Min Length", value="3", width=160, keyboard_type=ft.KeyboardType.NUMBER)
    output_path = ft.TextField(label="Output JSON Path", value="scraped_training_data.json", width=360)
    dataset_format_dd = ft.Dropdown(
        label="Dataset Format",
        options=[ft.dropdown.Option("ChatML"), ft.dropdown.Option("Standard")],
        value="ChatML",
        width=200,
        tooltip="Select output dataset format: ChatML (multi-turn conversations) or Standard (raw input/output pairs).",
    )

    # Pairing mode control
    multiturn_sw = ft.Switch(label="Multiturn", value=False)
    strategy_dd = ft.Dropdown(
        label="Context Strategy",
        value="cumulative",
        options=[ft.dropdown.Option("cumulative"), ft.dropdown.Option("last_k"), ft.dropdown.Option("quote_chain")],
        width=200,
    )
    k_field = ft.TextField(label="Last K", value="6", width=120, keyboard_type=ft.KeyboardType.NUMBER)
    max_chars_field = ft.TextField(label="Max Input Chars", value="", width=160, keyboard_type=ft.KeyboardType.NUMBER)
    merge_same_id_cb = ft.Checkbox(label="Merge same poster", value=True)
    require_question_cb = ft.Checkbox(label="Require question in context", value=False)

    def update_context_controls():
        is_ctx = bool(multiturn_sw.value)
        strategy_dd.visible = is_ctx
        k_field.visible = is_ctx
        max_chars_field.visible = is_ctx
        merge_same_id_cb.visible = is_ctx
        require_question_cb.visible = is_ctx
        page.update()
    multiturn_sw.on_change = lambda e: update_context_controls()
    update_context_controls()

    # Toggle visibility between 4chan, Reddit and StackExchange controls
    def update_source_controls():
        src = (source_dd.value or "").strip().lower()
        is_reddit = (src == "reddit")
        is_se = (src == "stackexchange")
        # Boards area (4chan only)
        try:
            is_4chan = not (is_reddit or is_se)
            boards_wrap.visible = is_4chan
            board_actions.visible = is_4chan
            board_warning.visible = is_4chan
            # Hide the entire 4chan Boards section when source is not 4chan
            try:
                ctl = boards_section_ref.get("control")
                if ctl is not None:
                    ctl.visible = is_4chan
            except Exception:
                pass
        except Exception:
            pass
        # Reddit params
        try:
            reddit_params_row.visible = is_reddit
        except Exception:
            pass
        # StackExchange params
        try:
            se_params_row.visible = is_se
        except Exception:
            pass
        # Parameters visibility
        max_threads.visible = not (is_reddit or is_se)  # 4chan-specific
        max_pairs.visible = not is_reddit               # used by 4chan and StackExchange
        # Pairing/context controls apply to 4chan and Reddit, hide for StackExchange
        for ctl in [multiturn_sw, strategy_dd, k_field, max_chars_field, merge_same_id_cb, require_question_cb]:
            try:
                ctl.visible = not is_se
            except Exception:
                pass
        page.update()

    source_dd.on_change = lambda e: (update_source_controls(), update_board_validation())

    scrape_prog = ft.ProgressBar(width=400, value=0)
    # Animated indicator shown while scraping to make progress feel more alive
    working_ring = ft.ProgressRing(width=20, height=20, value=None, visible=False)
    threads_label = ft.Text("Threads Visited: 0")
    pairs_label = ft.Text("Pairs Found: 0")
    stats_cards = ft.Row([
        ft.Container(pill("Threads Visited: 0", COLORS.BLUE, ICONS.TRAVEL_EXPLORE),
                     padding=10),
        ft.Container(pill("Pairs Found: 0", COLORS.GREEN, ICONS.CHAT),
                     padding=10),
    ])
    stats_label_map = {"threads": threads_label, "pairs": pairs_label}

    # Live log
    log_list = ft.ListView(expand=1, auto_scroll=True, spacing=4)
    log_placeholder = make_empty_placeholder("No logs yet", ICONS.TERMINAL)
    log_area = ft.Stack([log_list, log_placeholder], expand=True)

    # Preview host: flex-based two-column grid (ListView of Rows)
    preview_host = ft.ListView(expand=1, auto_scroll=False)
    preview_placeholder = make_empty_placeholder("Preview not available", ICONS.PREVIEW)
    preview_area = ft.Stack([preview_host, preview_placeholder], expand=True)

    # ---------- SETTINGS (Proxy) CONTROLS ----------
    proxy_enable_cb = ft.Checkbox(label="Enable proxy (override defaults)", value=False)
    proxy_url_tf = ft.TextField(
        label="Proxy URL (e.g., socks5h://127.0.0.1:9050)",
        value="",
        width=380,
    )
    use_env_cb = ft.Checkbox(label="Use environment proxies (HTTP(S)_PROXY)", value=False)

    def update_proxy_controls(_=None):
        en = bool(proxy_enable_cb.value)
        use_env = bool(use_env_cb.value)
        use_env_cb.visible = en
        proxy_url_tf.visible = en and not use_env
        page.update()

    proxy_enable_cb.on_change = update_proxy_controls
    use_env_cb.on_change = update_proxy_controls
    update_proxy_controls()

    # Shared unified settings file for Hugging Face, Runpod, and Ollama
    SETTINGS_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ff_settings.json")
    _HF_LEGACY_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hf_config.json")
    _RUNPOD_LEGACY_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runpod_config.json")
    _OLLAMA_LEGACY_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ollama_config.json")

    def _load_settings_file() -> dict:
        try:
            with open(SETTINGS_CFG_PATH, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            data: dict = {}
            # Best-effort migration from legacy per-service config files
            try:
                with open(_HF_LEGACY_CFG_PATH, "r", encoding="utf-8") as f:
                    hf_cfg = json.load(f) or {}
                tok = (hf_cfg.get("token") or "").strip()
                if tok:
                    data.setdefault("huggingface", {})["token"] = tok
            except Exception:
                pass
            try:
                with open(_RUNPOD_LEGACY_CFG_PATH, "r", encoding="utf-8") as f:
                    rp_cfg = json.load(f) or {}
                key = (rp_cfg.get("api_key") or "").strip()
                if key:
                    data.setdefault("runpod", {})["api_key"] = key
            except Exception:
                pass
            try:
                with open(_OLLAMA_LEGACY_CFG_PATH, "r", encoding="utf-8") as f:
                    ollama_cfg = json.load(f) or {}
                if isinstance(ollama_cfg, dict) and ollama_cfg:
                    data["ollama"] = ollama_cfg
            except Exception:
                pass
            return data

    def _save_settings_file(data: dict) -> None:
        try:
            with open(SETTINGS_CFG_PATH, "w", encoding="utf-8") as f:
                json.dump(data or {}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ---------- SETTINGS (Hugging Face) CONTROLS ----------

    def _load_hf_config() -> dict:
        try:
            all_cfg = _load_settings_file()
            hf_cfg = all_cfg.get("huggingface") or {}
        except Exception:
            hf_cfg = {}
        return {"token": (hf_cfg.get("token") or "")}

    def _save_hf_config(cfg: dict):
        try:
            all_cfg = _load_settings_file()
            if not isinstance(all_cfg, dict):
                all_cfg = {}
            tok = (cfg.get("token") or "").strip()
            section = all_cfg.get("huggingface") or {}
            if not isinstance(section, dict):
                section = {}
            section["token"] = tok
            all_cfg["huggingface"] = section
            _save_settings_file(all_cfg)
        except Exception:
            pass

    def _apply_hf_env_from_cfg(cfg: dict):
        tok = (cfg.get("token") or "").strip()
        if tok:
            os.environ["HF_TOKEN"] = tok
        else:
            try:
                if os.environ.get("HF_TOKEN"):
                    del os.environ["HF_TOKEN"]
            except Exception:
                pass

    _hf_cfg = _load_hf_config()
    _apply_hf_env_from_cfg(_hf_cfg)

    hf_token_tf = ft.TextField(label="Hugging Face API token", password=True, can_reveal_password=True, width=420)
    hf_status = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))

    async def on_test_hf():
        tok = (hf_token_tf.value or "").strip() or (_hf_cfg.get("token") or "").strip()
        if not tok:
            hf_status.value = "No token provided or saved"
            await safe_update(page)
            return
        if HfApi is None:
            hf_status.value = "huggingface_hub not available"
            await safe_update(page)
            return
        hf_status.value = "Testing token…"
        await safe_update(page)
        try:
            api = HfApi()
            who = await asyncio.to_thread(lambda: api.whoami(token=tok))
            name = who.get("name") or who.get("email") or who.get("username") or "user"
            hf_status.value = f"Valid ✓ — {name}"
        except Exception as e:
            hf_status.value = f"Invalid or error: {e}"
        await safe_update(page)

    def on_save_hf(_):
        tok = (hf_token_tf.value or "").strip()
        if not tok:
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Enter a token to save"))
                page.open(page.snack_bar)
            except Exception:
                pass
            return
        _hf_cfg["token"] = tok
        _save_hf_config(_hf_cfg)
        _apply_hf_env_from_cfg(_hf_cfg)
        hf_status.value = "Saved"
        page.update()

    def on_remove_hf(_):
        _hf_cfg["token"] = ""
        _save_hf_config(_hf_cfg)
        _apply_hf_env_from_cfg(_hf_cfg)
        hf_token_tf.value = ""
        hf_status.value = "Removed"
        page.update()

    hf_test_btn = ft.ElevatedButton("Test token", icon=ICONS.CHECK_CIRCLE, on_click=lambda e: page.run_task(on_test_hf))
    hf_save_btn = ft.OutlinedButton("Save", icon=ICONS.SAVE, on_click=on_save_hf)
    hf_remove_btn = ft.TextButton("Remove", icon=getattr(ICONS, "DELETE", ICONS.CANCEL), on_click=on_remove_hf)

    # ---------- SETTINGS (Runpod) CONTROLS ----------

    def _load_runpod_config() -> dict:
        try:
            all_cfg = _load_settings_file()
            rp_cfg = all_cfg.get("runpod") or {}
        except Exception:
            rp_cfg = {}
        return {"api_key": (rp_cfg.get("api_key") or "")}

    def _save_runpod_config(cfg: dict):
        try:
            all_cfg = _load_settings_file()
            if not isinstance(all_cfg, dict):
                all_cfg = {}
            key = (cfg.get("api_key") or "").strip()
            section = all_cfg.get("runpod") or {}
            if not isinstance(section, dict):
                section = {}
            section["api_key"] = key
            all_cfg["runpod"] = section
            _save_settings_file(all_cfg)
        except Exception:
            pass

    def _apply_runpod_env_from_cfg(cfg: dict):
        key = (cfg.get("api_key") or "").strip()
        if key:
            os.environ["RUNPOD_API_KEY"] = key
        else:
            try:
                if os.environ.get("RUNPOD_API_KEY"):
                    del os.environ["RUNPOD_API_KEY"]
            except Exception:
                pass

    _runpod_cfg = _load_runpod_config()
    _apply_runpod_env_from_cfg(_runpod_cfg)

    runpod_key_tf = ft.TextField(label="Runpod API key", password=True, can_reveal_password=True, width=420)
    runpod_status = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))

    async def on_test_runpod():
        key = (runpod_key_tf.value or "").strip() or (_runpod_cfg.get("api_key") or "").strip()
        if not key:
            runpod_status.value = "No key provided or saved"
            await safe_update(page)
            return
        runpod_status.value = "Testing key…"
        await safe_update(page)
        # Try a couple of public endpoints that typically require auth; report status.
        urls = [
            "https://api.runpod.ai/v2/endpoints",
            "https://api.runpod.io/v2/endpoints",
        ]
        last_err = None
        for u in urls:
            try:
                def do_req():
                    return httpx.get(u, headers={"Authorization": f"Bearer {key}"}, timeout=6)
                resp = await asyncio.to_thread(do_req)
                if resp.status_code == 200:
                    runpod_status.value = "Valid ✓ — endpoints accessible"
                    await safe_update(page)
                    return
                elif resp.status_code in (401, 403):
                    runpod_status.value = f"Invalid or unauthorized ({resp.status_code})"
                    await safe_update(page)
                    return
                else:
                    last_err = f"HTTP {resp.status_code}"
            except Exception as e:
                last_err = str(e)
        runpod_status.value = f"Could not verify key via public endpoints ({last_err or 'unknown error'})"
        await safe_update(page)

    def on_save_runpod(_):
        key = (runpod_key_tf.value or "").strip()
        if not key:
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Enter a key to save"))
                page.open(page.snack_bar)
            except Exception:
                pass
            return
        _runpod_cfg["api_key"] = key
        _save_runpod_config(_runpod_cfg)
        _apply_runpod_env_from_cfg(_runpod_cfg)
        runpod_status.value = "Saved"
        page.update()

    def on_remove_runpod(_):
        _runpod_cfg["api_key"] = ""
        _save_runpod_config(_runpod_cfg)
        _apply_runpod_env_from_cfg(_runpod_cfg)
        runpod_key_tf.value = ""
        runpod_status.value = "Removed"
        page.update()

    runpod_test_btn = ft.ElevatedButton("Test key", icon=ICONS.CHECK_CIRCLE, on_click=lambda e: page.run_task(on_test_runpod))
    runpod_save_btn = ft.OutlinedButton("Save", icon=ICONS.SAVE, on_click=on_save_runpod)
    runpod_remove_btn = ft.TextButton("Remove", icon=getattr(ICONS, "DELETE", ICONS.CANCEL), on_click=on_remove_runpod)

    # ---------- SETTINGS (Ollama) CONTROLS ----------
    # Load persisted Ollama settings (with defaults applied for UI bindings)
    _ollama_raw = load_ollama_config_helper()
    _ollama_cfg = {
        "enabled": bool(_ollama_raw.get("enabled", False)),
        "base_url": (_ollama_raw.get("base_url") or "http://127.0.0.1:11434"),
        "default_model": (_ollama_raw.get("default_model") or ""),
        "selected_model": (_ollama_raw.get("selected_model") or ""),
    }

    ollama_enable_cb = ft.Checkbox(label="Enable Ollama connection", value=_ollama_cfg.get("enabled", False))
    ollama_base_url_tf = ft.TextField(label="Ollama base URL", value=_ollama_cfg.get("base_url", "http://127.0.0.1:11434"), width=420)
    ollama_default_model_tf = ft.TextField(label="Preferred model (optional)", value=_ollama_cfg.get("default_model", ""), width=300)
    ollama_models_dd = ft.Dropdown(label="Available models", options=[], value=_ollama_cfg.get("selected_model") or None, width=420, disabled=True)
    ollama_status = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))

    def update_ollama_controls(_=None):
        en = bool(ollama_enable_cb.value)
        for c in [ollama_base_url_tf, ollama_default_model_tf, ollama_models_dd]:
            c.disabled = not en
        page.update()

    async def on_test_ollama():
        if not bool(ollama_enable_cb.value):
            return
        base = (ollama_base_url_tf.value or "http://127.0.0.1:11434").strip()
        ollama_status.value = f"Testing connection to {base}…"
        await safe_update(page)
        try:
            data = await fetch_ollama_tags_helper(base)
            models = [m.get("name", "") for m in (data.get("models", []) or []) if m.get("name")]
            ollama_models_dd.options = [ft.dropdown.Option(n) for n in models]
            if models and not ollama_models_dd.value:
                ollama_models_dd.value = models[0]
            ollama_models_dd.disabled = False
            ollama_status.value = f"Connected ✓ — {len(models)} models"
        except Exception as e:
            ollama_models_dd.options = []
            ollama_status.value = f"Failed to connect: {e}"
        await safe_update(page)

    async def on_refresh_models():
        await on_test_ollama()

    def on_save_ollama(_):
        cfg = {
            "enabled": bool(ollama_enable_cb.value),
            "base_url": (ollama_base_url_tf.value or "http://127.0.0.1:11434").strip(),
            "default_model": (ollama_default_model_tf.value or "").strip(),
            "selected_model": (ollama_models_dd.value or "").strip(),
        }
        save_ollama_config_helper(cfg)
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Ollama settings saved"))
            page.open(page.snack_bar)
        except Exception:
            pass
        page.update()

    ollama_test_btn = ft.ElevatedButton("Test connection", icon=ICONS.CHECK_CIRCLE, on_click=lambda e: page.run_task(on_test_ollama))
    ollama_refresh_btn = ft.TextButton("Refresh models", icon=REFRESH_ICON, on_click=lambda e: page.run_task(on_refresh_models))
    ollama_save_btn = ft.OutlinedButton("Save", icon=ICONS.SAVE, on_click=on_save_ollama)

    try:
        ollama_enable_cb.on_change = update_ollama_controls
    except Exception:
        pass
    update_ollama_controls()

    # Actions
    cancel_state = {"cancelled": False}

    # Start button with validation state (default enabled due to defaults)
    start_button = ft.ElevatedButton(
        "Start", icon=ICONS.PLAY_ARROW,
        on_click=lambda e: page.run_task(on_start_scrape),
        disabled=False,
    )

    # Refs to top-level sections so we can toggle visibility from state helpers
    boards_section_ref = {}
    progress_section_ref = {}
    log_section_ref = {}
    preview_section_ref = {}

    def update_board_validation():
        # If scraping 4chan, enforce board selection; Reddit/StackExchange don't require boards
        if source_dd.value in ("reddit", "stackexchange"):
            start_button.disabled = False
            board_warning.value = ""
        else:
            any_selected = any(p.data and p.data.get("selected") for p in board_pills)
            start_button.disabled = not any_selected
            board_warning.value = "Select at least one board to scrape." if not any_selected else ""
        page.update()

    def update_scrape_placeholders():
        try:
            has_logs = len(getattr(log_list, "controls", []) or []) > 0
            has_preview = len(getattr(preview_host, "controls", []) or []) > 0
            has_progress = bool(working_ring.visible) or ((scrape_prog.value or 0) > 0)
            # Stack-level placeholders inside sections
            log_placeholder.visible = not has_logs
            preview_placeholder.visible = not has_preview
            # Section-level visibility via refs from tab_scrape
            try:
                ctl = progress_section_ref.get("control")
                if ctl is not None:
                    # Progress only needed once a scrape has started (and remains after)
                    ctl.visible = has_progress
            except Exception:
                pass
            try:
                ctl = log_section_ref.get("control")
                if ctl is not None:
                    # Live Log only visible while a scrape is actively running
                    ctl.visible = bool(working_ring.visible)
            except Exception:
                pass
            try:
                ctl = preview_section_ref.get("control")
                if ctl is not None:
                    # Preview visible only after run completes and we have something to show
                    ctl.visible = (not bool(working_ring.visible)) and has_preview
            except Exception:
                pass
        except Exception:
            pass
        page.update()

    # Attach change callbacks after start_button exists
    for p in board_pills:
        if p.data is None:
            p.data = {}
        p.data["on_change"] = update_board_validation
    update_board_validation()

    async def on_start_scrape():
        cancel_state["cancelled"] = False
        log_list.controls.clear()
        preview_host.controls.clear()
        scrape_prog.value = 0
        working_ring.visible = True
        update_scrape_placeholders()
        # Collect selected boards (only for 4chan)
        selected_boards = [p.data.get("label") for p in board_pills if p.data and p.data.get("selected")]
        if source_dd.value == "4chan" and not selected_boards:
            page.snack_bar = ft.SnackBar(ft.Text("Select at least one board to scrape."))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        # Parse params safely
        try:
            mt = int(max_threads.value or 50)
        except Exception:
            mt = 50
        try:
            mp = int(max_pairs.value or 5000)
        except Exception:
            mp = 5000
        try:
            dl = float(delay.value or 1.0)
        except Exception:
            dl = 1.0
        try:
            ml = int(min_len.value or 3)
        except Exception:
            ml = 3
        out_path = output_path.value or "scraped_training_data.json"
        # Context params
        multiturn = bool(multiturn_sw.value)
        strat_val = (strategy_dd.value or "cumulative")
        try:
            k_val = int(k_field.value or 6)
        except Exception:
            k_val = 6
        try:
            max_chars_val = int(max_chars_field.value) if (max_chars_field.value or "").strip() != "" else None
        except Exception:
            max_chars_val = None

        # High-level run summary line
        if source_dd.value == "reddit":
            log_list.controls.append(ft.Text(
                f"Reddit URL: {reddit_url.value} | Max posts: {reddit_max_posts.value}"
            ))
        elif source_dd.value == "stackexchange":
            log_list.controls.append(ft.Text(
                f"StackExchange site: {se_site.value} | Max pairs: {max_pairs.value}"
            ))
        else:
            log_list.controls.append(ft.Text(
                f"Boards: {', '.join(selected_boards[:20])}{' ...' if len(selected_boards)>20 else ''}"
            ))
        # Log chosen dataset format
        try:
            log_list.controls.append(ft.Text(f"Dataset format: {dataset_format_dd.value}"))
        except Exception:
            pass
        await safe_update(page)
        # Now that we have at least one log entry, hide the 'No logs yet' placeholder
        update_scrape_placeholders()

        # Disable Start while running
        start_button.disabled = True
        start_button.text = "Running..."
        start_button.icon = ICONS.HOURGLASS_TOP if hasattr(ICONS, "HOURGLASS_TOP") else ICONS.HOURGLASS_EMPTY
        await safe_update(page)

        try:
            if source_dd.value == "reddit":
                # Branch: Reddit scraper
                try:
                    rp = int(reddit_max_posts.value or 30)
                except Exception:
                    rp = 30
                await run_reddit_scrape_helper(
                    page=page,
                    log_view=log_list,
                    prog=scrape_prog,
                    labels={"threads": threads_label, "pairs": pairs_label},
                    preview_host=preview_host,
                    cancel_flag=cancel_state,
                    url=reddit_url.value or "https://www.reddit.com/",
                    max_posts=rp,
                    delay=dl,
                    min_len_val=ml,
                    output_path=out_path,
                    multiturn=multiturn,
                    ctx_k=k_val,
                    ctx_max_chars=max_chars_val,
                    merge_same_id=bool(merge_same_id_cb.value),
                    require_question=bool(require_question_cb.value),
                    ui_proxy_enabled=bool(proxy_enable_cb.value),
                    ui_proxy_url=(proxy_url_tf.value or "").strip(),
                    ui_use_env_proxies=bool(use_env_cb.value),
                    dataset_format=(dataset_format_dd.value or "ChatML"),
                )
            elif source_dd.value == "stackexchange":
                await run_stackexchange_scrape_helper(
                    page=page,
                    log_view=log_list,
                    prog=scrape_prog,
                    labels={"threads": threads_label, "pairs": pairs_label},
                    preview_host=preview_host,
                    cancel_flag=cancel_state,
                    site=se_site.value or "stackoverflow",
                    max_pairs=mp,
                    delay=dl,
                    min_len_val=ml,
                    output_path=out_path,
                    ui_proxy_enabled=bool(proxy_enable_cb.value),
                    ui_proxy_url=(proxy_url_tf.value or "").strip(),
                    ui_use_env_proxies=bool(use_env_cb.value),
                    dataset_format=(dataset_format_dd.value or "ChatML"),
                )
            else:
                await run_real_scrape_helper(
                    page=page,
                    log_view=log_list,
                    prog=scrape_prog,
                    labels={"threads": threads_label, "pairs": pairs_label},
                    preview_host=preview_host,
                    cancel_flag=cancel_state,
                    boards=selected_boards,
                    max_threads=mt,
                    max_pairs_total=mp,
                    delay=dl,
                    min_len_val=ml,
                    output_path=out_path,
                    multiturn=multiturn,
                    ctx_strategy=strat_val,
                    ctx_k=k_val,
                    ctx_max_chars=max_chars_val,
                    merge_same_id=bool(merge_same_id_cb.value),
                    require_question=bool(require_question_cb.value),
                    ui_proxy_enabled=bool(proxy_enable_cb.value),
                    ui_proxy_url=(proxy_url_tf.value or "").strip(),
                    ui_use_env_proxies=bool(use_env_cb.value),
                    dataset_format=(dataset_format_dd.value or "ChatML"),
                )
        except Exception as e:
            try:
                log_list.controls.append(ft.Text(f"Scrape failed: {e}"))
            except Exception:
                pass
            page.snack_bar = ft.SnackBar(ft.Text(f"Scrape failed: {e}"))
            page.open(page.snack_bar)
            await safe_update(page)
        finally:
            start_button.disabled = False
            start_button.text = "Start"
            start_button.icon = ICONS.PLAY_ARROW
            working_ring.visible = False
            update_board_validation()
            update_scrape_placeholders()
            await safe_update(page)

    def on_cancel_scrape(_):
        cancel_state["cancelled"] = True

    def on_reset_scrape(_):
        cancel_state["cancelled"] = False
        log_list.controls.clear()
        preview_host.controls.clear()
        scrape_prog.value = 0
        threads_label.value = "Threads Visited: 0"
        pairs_label.value = "Pairs Found: 0"
        working_ring.visible = False
        update_board_validation()
        update_scrape_placeholders()

    def on_refresh_scrape(_):
        nonlocal boards, board_pills, boards_wrap
        # Reload boards from API and rebuild chips
        boards = load_4chan_boards()
        new_pills: List[ft.Container] = [
            make_selectable_pill(b, selected=(b in {"pol", "b", "x"}), base_color=COLORS.AMBER)
            for b in boards
        ]
        board_pills = new_pills
        if hasattr(boards_wrap, "controls"):
            boards_wrap.controls.clear()
            boards_wrap.controls.extend(board_pills)
        # Re-wire validation callbacks
        for p in board_pills:
            if p.data is None:
                p.data = {}
            p.data["on_change"] = update_board_validation
        # Reset scrape area UI
        on_reset_scrape(_)
        page.update()

    async def on_preview_dataset():
        """Open a modal dialog showing the full dataset from the output JSON path."""
        # Immediate feedback that the click was received
        page.snack_bar = ft.SnackBar(ft.Text("Opening dataset preview..."))
        page.open(page.snack_bar)
        await safe_update(page)

        # Resolve dataset path robustly (supports launching app from different CWDs)
        orig_path = output_path.value or "scraped_training_data.json"
        candidates = []
        if os.path.isabs(orig_path):
            candidates.append(orig_path)
        else:
            candidates.extend([
                orig_path,
                os.path.abspath(orig_path),
                os.path.join(os.getcwd(), orig_path),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), orig_path),
            ])
        # Deduplicate while preserving order
        seen = set(); resolved_list = []
        for pth in candidates:
            ap = os.path.abspath(pth)
            if ap not in seen:
                seen.add(ap); resolved_list.append(ap)
        existing = next((p for p in resolved_list if os.path.exists(p)), None)
        if not existing:
            page.snack_bar = ft.SnackBar(ft.Text(
                "Dataset file not found. Tried:\n" + "\n".join(resolved_list[:4])
            ))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        try:
            data = await asyncio.to_thread(
                lambda: json.load(open(existing, "r", encoding="utf-8"))
            )
            if not isinstance(data, list):
                raise ValueError("Expected a JSON list of records")
        except Exception as e:
            page.snack_bar = ft.SnackBar(ft.Text(f"Failed to open {existing}: {e}"))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        # Paginated flex-grid viewer to avoid heavy UI rendering for large datasets
        page_size = 100
        total = len(data)
        total_pages = max(1, (total + page_size - 1) // page_size)
        state = {"page": 0}

        grid_list = ft.ListView(expand=1, auto_scroll=False)
        info_text = ft.Text("")

        # Detect dataset type once (assumes uniform list)
        try:
            first = next((x for x in data if isinstance(x, dict)), {})
            is_chatml_dataset = isinstance(first.get("messages"), list)
        except Exception:
            is_chatml_dataset = False

        # Navigation buttons
        prev_btn = ft.TextButton("Prev")
        next_btn = ft.TextButton("Next")

        def _extract_pair(rec: dict) -> tuple[str, str]:
            """Return (input, output) for either Standard pairs or ChatML messages."""
            try:
                # ChatML detection: record has a list under 'messages'
                msgs = rec.get("messages")
                if isinstance(msgs, list) and msgs:
                    user_text = None
                    assistant_text = None
                    for m in msgs:
                        if not isinstance(m, dict):
                            continue
                        role = m.get("role")
                        text = m.get("content") or ""
                        if role == "user" and user_text is None and text:
                            user_text = text
                        elif role == "assistant" and user_text is not None and text:
                            assistant_text = text
                            break
                    if user_text and assistant_text:
                        return (user_text, assistant_text)
                # Fallback to Standard pairs
                return (str(rec.get("input", "") or ""), str(rec.get("output", "") or ""))
            except Exception:
                return ("", "")

        def render_page():
            start = state["page"] * page_size
            end = min(start + page_size, total)
            grid_list.controls.clear()
            # Compute dynamic flex for current page
            page_samples = [_extract_pair(r if isinstance(r, dict) else {}) for r in data[start:end]]
            lfx, rfx = compute_two_col_flex(page_samples)
            hdr_left = "User" if is_chatml_dataset else "Input"
            hdr_right = "Assistant" if is_chatml_dataset else "Output"
            grid_list.controls.append(two_col_header(hdr_left, hdr_right, left_flex=lfx, right_flex=rfx))
            for a, b in page_samples:
                grid_list.controls.append(two_col_row(a, b, lfx, rfx))
            info_text.value = f"Page {state['page']+1}/{total_pages} • Showing {start+1}-{end} of {total}"
            prev_btn.disabled = state["page"] <= 0
            next_btn.disabled = state["page"] >= (total_pages - 1)
            page.update()

        def on_prev(_):
            if state["page"] > 0:
                state["page"] -= 1
                render_page()

        def on_next(_):
            if state["page"] < (total_pages - 1):
                state["page"] += 1
                render_page()

        prev_btn.on_click = on_prev
        next_btn.on_click = on_next

        controls_bar = ft.Row([
            prev_btn,
            next_btn,
            info_text,
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        full_scroll = grid_list

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Dataset Viewer — {len(data)} rows"),
            content=ft.Container(
                width=900,
                height=600,
                content=ft.Column([
                    controls_bar,
                    ft.Container(full_scroll, expand=True),
                ], expand=True),
            ),
            actions=[],
        )

        def close_dlg(_):
            dlg.open = False
            page.update()

        dlg.actions = [ft.TextButton("Close", on_click=close_dlg)]
        # Ensure page updates on dismiss across Flet versions
        try:
            dlg.on_dismiss = lambda e: page.update()
        except Exception:
            pass
        # Prepare first page before showing
        render_page()
        # Try new page.open() API first; fall back to legacy page.dialog
        opened = False
        try:
            if hasattr(page, "open") and callable(getattr(page, "open")):
                page.open(dlg)
                opened = True
        except Exception:
            opened = False
        if not opened:
            page.dialog = dlg
            dlg.open = True
        await safe_update(page)

    async def on_preview_raw_dataset():
        """Open a modal dialog showing the raw JSON contents of the output path."""
        # Resolve dataset path similarly to on_preview_dataset
        orig_path = output_path.value or "scraped_training_data.json"
        candidates = []
        if os.path.isabs(orig_path):
            candidates.append(orig_path)
        else:
            candidates.extend([
                orig_path,
                os.path.abspath(orig_path),
                os.path.join(os.getcwd(), orig_path),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), orig_path),
            ])
        seen = set(); resolved_list = []
        for pth in candidates:
            ap = os.path.abspath(pth)
            if ap not in seen:
                seen.add(ap); resolved_list.append(ap)
        existing = next((p for p in resolved_list if os.path.exists(p)), None)
        if not existing:
            page.snack_bar = ft.SnackBar(ft.Text(
                "Dataset file not found. Tried:\n" + "\n".join(resolved_list[:4])
            ))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        # Read raw text content
        try:
            raw_text = await asyncio.to_thread(lambda: open(existing, "r", encoding="utf-8").read())
        except Exception as e:
            page.snack_bar = ft.SnackBar(ft.Text(f"Failed to read {existing}: {e}"))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        # Build a scrollable text area
        text_ctl = ft.Text(raw_text, size=12, no_wrap=False, max_lines=None, selectable=True)
        content_ctl = ft.Container(
            content=ft.Column([text_ctl], expand=True, scroll=ft.ScrollMode.AUTO, spacing=0),
            width=900,
            height=600,
        )

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Raw Dataset — {os.path.basename(existing)}"),
            content=content_ctl,
            actions=[],
        )

        def close_dlg(_):
            dlg.open = False
            page.update()

        dlg.actions = [ft.TextButton("Close", on_click=close_dlg)]
        try:
            dlg.on_dismiss = lambda e: page.update()
        except Exception:
            pass

        opened = False
        try:
            if hasattr(page, "open") and callable(getattr(page, "open")):
                page.open(dlg)
                opened = True
        except Exception:
            opened = False
        if not opened:
            page.dialog = dlg
            dlg.open = True
        await safe_update(page)

    scrape_actions = ft.Row([
        start_button,
        ft.OutlinedButton("Cancel", icon=ICONS.CANCEL, on_click=on_cancel_scrape),
        ft.TextButton("Reset", icon=ICONS.RESTART_ALT, on_click=on_reset_scrape),
        ft.TextButton("Refresh", icon=REFRESH_ICON, on_click=on_refresh_scrape),
    ], spacing=10)

    # Robust scheduler helper for async tasks
    def schedule_task(coro):
        try:
            if hasattr(page, "run_task") and callable(getattr(page, "run_task")):
                return page.run_task(coro)
        except Exception:
            pass
        try:
            return asyncio.create_task(coro())
        except Exception:
            # As a last resort, run in thread
            return asyncio.get_event_loop().run_in_executor(None, lambda: asyncio.run(coro()))

    def handle_preview_click(_):
        # Immediate feedback that click fired
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Opening dataset preview..."))
            page.snack_bar.open = True
            page.update()
        except Exception:
            pass
        schedule_task(on_preview_dataset)

    def handle_raw_preview_click(_):
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Opening raw dataset..."))
            page.snack_bar.open = True
            page.update()
        except Exception:
            pass
        schedule_task(on_preview_raw_dataset)

    # Source selector for dataset preview/processing
    source_mode = ft.Dropdown(
        options=[
            ft.dropdown.Option("JSON file"),
            ft.dropdown.Option("Merged dataset"),
        ],
        value="JSON file",
        width=180,
    )
    # Rows that are toggled by update_source_controls()
    reddit_params_row = ft.Row([reddit_url, reddit_max_posts], wrap=True, visible=False)
    se_params_row = ft.Row([se_site], wrap=True, visible=False)
    # Initialize source-specific visibility now that rows exist
    try:
        update_source_controls()
    except Exception:
        pass

    # Compose Scrape tab via builder (layout only; logic/state remain in main)
    scrape_tab = build_scrape_tab(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        source_dd=source_dd,
        board_actions=board_actions,
        boards_wrap=boards_wrap,
        board_warning=board_warning,
        reddit_params_row=reddit_params_row,
        se_params_row=se_params_row,
        max_threads=max_threads,
        max_pairs=max_pairs,
        delay=delay,
        min_len=min_len,
        output_path=output_path,
        dataset_format_dd=dataset_format_dd,
        multiturn_sw=multiturn_sw,
        strategy_dd=strategy_dd,
        k_field=k_field,
        max_chars_field=max_chars_field,
        merge_same_id_cb=merge_same_id_cb,
        require_question_cb=require_question_cb,
        scrape_actions=scrape_actions,
        scrape_prog=scrape_prog,
        working_ring=working_ring,
        stats_cards=stats_cards,
        threads_label=threads_label,
        pairs_label=pairs_label,
        log_area=log_area,
        preview_area=preview_area,
        handle_preview_click=handle_preview_click,
        handle_raw_preview_click=handle_raw_preview_click,
        boards_section_ref=boards_section_ref,
        progress_section_ref=progress_section_ref,
        log_section_ref=log_section_ref,
        preview_section_ref=preview_section_ref,
    )
    # Ensure initial visibility matches idle state (no logs or preview yet)
    update_scrape_placeholders()
    # Data source and processing controls
    data_file = ft.TextField(label="Data file (JSON)", value="scraped_training_data.json", width=360)
    merged_dir = ft.TextField(label="Merged dataset dir", value="merged_dataset", width=240)
    seed = ft.TextField(label="Seed", value="42", width=120, keyboard_type=ft.KeyboardType.NUMBER)
    shuffle = ft.Switch(label="Shuffle", value=True)
    val_slider = ft.Slider(min=0, max=0.2, value=0.01, divisions=20, label="{value}")
    test_slider = ft.Slider(min=0, max=0.2, value=0.0, divisions=20, label="{value}")
    min_len_b = ft.TextField(label="Min Length", value="1", width=120, keyboard_type=ft.KeyboardType.NUMBER)
    save_dir = ft.TextField(label="Save dir", value="hf_dataset", width=240)

    push_toggle = ft.Switch(label="Push to Hub", value=False)
    repo_id = ft.TextField(label="Repo ID", value="username/my-dataset", width=280)
    private = ft.Switch(label="Private", value=True)
    token_val_ui = ft.TextField(label="HF Token", password=True, can_reveal_password=True, width=320)
    saved_tok = ((_hf_cfg.get("token") or "").strip() if isinstance(_hf_cfg, dict) else "")
    token_val = saved_tok or token_val_ui.value

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

    # Toggle UI fields based on source selection (JSON vs Merged dataset)
    def on_source_change(_):
        mode = (source_mode.value or "JSON file").strip()
        is_json = mode == "JSON file"
        try:
            data_file.visible = is_json
            merged_dir.visible = not is_json
            # Enable JSON-only processing params for JSON mode; disable in merged mode
            for ctl in [seed, shuffle, min_len_b, val_slider, test_slider]:
                try:
                    ctl.disabled = not is_json
                except Exception:
                    pass
        except Exception:
            pass
        page.update()

    source_mode.on_change = on_source_change
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
    status_section_ref = {}

    cancel_build = {"cancelled": False}
    dd_ref = {"dd": None}
    push_state = {"inflight": False}
    push_ring = ft.ProgressRing(width=18, height=18, value=None, visible=False)
    # Reference to the model card preview container (assigned later)
    card_preview_container = None

    # --- Model Card Creator controls/state ---
    # Switch to enable custom model card instead of autogenerated
    use_custom_card = ft.Switch(label="Use custom model card (README.md)", value=False)

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

    card_preview_switch = ft.Switch(label="Live preview", value=False, disabled=True)
    card_preview_md = _make_md("")
    try:
        # Some Flet controls don't have 'visible'; guard accordingly
        card_preview_md.visible = False
    except Exception:
        pass

    # Dedicated preview container (hidden until we have content + preview enabled)
    card_preview_container = ft.Container(
        ft.Column([card_preview_md], scroll=ft.ScrollMode.AUTO, spacing=0),
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

    def _apply_preview_visibility():
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

    def _update_preview():
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

    async def _on_generate_from_dataset():
        # Generate using current built dataset (if available)
        dd = dd_ref.get("dd")
        if dd is None:
            page.snack_bar = ft.SnackBar(ft.Text("Build the dataset first to generate a default card."))
            page.open(page.snack_bar)
            await safe_update(page)
            return
        rid = (repo_id.value or "").strip()
        if not rid:
            page.snack_bar = ft.SnackBar(ft.Text("Enter Repo ID to generate a default card."))
            page.open(page.snack_bar)
            await safe_update(page)
            return
        try:
            use_custom_card.value = True
            _on_toggle_custom_card(None)
            content = await asyncio.to_thread(sd.build_dataset_card_content, dd, rid)
            card_editor.value = content
            _update_preview()
            await safe_update(page)
        except Exception as e:
            page.snack_bar = ft.SnackBar(ft.Text(f"Failed to generate card: {e}"))
            page.open(page.snack_bar)
            await safe_update(page)

    ollama_gen_status = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))

    async def _on_generate_with_ollama():
        # Generate using Ollama from the selected data file (JSON list of {input,output})
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
        model_name = (ollama_models_dd.value or cfg.get("selected_model") or cfg.get("default_model") or "").strip()
        if not model_name:
            page.snack_bar = ft.SnackBar(ft.Text("Select an Ollama model in Settings."))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        path = (data_file.value or "scraped_training_data.json").strip()
        try:
            records = await asyncio.to_thread(sd.load_records, path)
        except Exception as e:
            page.snack_bar = ft.SnackBar(ft.Text(f"Failed to load data file: {e}"))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        if not isinstance(records, list) or len(records) == 0:
            page.snack_bar = ft.SnackBar(ft.Text("Data file is empty or invalid (expected list of records)."))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        total_n = len(records)
        # Sample a small subset for context
        k = min(8, total_n)
        idxs = random.sample(range(total_n), k) if total_n >= k else list(range(total_n))
        samples = []
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
            size_cat = "n<1K" if total_n < 1_000 else ("1K<n<10K" if total_n < 10_000 else ("10K<n<100K" if total_n < 100_000 else ("100K<n<1M" if total_n < 1_000_000 else "n>1M")))

        rid = (repo_id.value or "username/dataset").strip()
        user_prompt = (
            f"You are an expert data curator. Create a professional Hugging Face dataset card (README.md) in Markdown for the dataset '{rid}'.\n"
            f"Use the provided random samples to infer characteristics. Include a YAML frontmatter header with tags, task_categories=text-generation, language=en, license=other, size_categories=[{size_cat}].\n"
            "Then include sections: Dataset Summary, Data Fields, Source and Collection, Splits (estimate if needed), Usage (datasets code snippet), Ethical Considerations and Warnings, Licensing, Example Records (re-embed the samples), How to Cite, Changelog.\n"
            "Keep the tone clear and factual. If unsure, state assumptions transparently."
        )
        samples_json = json.dumps(samples, ensure_ascii=False, indent=2)
        user_prompt += f"\n\nSamples (JSON):\n```json\n{samples_json}\n```\nTotal records (approx): {total_n}"

        system_prompt = (
            "You write concise, high-quality dataset cards for Hugging Face. Output ONLY valid Markdown starting with YAML frontmatter."
        )

        ollama_gen_status.value = f"Generating with Ollama model '{model_name}'…"
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
            ollama_gen_status.value = "Generated with Ollama ✓"
            await safe_update(page)
        except Exception as e:
            ollama_gen_status.value = f"Ollama generation failed: {e}"
            page.snack_bar = ft.SnackBar(ft.Text(ollama_gen_status.value))
            page.open(page.snack_bar)
            await safe_update(page)

    load_template_btn = ft.TextButton("Load simple template", icon=ICONS.ARTICLE, on_click=_on_load_simple_template)
    gen_from_ds_btn = ft.TextButton("Generate from built dataset", icon=ICONS.BUILD, on_click=lambda e: page.run_task(_on_generate_from_dataset))
    gen_with_ollama_btn = ft.ElevatedButton("Generate with Ollama", icon=getattr(ICONS, "SMART_TOY", ICONS.HUB), on_click=lambda e: page.run_task(_on_generate_with_ollama))
    clear_card_btn = ft.TextButton("Clear", icon=ICONS.BACKSPACE, on_click=lambda e: (setattr(card_editor, "value", ""), _update_preview(), page.update()))

    def update_status_placeholder():
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
            split_badges[k].content = pill(f"{label}: 0", split_meta[k][0], split_meta[k][1]).content
        push_state["inflight"] = False
        push_ring.visible = False
        # Re-enable push button if it was disabled
        for ctl in build_actions.controls:
            if isinstance(ctl, ft.TextButton) and "Push + Upload README" in getattr(ctl, "text", ""):
                ctl.disabled = False
        update_status_placeholder()

    async def on_build():
        # Delegate to helper to keep main.py slim
        hf_cfg_token = ((_hf_cfg.get("token") or "").strip() if isinstance(_hf_cfg, dict) else "")
        return await run_build_helper(
            page=page,
            source_mode=source_mode,
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
        # Delegate to helper to keep main.py slim
        hf_cfg_token = ((_hf_cfg.get("token") or "").strip() if isinstance(_hf_cfg, dict) else "")
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
            timeline.controls.append(ft.Row([ft.Icon(ICONS.CANCEL, color=COLORS.RED), ft.Text("Cancel requested — will stop ASAP")]))
            update_status_placeholder()
        except Exception:
            pass

    build_actions = ft.Row([
        ft.ElevatedButton("Build Dataset", icon=ICONS.BUILD, on_click=lambda e: page.run_task(on_build)),
        ft.OutlinedButton("Cancel", icon=ICONS.CANCEL, on_click=on_cancel_build),
        ft.TextButton("Refresh", icon=REFRESH_ICON, on_click=on_refresh_build),
        ft.TextButton("Push + Upload README", icon=ICONS.CLOUD_UPLOAD, on_click=lambda e: page.run_task(on_push_async)),
        push_ring,
    ], spacing=10)
    build_tab = build_build_tab(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        source_mode=source_mode,
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
        timeline=timeline,
        timeline_placeholder=timeline_placeholder,
        status_section_ref=status_section_ref,
    )
    update_status_placeholder()


    # ---------- MERGE DATASETS TAB ----------
    # Operation selector
    merge_op = ft.Dropdown(
        label="Operation",
        options=[
            ft.dropdown.Option("Concatenate"),
            ft.dropdown.Option("Interleave"),
        ],
        value="Concatenate",
        width=220,
    )

    # Dynamic dataset rows
    rows_host = ft.Column(spacing=8)

    def make_dataset_row():
        source_dd = ft.Dropdown(
            label="Source",
            options=[
                ft.dropdown.Option("Hugging Face"),
                ft.dropdown.Option("JSON file"),
            ],
            value="Hugging Face",
            width=160,
        )
        ds_id = ft.TextField(label="Dataset repo (e.g., username/dataset)", width=360, visible=True)
        split = ft.Dropdown(
            label="Split",
            options=[
                ft.dropdown.Option("train"),
                ft.dropdown.Option("validation"),
                ft.dropdown.Option("test"),
                ft.dropdown.Option("all"),
            ],
            value="train",
            width=160,
            visible=True,
        )
        config = ft.TextField(label="Config (optional)", width=180, visible=True)
        in_col = ft.TextField(label="Input column (optional)", width=200, visible=True)
        out_col = ft.TextField(label="Output column (optional)", width=200, visible=True)
        json_path = ft.TextField(label="JSON path", width=360, visible=False)
        remove_btn = ft.IconButton(ICONS.DELETE)
        row = ft.Row([source_dd, ds_id, split, config, in_col, out_col, json_path, remove_btn], spacing=10, wrap=True)

        # Keep references for later retrieval
        row.data = {
            "source": source_dd,
            "ds": ds_id,
            "split": split,
            "config": config,
            "in": in_col,
            "out": out_col,
            "json": json_path,
        }

        def on_source_change(_):
            is_hf = (getattr(source_dd, "value", "Hugging Face") or "Hugging Face") == "Hugging Face"
            ds_id.visible = is_hf
            split.visible = is_hf
            config.visible = is_hf
            in_col.visible = is_hf
            out_col.visible = is_hf
            json_path.visible = not is_hf
            try:
                page.update()
            except Exception:
                pass

        try:
            source_dd.on_change = on_source_change
        except Exception:
            pass

        def remove_row(_):
            try:
                rows_host.controls.remove(row)
                page.update()
            except Exception:
                pass

        remove_btn.on_click = remove_row
        return row

    def add_row(_=None):
        rows_host.controls.append(make_dataset_row())
        page.update()

    add_row_btn = ft.TextButton("Add Dataset", icon=ICONS.ADD, on_click=add_row)
    clear_btn = ft.TextButton("Clear", icon=ICONS.BACKSPACE, on_click=lambda e: (rows_host.controls.clear(), page.update()))

    # Output settings
    merge_output_format = ft.Dropdown(
        label="Output format",
        options=[ft.dropdown.Option("HF dataset dir"), ft.dropdown.Option("JSON file")],
        value="HF dataset dir",
        width=220,
    )
    merge_save_dir = ft.TextField(label="Save dir", value="merged_dataset", width=240)

    def update_output_controls(_=None):
        fmt = (merge_output_format.value or "").lower()
        if "json" in fmt:
            merge_save_dir.label = "Save file (.json)"
            if (merge_save_dir.value or "").strip() == "merged_dataset":
                merge_save_dir.value = "merged.json"
        else:
            merge_save_dir.label = "Save dir"
            if (merge_save_dir.value or "").strip() == "merged.json":
                merge_save_dir.value = "merged_dataset"
        try:
            page.update()
        except Exception:
            pass

    try:
        merge_output_format.on_change = update_output_controls
    except Exception:
        pass

    # Status & preview
    merge_timeline = ft.ListView(expand=1, auto_scroll=True, spacing=6)
    merge_timeline_placeholder = make_empty_placeholder("No status yet", ICONS.TASK)
    merge_preview_host = ft.ListView(expand=1, auto_scroll=False)
    merge_preview_placeholder = make_empty_placeholder("Preview not available", ICONS.PREVIEW)
    merge_status_section_ref = {}
    merge_preview_section_ref = {}

    merge_cancel = {"cancelled": False}
    merge_busy_ring = ft.ProgressRing(width=18, height=18, value=None, visible=False)

    def update_merge_placeholders():
        try:
            has_status = len(getattr(merge_timeline, "controls", []) or []) > 0
            has_preview = len(getattr(merge_preview_host, "controls", []) or []) > 0
            # Stack-level placeholders
            merge_timeline_placeholder.visible = not has_status
            merge_preview_placeholder.visible = not has_preview
            # Section-level visibility via refs from tab_merge
            try:
                ctl = merge_status_section_ref.get("control")
                if ctl is not None:
                    ctl.visible = has_status
            except Exception:
                pass
            try:
                ctl = merge_preview_section_ref.get("control")
                if ctl is not None:
                    ctl.visible = has_preview
            except Exception:
                pass
        except Exception:
            pass
        page.update()

    

    async def on_merge():
        # Delegate to helper to keep main.py slim
        return await run_merge_helper(
            page=page,
            rows_host=rows_host,
            merge_op=merge_op,
            merge_output_format=merge_output_format,
            merge_save_dir=merge_save_dir,
            merge_timeline=merge_timeline,
            merge_timeline_placeholder=merge_timeline_placeholder,
            merge_preview_host=merge_preview_host,
            merge_preview_placeholder=merge_preview_placeholder,
            merge_cancel=merge_cancel,
            merge_busy_ring=merge_busy_ring,
            download_button=download_merged_button,
            update_merge_placeholders=update_merge_placeholders,
        )


    def on_cancel_merge(_):
        merge_cancel["cancelled"] = True
        try:
            merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.CANCEL, color=COLORS.RED), ft.Text("Cancel requested — will stop ASAP")]))
            update_merge_placeholders(); page.update()
        except Exception:
            pass

    def on_refresh_merge(_):
        merge_cancel["cancelled"] = False
        merge_timeline.controls.clear()
        merge_preview_host.controls.clear()
        merge_busy_ring.visible = False
        download_merged_button.visible = False
        update_merge_placeholders(); page.update()

    async def on_preview_merged():
        # Delegate to helper preview
        return await preview_merged_helper(
            page=page,
            merge_output_format=merge_output_format,
            merge_save_dir=merge_save_dir,
        )

    def handle_merge_preview_click(_):
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Opening merged dataset preview..."))
            page.open(page.snack_bar)
            await_safe = False
            try:
                await_safe = hasattr(page, "update")
            except Exception:
                await_safe = False
            if await_safe:
                page.update()
        except Exception:
            pass
        # Use scheduler utility for consistency
        try:
            if hasattr(page, "run_task") and callable(getattr(page, "run_task")):
                page.run_task(on_preview_merged)
            else:
                asyncio.create_task(on_preview_merged())
        except Exception:
            asyncio.get_event_loop().run_in_executor(None, lambda: asyncio.run(on_preview_merged()))

    async def on_download_merged(e: ft.FilePickerResultEvent):
        """Handle downloading the merged dataset to a user-selected location."""
        logger.info(f"Download merged dataset called with destination: {e.path}")

        if e.path is None:
            logger.warning("Download cancelled: No destination selected")
            page.snack_bar = ft.SnackBar(ft.Text("No destination selected"))
            page.snack_bar.open = True
            await safe_update(page)
            return

        dest_dir = e.path
        orig_dir = merge_save_dir.value or "merged_dataset"
        fmt_now = (merge_output_format.value or "").lower()
        wants_json = ("json" in fmt_now) or (str(orig_dir).lower().endswith(".json"))

        logger.debug(f"Download params - dest_dir: {dest_dir}, orig_dir: {orig_dir}, wants_json: {wants_json}")

        # Find the source file
        candidates = []
        if os.path.isabs(orig_dir):
            candidates.append(orig_dir)
        else:
            candidates.extend([
                orig_dir,
                os.path.abspath(orig_dir),
                os.path.join(os.getcwd(), orig_dir),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), orig_dir),
            ])
        seen = set()
        resolved_list: List[str] = []
        for pth in candidates:
            ap = os.path.abspath(pth)
            if ap not in seen:
                seen.add(ap)
                resolved_list.append(ap)
        existing = next((p for p in resolved_list if os.path.exists(p)), None)

        logger.debug(f"Source search candidates: {candidates}")
        logger.debug(f"Found existing source: {existing}")

        if not existing:
            logger.error(f"Merged dataset not found. Searched: {orig_dir}")
            page.snack_bar = ft.SnackBar(ft.Text(f"Merged dataset not found. Searched: {orig_dir}"))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        # Use the original filename from the merge output path
        source_basename = os.path.basename(orig_dir)
        dest_path = os.path.join(dest_dir, source_basename)

        logger.info(f"Copying {source_basename} from {existing} to {dest_path}")

        try:
            import shutil
            if wants_json or os.path.isfile(existing):
                # Copy single file
                logger.debug(f"Starting file copy operation")
                await asyncio.to_thread(shutil.copy2, existing, dest_path)
                msg = f"Downloaded to {dest_path}"
                logger.info(f"File copy successful: {dest_path}")
            else:
                # Copy directory
                logger.debug(f"Starting directory copy operation")
                await asyncio.to_thread(shutil.copytree, existing, dest_path, dirs_exist_ok=True)
                msg = f"Downloaded to {dest_path}"
                logger.info(f"Directory copy successful: {dest_path}")

            page.snack_bar = ft.SnackBar(ft.Text(msg))
            page.open(page.snack_bar)
            await safe_update(page)
        except Exception as e:
            logger.error(f"Download failed - Source: {existing}, Dest: {dest_path}", exc_info=True)
            error_details = f"Download failed: {str(e)}"
            page.snack_bar = ft.SnackBar(ft.Text(error_details))
            page.open(page.snack_bar)
            await safe_update(page)

    async def handle_download_result(e: ft.FilePickerResultEvent):
        await on_download_merged(e)

    download_file_picker = ft.FilePicker(on_result=handle_download_result)
    page.overlay.append(download_file_picker)

    download_merged_button = ft.ElevatedButton(
        "Download Merged Dataset",
        icon=getattr(ICONS, "DOWNLOAD", ICONS.ARROW_DOWNWARD),
        on_click=lambda _: download_file_picker.get_directory_path(
            dialog_title="Select Download Location",
        ),
        visible=False,
    )

    merge_actions = ft.Row([
        ft.ElevatedButton("Merge Datasets", icon=ICONS.TABLE_VIEW, on_click=lambda e: page.run_task(on_merge)),
        ft.OutlinedButton("Cancel", icon=ICONS.CANCEL, on_click=on_cancel_merge),
        ft.TextButton("Refresh", icon=REFRESH_ICON, on_click=on_refresh_merge),
        ft.TextButton("Preview Merged", icon=ICONS.PREVIEW, on_click=handle_merge_preview_click),
        merge_busy_ring,
    ], spacing=10)

    merge_tab = build_merge_tab(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        merge_op=merge_op,
        rows_host=rows_host,
        add_row_btn=add_row_btn,
        clear_btn=clear_btn,
        merge_output_format=merge_output_format,
        merge_save_dir=merge_save_dir,
        merge_actions=merge_actions,
        merge_preview_host=merge_preview_host,
        merge_preview_placeholder=merge_preview_placeholder,
        merge_timeline=merge_timeline,
        merge_timeline_placeholder=merge_timeline_placeholder,
        download_button=download_merged_button,
        preview_section_ref=merge_preview_section_ref,
        status_section_ref=merge_status_section_ref,
    )
    update_merge_placeholders()


    # ---------- TRAINING TAB ----------
    # Dataset source
    train_source = ft.Dropdown(
        label="Dataset source",
        options=[
            ft.dropdown.Option("Hugging Face"),
            ft.dropdown.Option("JSON file"),
        ],
        value="Hugging Face",
        width=180,
    )
    train_hf_repo = ft.TextField(label="Dataset repo (e.g., username/dataset)", width=360, visible=True)
    train_hf_split = ft.Dropdown(
        label="Split",
        options=[ft.dropdown.Option("train"), ft.dropdown.Option("validation"), ft.dropdown.Option("test")],
        value="train",
        width=140,
        visible=True,
    )
    train_hf_config = ft.TextField(label="Config (optional)", width=180, visible=True)
    train_json_path = ft.TextField(label="JSON path", width=360, visible=False)

    def _update_train_source(_=None):
        is_hf = (getattr(train_source, "value", "Hugging Face") or "Hugging Face") == "Hugging Face"
        train_hf_repo.visible = is_hf
        train_hf_split.visible = is_hf
        train_hf_config.visible = is_hf
        train_json_path.visible = (not is_hf)
        try:
            page.update()
        except Exception:
            pass

    # Training parameters
    skill_level = ft.Dropdown(
        label="Skill level",
        options=[ft.dropdown.Option("Beginner"), ft.dropdown.Option("Expert")],
        value="Beginner",
        width=160,
    )
    beginner_mode_dd = ft.Dropdown(
        label="Beginner mode",
        options=[ft.dropdown.Option("Fastest"), ft.dropdown.Option("Cheapest")],
        value="Fastest",
        width=160,
        visible=True,
        tooltip="For Beginner: Fastest uses best GPU with aggressive params; Cheapest uses lowest-cost GPU with conservative params.",
    )
    # Expert-mode GPU picker (hidden by default)
    expert_gpu_dd = ft.Dropdown(
        label="GPU (Expert)",
        options=[ft.dropdown.Option("AUTO")],
        value="AUTO",
        width=260,
        visible=False,
        tooltip="Pick a GPU type available in the selected datacenter. 'AUTO' will pick the best available secure GPU.",
    )
    expert_spot_cb = ft.Checkbox(
        label="Use Spot (interruptible)",
        value=False,
        visible=False,
        tooltip="When enabled and available, a spot/interruptible pod is used.",
    )
    expert_gpu_refresh_btn = ft.IconButton(
        icon=getattr(ICONS, "REFRESH", getattr(ICONS, "AUTORENEW", getattr(ICONS, "UPDATE", getattr(ICONS, "SYNC", getattr(ICONS, "CACHED", ICONS.REFRESH))))),
        tooltip="Refresh available GPUs from Runpod",
        visible=False,
    )
    expert_gpu_busy = ft.ProgressRing(width=18, height=18, value=None, visible=False)
    # Map gpu_id -> availability flags to drive spot toggle enabling
    expert_gpu_avail: dict = {}
    def _update_expert_spot_enabled(_=None):
        try:
            gid = (expert_gpu_dd.value or "AUTO")
            flags = expert_gpu_avail.get(gid) or {}
            sec_ok = bool(flags.get("secureAvailable"))
            spot_ok = bool(flags.get("spotAvailable"))
            # Only enable checkbox if any mode is available; constrain value when not supported
            expert_spot_cb.disabled = not (spot_ok or sec_ok)
            if not spot_ok and bool(getattr(expert_spot_cb, "value", False)):
                expert_spot_cb.value = False
            expert_spot_cb.tooltip = f"Spot available: {spot_ok} • Secure available: {sec_ok}"
        except Exception:
            pass
        try:
            page.update()
        except Exception:
            pass
    expert_gpu_dd.on_change = _update_expert_spot_enabled
    base_model = ft.Dropdown(
        label="Base model",
        options=[
            ft.dropdown.Option("unsloth/Meta-Llama-3.1-8B-bnb-4bit"),  # Llama-3.1 15T tokens, 2x faster
            ft.dropdown.Option("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"),
            ft.dropdown.Option("unsloth/Meta-Llama-3.1-70B-bnb-4bit"),
            ft.dropdown.Option("unsloth/Meta-Llama-3.1-405B-bnb-4bit"),  # 4bit for 405B
            ft.dropdown.Option("unsloth/Mistral-Nemo-Base-2407-bnb-4bit"),  # New Mistral 12B, 2x faster
            ft.dropdown.Option("unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit"),
            ft.dropdown.Option("unsloth/mistral-7b-v0.3-bnb-4bit"),       # Mistral v3, 2x faster
            ft.dropdown.Option("unsloth/mistral-7b-instruct-v0.3-bnb-4bit"),
            ft.dropdown.Option("unsloth/Phi-3.5-mini-instruct"),          # Phi-3.5, 2x faster
            ft.dropdown.Option("unsloth/Phi-3-medium-4k-instruct"),
            ft.dropdown.Option("unsloth/gemma-2-9b-bnb-4bit"),
            ft.dropdown.Option("unsloth/gemma-2-27b-bnb-4bit"),
        ],
        value="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        width=320,
    )
    epochs_tf = ft.TextField(label="Epochs", value="3", width=120)
    lr_tf = ft.TextField(label="Learning rate", value="2e-4", width=160)
    batch_tf = ft.TextField(label="Per-device batch size", value="2", width=200)
    grad_acc_tf = ft.TextField(label="Grad accum steps", value="4", width=180)
    max_steps_tf = ft.TextField(label="Max steps", value="200", width=180)
    use_lora_cb = ft.Checkbox(label="Use LoRA", value=True)
    out_dir_tf = ft.TextField(label="Output dir", value="/data/outputs/runpod_run", width=260)
    # New HP toggles/fields
    packing_cb = ft.Checkbox(label="Packing", value=True, tooltip="Pack multiple samples into a sequence for higher utilization (if trainer supports).")
    auto_resume_cb = ft.Checkbox(label="Auto-resume", value=True, tooltip="Resume from latest checkpoint if container restarts.")
    push_cb = ft.Checkbox(
        label="Push to HF Hub",
        value=False,
        tooltip="When enabled, the trainer will attempt to push the final model/adapters to the Hugging Face Hub. Requires a valid HF token with write access.",
    )
    hf_repo_id_tf = ft.TextField(
        label="HF repo id (for push)",
        value="",
        width=280,
        hint_text="username/model-name",
        tooltip="Model repository on Hugging Face to push to (e.g., username/my-lora-model). You must own the repo or have write access and be authenticated.",
    )
    resume_from_tf = ft.TextField(
        label="Resume from (path)",
        value="",
        width=320,
        hint_text="/data/outputs/runpod_run/checkpoint-500",
        tooltip="Optional explicit checkpoint directory to resume from, inside the mounted volume (e.g., /data/outputs/runpod_run/checkpoint-500).",
    )

    # Info icons next to toggles/fields
    _info_icon = getattr(ICONS, "INFO_OUTLINE", getattr(ICONS, "INFO", getattr(ICONS, "HELP_OUTLINE", None)))

    packing_info = ft.IconButton(
        icon=_info_icon,
        tooltip="Packing combines multiple short samples into a fixed-length sequence to better utilize tokens (only if the training script supports it).",
        on_click=_mk_help_handler(
            "Packing: When enabled, the trainer may pack several shorter samples into a fixed-length training sequence to improve GPU utilization and throughput.\n\nWhen to use: If your training script supports packing and you have many short samples. If unsupported, leave it off."
        ),
    )
    auto_resume_info = ft.IconButton(
        icon=_info_icon,
        tooltip="Try to continue from the latest checkpoint in Output dir if the container restarts.",
        on_click=_mk_help_handler(
            "Auto-resume: On container restarts, the trainer looks for the latest checkpoint in your Output dir and continues training from it.\n\nRequirements: Keep Output dir on the persistent Runpod Network Volume and reuse the same Output dir for the same run."
        ),
    )
    push_info = ft.IconButton(
        icon=_info_icon,
        tooltip="Push final model/adapters to the Hugging Face Hub at the end of training.",
        on_click=_mk_help_handler(
            "Push to HF Hub: If enabled, the trainer will attempt to upload the resulting model (or LoRA adapters) to a Hugging Face model repository.\n\nProvide: • A valid HF token with write scope (Settings → Hugging Face Access) • The repo id as username/model-name.\nNote: Create the repo on the Hub first to ensure permissions."
        ),
    )
    hf_repo_info = ft.IconButton(
        icon=_info_icon,
        tooltip="Hugging Face model repo id, e.g., username/my-lora-model.",
        on_click=_mk_help_handler(
            "HF repo id (for push): The target model repository on Hugging Face to push your trained weights/adapters to.\n\nFormat: username/model-name (e.g., sbussiso/my-lora-phi3). You must own the repo or have collaborator write access. Authenticate via Settings → Hugging Face Access or HF_TOKEN env."
        ),
    )
    resume_info = ft.IconButton(
        icon=_info_icon,
        tooltip="Explicit checkpoint path inside /data, e.g., /data/outputs/runpod_run/checkpoint-500",
        on_click=_mk_help_handler(
            "Resume from (path): Force the trainer to resume from a specific checkpoint directory.\n\nExample: /data/outputs/runpod_run/checkpoint-500 or /data/outputs/runpod_run/last. Must exist on the mounted volume. Leave blank to let Auto-resume find the latest checkpoint automatically (if supported)."
        ),
    )

    # Group each control with its info icon
    packing_row = ft.Row([packing_cb, packing_info], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)
    auto_resume_row = ft.Row([auto_resume_cb, auto_resume_info], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)
    push_row = ft.Row([push_cb, push_info], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)
    hf_repo_row = ft.Row([hf_repo_id_tf, hf_repo_info], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)
    resume_from_row = ft.Row([resume_from_tf, resume_info], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    # Advanced parameters (Expert mode)
    warmup_steps_tf = ft.TextField(label="Warmup steps", value="10", width=140)
    weight_decay_tf = ft.TextField(label="Weight decay", value="0.01", width=140)
    lr_sched_dd = ft.Dropdown(
        label="LR scheduler",
        options=[ft.dropdown.Option("linear"), ft.dropdown.Option("cosine"), ft.dropdown.Option("constant")],
        value="linear",
        width=160,
    )
    optim_dd = ft.Dropdown(
        label="Optimizer",
        options=[ft.dropdown.Option("adamw_8bit"), ft.dropdown.Option("adamw_torch")],
        value="adamw_8bit",
        width=180,
    )
    logging_steps_tf = ft.TextField(label="Logging steps", value="25", width=160)
    logging_first_step_cb = ft.Checkbox(label="Log first step", value=True)
    disable_tqdm_cb = ft.Checkbox(label="Disable tqdm", value=False)
    seed_tf = ft.TextField(label="Seed", value="3407", width=140)
    save_strategy_dd = ft.Dropdown(
        label="Save strategy",
        options=[ft.dropdown.Option("epoch"), ft.dropdown.Option("steps"), ft.dropdown.Option("no")],
        value="epoch",
        width=160,
    )
    save_total_limit_tf = ft.TextField(label="Save total limit", value="2", width=180)
    pin_memory_cb = ft.Checkbox(label="Pin dataloader memory", value=False)
    report_to_dd = ft.Dropdown(
        label="Report to",
        options=[ft.dropdown.Option("none"), ft.dropdown.Option("wandb")],
        value="none",
        width=160,
    )
    fp16_cb = ft.Checkbox(label="Use FP16", value=True)
    bf16_cb = ft.Checkbox(label="Use BF16 (if supported)", value=False)

    advanced_params_section = ft.Column([
        ft.Row([warmup_steps_tf, weight_decay_tf, lr_sched_dd, optim_dd], wrap=True),
        ft.Row([logging_steps_tf, logging_first_step_cb, disable_tqdm_cb, seed_tf], wrap=True),
        ft.Row([save_strategy_dd, save_total_limit_tf, pin_memory_cb, report_to_dd], wrap=True),
        ft.Row([fp16_cb, bf16_cb], wrap=True),
    ], spacing=8)

    # Configuration mode controls
    config_mode_dd = ft.Dropdown(
        label="Mode",
        options=[ft.dropdown.Option("Normal"), ft.dropdown.Option("Configuration")],
        value="Normal",
        width=180,
        tooltip="Configuration mode: load a saved config and run with minimal inputs.",
    )
    config_files_dd = ft.Dropdown(label="Saved config", options=[], width=420)
    config_refresh_btn = ft.TextButton("Refresh", icon=REFRESH_ICON)
    load_config_btn = ft.ElevatedButton(
        "Load Config",
        icon=getattr(ICONS, "FILE_OPEN", getattr(ICONS, "FOLDER_OPEN", ICONS.UPLOAD)),
    )
    config_save_current_btn = ft.ElevatedButton(
        "Save Current",
        icon=getattr(ICONS, "SAVE", ICONS.CHECK),
        tooltip="Save the current training setup as a reusable config",
    )
    config_edit_btn = ft.TextButton(
        "Edit",
        icon=getattr(ICONS, "EDIT", getattr(ICONS, "MODE_EDIT", ICONS.SETTINGS)),
        tooltip="Edit selected config file",
        disabled=True,
    )
    config_rename_btn = ft.TextButton(
        "Rename",
        icon=getattr(ICONS, "DRIVE_FILE_RENAME_OUTLINE", getattr(ICONS, "EDIT", ICONS.SETTINGS)),
        tooltip="Rename selected config file",
        disabled=True,
    )
    config_delete_btn = ft.TextButton(
        "Delete",
        icon=getattr(ICONS, "DELETE", getattr(ICONS, "DELETE_FOREVER", ICONS.CLOSE)),
        tooltip="Delete selected config file",
        disabled=True,
    )
    config_summary_txt = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))
    # Container for file controls (visibility toggled by mode)
    config_files_row = ft.Row(
        [
            config_files_dd,
            config_refresh_btn,
            load_config_btn,
            config_save_current_btn,
            config_edit_btn,
            config_rename_btn,
            config_delete_btn,
        ],
        wrap=True,
    )

    def _saved_configs_dir() -> str:
        return saved_configs_dir_helper()

    def _list_saved_configs() -> List[str]:
        return list_saved_configs_helper(_saved_configs_dir())


    def _collect_local_ui_state() -> dict:
        data: dict = {}
        try:
            data["host_dir"] = (local_host_dir_tf.value or "")
        except Exception:
            data["host_dir"] = ""
        try:
            data["container_name"] = (local_container_name_tf.value or "")
        except Exception:
            data["container_name"] = ""
        try:
            data["docker_image"] = (docker_image_tf.value or "")
        except Exception:
            data["docker_image"] = ""
        try:
            data["use_gpu"] = bool(getattr(local_use_gpu_cb, "value", False))
        except Exception:
            data["use_gpu"] = False
        try:
            data["pass_hf_token"] = bool(getattr(local_pass_hf_token_cb, "value", False))
        except Exception:
            data["pass_hf_token"] = False
        return data

    def _update_config_buttons_enabled(_=None):
        try:
            has_sel = bool((config_files_dd.value or "").strip())
            config_edit_btn.disabled = not has_sel
            config_rename_btn.disabled = not has_sel
            config_delete_btn.disabled = not has_sel
        except Exception:
            pass
        try:
            page.update()
        except Exception:
            pass

    def _refresh_config_list(_=None):
        """Refresh saved config dropdown, filtered by current training target.

        Configs without an explicit meta.train_target are shown for all targets
        (backward compatibility)."""
        try:
            all_files = _list_saved_configs()
        except Exception:
            all_files = []
        files: List[str] = []
        try:
            cur_target = (train_target_dd.value or "Runpod - Pod").strip().lower()
        except Exception:
            cur_target = "runpod - pod"
        for name in all_files:
            tgt_val = ""
            try:
                path = os.path.join(_saved_configs_dir(), name)
                conf = _read_json_file(path) or {}
                meta = conf.get("meta") or {}
                tgt_val = str(meta.get("train_target") or "").strip().lower()
            except Exception:
                tgt_val = ""
            # No target recorded: show for all targets
            if not tgt_val:
                files.append(name)
                continue
            # Target-aware filtering
            if cur_target.startswith("runpod - pod"):
                if tgt_val.startswith("runpod - pod") or tgt_val == "runpod":
                    files.append(name)
            elif cur_target.startswith("local"):
                if tgt_val.startswith("local"):
                    files.append(name)
            else:
                # Unknown target label; do not hide config
                files.append(name)
        try:
            config_files_dd.options = [ft.dropdown.Option(f) for f in files]
            cur_val = (config_files_dd.value or "").strip()
            if cur_val and (cur_val not in files):
                config_files_dd.value = files[0] if files else None
            if files and not config_files_dd.value:
                config_files_dd.value = files[0]
            if not files:
                config_files_dd.value = None
        except Exception:
            pass
        _update_config_buttons_enabled()

    def _read_json_file(path: str) -> Optional[dict]:
        return read_json_file_helper(path)

    def _validate_config(conf: dict) -> Tuple[bool, str]:
        return validate_config_helper(conf)


    def _build_config_payload_from_ui() -> dict:
        hp = _build_hp() or {}
        try:
            tgt = train_target_dd.value or "Runpod - Pod"
        except Exception:
            tgt = "Runpod - Pod"
        meta = {
            "skill_level": skill_level.value,
            "beginner_mode": beginner_mode_dd.value if (skill_level.value or "") == "Beginner" else "",
            "train_target": tgt,
        }
        payload: dict = {"hp": hp, "meta": meta}
        # Include Runpod infra UI when on Runpod target
        try:
            if (tgt or "").lower().startswith("runpod - pod"):
                payload["infra_ui"] = {
                    "dc": (rp_dc_tf.value or "US-NC-1"),
                    "vol_name": (rp_vol_name_tf.value or "unsloth-volume"),
                    "vol_size": int(float(rp_vol_size_tf.value or "50")),
                    "resize_if_smaller": bool(getattr(rp_resize_cb, "value", True)),
                    "tpl_name": (rp_tpl_name_tf.value or "unsloth-trainer-template"),
                    "image": (rp_image_tf.value or "docker.io/sbussiso/unsloth-trainer:latest"),
                    "container_disk": int(float(rp_container_disk_tf.value or "30")),
                    "pod_volume_gb": int(float(rp_volume_in_gb_tf.value or "0")),
                    "mount_path": (rp_mount_path_tf.value or "/data"),
                    "category": (rp_category_tf.value or "NVIDIA"),
                    "public": bool(getattr(rp_public_cb, "value", False)),
                }
        except Exception:
            pass
        # Always include local UI so configs are reusable for local runs too
        try:
            payload["local_ui"] = _collect_local_ui_state()
        except Exception:
            pass
        return payload

    def _apply_config_to_ui(conf: dict) -> None:
        if not isinstance(conf, dict):
            return
        hp = conf.get("hp") or {}
        try:
            base_model.value = hp.get("base_model", base_model.value)
            epochs_tf.value = str(hp.get("epochs", epochs_tf.value))
            lr_tf.value = str(hp.get("lr", lr_tf.value))
            batch_tf.value = str(hp.get("bsz", batch_tf.value))
            grad_acc_tf.value = str(hp.get("grad_accum", grad_acc_tf.value))
            max_steps_tf.value = str(hp.get("max_steps", max_steps_tf.value))
            use_lora_cb.value = bool(hp.get("use_lora", use_lora_cb.value))
            out_dir_tf.value = hp.get("output_dir", out_dir_tf.value)
            # dataset
            if "hf_dataset_id" in hp:
                train_source.value = "Hugging Face"
                train_hf_repo.value = hp.get("hf_dataset_id", "")
                train_hf_split.value = hp.get("hf_dataset_split", "train")
                train_hf_config.value = hp.get("hf_dataset_config", train_hf_config.value)
            elif "json_path" in hp:
                train_source.value = "JSON file"
                train_json_path.value = hp.get("json_path", "")
            # toggles
            packing_cb.value = bool(hp.get("packing", packing_cb.value))
            auto_resume_cb.value = bool(hp.get("auto_resume", auto_resume_cb.value))
            push_cb.value = bool(hp.get("push", push_cb.value))
            hf_repo_id_tf.value = hp.get("hf_repo_id", hf_repo_id_tf.value or "")
            resume_from_tf.value = hp.get("resume_from", resume_from_tf.value or "")
        except Exception:
            pass
        # reflect meta.skill_level and meta.beginner_mode in UI without overriding HP
        try:
            meta = conf.get("meta") or {}
            skill = str(meta.get("skill_level") or "").strip().lower()
            mode = str(meta.get("beginner_mode") or "").strip().lower()
            if skill in ("beginner", "expert"):
                try:
                    train_state["suppress_skill_defaults"] = True
                except Exception:
                    pass
                try:
                    skill_level.value = "Beginner" if skill == "beginner" else "Expert"
                    if skill == "beginner" and mode in ("fastest", "cheapest"):
                        beginner_mode_dd.value = "Fastest" if mode == "fastest" else "Cheapest"
                    _update_skill_controls()
                except Exception:
                    pass
                finally:
                    try:
                        train_state["suppress_skill_defaults"] = False
                    except Exception:
                        pass
            # training target (Runpod vs local)
            try:
                tgt = str(meta.get("train_target") or "").strip()
                if tgt:
                    train_target_dd.value = tgt
                    try:
                        _update_training_target()
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass
        # infra UI
        iu = conf.get("infra_ui") or {}
        try:
            rp_dc_tf.value = iu.get("dc", rp_dc_tf.value)
            rp_vol_name_tf.value = iu.get("vol_name", rp_vol_name_tf.value)
            rp_vol_size_tf.value = str(iu.get("vol_size", rp_vol_size_tf.value))
            rp_resize_cb.value = bool(iu.get("resize_if_smaller", rp_resize_cb.value))
            rp_tpl_name_tf.value = iu.get("tpl_name", rp_tpl_name_tf.value)
            rp_image_tf.value = iu.get("image", rp_image_tf.value)
            rp_container_disk_tf.value = str(iu.get("container_disk", rp_container_disk_tf.value))
            rp_volume_in_gb_tf.value = str(iu.get("pod_volume_gb", rp_volume_in_gb_tf.value))
            rp_mount_path_tf.value = iu.get("mount_path", rp_mount_path_tf.value)
            rp_category_tf.value = iu.get("category", rp_category_tf.value)
            rp_public_cb.value = bool(iu.get("public", rp_public_cb.value))
        except Exception:
            pass
        # local UI (local Docker training settings)
        try:
            lu = conf.get("local_ui") or {}
            try:
                local_host_dir_tf.value = lu.get("host_dir", local_host_dir_tf.value)
            except Exception:
                pass
            try:
                local_container_name_tf.value = lu.get("container_name", local_container_name_tf.value)
            except Exception:
                pass
            try:
                docker_image_tf.value = lu.get("docker_image", docker_image_tf.value)
            except Exception:
                pass
            try:
                local_use_gpu_cb.value = bool(lu.get("use_gpu", getattr(local_use_gpu_cb, "value", False)))
            except Exception:
                pass
            try:
                local_pass_hf_token_cb.value = bool(lu.get("pass_hf_token", getattr(local_pass_hf_token_cb, "value", False)))
            except Exception:
                pass
        except Exception:
            pass
        # summary
        try:
            m = hp.get("base_model", "")
            ds = hp.get("hf_dataset_id") or hp.get("json_path") or ""
            config_summary_txt.value = f"Model: {m} • Dataset: {ds}"
        except Exception:
            pass
        try:
            _update_train_source()
            page.update()
        except Exception:
            pass

    async def on_load_config():
        name = (config_files_dd.value or "").strip()
        if not name:
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Select a config to load."))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return
        path = os.path.join(_saved_configs_dir(), name)
        conf = _read_json_file(path)
        if not isinstance(conf, dict):
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Failed to load config: invalid JSON or unreadable file."))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return
        ok, msg = _validate_config(conf)
        if not ok:
            try:
                page.snack_bar = ft.SnackBar(ft.Text(f"Invalid config: {msg}"))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return
        # Persist and apply
        train_state["loaded_config"] = conf
        try:
            train_state["loaded_config_name"] = name
        except Exception:
            pass
        try:
            set_last_used_config_name_helper(name)
        except Exception:
            pass
        _apply_config_to_ui(conf)
        try:
            page.snack_bar = ft.SnackBar(ft.Text(f"Loaded config: {name}"))
            page.snack_bar.open = True
            await safe_update(page)
        except Exception:
            pass

    async def on_rename_config():
        name = (config_files_dd.value or "").strip()
        if not name:
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Select a config to rename."))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return
        new_tf = ft.TextField(label="New name", value=name, width=420)

        async def _do_rename(_=None):
            try:
                new_name = (new_tf.value or name).strip()
                if not new_name:
                    return
                if not new_name.lower().endswith('.json'):
                    new_name = f"{new_name}.json"
                d = _saved_configs_dir()
                src = os.path.join(d, name)
                dst = os.path.join(d, new_name)
                # No-op if unchanged (case-insensitive on Windows)
                try:
                    if os.path.normcase(os.path.abspath(src)) == os.path.normcase(os.path.abspath(dst)):
                        dlg.open = False
                        await safe_update(page)
                        return
                except Exception:
                    pass
                if os.path.exists(dst):
                    page.snack_bar = ft.SnackBar(ft.Text("A config with that name already exists."))
                    page.snack_bar.open = True
                    await safe_update(page)
                    return
                os.rename(src, dst)
                _refresh_config_list()
                try:
                    config_files_dd.value = new_name
                except Exception:
                    pass
                page.snack_bar = ft.SnackBar(ft.Text(f"Renamed to: {new_name}"))
                page.snack_bar.open = True
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Rename failed: {ex}"))
                page.snack_bar.open = True
            finally:
                try:
                    dlg.open = False
                    await safe_update(page)
                except Exception:
                    pass

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(getattr(ICONS, "DRIVE_FILE_RENAME_OUTLINE", getattr(ICONS, "EDIT", ICONS.SETTINGS)), color=ACCENT_COLOR),
                ft.Text("Rename configuration"),
            ], alignment=ft.MainAxisAlignment.START),
            content=ft.Column([ft.Text("Choose a new filename (JSON extension optional)."), new_tf], tight=True, spacing=6),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: (setattr(dlg, "open", False), page.update())),
                ft.ElevatedButton("Rename", icon=getattr(ICONS, "CHECK", ICONS.SAVE), on_click=lambda e: page.run_task(_do_rename)),
            ],
        )
        page.dialog = dlg
        dlg.open = True
        await safe_update(page)

    async def on_edit_config():
        name = (config_files_dd.value or "").strip()
        if not name:
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Select a config to edit."))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return
        d = _saved_configs_dir()
        path = os.path.join(d, name)
        raw_text = ""
        conf = None
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except Exception:
            raw_text = ""
        if not raw_text:
            conf = _read_json_file(path) or {}
            try:
                raw_text = json.dumps(conf, indent=2, ensure_ascii=False)
            except Exception:
                raw_text = "{}"

        editor_tf = ft.TextField(
            label=f"Editing: {name}",
            value=raw_text,
            multiline=True,
            max_lines=24,
            width=900,
            tooltip="Edit JSON configuration. Use Save to validate and write changes.",
        )
        status_txt = ft.Text("", size=12, color=WITH_OPACITY(0.8, BORDER_BASE))

        async def _validate_only(_=None):
            try:
                data = json.loads(editor_tf.value or "{}")
            except Exception as ex:
                status_txt.value = f"JSON error: {ex}"
                try:
                    status_txt.color = COLORS.RED
                except Exception:
                    pass
                await safe_update(page)
                return
            ok, msg = _validate_config(data)
            if ok:
                status_txt.value = f"Valid ✓ {msg or ''}"
                try:
                    status_txt.color = getattr(COLORS, "GREEN", COLORS.SECONDARY)
                except Exception:
                    pass
            else:
                status_txt.value = f"Invalid: {msg}"
                try:
                    status_txt.color = COLORS.RED
                except Exception:
                    pass
            await safe_update(page)

        async def _save_edits(_=None):
            try:
                data = json.loads(editor_tf.value or "{}")
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"JSON error: {ex}"))
                page.snack_bar.open = True
                await safe_update(page)
                return
            ok, msg = _validate_config(data)
            if not ok:
                page.snack_bar = ft.SnackBar(ft.Text(f"Invalid config: {msg}"))
                page.snack_bar.open = True
                await safe_update(page)
                return
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(data, indent=2, ensure_ascii=False))
                    f.write("\n")
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Save failed: {ex}"))
                page.snack_bar.open = True
                await safe_update(page)
                return
            # Refresh list and update loaded config/UI if applicable
            try:
                _refresh_config_list()
            except Exception:
                pass
            try:
                if (train_state.get("loaded_config_name") or "") == name:
                    train_state["loaded_config"] = data
                    _apply_config_to_ui(data)
            except Exception:
                pass
            page.snack_bar = ft.SnackBar(ft.Text(f"Saved: {name}"))
            page.snack_bar.open = True
            try:
                edit_dlg.open = False
                await safe_update(page)
            except Exception:
                pass

        edit_dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(getattr(ICONS, "EDIT", getattr(ICONS, "MODE_EDIT", ICONS.SETTINGS)), color=ACCENT_COLOR),
                ft.Text("Edit configuration"),
            ], alignment=ft.MainAxisAlignment.START),
            content=ft.Column([
                ft.Text("Update the JSON below, then click Save. Use Validate to check without saving."),
                editor_tf,
                status_txt,
            ], tight=True, spacing=8),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: (setattr(edit_dlg, "open", False), page.update())),
                ft.OutlinedButton("Validate", icon=getattr(ICONS, "CHECK_CIRCLE", ICONS.CHECK), on_click=lambda e: page.run_task(_validate_only)),
                ft.ElevatedButton("Save", icon=getattr(ICONS, "SAVE", ICONS.CHECK), on_click=lambda e: page.run_task(_save_edits)),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        # Open dialog using the resilient pattern
        opened = False
        try:
            if hasattr(page, "open") and callable(getattr(page, "open")):
                page.open(edit_dlg)
                opened = True
        except Exception:
            opened = False
        if not opened:
            page.dialog = edit_dlg
            edit_dlg.open = True
        await safe_update(page)

    async def on_delete_config():
        name = (config_files_dd.value or "").strip()
        if not name:
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Select a config to delete."))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return
        confirm_dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(getattr(ICONS, "DELETE", getattr(ICONS, "DELETE_FOREVER", ICONS.CLOSE)), color=COLORS.RED),
                ft.Text("Delete configuration?"),
            ], alignment=ft.MainAxisAlignment.START),
            content=ft.Text(f"This will permanently delete '{name}'."),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: (setattr(confirm_dlg, "open", False), page.update())),
                ft.ElevatedButton("Delete", icon=getattr(ICONS, "CHECK", ICONS.DELETE), on_click=lambda e: page.run_task(_do_delete)),
            ],
        )
        # Open confirm dialog using same pattern as dataset previews
        opened = False
        try:
            if hasattr(page, "open") and callable(getattr(page, "open")):
                page.open(confirm_dlg)
                opened = True
        except Exception:
            opened = False
        if not opened:
            page.dialog = confirm_dlg
            confirm_dlg.open = True
        await safe_update(page)

        async def _do_delete(_=None):
            try:
                d = _saved_configs_dir()
                path = os.path.join(d, name)
                os.remove(path)
                _refresh_config_list()
                try:
                    if (train_state.get("loaded_config_name") or "") == name:
                        train_state["loaded_config_name"] = ""
                        train_state["loaded_config"] = {}
                        config_summary_txt.value = ""
                except Exception:
                    pass
                page.snack_bar = ft.SnackBar(ft.Text(f"Deleted: {name}"))
                page.snack_bar.open = True
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Delete failed: {ex}"))
                page.snack_bar.open = True
            finally:
                try:
                    confirm_dlg.open = False
                    await safe_update(page)
                except Exception:
                    pass

    # Progress & logs
    train_progress = ft.ProgressBar(value=0.0, width=400)
    train_prog_label = ft.Text("Progress: 0%")
    train_timeline = ft.ListView([], spacing=4, auto_scroll=True, expand=True)
    train_timeline_placeholder = make_empty_placeholder("No training logs yet", getattr(ICONS, "SCIENCE", ICONS.PLAY_CIRCLE))

    def _update_progress_from_logs(new_lines: List[str]) -> None:
        """Parse progress from recent log lines and update the progress bar and label.
        Supported patterns:
        - Percent like "12%"
        - Step/total like "Step 100/1000", "global_step 1234/5000", "Step 10 of 100"
        - Epoch progress like "Epoch 1/3" or "Epoch [1/3]"
        If no numeric progress found, show the most recent log line (truncated).
        """
        try:
            last_val = float(train_state.get("progress") or 0.0)
        except Exception:
            last_val = 0.0

        if not new_lines:
            return

        pct_found: Optional[int] = None
        for line in reversed(list(new_lines)):
            s = str(line)
            # 1) Percentage patterns
            m = re.search(r"(\d{1,3})\s?%", s)
            if m:
                try:
                    pct = int(m.group(1))
                    if 0 <= pct <= 100:
                        pct_found = pct
                        break
                except Exception:
                    pass

            # 2) Step/total patterns (Step 100/1000, global_step 12/500, Iteration 5 of 20)
            m = re.search(r"(?:global[_ ]?step|steps?|iter(?:ation)?|it(?:er)?)\s*[:=]?\s*(\d+)\s*(?:/|of)\s*(\d+)", s, re.IGNORECASE)
            if m:
                try:
                    cur = int(m.group(1)); tot = int(m.group(2))
                    if tot > 0:
                        pct_found = max(0, min(100, int(cur * 100 / tot)))
                        break
                except Exception:
                    pass

            # 3) Epoch progress patterns (Epoch 1/3, Epoch [1/3])
            m = re.search(r"epoch\s*[:#]?\s*(\d+)\s*/\s*(\d+)", s, re.IGNORECASE)
            if not m:
                m = re.search(r"epoch\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]", s, re.IGNORECASE)
            if m:
                try:
                    cur = int(m.group(1)); tot = int(m.group(2))
                    if tot > 0:
                        # Use (cur-1)/tot as coarse progress into epochs
                        pct_found = max(0, min(100, int(((cur - 1) / tot) * 100)))
                        break
                except Exception:
                    pass

        try:
            if pct_found is not None:
                v = max(last_val, min(1.0, pct_found / 100.0))
                train_progress.value = v
                train_prog_label.value = f"Progress: {int(v * 100)}%"
                try:
                    train_state["progress"] = v
                except Exception:
                    pass
            else:
                # No numeric progress found; show latest line text
                latest = str(new_lines[-1]) if new_lines else ""
                if latest:
                    max_len = 120
                    disp = latest if len(latest) <= max_len else (latest[: max_len - 1] + "…")
                    train_prog_label.value = disp
        except Exception:
            # Never let UI updates crash the loop
            pass

    def update_train_placeholders():
        try:
            has_logs = len(getattr(train_timeline, "controls", []) or []) > 0
            train_timeline_placeholder.visible = not has_logs
        except Exception:
            pass

    cancel_train = {"cancelled": False}
    train_state = {"running": False, "pod_id": None, "infra": None, "api_key": "", "loaded_config": None, "suppress_skill_defaults": False}

    def _update_skill_controls(_=None):
        level = (skill_level.value or "Beginner").lower()
        is_beginner = (level == "beginner")
        # Hide some tweak knobs for beginners
        for ctl in [lr_tf, batch_tf, grad_acc_tf, max_steps_tf]:
            try:
                ctl.visible = (not is_beginner)
            except Exception:
                pass
        # Advanced block
        try:
            advanced_params_section.visible = (not is_beginner)
        except Exception:
            pass
        # Beginner target control visibility
        try:
            beginner_mode_dd.visible = is_beginner
        except Exception:
            pass
        # Expert GPU picker visibility
        try:
            expert_gpu_dd.visible = (not is_beginner)
            # Spot is only meaningful for Runpod target
            tgt = (train_target_dd.value or "Runpod - Pod").lower()
            if tgt.startswith("runpod - pod"):
                expert_spot_cb.visible = (not is_beginner)
                expert_spot_cb.disabled = False
            else:
                expert_spot_cb.value = False
                expert_spot_cb.disabled = True
                expert_spot_cb.visible = False
            expert_gpu_refresh_btn.visible = (not is_beginner)
            if is_beginner:
                expert_gpu_busy.visible = False
        except Exception:
            pass
        # Local-mode beginner uses simple GPU checkbox; expert uses dropdown
        try:
            tgt2 = (train_target_dd.value or "Runpod - Pod").lower()
            if not tgt2.startswith("runpod - pod"):
                local_use_gpu_cb.visible = is_beginner
        except Exception:
            pass
        suppress = bool(train_state.get("suppress_skill_defaults"))
        # Set beginner defaults (depend on beginner mode)
        if is_beginner and (not suppress):
            try:
                mode = (beginner_mode_dd.value or "Fastest").lower()
                epochs_tf.value = epochs_tf.value or "1"
                if mode == "fastest":
                    lr_tf.value = "2e-4"
                    batch_tf.value = "4"
                    grad_acc_tf.value = "1"
                    max_steps_tf.value = max_steps_tf.value or "200"
                else:
                    lr_tf.value = "2e-5"
                    batch_tf.value = "2"
                    grad_acc_tf.value = "4"
                    max_steps_tf.value = "200"
            except Exception:
                pass
        # If switching to Expert for the first time, lazily refresh GPU list
        try:
            if (not is_beginner) and (len(getattr(expert_gpu_dd, "options", []) or []) <= 1):
                tgt = (train_target_dd.value or "Runpod - Pod").lower()
                if tgt.startswith("runpod - pod"):
                    schedule_task(refresh_expert_gpus)
                else:
                    schedule_task(refresh_local_gpus)
        except Exception:
            pass
        try:
            page.update()
        except Exception:
            pass

    skill_level.on_change = _update_skill_controls
    beginner_mode_dd.on_change = _update_skill_controls
    train_source.on_change = _update_train_source
    # Initialize skill-level dependent visibility once
    try:
        _update_skill_controls()
    except Exception:
        pass

    def _build_hp() -> dict:
        """Build train.py flags via helper (delegated)."""
        return build_hp_from_controls_helper(
            train_source=train_source,
            train_hf_repo=train_hf_repo,
            train_hf_split=train_hf_split,
            train_hf_config=train_hf_config,
            train_json_path=train_json_path,
            base_model=base_model,
            out_dir_tf=out_dir_tf,
            epochs_tf=epochs_tf,
            lr_tf=lr_tf,
            batch_tf=batch_tf,
            grad_acc_tf=grad_acc_tf,
            max_steps_tf=max_steps_tf,
            use_lora_cb=use_lora_cb,
            packing_cb=packing_cb,
            auto_resume_cb=auto_resume_cb,
            push_cb=push_cb,
            hf_repo_id_tf=hf_repo_id_tf,
            resume_from_tf=resume_from_tf,
            warmup_steps_tf=warmup_steps_tf,
            weight_decay_tf=weight_decay_tf,
            lr_sched_dd=lr_sched_dd,
            optim_dd=optim_dd,
            logging_steps_tf=logging_steps_tf,
            logging_first_step_cb=logging_first_step_cb,
            disable_tqdm_cb=disable_tqdm_cb,
            seed_tf=seed_tf,
            save_strategy_dd=save_strategy_dd,
            save_total_limit_tf=save_total_limit_tf,
            pin_memory_cb=pin_memory_cb,
            report_to_dd=report_to_dd,
            fp16_cb=fp16_cb,
            bf16_cb=bf16_cb,
        )

    async def _on_pod_created(data: dict):
        try:
            hp = (data or {}).get("hp") or {}
            chosen_gpu_type_id = (data or {}).get("chosen_gpu_type_id")
            chosen_interruptible = bool((data or {}).get("chosen_interruptible"))
            default_name = f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(hp.get('base_model','model')).replace('/', '_')}.json"
            name_tf = ft.TextField(label="Save as", value=default_name, width=420)

            def _collect_infra_ui_state() -> dict:
                return {
                    "dc": (rp_dc_tf.value or "US-NC-1"),
                    "vol_name": (rp_vol_name_tf.value or "unsloth-volume"),
                    "vol_size": int(float(rp_vol_size_tf.value or "50")),
                    "resize_if_smaller": bool(getattr(rp_resize_cb, "value", True)),
                    "tpl_name": (rp_tpl_name_tf.value or "unsloth-trainer-template"),
                    "image": (rp_image_tf.value or "docker.io/sbussiso/unsloth-trainer:latest"),
                    "container_disk": int(float(rp_container_disk_tf.value or "30")),
                    "pod_volume_gb": int(float(rp_volume_in_gb_tf.value or "0")),
                    "mount_path": (rp_mount_path_tf.value or "/data"),
                    "category": (rp_category_tf.value or "NVIDIA"),
                    "public": bool(getattr(rp_public_cb, "value", False)),
                }

            payload = {
                "hp": hp,
                "infra_ui": _collect_infra_ui_state(),
                "meta": {
                    "skill_level": skill_level.value,
                    "beginner_mode": beginner_mode_dd.value if (skill_level.value or "") == "Beginner" else "",
                },
                "pod": {
                    "gpu_type_id": chosen_gpu_type_id,
                    "interruptible": bool(chosen_interruptible),
                },
            }

            def _do_save(_=None):
                name = (name_tf.value or default_name).strip()
                d = _saved_configs_dir()
                path = os.path.join(d, name if name.endswith('.json') else f"{name}.json")
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                    page.snack_bar = ft.SnackBar(ft.Text(f"Saved config: {os.path.basename(path)}"))
                    page.snack_bar.open = True
                    _refresh_config_list()
                except Exception as ex:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Failed to save config: {ex}"))
                    page.snack_bar.open = True
                try:
                    dlg.open = False
                    page.update()
                except Exception:
                    pass

            dlg = ft.AlertDialog(
                modal=True,
                title=ft.Row([
                    ft.Icon(getattr(ICONS, "SAVE_ALT", ICONS.SAVE), color=ACCENT_COLOR),
                    ft.Text("Save this training setup?"),
                ], alignment=ft.MainAxisAlignment.START),
                content=ft.Column([
                    ft.Text("You can reuse this configuration later via Training → Configuration mode."),
                    name_tf,
                ], tight=True, spacing=6),
                actions=[
                    ft.TextButton("Skip", on_click=lambda e: (setattr(dlg, "open", False), page.update())),
                    ft.ElevatedButton("Save", icon=getattr(ICONS, "SAVE", ICONS.CHECK), on_click=_do_save),
                ],
            )
            try:
                dlg.on_dismiss = lambda e: page.update()
            except Exception:
                pass
            opened = False
            try:
                if hasattr(page, "open") and callable(getattr(page, "open")):
                    page.open(dlg)
                    opened = True
            except Exception:
                opened = False
            if not opened:
                page.dialog = dlg
                dlg.open = True
            train_timeline.controls.append(ft.Row([ft.Icon(getattr(ICONS, "SAVE", ICONS.SAVE_ALT), color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text("Opened Save Configuration dialog")]))
            update_train_placeholders(); await safe_update(page)
            try:
                await asyncio.sleep(0.05)
            except Exception:
                pass
        except Exception:
            pass

    async def on_start_training():
        return await run_pod_training_helper(
            page=page,
            rp_pod_module=rp_pod,
            train_state=train_state,
            cancel_train=cancel_train,
            _runpod_cfg=_runpod_cfg,
            rp_temp_key_tf=rp_temp_key_tf,
            config_mode_dd=config_mode_dd,
            config_files_dd=config_files_dd,
            skill_level=skill_level,
            beginner_mode_dd=beginner_mode_dd,
            expert_gpu_dd=expert_gpu_dd,
            expert_spot_cb=expert_spot_cb,
            base_model=base_model,
            train_source=train_source,
            train_hf_repo=train_hf_repo,
            train_hf_split=train_hf_split,
            train_json_path=train_json_path,
            train_timeline=train_timeline,
            train_progress=train_progress,
            train_prog_label=train_prog_label,
            start_train_btn=start_train_btn,
            stop_train_btn=stop_train_btn,
            refresh_train_btn=refresh_train_btn,
            restart_container_btn=restart_container_btn,
            open_runpod_btn=open_runpod_btn,
            open_web_terminal_btn=open_web_terminal_btn,
            copy_ssh_btn=copy_ssh_btn,
            auto_terminate_cb=auto_terminate_cb,
            update_train_placeholders=update_train_placeholders,
            refresh_teardown_ui_fn=_refresh_teardown_ui,
            update_progress_from_logs=_update_progress_from_logs,
            build_hp_fn=_build_hp,
            on_pod_created=_on_pod_created,
        )

    async def on_restart_container():
        return await restart_pod_container_helper(
            page=page,
            rp_pod_module=rp_pod,
            train_state=train_state,
            config_mode_dd=config_mode_dd,
            config_files_dd=config_files_dd,
            train_timeline=train_timeline,
            update_train_placeholders=update_train_placeholders,
            build_hp_fn=_build_hp,
        )

    def on_open_runpod(_):
        return open_runpod_helper(page, train_state, train_timeline)

    def on_open_web_terminal(_):
        return open_web_terminal_helper(page, train_state, train_timeline)

    async def on_copy_ssh_command(_):
        return await copy_ssh_command_helper(
            page=page,
            rp_pod_module=rp_pod,
            train_state=train_state,
            train_timeline=train_timeline,
        )

    def on_stop_training(_):
        if not train_state.get("running"):
            # If a pod exists but not marked running, allow manual terminate
            pod_id = train_state.get("pod_id")
            if pod_id:
                try:
                    async def _terminate():
                        try:
                            await asyncio.to_thread(
                                rp_pod.delete_pod,
                                (train_state.get("api_key") or os.environ.get("RUNPOD_API_KEY") or "").strip(),
                                pod_id,
                            )
                        finally:
                            try:
                                train_timeline.controls.append(ft.Row([ft.Icon(ICONS.CANCEL, color=COLORS.RED), ft.Text("Termination requested for existing pod")]))
                                # Reset UI to idle state
                                start_train_btn.visible = True
                                start_train_btn.disabled = False
                                stop_train_btn.disabled = True
                                refresh_train_btn.disabled = False
                                train_state["pod_id"] = None
                                update_train_placeholders(); await safe_update(page)
                            except Exception:
                                pass
                    schedule_task(_terminate)
                except Exception:
                    pass
            return
        cancel_train["cancelled"] = True
        try:
            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.CANCEL, color=COLORS.RED), ft.Text("Cancel requested — will stop ASAP")]))
            # Prevent multiple stop presses
            stop_train_btn.disabled = True
            update_train_placeholders(); page.update()
        except Exception:
            pass

    def on_refresh_training(_):
        try:
            # Clear current log space only; keep progress and state intact
            train_timeline.controls.clear()
            train_state["log_seen"] = set()
            update_train_placeholders(); page.update()
        except Exception:
            pass

    # ---------- Runpod Infrastructure (Ensure volume + template) ----------
    rp_dc_tf = ft.TextField(label="Datacenter ID", value="US-NC-1", width=140)
    rp_vol_name_tf = ft.TextField(label="Volume name", value="unsloth-volume", width=220)
    rp_vol_size_tf = ft.TextField(label="Volume size (GB)", value="50", width=180)
    rp_resize_cb = ft.Checkbox(label="Resize if smaller", value=True)

    # Info icon for "Resize if smaller": explains that existing smaller volumes will be expanded (never shrunk)
    rp_resize_info = ft.IconButton(
        icon=getattr(ICONS, "INFO_OUTLINE", getattr(ICONS, "INFO", getattr(ICONS, "HELP_OUTLINE", None))),
        tooltip="If enabled, an existing volume smaller than the requested size will be expanded. It never shrinks volumes.",
        on_click=_mk_help_handler(
            "When ensuring the Runpod Network Volume: if a volume with this name already exists and its size is smaller than the size you specify, it will be automatically increased to match your requested size. Existing volumes are never shrunk."
        ),
    )

    # Keep the info icon on the right of the checkbox by grouping them together
    rp_resize_row = ft.Row([rp_resize_cb, rp_resize_info], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    rp_tpl_name_tf = ft.TextField(label="Template name", value="unsloth-trainer-template", width=260)
    rp_image_tf = ft.TextField(label="Image name", value="docker.io/sbussiso/unsloth-trainer:latest", width=360)
    rp_container_disk_tf = ft.TextField(label="Container disk (GB)", value="30", width=200)
    rp_volume_in_gb_tf = ft.TextField(label="Pod volume (GB)", value="0", width=180, tooltip="Optional pod-local disk, not the network volume")
    rp_mount_path_tf = ft.TextField(
        label="Mount path",
        value="/data",
        width=220,
        tooltip="Avoid mounting at /workspace to prevent hiding train.py inside the image. /data is recommended.")
    rp_category_tf = ft.TextField(label="Category", value="NVIDIA", width=160)
    rp_public_cb = ft.Checkbox(label="Public template", value=False)

    # Temporary API key input (overrides Settings key for this session)
    rp_temp_key_tf = ft.TextField(
        label="Runpod API key (temp)",
        password=True,
        can_reveal_password=True,
        width=420,
        tooltip="Optional. Overrides Settings key for this run. You can also set RUNPOD_API_KEY env var.",
    )

    # Info icon for "Public template": clarifies visibility and considerations
    rp_public_info = ft.IconButton(
        icon=getattr(ICONS, "INFO_OUTLINE", getattr(ICONS, "INFO", getattr(ICONS, "HELP_OUTLINE", None))),
        tooltip="Make this template visible to all Runpod users. Be mindful of sensitive env vars.",
        on_click=_mk_help_handler(
            "Public templates are discoverable by other Runpod users. Others can launch pods using this template. If your image is private or requires registry auth, they will need access to run it. Avoid putting sensitive environment variables in the template."
        ),
    )
    rp_public_row = ft.Row([rp_public_cb, rp_public_info], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    # (SSH exposure option removed by request)

    rp_infra_busy = ft.ProgressRing(visible=False)

    async def on_ensure_infra():
        return await ensure_infrastructure_helper(
            page=page,
            rp_infra_module=rp_infra,
            _hf_cfg=_hf_cfg,
            _runpod_cfg=_runpod_cfg,
            rp_dc_tf=rp_dc_tf,
            rp_vol_name_tf=rp_vol_name_tf,
            rp_vol_size_tf=rp_vol_size_tf,
            rp_resize_cb=rp_resize_cb,
            rp_tpl_name_tf=rp_tpl_name_tf,
            rp_image_tf=rp_image_tf,
            rp_container_disk_tf=rp_container_disk_tf,
            rp_volume_in_gb_tf=rp_volume_in_gb_tf,
            rp_mount_path_tf=rp_mount_path_tf,
            rp_category_tf=rp_category_tf,
            rp_public_cb=rp_public_cb,
            rp_temp_key_tf=rp_temp_key_tf,
            rp_infra_busy=rp_infra_busy,
            train_timeline=train_timeline,
            refresh_expert_gpus_fn=lambda: page.run_task(refresh_expert_gpus),
            refresh_teardown_ui_fn=_refresh_teardown_ui,
            dataset_section=dataset_section,
            train_params_section=train_params_section,
            start_train_btn=start_train_btn,
            stop_train_btn=stop_train_btn,
            refresh_train_btn=refresh_train_btn,
            _update_mode_visibility=_update_mode_visibility,
        )

    def on_click_ensure_infra(e):
        # Immediate feedback to confirm click registered and show activity
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Ensuring Runpod infrastructure…"))
            page.snack_bar.open = True
            rp_infra_busy.visible = True
            update_train_placeholders(); page.update()
        except Exception:
            pass
        schedule_task(on_ensure_infra)

    ensure_infra_btn = ft.ElevatedButton(
        "Ensure Infrastructure",
        icon=getattr(ICONS, "CLOUD_DONE", getattr(ICONS, "CLOUD", ICONS.SETTINGS)),
        on_click=on_click_ensure_infra,
    )
    try:
        if rp_infra is None:
            ensure_infra_btn.disabled = True
            ensure_infra_btn.tooltip = "Runpod infra helper unavailable. Install/enable runpod ensure_infra."
    except Exception:
        pass
    rp_infra_actions = ft.Row([
        ensure_infra_btn,
        rp_infra_busy,
    ], spacing=10)

    # Populate Expert GPU dropdown from Runpod based on datacenter (delegated)
    async def refresh_expert_gpus(_=None):
        return await refresh_expert_gpus_helper(
            page=page,
            rp_pod_module=rp_pod,
            train_state=train_state,
            _runpod_cfg=_runpod_cfg,
            rp_temp_key_tf=rp_temp_key_tf,
            rp_dc_tf=rp_dc_tf,
            expert_gpu_busy=expert_gpu_busy,
            expert_gpu_dd=expert_gpu_dd,
            expert_spot_cb=expert_spot_cb,
            expert_gpu_avail=expert_gpu_avail,
            _update_expert_spot_enabled=_update_expert_spot_enabled,
        )

    # Populate Expert GPU dropdown from LOCAL system GPUs (delegated)
    async def refresh_local_gpus(_=None):
        return await refresh_local_gpus_helper(
            page=page,
            expert_gpu_busy=expert_gpu_busy,
            expert_gpu_dd=expert_gpu_dd,
            expert_spot_cb=expert_spot_cb,
            expert_gpu_avail=expert_gpu_avail,
        )

    # Wire refresh button (dispatch based on training target)
    def on_click_expert_gpu_refresh(e):
        try:
            tgt = (train_target_dd.value or "Runpod - Pod").lower()
            if tgt.startswith("runpod - pod"):
                page.run_task(refresh_expert_gpus)
            else:
                page.run_task(refresh_local_gpus)
        except Exception:
            pass
    try:
        expert_gpu_refresh_btn.on_click = on_click_expert_gpu_refresh
    except Exception:
        pass

    # Compact infra action for Configuration mode (button only)
    rp_infra_compact_row = ft.Row([
        ft.OutlinedButton(
            "Ensure Infrastructure",
            icon=getattr(ICONS, "CLOUD_DONE", getattr(ICONS, "CLOUD", ICONS.SETTINGS)),
            on_click=on_click_ensure_infra,
        ),
    ], spacing=10)

    # Configuration section wrapper to place in Training tab (delegated)
    config_section = build_config_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        config_mode_dd=config_mode_dd,
        config_files_row=config_files_row,
        config_summary_txt=config_summary_txt,
        rp_infra_compact_row=rp_infra_compact_row,
    )

    # Group all Runpod infrastructure controls to toggle as one section (delegated)
    rp_infra_panel = build_rp_infra_panel(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        rp_dc_tf=rp_dc_tf,
        rp_vol_name_tf=rp_vol_name_tf,
        rp_vol_size_tf=rp_vol_size_tf,
        rp_resize_row=rp_resize_row,
        rp_tpl_name_tf=rp_tpl_name_tf,
        rp_image_tf=rp_image_tf,
        rp_container_disk_tf=rp_container_disk_tf,
        rp_volume_in_gb_tf=rp_volume_in_gb_tf,
        rp_mount_path_tf=rp_mount_path_tf,
        rp_category_tf=rp_category_tf,
        rp_public_row=rp_public_row,
        rp_temp_key_tf=rp_temp_key_tf,
        rp_infra_actions=rp_infra_actions,
    )

    # Training action buttons are disabled until infra is ready
    start_train_btn = ft.ElevatedButton(
        "Start Training",
        icon=getattr(ICONS, "SCIENCE", ICONS.PLAY_CIRCLE),
        on_click=lambda e: page.run_task(on_start_training),
        disabled=True,
    )
    stop_train_btn = ft.OutlinedButton(
        "Stop",
        icon=ICONS.STOP_CIRCLE,
        on_click=on_stop_training,
        disabled=True,
    )
    refresh_train_btn = ft.TextButton(
        "Refresh",
        icon=REFRESH_ICON,
        on_click=on_refresh_training,
        disabled=True,
    )
    restart_container_btn = ft.OutlinedButton(
        "Restart Container",
        icon=getattr(ICONS, "RESTART_ALT", getattr(ICONS, "REFRESH", ICONS.SETTINGS)),
        on_click=lambda e: page.run_task(on_restart_container),
        disabled=True,
    )
    auto_terminate_cb = ft.Checkbox(label="Auto-terminate on finish", value=True, tooltip="Delete pod automatically when training reaches a terminal state.")
    open_runpod_btn = ft.TextButton(
        "Open in Runpod",
        icon=getattr(ICONS, "OPEN_IN_NEW", getattr(ICONS, "LINK", ICONS.SETTINGS)),
        on_click=on_open_runpod,
        disabled=True,
    )
    open_web_terminal_btn = ft.TextButton(
        "Open Web Terminal",
        icon=getattr(ICONS, "TERMINAL", getattr(ICONS, "CODE", ICONS.OPEN_IN_NEW)),
        on_click=on_open_web_terminal,
        disabled=True,
        tooltip="Opens the pod page; then click Connect → Open Web Terminal",
    )
    copy_ssh_btn = ft.TextButton(
        "Copy SSH Command",
        icon=getattr(ICONS, "CONTENT_COPY", getattr(ICONS, "COPY", ICONS.LINK)),
        on_click=lambda e: page.run_task(on_copy_ssh_command),
        disabled=True,
        tooltip="Copies an SSH command for this pod to your clipboard.",
    )
    save_config_bottom_btn = ft.TextButton(
        "Save current setup",
        icon=getattr(ICONS, "SAVE", ICONS.CHECK),
        tooltip="Save the current training setup (dataset, hyperparameters, target, and infra) as a reusable config.",
        on_click=lambda e: page.run_task(on_save_current_config),
    )
    train_actions = ft.Row([
        start_train_btn,
        stop_train_btn,
        refresh_train_btn,
        restart_container_btn,
        open_runpod_btn,
        open_web_terminal_btn,
        copy_ssh_btn,
        auto_terminate_cb,
        save_config_bottom_btn,
    ], spacing=10)

    # ---------- Teardown Section (Volume/Template/Pod) ----------
    td_title = section_title(
        "Teardown",
        getattr(ICONS, "DELETE_FOREVER", getattr(ICONS, "DELETE", ICONS.CLOSE)),
        "Select infrastructure items to delete. Teardown All removes all related items.",
        on_help_click=_mk_help_handler("Delete Runpod Template and/or Network Volume. If a pod exists, you can delete it too."),
    )
    td_template_cb = ft.Checkbox(label="Template: (none)", value=False, visible=False)
    td_volume_cb = ft.Checkbox(label="Volume: (none)", value=False, visible=False)
    td_pod_cb = ft.Checkbox(label="Pod: (none)", value=False, visible=False)
    td_busy = ft.ProgressRing(visible=False)

    async def _refresh_teardown_ui(_=None):
        return await refresh_teardown_ui_helper(
            page=page,
            rp_pod_module=rp_pod,
            train_state=train_state,
            td_template_cb=td_template_cb,
            td_volume_cb=td_volume_cb,
            td_pod_cb=td_pod_cb,
            teardown_section=teardown_section,
            _runpod_cfg=_runpod_cfg,
            rp_temp_key_tf=rp_temp_key_tf,
        )

    async def _do_teardown(selected_all: bool = False):
        return await do_teardown_helper(
            page=page,
            rp_pod_module=rp_pod,
            rp_infra_module=rp_infra,
            train_state=train_state,
            td_template_cb=td_template_cb,
            td_volume_cb=td_volume_cb,
            td_pod_cb=td_pod_cb,
            td_busy=td_busy,
            train_timeline=train_timeline,
            update_train_placeholders=update_train_placeholders,
            _runpod_cfg=_runpod_cfg,
            rp_temp_key_tf=rp_temp_key_tf,
            refresh_teardown_ui_fn=_refresh_teardown_ui,
            selected_all=selected_all,
        )

    async def on_teardown_selected(_=None):
        return await confirm_teardown_selected_helper(
            page=page,
            td_template_cb=td_template_cb,
            td_volume_cb=td_volume_cb,
            td_pod_cb=td_pod_cb,
            td_busy=td_busy,
            train_timeline=train_timeline,
            update_train_placeholders=update_train_placeholders,
            _runpod_cfg=_runpod_cfg,
            rp_temp_key_tf=rp_temp_key_tf,
            rp_pod_module=rp_pod,
            rp_infra_module=rp_infra,
            train_state=train_state,
            refresh_teardown_ui_fn=_refresh_teardown_ui,
        )

    async def on_teardown_all(_=None):
        return await confirm_teardown_all_helper(
            page=page,
            td_template_cb=td_template_cb,
            td_volume_cb=td_volume_cb,
            td_pod_cb=td_pod_cb,
            td_busy=td_busy,
            train_timeline=train_timeline,
            update_train_placeholders=update_train_placeholders,
            _runpod_cfg=_runpod_cfg,
            rp_temp_key_tf=rp_temp_key_tf,
            rp_pod_module=rp_pod,
            rp_infra_module=rp_infra,
            train_state=train_state,
            refresh_teardown_ui_fn=_refresh_teardown_ui,
        )

    teardown_section = build_teardown_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        td_template_cb=td_template_cb,
        td_volume_cb=td_volume_cb,
        td_pod_cb=td_pod_cb,
        td_busy=td_busy,
        on_teardown_selected_cb=lambda e: page.run_task(on_teardown_selected),
        on_teardown_all_cb=lambda e: page.run_task(on_teardown_all),
    )

    # Sections hidden until infrastructure is ensured successfully (delegated)
    dataset_section = build_dataset_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        train_source=train_source,
        train_hf_repo=train_hf_repo,
        train_hf_split=train_hf_split,
        train_hf_config=train_hf_config,
        train_json_path=train_json_path,
        visible=False,
    )

    train_params_section = build_train_params_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        skill_level=skill_level,
        beginner_mode_dd=beginner_mode_dd,
        expert_gpu_dd=expert_gpu_dd,
        expert_gpu_busy=expert_gpu_busy,
        expert_spot_cb=expert_spot_cb,
        expert_gpu_refresh_btn=expert_gpu_refresh_btn,
        base_model=base_model,
        epochs_tf=epochs_tf,
        lr_tf=lr_tf,
        batch_tf=batch_tf,
        grad_acc_tf=grad_acc_tf,
        max_steps_tf=max_steps_tf,
        use_lora_cb=use_lora_cb,
        out_dir_tf=out_dir_tf,
        packing_row=packing_row,
        auto_resume_row=auto_resume_row,
        push_row=push_row,
        hf_repo_row=hf_repo_row,
        resume_from_row=resume_from_row,
        advanced_params_section=advanced_params_section,
        visible=False,
    )

    # Combined holder for Dataset + Training Params. Styled depending on training target.
    ds_tp_group_container = ft.Container(
        content=ft.Column([
            dataset_section,
            train_params_section,
        ], spacing=12),
    )

    # Update visibility based on mode selection
    def _update_mode_visibility(_=None):
        mode = (config_mode_dd.value or "Normal").lower()
        is_cfg = mode.startswith("config")
        # Determine whether the current training target is Runpod or local
        try:
            tgt_val = (train_target_dd.value or "Runpod - Pod").lower()
            is_pod_target = tgt_val.startswith("runpod - pod")
        except Exception:
            is_pod_target = True
        try:
            config_files_row.visible = is_cfg
            # Keep rename/delete buttons in sync with selection when toggling mode
            _update_config_buttons_enabled()
            update_train_placeholders(); page.update()
        except Exception:
            pass
        try:
            dataset_section.visible = (not is_cfg)
            train_params_section.visible = (not is_cfg)
            # Runpod infrastructure UI is only meaningful for Runpod target
            rp_infra_panel.visible = (not is_cfg) and is_pod_target
            rp_infra_compact_row.visible = is_cfg and is_pod_target
            # Hide the grouped wrapper entirely in Config mode (Runpod)
            try:
                ds_tp_group_container.visible = (not is_cfg)
            except Exception:
                pass
        except Exception:
            pass
        try:
            page.update()
        except Exception:
            pass

    # Training target selector (top of Training tab)
    train_target_dd = ft.Dropdown(
        label="Training target",
        options=[
            ft.dropdown.Option("Runpod - Pod"),
            ft.dropdown.Option("local"),
        ],
        value="Runpod - Pod",
        width=420,
        tooltip="Choose where training runs. 'Runpod - Pod' uses the Runpod workflow; 'local' runs via Docker on this machine.",
    )

    # Hook handlers and initialize config list
    config_mode_dd.on_change = _update_mode_visibility
    config_files_dd.on_change = _update_config_buttons_enabled
    config_refresh_btn.on_click = _refresh_config_list
    load_config_btn.on_click = lambda e: page.run_task(on_load_config)
    config_save_current_btn.on_click = lambda e: page.run_task(on_save_current_config)
    config_edit_btn.on_click = lambda e: page.run_task(on_edit_config)
    config_rename_btn.on_click = lambda e: page.run_task(on_rename_config)
    config_delete_btn.on_click = lambda e: page.run_task(on_delete_config)
    # Auto-load last used config on startup if available, and set training target from it
    try:
        last_name = get_last_used_config_name_helper()
    except Exception:
        last_name = None
    if last_name:
        conf_for_last: dict = {}
        try:
            d = _saved_configs_dir()
            path = os.path.join(d, last_name)
            conf_for_last = _read_json_file(path) or {}
        except Exception:
            conf_for_last = {}
        # Try to set training target from config meta before refreshing list
        try:
            meta = conf_for_last.get("meta") or {}
        except Exception:
            meta = {}
        try:
            tgt = str(meta.get("train_target") or "").strip() or "Runpod - Pod"
        except Exception:
            tgt = "Runpod - Pod"
        try:
            train_target_dd.value = tgt
        except Exception:
            pass
        try:
            _refresh_config_list()
        except Exception:
            pass
        try:
            opts = getattr(config_files_dd, "options", []) or []
            keys = {getattr(o, "key", None) or getattr(o, "text", "") for o in opts}
            if last_name in keys:
                config_files_dd.value = last_name
                schedule_task(on_load_config)
        except Exception:
            pass
    else:
        try:
            _refresh_config_list()
        except Exception:
            pass
    # Ensure initial visibility matches the selected mode
    _update_mode_visibility()

    # -------- LOCAL TRAINING: System Specs --------
    # Delegated to helpers.local_specs (gather_local_specs, refresh_local_gpus)

    # Controls to show specs
    local_os_txt = ft.Text("", selectable=True)
    local_py_txt = ft.Text("")
    local_cpu_txt = ft.Text("")
    local_ram_txt = ft.Text("")
    local_disk_txt = ft.Text("")
    local_torch_txt = ft.Text("")
    local_cuda_txt = ft.Text("")
    local_gpus_txt = ft.Text("", selectable=True)
    local_capability_txt = ft.Text("", weight=ft.FontWeight.W_600)
    # Red flags UI (hidden when none)
    local_flags_col = ft.Column([], spacing=2)
    local_flags_box = ft.Column([
        ft.Row([
            ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED_400, size=18),
            ft.Text("Potential issues", weight=ft.FontWeight.W_600, color=COLORS.RED_400),
        ], spacing=6),
        local_flags_col,
    ], spacing=6, visible=False)

    # Docker image pull controls (Local section)
    # Use the same default image as the Runpod template to avoid tag mismatches.
    DEFAULT_DOCKER_IMAGE = "docker.io/sbussiso/unsloth-trainer:latest"
    docker_image_tf = ft.TextField(
        label="Docker image",
        value=DEFAULT_DOCKER_IMAGE,
        width=600,
        dense=True,
        hint_text="e.g., repo/image:tag",
    )
    docker_status = ft.Text("")
    docker_log_timeline = ft.ListView(expand=True, spacing=4, auto_scroll=True)
    docker_log_placeholder = ft.Text(
        "Docker pull logs will appear here after starting a pull.",
        color=WITH_OPACITY(0.5, BORDER_BASE),
    )
    docker_pull_ring = ft.ProgressRing(width=18, height=18, value=None, visible=False)
    docker_pull_btn = ft.ElevatedButton(
        "Pull image",
        icon=getattr(ICONS, "CLOUD_DOWNLOAD", getattr(ICONS, "DOWNLOAD", ICONS.CLOUD)),
    )
    # Bind pull action
    docker_pull_btn.on_click = lambda e: page.run_task(on_docker_pull)

    def _render_local_specs(data: dict):
        try:
            local_os_txt.value = f"OS: {data.get('os') or 'N/A'}"
            local_py_txt.value = f"Python: {data.get('python') or 'N/A'}"
            local_cpu_txt.value = f"CPU cores: {data.get('cpu_cores') or 'N/A'}"
            rg = data.get('ram_gb')
            local_ram_txt.value = f"RAM: {rg} GB" if rg is not None else "RAM: N/A"
            df = data.get('disk_free_gb')
            local_disk_txt.value = f"Disk free: {df} GB" if df is not None else "Disk free: N/A"
            local_torch_txt.value = f"PyTorch installed: {bool(data.get('torch_installed'))}"
            local_cuda_txt.value = f"CUDA available: {bool(data.get('cuda_available'))}"
            gpus_list = data.get('gpus') or []
            if gpus_list:
                try:
                    lines = []
                    for g in gpus_list:
                        nm = g.get("name") or "GPU"
                        vr = g.get("vram_gb")
                        lines.append(f"- {nm} — VRAM: {vr} GB" if vr is not None else f"- {nm}")
                    local_gpus_txt.value = "\n".join(lines)
                except Exception:
                    local_gpus_txt.value = str(gpus_list)
            else:
                local_gpus_txt.value = "(none)"
            # Red flags rendering
            flags = list(data.get("red_flags") or [])
            if flags:
                local_flags_col.controls = [
                    ft.Row([
                        ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED_400, size=16),
                        ft.Text(str(msg), color=COLORS.RED_400),
                    ], spacing=6)
                    for msg in flags
                ]
                local_flags_box.visible = True
            else:
                local_flags_col.controls = []
                local_flags_box.visible = False
            local_capability_txt.value = str(data.get('capability') or '')
        except Exception:
            pass

    async def on_refresh_local_specs(e=None):
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Gathering local system specs..."))
            page.open(page.snack_bar)
            await safe_update(page)
        except Exception:
            pass
        data = gather_local_specs_helper()
        _render_local_specs(data)
        try:
            await safe_update(page)
        except Exception:
            pass

    # ---------- LOCAL DOCKER: Run Training ----------
    # Controls for local docker run
    local_host_dir_tf = ft.TextField(
        label="Host data directory (mounted to /data)",
        width=600,
        dense=True,
        hint_text="Absolute path on your machine to mount as /data in the container",
    )
    # Directory picker for local host dir
    def _on_local_dir_picked(e):
        try:
            # e.path is set for get_directory_path
            sel = getattr(e, "path", None) or ""
            if sel:
                local_host_dir_tf.value = sel
            page.update()
        except Exception:
            pass
    local_dir_picker = ft.FilePicker(on_result=_on_local_dir_picked)
    try:
        page.overlay.append(local_dir_picker)
    except Exception:
        pass
    local_browse_btn = ft.OutlinedButton(
        "Browse…",
        icon=getattr(ICONS, "FOLDER_OPEN", getattr(ICONS, "FOLDER", ICONS.SEARCH)),
        on_click=lambda e: local_dir_picker.get_directory_path(dialog_title="Select host data directory to mount as /data"),
    )
    local_container_name_tf = ft.TextField(
        label="Container name",
        value=f"ds-local-train-{int(time.time())}",
        width=280,
        dense=True,
    )
    # Default GPU toggle based on quick capability probe when first rendered; will update on refresh
    local_use_gpu_cb = ft.Checkbox(label="Use NVIDIA GPU (adds --gpus all)", value=False)
    local_pass_hf_token_cb = ft.Checkbox(label="Pass HF token to container (HF_TOKEN / HUGGINGFACE_HUB_TOKEN)")
    local_train_status = ft.Text("")
    local_save_config_btn = ft.OutlinedButton(
        "Save current setup",
        icon=getattr(ICONS, "SAVE", ICONS.CHECK),
        tooltip="Save the current training setup (dataset, hyperparameters, target, and local Docker settings) as a reusable config.",
        on_click=lambda e: page.run_task(on_save_current_config),
    )
    local_train_progress = ft.ProgressBar(value=0.0, width=280)
    local_train_prog_label = ft.Text("Idle")
    local_train_timeline = ft.ListView(expand=True, spacing=4, auto_scroll=True)
    local_train_timeline_placeholder = ft.Text(
        "Logs will appear here after starting local training.",
        color=WITH_OPACITY(0.5, BORDER_BASE),
    )
    # Quick local inference controls (initially hidden until a local adapter is available)
    local_infer_status = ft.Text(
        "Quick local inference will be enabled after a successful local training run.",
        color=WITH_OPACITY(0.6, BORDER_BASE),
    )
    local_infer_meta = ft.Text(
        "",
        size=12,
        color=WITH_OPACITY(0.65, BORDER_BASE),
    )
    local_infer_prompt_tf = ft.TextField(
        label="Quick local inference prompt",
        multiline=True,
        min_lines=3,
        max_lines=6,
        width=1000,
        dense=True,
    )
    local_infer_preset_dd = ft.Dropdown(
        label="Preset",
        options=[
            ft.dropdown.Option("Deterministic"),
            ft.dropdown.Option("Balanced"),
            ft.dropdown.Option("Creative"),
        ],
        value="Balanced",
        width=220,
    )
    local_infer_temp_slider = ft.Slider(
        label="Temperature: {value}",
        min=0.1,
        max=1.2,
        divisions=11,
        value=0.7,
        width=320,
    )
    local_infer_max_tokens_slider = ft.Slider(
        label="Max new tokens: {value}",
        min=64,
        max=512,
        divisions=14,
        value=256,
        width=320,
    )
    local_infer_output = ft.ListView(expand=True, spacing=4, auto_scroll=True)
    local_infer_output_placeholder = ft.Text(
        "Responses will appear here after running inference.",
        color=WITH_OPACITY(0.5, BORDER_BASE),
    )
    local_infer_btn = ft.ElevatedButton(
        "Run Inference",
        icon=getattr(ICONS, "PLAY_CIRCLE", ICONS.PLAY_ARROW),
    )
    local_infer_clear_btn = ft.TextButton(
        "Clear history",
        icon=getattr(ICONS, "DELETE_SWEEP", getattr(ICONS, "DELETE", ICONS.CLOSE)),
    )
    local_infer_group_container = ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Quick Local Inference",
                    getattr(ICONS, "PSYCHOLOGY", getattr(ICONS, "CHAT", ICONS.PLAY_CIRCLE)),
                    "Test the latest local adapter with a prompt to verify its behavior.",
                    on_help_click=_mk_help_handler(
                        "Runs the fine-tuned adapter locally so you can sanity-check training results.",
                    ),
                ),
                local_infer_status,
                local_infer_meta,
                local_infer_preset_dd,
                local_infer_prompt_tf,
                ft.Row([
                    local_infer_temp_slider,
                    local_infer_max_tokens_slider,
                ], wrap=True, spacing=10),
                ft.Row([local_infer_btn, local_infer_clear_btn], wrap=True, spacing=10),
                ft.Container(
                    ft.Stack([local_infer_output, local_infer_output_placeholder], expand=True),
                    height=220,
                    width=1000,
                    border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                    border_radius=8,
                    padding=10,
                ),
            ],
            spacing=8,
        ),
        visible=False,
    )
    # Buffer for saving logs
    local_log_buffer: List[str] = []
    def on_local_infer_clear(e=None):
        try:
            local_infer_output.controls.clear()
            try:
                local_infer_output_placeholder.visible = True
            except Exception:
                pass
            local_infer_status.value = "Idle — history cleared."
            page.update()
        except Exception:
            pass
    def on_local_infer_preset_change(e=None):
        try:
            name = (local_infer_preset_dd.value or "Balanced").lower()
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
            local_infer_temp_slider.value = t
            local_infer_max_tokens_slider.value = n
            page.update()
        except Exception:
            pass
    # File picker + button to save logs
    def _on_save_logs(e):
        try:
            path = getattr(e, "path", None)
            if not path:
                return
            txt = "\n".join(local_log_buffer)
            if not txt.endswith("\n"):
                txt += "\n"
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            local_train_status.value = f"Saved logs to: {path}"
            page.update()
        except Exception as ex:
            local_train_status.value = f"Failed to save logs: {ex}"
            try:
                page.update()
            except Exception:
                pass
    local_logs_picker = ft.FilePicker(on_result=_on_save_logs)
    try:
        page.overlay.append(local_logs_picker)
    except Exception:
        pass
    try:
        local_infer_preset_dd.on_change = on_local_infer_preset_change
    except Exception:
        pass
    local_save_logs_btn = ft.OutlinedButton(
        "Download logs",
        icon=getattr(ICONS, "DOWNLOAD", getattr(ICONS, "SAVE_ALT", ICONS.SAVE)),
        on_click=lambda e: local_logs_picker.save_file(
            dialog_title="Save training logs",
            file_name=f"local-train-{int(time.time())}.log",
            allowed_extensions=["txt", "log"],
        ),
        disabled=True,
    )

    # Keep minimal state for local process/container
    train_state.setdefault("local", {})

    def _update_local_gpu_default_from_specs():
        try:
            data = gather_local_specs_helper()
            local_use_gpu_cb.value = bool(data.get("cuda_available"))
        except Exception:
            pass

    _update_local_gpu_default_from_specs()

    async def on_start_local_training(e=None):
        await run_local_training_helper(
            page=page,
            train_state=train_state,
            hf_token_tf=hf_token_tf,
            skill_level=skill_level,
            expert_gpu_dd=expert_gpu_dd,
            local_use_gpu_cb=local_use_gpu_cb,
            local_pass_hf_token_cb=local_pass_hf_token_cb,
            proxy_enable_cb=proxy_enable_cb,
            use_env_cb=use_env_cb,
            proxy_url_tf=proxy_url_tf,
            docker_image_tf=docker_image_tf,
            train_source=train_source,
            train_hf_repo=train_hf_repo,
            train_hf_split=train_hf_split,
            train_json_path=train_json_path,
            local_host_dir_tf=local_host_dir_tf,
            local_container_name_tf=local_container_name_tf,
            local_train_status=local_train_status,
            local_train_progress=local_train_progress,
            local_train_prog_label=local_train_prog_label,
            local_train_timeline=local_train_timeline,
            local_train_timeline_placeholder=local_train_timeline_placeholder,
            local_log_buffer=local_log_buffer,
            local_save_logs_btn=local_save_logs_btn,
            local_start_btn=local_start_btn,
            local_stop_btn=local_stop_btn,
            build_hp_fn=_build_hp,
            DEFAULT_DOCKER_IMAGE=DEFAULT_DOCKER_IMAGE,
            rp_pod_module=rp_pod,
            ICONS_module=ICONS,
        )
        try:
            info = train_state.get("local_infer") or {}
            adapter_path = (info.get("adapter_path") or "").strip()
            base_model_name = (info.get("base_model") or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit").strip()
            if adapter_path and os.path.isdir(adapter_path):
                local_infer_group_container.visible = True
                local_infer_status.value = "Idle — ready for local inference."
                local_infer_meta.value = f"Adapter: {adapter_path} • Base model: {base_model_name}"
            else:
                local_infer_status.value = "Quick local inference not available yet. Ensure training completed successfully."
                local_infer_meta.value = ""
            await safe_update(page)
        except Exception:
            pass

    async def on_stop_local_training(e=None):
        return await stop_local_training_helper(
            page=page,
            train_state=train_state,
            local_train_status=local_train_status,
            local_start_btn=local_start_btn,
            local_stop_btn=local_stop_btn,
            local_train_progress=local_train_progress,
            local_train_prog_label=local_train_prog_label,
        )

    async def on_local_infer_generate(e=None):
        prompt = (local_infer_prompt_tf.value or "").strip()
        if not prompt:
            local_infer_status.value = "Enter a prompt to test the latest local adapter."
            await safe_update(page)
            return
        info = train_state.get("local_infer") or {}
        adapter_path = (info.get("adapter_path") or "").strip()
        base_model_name = (info.get("base_model") or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit").strip()
        if (not adapter_path) or (not os.path.isdir(adapter_path)):
            local_infer_status.value = "Quick local inference is not ready. Run a successful local training first."
            await safe_update(page)
            return
        try:
            max_tokens = int(getattr(local_infer_max_tokens_slider, "value", 256) or 256)
        except Exception:
            max_tokens = 256
        try:
            temperature = float(getattr(local_infer_temp_slider, "value", 0.7) or 0.7)
        except Exception:
            temperature = 0.7
        if max_tokens <= 0:
            max_tokens = 1
        if temperature <= 0:
            temperature = 0.1
        loaded = bool(info.get("model_loaded"))
        local_infer_btn.disabled = True
        if not loaded:
            local_infer_status.value = "Loading model and generating response..."
        else:
            local_infer_status.value = "Generating response..."
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
                local_infer_output_placeholder.visible = False
            except Exception:
                pass
            local_infer_output.controls.append(
                ft.Column(
                    [
                        ft.Text("Prompt", weight=getattr(ft.FontWeight, "BOLD", None)),
                        ft.Text(prompt),
                        ft.Text("Response", weight=getattr(ft.FontWeight, "BOLD", None)),
                        ft.Text(text),
                    ],
                    spacing=4,
                )
            )
            try:
                train_state.setdefault("local_infer", {})
                train_state["local_infer"]["model_loaded"] = True
            except Exception:
                pass
            local_infer_status.value = "Idle — last inference complete."
        except Exception as ex:
            local_infer_status.value = f"Inference failed: {ex}"
        finally:
            local_infer_btn.disabled = False
            await safe_update(page)

    local_start_btn = ft.ElevatedButton(
        "Start Local Training",
        icon=getattr(ICONS, "PLAY_CIRCLE", ICONS.PLAY_ARROW),
        on_click=lambda e: page.run_task(on_start_local_training),
    )
    local_stop_btn = ft.OutlinedButton(
        "Stop",
        icon=getattr(ICONS, "STOP_CIRCLE", ICONS.STOP),
        disabled=True,
        on_click=lambda e: page.run_task(on_stop_local_training),
    )
    local_infer_btn.on_click = lambda e: page.run_task(on_local_infer_generate)
    local_infer_clear_btn.on_click = on_local_infer_clear

    async def on_docker_pull(e=None):
        logger.info("Docker pull button clicked")
        try:
            logger.info(
                "Docker pull UI before update: disabled=%s text=%r ring_visible=%s",
                getattr(docker_pull_btn, "disabled", None),
                getattr(docker_pull_btn, "text", None),
                getattr(docker_pull_ring, "visible", None),
            )
            docker_pull_btn.disabled = True
            docker_pull_btn.text = "Pulling..."
            docker_pull_ring.visible = True
            await safe_update(page)
            logger.info(
                "Docker pull UI after set busy: disabled=%s text=%r ring_visible=%s",
                docker_pull_btn.disabled,
                getattr(docker_pull_btn, "text", None),
                docker_pull_ring.visible,
            )
        except Exception:
            logger.exception("Failed to update Docker pull UI state before pull")
        try:
            logger.info("Starting on_docker_pull_helper (Docker CLI pull)")
            await on_docker_pull_helper(
                page=page,
                ICONS=ICONS,
                COLORS=COLORS,
                docker_image_tf=docker_image_tf,
                docker_status=docker_status,
                DEFAULT_DOCKER_IMAGE=DEFAULT_DOCKER_IMAGE,
                docker_log_timeline=docker_log_timeline,
                docker_log_placeholder=docker_log_placeholder,
            )
            logger.info("on_docker_pull_helper completed")
        except Exception:
            logger.exception("on_docker_pull_helper raised an error")
        finally:
            try:
                docker_pull_btn.disabled = False
                docker_pull_btn.text = "Pull image"
                docker_pull_ring.visible = False
                await safe_update(page)
                logger.info(
                    "Docker pull UI after reset: disabled=%s text=%r ring_visible=%s",
                    docker_pull_btn.disabled,
                    getattr(docker_pull_btn, "text", None),
                    docker_pull_ring.visible,
                )
            except Exception:
                logger.exception("Failed to reset Docker pull UI state after pull")

    # Local specs + Docker pull / local training container (delegated layout)
    local_specs_container = build_local_specs_container(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        REFRESH_ICON=REFRESH_ICON,
        local_os_txt=local_os_txt,
        local_py_txt=local_py_txt,
        local_cpu_txt=local_cpu_txt,
        local_ram_txt=local_ram_txt,
        local_disk_txt=local_disk_txt,
        local_torch_txt=local_torch_txt,
        local_cuda_txt=local_cuda_txt,
        local_gpus_txt=local_gpus_txt,
        local_flags_box=local_flags_box,
        local_capability_txt=local_capability_txt,
        docker_image_tf=docker_image_tf,
        docker_pull_btn=docker_pull_btn,
        docker_pull_ring=docker_pull_ring,
        docker_status=docker_status,
        docker_log_timeline=docker_log_timeline,
        docker_log_placeholder=docker_log_placeholder,
        refresh_specs_click_cb=lambda e: page.run_task(on_refresh_local_specs),
        local_host_dir_tf=local_host_dir_tf,
        local_browse_btn=local_browse_btn,
        local_container_name_tf=local_container_name_tf,
        local_use_gpu_cb=local_use_gpu_cb,
        local_pass_hf_token_cb=local_pass_hf_token_cb,
        local_train_progress=local_train_progress,
        local_train_prog_label=local_train_prog_label,
        local_save_logs_btn=local_save_logs_btn,
        local_train_timeline=local_train_timeline,
        local_train_timeline_placeholder=local_train_timeline_placeholder,
        local_start_btn=local_start_btn,
        local_stop_btn=local_stop_btn,
        local_train_status=local_train_status,
        local_infer_group_container=local_infer_group_container,
        local_save_config_btn=local_save_config_btn,
        mk_help_handler=_mk_help_handler,
    )

    # Pod logs section (delegated)
    pod_logs_section = build_pod_logs_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        train_progress=train_progress,
        train_prog_label=train_prog_label,
        train_timeline=train_timeline,
        train_timeline_placeholder=train_timeline_placeholder,
        mk_help_handler=_mk_help_handler,
    )

    # Wrap the existing Training content so we can hide it for non-pod targets (delegated)
    pod_content_container = build_pod_content_container(
        config_section=config_section,
        rp_infra_panel=rp_infra_panel,
        ds_tp_group_container=ds_tp_group_container,
        pod_logs_section=pod_logs_section,
        teardown_section=teardown_section,
        train_actions=train_actions,
    )

    train_target_profiles = {"runpod": {}, "local": {}}
    train_target_state = {"current": "runpod"}

    def _target_key_from_value(val: str) -> str:
        v = (val or "").lower()
        if v.startswith("runpod - pod"):
            return "runpod"
        return "local"

    def _snapshot_hp_for_target(target_key: str):
        prof = train_target_profiles.get(target_key)
        if prof is None:
            prof = {}
            train_target_profiles[target_key] = prof
        prof["train_source"] = train_source.value
        prof["train_hf_repo"] = train_hf_repo.value
        prof["train_hf_split"] = train_hf_split.value
        prof["train_hf_config"] = train_hf_config.value
        prof["train_json_path"] = train_json_path.value
        prof["base_model"] = base_model.value
        prof["epochs"] = epochs_tf.value
        prof["lr"] = lr_tf.value
        prof["batch"] = batch_tf.value
        prof["grad_acc"] = grad_acc_tf.value
        prof["max_steps"] = max_steps_tf.value
        prof["use_lora"] = bool(getattr(use_lora_cb, "value", False))
        prof["out_dir"] = out_dir_tf.value
        prof["packing"] = bool(getattr(packing_cb, "value", False))
        prof["auto_resume"] = bool(getattr(auto_resume_cb, "value", False))
        prof["push"] = bool(getattr(push_cb, "value", False))
        prof["hf_repo_id"] = hf_repo_id_tf.value
        prof["resume_from"] = resume_from_tf.value
        prof["warmup_steps"] = warmup_steps_tf.value
        prof["weight_decay"] = weight_decay_tf.value
        prof["lr_sched"] = lr_sched_dd.value
        prof["optim"] = optim_dd.value
        prof["logging_steps"] = logging_steps_tf.value
        prof["logging_first_step"] = bool(getattr(logging_first_step_cb, "value", False))
        prof["disable_tqdm"] = bool(getattr(disable_tqdm_cb, "value", False))
        prof["seed"] = seed_tf.value
        prof["save_strategy"] = save_strategy_dd.value
        prof["save_total_limit"] = save_total_limit_tf.value
        prof["pin_memory"] = bool(getattr(pin_memory_cb, "value", False))
        prof["report_to"] = report_to_dd.value
        prof["fp16"] = bool(getattr(fp16_cb, "value", False))
        prof["bf16"] = bool(getattr(bf16_cb, "value", False))

    def _apply_hp_for_target(target_key: str):
        prof = train_target_profiles.get(target_key) or {}
        if (not prof) and (target_key == "local"):
            base_prof = train_target_profiles.get("runpod") or {}
            prof = dict(base_prof)
            out_dir_val = (prof.get("out_dir") or "").strip()
            if (not out_dir_val) or (out_dir_val == "/data/outputs/runpod_run"):
                prof["out_dir"] = "/data/outputs/local_run"
            train_target_profiles["local"] = prof
        v = prof.get("train_source")
        if v is not None:
            train_source.value = v
        v = prof.get("train_hf_repo")
        if v is not None:
            train_hf_repo.value = v
        v = prof.get("train_hf_split")
        if v is not None:
            train_hf_split.value = v
        v = prof.get("train_hf_config")
        if v is not None:
            train_hf_config.value = v
        v = prof.get("train_json_path")
        if v is not None:
            train_json_path.value = v
        v = prof.get("base_model")
        if v is not None:
            base_model.value = v
        v = prof.get("epochs")
        if v is not None:
            epochs_tf.value = v
        v = prof.get("lr")
        if v is not None:
            lr_tf.value = v
        v = prof.get("batch")
        if v is not None:
            batch_tf.value = v
        v = prof.get("grad_acc")
        if v is not None:
            grad_acc_tf.value = v
        v = prof.get("max_steps")
        if v is not None:
            max_steps_tf.value = v
        if "use_lora" in prof:
            use_lora_cb.value = bool(prof.get("use_lora"))
        v = prof.get("out_dir")
        if v is not None:
            out_dir_tf.value = v
        if "packing" in prof:
            packing_cb.value = bool(prof.get("packing"))
        if "auto_resume" in prof:
            auto_resume_cb.value = bool(prof.get("auto_resume"))
        if "push" in prof:
            push_cb.value = bool(prof.get("push"))
        v = prof.get("hf_repo_id")
        if v is not None:
            hf_repo_id_tf.value = v
        v = prof.get("resume_from")
        if v is not None:
            resume_from_tf.value = v
        v = prof.get("warmup_steps")
        if v is not None:
            warmup_steps_tf.value = v
        v = prof.get("weight_decay")
        if v is not None:
            weight_decay_tf.value = v
        v = prof.get("lr_sched")
        if v is not None:
            lr_sched_dd.value = v
        v = prof.get("optim")
        if v is not None:
            optim_dd.value = v
        v = prof.get("logging_steps")
        if v is not None:
            logging_steps_tf.value = v
        if "logging_first_step" in prof:
            logging_first_step_cb.value = bool(prof.get("logging_first_step"))
        if "disable_tqdm" in prof:
            disable_tqdm_cb.value = bool(prof.get("disable_tqdm"))
        v = prof.get("seed")
        if v is not None:
            seed_tf.value = v
        v = prof.get("save_strategy")
        if v is not None:
            save_strategy_dd.value = v
        v = prof.get("save_total_limit")
        if v is not None:
            save_total_limit_tf.value = v
        if "pin_memory" in prof:
            pin_memory_cb.value = bool(prof.get("pin_memory"))
        v = prof.get("report_to")
        if v is not None:
            report_to_dd.value = v
        if "fp16" in prof:
            fp16_cb.value = bool(prof.get("fp16"))
        if "bf16" in prof:
            bf16_cb.value = bool(prof.get("bf16"))

    def _update_training_target(_=None):
        val = train_target_dd.value or "Runpod - Pod"
        try:
            new_key = _target_key_from_value(val)
            prev_key = train_target_state.get("current") or "runpod"
            if new_key != prev_key:
                try:
                    _snapshot_hp_for_target(prev_key)
                except Exception:
                    pass
                try:
                    _apply_hp_for_target(new_key)
                except Exception:
                    pass
                train_target_state["current"] = new_key
        except Exception:
            pass
        # Refresh config list when target changes so Runpod/local configs are separated
        try:
            _refresh_config_list()
        except Exception:
            pass
        target = (val or "").lower()
        is_pod = target.startswith("runpod - pod")
        try:
            # Always show the wrapper; toggle inner sections as needed
            pod_content_container.visible = True
            # Show Local Specs when local
            local_specs_container.visible = (not is_pod)
            # Toggle Runpod-only panels and pod log/actions
            try:
                rp_infra_panel.visible = is_pod
            except Exception:
                pass
            try:
                pod_logs_section.visible = is_pod
            except Exception:
                pass
            try:
                train_progress.visible = is_pod
                train_prog_label.visible = is_pod
                train_timeline.visible = is_pod
                train_timeline_placeholder.visible = is_pod
            except Exception:
                pass
            try:
                teardown_section.visible = is_pod
            except Exception:
                pass
            try:
                train_actions.visible = is_pod
            except Exception:
                pass
            # Configuration section is available for managing configs; keep it visible for all targets
            try:
                config_section.visible = True
            except Exception:
                pass
            # Recompose the Dataset + Training Params grouping depending on target
            try:
                if is_pod:
                    # Runpod view: bordered form like local
                    ds_tp_group_container.content = ft.Column([
                        section_title(
                            "Runpod: Dataset & Params",
                            getattr(ICONS, "LIST_ALT", getattr(ICONS, "DESCRIPTION", ICONS.SETTINGS)),
                            "Choose dataset and configure training parameters for Runpod pods.",
                            on_help_click=_mk_help_handler("Choose dataset source and configure training parameters for Runpod pod training."),
                        ),
                        ft.Container(
                            content=ft.Column([
                                dataset_section,
                                train_params_section,
                            ], spacing=0),
                            width=1000,
                            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                            border_radius=8,
                            padding=10,
                        ),
                    ], spacing=12)
                else:
                    # Local view: add a titled bordered form around Dataset + Training Params
                    ds_tp_group_container.content = ft.Column([
                        section_title(
                            "Local Training: Dataset & Params",
                            getattr(ICONS, "LIST_ALT", getattr(ICONS, "DESCRIPTION", ICONS.SETTINGS)),
                            "Choose dataset and set training parameters for local runs.",
                            on_help_click=_mk_help_handler("Choose dataset source and configure training parameters for local Docker runs."),
                        ),
                        ft.Container(
                            content=ft.Column([
                                dataset_section,
                                train_params_section,
                            ], spacing=0),
                            width=1000,
                            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                            border_radius=8,
                            padding=10,
                        ),
                    ], spacing=12)
            except Exception:
                pass
        except Exception:
            pass
        # Ensure normal/config mode visibility still applies for both targets
        try:
            _update_mode_visibility()
        except Exception:
            pass
        if is_pod:
            # Restore Runpod spot visibility depending on skill
            try:
                expert_spot_cb.disabled = False
                expert_spot_cb.visible = ((skill_level.value or "Beginner").lower() != "beginner")
            except Exception:
                pass
            # If switching back to Runpod and expert GPU list looks local/unpopulated, refresh Runpod GPUs
            try:
                is_beginner = ((skill_level.value or "Beginner").lower() == "beginner")
                if not is_beginner:
                    opts = (getattr(expert_gpu_dd, "options", []) or [])
                    dd_tip = (getattr(expert_gpu_dd, "tooltip", "") or "").lower()
                    if (len(opts) <= 1) or ("local" in dd_tip):
                        if hasattr(page, "run_task"):
                            page.run_task(refresh_expert_gpus)
            except Exception:
                pass
        else:
            # When switching to local, refresh specs
            try:
                if hasattr(page, "run_task"):
                    page.run_task(on_refresh_local_specs)
            except Exception:
                pass
            # Local target always uses Normal semantics; force dataset/params visible
            try:
                dataset_section.visible = True
                train_params_section.visible = True
                ds_tp_group_container.visible = True
            except Exception:
                pass
            # Hide Runpod-only spot toggle; use local GPUs for expert picker
            try:
                expert_spot_cb.value = False
                expert_spot_cb.disabled = True
                expert_spot_cb.visible = False
            except Exception:
                pass
            # Show/hide local GPU checkbox based on skill (beginner uses checkbox, expert uses dropdown)
            try:
                is_beginner = ((skill_level.value or "Beginner").lower() == "beginner")
                local_use_gpu_cb.visible = is_beginner
            except Exception:
                pass
            # If expert mode and dropdown not populated with locals yet, populate
            try:
                if ((skill_level.value or "Beginner").lower() != "beginner") and (len(getattr(expert_gpu_dd, "options", []) or []) <= 1):
                    if hasattr(page, "run_task"):
                        page.run_task(refresh_local_gpus)
            except Exception:
                pass
        try:
            page.update()
        except Exception:
            pass

    train_target_dd.on_change = _update_training_target
    # Initialize visibility based on default/loaded target
    _update_training_target()

    training_tab = build_training_tab(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        train_target_dd=train_target_dd,
        pod_content_container=pod_content_container,
        local_specs_container=local_specs_container,
    )

    # ---------- SETTINGS TAB (Proxy config + Ollama) ----------
    settings_tab = build_settings_tab(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        proxy_enable_cb=proxy_enable_cb,
        use_env_cb=use_env_cb,
        proxy_url_tf=proxy_url_tf,
        hf_token_tf=hf_token_tf,
        hf_status=hf_status,
        hf_test_btn=hf_test_btn,
        hf_save_btn=hf_save_btn,
        hf_remove_btn=hf_remove_btn,
        runpod_key_tf=runpod_key_tf,
        runpod_status=runpod_status,
        runpod_test_btn=runpod_test_btn,
        runpod_save_btn=runpod_save_btn,
        runpod_remove_btn=runpod_remove_btn,
        ollama_enable_cb=ollama_enable_cb,
        ollama_base_url_tf=ollama_base_url_tf,
        ollama_default_model_tf=ollama_default_model_tf,
        ollama_models_dd=ollama_models_dd,
        ollama_test_btn=ollama_test_btn,
        ollama_refresh_btn=ollama_refresh_btn,
        ollama_save_btn=ollama_save_btn,
        ollama_status=ollama_status,
        REFRESH_ICON=REFRESH_ICON,
    )
    
    # ---- Dataset Analysis tab: UI controls for builder ----
    def kpi_tile(title: str, value, subtitle: str = "", icon=None):
        # Accept either a string or a Flet control for value, so we can update it dynamically later.
        val_ctrl = value if isinstance(value, ft.Control) else ft.Text(str(value), size=18, weight=ft.FontWeight.W_600)
        return ft.Container(
            content=ft.Row([
                ft.Icon(icon or getattr(ICONS, "INSIGHTS", ICONS.SEARCH), size=20, color=ACCENT_COLOR),
                ft.Column([
                    ft.Text(title, size=12, color=WITH_OPACITY(0.7, BORDER_BASE)),
                    val_ctrl,
                    ft.Text(subtitle, size=11, color=WITH_OPACITY(0.6, BORDER_BASE)) if subtitle else ft.Container(),
                ], spacing=2),
            ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            width=230,
            padding=12,
            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
            border_radius=8,
        )

    analysis_overview_note = ft.Text(
        "Click Analyze to compute dataset insights: totals, lengths, duplicates, sentiment, class balance, and samples.",
        size=12,
        color=WITH_OPACITY(0.7, BORDER_BASE),
    )

    # Sentiment controls (dynamic)
    sent_pos_label = ft.Text("Positive", width=90)
    sent_pos_bar = ft.ProgressBar(value=0.0, width=240)
    sent_pos_pct = ft.Text("0%", width=50, text_align=ft.TextAlign.END)
    sent_neu_label = ft.Text("Neutral", width=90)
    sent_neu_bar = ft.ProgressBar(value=0.0, width=240)
    sent_neu_pct = ft.Text("0%", width=50, text_align=ft.TextAlign.END)
    sent_neg_label = ft.Text("Negative", width=90)
    sent_neg_bar = ft.ProgressBar(value=0.0, width=240)
    sent_neg_pct = ft.Text("0%", width=50, text_align=ft.TextAlign.END)
    sentiment_row = ft.Column([
        ft.Row([sent_pos_label, sent_pos_bar, sent_pos_pct], vertical_alignment=ft.CrossAxisAlignment.CENTER),
        ft.Row([sent_neu_label, sent_neu_bar, sent_neu_pct], vertical_alignment=ft.CrossAxisAlignment.CENTER),
        ft.Row([sent_neg_label, sent_neg_bar, sent_neg_pct], vertical_alignment=ft.CrossAxisAlignment.CENTER),
    ], spacing=6)

    # Class balance proxy (dynamic) — we use input length buckets: Short/Medium/Long
    class_a_label = ft.Text("Short", width=90)
    class_a_bar = ft.ProgressBar(value=0.0, width=240)
    class_a_pct = ft.Text("0%", width=50, text_align=ft.TextAlign.END)
    class_b_label = ft.Text("Medium", width=90)
    class_b_bar = ft.ProgressBar(value=0.0, width=240)
    class_b_pct = ft.Text("0%", width=50, text_align=ft.TextAlign.END)
    class_c_label = ft.Text("Long", width=90)
    class_c_bar = ft.ProgressBar(value=0.0, width=240)
    class_c_pct = ft.Text("0%", width=50, text_align=ft.TextAlign.END)
    class_balance_row = ft.Column([
        ft.Row([class_a_label, class_a_bar, class_a_pct]),
        ft.Row([class_b_label, class_b_bar, class_b_pct]),
        ft.Row([class_c_label, class_c_bar, class_c_pct]),
    ], spacing=6)

    # Wrap Sentiment and Class Balance into sections to toggle visibility later
    sentiment_section = ft.Container(
        sentiment_row,
        padding=8,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        visible=False,
    )
    class_balance_section = ft.Container(
        class_balance_row,
        padding=8,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        visible=False,
    )

    # Grid table view for detailed samples (dynamic)
    SAMPLE_INPUT_W = 420
    SAMPLE_OUTPUT_W = 420
    SAMPLE_LEN_W = 70
    samples_grid = ft.DataTable(
        column_spacing=12,
        data_row_min_height=40,
        heading_row_height=40,
        columns=[
            ft.DataColumn(ft.Container(width=SAMPLE_INPUT_W, content=ft.Text("Input"))),
            ft.DataColumn(ft.Container(width=SAMPLE_OUTPUT_W, content=ft.Text("Output"))),
            ft.DataColumn(ft.Container(width=SAMPLE_LEN_W, content=ft.Text("In len", text_align=ft.TextAlign.END))),
            ft.DataColumn(ft.Container(width=SAMPLE_LEN_W, content=ft.Text("Out len", text_align=ft.TextAlign.END))),
        ],
        rows=[],
    )

    # Extra metrics table (for optional modules)
    extra_metrics_table = ft.DataTable(
        column_spacing=12,
        data_row_min_height=32,
        heading_row_height=36,
        columns=[
            ft.DataColumn(ft.Container(width=220, content=ft.Text("Metric"))),
            ft.DataColumn(ft.Container(width=560, content=ft.Text("Value"))),
        ],
        rows=[],
    )
    extra_metrics_section = ft.Container(
        extra_metrics_table,
        padding=8,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        visible=False,
    )

    # Samples section wrapper (hidden until results are available)
    samples_section = ft.Container(
        samples_grid,
        padding=8,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        visible=False,
    )

    # Dataset selector controls for Analysis (HF or JSON)
    analysis_source_dd = ft.Dropdown(
        label="Dataset source",
        options=[ft.dropdown.Option("Hugging Face"), ft.dropdown.Option("JSON file")],
        value="Hugging Face",
        width=180,
    )
    analysis_hf_repo = ft.TextField(label="Dataset repo (e.g., username/dataset)", width=360, visible=True)
    analysis_hf_split = ft.TextField(label="Split", value="train", width=120, visible=True)
    analysis_hf_config = ft.TextField(label="Config (optional)", width=180, visible=True)
    analysis_json_path = ft.TextField(label="JSON path", width=360, visible=False)

    analysis_dataset_hint = ft.Text("Select a dataset to analyze.", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))
    # Analysis runtime settings (UI only)
    analysis_backend_dd = ft.Dropdown(
        label="Backend",
        options=[ft.dropdown.Option("HF Inference API"), ft.dropdown.Option("Local (Transformers)")],
        value="HF Inference API",
        width=220,
    )
    analysis_hf_token_tf = ft.TextField(
        label="HF token (optional)",
        width=360,
        password=True,
        can_reveal_password=True,
        visible=True,
    )
    analysis_sample_size_tf = ft.TextField(label="Sample size", value="5000", width=140)

    # Analysis module toggles
    cb_basic_stats = ft.Checkbox(label="Basic Stats", value=True, tooltip="Record count and average input/output lengths.")
    cb_duplicates = ft.Checkbox(label="Duplicates & Similarity", tooltip="Approximate duplicate/similarity detection via hashing heuristics.")
    cb_sentiment = ft.Checkbox(label="Sentiment", value=True, tooltip="Heuristic sentiment distribution over sampled records.")
    cb_class_balance = ft.Checkbox(label="Class balance", value=True, tooltip="Distribution of labels/classes if present.")
    cb_coverage_overlap = ft.Checkbox(label="Coverage Overlap", tooltip="Overlap of input and output tokens (higher may indicate copying).")
    cb_data_leakage = ft.Checkbox(label="Data Leakage Check", tooltip="Flags potential target text appearing in inputs.")
    cb_conversation_depth = ft.Checkbox(label="Conversation Depth", tooltip="Estimated turns/exchanges in dialogue-like data.")
    cb_speaker_balance = ft.Checkbox(label="Speaker Balance", tooltip="Balance of speakers/roles when such tags exist.")
    cb_question_statement = ft.Checkbox(label="Question vs Statement", tooltip="Ratio of questions to statements in inputs.")
    cb_readability = ft.Checkbox(label="Readability", tooltip="Simple readability proxy (length, punctuation).")
    cb_ner = ft.Checkbox(label="NER", tooltip="Counts of proper nouns/capitalized tokens as NER proxy.")
    cb_toxicity = ft.Checkbox(label="Toxicity / Safety", tooltip="Flags profanity or unsafe terms (heuristic).")
    cb_politeness = ft.Checkbox(label="Politeness / Formality", tooltip="Presence of polite markers (please, thanks, etc.).")
    cb_dialogue_acts = ft.Checkbox(label="Dialogue Acts", tooltip="Heuristic dialogue acts (question/command/statement).")
    cb_topics = ft.Checkbox(label="Topics / Clustering", tooltip="Top keywords proxy for topics.")
    cb_alignment = ft.Checkbox(label="Alignment (Similarity/NLI)", tooltip="Rough input/output semantic alignment proxy.")
    # Select-all toggle for analysis modules
    select_all_modules_cb = ft.Checkbox(label="Select all", value=False)

    # Analyze button; enabled only when dataset is selected
    analyze_btn = ft.ElevatedButton(
        "Analyze dataset",
        icon=getattr(ICONS, "INSIGHTS", getattr(ICONS, "ANALYTICS", ICONS.SEARCH)),
        disabled=True,
        on_click=lambda e: page.run_task(on_analyze),
    )
    # Ensure there's always a snackbar to open (handle older Flet without attribute)
    if not getattr(page, "snack_bar", None):
        page.snack_bar = ft.SnackBar(ft.Text("Analysis ready."))

    def _validate_analysis_dataset(_=None):
        try:
            src = (analysis_source_dd.value or "Hugging Face")
        except Exception:
            src = "Hugging Face"
        repo = (analysis_hf_repo.value or "").strip()
        jpath = (analysis_json_path.value or "").strip()
        if src == "Hugging Face":
            valid = bool(repo)
            desc = f"Selected: HF {repo} [{(analysis_hf_split.value or 'train').strip()}]"
        else:
            valid = bool(jpath)
            desc = f"Selected: JSON {jpath}" if jpath else "Select a JSON file path"
        analyze_btn.disabled = not valid
        analysis_dataset_hint.value = desc if valid else "Select a dataset to analyze."
        try:
            page.update()
        except Exception:
            pass

    def _update_analysis_source(_=None):
        is_hf = (getattr(analysis_source_dd, "value", "Hugging Face") or "Hugging Face") == "Hugging Face"
        analysis_hf_repo.visible = is_hf
        analysis_hf_split.visible = is_hf
        analysis_hf_config.visible = is_hf
        analysis_json_path.visible = not is_hf
        _validate_analysis_dataset()

    def _update_analysis_backend(_=None):
        use_api = (getattr(analysis_backend_dd, "value", "HF Inference API") or "HF Inference API") == "HF Inference API"
        analysis_hf_token_tf.visible = use_api
        try:
            page.update()
        except Exception:
            pass

    # Wire up events
    analysis_source_dd.on_change = _update_analysis_source
    analysis_hf_repo.on_change = _validate_analysis_dataset
    analysis_hf_split.on_change = _validate_analysis_dataset
    analysis_json_path.on_change = _validate_analysis_dataset
    analysis_backend_dd.on_change = _update_analysis_backend

    # Helpers for analysis modules selection
    def _all_analysis_modules():
        return [
            cb_basic_stats,
            cb_duplicates,
            cb_sentiment,
            cb_class_balance,
            cb_coverage_overlap,
            cb_data_leakage,
            cb_conversation_depth,
            cb_speaker_balance,
            cb_question_statement,
            cb_readability,
            cb_ner,
            cb_toxicity,
            cb_politeness,
            cb_dialogue_acts,
            cb_topics,
            cb_alignment,
        ]

    def _sync_select_all_modules():
        try:
            select_all_modules_cb.value = all(bool(getattr(m, "value", False)) for m in _all_analysis_modules())
            page.update()
        except Exception:
            pass

    def _on_select_all_modules_change(_):
        try:
            val = bool(getattr(select_all_modules_cb, "value", False))
            for m in _all_analysis_modules():
                m.value = val
            page.update()
        except Exception:
            pass

    def _on_module_cb_change(_):
        _sync_select_all_modules()

    # Attach module checkbox events
    try:
        select_all_modules_cb.on_change = _on_select_all_modules_change
        for _m in _all_analysis_modules():
            _m.on_change = _on_module_cb_change
    except Exception:
        pass

    # --- Analysis backend state & handler ---
    analysis_state = {"running": False}
    analysis_busy_ring = ft.ProgressRing(value=None, visible=False, width=18, height=18)

    # KPI dynamic value controls
    kpi_total_value = ft.Text("—", size=18, weight=ft.FontWeight.W_600)
    kpi_avg_in_value = ft.Text("—", size=18, weight=ft.FontWeight.W_600)
    kpi_avg_out_value = ft.Text("—", size=18, weight=ft.FontWeight.W_600)
    kpi_dupe_value = ft.Text("—", size=18, weight=ft.FontWeight.W_600)

    async def on_analyze(_=None):
        if analysis_state.get("running"):
            return
        analysis_state["running"] = True
        try:
            analyze_btn.disabled = True
            analysis_busy_ring.visible = True
            # Hide results while computing a fresh run
            overview_block.visible = False
            sentiment_block.visible = False
            class_balance_block.visible = False
            extra_metrics_block.visible = False
            samples_block.visible = False
            samples_section.visible = False
            div_overview.visible = False
            div_sentiment.visible = False
            div_class.visible = False
            div_extra.visible = False
            div_samples.visible = False
            await safe_update(page)

            src = (analysis_source_dd.value or "Hugging Face")
            repo = (analysis_hf_repo.value or "").strip()
            split = (analysis_hf_split.value or "train").strip()
            cfg = (analysis_hf_config.value or "").strip() or None
            jpath = (analysis_json_path.value or "").strip()
            try:
                sample_size = int(float((analysis_sample_size_tf.value or "5000").strip()))
                sample_size = max(1, min(250000, sample_size))
            except Exception:
                sample_size = 5000

            # Load examples as list[{input, output}]
            examples: list[dict] = []
            total_records = 0

            if src == "Hugging Face":
                if load_dataset is None:
                    raise RuntimeError("datasets library not available — cannot load from Hub")

                async def _load_hf(repo_id: str, sp: str, name: Optional[str]):
                    def do_load():
                        return load_dataset(repo_id, split=sp, name=name)
                    try:
                        return await asyncio.to_thread(do_load)
                    except Exception as e:
                        msg = str(e).lower()
                        if (get_dataset_config_names is not None) and ("config name is missing" in msg or "config name is required" in msg):
                            try:
                                cfgs = await asyncio.to_thread(lambda: get_dataset_config_names(repo_id))
                            except Exception:
                                cfgs = []
                            pick = None
                            for pref in ("main", "default", "socratic"):
                                if pref in cfgs:
                                    pick = pref
                                    break
                            if not pick and cfgs:
                                pick = cfgs[0]
                            if pick:
                                return await asyncio.to_thread(lambda: load_dataset(repo_id, split=sp, name=pick))
                        raise

                ds = await _load_hf(repo, split, cfg)
                try:
                    names = list(getattr(ds, "column_names", []) or [])
                except Exception:
                    names = []
                inn, outn = guess_input_output_columns(names)
                if not inn or not outn:
                    # If already in expected schema, allow it
                    if "input" in names and "output" in names:
                        inn, outn = "input", "output"
                    else:
                        raise RuntimeError(f"Could not resolve input/output columns for {repo} (have: {', '.join(names)})")

                # Prepare two-column view
                def mapper(batch):
                    srcs = batch.get(inn, [])
                    tgts = batch.get(outn, [])
                    return {
                        "input": ["" if v is None else str(v).strip() for v in srcs],
                        "output": ["" if v is None else str(v).strip() for v in tgts],
                    }

                try:
                    mapped = await asyncio.to_thread(
                        lambda: ds.map(mapper, batched=True, remove_columns=list(getattr(ds, "column_names", []) or []))
                    )
                except Exception:
                    # Fallback: iterate to python list
                    tmp = []
                    for r in ds:
                        tmp.append({
                            "input": "" if r.get(inn) is None else str(r.get(inn)).strip(),
                            "output": "" if r.get(outn) is None else str(r.get(outn)).strip(),
                        })
                    from_list = await asyncio.to_thread(lambda: Dataset.from_list(tmp) if Dataset is not None else None)
                    mapped = from_list if from_list is not None else tmp  # may be a list if datasets missing

                # Select sample
                try:
                    total_records = len(mapped)
                except Exception:
                    total_records = 0
                if hasattr(mapped, "select"):
                    k = min(sample_size, total_records)
                    idxs = list(range(total_records)) if k >= total_records else random.sample(range(total_records), k)
                    batch = await asyncio.to_thread(lambda: mapped.select(idxs))
                    examples = [{"input": (r.get("input", "") or ""), "output": (r.get("output", "") or "")} for r in batch]
                else:
                    # mapped is already a python list
                    total_records = len(mapped)
                    if total_records > sample_size:
                        idxs = random.sample(range(total_records), sample_size)
                        examples = [mapped[i] for i in idxs]
                    else:
                        examples = list(mapped)

            else:
                # JSON file
                if not jpath:
                    raise RuntimeError("Provide a JSON path")
                try:
                    records = await asyncio.to_thread(sd.load_records, jpath)
                except Exception as e:
                    raise RuntimeError(f"Failed to read JSON: {e}")
                try:
                    ex0 = await asyncio.to_thread(sd.normalize_records, records, 1)
                except Exception:
                    ex0 = []
                    for r in records or []:
                        if isinstance(r, dict):
                            a = str((r.get("input") or "")).strip()
                            b = str((r.get("output") or "")).strip()
                            if a and b:
                                ex0.append({"input": a, "output": b})
                total_records = len(ex0)
                if total_records > sample_size:
                    idxs = random.sample(range(total_records), sample_size)
                    examples = [ex0[i] for i in idxs]
                else:
                    examples = ex0

            used_n = len(examples)
            if used_n == 0:
                raise RuntimeError("No examples found to analyze")

            # Compute metrics (gated by module toggles where applicable)
            do_basic = bool(getattr(cb_basic_stats, "value", True))
            do_dupe = bool(getattr(cb_duplicates, "value", False))
            do_sent = bool(getattr(cb_sentiment, "value", True))
            do_cls = bool(getattr(cb_class_balance, "value", True))
            do_cov = bool(getattr(cb_coverage_overlap, "value", False))
            do_leak = bool(getattr(cb_data_leakage, "value", False))
            do_depth = bool(getattr(cb_conversation_depth, "value", False))
            do_speaker = bool(getattr(cb_speaker_balance, "value", False))
            do_qstmt = bool(getattr(cb_question_statement, "value", False))
            do_read = bool(getattr(cb_readability, "value", False))
            do_ner = bool(getattr(cb_ner, "value", False))
            do_toxic = bool(getattr(cb_toxicity, "value", False))
            do_polite = bool(getattr(cb_politeness, "value", False))
            do_dacts = bool(getattr(cb_dialogue_acts, "value", False))
            do_topics = bool(getattr(cb_topics, "value", False))
            do_align = bool(getattr(cb_alignment, "value", False))

            in_lens = [len(str(x.get("input", ""))) for x in examples]
            out_lens = [len(str(x.get("output", ""))) for x in examples]

            avg_in = avg_out = 0.0
            if do_basic:
                avg_in = sum(in_lens) / max(1, used_n)
                avg_out = sum(out_lens) / max(1, used_n)

            dup_pct = None
            if do_dupe:
                unique_pairs = len({(str(x.get("input", "")), str(x.get("output", ""))) for x in examples})
                dup_pct = 100.0 * (1.0 - (unique_pairs / max(1, used_n)))

            # Sentiment proxy via tiny lexicon (gated)
            POS = {"good", "great", "love", "awesome", "nice", "excellent", "happy", "lol", "thanks", "cool"}
            NEG = {"bad", "hate", "terrible", "awful", "angry", "sad", "stupid", "dumb", "wtf", "idiot", "trash"}
            pos = neu = neg = 0
            if do_sent:
                for ex in examples:
                    txt = f"{ex.get('input','')} {ex.get('output','')}".lower()
                    score = sum(1 for w in POS if w in txt) - sum(1 for w in NEG if w in txt)
                    if score > 0:
                        pos += 1
                    elif score < 0:
                        neg += 1
                    else:
                        neu += 1
                pos_p = pos / used_n
                neu_p = neu / used_n
                neg_p = neg / used_n
            else:
                pos_p = neu_p = neg_p = 0.0

            # Length buckets (Short/Medium/Long) for input (gated)
            if do_cls:
                short = sum(1 for L in in_lens if L <= 128)
                medium = sum(1 for L in in_lens if 129 <= L <= 512)
                long = used_n - short - medium
                a_p = short / used_n
                b_p = medium / used_n
                c_p = long / used_n
            else:
                a_p = b_p = c_p = 0.0

            # Update UI controls
            kpi_total_value.value = f"{used_n:,}" if do_basic else "—"
            kpi_avg_in_value.value = (f"{avg_in:.0f} chars" if do_basic else "—")
            kpi_avg_out_value.value = (f"{avg_out:.0f} chars" if do_basic else "—")
            kpi_dupe_value.value = (f"{dup_pct:.1f}%" if (do_dupe and dup_pct is not None) else "—")

            # Sentiment section
            sentiment_section.visible = do_sent
            sent_pos_bar.value = pos_p
            sent_pos_pct.value = f"{int(pos_p * 100)}%"
            sent_neu_bar.value = neu_p
            sent_neu_pct.value = f"{int(neu_p * 100)}%"
            sent_neg_bar.value = neg_p
            sent_neg_pct.value = f"{int(neg_p * 100)}%"

            # Class balance section
            class_balance_section.visible = do_cls
            class_a_label.value = "Short"
            class_a_bar.value = a_p
            class_a_pct.value = f"{int(a_p * 100)}%"
            class_b_label.value = "Medium"
            class_b_bar.value = b_p
            class_b_pct.value = f"{int(b_p * 100)}%"
            class_c_label.value = "Long"
            class_c_bar.value = c_p
            class_c_pct.value = f"{int(c_p * 100)}%"

            # Compute Extra metrics based on selected modules
            extra_rows: list[ft.DataRow] = []

            def _tokens(s: str) -> list[str]:
                return re.findall(r"[A-Za-z0-9']+", s.lower())

            def _token_set(s: str) -> set[str]:
                return set(_tokens(s))

            def _jaccard(a: set[str], b: set[str]) -> float:
                if not a and not b:
                    return 1.0
                inter = len(a & b)
                union = len(a | b)
                return inter / union if union else 0.0

            if any([do_cov, do_leak, do_depth, do_speaker, do_qstmt, do_read, do_ner, do_toxic, do_polite, do_dacts, do_topics, do_align]):
                # Precompute tokens
                in_tokens = [_token_set(str(ex.get("input", ""))) for ex in examples]
                out_tokens = [_token_set(str(ex.get("output", ""))) for ex in examples]

                if do_cov:
                    cover_vals = []
                    for ti, to in zip(in_tokens, out_tokens):
                        cover = (len(ti & to) / max(1, len(to))) if to else 0.0
                        cover_vals.append(cover)
                    cover_avg = sum(cover_vals) / len(cover_vals)
                    extra_rows.append(ft.DataRow(cells=[
                        ft.DataCell(ft.Text("Coverage overlap")),
                        ft.DataCell(ft.Text(f"{cover_avg*100:.1f}%")),
                    ]))

                if do_align:
                    jac_vals = []
                    for ti, to in zip(in_tokens, out_tokens):
                        jac_vals.append(_jaccard(ti, to))
                    jac_avg = sum(jac_vals) / len(jac_vals)
                    extra_rows.append(ft.DataRow(cells=[
                        ft.DataCell(ft.Text("Alignment (Jaccard)")),
                        ft.DataCell(ft.Text(f"{jac_avg*100:.1f}%")),
                    ]))

                if do_leak:
                    leak = 0
                    for ex in examples:
                        a = str(ex.get("input", "")).lower()
                        b = str(ex.get("output", "")).lower()
                        if (a and b) and (a in b or b in a):
                            leak += 1
                    leak_p = leak / used_n
                    extra_rows.append(ft.DataRow(cells=[
                        ft.DataCell(ft.Text("Data leakage risk")),
                        ft.DataCell(ft.Text(f"{leak_p*100:.1f}%")),
                    ]))

                if do_depth:
                    def _turns(text: str) -> int:
                        tl = text.lower()
                        m = len(re.findall(r"\b(user|assistant|system)\s*:", tl))
                        if m:
                            return m
                        lines = [ln for ln in text.splitlines() if ln.strip()]
                        return max(1, len(lines))
                    turns = [max(_turns(str(ex.get("input",""))), 1) for ex in examples]
                    avg_turns = sum(turns) / len(turns)
                    extra_rows.append(ft.DataRow(cells=[
                        ft.DataCell(ft.Text("Avg turns (approx)")),
                        ft.DataCell(ft.Text(f"{avg_turns:.1f}")),
                    ]))

                if do_speaker:
                    shares = []
                    for ex in examples:
                        a = str(ex.get("input", ""))
                        b = str(ex.get("output", ""))
                        tot = len(a) + len(b)
                        shares.append((len(a) / tot) if tot else 0.0)
                    share_avg = sum(shares) / len(shares)
                    extra_rows.append(ft.DataRow(cells=[
                        ft.DataCell(ft.Text("Speaker balance (input share)")),
                        ft.DataCell(ft.Text(f"{share_avg*100:.1f}%")),
                    ]))

                if do_qstmt:
                    q = 0
                    for ex in examples:
                        a = str(ex.get("input", ""))
                        if a.strip().endswith("?"):
                            q += 1
                    q_p = q / used_n
                    extra_rows.append(ft.DataRow(cells=[
                        ft.DataCell(ft.Text("Questions (inputs)")),
                        ft.DataCell(ft.Text(f"{q_p*100:.1f}%")),
                    ]))

                if do_read:
                    vowels = set("aeiouy")
                    def _syllables(word: str) -> int:
                        w = word.lower()
                        groups = re.findall(r"[aeiouy]+", w)
                        return max(1, len(groups))
                    def _readability(text: str) -> float:
                        toks = _tokens(text)
                        words = max(1, len(toks))
                        sentences = max(1, len(re.findall(r"[.!?]", text)))
                        syll = sum(_syllables(t) for t in toks)
                        # Flesch Reading Ease (approx)
                        return 206.835 - 1.015*(words/sentences) - 84.6*(syll/words)
                    scores = [_readability(str(ex.get("input",""))) for ex in examples]
                    score_avg = sum(scores)/len(scores)
                    extra_rows.append(ft.DataRow(cells=[
                        ft.DataCell(ft.Text("Readability (Flesch approx)")),
                        ft.DataCell(ft.Text(f"{score_avg:.1f}")),
                    ]))

                if do_ner:
                    def _capwords(text: str) -> int:
                        # Count capitalized words not at sentence start (rough proxy)
                        toks = re.findall(r"\b[A-Z][a-z]+\b", text)
                        return len(toks)
                    ents = [_capwords(str(ex.get("input",""))) for ex in examples]
                    ents_avg = sum(ents)/len(ents)
                    extra_rows.append(ft.DataRow(cells=[
                        ft.DataCell(ft.Text("NER (capwords avg)")),
                        ft.DataCell(ft.Text(f"{ents_avg:.2f}")),
                    ]))

                if do_toxic:
                    tox = 0
                    for ex in examples:
                        txt = f"{ex.get('input','')} {ex.get('output','')}".lower()
                        if any(w in txt for w in NEG):
                            tox += 1
                    tox_p = tox / used_n
                    extra_rows.append(ft.DataRow(cells=[
                        ft.DataCell(ft.Text("Toxicity flagged")),
                        ft.DataCell(ft.Text(f"{tox_p*100:.1f}%")),
                    ]))

                if do_polite:
                    POLITE = {"please", "thank", "thanks", "kindly", "sir", "madam", "regards"}
                    pol = 0
                    for ex in examples:
                        txt = f"{ex.get('input','')} {ex.get('output','')}".lower()
                        if any(w in txt for w in POLITE):
                            pol += 1
                    pol_p = pol / used_n
                    extra_rows.append(ft.DataRow(cells=[
                        ft.DataCell(ft.Text("Politeness flagged")),
                        ft.DataCell(ft.Text(f"{pol_p*100:.1f}%")),
                    ]))

                if do_dacts:
                    q = c = s = 0
                    for ex in examples:
                        a = str(ex.get("input", "")).strip()
                        al = a.lower()
                        if a.endswith("?"):
                            q += 1
                        elif al.startswith(("please ", "do ", "go ", "make ", "provide ", "give ", "show ")):
                            c += 1
                        else:
                            s += 1
                    q_p = q/used_n
                    c_p = c/used_n
                    s_p = s/used_n
                    extra_rows.append(ft.DataRow(cells=[
                        ft.DataCell(ft.Text("Dialogue acts (Q/C/S)")),
                        ft.DataCell(ft.Text(f"{int(q_p*100)}/{int(c_p*100)}/{int(s_p*100)}%")),
                    ]))

                if do_topics:
                    STOP = {"the","a","an","and","or","to","is","are","was","were","of","for","in","on","at","it","this","that","i","you","he","she","they","we","with"}
                    freq = Counter()
                    for ex in examples:
                        freq.update([t for t in _tokens(str(ex.get("input",""))) if t not in STOP and len(t) > 2])
                    top = ", ".join([w for w,_ in freq.most_common(5)]) or "(none)"
                    extra_rows.append(ft.DataRow(cells=[
                        ft.DataCell(ft.Text("Top keywords")),
                        ft.DataCell(ft.Text(top)),
                    ]))

            extra_metrics_table.rows = extra_rows
            extra_metrics_section.visible = len(extra_rows) > 0

            # Reveal result blocks now that real values are computed
            kpi_total_tile.visible = do_basic
            kpi_avg_in_tile.visible = do_basic
            kpi_avg_out_tile.visible = do_basic
            kpi_dupe_tile.visible = do_dupe
            overview_block.visible = (do_basic or do_dupe)
            sentiment_block.visible = do_sent
            class_balance_block.visible = do_cls
            extra_metrics_block.visible = len(extra_rows) > 0
            samples_section.visible = True
            samples_block.visible = True
            # Toggle dividers to match block visibility
            div_overview.visible = overview_block.visible
            div_sentiment.visible = sentiment_block.visible
            div_class.visible = class_balance_block.visible
            div_extra.visible = extra_metrics_block.visible
            div_samples.visible = samples_block.visible

            # Samples grid (up to 10)
            try:
                show_n = min(10, used_n)
                rows = []
                for i in range(show_n):
                    ex = examples[i]
                    a = str(ex.get("input", ""))
                    b = str(ex.get("output", ""))
                    # Scrollable text cells with fixed width for neat column layout
                    a_cell = ft.Container(
                        width=SAMPLE_INPUT_W,
                        content=ft.Row([ft.Text(a, no_wrap=True, selectable=True)], scroll=ft.ScrollMode.AUTO),
                    )
                    b_cell = ft.Container(
                        width=SAMPLE_OUTPUT_W,
                        content=ft.Row([ft.Text(b, no_wrap=True, selectable=True)], scroll=ft.ScrollMode.AUTO),
                    )
                    inlen_cell = ft.Container(width=SAMPLE_LEN_W, content=ft.Text(str(len(a)), text_align=ft.TextAlign.END))
                    outlen_cell = ft.Container(width=SAMPLE_LEN_W, content=ft.Text(str(len(b)), text_align=ft.TextAlign.END))
                    rows.append(ft.DataRow(cells=[
                        ft.DataCell(a_cell),
                        ft.DataCell(b_cell),
                        ft.DataCell(inlen_cell),
                        ft.DataCell(outlen_cell),
                    ]))
                samples_grid.rows = rows
            except Exception:
                pass

            try:
                modules_used = []
                if do_basic:
                    modules_used.append("Basic Stats")
                if do_dupe:
                    modules_used.append("Duplicates")
                if do_sent:
                    modules_used.append("Sentiment")
                if do_cls:
                    modules_used.append("Class balance")
                if do_cov:
                    modules_used.append("Coverage Overlap")
                if do_leak:
                    modules_used.append("Data Leakage Check")
                if do_depth:
                    modules_used.append("Conversation Depth")
                if do_speaker:
                    modules_used.append("Speaker Balance")
                if do_qstmt:
                    modules_used.append("Question vs Statement")
                if do_read:
                    modules_used.append("Readability")
                if do_ner:
                    modules_used.append("NER")
                if do_toxic:
                    modules_used.append("Toxicity / Safety")
                if do_polite:
                    modules_used.append("Politeness / Formality")
                if do_dacts:
                    modules_used.append("Dialogue Acts")
                if do_topics:
                    modules_used.append("Topics / Clustering")
                if do_align:
                    modules_used.append("Alignment (Similarity/NLI)")
                mod_txt = " | Modules: " + ", ".join(modules_used) if modules_used else ""
                analysis_overview_note.value = (
                    f"Analyzed {used_n:,} records" + (f" (sampled from {total_records:,})" if total_records > used_n else "") + mod_txt
                )
            except Exception:
                pass

            await safe_update(page)
        except Exception as e:
            page.snack_bar = ft.SnackBar(ft.Text(f"Analysis failed: {e}"))
            page.snack_bar.open = True
            await safe_update(page)
        finally:
            analysis_busy_ring.visible = False
            analyze_btn.disabled = False
            analysis_state["running"] = False
            await safe_update(page)

    # Helper: build a table layout for module checkboxes (3 columns)
    def _build_modules_table():
        mods = _all_analysis_modules()
        columns = [ft.DataColumn(ft.Text("")), ft.DataColumn(ft.Text("")), ft.DataColumn(ft.Text(""))]
        rows: list[ft.DataRow] = []
        def _cell_with_help(ctrl):
            try:
                tip = getattr(ctrl, "tooltip", None)
            except Exception:
                tip = None
            # Try to add a small clickable info icon next to control
            try:
                _info_icon_name = getattr(
                    ICONS,
                    "INFO_OUTLINE",
                    getattr(ICONS, "INFO", getattr(ICONS, "HELP_OUTLINE", getattr(ICONS, "HELP", None))),
                )
                def _on_help_click(e, text=tip):
                    try:
                        dlg = ft.AlertDialog(title=ft.Text("About module"), content=ft.Text(text or ""))
                        page.dialog = dlg
                        dlg.open = True
                        page.update()
                    except Exception:
                        try:
                            page.snack_bar = ft.SnackBar(ft.Text(text or ""))
                            page.snack_bar.open = True
                            page.update()
                        except Exception:
                            pass
                help_btn = None
                try:
                    help_btn = ft.IconButton(icon=_info_icon_name, icon_color=WITH_OPACITY(0.6, BORDER_BASE), tooltip=tip or "Module help", on_click=_on_help_click)
                except Exception:
                    try:
                        help_btn = ft.Icon(_info_icon_name, size=16, color=WITH_OPACITY(0.6, BORDER_BASE))
                        help_btn = ft.Tooltip(message=tip or "Module help", content=help_btn)
                    except Exception:
                        help_btn = None
                if help_btn is not None:
                    return ft.Row([ctrl, help_btn], spacing=4, alignment=ft.MainAxisAlignment.START)
            except Exception:
                pass
            # Fallback: return control as-is
            return ctrl
        for i in range(0, len(mods), 3):
            c1 = ft.DataCell(_cell_with_help(mods[i]))
            c2 = ft.DataCell(_cell_with_help(mods[i + 1])) if i + 1 < len(mods) else ft.DataCell(ft.Container())
            c3 = ft.DataCell(_cell_with_help(mods[i + 2])) if i + 2 < len(mods) else ft.DataCell(ft.Container())
            rows.append(ft.DataRow(cells=[c1, c2, c3]))
        return ft.DataTable(columns=columns, rows=rows)

    # Blocks for results sections: hidden until real results are computed
    kpi_total_tile = kpi_tile("Total records", kpi_total_value, icon=getattr(ICONS, "TABLE_VIEW", getattr(ICONS, "LIST", ICONS.SEARCH)))
    kpi_avg_in_tile = kpi_tile("Avg input length", kpi_avg_in_value, icon=getattr(ICONS, "TEXT_FIELDS", getattr(ICONS, "TEXT_FIELDS_OUTLINED", ICONS.SEARCH)))
    kpi_avg_out_tile = kpi_tile("Avg output length", kpi_avg_out_value, icon=getattr(ICONS, "TEXT_FIELDS", getattr(ICONS, "TEXT_FIELDS_OUTLINED", ICONS.SEARCH)))
    kpi_dupe_tile = kpi_tile("Duplicates", kpi_dupe_value, icon=getattr(ICONS, "CONTENT_COPY", getattr(ICONS, "COPY_ALL", ICONS.SEARCH)))
    overview_row = ft.Row([
        kpi_total_tile,
        kpi_avg_in_tile,
        kpi_avg_out_tile,
        kpi_dupe_tile,
    ], wrap=True, spacing=12)
    overview_block = ft.Container(
        content=ft.Column([
            section_title(
                "Overview",
                getattr(ICONS, "DASHBOARD", getattr(ICONS, "INSIGHTS", ICONS.SEARCH)),
                "Key KPIs such as total records, average input/output lengths, and duplicate rate (if enabled).",
                on_help_click=_mk_help_handler("Key KPIs such as total records, average input/output lengths, and duplicate rate (if enabled)."),
            ),
            overview_row,
        ], spacing=6),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
        border_radius=8,
        padding=10,
        visible=False,
    )

    sentiment_block = ft.Container(
        content=ft.Column([
            section_title(
                "Sentiment",
                getattr(ICONS, "EMOJI_EMOTIONS", getattr(ICONS, "INSERT_EMOTICON", ICONS.SEARCH)),
                "Heuristic sentiment distribution computed over sampled records.",
                on_help_click=_mk_help_handler("Heuristic sentiment distribution computed over sampled records."),
            ),
            sentiment_section,
        ], spacing=6),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
        border_radius=8,
        padding=10,
        visible=False,
    )

    class_balance_block = ft.Container(
        content=ft.Column([
            section_title(
                "Class balance",
                getattr(ICONS, "DONUT_SMALL", getattr(ICONS, "PIE_CHART", ICONS.SEARCH)),
                "Distribution of labels/classes if present in your dataset.",
                on_help_click=_mk_help_handler("Distribution of labels/classes if present in your dataset."),
            ),
            class_balance_section,
        ], spacing=6),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
        border_radius=8,
        padding=10,
        visible=False,
    )

    extra_metrics_block = ft.Container(
        content=ft.Column([
            section_title(
                "Extra metrics",
                getattr(ICONS, "INSIGHTS", getattr(ICONS, "ANALYTICS", ICONS.SEARCH)),
                "Lightweight proxies: coverage overlap, leakage check, depth, speaker balance, Q vs statement, readability, NER proxy, toxicity, politeness, dialogue acts, topics, alignment.",
                on_help_click=_mk_help_handler("Lightweight proxies: coverage overlap, leakage check, depth, speaker balance, Q vs statement, readability, NER proxy, toxicity, politeness, dialogue acts, topics, alignment."),
            ),
            extra_metrics_section,
        ], spacing=6),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
        border_radius=8,
        padding=10,
        visible=False,
    )

    samples_block = ft.Container(
        content=ft.Column([
            section_title(
                "Samples",
                getattr(ICONS, "LIST", getattr(ICONS, "LIST_ALT", ICONS.SEARCH)),
                "Random sample rows for quick spot checks (input/output and lengths).",
                on_help_click=_mk_help_handler("Random sample rows for quick spot checks (input/output and lengths)."),
            ),
            samples_section,
        ], spacing=6),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
        border_radius=8,
        padding=10,
        visible=False,
    )

    # Named dividers for each results block (hidden until analysis produces output)
    div_overview = ft.Divider(visible=False)
    div_sentiment = ft.Divider(visible=False)
    div_class = ft.Divider(visible=False)
    div_extra = ft.Divider(visible=False)
    div_samples = ft.Divider(visible=False)

    analysis_tab = build_analysis_tab(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        analyze_btn=analyze_btn,
        analysis_busy_ring=analysis_busy_ring,
        analysis_source_dd=analysis_source_dd,
        analysis_hf_repo=analysis_hf_repo,
        analysis_hf_split=analysis_hf_split,
        analysis_hf_config=analysis_hf_config,
        analysis_json_path=analysis_json_path,
        analysis_dataset_hint=analysis_dataset_hint,
        select_all_modules_cb=select_all_modules_cb,
        _build_modules_table=_build_modules_table,
        analysis_backend_dd=analysis_backend_dd,
        analysis_hf_token_tf=analysis_hf_token_tf,
        analysis_sample_size_tf=analysis_sample_size_tf,
        analysis_overview_note=analysis_overview_note,
        div_overview=div_overview,
        overview_block=overview_block,
        div_sentiment=div_sentiment,
        sentiment_block=sentiment_block,
        div_class=div_class,
        class_balance_block=class_balance_block,
        div_extra=div_extra,
        extra_metrics_block=extra_metrics_block,
        div_samples=div_samples,
        samples_block=samples_block,
    )
    
    # Tabs and welcome screen
    tabs = ft.Tabs(
        tabs=[
            ft.Tab(text="Scrape", icon=ICONS.SEARCH, content=scrape_tab),
            ft.Tab(text="Build / Publish", icon=ICONS.BUILD_CIRCLE_OUTLINED, content=build_tab),
            ft.Tab(text="Dataset Analysis", icon=getattr(ICONS, "INSIGHTS", ICONS.ANALYTICS), content=analysis_tab),
            ft.Tab(text="Merge Datasets", icon=getattr(ICONS, "MERGE_TYPE", ICONS.TABLE_VIEW), content=merge_tab),
            ft.Tab(text="Training", icon=getattr(ICONS, "SCIENCE", ICONS.PLAY_CIRCLE), content=training_tab),
            ft.Tab(text="Settings", icon=ICONS.SETTINGS, content=settings_tab),
        ],
        expand=1,
    )

    def show_main_app(_=None):
        try:
            page.controls.clear()
            page.add(tabs)
            try:
                page.floating_action_button = ft.FloatingActionButton(
                    icon=(INFO_ICON or getattr(ICONS, "INFO", None)),
                    tooltip="User guide (F1)",
                    on_click=lambda e: open_user_guide(e),
                )
            except Exception:
                pass
            try:
                page.update()
            except Exception:
                pass
            # Initialize visibility and dependent UI now that tabs exist
            update_source_controls()
            try:
                _update_train_source()
            except Exception:
                pass
            try:
                _update_skill_controls()
            except Exception:
                pass
            try:
                _update_analysis_source()
            except Exception:
                pass
            try:
                _update_analysis_backend()
            except Exception:
                pass
            try:
                _sync_select_all_modules()
            except Exception:
                pass
        except Exception:
            pass

    welcome_logo = ft.Image(src="img/FineForge-logo.png", width=240, height=240, fit=ft.ImageFit.CONTAIN)
    welcome_tagline = ft.Text("Curate, analyze, and fine‑tune datasets with a beautiful desktop UI.", color=WITH_OPACITY(0.8, BORDER_BASE))
    start_btn = ft.FilledButton("Start", icon=getattr(ICONS, "PLAY_ARROW", getattr(ICONS, "PLAY_CIRCLE", None)), on_click=show_main_app)

    welcome_view = ft.Container(
        expand=1,
        content=ft.Column([
            welcome_logo,
            ft.Container(height=8),
            welcome_tagline,
            ft.Container(height=16),
            start_btn,
        ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6),
        alignment=ft.alignment.center,
        padding=20,
    )

    try:
        page.controls.clear()
        page.add(welcome_view)
        page.floating_action_button = None
        page.update()
    except Exception:
        pass

if __name__ == "__main__":
    ft.app(target=main)
