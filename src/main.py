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
from helpers.datasets import guess_input_output_columns
from ui.tabs.tab_settings import build_settings_tab
from ui.tabs.tab_scrape import build_scrape_tab
from ui.tabs.tab_build import build_build_tab
from ui.tabs.tab_merge import build_merge_tab
from ui.tabs.tab_analysis import build_analysis_tab
from ui.tabs.inference_controller import build_inference_tab_with_logic
from ui.tabs.training_controller import build_training_tab_with_logic
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
        leading=ft.IconButton(
            icon=getattr(ICONS, "DATASET_LINKED_OUTLINED", getattr(ICONS, "DATASET", getattr(ICONS, "DESCRIPTION", None))),
            tooltip="Open FineFoundry website",
            on_click=lambda e: page.launch_url("https://finefoundry.fly.dev/"),
        ),
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

    # Build Training tab via the dedicated controller (refactored wiring)
    training_tab, train_state = build_training_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
        _hf_cfg=_hf_cfg,
        _runpod_cfg=_runpod_cfg,
        hf_token_tf=hf_token_tf,
        proxy_enable_cb=proxy_enable_cb,
        use_env_cb=use_env_cb,
        proxy_url_tf=proxy_url_tf,
    )
    # ---------- INFERENCE TAB (global inference over fine-tuned models) ----------

    inference_tab = build_inference_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
        train_state=train_state,
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
            ft.Tab(text="Inference", icon=getattr(ICONS, "PSYCHOLOGY", getattr(ICONS, "CHAT", ICONS.PLAY_CIRCLE)), content=inference_tab),
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
