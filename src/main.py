import asyncio
import os
import sys
import time
import shutil
from typing import Optional
import httpx


import flet as ft

# Runpod infra helper (pods are managed via controllers; main only needs optional infra module)
try:
    from runpod import ensure_infra as rp_infra
except Exception:
    import sys as __sys2

    __sys2.path.append(os.path.dirname(__file__))
    try:
        from runpod import ensure_infra as rp_infra
    except Exception:
        rp_infra = None

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
from helpers.ui import (
    WITH_OPACITY,
    section_title,
)
from ui.tabs.tab_settings import build_settings_tab
from ui.tabs.scrape_controller import build_scrape_tab_with_logic
from ui.tabs.build_controller import build_build_tab_with_logic
from ui.tabs.merge_controller import build_merge_tab_with_logic
from ui.tabs.analysis_controller import build_analysis_tab_with_logic
from ui.tabs.inference_controller import build_inference_tab_with_logic
from ui.tabs.training_controller import build_training_tab_with_logic
from helpers.settings_ollama import (
    load_config as load_ollama_config_helper,
    save_config as save_ollama_config_helper,
    fetch_tags as fetch_ollama_tags_helper,
)
from helpers.settings import (
    load_hf_config as load_hf_config_from_db,
    save_hf_config as save_hf_config_to_db,
    load_runpod_config as load_runpod_config_from_db,
    save_runpod_config as save_runpod_config_to_db,
    load_offline_mode as load_offline_mode_from_db,
    save_offline_mode as save_offline_mode_to_db,
)
from db import (
    init_db,
    get_db_path,
    get_connection,
    get_log_count,
    clear_logs,
    delete_training_run,
    get_managed_storage_root,
    close_all_connections,
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
            "Collect data from sources (4chan, Reddit, StackExchange, or generate synthetic data from documents), build/merge datasets, train locally or on Runpod, and analyze data quality.\n"
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
    # Helper: robust Markdown factory for the in-app guide
    def _make_md(md_text: str) -> ft.Markdown:
        return ft.Markdown(md_text, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB)

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
FineFoundry is a desktop studio to collect data, merge, analyze, build/publish, train, and run inference on fine-tuned adapters.

Tabs:
- Scrape
- Build / Publish
- Training
- Inference
- Merge Datasets
- Dataset Analysis
- Settings

## Quick Navigation
- Top‑right icons: Refresh • Theme toggle
- User guide: press F1 or click the bottom‑right Help FAB.
- Use the tabs to switch workflows.

## Scrape
- Choose source: 4chan, Reddit, StackExchange, or **Synthetic** (generate from documents).
- For scraping: configure parameters (max pairs, threads, delays, length filters).
- For synthetic: select files (PDF, DOCX, PPTX, HTML, TXT) or URLs to generate QA pairs, Chain of Thought, or summaries.
- Preview results in a two‑column grid (input/output).
- All scraped/generated data is saved to the SQLite database as a new session.
- Proxy support per scraper; system env proxies optional.

## Build / Publish
- Create train/val/test splits with Hugging Face `datasets`.
- Save locally and optionally push to the Hugging Face Hub.
- Provide `HF_TOKEN` in Settings or via environment/CLI login.

Build / Publish in the GUI is database-first:
- Select a database scrape session as the source.
- Optionally enable Hub push (disabled in Offline Mode).

## Merge Datasets
- Combine database sessions (and optionally Hugging Face datasets when online).
- Auto‑map `input`/`output` columns; filter empty rows.
- Merged data is saved to the database as a new session.

## Training
- Training target: choose **Runpod - Pod** (remote GPU pod) or **local** (runs directly on this machine).
- Hyperparameters: base model, epochs, learning rate, batch size, grad accumulation, max steps, packing, resume.
- Outputs & checkpoints: saved under **Output dir** inside the selected Training run's managed storage.
- Hugging Face auth: save a token in **Settings → Hugging Face** or export `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`. Required for pushing models or using private datasets.

Offline Mode notes:
- Runpod training is disabled (local only).
- Hub push is disabled.

### Local Training
- Runs the integrated Unsloth trainer directly via a local Python subprocess.
- Select or create a **Training run** for managed storage.
- Configure GPU usage and whether to **Pass HF token to trainer** (for private datasets).
- Click **Start Local Training** to run training on your machine.
- After a successful local run, a **Quick Local Inference** panel appears so you can test the trained adapter immediately.

### Quick Local Inference
- Loads the base model and LoRA adapter from your last successful local training run.
- Controls: prompt box, **Generate** button, temperature slider, max tokens slider, presets (Deterministic / Balanced / Creative), **Clear history**, and inline model info.
- Useful for quick sanity checks of a new run without leaving the app.

## Inference
- Select a **completed training run** to load its adapter for inference.
- Use **Use latest completed run** for a one-click shortcut.
- Sample prompts can be pulled from any saved dataset session in the database.

### Configuration (save / load setups)
- Mode:
  - **Normal**: edit dataset + hyperparameters directly.
  - **Configuration**: pick a saved config and run with minimal inputs.
- Use **Save current setup** (in the Configuration section or near the training controls) to snapshot:
  - Dataset + hyperparameters
  - Training target (Runpod or local)
  - Runpod infrastructure or local settings
- Saved configs are stored in the database. The last used config auto‑loads on startup so you can continue where you left off.

## Dataset Analysis
- Select dataset source (Database or Hugging Face when online) and click **Analyze dataset**.
- Use **Select all** to toggle modules. Only enabled modules are computed and shown.
- Modules: Basic Stats • Duplicates • Sentiment • Class balance (length) •
  Extra proxies (Coverage overlap, Data leakage, Conversation depth, Speaker balance,
  Question vs Statement, Readability, NER proxy, Toxicity, Politeness, Dialogue Acts, Topics, Alignment).
- Summary lists active modules after each run; sample rows are shown for quick checks.

Offline Mode notes:
- Hugging Face dataset sources and HF Inference API are disabled.

## Settings
- Hugging Face: save access token.
- Proxies: per‑scraper defaults and/or system env proxies.
- Runpod: save API key; standard mount path is `/data`.
- Ollama: enable connection, set base URL, list/select models; used for dataset cards.

Offline Mode is a global switch:
- Scrape becomes synthetic-only.
- Hugging Face datasets / Hub pushes / HF Inference API are disabled.
- Runpod training is disabled.

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
                    return ft.IconButton(
                        icon=icon_const, tooltip=tooltip, on_click=lambda e: page.run_task(on_click_cb)
                    )
                return ft.IconButton(icon=icon_const, tooltip=tooltip, on_click=lambda e: on_click_cb(e))
        except Exception:
            pass
        if getattr(on_click_cb, "__name__", "").endswith("_async"):
            return ft.TextButton(text_fallback or tooltip, on_click=lambda e: page.run_task(on_click_cb))
        return ft.TextButton(text_fallback or tooltip, on_click=lambda e: on_click_cb(e))

    # Global online/offline status indicator shown next to app title
    offline_status_dot = ft.Container(
        width=10,
        height=10,
        border_radius=50,
        bgcolor=COLORS.GREEN,
    )
    offline_status_label = ft.Text(
        "Online",
        size=12,
        color=WITH_OPACITY(0.8, BORDER_BASE),
    )
    offline_status_row = ft.Row(
        [offline_status_dot, offline_status_label],
        spacing=4,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    page.appbar = ft.AppBar(
        leading=ft.IconButton(
            icon=getattr(
                ICONS, "DATASET_LINKED_OUTLINED", getattr(ICONS, "DATASET", getattr(ICONS, "DESCRIPTION", None))
            ),
            tooltip="Open FineFoundry website",
            on_click=lambda e: page.launch_url("https://finefoundry.fly.dev/"),
        ),
        title=ft.Row(
            [
                ft.Text(APP_TITLE, weight=ft.FontWeight.BOLD),
                offline_status_row,
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        center_title=False,
        bgcolor=WITH_OPACITY(0.03, COLORS.AMBER),
        actions=[
            _appbar_action(
                REFRESH_ICON or getattr(ICONS, "SYNC", getattr(ICONS, "CACHED", None)),
                "Refresh app",
                refresh_app,
                text_fallback="Refresh",
            ),
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
    # Built via ui.tabs.scrape_controller.build_scrape_tab_with_logic (controller-based wiring)

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

    # Initialize database
    try:
        init_db()
    except Exception as e:
        logger.warning(f"Database initialization warning: {e}")

    # ---------- SETTINGS (Offline Mode) CONTROLS ----------
    try:
        _offline_mode_enabled = bool(load_offline_mode_from_db())
    except Exception:
        _offline_mode_enabled = True

    offline_mode_sw = ft.Switch(
        label="Offline mode (synthetic-only data sources, no remote scrapers)",
        value=_offline_mode_enabled,
    )

    def on_offline_mode_change(e):
        try:
            save_offline_mode_to_db(bool(offline_mode_sw.value))
        except Exception:
            pass
        try:
            hooks = getattr(offline_mode_sw, "data", None)
            if isinstance(hooks, dict):
                for fn in hooks.values():
                    try:
                        if callable(fn):
                            fn(e)
                    except Exception:
                        pass
        except Exception:
            pass
        # Refresh global online/offline status indicator in the AppBar
        try:
            update_tab_offline_badges()
        except Exception:
            pass
        try:
            page.update()
        except Exception:
            pass

    offline_mode_sw.on_change = on_offline_mode_change

    # ---------- SETTINGS (Hugging Face) CONTROLS ----------
    # Now using SQLite database via helpers.settings

    def _load_hf_config() -> dict:
        try:
            return load_hf_config_from_db()
        except Exception:
            return {"token": ""}

    def _save_hf_config(cfg: dict):
        try:
            save_hf_config_to_db(cfg)
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
    _hf_has_token = bool((_hf_cfg.get("token") or "").strip())

    hf_token_tf = ft.TextField(
        label="Hugging Face API token",
        password=True,
        can_reveal_password=True,
        width=420,
    )
    # If a token is already saved, lock and mask the field so the raw value is never shown
    if _hf_has_token:
        hf_token_tf.value = "••••••••••••"
        hf_token_tf.password = True
        hf_token_tf.can_reveal_password = False
        hf_token_tf.read_only = True
        hf_token_tf.hint_text = "Token saved (hidden). Click Remove to clear or paste a new one."

    hf_status = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))

    async def on_test_hf():
        if bool(getattr(offline_mode_sw, "value", False)):
            hf_status.value = "Offline mode is enabled; network token checks are disabled."
            await safe_update(page)
            return

        raw = (hf_token_tf.value or "").strip()
        cfg_tok = (_hf_cfg.get("token") or "").strip()
        tok = cfg_tok if hf_token_tf.read_only else (raw or cfg_tok)
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
        # After saving, lock and mask the field so the token is not editable or revealable
        hf_token_tf.value = "••••••••••••"
        hf_token_tf.password = True
        hf_token_tf.can_reveal_password = False
        hf_token_tf.read_only = True
        hf_token_tf.hint_text = "Token saved (hidden). Click Remove to clear or paste a new one."
        try:
            hf_save_btn.disabled = True
        except Exception:
            pass
        hf_status.value = "Saved"
        page.update()

    def on_remove_hf(_):
        _hf_cfg["token"] = ""
        _save_hf_config(_hf_cfg)
        _apply_hf_env_from_cfg(_hf_cfg)
        hf_token_tf.value = ""
        # Unlock the field so a new token can be entered
        hf_token_tf.read_only = False
        hf_token_tf.can_reveal_password = True
        hf_token_tf.hint_text = ""
        try:
            hf_save_btn.disabled = False
        except Exception:
            pass
        hf_status.value = "Removed"
        page.update()

    hf_test_btn = ft.ElevatedButton("Test token", icon=ICONS.CHECK_CIRCLE, on_click=lambda e: page.run_task(on_test_hf))
    hf_save_btn = ft.OutlinedButton("Save", icon=ICONS.SAVE, on_click=on_save_hf)
    if _hf_has_token:
        hf_save_btn.disabled = True
    hf_remove_btn = ft.TextButton("Remove", icon=getattr(ICONS, "DELETE", ICONS.CANCEL), on_click=on_remove_hf)

    # ---------- SETTINGS (Runpod) CONTROLS ----------
    # Now using SQLite database via helpers.settings

    def _load_runpod_config() -> dict:
        try:
            return load_runpod_config_from_db()
        except Exception:
            return {"api_key": ""}

    def _save_runpod_config(cfg: dict):
        try:
            save_runpod_config_to_db(cfg)
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
    _runpod_has_key = bool((_runpod_cfg.get("api_key") or "").strip())

    runpod_key_tf = ft.TextField(
        label="Runpod API key",
        password=True,
        can_reveal_password=True,
        width=420,
    )
    if _runpod_has_key:
        runpod_key_tf.value = "••••••••••••"
        runpod_key_tf.password = True
        runpod_key_tf.can_reveal_password = False
        runpod_key_tf.read_only = True
        runpod_key_tf.hint_text = "Key saved (hidden). Click Remove to clear or paste a new one."

    runpod_status = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))

    async def on_test_runpod():
        if bool(getattr(offline_mode_sw, "value", False)):
            runpod_status.value = "Offline mode is enabled; Runpod endpoint checks are disabled."
            await safe_update(page)
            return

        raw = (runpod_key_tf.value or "").strip()
        cfg_key = (_runpod_cfg.get("api_key") or "").strip()
        key = cfg_key if runpod_key_tf.read_only else (raw or cfg_key)
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
        runpod_key_tf.value = "••••••••••••"
        runpod_key_tf.password = True
        runpod_key_tf.can_reveal_password = False
        runpod_key_tf.read_only = True
        runpod_key_tf.hint_text = "Key saved (hidden). Click Remove to clear or paste a new one."
        try:
            runpod_save_btn.disabled = True
        except Exception:
            pass
        runpod_status.value = "Saved"
        page.update()

    def on_remove_runpod(_):
        _runpod_cfg["api_key"] = ""
        _save_runpod_config(_runpod_cfg)
        _apply_runpod_env_from_cfg(_runpod_cfg)
        runpod_key_tf.value = ""
        runpod_key_tf.read_only = False
        runpod_key_tf.can_reveal_password = True
        runpod_key_tf.hint_text = ""
        try:
            runpod_save_btn.disabled = False
        except Exception:
            pass
        runpod_status.value = "Removed"
        page.update()

    runpod_test_btn = ft.ElevatedButton(
        "Test key", icon=ICONS.CHECK_CIRCLE, on_click=lambda e: page.run_task(on_test_runpod)
    )
    runpod_save_btn = ft.OutlinedButton("Save", icon=ICONS.SAVE, on_click=on_save_runpod)
    if _runpod_has_key:
        runpod_save_btn.disabled = True
    runpod_remove_btn = ft.TextButton("Remove", icon=getattr(ICONS, "DELETE", ICONS.CANCEL), on_click=on_remove_runpod)

    # ---------- SETTINGS (Ollama) CONTROLS ----------
    # Load persisted Ollama settings (with defaults applied for UI bindings)
    _ollama_raw = load_ollama_config_helper() or {}
    # Detect whether we appear to have an existing Ollama configuration
    _ollama_has_cfg = any(
        bool((str(_ollama_raw.get(k) or "").strip())) for k in ("base_url", "default_model", "selected_model")
    )
    # If "enabled" was never stored but config exists (pre-refactor installs), treat it as enabled by default
    if "enabled" in _ollama_raw:
        _ollama_enabled = bool(_ollama_raw.get("enabled", False))
    else:
        _ollama_enabled = _ollama_has_cfg

    _ollama_cfg = {
        "enabled": _ollama_enabled,
        "base_url": (_ollama_raw.get("base_url") or "http://127.0.0.1:11434"),
        "default_model": (_ollama_raw.get("default_model") or ""),
        "selected_model": (_ollama_raw.get("selected_model") or ""),
    }

    ollama_enable_cb = ft.Checkbox(label="Enable Ollama connection", value=_ollama_cfg.get("enabled", False))
    ollama_base_url_tf = ft.TextField(
        label="Ollama base URL", value=_ollama_cfg.get("base_url", "http://127.0.0.1:11434"), width=420
    )
    ollama_models_dd = ft.Dropdown(
        label="Available models", options=[], value=_ollama_cfg.get("selected_model") or None, width=420, disabled=True
    )
    ollama_status = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))
    if _ollama_has_cfg:
        if _ollama_enabled:
            ollama_status.value = (
                f"Saved Ollama settings found for {_ollama_cfg['base_url']}. Click 'Test connection' to verify."
            )
        else:
            ollama_status.value = "Saved Ollama settings found (connection disabled). Enable to use."

    def update_ollama_controls(_=None):
        en = bool(ollama_enable_cb.value)
        # Base URL: disabled when not enabled; read-only when we already have a saved config and it's enabled
        ollama_base_url_tf.disabled = not en
        try:
            if _ollama_has_cfg and en:
                ollama_base_url_tf.read_only = True
            else:
                ollama_base_url_tf.read_only = False
        except Exception:
            pass
        # Other controls follow the enable checkbox
        for c in [ollama_models_dd]:
            c.disabled = not en
        page.update()

    def mark_ollama_dirty(_=None):
        """Enable the Ollama Save button when the user changes settings (e.g., model/default/enabled)."""
        try:
            ollama_save_btn.disabled = False
        except Exception:
            pass
        try:
            page.update()
        except Exception:
            pass

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
        nonlocal _ollama_has_cfg, _ollama_enabled
        selected_model = (ollama_models_dd.value or "").strip()
        cfg = {
            "enabled": bool(ollama_enable_cb.value),
            "base_url": (ollama_base_url_tf.value or "http://127.0.0.1:11434").strip(),
            # Keep default_model for backward compatibility, but mirror the selected dropdown model
            "default_model": selected_model,
            "selected_model": selected_model,
        }
        save_ollama_config_helper(cfg)
        # Refresh local flags so base URL locking and status reflect the newly saved config
        _ollama_has_cfg = any(bool((cfg.get(k) or "").strip()) for k in ("base_url", "default_model", "selected_model"))
        _ollama_enabled = bool(cfg.get("enabled", False))
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Ollama settings saved"))
            page.open(page.snack_bar)
        except Exception:
            pass
        # After saving, treat config as clean again until further user changes
        try:
            ollama_save_btn.disabled = True
        except Exception:
            pass
        update_ollama_controls()

    def on_reset_ollama(_):
        nonlocal _ollama_has_cfg, _ollama_enabled
        _ollama_has_cfg = False
        _ollama_enabled = False
        try:
            save_ollama_config_helper({})
        except Exception:
            pass
        # Reset controls to defaults and unlock base URL
        ollama_enable_cb.value = False
        ollama_base_url_tf.value = "http://127.0.0.1:11434"
        ollama_base_url_tf.read_only = False
        ollama_models_dd.options = []
        ollama_models_dd.value = None
        ollama_models_dd.disabled = True
        ollama_status.value = (
            "Ollama settings reset. Enable connection, set base URL, and click Save to configure a new server."
        )
        try:
            ollama_save_btn.disabled = False
        except Exception:
            pass
        update_ollama_controls()

    ollama_test_btn = ft.ElevatedButton(
        "Test connection", icon=ICONS.CHECK_CIRCLE, on_click=lambda e: page.run_task(on_test_ollama)
    )
    ollama_refresh_btn = ft.TextButton(
        "Refresh models", icon=REFRESH_ICON, on_click=lambda e: page.run_task(on_refresh_models)
    )
    ollama_save_btn = ft.OutlinedButton("Save", icon=ICONS.SAVE, on_click=on_save_ollama)
    if _ollama_has_cfg:
        # Existing config is considered already saved; require a user change to re-enable Save
        ollama_save_btn.disabled = True
    ollama_reset_btn = ft.TextButton(
        "Reset settings",
        icon=getattr(ICONS, "BACKSPACE", ICONS.CANCEL),
        on_click=on_reset_ollama,
    )

    def on_ollama_enable_changed(e):
        # Update control states and mark config as having unsaved changes
        update_ollama_controls(e)
        mark_ollama_dirty(e)

    try:
        ollama_enable_cb.on_change = on_ollama_enable_changed
        ollama_models_dd.on_change = mark_ollama_dirty
    except Exception:
        pass
    update_ollama_controls()
    try:
        if _ollama_has_cfg and _ollama_enabled and hasattr(page, "run_task"):
            page.run_task(on_test_ollama)
    except Exception:
        pass

    db_info_text = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))
    db_summary_text = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))

    def _format_bytes(num: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if num < 1024 or unit == "TB":
                return f"{num:.1f} {unit}"
            num /= 1024.0
        return f"{num:.1f} TB"

    def refresh_db_info(_=None):
        try:
            init_db()
            db_path = get_db_path()
            size_bytes = 0
            try:
                if os.path.exists(db_path):
                    size_bytes = os.path.getsize(db_path)
            except Exception:
                size_bytes = 0
            conn = get_connection()
            cursor = conn.cursor()

            def _count(table: str) -> int:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row = cursor.fetchone()
                    return int(row[0]) if row is not None else 0
                except Exception:
                    return 0

            settings_count = _count("settings")
            cfg_count = _count("training_configs")
            sessions_count = _count("scrape_sessions")
            pairs_count = _count("scraped_pairs")
            runs_count = _count("training_runs")
            try:
                logs_count = int(get_log_count())
            except Exception:
                logs_count = 0

            db_info_text.value = f"Path: {db_path}"
            human_size = _format_bytes(size_bytes) if size_bytes > 0 else "0 B"
            db_summary_text.value = (
                f"Size {human_size} · Settings {settings_count} · Training configs {cfg_count} · "
                f"Scrape sessions {sessions_count} / pairs {pairs_count} · Training runs {runs_count} · Logs {logs_count}"
            )
        except Exception as e:
            db_summary_text.value = f"Error reading database info: {e}"
        try:
            page.update()
        except Exception:
            pass

    db_refresh_btn = ft.TextButton(
        "Refresh info",
        icon=REFRESH_ICON,
        on_click=refresh_db_info,
    )

    def on_clear_logs_click(_):
        try:
            current_count = int(get_log_count())
        except Exception:
            current_count = 0

        description = "This will permanently delete all application logs stored in the database."
        if current_count:
            description += f"\nCurrent log entries: {current_count}."

        confirm_dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(
                        getattr(ICONS, "DELETE_FOREVER", getattr(ICONS, "DELETE", ICONS.CLOSE)),
                        color=getattr(COLORS, "RED", COLORS.ERROR),
                    ),
                    ft.Text("Clear logs?"),
                ],
                alignment=ft.MainAxisAlignment.START,
            ),
            content=ft.Text(description),
            actions=[],
        )

        def _close_dialog():
            try:
                confirm_dlg.open = False
                page.update()
            except Exception:
                pass

        def _do_clear(_e):
            try:
                deleted = int(clear_logs())
                try:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Cleared {deleted} log entries"))
                    page.open(page.snack_bar)
                except Exception:
                    pass
            except Exception as ex:
                try:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Failed to clear logs: {ex}"))
                    page.open(page.snack_bar)
                except Exception:
                    pass
            refresh_db_info()
            _close_dialog()

        try:
            confirm_dlg.actions = [
                ft.TextButton("Cancel", on_click=lambda e: _close_dialog()),
                ft.ElevatedButton(
                    "Delete logs",
                    icon=getattr(ICONS, "CHECK", ICONS.DELETE),
                    on_click=_do_clear,
                ),
            ]
        except Exception:
            pass

        try:
            if hasattr(page, "open") and callable(getattr(page, "open")):
                page.open(confirm_dlg)
            else:
                page.dialog = confirm_dlg
                confirm_dlg.open = True
            page.update()
        except Exception:
            try:
                page.dialog = confirm_dlg
                confirm_dlg.open = True
                page.update()
            except Exception:
                pass

    db_clear_logs_btn = ft.TextButton(
        "Clear logs",
        icon=getattr(ICONS, "DELETE_FOREVER", getattr(ICONS, "DELETE", ICONS.CANCEL)),
        on_click=on_clear_logs_click,
    )

    def on_clear_scraped_click(_):
        # Delete all scrape sessions and scraped pairs
        try:
            init_db()
            conn = get_connection()
            cursor = conn.cursor()

            def _count(table: str) -> int:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row = cursor.fetchone()
                    return int(row[0]) if row is not None else 0
                except Exception:
                    return 0

            sessions_count = _count("scrape_sessions")
            pairs_count = _count("scraped_pairs")
        except Exception:
            sessions_count = 0
            pairs_count = 0

        description = "This will permanently delete all scraped sessions and pairs."
        if sessions_count or pairs_count:
            description += f"\nCurrent sessions: {sessions_count}, pairs: {pairs_count}."

        confirm_dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(
                        getattr(ICONS, "DELETE_FOREVER", getattr(ICONS, "DELETE", ICONS.CLOSE)),
                        color=getattr(COLORS, "RED", getattr(COLORS, "ERROR", None)),
                    ),
                    ft.Text("Clear scraped data?"),
                ],
                alignment=ft.MainAxisAlignment.START,
            ),
            content=ft.Text(description),
            actions=[],
        )

        def _close_dialog():
            try:
                confirm_dlg.open = False
                page.update()
            except Exception:
                pass

        def _do_clear(_e):
            try:
                init_db()
                conn2 = get_connection()
                cur2 = conn2.cursor()
                cur2.execute("DELETE FROM scrape_sessions")
                deleted_sessions = int(cur2.rowcount or 0)
                conn2.commit()
                try:
                    page.snack_bar = ft.SnackBar(
                        ft.Text(f"Cleared scraped data: {deleted_sessions} sessions (and all pairs)")
                    )
                    page.open(page.snack_bar)
                except Exception:
                    pass
            except Exception as ex:
                try:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Failed to clear scraped data: {ex}"))
                    page.open(page.snack_bar)
                except Exception:
                    pass
            refresh_db_info()
            _close_dialog()

        try:
            confirm_dlg.actions = [
                ft.TextButton("Cancel", on_click=lambda e: _close_dialog()),
                ft.ElevatedButton(
                    "Delete scraped data",
                    icon=getattr(ICONS, "CHECK", ICONS.DELETE),
                    on_click=_do_clear,
                ),
            ]
        except Exception:
            pass

        try:
            if hasattr(page, "open") and callable(getattr(page, "open")):
                page.open(confirm_dlg)
            else:
                page.dialog = confirm_dlg
                confirm_dlg.open = True
            page.update()
        except Exception:
            try:
                page.dialog = confirm_dlg
                confirm_dlg.open = True
                page.update()
            except Exception:
                pass

    db_clear_scraped_btn = ft.TextButton(
        "Clear scraped data",
        icon=getattr(ICONS, "DELETE_SWEEP", getattr(ICONS, "DELETE", ICONS.CANCEL)),
        on_click=on_clear_scraped_click,
    )

    def on_clear_configs_click(_):
        # Delete all training configs and last-used pointer
        try:
            init_db()
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM training_configs")
            row = cursor.fetchone()
            cfg_count = int(row[0]) if row is not None else 0
        except Exception:
            cfg_count = 0

        description = "This will permanently delete all saved training configurations."
        if cfg_count:
            description += f"\nCurrent configs: {cfg_count}."

        confirm_dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(
                        getattr(ICONS, "DELETE_FOREVER", getattr(ICONS, "DELETE", ICONS.CLOSE)),
                        color=getattr(COLORS, "RED", getattr(COLORS, "ERROR", None)),
                    ),
                    ft.Text("Clear training configs?"),
                ],
                alignment=ft.MainAxisAlignment.START,
            ),
            content=ft.Text(description),
            actions=[],
        )

        def _close_dialog():
            try:
                confirm_dlg.open = False
                page.update()
            except Exception:
                pass

        def _do_clear(_e):
            try:
                init_db()
                conn2 = get_connection()
                cur2 = conn2.cursor()
                cur2.execute("DELETE FROM training_configs")
                deleted_cfgs = int(cur2.rowcount or 0)
                # Clear last_used_config pointer
                try:
                    cur2.execute("DELETE FROM app_state WHERE key = 'last_used_config'")
                except Exception:
                    pass
                conn2.commit()
                try:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Cleared {deleted_cfgs} training configs"))
                    page.open(page.snack_bar)
                except Exception:
                    pass
            except Exception as ex:
                try:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Failed to clear configs: {ex}"))
                    page.open(page.snack_bar)
                except Exception:
                    pass
            refresh_db_info()
            _close_dialog()

        try:
            confirm_dlg.actions = [
                ft.TextButton("Cancel", on_click=lambda e: _close_dialog()),
                ft.ElevatedButton(
                    "Delete configs",
                    icon=getattr(ICONS, "CHECK", ICONS.DELETE),
                    on_click=_do_clear,
                ),
            ]
        except Exception:
            pass

        try:
            if hasattr(page, "open") and callable(getattr(page, "open")):
                page.open(confirm_dlg)
            else:
                page.dialog = confirm_dlg
                confirm_dlg.open = True
            page.update()
        except Exception:
            try:
                page.dialog = confirm_dlg
                confirm_dlg.open = True
                page.update()
            except Exception:
                pass

    db_clear_configs_btn = ft.TextButton(
        "Clear training configs",
        icon=getattr(ICONS, "DELETE_SWEEP", getattr(ICONS, "DELETE", ICONS.CANCEL)),
        on_click=on_clear_configs_click,
    )

    def on_clear_runs_click(_):
        # Delete all training runs and their managed storage
        try:
            init_db()
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM training_runs")
            ids = [int(row[0]) for row in cursor.fetchall()]
            run_count = len(ids)
        except Exception:
            ids = []
            run_count = 0

        storage_root = ""
        try:
            storage_root = get_managed_storage_root()
        except Exception:
            storage_root = ""

        description = "This will delete all training run records and their managed storage directories."
        if run_count:
            description += f"\nCurrent training runs: {run_count}."
        if storage_root:
            description += f"\nManaged storage root: {storage_root}."

        confirm_dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(
                        getattr(ICONS, "DELETE_FOREVER", getattr(ICONS, "DELETE", ICONS.CLOSE)),
                        color=getattr(COLORS, "RED", getattr(COLORS, "ERROR", None)),
                    ),
                    ft.Text("Clear training runs?"),
                ],
                alignment=ft.MainAxisAlignment.START,
            ),
            content=ft.Text(description),
            actions=[],
        )

        def _close_dialog():
            try:
                confirm_dlg.open = False
                page.update()
            except Exception:
                pass

        def _do_clear(_e):
            cleared = 0
            try:
                for rid in ids:
                    try:
                        if delete_training_run(rid):
                            cleared += 1
                    except Exception:
                        pass
                try:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Cleared {cleared} training runs"))
                    page.open(page.snack_bar)
                except Exception:
                    pass
            except Exception as ex:
                try:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Failed to clear training runs: {ex}"))
                    page.open(page.snack_bar)
                except Exception:
                    pass
            refresh_db_info()
            _close_dialog()

        try:
            confirm_dlg.actions = [
                ft.TextButton("Cancel", on_click=lambda e: _close_dialog()),
                ft.ElevatedButton(
                    "Delete training runs",
                    icon=getattr(ICONS, "CHECK", ICONS.DELETE),
                    on_click=_do_clear,
                ),
            ]
        except Exception:
            pass

        try:
            if hasattr(page, "open") and callable(getattr(page, "open")):
                page.open(confirm_dlg)
            else:
                page.dialog = confirm_dlg
                confirm_dlg.open = True
            page.update()
        except Exception:
            try:
                page.dialog = confirm_dlg
                confirm_dlg.open = True
                page.update()
            except Exception:
                pass

    db_clear_runs_btn = ft.TextButton(
        "Clear training runs",
        icon=getattr(ICONS, "DELETE_SWEEP", getattr(ICONS, "DELETE", ICONS.CANCEL)),
        on_click=on_clear_runs_click,
    )

    def on_factory_reset_click(_):
        # Full reset: delete DB file and managed storage, then re-init
        try:
            db_path = get_db_path()
        except Exception:
            db_path = "<unknown>"

        try:
            storage_root = get_managed_storage_root()
        except Exception:
            storage_root = ""

        description = (
            "This will perform a factory reset of FineFoundry's database and managed training storage.\n"
            "All settings, scraped data, training configs, training runs, and logs will be permanently deleted."
        )
        description += f"\nDatabase: {db_path}"
        if storage_root:
            description += f"\nTraining storage: {storage_root}"

        confirm_dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(
                        getattr(ICONS, "DELETE_FOREVER", getattr(ICONS, "DELETE", ICONS.CLOSE)),
                        color=getattr(COLORS, "RED", getattr(COLORS, "ERROR", None)),
                    ),
                    ft.Text("Factory reset?"),
                ],
                alignment=ft.MainAxisAlignment.START,
            ),
            content=ft.Text(description),
            actions=[],
        )

        def _close_dialog():
            try:
                confirm_dlg.open = False
                page.update()
            except Exception:
                pass

        def _do_reset(_e):
            errors = []
            try:
                close_all_connections()
            except Exception as ex:
                errors.append(f"close connections: {ex}")

            try:
                if db_path and os.path.exists(db_path):
                    os.remove(db_path)
            except Exception as ex:
                errors.append(f"remove db: {ex}")

            try:
                if storage_root and os.path.isdir(storage_root):
                    shutil.rmtree(storage_root)
            except Exception as ex:
                errors.append(f"remove storage: {ex}")

            try:
                init_db()
            except Exception as ex:
                errors.append(f"re-init db: {ex}")

            refresh_db_info()

            msg = "Factory reset complete."
            if errors:
                msg += " Issues: " + "; ".join(errors)
            try:
                page.snack_bar = ft.SnackBar(ft.Text(msg))
                page.open(page.snack_bar)
            except Exception:
                pass
            _close_dialog()

        try:
            confirm_dlg.actions = [
                ft.TextButton("Cancel", on_click=lambda e: _close_dialog()),
                ft.ElevatedButton(
                    "Factory reset",
                    icon=getattr(ICONS, "CHECK", ICONS.DELETE),
                    on_click=_do_reset,
                ),
            ]
        except Exception:
            pass

        try:
            if hasattr(page, "open") and callable(getattr(page, "open")):
                page.open(confirm_dlg)
            else:
                page.dialog = confirm_dlg
                confirm_dlg.open = True
            page.update()
        except Exception:
            try:
                page.dialog = confirm_dlg
                confirm_dlg.open = True
                page.update()
            except Exception:
                pass

    db_factory_reset_btn = ft.TextButton(
        "Factory reset (DB + storage)",
        icon=getattr(ICONS, "DELETE_FOREVER", getattr(ICONS, "DELETE", ICONS.CANCEL)),
        on_click=on_factory_reset_click,
    )

    refresh_db_info()

    # ---------- SETTINGS (System Check) CONTROLS ----------

    system_check_status = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))
    system_check_log = ft.ListView(expand=False, spacing=2, auto_scroll=True)
    system_check_log_container = ft.Container(
        content=system_check_log,
        height=180,
        border=ft.border.all(1, WITH_OPACITY(0.15, BORDER_BASE)),
        border_radius=4,
        padding=8,
        visible=False,
    )

    system_check_summary = ft.Column(spacing=4)
    system_check_summary_container = ft.Container(
        content=system_check_summary,
        padding=8,
        border=ft.border.all(1, WITH_OPACITY(0.08, BORDER_BASE)),
        border_radius=6,
        visible=False,
    )

    async def on_run_system_check():
        """Run a diagnostics pipeline and stream output into the Settings tab.

        Steps (all executed even if earlier ones fail):
        - feature-specific pytest groups (scraping, merge/build, training/inference)
        - full pytest test suite
        - coverage run --source=src -m pytest
        - coverage report -m
        """

        # Prevent duplicate runs
        if getattr(system_check_btn, "disabled", False):
            return

        system_check_status.value = "Running diagnostics (pytest & coverage)…"
        system_check_log_container.visible = True
        system_check_log.controls.clear()
        system_check_summary.controls.clear()
        system_check_summary_container.visible = False
        system_check_btn.disabled = True
        await safe_update(page)

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        step_results: list[tuple[str, int]] = []

        async def _run_step(title: str, args: list[str]) -> int:
            """Run a diagnostic step and stream its output.

            Args:
                title: Human-readable step name.
                args: Arguments passed to the current Python executable.
            """

            # Header for this step
            system_check_log.controls.append(ft.Text(f"=== {title} ===", size=12, weight=ft.FontWeight.BOLD))
            await safe_update(page)

            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=project_root,
                )
            except Exception as e:  # pragma: no cover - environment-specific
                system_check_log.controls.append(
                    ft.Text(f"[{title}] failed to start: {e}", size=11, color=WITH_OPACITY(0.9, COLORS.RED))
                )
                await safe_update(page)
                return 1

            assert proc.stdout is not None
            while True:
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode(errors="replace").rstrip()
                if not line:
                    continue
                system_check_log.controls.append(ft.Text(line, size=11))
                # Keep log bounded
                if len(system_check_log.controls) > 1000:
                    system_check_log.controls.pop(0)
                await safe_update(page)

            rc = await proc.wait()
            step_results.append((title, rc))
            system_check_log.controls.append(
                ft.Text(f"[{title}] exit code: {rc}", size=11, color=WITH_OPACITY(0.8, BORDER_BASE))
            )
            await safe_update(page)
            return rc

        any_failures = False
        try:
            # 0) Domain-specific pytest groups
            scraping_tests = [
                "tests/unit/test_scraper_utils.py",
                "tests/unit/test_scrape_orchestration.py",
            ]
            if (
                await _run_step(
                    "Scraping & scrape orchestration tests",
                    ["-m", "pytest", *scraping_tests],
                )
                != 0
            ):
                any_failures = True

            merge_build_tests = [
                "tests/unit/test_merge.py",
                "tests/unit/test_build.py",
            ]
            if (
                await _run_step(
                    "Merge & build pipeline tests",
                    ["-m", "pytest", *merge_build_tests],
                )
                != 0
            ):
                any_failures = True

            training_tests = [
                "tests/unit/test_training_config.py",
            ]
            if (
                await _run_step(
                    "Training config & local training infra tests",
                    ["-m", "pytest", *training_tests],
                )
                != 0
            ):
                any_failures = True

            inference_tests = [
                "tests/unit/test_local_inference.py",
                "tests/unit/test_training_controller_local_infer.py",
            ]
            if (
                await _run_step(
                    "Quick local inference & UI wiring tests",
                    ["-m", "pytest", *inference_tests],
                )
                != 0
            ):
                any_failures = True

            # 1) Full unit/integration suite
            if await _run_step("Full test suite (pytest tests)", ["-m", "pytest", "tests"]) != 0:
                any_failures = True

            # 2) Coverage run + report
            if await _run_step("coverage run", ["-m", "coverage", "run", "--source=src", "-m", "pytest"]) != 0:
                any_failures = True
            if await _run_step("coverage report", ["-m", "coverage", "report", "-m"]) != 0:
                any_failures = True

            # Build grouped visual summary under the log
            system_check_summary.controls.clear()
            ok_color = getattr(COLORS, "GREEN_400", getattr(COLORS, "GREEN", BORDER_BASE))
            err_color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", BORDER_BASE))

            # High-level subtitle above the grouped sections
            system_check_summary.controls.append(
                ft.Text(
                    "System health summary by area:",
                    size=12,
                    color=WITH_OPACITY(0.8, BORDER_BASE),
                )
            )

            def _section_for_title(title: str) -> str:
                if "Scraping & scrape" in title:
                    return "Data Collection"
                if "Merge & build" in title:
                    return "Dataset Build"
                if "Training config" in title or "Quick local inference" in title:
                    return "Training & Inference"
                return "Overall Health"

            grouped: dict[str, list[tuple[str, int]]] = {}
            for title, rc in step_results:
                sec = _section_for_title(title)
                grouped.setdefault(sec, []).append((title, rc))

            section_order = [
                "Data Collection",
                "Dataset Build",
                "Training & Inference",
                "Overall Health",
            ]

            for section in section_order:
                if section not in grouped:
                    continue
                rows_for_section = grouped[section]
                all_ok = all(rc == 0 for _, rc in rows_for_section)
                section_color = ok_color if all_ok else err_color

                row_controls = []
                for title, rc in rows_for_section:
                    ok = rc == 0
                    icon_const = (
                        getattr(ICONS, "CHECK_CIRCLE", getattr(ICONS, "CHECK", getattr(ICONS, "DONE", None)))
                        if ok
                        else getattr(ICONS, "ERROR", getattr(ICONS, "CANCEL", getattr(ICONS, "WARNING", None)))
                    )
                    icon_color = ok_color if ok else err_color
                    label = "Passed" if ok else "Failed"
                    row_controls.append(
                        ft.Row(
                            [
                                ft.Icon(icon_const, color=icon_color, size=16)
                                if icon_const is not None
                                else ft.Container(),
                                ft.Text(
                                    f"{title}: {label} (exit {rc})",
                                    size=12,
                                    color=WITH_OPACITY(0.9, BORDER_BASE),
                                ),
                            ],
                            spacing=6,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        )
                    )

                system_check_summary.controls.append(
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Text(
                                    section,
                                    size=13,
                                    weight=ft.FontWeight.BOLD,
                                    color=WITH_OPACITY(0.95, BORDER_BASE),
                                ),
                                *row_controls,
                            ],
                            spacing=6,
                        ),
                        padding=10,
                        border=ft.border.all(1, WITH_OPACITY(0.18, section_color)),
                        border_radius=10,
                        bgcolor=WITH_OPACITY(0.03 if all_ok else 0.06, section_color),
                    )
                )
            system_check_summary_container.visible = True

            if any_failures:
                system_check_status.value = "Diagnostics complete: some steps failed (see summary below)."
            else:
                system_check_status.value = "Diagnostics complete: all steps passed (see summary below)."
        except Exception as e:  # pragma: no cover - defensive
            system_check_status.value = f"Diagnostics error: {e}"
        finally:
            system_check_btn.disabled = False
            await safe_update(page)

    def _on_save_system_log(e):
        try:
            path = getattr(e, "path", None)
            if not path:
                return
            lines = []
            for ctl in system_check_log.controls:
                if isinstance(ctl, ft.Text):
                    lines.append(str(getattr(ctl, "value", "")))
            txt = "\n".join(lines)
            if txt and not txt.endswith("\n"):
                txt += "\n"
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            system_check_status.value = f"Saved diagnostics log to: {path}"
            try:
                page.update()
            except Exception:
                pass
        except Exception as ex:  # pragma: no cover - defensive
            system_check_status.value = f"Failed to save diagnostics log: {ex}"
            try:
                page.update()
            except Exception:
                pass

    system_check_log_picker = ft.FilePicker(on_result=_on_save_system_log)
    try:
        page.overlay.append(system_check_log_picker)
    except Exception:
        pass

    system_check_btn = ft.ElevatedButton(
        "Run system diagnostics",
        icon=getattr(ICONS, "SCIENCE", getattr(ICONS, "PLAY_CIRCLE", getattr(ICONS, "BUG_REPORT", None))),
        on_click=lambda e: page.run_task(on_run_system_check),
    )

    system_check_download_btn = ft.OutlinedButton(
        "Download diagnostics log",
        icon=getattr(ICONS, "DOWNLOAD", getattr(ICONS, "SAVE_ALT", ICONS.SAVE)),
        on_click=lambda e: system_check_log_picker.save_file(
            dialog_title="Save diagnostics log",
            file_name=f"diagnostics-{int(time.time())}.log",
            allowed_extensions=["txt", "log"],
        ),
    )
    # Compose Scrape tab via dedicated controller (refactored wiring)
    scrape_tab = build_scrape_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
        proxy_enable_cb=proxy_enable_cb,
        use_env_cb=use_env_cb,
        proxy_url_tf=proxy_url_tf,
        offline_mode_sw=offline_mode_sw,
    )

    # Compose Publish tab via dedicated controller (refactored wiring)
    build_tab = build_build_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
        _hf_cfg=_hf_cfg,
        ollama_enable_cb=ollama_enable_cb,
        ollama_models_dd=ollama_models_dd,
        offline_mode_sw=offline_mode_sw,
    )

    # ---------- MERGE DATASETS TAB ----------
    merge_tab = build_merge_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
        offline_mode_sw=offline_mode_sw,
    )

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
        offline_mode_sw=offline_mode_sw,
    )
    # ---------- INFERENCE TAB (global inference over fine-tuned models) ----------

    inference_tab = build_inference_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
        train_state=train_state,
    )

    # ---------- SETTINGS TAB (Proxy config + Offline Mode + Ollama + Database & Storage + System Check) ----------
    settings_tab = build_settings_tab(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        proxy_enable_cb=proxy_enable_cb,
        use_env_cb=use_env_cb,
        proxy_url_tf=proxy_url_tf,
        offline_mode_sw=offline_mode_sw,
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
        ollama_models_dd=ollama_models_dd,
        ollama_test_btn=ollama_test_btn,
        ollama_refresh_btn=ollama_refresh_btn,
        ollama_save_btn=ollama_save_btn,
        ollama_reset_btn=ollama_reset_btn,
        ollama_status=ollama_status,
        REFRESH_ICON=REFRESH_ICON,
        db_info_text=db_info_text,
        db_summary_text=db_summary_text,
        db_refresh_btn=db_refresh_btn,
        db_clear_logs_btn=db_clear_logs_btn,
        db_clear_scraped_btn=db_clear_scraped_btn,
        db_clear_configs_btn=db_clear_configs_btn,
        db_clear_runs_btn=db_clear_runs_btn,
        db_factory_reset_btn=db_factory_reset_btn,
        system_check_status=system_check_status,
        system_check_btn=system_check_btn,
        system_check_log_container=system_check_log_container,
        system_check_summary_container=system_check_summary_container,
        system_check_download_btn=system_check_download_btn,
    )

    # Compose Dataset Analysis tab via dedicated controller (refactored wiring)
    analysis_tab = build_analysis_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
        offline_mode_sw=offline_mode_sw,
    )

    # Tabs and welcome screen
    data_sources_tab = ft.Tab(text="Data Sources", icon=ICONS.SEARCH, content=scrape_tab)
    build_publish_tab = ft.Tab(text="Publish", icon=ICONS.BUILD_CIRCLE_OUTLINED, content=build_tab)
    analysis_tab_tab = ft.Tab(
        text="Dataset Analysis", icon=getattr(ICONS, "INSIGHTS", ICONS.ANALYTICS), content=analysis_tab
    )
    merge_datasets_tab = ft.Tab(
        text="Merge Datasets", icon=getattr(ICONS, "MERGE_TYPE", ICONS.TABLE_VIEW), content=merge_tab
    )
    training_tab_tab = ft.Tab(text="Training", icon=getattr(ICONS, "SCIENCE", ICONS.PLAY_CIRCLE), content=training_tab)
    inference_tab_tab = ft.Tab(
        text="Inference",
        icon=getattr(ICONS, "PSYCHOLOGY", getattr(ICONS, "CHAT", ICONS.PLAY_CIRCLE)),
        content=inference_tab,
    )
    settings_tab_tab = ft.Tab(text="Settings", icon=ICONS.SETTINGS, content=settings_tab)

    tabs = ft.Tabs(
        tabs=[
            data_sources_tab,
            analysis_tab_tab,
            merge_datasets_tab,
            training_tab_tab,
            inference_tab_tab,
            build_publish_tab,
            settings_tab_tab,
        ],
        expand=1,
    )

    def update_tab_offline_badges():
        """Update the global online/offline status indicator in the AppBar."""
        try:
            is_offline = bool(getattr(offline_mode_sw, "value", False))
        except Exception:
            is_offline = False
        try:
            if is_offline:
                # Offline: red dot and 'Offline' label
                try:
                    offline_status_dot.bgcolor = getattr(COLORS, "RED", "#d32f2f")
                except Exception:
                    pass
                offline_status_label.value = "Offline"
            else:
                # Online: green dot and 'Online' label
                try:
                    offline_status_dot.bgcolor = getattr(COLORS, "GREEN", "#388e3c")
                except Exception:
                    pass
                offline_status_label.value = "Online"
        except Exception:
            pass

    try:
        update_tab_offline_badges()
    except Exception:
        pass

    def show_main_app(_=None):
        try:
            page.controls.clear()
            page.add(tabs)
            try:
                page.floating_action_button = ft.FloatingActionButton(
                    icon=getattr(ICONS, "INFO", None),
                    tooltip="User guide (F1)",
                    on_click=lambda e: open_user_guide(e),
                )
            except Exception:
                pass
            try:
                page.update()
            except Exception:
                pass
            # Training tab manages its own visibility and skill controls internally.
        except Exception:
            pass

    welcome_logo = ft.Image(src="img/FineForge-logo.png", width=240, height=240, fit=ft.ImageFit.CONTAIN)
    welcome_tagline = ft.Text(
        "Curate, analyze, and fine‑tune datasets with a beautiful desktop UI.", color=WITH_OPACITY(0.8, BORDER_BASE)
    )
    start_btn = ft.FilledButton(
        "Start", icon=getattr(ICONS, "PLAY_ARROW", getattr(ICONS, "PLAY_CIRCLE", None)), on_click=show_main_app
    )

    welcome_view = ft.Container(
        expand=1,
        content=ft.Column(
            [
                welcome_logo,
                ft.Container(height=8),
                welcome_tagline,
                ft.Container(height=16),
                start_btn,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=6,
        ),
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
