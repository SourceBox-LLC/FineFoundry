import asyncio
import os
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
)
from db import init_db

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
FineFoundry is a desktop studio to collect data, merge, analyze, build/publish, and train LLM datasets.

Tabs:
- Data Sources
- Build / Publish
- Training
- Merge Datasets
- Dataset Analysis
- Settings

## Quick Navigation
- Top‑right icons: Refresh • Theme toggle
- User guide: press F1 or click the bottom‑right Help FAB.
- Use the tabs to switch workflows.

## Data Sources
- Choose source: 4chan, Reddit, StackExchange, or **Synthetic** (generate from documents).
- For scraping: configure parameters (max pairs, threads, delays, length filters).
- For synthetic: select files (PDF, DOCX, PPTX, HTML, TXT) or URLs to generate QA pairs, Chain of Thought, or summaries.
- Preview results in a two‑column grid (input/output).
- Proxy support per scraper; system env proxies optional.

## Build / Publish
- Create train/val/test splits with Hugging Face `datasets`.
- Save locally and optionally push to the Hugging Face Hub.
- Provide `HF_TOKEN` in Settings or via environment/CLI login.

## Merge Datasets
- Combine database sessions and/or HF datasets.
- Auto‑map `input`/`output` columns; filter empty rows.
- Merged data is saved to the database as a new session.

## Training
- Training target: choose **Runpod - Pod** (remote GPU pod) or **local** (Docker on this machine).
- Hyperparameters: base model, epochs, learning rate, batch size, grad accumulation, max steps, packing, resume.
- Outputs & checkpoints: saved under **Output dir**. For containers, use paths under `/data/...` so they map back into your mounted host folder.
- Hugging Face auth: save a token in **Settings → Hugging Face** or export `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`. Required for pushing models or using private datasets.

### Local Training (Docker)
- Requires Docker Desktop / `docker` CLI available on your machine.
- Select or create a **Training run** for managed storage; the run's directory is mounted at `/data` inside the container.
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
- Saved configs are stored in the database. The last used config auto‑loads on startup so you can continue where you left off.

## Dataset Analysis
- Select dataset source (Database or HF) and click **Analyze dataset**.
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
                    return ft.IconButton(
                        icon=icon_const, tooltip=tooltip, on_click=lambda e: page.run_task(on_click_cb)
                    )
                return ft.IconButton(icon=icon_const, tooltip=tooltip, on_click=lambda e: on_click_cb(e))
        except Exception:
            pass
        if getattr(on_click_cb, "__name__", "").endswith("_async"):
            return ft.TextButton(text_fallback or tooltip, on_click=lambda e: page.run_task(on_click_cb))
        return ft.TextButton(text_fallback or tooltip, on_click=lambda e: on_click_cb(e))

    page.appbar = ft.AppBar(
        leading=ft.IconButton(
            icon=getattr(
                ICONS, "DATASET_LINKED_OUTLINED", getattr(ICONS, "DATASET", getattr(ICONS, "DESCRIPTION", None))
            ),
            tooltip="Open FineFoundry website",
            on_click=lambda e: page.launch_url("https://finefoundry.fly.dev/"),
        ),
        title=ft.Text(APP_TITLE, weight=ft.FontWeight.BOLD),
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

    runpod_test_btn = ft.ElevatedButton(
        "Test key", icon=ICONS.CHECK_CIRCLE, on_click=lambda e: page.run_task(on_test_runpod)
    )
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
    ollama_base_url_tf = ft.TextField(
        label="Ollama base URL", value=_ollama_cfg.get("base_url", "http://127.0.0.1:11434"), width=420
    )
    ollama_default_model_tf = ft.TextField(
        label="Preferred model (optional)", value=_ollama_cfg.get("default_model", ""), width=300
    )
    ollama_models_dd = ft.Dropdown(
        label="Available models", options=[], value=_ollama_cfg.get("selected_model") or None, width=420, disabled=True
    )
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

    ollama_test_btn = ft.ElevatedButton(
        "Test connection", icon=ICONS.CHECK_CIRCLE, on_click=lambda e: page.run_task(on_test_ollama)
    )
    ollama_refresh_btn = ft.TextButton(
        "Refresh models", icon=REFRESH_ICON, on_click=lambda e: page.run_task(on_refresh_models)
    )
    ollama_save_btn = ft.OutlinedButton("Save", icon=ICONS.SAVE, on_click=on_save_ollama)

    try:
        ollama_enable_cb.on_change = update_ollama_controls
    except Exception:
        pass
    update_ollama_controls()
    # Compose Scrape tab via dedicated controller (refactored wiring)
    scrape_tab = build_scrape_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
        proxy_enable_cb=proxy_enable_cb,
        use_env_cb=use_env_cb,
        proxy_url_tf=proxy_url_tf,
    )

    # Compose Build / Publish tab via dedicated controller (refactored wiring)
    build_tab = build_build_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
        _hf_cfg=_hf_cfg,
        ollama_enable_cb=ollama_enable_cb,
        ollama_models_dd=ollama_models_dd,
    )

    # ---------- MERGE DATASETS TAB ----------
    merge_tab = build_merge_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
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

    # Compose Dataset Analysis tab via dedicated controller (refactored wiring)
    analysis_tab = build_analysis_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
    )

    # Tabs and welcome screen
    tabs = ft.Tabs(
        tabs=[
            ft.Tab(text="Data Sources", icon=ICONS.SEARCH, content=scrape_tab),
            ft.Tab(text="Build / Publish", icon=ICONS.BUILD_CIRCLE_OUTLINED, content=build_tab),
            ft.Tab(text="Dataset Analysis", icon=getattr(ICONS, "INSIGHTS", ICONS.ANALYTICS), content=analysis_tab),
            ft.Tab(text="Merge Datasets", icon=getattr(ICONS, "MERGE_TYPE", ICONS.TABLE_VIEW), content=merge_tab),
            ft.Tab(text="Training", icon=getattr(ICONS, "SCIENCE", ICONS.PLAY_CIRCLE), content=training_tab),
            ft.Tab(
                text="Inference",
                icon=getattr(ICONS, "PSYCHOLOGY", getattr(ICONS, "CHAT", ICONS.PLAY_CIRCLE)),
                content=inference_tab,
            ),
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
