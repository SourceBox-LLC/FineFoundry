import asyncio
import random
import time
import os
import shutil
from datetime import datetime
from typing import List, Optional, Tuple, Callable
import json
import re
from collections import Counter
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

import flet as ft
import httpx
import fourchan_scraper as sc
import requests
try:
    import save_dataset as sd
except Exception:
    import sys as _sys
    _sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import save_dataset as sd

# Runpod infra helper (local module)
try:
    import ensure_infra as rp_infra
except Exception:
    import sys as __sys
    __sys.path.append(os.path.dirname(__file__))
    import ensure_infra as rp_infra

# Runpod pod helper (local module)
try:
    import runpod_pod as rp_pod
except Exception:
    import sys as __sys2
    __sys2.path.append(os.path.dirname(__file__))
    import runpod_pod as rp_pod

import reddit_scraper as rdt
import stackexchange_scraper as sx
try:
    # Hugging Face datasets (for merge tab)
    from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, get_dataset_config_names, load_from_disk
    try:
        from datasets import interleave_datasets as hf_interleave
    except Exception:
        hf_interleave = None
except Exception:
    load_dataset = None
    Dataset = None
    DatasetDict = None
    concatenate_datasets = None
    hf_interleave = None
    get_dataset_config_names = None
    load_from_disk = None

try:
    from huggingface_hub import HfApi, HfFolder
except Exception:
    HfApi = None
    HfFolder = None

# Robust color aliasing: prefer ft.Colors, fall back to ft.colors if present
if hasattr(ft, "Colors"):
    COLORS = ft.Colors
else:
    COLORS = getattr(ft, "colors", None)

# Robust icons aliasing: prefer ft.Icons, fall back to ft.icons if present
if hasattr(ft, "Icons"):
    ICONS = ft.Icons
else:
    ICONS = getattr(ft, "icons", None)

# Common icon fallbacks
REFRESH_ICON = getattr(ICONS, "REFRESH", getattr(ICONS, "AUTORENEW", ICONS.RESTART_ALT))

async def safe_update(page: ft.Page):
    """Update the page across Flet versions (async if available, else sync)."""
    if hasattr(page, "update_async"):
        return await page.update_async()
    return page.update()

def WITH_OPACITY(opacity: float, color):
    """Apply opacity if supported in this Flet build; otherwise return color as-is."""
    # Try ft.colors.with_opacity
    if hasattr(ft, "colors") and hasattr(ft.colors, "with_opacity"):
        try:
            return ft.colors.with_opacity(opacity, color)
        except Exception:
            pass
    # Try Colors.with_opacity
    if hasattr(COLORS, "with_opacity"):
        try:
            return COLORS.with_opacity(opacity, color)
        except Exception:
            pass
    return color

APP_TITLE = "Dataset Studio"
ACCENT_COLOR = COLORS.AMBER
BORDER_BASE = getattr(COLORS, "ON_SURFACE", getattr(COLORS, "GREY", "#e0e0e0"))

# Fallback board list (used if API unavailable)
DEFAULT_BOARDS: List[str] = [
    # SFW
    "a", "c", "w", "m", "cgl", "cm", "f", "n", "jp", "vt", "vp",
    "v", "vg", "vr", "vm", "vmg", "vst", "co", "g", "tv", "k", "o",
    "an", "tg", "sp", "xs", "sci", "his", "int", "out", "toy", "i", "po",
    "p", "ck", "ic", "wg", "lit", "mu", "fa", "3", "gd", "diy", "wsg",
    "biz", "trv", "fit", "s4s", "adv", "news", "qa", "qst",
    # 18+
    "b", "r9k", "pol", "soc", "s", "hc", "hm", "h", "e", "u", "d", "y",
    "t", "hr", "gif", "aco", "trash", "mlp", "bant", "x",
]


def load_4chan_boards() -> List[str]:
    """Load all 4chan board codes from public API with a safe fallback."""
    url = "https://a.4cdn.org/boards.json"
    try:
        with urlopen(url, timeout=6) as resp:
            data = json.load(resp)
            names = sorted([b.get("board") for b in data.get("boards", []) if b.get("board")])
            if names:
                return names
    except Exception:
        pass
    return DEFAULT_BOARDS


def pill(text: str, color: str, icon: Optional[str] = None) -> ft.Container:
    return ft.Container(
        content=ft.Row([
            ft.Icon(icon, size=14, color=COLORS.WHITE) if icon else ft.Container(),
            ft.Text(text, size=12, weight=ft.FontWeight.W_600, color=COLORS.WHITE),
        ], spacing=6, alignment=ft.MainAxisAlignment.CENTER),
        bgcolor=color,
        padding=ft.padding.symmetric(6, 6),
        border_radius=999,
    )


def make_board_chip(text: str, selected: bool, base_color) :
    """Return a chip-like control compatible with current Flet version.
    Tries FilterChip -> ChoiceChip -> Chip -> styled Container.
    """
    tooltip = f"/{text}/"
    # Preferred: FilterChip
    if hasattr(ft, "FilterChip"):
        return ft.FilterChip(
            text=text,
            selected=selected,
            bgcolor=WITH_OPACITY(0.1, base_color),
            selected_color=base_color,
            tooltip=tooltip,
        )
    # Next: ChoiceChip (API might be similar)
    if hasattr(ft, "ChoiceChip"):
        try:
            return ft.ChoiceChip(text=text, selected=selected, tooltip=tooltip)
        except Exception:
            pass
    # Next: Chip (non-selectable)
    if hasattr(ft, "Chip"):
        try:
            return ft.Chip(label=ft.Text(text), tooltip=tooltip, bgcolor=WITH_OPACITY(0.1, base_color))
        except Exception:
            pass
    # Fallback: simple container
    return ft.Container(
        content=ft.Text(text),
        bgcolor=WITH_OPACITY(0.1, base_color) if selected else None,
        border=ft.border.all(1, WITH_OPACITY(0.2, BORDER_BASE)),
        border_radius=16,
        padding=ft.padding.symmetric(8, 6),
        tooltip=tooltip,
    )


def section_title(title: str, icon: str, help_text: Optional[str] = None, on_help_click: Optional[Callable] = None) -> ft.Row:
    controls = [
        ft.Icon(icon, color=ACCENT_COLOR),
        ft.Text(title, size=16, weight=ft.FontWeight.BOLD),
    ]
    if help_text:
        try:
            _info_icon_name = getattr(
                ICONS,
                "INFO_OUTLINE",
                getattr(ICONS, "INFO", getattr(ICONS, "HELP_OUTLINE", getattr(ICONS, "HELP", None))),
            )
            if _info_icon_name is None:
                raise AttributeError("No suitable info icon available")
            try:
                controls.append(
                    ft.IconButton(
                        icon=_info_icon_name,
                        icon_color=WITH_OPACITY(0.8, BORDER_BASE),
                        tooltip=help_text,
                        on_click=on_help_click,
                    )
                )
            except Exception:
                # Fallback to Tooltip wrapper if IconButton not available
                info_ic = ft.Icon(_info_icon_name, size=16, color=WITH_OPACITY(0.8, BORDER_BASE))
                try:
                    controls.append(ft.Tooltip(message=help_text, content=ft.Container(content=info_ic, padding=0)))
                except Exception:
                    controls.append(ft.Container(content=info_ic, tooltip=help_text))
        except Exception:
            # Last resort: simple text with optional tooltip
            try:
                controls.append(ft.Tooltip(message=help_text, content=ft.Text("ⓘ")))
            except Exception:
                controls.append(ft.Text("ⓘ"))
    return ft.Row(controls)


def make_wrap(controls: list, spacing: int = 6, run_spacing: int = 6):
    """Return a wrapping layout compatible with current Flet version.
    Tries Wrap -> Row(wrap=True) -> Row -> Column.
    """
    # Preferred: Wrap
    if hasattr(ft, "Wrap"):
        try:
            return ft.Wrap(controls, spacing=spacing, run_spacing=run_spacing)
        except Exception:
            pass
    # Next: Row with wrap
    try:
        return ft.Row(controls, wrap=True, spacing=spacing, run_spacing=run_spacing,
                      alignment=ft.MainAxisAlignment.START)
    except TypeError:
        # Older Row without run_spacing or wrap
        try:
            return ft.Row(controls, spacing=spacing, alignment=ft.MainAxisAlignment.START)
        except Exception:
            pass
    # Fallback: Column (no wrapping)
    return ft.Column(controls, spacing=spacing)


def make_selectable_pill(label: str, selected: bool = False, base_color: Optional[str] = None, on_change=None) -> ft.Container:
    """Create a selectable pill using a Container, compatible with older Flet builds."""
    base_color = base_color or ACCENT_COLOR
    pill = ft.Container(
        content=ft.Text(label),
        bgcolor=WITH_OPACITY(0.15, base_color) if selected else None,
        border=ft.border.all(1, WITH_OPACITY(0.2, BORDER_BASE)),
        border_radius=16,
        padding=ft.padding.symmetric(8, 6),
        tooltip=f"/{label}/",
    )
    # Store state in .data
    pill.data = {"label": label, "selected": bool(selected), "base_color": base_color, "on_change": on_change}

    def toggle(_):
        d = pill.data
        d["selected"] = not d.get("selected", False)
        pill.bgcolor = WITH_OPACITY(0.15, base_color) if d["selected"] else None
        pill.update()
        cb = d.get("on_change")
        if callable(cb):
            cb()

    pill.on_click = toggle
    return pill


def make_empty_placeholder(text: str, icon) -> ft.Container:
    """Centered, subtle placeholder shown when a panel has no content."""
    return ft.Container(
        content=ft.Column(
            [
                ft.Icon(icon, color=WITH_OPACITY(0.45, BORDER_BASE), size=18),
                ft.Text(text, size=12, color=WITH_OPACITY(0.7, BORDER_BASE)),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=6,
        ),
        padding=10,
    )


def _env_truthy(val: Optional[str]) -> bool:
    """Interpret common truthy strings from environment variables."""
    try:
        return str(val).strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        return False


def apply_proxy_from_env() -> str:
    """Apply proxy settings from environment to both scrapers.

    Env vars:
    - TOR_PROXY / PROXY_URL: e.g., socks5h://127.0.0.1:9050
    - USE_ENV_PROXIES: if truthy, allow requests to use HTTP(S)_PROXY from env
    Returns a short status string for logging.
    """
    raw_proxy = os.getenv("TOR_PROXY") or os.getenv("PROXY_URL")
    raw_use_env = os.getenv("USE_ENV_PROXIES")
    use_env = _env_truthy(raw_use_env) if raw_use_env is not None else None

    # 4chan scraper
    try:
        if raw_proxy is not None and hasattr(sc, "PROXY_URL"):
            sc.PROXY_URL = raw_proxy
        if use_env is not None and hasattr(sc, "USE_ENV_PROXIES"):
            sc.USE_ENV_PROXIES = bool(use_env)
        if hasattr(sc, "apply_session_config"):
            sc.apply_session_config()
    except Exception:
        pass

    # Reddit scraper (optional)
    try:
        if rdt is not None:
            if raw_proxy is not None and hasattr(rdt, "PROXY_URL"):
                rdt.PROXY_URL = raw_proxy
            if use_env is not None and hasattr(rdt, "USE_ENV_PROXIES"):
                rdt.USE_ENV_PROXIES = bool(use_env)
            if hasattr(rdt, "apply_session_config"):
                rdt.apply_session_config()
    except Exception:
        pass

    # StackExchange scraper (optional)
    try:
        if sx is not None:
            if raw_proxy is not None and hasattr(sx, "PROXY_URL"):
                sx.PROXY_URL = raw_proxy
            if use_env is not None and hasattr(sx, "USE_ENV_PROXIES"):
                sx.USE_ENV_PROXIES = bool(use_env)
            if hasattr(sx, "apply_session_config"):
                sx.apply_session_config()
    except Exception:
        pass

    # Determine effective configuration for logging
    if use_env is True:
        return "Proxy: using environment proxies (USE_ENV_PROXIES=on)"
    if raw_proxy:
        return f"Proxy: routing via {raw_proxy}"

    # No env overrides provided; report module defaults
    try:
        eff_env = bool(getattr(sc, "USE_ENV_PROXIES", False)) or bool(getattr(rdt, "USE_ENV_PROXIES", False) if rdt is not None else False)
    except Exception:
        eff_env = False
    if eff_env:
        return "Proxy: using environment proxies (module default)"

    eff_proxy = None
    try:
        eff_proxy = getattr(rdt, "PROXY_URL", None) if rdt is not None else None
    except Exception:
        eff_proxy = None
    if not eff_proxy:
        try:
            eff_proxy = getattr(sc, "PROXY_URL", None)
        except Exception:
            eff_proxy = None
    if eff_proxy:
        return f"Proxy: routing via {eff_proxy} (module default)"
    return "Proxy: disabled (no proxy configured)"


def apply_proxy_from_ui(enabled: bool, proxy_url: Optional[str], use_env: bool) -> str:
    """Apply proxy settings based on UI controls.

    Priority:
    - If not enabled: disable all proxies for both scrapers.
    - If enabled and use_env: allow requests to use environment proxies.
    - If enabled and not use_env: route via explicit proxy_url (if provided), else disable.
    Returns a status string for logging.
    """
    try:
        # 4chan
        if hasattr(sc, "USE_ENV_PROXIES"):
            sc.USE_ENV_PROXIES = bool(use_env) if enabled else False
        if hasattr(sc, "PROXY_URL"):
            sc.PROXY_URL = (proxy_url or None) if (enabled and not use_env) else None
        if hasattr(sc, "apply_session_config"):
            sc.apply_session_config()
    except Exception:
        pass

    try:
        # Reddit (optional)
        if rdt is not None:
            if hasattr(rdt, "USE_ENV_PROXIES"):
                rdt.USE_ENV_PROXIES = bool(use_env) if enabled else False
            if hasattr(rdt, "PROXY_URL"):
                rdt.PROXY_URL = (proxy_url or None) if (enabled and not use_env) else None
            if hasattr(rdt, "apply_session_config"):
                rdt.apply_session_config()
    except Exception:
        pass

    try:
        # StackExchange
        if hasattr(sx, "USE_ENV_PROXIES"):
            sx.USE_ENV_PROXIES = bool(use_env) if enabled else False
        if hasattr(sx, "PROXY_URL"):
            sx.PROXY_URL = (proxy_url or None) if (enabled and not use_env) else None
        if hasattr(sx, "apply_session_config"):
            sx.apply_session_config()
    except Exception:
        pass

    if not enabled:
        return "Proxy: disabled via UI"
    if use_env:
        return "Proxy: using environment proxies (UI)"
    if proxy_url:
        return f"Proxy: routing via {proxy_url} (UI)"
    return "Proxy: disabled (no proxy URL provided)"


def cell_text(text: str, width: int | None = None, size: int = 13) -> ft.Text:
    """Create text for DataTable cells that wraps properly within its cell.
    Do not force width or VISIBLE overflow to avoid overlap; let the table layout size cells.
    """
    try:
        return ft.Text(
            text or "",
            no_wrap=False,
            max_lines=None,
            size=size,
        )
    except Exception:
        # Fallback if some args are unsupported
        return ft.Text(text or "")

def get_preview_col_width(page: ft.Page) -> int | None:
    """Best-effort column width for preview tables to keep columns aligned.
    Uses window width when available; returns a conservative fallback otherwise.
    """
    try:
        w = getattr(page, "window_width", None) or getattr(page, "width", None)
        if w:
            usable = max(600, int(w - 220))  # account for paddings/controls
            return max(260, int(usable / 2))
    except Exception:
        pass
    return 420

def _estimate_two_col_ratio(samples: list[tuple[str, str]]) -> float:
    """Estimate width ratio for col A (0..1) based on average content length.
    Keeps result within [0.35, 0.65] to avoid extreme skews.
    """
    if not samples:
        return 0.5
    a = sum(min(len(x or ""), 400) for x, _ in samples) / len(samples)
    b = sum(min(len(y or ""), 400) for _, y in samples) / len(samples)
    total = a + b
    if total <= 0:
        return 0.5
    r = a / total
    # Clamp
    return max(0.35, min(0.65, r))

def compute_two_col_widths(page: ft.Page, samples: list[tuple[str, str]], *,
                           total_px: int | None = None, spacing_px: int = 16,
                           min_px_each: int = 180) -> tuple[int, int]:
    """Compute two column widths that sum to available width minus spacing.
    If total_px isn't provided, derive from page width conservatively.
    """
    try:
        if total_px is None:
            w = getattr(page, "window_width", None) or getattr(page, "width", None)
            if w:
                total_px = max(600, int(w - 220))
        if total_px is None:
            total_px = 840
    except Exception:
        total_px = 840
    # Space available for both columns combined
    usable = max(2 * min_px_each + 10, total_px)
    ratio = _estimate_two_col_ratio(samples)
    w1 = max(min_px_each, int(usable * ratio) - spacing_px // 2)
    w2 = max(min_px_each, usable - spacing_px - w1)
    return (w1, w2)

def compute_two_col_flex(samples: list[tuple[str, str]]) -> tuple[int, int]:
    """Return left/right flex factors based on content ratio with sane clamps."""
    r = _estimate_two_col_ratio(samples)
    l = max(1, int(round(r * 100)))
    return l, max(1, 100 - l)

def two_col_header(left: str = "Input", right: str = "Output", *, left_flex: int = 50, right_flex: int = 50) -> ft.Container:
    hdr = ft.Row([
        ft.Container(ft.Text(left, weight=ft.FontWeight.BOLD, size=13), expand=left_flex, padding=4),
        ft.Container(ft.Text(right, weight=ft.FontWeight.BOLD, size=13), expand=right_flex, padding=4),
    ], spacing=12, alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START)
    return ft.Container(
        content=hdr,
        padding=ft.padding.only(left=6, right=6, bottom=6),
        border=ft.border.only(bottom=ft.border.BorderSide(1, WITH_OPACITY(0.12, BORDER_BASE))),
    )

def two_col_row(a: str, b: str, left_flex: int, right_flex: int) -> ft.Container:
    """Compact, uniform two-column row; cells are fixed-height and internally scrollable."""
    CELL_H = 88
    COL_SPACING = 12

    def scroll_cell(text: str) -> ft.Container:
        inner = ft.Column([
            ft.Text(text or "", no_wrap=False, max_lines=None, size=13)
        ], scroll=ft.ScrollMode.AUTO, spacing=0)
        return ft.Container(content=inner, height=CELL_H, padding=6,
                            border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
                            border_radius=6)

    row = ft.Row(
        [
            ft.Container(content=scroll_cell(a), expand=left_flex),
            ft.Container(content=scroll_cell(b), expand=right_flex),
        ],
        spacing=COL_SPACING,
        alignment=ft.MainAxisAlignment.START,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )
    return ft.Container(
        content=row,
        padding=ft.padding.symmetric(6, 6),
        border=ft.border.only(bottom=ft.border.BorderSide(1, WITH_OPACITY(0.06, BORDER_BASE))),
    )

async def simulate_scrape(page: ft.Page, log_view: ft.ListView, prog: ft.ProgressBar,
                          stats_labels: dict, preview_host: ft.ListView,
                          cancel_flag: dict) -> None:
    log = lambda m: (log_view.controls.append(ft.Text(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")),
                     page.update())

    total_steps = 120
    visited = 0
    pairs = 0
    prog.value = 0
    log("Starting scrape across selected boards...")
    page.snack_bar = ft.SnackBar(ft.Text("Scrape started 🚀"))
    page.snack_bar.open = True
    await safe_update(page)

    for step in range(1, total_steps + 1):
        if cancel_flag.get("cancelled"):
            log("Scrape cancelled by user.")
            page.snack_bar = ft.SnackBar(ft.Text("Scrape cancelled ✋"))
            page.snack_bar.open = True
            await safe_update(page)
            return
        await asyncio.sleep(0.04 + random.uniform(0, 0.02))
        prog.value = step / total_steps
        # Fake stats
        bump_threads = random.choice([0, 1])
        visited += bump_threads
        pairs += random.randint(10, 40)
        stats_labels["threads"].value = f"Threads Visited: {visited}"
        stats_labels["pairs"].value = f"Pairs Found: {pairs}"
        if step % 10 == 0:
            log(f"Visited ~{visited} threads, accumulated ~{pairs} pairs...")
        await safe_update(page)

    # Build preview rows (mock) into flex grid with scrollable cells
    preview_host.controls.clear()
    sample_pairs = [(f"What do you think about topic #{i}?", f"Here's a witty response #{i}.") for i in range(1, 11)]
    lfx, rfx = compute_two_col_flex(sample_pairs)
    preview_host.controls.append(two_col_header(left_flex=lfx, right_flex=rfx))
    for a, b in sample_pairs:
        preview_host.controls.append(two_col_row(a, b, lfx, rfx))
    log("Scrape completed successfully.")
    page.snack_bar = ft.SnackBar(ft.Text("Scrape complete! 🎉"))
    page.snack_bar.open = True
    await safe_update(page)

async def run_reddit_scrape(
    page: ft.Page,
    log_view: ft.ListView,
    prog: ft.ProgressBar,
    labels: dict,
    preview_host: ft.ListView,
    cancel_flag: dict,
    url: str,
    max_posts: int,
    delay: float,
    min_len_val: int,
    output_path: str,
    pairing_mode: str,
    ctx_k: int,
    ctx_max_chars: Optional[int],
    merge_same_id: bool,
    require_question: bool,
    ui_proxy_enabled: bool,
    ui_proxy_url: Optional[str],
    ui_use_env_proxies: bool,
) -> None:
    """Run the Reddit scraper in a worker thread and integrate results into the UI."""
    def log(msg: str):
        log_view.controls.append(ft.Text(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"))
    await safe_update(page)

    if rdt is None:
        log("Reddit scraper module not available.")
        await safe_update(page)
        return

    # Configure reddit module from UI inputs
    try:
        # Reset run-scoped counters to make stop caps reliable
        rdt.START_TS = time.time()
        rdt.REQUESTS_MADE = 0
        rdt.MAX_REQUESTS_TOTAL = None
        rdt.STOP_AFTER_SECONDS = None

        rdt.DEFAULT_URL = (url or rdt.DEFAULT_URL).strip()
        rdt.MAX_POSTS = int(max_posts)
        rdt.REQUEST_DELAY = max(0.0, float(delay))
        # Always build dataset; we'll copy pairs to desired path
        rdt.BUILD_DATASET = True
        rdt.PAIRING_MODE = "contextual" if (pairing_mode or "normal") == "contextual" else "parent_child"
        rdt.CONTEXT_K = int(ctx_k)
        rdt.MAX_INPUT_CHARS = None if (ctx_max_chars is None) else int(ctx_max_chars)
        rdt.MERGE_SAME_AUTHOR = bool(merge_same_id)
        rdt.REQUIRE_QUESTION = bool(require_question)
        rdt.MIN_LEN = max(0, int(min_len_val))
        # Keep dumps temp so we can clean after copying pairs
        rdt.OUTPUT_DIR = None
        rdt.USE_TEMP_DUMP = True
    except Exception as e:
        log(f"Invalid Reddit configuration: {e}")
        await safe_update(page)
        return

    # Apply proxy settings from UI (overrides env/defaults)
    try:
        pmsg = apply_proxy_from_ui(bool(ui_proxy_enabled), ui_proxy_url, bool(ui_use_env_proxies))
        log(pmsg)
    except Exception:
        pass

    prog.value = 0
    labels.get("threads").value = "Threads Visited: 0"
    labels.get("pairs").value = "Pairs Found: 0"
    await safe_update(page)

    log("Starting Reddit scrape...")
    await safe_update(page)

    # Kick off the blocking scraper in a background thread
    fut = asyncio.create_task(asyncio.to_thread(rdt.run))

    # A soft progress pulse and cooperative cancellation monitor
    async def pulse_and_watch():
        try:
            tick = 0.0
            while not fut.done():
                if cancel_flag.get("cancelled"):
                    # Force the worker to abort on next HTTP call
                    rdt.MAX_REQUESTS_TOTAL = 0
                    rdt.STOP_AFTER_SECONDS = 0
                # Pulse progress to indicate activity
                cur = (prog.value or 0.0)
                cur = 0.4 if cur >= 0.9 else (cur + 0.04)
                prog.value = cur
                await safe_update(page)
                await asyncio.sleep(0.35)
        except Exception:
            # Ignore UI pulse errors
            pass

    pulse_task = asyncio.create_task(pulse_and_watch())

    base_out = None
    pairs_src = None
    try:
        base_out, pairs_src = await fut
    except Exception as e:
        if cancel_flag.get("cancelled"):
            log("Scrape cancelled by user.")
            page.snack_bar = ft.SnackBar(ft.Text("Scrape cancelled ✋"))
            page.snack_bar.open = True
        else:
            log(f"Reddit scrape failed: {e}")
            page.snack_bar = ft.SnackBar(ft.Text(f"Reddit scrape failed: {e}"))
            page.snack_bar.open = True
        await safe_update(page)
        try:
            pulse_task.cancel()
        except Exception:
            pass
        return
    finally:
        try:
            pulse_task.cancel()
        except Exception:
            pass

    # Completed
    prog.value = 1.0
    await safe_update(page)

    # Copy pairs to desired output path and load for preview
    pairs_count = 0
    preview_pairs = []
    try:
        if pairs_src is not None and os.path.exists(str(pairs_src)):
            dest = output_path or "scraped_training_data.json"
            dest_abs = os.path.abspath(dest)
            os.makedirs(os.path.dirname(dest_abs) or ".", exist_ok=True)
            # Copy contents
            txt = await asyncio.to_thread(lambda: open(str(pairs_src), "r", encoding="utf-8").read())
            await asyncio.to_thread(lambda: open(dest_abs, "w", encoding="utf-8").write(txt))
            log(f"Copied pairs to: {dest_abs}")
            # Load for preview
            data = await asyncio.to_thread(lambda: json.loads(txt))
            if isinstance(data, list):
                pairs_count = len(data)
                preview_pairs = [(d.get("input", "") or "", d.get("output", "") or "") for d in data[:10]]
        else:
            log("No pairs JSON produced (pairs_src missing).")
    except Exception as e:
        log(f"Failed to copy/load pairs: {e}")
    await safe_update(page)

    # Read index.json for post count (threads label)
    try:
        idx_path = os.path.join(str(base_out), "index.json") if base_out else None
        if idx_path and os.path.exists(idx_path):
            idx = await asyncio.to_thread(lambda: json.load(open(idx_path, "r", encoding="utf-8")))
            pc = int(idx.get("post_count") or 0)
            labels.get("threads").value = f"Posts processed: {pc}"
    except Exception:
        pass

    labels.get("pairs").value = f"Pairs Found: {pairs_count}"

    # Populate preview grid
    try:
        preview_host.controls.clear()
        lfx, rfx = compute_two_col_flex(preview_pairs)
        preview_host.controls.append(two_col_header(left_flex=lfx, right_flex=rfx))
        for a, b in preview_pairs:
            preview_host.controls.append(two_col_row(a, b, lfx, rfx))
    except Exception as e:
        log(f"Failed to render preview: {e}")
    await safe_update(page)

    # Cleanup temporary dump folder
    try:
        if base_out and os.path.isdir(str(base_out)) and getattr(rdt, "USE_TEMP_DUMP", False):
            shutil.rmtree(str(base_out), ignore_errors=True)
            log(f"Cleaned up temp dump: {base_out}")
    except Exception as e:
        log(f"Cleanup warning: {e}")
    page.snack_bar = ft.SnackBar(ft.Text("Scrape complete! 🎉"))
    page.snack_bar.open = True
    await safe_update(page)


async def run_real_scrape(
    page: ft.Page,
    log_view: ft.ListView,
    prog: ft.ProgressBar,
    labels: dict,
    preview_host: ft.ListView,
    cancel_flag: dict,
    boards: List[str],
    max_threads: int,
    max_pairs_total: int,
    delay: float,
    min_len_val: int,
    output_path: str,
    pairing_mode: str,
    ctx_strategy: str,
    ctx_k: int,
    ctx_max_chars: Optional[int],
    merge_same_id: bool,
    require_question: bool,
    ui_proxy_enabled: bool,
    ui_proxy_url: Optional[str],
    ui_use_env_proxies: bool,
) -> None:
    def log(msg: str):
        log_view.controls.append(ft.Text(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"))
    await safe_update(page)

    total_boards = len(boards)
    if total_boards == 0:
        return
    prog.value = 0
    pairs_accum: List[dict] = []

    # Apply proxy settings from UI (overrides env/defaults)
    try:
        pmsg = apply_proxy_from_ui(bool(ui_proxy_enabled), ui_proxy_url, bool(ui_use_env_proxies))
        log(pmsg)
    except Exception:
        pass

    for idx, b in enumerate(boards, start=1):
        if cancel_flag.get("cancelled"):
            log("Scrape cancelled by user.")
            page.snack_bar = ft.SnackBar(ft.Text("Scrape cancelled ✋"))
            page.snack_bar.open = True
            await safe_update(page)
            return

        remaining = max_pairs_total - len(pairs_accum)
        if remaining <= 0:
            break

        log(f"Scraping /{b}/ (up to {remaining} pairs) — mode={pairing_mode}")
        try:
            data = await asyncio.to_thread(
                sc.scrape,
                board=b,
                max_threads=max_threads,
                max_pairs=remaining,
                delay=delay,
                min_len=min_len_val,
                mode=pairing_mode,
                strategy=ctx_strategy,
                k=ctx_k,
                max_chars=ctx_max_chars,
                merge_same_id=merge_same_id,
                require_question=require_question,
            )
        except Exception as e:
            log(f"Error scraping /{b}/: {e}")
            await safe_update(page)
            continue

        pairs_accum.extend(data)
        labels.get("pairs").value = f"Pairs Found: {len(pairs_accum)}"
        labels.get("threads").value = f"Boards processed: {idx}/{total_boards}"
        prog.value = idx / total_boards
        log(f"/{b}/ -> {len(data)} pairs (total {len(pairs_accum)})")
        await safe_update(page)

    # Write JSON
    try:
        await asyncio.to_thread(
            lambda: open(output_path, "w", encoding="utf-8").write(json.dumps(pairs_accum, ensure_ascii=False, indent=4))
        )
        log(f"Wrote {len(pairs_accum)} records to {output_path}")
    except Exception as e:
        log(f"Failed to write {output_path}: {e}")
        await safe_update(page)
        return

    # Populate preview with flex grid and scrollable cells
    preview_host.controls.clear()
    head = pairs_accum[:10]
    sample_pairs = [(ex.get("input", "") or "", ex.get("output", "") or "") for ex in head]
    lfx, rfx = compute_two_col_flex(sample_pairs)
    preview_host.controls.append(two_col_header(left_flex=lfx, right_flex=rfx))
    for a, b in sample_pairs:
        preview_host.controls.append(two_col_row(a, b, lfx, rfx))
    page.snack_bar = ft.SnackBar(ft.Text("Scrape complete! 🎉"))
    page.snack_bar.open = True
    await safe_update(page)

async def run_stackexchange_scrape(
    page: ft.Page,
    log_view: ft.ListView,
    prog: ft.ProgressBar,
    labels: dict,
    preview_host: ft.ListView,
    cancel_flag: dict,
    site: str,
    max_pairs: int,
    delay: float,
    min_len_val: int,
    output_path: str,
    ui_proxy_enabled: bool,
    ui_proxy_url: Optional[str],
    ui_use_env_proxies: bool,
) -> None:
    """Run the Stack Exchange scraper in a worker thread and integrate results into the UI."""
    def log(msg: str):
        log_view.controls.append(ft.Text(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"))
    await safe_update(page)

    # Apply proxy settings from UI (overrides env/defaults)
    try:
        pmsg = apply_proxy_from_ui(bool(ui_proxy_enabled), ui_proxy_url, bool(ui_use_env_proxies))
        log(pmsg)
    except Exception:
        pass

    prog.value = 0
    labels.get("threads").value = "Pages processed: 0"
    labels.get("pairs").value = "Pairs Found: 0"
    await safe_update(page)

    log(f"Starting StackExchange scrape (site={site}, max_pairs={max_pairs})...")
    await safe_update(page)

    # Kick off the blocking scraper in a background thread with cancellation support
    fut = asyncio.create_task(asyncio.to_thread(
        sx.scrape,
        site=site or "stackoverflow",
        max_pairs=int(max_pairs),
        delay=float(delay),
        min_len=int(min_len_val),
        cancel_cb=lambda: bool(cancel_flag.get("cancelled")),
    ))

    # Progress pulse and cooperative cancellation monitor
    async def pulse_and_watch():
        try:
            while not fut.done():
                # Nothing to force-cancel; cancel_cb will be polled in the worker
                cur = (prog.value or 0.0)
                cur = 0.4 if cur >= 0.9 else (cur + 0.04)
                prog.value = cur
                await safe_update(page)
                await asyncio.sleep(0.35)
        except Exception:
            # Ignore UI pulse errors
            pass

    pulse_task = asyncio.create_task(pulse_and_watch())

    try:
        results = await fut
    except Exception as e:
        if cancel_flag.get("cancelled"):
            log("Scrape cancelled by user.")
            page.snack_bar = ft.SnackBar(ft.Text("Scrape cancelled ✋"))
            page.snack_bar.open = True
        else:
            log(f"StackExchange scrape failed: {e}")
            page.snack_bar = ft.SnackBar(ft.Text(f"StackExchange scrape failed: {e}"))
            page.snack_bar.open = True
        await safe_update(page)
        try:
            pulse_task.cancel()
        except Exception:
            pass
        return
    finally:
        try:
            pulse_task.cancel()
        except Exception:
            pass

    # Completed
    prog.value = 1.0
    await safe_update(page)

    # Write JSON
    pairs_count = 0
    preview_pairs: List[Tuple[str, str]] = []
    try:
        await asyncio.to_thread(
            lambda: open(output_path, "w", encoding="utf-8").write(json.dumps(results or [], ensure_ascii=False, indent=4))
        )
        log(f"Wrote {len(results or [])} records to {output_path}")
        pairs_count = len(results or [])
        preview_pairs = [
            (d.get("input", "") or "", d.get("output", "") or "") for d in (results or [])[:10]
        ]
    except Exception as e:
        log(f"Failed to write results: {e}")
    await safe_update(page)

    labels.get("pairs").value = f"Pairs Found: {pairs_count}"
    labels.get("threads").value = f"Questions processed: {pairs_count}"

    # Populate preview grid
    try:
        preview_host.controls.clear()
        lfx, rfx = compute_two_col_flex(preview_pairs)
        preview_host.controls.append(two_col_header(left_flex=lfx, right_flex=rfx))
        for a, b in preview_pairs:
            preview_host.controls.append(two_col_row(a, b, lfx, rfx))
    except Exception as e:
        log(f"Failed to render preview: {e}")
    page.snack_bar = ft.SnackBar(ft.Text("Scrape complete! 🎉"))
    page.snack_bar.open = True
    await safe_update(page)

async def simulate_build(page: ft.Page, timeline: ft.Column, split_badges: dict,
                         split_meta: dict, cancel_flag: dict) -> None:
    def add_step(text: str, color: str, icon: str):
        timeline.controls.append(
            ft.Row([
                ft.Icon(icon, color=color),
                ft.Text(text),
            ])
        )

    add_step("Validating inputs", COLORS.BLUE, ICONS.TASK_ALT)
    await safe_update(page); await asyncio.sleep(0.4)
    if cancel_flag.get("cancelled"):
        add_step("Build cancelled by user", COLORS.RED, ICONS.CANCEL)
        await safe_update(page)
        return

    add_step("Loading and normalizing data", COLORS.BLUE, ICONS.SHUFFLE)
    await safe_update(page); await asyncio.sleep(0.6)

    add_step("Creating splits (train/val/test)", COLORS.BLUE, ICONS.CALENDAR_VIEW_MONTH)
    await safe_update(page); await asyncio.sleep(0.8)

    # Fake split sizes
    total = random.randint(5000, 9000)
    val = int(total * random.uniform(0.01, 0.1))
    test = int(total * random.uniform(0.0, 0.05))
    train = total - val - test
    split_badges["train"].content = pill(f"Train: {train}", split_meta["train"][0], split_meta["train"][1]).content
    split_badges["val"].content = pill(f"Val: {val}", split_meta["val"][0], split_meta["val"][1]).content
    split_badges["test"].content = pill(f"Test: {test}", split_meta["test"][0], split_meta["test"][1]).content
    await safe_update(page); await asyncio.sleep(0.3)

    add_step("Building dataset card (README)", COLORS.BLUE, ICONS.ARTICLE)
    await safe_update(page); await asyncio.sleep(0.5)

    add_step("Done!", COLORS.GREEN, ICONS.CHECK_CIRCLE)
    page.snack_bar = ft.SnackBar(ft.Text("Build complete ✨"))
    page.snack_bar.open = True
    await safe_update(page)


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
            "4chan Dataset Studio\n\n"
            "Mocked GUI built with Flet. Scrape 4chan (mock), build datasets (mock), and style with flair.\n"
            "No real backend calls are made in this demo."
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

    page.appbar = ft.AppBar(
        leading=ft.Icon(ICONS.DATASET_LINKED_OUTLINED),
        title=ft.Text(APP_TITLE, weight=ft.FontWeight.BOLD),
        center_title=False,
        bgcolor=WITH_OPACITY(0.03, COLORS.AMBER),
        actions=[
            ft.IconButton(ICONS.INFO_OUTLINE, tooltip="About", on_click=open_about),
            ft.IconButton(ICONS.DARK_MODE_OUTLINED, tooltip="Toggle theme", on_click=toggle_theme),
        ],
    )

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
                    page.snack_bar.open = True
                    page.update()
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
    reddit_url = ft.TextField(label="Reddit URL (subreddit or post)", value="https://www.reddit.com/r/Conservative/", width=420)
    reddit_max_posts = ft.TextField(label="Max Posts (Reddit)", value="30", width=180, keyboard_type=ft.KeyboardType.NUMBER)
    se_site = ft.TextField(label="StackExchange Site", value="stackoverflow", width=260)
    max_threads = ft.TextField(label="Max Threads", value="50", width=160, keyboard_type=ft.KeyboardType.NUMBER)
    max_pairs = ft.TextField(label="Max Pairs", value="5000", width=160, keyboard_type=ft.KeyboardType.NUMBER)
    delay = ft.TextField(label="Delay (s)", value="1.0", width=160, keyboard_type=ft.KeyboardType.NUMBER)
    min_len = ft.TextField(label="Min Length", value="3", width=160, keyboard_type=ft.KeyboardType.NUMBER)
    output_path = ft.TextField(label="Output JSON Path", value="scraped_training_data.json", width=360)

    # Contextual pairing controls
    pair_mode = ft.Dropdown(
        label="Pairing Mode",
        value="normal",
        options=[
            ft.dropdown.Option("normal"),
            ft.dropdown.Option("contextual"),
        ],
        width=200,
    )
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
        is_ctx = (pair_mode.value == "contextual")
        strategy_dd.visible = is_ctx
        k_field.visible = is_ctx
        max_chars_field.visible = is_ctx
        merge_same_id_cb.visible = is_ctx
        require_question_cb.visible = is_ctx
        page.update()
    pair_mode.on_change = lambda e: update_context_controls()
    update_context_controls()

    # Toggle visibility between 4chan, Reddit and StackExchange controls
    def update_source_controls():
        src = (source_dd.value or "").strip().lower()
        is_reddit = (src == "reddit")
        is_se = (src == "stackexchange")
        # Boards area (4chan only)
        try:
            boards_wrap.visible = not (is_reddit or is_se)
            board_actions.visible = not (is_reddit or is_se)
            board_warning.visible = not (is_reddit or is_se)
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
        for ctl in [pair_mode, strategy_dd, k_field, max_chars_field, merge_same_id_cb, require_question_cb]:
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

    # ---------- SETTINGS (Hugging Face) CONTROLS ----------
    HF_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hf_config.json")

    def _load_hf_config() -> dict:
        try:
            with open(HF_CFG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
        except Exception:
            cfg = {}
        return {"token": (cfg.get("token") or "")}

    def _save_hf_config(cfg: dict):
        try:
            with open(HF_CFG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
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
                page.snack_bar.open = True
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
    RUNPOD_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runpod_config.json")

    def _load_runpod_config() -> dict:
        try:
            with open(RUNPOD_CFG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
        except Exception:
            cfg = {}
        return {"api_key": (cfg.get("api_key") or "")}

    def _save_runpod_config(cfg: dict):
        try:
            with open(RUNPOD_CFG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
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
                page.snack_bar.open = True
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
    # Simple persistence path (project root)
    OLLAMA_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ollama_config.json")

    def _load_ollama_config() -> dict:
        try:
            with open(OLLAMA_CFG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
        except Exception:
            cfg = {}
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "base_url": (cfg.get("base_url") or "http://127.0.0.1:11434"),
            "default_model": (cfg.get("default_model") or ""),
            "selected_model": (cfg.get("selected_model") or ""),
        }

    def _save_ollama_config(cfg: dict):
        try:
            with open(OLLAMA_CFG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    _ollama_cfg = _load_ollama_config()

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

    async def _fetch_ollama_tags(base_url: str) -> dict:
        url = f"{(base_url or '').rstrip('/')}/api/tags"
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json()

    async def on_test_ollama():
        if not bool(ollama_enable_cb.value):
            return
        base = (ollama_base_url_tf.value or "http://127.0.0.1:11434").strip()
        ollama_status.value = f"Testing connection to {base}…"
        await safe_update(page)
        try:
            data = await _fetch_ollama_tags(base)
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
        _save_ollama_config(cfg)
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Ollama settings saved"))
            page.snack_bar.open = True
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
            log_placeholder.visible = len(getattr(log_list, "controls", []) or []) == 0
            preview_placeholder.visible = len(getattr(preview_host, "controls", []) or []) == 0
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
            page.snack_bar.open = True
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
        mode_val = (pair_mode.value or "normal")
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
        await safe_update(page)

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
                await run_reddit_scrape(
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
                    pairing_mode=mode_val,
                    ctx_k=k_val,
                    ctx_max_chars=max_chars_val,
                    merge_same_id=bool(merge_same_id_cb.value),
                    require_question=bool(require_question_cb.value),
                    ui_proxy_enabled=bool(proxy_enable_cb.value),
                    ui_proxy_url=(proxy_url_tf.value or "").strip(),
                    ui_use_env_proxies=bool(use_env_cb.value),
                )
            elif source_dd.value == "stackexchange":
                await run_stackexchange_scrape(
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
                )
            else:
                await run_real_scrape(
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
                    pairing_mode=mode_val,
                    ctx_strategy=strat_val,
                    ctx_k=k_val,
                    ctx_max_chars=max_chars_val,
                    merge_same_id=bool(merge_same_id_cb.value),
                    require_question=bool(require_question_cb.value),
                    ui_proxy_enabled=bool(proxy_enable_cb.value),
                    ui_proxy_url=(proxy_url_tf.value or "").strip(),
                    ui_use_env_proxies=bool(use_env_cb.value),
                )
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
        page.snack_bar.open = True
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
            page.snack_bar.open = True
            await safe_update(page)
            return

        try:
            data = await asyncio.to_thread(
                lambda: json.load(open(existing, "r", encoding="utf-8"))
            )
            if not isinstance(data, list):
                raise ValueError("Expected a JSON list of {input,output} records")
        except Exception as e:
            page.snack_bar = ft.SnackBar(ft.Text(f"Failed to open {existing}: {e}"))
            page.snack_bar.open = True
            await safe_update(page)
            return

        # Paginated flex-grid viewer to avoid heavy UI rendering for large datasets
        page_size = 100
        total = len(data)
        total_pages = max(1, (total + page_size - 1) // page_size)
        state = {"page": 0}

        grid_list = ft.ListView(expand=1, auto_scroll=False)
        info_text = ft.Text("")

        # Navigation buttons
        prev_btn = ft.TextButton("Prev")
        next_btn = ft.TextButton("Next")

        def render_page():
            start = state["page"] * page_size
            end = min(start + page_size, total)
            grid_list.controls.clear()
            # Compute dynamic flex for current page
            page_samples = [(r.get("input", "") or "", r.get("output", "") or "") for r in data[start:end]]
            lfx, rfx = compute_two_col_flex(page_samples)
            grid_list.controls.append(two_col_header(left_flex=lfx, right_flex=rfx))
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

    # Reddit params row (hidden by default)
    reddit_params_row = ft.Row([reddit_url, reddit_max_posts], wrap=True, visible=False)
    # StackExchange params row (hidden by default)
    se_params_row = ft.Row([se_site], wrap=True, visible=False)

    scrape_tab = ft.Container(
        content=ft.Column([
            section_title(
                "Source",
                ICONS.DASHBOARD,
                "Choose a data source. Options: 4chan, Reddit, StackExchange.",
                on_help_click=_mk_help_handler("Choose a data source. Options: 4chan, Reddit, StackExchange."),
            ),
            ft.Row([source_dd], wrap=True),
            section_title(
                "4chan Boards",
                ICONS.DASHBOARD,
                "Select which 4chan boards to scrape.",
                on_help_click=_mk_help_handler("Select which 4chan boards to scrape."),
            ),
            board_actions,
            boards_wrap,
            board_warning,
            ft.Divider(),
            section_title(
                "Parameters",
                ICONS.TUNE,
                "Set scraping limits and pairing behavior. Context options appear when applicable.",
                on_help_click=_mk_help_handler("Set scraping limits and pairing behavior. Context options appear when applicable."),
            ),
            reddit_params_row,
            se_params_row,
            ft.Row([max_threads, max_pairs, delay, min_len, output_path], wrap=True),
            ft.Row([pair_mode, strategy_dd, k_field, max_chars_field], wrap=True),
            ft.Row([merge_same_id_cb, require_question_cb], wrap=True),
            scrape_actions,
            ft.Container(height=10),
            section_title(
                "Progress",
                ICONS.TIMELAPSE,
                "Shows current task progress and counters.",
                on_help_click=_mk_help_handler("Shows current task progress and counters."),
            ),
            ft.Row([scrape_prog, working_ring, ft.Text("Working...")], spacing=16),
            stats_cards,
            ft.Row([threads_label, pairs_label], spacing=20),
            ft.Divider(),
            section_title(
                "Live Log",
                ICONS.TERMINAL,
                "Streaming log of scraping activity.",
                on_help_click=_mk_help_handler("Streaming log of scraping activity."),
            ),
            ft.Container(log_area, height=180, border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                         border_radius=8, padding=10),
            section_title(
                "Preview",
                ICONS.PREVIEW,
                "Quick sample preview of scraped pairs.",
                on_help_click=_mk_help_handler("Quick sample preview of scraped pairs."),
            ),
            ft.Container(preview_area, height=240, border_radius=8,
                         border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)), padding=6),
            ft.Row([
                ft.ElevatedButton(
                    "Preview Dataset", icon=ICONS.PREVIEW,
                    on_click=handle_preview_click,
                )
            ], alignment=ft.MainAxisAlignment.END),
        ], scroll=ft.ScrollMode.AUTO, spacing=12),
        padding=16,
    )


    # ---------- BUILD/PUBLISH TAB ----------
    source_mode = ft.Dropdown(
        label="Source",
        options=[
            ft.dropdown.Option("JSON file"),
            ft.dropdown.Option("Merged dataset"),
        ],
        value="JSON file",
        width=180,
    )
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
        if total >= 0.9:  # generous limit for demo
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

    # Split badges (mock values updated during build)
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

    cancel_build = {"cancelled": False}
    dd_ref = {"dd": None}
    push_state = {"inflight": False}
    push_ring = ft.ProgressRing(width=18, height=18, value=None, visible=False)

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

    def _on_toggle_custom_card(_):
        enabled = bool(use_custom_card.value)
        try:
            card_editor.disabled = not enabled
            card_preview_switch.disabled = not enabled
            if hasattr(card_preview_md, "visible"):
                card_preview_md.visible = enabled and bool(card_preview_switch.value)
        except Exception:
            pass
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

    async def _on_generate_from_dataset(_):
        # Generate using current built dataset (if available)
        dd = dd_ref.get("dd")
        if dd is None:
            page.snack_bar = ft.SnackBar(ft.Text("Build the dataset first to generate a default card."))
            page.snack_bar.open = True
            await safe_update(page)
            return
        rid = (repo_id.value or "").strip()
        if not rid:
            page.snack_bar = ft.SnackBar(ft.Text("Enter Repo ID to generate a default card."))
            page.snack_bar.open = True
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
            page.snack_bar.open = True
            await safe_update(page)

    async def _ollama_chat(base_url: str, model: str, messages: List[dict]) -> str:
        url = f"{(base_url or '').rstrip('/')}/api/chat"
        payload = {"model": model, "messages": messages, "stream": False}
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            # Ollama non-stream response typically: { ..., "message": {"role":"assistant","content":"..."} }
            msg = data.get("message") or {}
            content = msg.get("content") or data.get("content") or ""
            return str(content)

    ollama_gen_status = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))

    async def _on_generate_with_ollama(_):
        # Generate using Ollama from the selected data file (JSON list of {input,output})
        try:
            if not bool(ollama_enable_cb.value):
                page.snack_bar = ft.SnackBar(ft.Text("Enable Ollama in Settings first."))
                page.snack_bar.open = True
                await safe_update(page)
                return
        except Exception:
            pass

        cfg = _load_ollama_config()
        base_url = (cfg.get("base_url") or "http://127.0.0.1:11434").strip()
        model_name = (ollama_models_dd.value or cfg.get("selected_model") or cfg.get("default_model") or "").strip()
        if not model_name:
            page.snack_bar = ft.SnackBar(ft.Text("Select an Ollama model in Settings."))
            page.snack_bar.open = True
            await safe_update(page)
            return

        path = (data_file.value or "scraped_training_data.json").strip()
        try:
            records = await asyncio.to_thread(sd.load_records, path)
        except Exception as e:
            page.snack_bar = ft.SnackBar(ft.Text(f"Failed to load data file: {e}"))
            page.snack_bar.open = True
            await safe_update(page)
            return

        if not isinstance(records, list) or len(records) == 0:
            page.snack_bar = ft.SnackBar(ft.Text("Data file is empty or invalid (expected list of records)."))
            page.snack_bar.open = True
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
            md = await _ollama_chat(
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
            page.snack_bar.open = True
            await safe_update(page)

    load_template_btn = ft.TextButton("Load simple template", icon=ICONS.ARTICLE, on_click=_on_load_simple_template)
    gen_from_ds_btn = ft.TextButton("Generate from built dataset", icon=ICONS.BUILD, on_click=lambda e: page.run_task(_on_generate_from_dataset))
    gen_with_ollama_btn = ft.ElevatedButton("Generate with Ollama", icon=getattr(ICONS, "SMART_TOY", ICONS.HUB), on_click=lambda e: page.run_task(_on_generate_with_ollama))
    clear_card_btn = ft.TextButton("Clear", icon=ICONS.BACKSPACE, on_click=lambda e: (setattr(card_editor, "value", ""), _update_preview(), page.update()))

    def update_status_placeholder():
        try:
            timeline_placeholder.visible = len(getattr(timeline, "controls", []) or []) == 0
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
        cancel_build["cancelled"] = False
        timeline.controls.clear()
        for k in split_badges:
            label = {"train": "Train", "val": "Val", "test": "Test"}[k]
            split_badges[k].content = pill(f"{label}: 0", split_meta[k][0], split_meta[k][1]).content
        await safe_update(page)
        # Hide placeholder now that we will append steps
        try:
            timeline_placeholder.visible = False
        except Exception:
            pass

        # Parse inputs
        data_path = data_file.value or "scraped_training_data.json"
        source_val = (source_mode.value or "JSON file").strip()
        merged_path = merged_dir.value or "merged_dataset"
        try:
            seed_val = int(seed.value or 42)
        except Exception:
            seed_val = 42
        shuffle_val = bool(shuffle.value)
        try:
            val_frac = float(val_slider.value or 0.0)
        except Exception:
            val_frac = 0.0
        try:
            test_frac = float(test_slider.value or 0.0)
        except Exception:
            test_frac = 0.0
        try:
            min_len_val = int(min_len_b.value or 1)
        except Exception:
            min_len_val = 1
        out_dir = save_dir.value or "hf_dataset"
        do_push = bool(push_toggle.value)
        repo = (repo_id.value or "").strip()
        is_private = bool(private.value)
        token_val_ui = (token_val_ui.value or "").strip()
        saved_tok = ((_hf_cfg.get("token") or "").strip() if isinstance(_hf_cfg, dict) else "")
        token_val = saved_tok or token_val_ui

        def add_step(text: str, color, icon):
            timeline.controls.append(ft.Row([ft.Icon(icon, color=color), ft.Text(text)]))
            try:
                timeline_placeholder.visible = len(timeline.controls) == 0
            except Exception:
                pass

        # If using locally merged dataset, skip JSON pipeline and load from disk
        if source_val == "Merged dataset":
            add_step(f"Loading merged dataset from: {merged_path}", COLORS.BLUE, ICONS.UPLOAD_FILE)
            await safe_update(page)
            try:
                loaded = await asyncio.to_thread(load_from_disk, merged_path)
            except Exception as e:
                add_step(f"Failed loading merged dataset: {e}", COLORS.RED, ICONS.ERROR_OUTLINE)
                page.snack_bar = ft.SnackBar(ft.Text(f"Merged dataset not found or invalid at '{merged_path}'."))
                page.snack_bar.open = True
                await safe_update(page)
                return

            # Coerce to DatasetDict if needed
            try:
                is_dd = isinstance(loaded, DatasetDict)
            except Exception:
                is_dd = False
            if not is_dd:
                try:
                    # Treat as single-train dataset
                    loaded = DatasetDict({"train": loaded})
                except Exception:
                    add_step("Loaded object is not a datasets.Dataset or DatasetDict", COLORS.RED, ICONS.ERROR_OUTLINE)
                    await safe_update(page)
                    return

            dd = loaded

            # Update split badges with counts
            train_n = len(dd.get("train", []))
            val_n = len(dd.get("validation", [])) if "validation" in dd else 0
            test_n = len(dd.get("test", [])) if "test" in dd else 0
            split_badges["train"].content = pill(f"Train: {train_n}", split_meta["train"][0], split_meta["train"][1]).content
            split_badges["val"].content = pill(f"Val: {val_n}", split_meta["val"][0], split_meta["val"][1]).content
            split_badges["test"].content = pill(f"Test: {test_n}", split_meta["test"][0], split_meta["test"][1]).content
            await safe_update(page)

            # Save to disk with heartbeat
            save_text = ft.Text(f"Saving dataset to {out_dir}")
            timeline.controls.append(ft.Row([ft.Icon(ICONS.SAVE_ALT, color=COLORS.BLUE), save_text]))
            await safe_update(page)
            save_task = asyncio.create_task(asyncio.to_thread(lambda: (os.makedirs(out_dir, exist_ok=True), dd.save_to_disk(out_dir))))
            i = 0
            while not save_task.done():
                if cancel_build["cancelled"]:
                    save_text.value = "Cancel requested — waiting for current save to finish…"
                else:
                    save_text.value = f"Saving dataset{'.' * (i % 4)}"
                i += 1
                await safe_update(page)
                await asyncio.sleep(0.4)
            try:
                await save_task
            except Exception as e:
                save_text.value = f"Save failed: {e}"
                timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED), ft.Text("Save failed")]))
                await safe_update(page)
                return
            save_text.value = "Saved dataset"
            timeline.controls.append(ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text("Save complete")]))
            await safe_update(page)
            dd_ref["dd"] = dd

            # Optional push
            if do_push:
                if not repo:
                    add_step("Missing Repo ID for push", COLORS.RED, ICONS.ERROR_OUTLINE)
                    await safe_update(page)
                    return
                if not token_val:
                    try:
                        token_val = os.environ.get("HF_TOKEN") or sd.HfFolder.get_token()
                    except Exception:
                        token_val = ""
                if not token_val:
                    add_step("No HF token found — cannot push", COLORS.RED, ICONS.ERROR_OUTLINE)
                    await safe_update(page)
                    return
                add_step(f"Pushing to Hub: {repo}", COLORS.BLUE, ICONS.CLOUD_UPLOAD)
                await safe_update(page)
                try:
                    push_text = ft.Text(f"Pushing to Hub: {repo}")
                    timeline.controls.append(ft.Row([ft.Icon(ICONS.CLOUD_UPLOAD, color=COLORS.BLUE), push_text]))
                    await safe_update(page)
                    push_task = asyncio.create_task(asyncio.to_thread(sd.push_to_hub, dd, repo, is_private, token_val))
                    j = 0
                    while not push_task.done():
                        if cancel_build["cancelled"]:
                            push_text.value = "Cancel requested — waiting for current upload to finish…"
                        else:
                            push_text.value = f"Uploading{'.' * (j % 4)}"
                        j += 1
                        await safe_update(page)
                        await asyncio.sleep(0.6)
                    await push_task

                    if cancel_build["cancelled"]:
                        timeline.controls.append(ft.Row([ft.Icon(ICONS.CANCEL, color=COLORS.RED), ft.Text("Build cancelled by user")]))
                        await safe_update(page)
                        return

                    # Prepare README (custom or autogenerated) and upload
                    readme_text = ft.Text("Preparing dataset card (README)")
                    timeline.controls.append(ft.Row([ft.Icon(ICONS.ARTICLE, color=COLORS.BLUE), readme_text]))
                    await safe_update(page)
                    # Prefer custom content if enabled and non-empty
                    _custom_enabled = bool(getattr(use_custom_card, "value", False))
                    _custom_text = (getattr(card_editor, "value", "") or "").strip()
                    if _custom_enabled and _custom_text:
                        readme = _custom_text
                    else:
                        readme = await asyncio.to_thread(sd.build_dataset_card_content, dd, repo)
                    readme_text.value = "Prepared dataset card"
                    await safe_update(page)

                    up_text = ft.Text("Uploading README.md")
                    timeline.controls.append(ft.Row([ft.Icon(ICONS.ARTICLE, color=COLORS.BLUE), up_text]))
                    await safe_update(page)
                    up_task = asyncio.create_task(asyncio.to_thread(sd.upload_readme, repo, token_val, readme))
                    k = 0
                    while not up_task.done():
                        if cancel_build["cancelled"]:
                            up_text.value = "Cancel requested — waiting for current upload to finish…"
                        else:
                            up_text.value = f"Uploading README{'.' * (k % 4)}"
                        k += 1
                        await safe_update(page)
                        await asyncio.sleep(0.6)
                    await up_task

                    _url = f"https://huggingface.co/datasets/{repo}"
                    timeline.controls.append(
                        ft.Row([
                            ft.Icon(ICONS.OPEN_IN_NEW, color=COLORS.BLUE),
                            ft.TextButton("Open on Hugging Face", on_click=lambda e, u=_url: page.launch_url(u)),
                        ])
                    )
                    add_step("Push complete!", COLORS.GREEN, ICONS.CHECK_CIRCLE)
                    page.snack_bar = ft.SnackBar(ft.Text("Pushed to Hub 🚀"))
                    page.snack_bar.open = True
                    await safe_update(page)
                except Exception as e:
                    add_step(f"Push failed: {e}", COLORS.RED, ICONS.ERROR_OUTLINE)
                    await safe_update(page)

            return

        # Validate splits
        if val_frac + test_frac >= 1.0:
            add_step("Invalid split: val+test must be < 1.0", COLORS.RED, ICONS.ERROR_OUTLINE)
            page.snack_bar = ft.SnackBar(ft.Text("Invalid split: val+test must be < 1.0"))
            page.snack_bar.open = True
            await safe_update(page)
            return

        # Step 1: Load
        add_step("Loading data", COLORS.BLUE, ICONS.UPLOAD_FILE)
        await safe_update(page)
        try:
            records = await asyncio.to_thread(sd.load_records, data_path)
        except Exception as e:
            add_step(f"Failed loading data: {e}", COLORS.RED, ICONS.ERROR_OUTLINE)
            await safe_update(page)
            return
        if cancel_build["cancelled"]:
            add_step("Build cancelled by user", COLORS.RED, ICONS.CANCEL)
            await safe_update(page)
            return

        # Step 2: Normalize
        add_step("Normalizing records", COLORS.BLUE, ICONS.SHUFFLE)
        await safe_update(page)
        try:
            examples = await asyncio.to_thread(sd.normalize_records, records, min_len_val)
        except Exception as e:
            add_step(f"Normalization failed: {e}", COLORS.RED, ICONS.ERROR_OUTLINE)
            await safe_update(page)
            return
        if not examples:
            add_step("No valid examples after normalization", COLORS.RED, ICONS.ERROR_OUTLINE)
            page.snack_bar = ft.SnackBar(ft.Text("No valid examples after normalization."))
            page.snack_bar.open = True
            await safe_update(page)
            return
        if cancel_build["cancelled"]:
            add_step("Build cancelled by user", COLORS.RED, ICONS.CANCEL)
            await safe_update(page)
            return

        # Step 3: Splits
        add_step("Creating splits (train/val/test)", COLORS.BLUE, ICONS.CALENDAR_VIEW_MONTH)
        await safe_update(page)
        try:
            dd = await asyncio.to_thread(
                sd.build_dataset_dict,
                examples,
                val_frac,
                test_frac,
                shuffle_val,
                seed_val,
            )
        except Exception as e:
            add_step(f"Split creation failed: {e}", COLORS.RED, ICONS.ERROR_OUTLINE)
            await safe_update(page)
            return

        # Update split badges with real counts
        train_n = len(dd.get("train", []))
        val_n = len(dd.get("validation", [])) if "validation" in dd else 0
        test_n = len(dd.get("test", [])) if "test" in dd else 0
        split_badges["train"].content = pill(f"Train: {train_n}", split_meta["train"][0], split_meta["train"][1]).content
        split_badges["val"].content = pill(f"Val: {val_n}", split_meta["val"][0], split_meta["val"][1]).content
        split_badges["test"].content = pill(f"Test: {test_n}", split_meta["test"][0], split_meta["test"][1]).content
        await safe_update(page)
        if cancel_build["cancelled"]:
            add_step("Build cancelled by user", COLORS.RED, ICONS.CANCEL)
            await safe_update(page)
            return

        # Step 4: Save to disk (with heartbeat + cancel-aware)
        # Row that we can live-update
        save_text = ft.Text(f"Saving dataset to {out_dir}")
        timeline.controls.append(ft.Row([ft.Icon(ICONS.SAVE_ALT, color=COLORS.BLUE), save_text]))
        await safe_update(page)
        # Kick off blocking save in a worker thread and heartbeat the UI
        save_task = asyncio.create_task(asyncio.to_thread(lambda: (os.makedirs(out_dir, exist_ok=True), dd.save_to_disk(out_dir))))
        i = 0
        while not save_task.done():
            if cancel_build["cancelled"]:
                save_text.value = "Cancel requested — waiting for current save to finish…"
            else:
                save_text.value = f"Saving dataset{'.' * (i % 4)}"
            i += 1
            await safe_update(page)
            await asyncio.sleep(0.4)
        try:
            await save_task
        except Exception as e:
            save_text.value = f"Save failed: {e}"
            timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED), ft.Text("Save failed")]))
            await safe_update(page)
            return
        save_text.value = "Saved dataset"
        timeline.controls.append(ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text("Save complete")]))
        await safe_update(page)
        dd_ref["dd"] = dd
        if cancel_build["cancelled"]:
            timeline.controls.append(ft.Row([ft.Icon(ICONS.CANCEL, color=COLORS.RED), ft.Text("Build cancelled by user")]))
            await safe_update(page)
            return

        # Optional push
        if do_push:
            if not repo:
                add_step("Missing Repo ID for push", COLORS.RED, ICONS.ERROR_OUTLINE)
                await safe_update(page)
                return
            if not token_val:
                try:
                    token_val = os.environ.get("HF_TOKEN") or sd.HfFolder.get_token()
                except Exception:
                    token_val = os.environ.get("HF_TOKEN") or ""
            if not token_val:
                add_step("No HF token found — cannot push", COLORS.RED, ICONS.ERROR_OUTLINE)
                await safe_update(page)
                return
            add_step(f"Pushing to Hub: {repo}", COLORS.BLUE, ICONS.CLOUD_UPLOAD)
            await safe_update(page)
            try:
                # Push with heartbeat
                push_text = ft.Text(f"Pushing to Hub: {repo}")
                timeline.controls.append(ft.Row([ft.Icon(ICONS.CLOUD_UPLOAD, color=COLORS.BLUE), push_text]))
                await safe_update(page)
                push_task = asyncio.create_task(asyncio.to_thread(sd.push_to_hub, dd, repo, is_private, token_val))
                j = 0
                while not push_task.done():
                    if cancel_build["cancelled"]:
                        push_text.value = "Cancel requested — waiting for current upload to finish…"
                    else:
                        push_text.value = f"Uploading{'.' * (j % 4)}"
                    j += 1
                    await safe_update(page)
                    await asyncio.sleep(0.6)
                await push_task

                if cancel_build["cancelled"]:
                    timeline.controls.append(ft.Row([ft.Icon(ICONS.CANCEL, color=COLORS.RED), ft.Text("Build cancelled by user")]))
                    await safe_update(page)
                    return

                # Prepare README content (custom or autogenerated)
                readme_text = ft.Text("Preparing dataset card (README)")
                timeline.controls.append(ft.Row([ft.Icon(ICONS.ARTICLE, color=COLORS.BLUE), readme_text]))
                await safe_update(page)
                _custom_enabled = bool(getattr(use_custom_card, "value", False))
                _custom_text = (getattr(card_editor, "value", "") or "").strip()
                if _custom_enabled and _custom_text:
                    readme = _custom_text
                else:
                    readme = await asyncio.to_thread(sd.build_dataset_card_content, dd, repo)
                readme_text.value = "Prepared dataset card"
                await safe_update(page)

                # Upload README with heartbeat
                up_text = ft.Text("Uploading README.md")
                timeline.controls.append(ft.Row([ft.Icon(ICONS.ARTICLE, color=COLORS.BLUE), up_text]))
                await safe_update(page)
                up_task = asyncio.create_task(asyncio.to_thread(sd.upload_readme, repo, token_val, readme))
                k = 0
                while not up_task.done():
                    if cancel_build["cancelled"]:
                        up_text.value = "Cancel requested — waiting for current upload to finish…"
                    else:
                        up_text.value = f"Uploading README{'.' * (k % 4)}"
                    k += 1
                    await safe_update(page)
                    await asyncio.sleep(0.6)
                await up_task

                # Add link to dataset on Hub
                _url = f"https://huggingface.co/datasets/{repo}"
                timeline.controls.append(
                    ft.Row([
                        ft.Icon(ICONS.OPEN_IN_NEW, color=COLORS.BLUE),
                        ft.TextButton("Open on Hugging Face", on_click=lambda e, u=_url: page.launch_url(u)),
                    ])
                )
                add_step("Push complete!", COLORS.GREEN, ICONS.CHECK_CIRCLE)
                page.snack_bar = ft.SnackBar(ft.Text("Pushed to Hub 🚀"))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception as e:
                add_step(f"Push failed: {e}", COLORS.RED, ICONS.ERROR_OUTLINE)
                await safe_update(page)

    async def on_push_async():
        # Push using the most recently built dataset (if available)
        if push_state["inflight"]:
            return
        dd = dd_ref.get("dd")
        if dd is None:
            page.snack_bar = ft.SnackBar(ft.Text("Build the dataset first."))
            page.snack_bar.open = True
            await safe_update(page)
            return
        repo = (repo_id.value or "").strip()
        saved_tok = ((_hf_cfg.get("token") or "").strip() if isinstance(_hf_cfg, dict) else "")
        tok = (token_val_ui.value or "").strip() or saved_tok
        if not tok:
            try:
                tok = os.environ.get("HF_TOKEN") or getattr(sd.HfFolder, "get_token", lambda: "")()
            except Exception:
                tok = ""
        if not repo or not tok:
            page.snack_bar = ft.SnackBar(ft.Text("Repo ID and a valid HF token are required."))
            page.snack_bar.open = True
            await safe_update(page)
            return

        # UI: show inflight state
        push_state["inflight"] = True
        push_ring.visible = True
        # Disable the Push button while pushing
        for ctl in build_actions.controls:
            if isinstance(ctl, ft.TextButton) and "Push + Upload README" in getattr(ctl, "text", ""):
                ctl.disabled = True
        timeline.controls.append(ft.Row([ft.Icon(ICONS.CLOUD_UPLOAD, color=COLORS.BLUE), ft.Text(f"Pushing to Hub: {repo}")]))
        await safe_update(page)
        update_status_placeholder()

        try:
            await asyncio.to_thread(sd.push_to_hub, dd, repo, bool(private.value), tok)
            timeline.controls.append(ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text("Dataset pushed to Hub")]))
            await safe_update(page)

            _custom_enabled = bool(getattr(use_custom_card, "value", False))
            _custom_text = (getattr(card_editor, "value", "") or "").strip()
            if _custom_enabled and _custom_text:
                readme = _custom_text
            else:
                readme = await asyncio.to_thread(sd.build_dataset_card_content, dd, repo)
            await asyncio.to_thread(sd.upload_readme, repo, tok, readme)
            timeline.controls.append(ft.Row([ft.Icon(ICONS.ARTICLE, color=COLORS.GREEN), ft.Text("Uploaded dataset card (README)")]))
            # Add link to dataset on Hub
            _url = f"https://huggingface.co/datasets/{repo}"
            timeline.controls.append(
                ft.Row([
                    ft.Icon(ICONS.OPEN_IN_NEW, color=COLORS.BLUE),
                    ft.TextButton("Open on Hugging Face", on_click=lambda e, u=_url: page.launch_url(u)),
                ])
            )
            timeline.controls.append(ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text("Push complete!")]))
            page.snack_bar = ft.SnackBar(ft.Text("Pushed to Hub 🚀"))
            page.snack_bar.open = True
            await safe_update(page)
        except Exception as e:
            timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED), ft.Text(f"Push failed: {e}")]))
            await safe_update(page)
        finally:
            push_state["inflight"] = False
            push_ring.visible = False
            # Re-enable Push button
            for ctl in build_actions.controls:
                if isinstance(ctl, ft.TextButton) and "Push + Upload README" in getattr(ctl, "text", ""):
                    ctl.disabled = False
            update_status_placeholder()
            await safe_update(page)

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

    build_tab = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Container(
                    content=ft.Column([
                        section_title(
                            "Dataset Params",
                            ICONS.SETTINGS,
                            "Choose input source, preprocessing, and output path for building a dataset.",
                            on_help_click=_mk_help_handler("Choose input source, preprocessing, and output path for building a dataset."),
                        ),
                        ft.Row([source_mode, data_file, merged_dir, seed, shuffle, min_len_b, save_dir], wrap=True),
                        ft.Divider(),
                        section_title(
                            "Splits",
                            ICONS.TABLE_VIEW,
                            "Configure validation and test fractions; train is the remainder.",
                            on_help_click=_mk_help_handler("Configure validation and test fractions; train is the remainder."),
                        ),
                        ft.Row([
                            ft.Column([
                                ft.Text("Validation Fraction"), val_slider,
                                ft.Text("Test Fraction"), test_slider,
                                split_error,
                            ], width=360),
                            ft.Row([split_badges["train"], split_badges["val"], split_badges["test"]], spacing=10),
                        ], wrap=True, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        ft.Divider(),
                        section_title(
                            "Push to Hub",
                            ICONS.PUBLIC,
                            "Optionally upload your dataset to the Hugging Face Hub.",
                            on_help_click=_mk_help_handler("Optionally upload your dataset to the Hugging Face Hub."),
                        ),
                        ft.Row([push_toggle, repo_id, private, token_val_ui], wrap=True),
                        build_actions,
                        ft.Divider(),
                        section_title(
                            "Model Card Creator",
                            ICONS.ARTICLE,
                            "Draft and preview the README dataset card; can generate from template or dataset.",
                            on_help_click=_mk_help_handler("Draft and preview the README dataset card; can generate from template or dataset."),
                        ),
                        ft.Row([use_custom_card, card_preview_switch], wrap=True),
                        ft.Row([load_template_btn, gen_from_ds_btn, gen_with_ollama_btn, clear_card_btn], wrap=True),
                        ft.Row([ollama_gen_status], wrap=True),
                        # Wrap editor in a scrollable container so long content can be viewed
                        ft.Container(
                            ft.Column([card_editor], scroll=ft.ScrollMode.AUTO, spacing=0),
                            height=300,
                            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                            border_radius=8,
                            padding=8,
                        ),
                        # Wrap markdown preview in a scrollable container as well
                        ft.Container(
                            ft.Column([card_preview_md], scroll=ft.ScrollMode.AUTO, spacing=0),
                            height=300,
                            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                            border_radius=8,
                            padding=8,
                        ),
                        ft.Divider(),
                        section_title(
                            "Status",
                            ICONS.TASK,
                            "Build timeline with step-by-step status.",
                            on_help_click=_mk_help_handler("Build timeline with step-by-step status."),
                        ),
                        ft.Container(
                            ft.Stack([timeline, timeline_placeholder], expand=True),
                            height=260,
                            width=1200,
                            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                            border_radius=8,
                            padding=10,
                        ),
                    ], spacing=12),
                    width=1000,
                )
            ], alignment=ft.MainAxisAlignment.CENTER)
        ], scroll=ft.ScrollMode.AUTO, spacing=0),
        padding=16,
    )


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

    merge_cancel = {"cancelled": False}
    merge_busy_ring = ft.ProgressRing(width=18, height=18, value=None, visible=False)

    def update_merge_placeholders():
        try:
            merge_timeline_placeholder.visible = len(getattr(merge_timeline, "controls", []) or []) == 0
            merge_preview_placeholder.visible = len(getattr(merge_preview_host, "controls", []) or []) == 0
        except Exception:
            pass
        page.update()

    def _guess_cols(names: list[str]) -> tuple[Optional[str], Optional[str]]:
        low = {n.lower(): n for n in names}
        in_cands = [
            "input", "prompt", "question", "instruction", "source", "text", "query", "context", "post",
        ]
        out_cands = [
            "output", "response", "answer", "completion", "target", "label", "reply",
        ]
        inn = next((low[x] for x in in_cands if x in low), None)
        outn = next((low[x] for x in out_cands if x in low), None)
        # Common paired fallbacks
        if inn is None and outn is None:
            if "question" in low and "answer" in low:
                return low["question"], low["answer"]
        return inn, outn

    async def _load_and_prepare(repo: str, split: str, config: Optional[str], in_col: Optional[str], out_col: Optional[str]):
        if load_dataset is None:
            raise RuntimeError("datasets library not available — cannot load from Hub")
        # Load
        def do_load():
            if split == "all":
                dd = load_dataset(repo, name=(config or None))
                return dd
            return load_dataset(repo, split=split, name=(config or None))

        try:
            obj = await asyncio.to_thread(do_load)
        except Exception as e:
            # Handle datasets that require an explicit config
            msg = str(e).lower()
            auto_loaded = False
            if (get_dataset_config_names is not None) and ("config name is missing" in msg or "config name is required" in msg):
                try:
                    cfgs = await asyncio.to_thread(lambda: get_dataset_config_names(repo))
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
                    # Log auto-pick to timeline
                    try:
                        merge_timeline.controls.append(ft.Row([
                            ft.Icon(ICONS.INFO, color=WITH_OPACITY(0.8, ACCENT_COLOR)),
                            ft.Text(f"'{repo}' requires a config; using '{pick}' automatically"),
                        ]))
                        await safe_update(page)
                    except Exception:
                        pass
                    def do_load_cfg():
                        if split == "all":
                            return load_dataset(repo, name=pick)
                        return load_dataset(repo, split=split, name=pick)
                    obj = await asyncio.to_thread(do_load_cfg)
                    auto_loaded = True
            if not auto_loaded:
                raise

        # Normalize to list[Dataset]
        ds_list: list = []
        if isinstance(obj, dict) or (DatasetDict is not None and isinstance(obj, DatasetDict)):
            for k in ["train", "validation", "test"]:
                try:
                    if k in obj:
                        ds_list.append(obj[k])
                except Exception:
                    pass
            # Fallback to any other splits
            try:
                for k in getattr(obj, "keys", lambda: [])():
                    if k not in {"train", "validation", "test"}:
                        ds_list.append(obj[k])
            except Exception:
                pass
        else:
            ds_list = [obj]

        prepped = []
        for ds in ds_list:
            try:
                names = list(getattr(ds, "column_names", []) or [])
            except Exception:
                names = []
            # Resolve columns
            inn = (in_col or "").strip() or None
            outn = (out_col or "").strip() or None
            if not inn or inn not in names or not outn or outn not in names:
                gi, go = _guess_cols(names)
                inn = inn if (inn and inn in names) else gi
                outn = outn if (outn and outn in names) else go
            if not inn or not outn:
                raise RuntimeError(f"Could not resolve input/output columns for {repo} (have: {', '.join(names)})")

            def mapper(batch):
                # batched mapping
                src = batch.get(inn, [])
                tgt = batch.get(outn, [])
                return {
                    "input": ["" if v is None else str(v).strip() for v in src],
                    "output": ["" if v is None else str(v).strip() for v in tgt],
                }

            try:
                mapped = await asyncio.to_thread(
                    lambda: ds.map(mapper, batched=True, remove_columns=list(getattr(ds, "column_names", []) or []))
                )
            except Exception:
                # Fallback: construct from python list (may be slower)
                try:
                    to_list = [
                        {"input": "" if r.get(inn) is None else str(r.get(inn)).strip(),
                         "output": "" if r.get(outn) is None else str(r.get(outn)).strip()}
                        for r in ds
                    ]
                    mapped = await asyncio.to_thread(lambda: Dataset.from_list(to_list))
                except Exception as e:
                    raise RuntimeError(f"Failed to map columns for {repo}: {e}")

            # Optional filtering of empty rows
            try:
                mapped = await asyncio.to_thread(lambda: mapped.filter(lambda r: (len(r.get("input", "") or "") > 0 and len(r.get("output", "") or "") > 0)))
            except Exception:
                pass

            prepped.append(mapped)
            if merge_cancel.get("cancelled"):
                break
        return prepped

    async def on_merge():
        merge_cancel["cancelled"] = False
        merge_timeline.controls.clear()
        merge_preview_host.controls.clear()
        merge_busy_ring.visible = True
        update_merge_placeholders()
        await safe_update(page)

        # Validate inputs
        rows = [r for r in list(getattr(rows_host, "controls", []) or []) if isinstance(r, ft.Row)]
        entries = []
        for r in rows:
            d = getattr(r, "data", None) or {}
            src_dd = d.get("source")
            ds_tf = d.get("ds")
            sp_dd = d.get("split")
            cfg_tf = d.get("config")
            in_tf = d.get("in")
            out_tf = d.get("out")
            json_tf = d.get("json")
            src = (getattr(src_dd, "value", "Hugging Face") or "Hugging Face") if src_dd else "Hugging Face"
            repo = (getattr(ds_tf, "value", "") or "").strip() if ds_tf else ""
            json_path = (getattr(json_tf, "value", "") or "").strip() if json_tf else ""
            if (src == "Hugging Face" and repo) or (src == "JSON file" and json_path):
                entries.append({
                    "source": src,
                    "repo": repo,
                    "split": (getattr(sp_dd, "value", "train") or "train") if sp_dd else "train",
                    "config": (getattr(cfg_tf, "value", "") or "").strip() if cfg_tf else "",
                    "in": (getattr(in_tf, "value", "") or "").strip() if in_tf else "",
                    "out": (getattr(out_tf, "value", "") or "").strip() if out_tf else "",
                    "json": json_path,
                })
        if len(entries) < 2:
            merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED), ft.Text("Add at least two datasets")]))
            update_merge_placeholders(); await safe_update(page)
            merge_busy_ring.visible = False
            await safe_update(page)
            return

        out_path = merge_save_dir.value or "merged_dataset"
        op = merge_op.value or "Concatenate"
        fmt = (merge_output_format.value or "HF dataset dir").lower()
        output_json = ("json" in fmt) or (out_path.lower().endswith(".json"))
        # Auto-infer JSON output if all sources are JSON files and user didn't explicitly choose JSON
        try:
            if (not output_json) and all(ent.get("source") == "JSON file" for ent in entries):
                output_json = True
                if not out_path.lower().endswith(".json"):
                    out_path = f"{out_path}.json"
                # Reflect this choice in the UI so Preview button resolves correctly
                try:
                    merge_save_dir.value = out_path
                    if merge_output_format is not None:
                        merge_output_format.value = "JSON file"
                    update_output_controls()
                except Exception:
                    pass
                await safe_update(page)
        except Exception:
            pass

        # Load and map each dataset
        hf_prepped = []  # list[Dataset]
        json_sources: list[list[dict]] = []  # list of lists for interleave/concat
        for i, ent in enumerate(entries, start=1):
            if merge_cancel.get("cancelled"):
                break
            src = ent.get("source", "Hugging Face")
            label = ent['repo'] if src == "Hugging Face" else ent.get("json", "(json)")
            split_lbl = ent['split'] if src == "Hugging Face" else "-"
            merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.DOWNLOAD, color=COLORS.BLUE), ft.Text(f"Loading {label} [{split_lbl}]…")]))
            update_merge_placeholders(); await safe_update(page)
            try:
                if src == "Hugging Face":
                    dss = await _load_and_prepare(ent["repo"], ent["split"], ent["config"], ent["in"], ent["out"])
                    hf_prepped.extend(dss)
                    merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text(f"Prepared {ent['repo']}")]))
                else:
                    # JSON file source
                    path = ent.get("json")
                    if not path:
                        raise RuntimeError("JSON path required")
                    try:
                        records = await asyncio.to_thread(sd.load_records, path)
                    except Exception as e:
                        raise RuntimeError(f"Failed to read JSON: {e}")
                    try:
                        examples = await asyncio.to_thread(sd.normalize_records, records, 1)
                    except Exception:
                        # Fallback: minimal normalization
                        examples = []
                        for r in records or []:
                            if isinstance(r, dict):
                                a = str((r.get("input") or "")).strip()
                                b = str((r.get("output") or "")).strip()
                                if a and b:
                                    examples.append({"input": a, "output": b})
                    if output_json:
                        json_sources.append(examples)
                    else:
                        if Dataset is None:
                            raise RuntimeError("datasets library unavailable to convert JSON -> HF")
                        ds = await asyncio.to_thread(lambda: Dataset.from_list(examples))
                        hf_prepped.append(ds)
                    merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text(f"Prepared {os.path.basename(path)}")]))
            except Exception as e:
                merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED), ft.Text(f"Failed {label}: {e}")]))
                await safe_update(page)
                merge_busy_ring.visible = False
                update_merge_placeholders()
                return
            await safe_update(page)

        if merge_cancel.get("cancelled"):
            merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.CANCEL, color=COLORS.RED), ft.Text("Merge cancelled by user")]))
            merge_busy_ring.visible = False
            update_merge_placeholders(); await safe_update(page)
            return

        # Convert HF prepped to JSON if output is JSON
        if output_json and hf_prepped:
            for ds in hf_prepped:
                try:
                    exs = []
                    for rec in ds:
                        exs.append({"input": (rec.get("input", "") or ""), "output": (rec.get("output", "") or "")})
                    json_sources.append(exs)
                except Exception:
                    pass

        # Merge
        try:
            if output_json:
                if not json_sources and not hf_prepped:
                    raise RuntimeError("No datasets to merge after preparation")
                # Interleave or concatenate JSON sources
                merged_examples: list[dict] = []
                if op == "Interleave" and len(json_sources) > 1:
                    # Round-robin across sources
                    indices = [0] * len(json_sources)
                    total = sum(len(s) for s in json_sources)
                    while len(merged_examples) < total:
                        for i, s in enumerate(json_sources):
                            if indices[i] < len(s):
                                merged_examples.append(s[indices[i]])
                                indices[i] += 1
                    # no shuffle; deterministic RR
                else:
                    for s in json_sources:
                        merged_examples.extend(s)
                merged_len = len(merged_examples)
                merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.TABLE_VIEW, color=COLORS.BLUE), ft.Text(f"Merged rows: {merged_len}")]))
                await safe_update(page)
            else:
                prepped_all = hf_prepped
                if not prepped_all:
                    raise RuntimeError("No datasets to merge after preparation")
                if op == "Concatenate" or len(prepped_all) == 1:
                    merged = await asyncio.to_thread(lambda: concatenate_datasets(prepped_all))
                else:
                    if hf_interleave is not None:
                        merged = await asyncio.to_thread(lambda: hf_interleave(prepped_all, probabilities=None, seed=42))
                    else:
                        # Fallback: concatenate + shuffle (approximate interleave)
                        tmp = await asyncio.to_thread(lambda: concatenate_datasets(prepped_all))
                        try:
                            merged = await asyncio.to_thread(lambda: tmp.shuffle(seed=42))
                            merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.SHUFFLE, color=COLORS.ORANGE), ft.Text("Interleave not available — using shuffle fallback")]))
                        except Exception:
                            merged = tmp
                merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.TABLE_VIEW, color=COLORS.BLUE), ft.Text(f"Merged rows: {len(merged)}")]))
                await safe_update(page)
        except Exception as e:
            merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED), ft.Text(f"Merge failed: {e}")]))
            merge_busy_ring.visible = False
            update_merge_placeholders(); await safe_update(page)
            return

        # Save to disk
        try:
            if output_json:
                target = out_path
                # ensure dir exists if nested
                try:
                    dname = os.path.dirname(target)
                    if dname:
                        await asyncio.to_thread(lambda: os.makedirs(dname, exist_ok=True))
                except Exception:
                    pass
                merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.SAVE_ALT, color=COLORS.BLUE), ft.Text(f"Saving JSON to {target}")]))
                await safe_update(page)
                await asyncio.to_thread(lambda: open(target, "w", encoding="utf-8").write(json.dumps(merged_examples, ensure_ascii=False, indent=2)))
                merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text("Save complete")]))
                await safe_update(page)
            else:
                out_dir = out_path
                dd = DatasetDict({"train": merged}) if DatasetDict is not None else {"train": merged}
                merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.SAVE_ALT, color=COLORS.BLUE), ft.Text(f"Saving to {out_dir}")]))
                await safe_update(page)
                await asyncio.to_thread(lambda: (os.makedirs(out_dir, exist_ok=True), dd.save_to_disk(out_dir)))
                merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text("Save complete")]))
                await safe_update(page)
        except Exception as e:
            merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED), ft.Text(f"Save failed: {e}")]))
            merge_busy_ring.visible = False
            update_merge_placeholders(); await safe_update(page)
            return

        # Preview first N rows in-place
        try:
            merge_preview_host.controls.clear()
            pairs = []
            if output_json:
                head_n = min(12, len(merged_examples))
                for rec in merged_examples[:head_n]:
                    pairs.append(((rec.get("input", "") or ""), (rec.get("output", "") or "")))
            else:
                head_n = min(12, len(merged))
                idxs = list(range(head_n))
                head = await asyncio.to_thread(lambda: merged.select(idxs)) if head_n > 0 else None
                if head is not None:
                    for rec in head:
                        pairs.append((rec.get("input", "") or "", rec.get("output", "") or ""))
            lfx, rfx = compute_two_col_flex(pairs)
            merge_preview_host.controls.append(two_col_header(left_flex=lfx, right_flex=rfx))
            for a, b in pairs:
                merge_preview_host.controls.append(two_col_row(a, b, lfx, rfx))
        except Exception:
            pass

        merge_busy_ring.visible = False
        page.snack_bar = ft.SnackBar(ft.Text("Merge complete ✨"))
        page.snack_bar.open = True
        update_merge_placeholders(); await safe_update(page)

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
        update_merge_placeholders(); page.update()

    async def on_preview_merged():
        """Open a modal dialog showing the merged dataset saved to disk (DatasetDict or JSON)."""
        # Immediate feedback
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Opening merged dataset preview..."))
            page.snack_bar.open = True
            await safe_update(page)
        except Exception:
            pass

        # Resolve save dir robustly
        orig_dir = merge_save_dir.value or "merged_dataset"
        fmt_now = (merge_output_format.value or "").lower()
        wants_json = ("json" in fmt_now) or (str(orig_dir).lower().endswith(".json"))
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
        seen = set(); resolved_list = []
        for pth in candidates:
            ap = os.path.abspath(pth)
            if ap not in seen:
                seen.add(ap); resolved_list.append(ap)
        existing = next((p for p in resolved_list if os.path.exists(p)), None)
        if not existing:
            page.snack_bar = ft.SnackBar(ft.Text(
                "Merged dataset not found. Tried:\n" + "\n".join(resolved_list[:4])
            ))
            page.snack_bar.open = True
            await safe_update(page)
            return

        if wants_json:
            # Load JSON and preview
            try:
                data = await asyncio.to_thread(sd.load_records, existing)
            except Exception as e:
                page.snack_bar = ft.SnackBar(ft.Text(f"Failed to read JSON: {e}"))
                page.snack_bar.open = True
                await safe_update(page)
                return
            try:
                data = await asyncio.to_thread(sd.normalize_records, data, 1)
            except Exception:
                pass
            total = len(data or [])
            page_size = 100
            total_pages = max(1, (total + page_size - 1) // page_size)
            state = {"page": 0}

            grid_list = ft.ListView(expand=1, auto_scroll=False)
            info_text = ft.Text("")
            prev_btn = ft.TextButton("Prev")
            next_btn = ft.TextButton("Next")

            def render_page_json():
                start = state["page"] * page_size
                end = min(start + page_size, total)
                grid_list.controls.clear()
                pairs = []
                for rec in (data or [])[start:end]:
                    pairs.append(((rec.get("input", "") or ""), (rec.get("output", "") or "")))
                lfx, rfx = compute_two_col_flex(pairs)
                grid_list.controls.append(two_col_header(left_flex=lfx, right_flex=rfx))
                for a, b in pairs:
                    grid_list.controls.append(two_col_row(a, b, lfx, rfx))
                info_text.value = f"Page {state['page']+1}/{total_pages} • Showing {start+1}-{end} of {total}"
                prev_btn.disabled = state["page"] <= 0
                next_btn.disabled = state["page"] >= (total_pages - 1)
                page.update()

            def on_prev_json(_):
                if state["page"] > 0:
                    state["page"] -= 1
                    render_page_json()

            def on_next_json(_):
                if state["page"] < (total_pages - 1):
                    state["page"] += 1
                    render_page_json()

            prev_btn.on_click = on_prev_json
            next_btn.on_click = on_next_json

            controls_bar = ft.Row([
                prev_btn,
                next_btn,
                info_text,
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

            dlg = ft.AlertDialog(
                modal=True,
                title=ft.Text(f"Merged Dataset Viewer — {total} rows"),
                content=ft.Container(
                    width=900,
                    height=600,
                    content=ft.Column([
                        controls_bar,
                        ft.Container(grid_list, expand=True),
                    ], expand=True),
                ),
                actions=[],
            )

            def close_dlg_json(_):
                dlg.open = False
                page.update()

            dlg.actions = [ft.TextButton("Close", on_click=close_dlg_json)]
            try:
                dlg.on_dismiss = lambda e: page.update()
            except Exception:
                pass
            render_page_json()
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
            return

        if load_from_disk is None:
            page.snack_bar = ft.SnackBar(ft.Text("datasets.load_from_disk unavailable — cannot open preview"))
            page.snack_bar.open = True
            await safe_update(page)
            return

        # Load dataset from disk
        try:
            obj = await asyncio.to_thread(lambda: load_from_disk(existing))
        except Exception as e:
            page.snack_bar = ft.SnackBar(ft.Text(f"Failed to load dataset from {existing}: {e}"))
            page.snack_bar.open = True
            await safe_update(page)
            return

        # Extract a Dataset
        ds = None
        try:
            if DatasetDict is not None and isinstance(obj, DatasetDict):
                for k in ["train", "validation", "test"]:
                    if k in obj:
                        ds = obj[k]; break
                if ds is None:
                    # fallback: any split
                    for k in getattr(obj, "keys", lambda: [])():
                        ds = obj[k]; break
            else:
                ds = obj
        except Exception:
            ds = obj
        if ds is None:
            page.snack_bar = ft.SnackBar(ft.Text("No split found to preview"))
            page.snack_bar.open = True
            await safe_update(page)
            return

        # Pagination state
        try:
            total = len(ds)
        except Exception:
            try:
                total = int(getattr(ds, "num_rows", 0))
            except Exception:
                total = 0
        page_size = 100
        total_pages = max(1, (total + page_size - 1) // page_size)
        state = {"page": 0}

        grid_list = ft.ListView(expand=1, auto_scroll=False)
        info_text = ft.Text("")
        prev_btn = ft.TextButton("Prev")
        next_btn = ft.TextButton("Next")

        def render_page():
            start = state["page"] * page_size
            end = min(start + page_size, total)
            grid_list.controls.clear()
            # Fetch a small slice efficiently
            try:
                idxs = list(range(start, end))
                page_ds = ds.select(idxs)
            except Exception:
                page_ds = None
            pairs = []
            try:
                if page_ds is not None:
                    for rec in page_ds:
                        pairs.append((rec.get("input", "") or "", rec.get("output", "") or ""))
            except Exception:
                pass

            lfx, rfx = compute_two_col_flex(pairs)
            grid_list.controls.append(two_col_header(left_flex=lfx, right_flex=rfx))
            for a, b in pairs:
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

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Merged Dataset Viewer — {total} rows"),
            content=ft.Container(
                width=900,
                height=600,
                content=ft.Column([
                    controls_bar,
                    ft.Container(grid_list, expand=True),
                ], expand=True),
            ),
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
        render_page()
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

    def handle_merge_preview_click(_):
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Opening merged dataset preview..."))
            page.snack_bar.open = True
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

    merge_actions = ft.Row([
        ft.ElevatedButton("Merge Datasets", icon=ICONS.TABLE_VIEW, on_click=lambda e: page.run_task(on_merge)),
        ft.OutlinedButton("Cancel", icon=ICONS.CANCEL, on_click=on_cancel_merge),
        ft.TextButton("Refresh", icon=REFRESH_ICON, on_click=on_refresh_merge),
        ft.TextButton("Preview Merged", icon=ICONS.PREVIEW, on_click=handle_merge_preview_click),
        merge_busy_ring,
    ], spacing=10)

    merge_tab = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Container(
                    content=ft.Column([
                        section_title(
                            "Merge Datasets",
                            ICONS.TABLE_VIEW,
                            "Combine datasets and unify columns into input/output schema.",
                            on_help_click=_mk_help_handler("Combine datasets and unify columns into input/output schema."),
                        ),
                        ft.Text("Combine multiple datasets (Hugging Face or local JSON). Map columns to a unified input/output schema and merge.", size=12, color=WITH_OPACITY(0.7, BORDER_BASE)),
                        ft.Divider(),
                        section_title(
                            "Operation",
                            ICONS.SHUFFLE,
                            "Choose how to merge rows (e.g., concatenate).",
                            on_help_click=_mk_help_handler("Choose how to merge rows (e.g., concatenate)."),
                        ),
                        ft.Row([merge_op], wrap=True),
                        ft.Divider(),
                        section_title(
                            "Datasets",
                            ICONS.TABLE_VIEW,
                            "Add datasets from HF or local JSON and map columns.",
                            on_help_click=_mk_help_handler("Add datasets from HF or local JSON and map columns."),
                        ),
                        ft.Row([add_row_btn, clear_btn], spacing=8),
                        rows_host,
                        ft.Divider(),
                        section_title(
                            "Output",
                            ICONS.SAVE_ALT,
                            "Set output format and save directory.",
                            on_help_click=_mk_help_handler("Set output format and save directory."),
                        ),
                        ft.Row([merge_output_format, merge_save_dir], wrap=True),
                        merge_actions,
                        ft.Divider(),
                        section_title(
                            "Preview",
                            ICONS.PREVIEW,
                            "Shows a sample of the merged result.",
                            on_help_click=_mk_help_handler("Shows a sample of the merged result."),
                        ),
                        ft.Container(ft.Stack([merge_preview_host, merge_preview_placeholder], expand=True),
                                     height=220,
                                     width=1000,
                                     border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                                     border_radius=8,
                                     padding=6,
                        ),
                        ft.Divider(),
                        section_title(
                            "Status",
                            ICONS.TASK,
                            "Merge timeline and diagnostics.",
                            on_help_click=_mk_help_handler("Merge timeline and diagnostics."),
                        ),
                        ft.Container(
                            ft.Stack([merge_timeline, merge_timeline_placeholder], expand=True),
                            height=200,
                            width=1000,
                            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                            border_radius=8,
                            padding=10,
                        ),
                    ], spacing=12),
                    width=1000,
                )
            ], alignment=ft.MainAxisAlignment.CENTER)
        ], scroll=ft.ScrollMode.AUTO, spacing=0),
        padding=16,
    )


    # ---------- TRAINING TAB (Mock UI) ----------
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
        [config_files_dd, config_refresh_btn, load_config_btn, config_edit_btn, config_rename_btn, config_delete_btn],
        wrap=True,
    )

    def _saved_configs_dir() -> str:
        root = os.path.dirname(os.path.dirname(__file__))
        d = os.path.join(root, "saved_configs")
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
        return d

    def _list_saved_configs() -> List[str]:
        d = _saved_configs_dir()
        try:
            files = [f for f in os.listdir(d) if f.lower().endswith(".json")]
        except Exception:
            files = []
        return sorted(files)

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
        try:
            files = _list_saved_configs()
            config_files_dd.options = [ft.dropdown.Option(f) for f in files]
            if files and not config_files_dd.value:
                config_files_dd.value = files[0]
            if not files:
                config_files_dd.value = None
        except Exception:
            pass
        _update_config_buttons_enabled()

    def _read_json_file(path: str) -> Optional[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _validate_config(conf: dict) -> Tuple[bool, str]:
        """Lightweight schema validation for saved training configs.
        Returns (ok, message). On ok=False, message explains the issue.
        """
        if not isinstance(conf, dict):
            return False, "Config is not a JSON object."
        hp = conf.get("hp")
        if not isinstance(hp, dict):
            return False, "Missing or invalid 'hp' section."
        # Minimal required fields for a viable training run
        required = ["base_model", "epochs", "lr", "bsz", "grad_accum", "max_steps", "output_dir"]
        missing = [k for k in required if not str(hp.get(k) or "").strip()]
        if missing:
            return False, f"Missing required hp fields: {', '.join(missing)}"
        # Dataset hint (optional but helpful)
        if not (hp.get("hf_dataset_id") or hp.get("json_path")):
            return True, "Note: no dataset specified (hf_dataset_id/json_path)."
        return True, "OK"

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
            rp_tb_cb.value = bool(iu.get("tensorboard", rp_tb_cb.value))
            rp_ssh_cb.value = bool(iu.get("ssh", rp_ssh_cb.value))
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
    train_state = {"running": False, "pod_id": None, "infra": None, "api_key": "", "loaded_config": None}

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
        # Set beginner defaults (depend on beginner mode)
        if is_beginner:
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
        """Build train.py flags. Uses underscore keys matching script flags.
        Only emits a safe whitelist of flags.
        """
        src = train_source.value or "Hugging Face"
        repo = (train_hf_repo.value or "").strip()
        split = (train_hf_split.value or "train").strip()
        cfg = (train_hf_config.value or "").strip()
        jpath = (train_json_path.value or "").strip()
        model = (base_model.value or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit").strip()
        out_dir = (out_dir_tf.value or "/data/outputs/runpod_run").strip()
        # Core hparams
        hp: dict = {
            "base_model": model,
            "epochs": (epochs_tf.value or "3").strip(),
            "lr": (lr_tf.value or "2e-4").strip(),
            "bsz": (batch_tf.value or "2").strip(),
            "grad_accum": (grad_acc_tf.value or "4").strip(),
            "max_steps": (max_steps_tf.value or "200").strip(),
            "use_lora": bool(getattr(use_lora_cb, "value", False)),
            "output_dir": out_dir,
        }
        # Dataset flags
        if (src == "Hugging Face") and repo:
            hp["hf_dataset_id"] = repo
            hp["hf_dataset_split"] = split
        elif jpath:
            hp["json_path"] = jpath
        # Add optional toggles
        if bool(getattr(packing_cb, "value", False)):
            hp["packing"] = True
        if bool(getattr(auto_resume_cb, "value", False)):
            hp["auto_resume"] = True
        if bool(getattr(push_cb, "value", False)):
            hp["push"] = True
        _hf_repo_id = (hf_repo_id_tf.value or "").strip()
        if _hf_repo_id:
            hp["hf_repo_id"] = _hf_repo_id
        _resume_from = (resume_from_tf.value or "").strip()
        if _resume_from:
            hp["resume_from"] = _resume_from
        return hp

    async def on_start_training():
        if train_state.get("running"):
            return
        cancel_train["cancelled"] = False
        train_state["running"] = True

        # Ensure infra and API key
        infra = train_state.get("infra") or {}
        tpl_id = ((infra.get("template") or {}).get("id") or "").strip()
        vol_id = ((infra.get("volume") or {}).get("id") or "").strip()
        if not tpl_id or not vol_id:
            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.WARNING, color=COLORS.RED), ft.Text("Runpod infrastructure not ready. Click Ensure Infrastructure first.")]))
            train_state["running"] = False
            # (removed: was incorrectly enabling restart/open before pod exists)
            update_train_placeholders(); await safe_update(page)
            return
        saved_key = ((train_state.get("api_key") or (_runpod_cfg.get("api_key") if isinstance(_runpod_cfg, dict) else "") or "").strip())
        temp_key = (rp_temp_key_tf.value or "").strip()
        api_key = saved_key or temp_key or (os.environ.get("RUNPOD_API_KEY") or "").strip()
        if not api_key:
            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.WARNING, color=COLORS.RED), ft.Text("Runpod API key missing. Set it in Settings or temp field.")]))
            train_state["running"] = False
            update_train_placeholders(); await safe_update(page)
            return

        # Persist key and set running UI state
        try:
            train_state["api_key"] = api_key
        except Exception:
            pass
        try:
            start_train_btn.disabled = True
            start_train_btn.visible = False
            stop_train_btn.disabled = False
            refresh_train_btn.disabled = False
        except Exception:
            pass
        # Reset log de-duplication buffer for this session
        train_state["log_seen"] = set()
        # Initialize progress display
        try:
            train_progress.value = 0.0
            train_prog_label.value = "Starting..."
            train_state["progress"] = 0.0
        except Exception:
            pass
        await safe_update(page)

        # Build flags for train.py (honor Configuration mode if a config is loaded)
        using_loaded_hp = False
        hp = None
        try:
            mode_val = (config_mode_dd.value or "Normal").lower()
            if mode_val.startswith("config"):
                cfg = train_state.get("loaded_config") or {}
                cfg_hp = cfg.get("hp") if isinstance(cfg, dict) else None
                if isinstance(cfg_hp, dict) and cfg_hp:
                    hp = dict(cfg_hp)
                    using_loaded_hp = True
        except Exception:
            pass
        if not isinstance(hp, dict) or not hp:
            hp = _build_hp()

        # Beginner mode presets: choose GPU & adjust params
        level = (skill_level.value or "Beginner").lower()
        is_beginner = (level == "beginner")
        beginner_mode = (beginner_mode_dd.value or "Fastest").lower() if is_beginner else ""

        chosen_gpu_type_id = "AUTO"
        chosen_interruptible = False

        if is_beginner and (not using_loaded_hp):
            if beginner_mode == "fastest":
                # Aggressive for speed: rely on best GPU (secure), larger per-device batch, minimal GA
                try:
                    hp["epochs"] = (epochs_tf.value or "1").strip() or "1"
                    hp["lr"] = "2e-4"
                    hp["bsz"] = "4"
                    hp["grad_accum"] = "1"
                    hp["max_steps"] = (max_steps_tf.value or "200").strip() or "200"
                except Exception:
                    pass
                # Keep AUTO + secure (non-interruptible)
                chosen_gpu_type_id = "AUTO"
                chosen_interruptible = False
            elif beginner_mode == "cheapest":
                # Conservative defaults already set; pick the lowest-cost GPU (spot preferred)
                dc_id = (((infra.get("volume") or {}).get("dc") or "").strip()) or "US-NC-1"
                try:
                    cheapest_gpu, is_spot = await asyncio.to_thread(rp_pod.discover_cheapest_gpu, api_key, dc_id, 1)
                    chosen_gpu_type_id = cheapest_gpu
                    chosen_interruptible = bool(is_spot)
                except Exception as e:
                    train_timeline.controls.append(ft.Row([ft.Icon(ICONS.WARNING, color=COLORS.RED), ft.Text(f"Cheapest GPU discovery failed, using AUTO: {e}")]))

        # Logs
        # Log dataset choice
        src = train_source.value or "Hugging Face"
        repo = (train_hf_repo.value or "").strip()
        split = (train_hf_split.value or "train").strip()
        jpath = (train_json_path.value or "").strip()
        model = base_model.value or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
        ds_desc = f"HF: {repo} [{split}]" if (src == "Hugging Face") else (f"JSON: {jpath}" if jpath else "JSON: (unset)")
        train_timeline.controls.append(ft.Row([ft.Icon(getattr(ICONS, "SCIENCE", ICONS.PLAY_CIRCLE), color=ACCENT_COLOR), ft.Text("Creating Runpod pod and starting training…")]))
        train_timeline.controls.append(ft.Row([ft.Icon(ICONS.TABLE_VIEW, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"Dataset: {ds_desc}")]))
        train_timeline.controls.append(ft.Row([ft.Icon(ICONS.SETTINGS, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"Model={model} • Epochs={hp.get('epochs')} • LR={hp.get('lr')} • BSZ={hp.get('bsz')} • GA={hp.get('grad_accum')}")]))
        if is_beginner:
            try:
                if beginner_mode == "fastest":
                    bm_text = "Beginner: Fastest — using best GPU (secure) with aggressive params"
                else:
                    bm_text = f"Beginner: Cheapest — selecting lowest-cost GPU ({'spot' if chosen_interruptible else 'secure'}) with conservative params"
                train_timeline.controls.append(ft.Row([ft.Icon(ICONS.SETTINGS, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(bm_text)]))
            except Exception:
                pass
        update_train_placeholders(); await safe_update(page)

        # Create pod
        try:
            def _mk_pod():
                return rp_pod.create_pod(
                    api_key=api_key,
                    template_id=tpl_id,
                    volume_id=vol_id,
                    pod_name=f"ds-train-{int(time.time())}",
                    hp=hp,
                    gpu_type_id=chosen_gpu_type_id,
                    gpu_count=1,
                    interruptible=chosen_interruptible
                )
            pod = await asyncio.to_thread(_mk_pod)
            pod_id = pod.get("id") or pod.get("podId") or pod.get("pod_id")
            train_state["pod_id"] = pod_id
            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.CLOUD, color=ACCENT_COLOR), ft.Text(f"Pod created: {pod_id}")]))
            # Enable actions now that a pod exists
            # Refresh teardown UI to expose Pod checkbox
            try:
                await _refresh_teardown_ui()
            except Exception:
                pass
            try:
                restart_container_btn.disabled = False
                open_runpod_btn.disabled = False
                open_web_terminal_btn.disabled = False
                copy_ssh_btn.disabled = False
            except Exception:
                pass
            # Offer to save setup immediately after successful start (always)
            try:
                def _collect_infra_ui_state() -> dict:
                    return {
                        "dc": (rp_dc_tf.value or "US-NC-1"),
                        "vol_name": (rp_vol_name_tf.value or "unsloth-volume"),
                        "vol_size": int(float(rp_vol_size_tf.value or "50")),
                        "resize_if_smaller": bool(getattr(rp_resize_cb, "value", True)),
                        "tpl_name": (rp_tpl_name_tf.value or "unsloth-trainer-template"),
                        "image": (rp_image_tf.value or ""),
                        "container_disk": int(float(rp_container_disk_tf.value or "30")),
                        "pod_volume_gb": int(float(rp_volume_in_gb_tf.value or "0")),
                        "mount_path": (rp_mount_path_tf.value or "/data"),
                        "category": (rp_category_tf.value or "NVIDIA"),
                        "public": bool(getattr(rp_public_cb, "value", False)),
                        "tensorboard": bool(getattr(rp_tb_cb, "value", False)),
                        "ssh": bool(getattr(rp_ssh_cb, "value", False)),
                    }
                payload = {
                    "version": 1,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "hp": hp,
                    "infra_ui": _collect_infra_ui_state(),
                    "infra_ids": train_state.get("infra") or {},
                    "meta": {
                        "skill_level": skill_level.value,
                        "beginner_mode": beginner_mode_dd.value if (skill_level.value or "") == "Beginner" else "",
                    },
                }
                default_name = f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(hp.get('base_model','model')).replace('/', '_')}.json"
                name_tf = ft.TextField(label="Save as", value=default_name, width=420)

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
                # Open dialog using same pattern as dataset previews
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
                # Yield once to ensure modal renders before entering the polling loop
                try:
                    await asyncio.sleep(0.05)
                except Exception:
                    pass
            except Exception as ex:
                try:
                    train_timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR, color=COLORS.RED), ft.Text(f"Save dialog error: {ex}")]))
                    update_train_placeholders(); await safe_update(page)
                except Exception:
                    pass
            # Log exact GPU type selected
            try:
                gpu_id = None
                # Try fields from create response
                gpu_id = pod.get("gpuTypeId") or pod.get("gpu_type_id")
                if not gpu_id:
                    gids = pod.get("gpuTypeIds") or pod.get("gpu_type_ids")
                    if isinstance(gids, list) and gids:
                        gpu_id = gids[0]
                # Fallback: fetch pod details
                if not gpu_id and pod_id:
                    try:
                        info = await asyncio.to_thread(rp_pod.get_pod, api_key, pod_id)
                    except Exception:
                        info = None
                    if isinstance(info, dict):
                        gpu_id = info.get("gpuTypeId") or info.get("gpu_type_id")
                        if not gpu_id:
                            gids2 = info.get("gpuTypeIds") or info.get("gpu_type_ids")
                            if isinstance(gids2, list) and gids2:
                                gpu_id = gids2[0]
                # Determine spot/secure
                is_spot = None
                cloud_type = (pod.get("cloudType") or pod.get("cloud_type") or "").upper()
                if cloud_type:
                    is_spot = (cloud_type == "COMMUNITY")
                else:
                    val = pod.get("interruptible")
                    if val is None:
                        val = pod.get("isInterruptible") or pod.get("spot")
                    is_spot = bool(val) if val is not None else bool(chosen_interruptible)
                # Fallback display if still unknown
                if not gpu_id:
                    gpu_id = chosen_gpu_type_id if chosen_gpu_type_id != "AUTO" else "(auto-best)"
                txt = f"GPU type selected: {gpu_id} • {'spot' if is_spot else 'secure'}"
                train_timeline.controls.append(ft.Row([ft.Icon(getattr(ICONS, 'MEMORY', getattr(ICONS, 'COMPUTER', ICONS.SETTINGS)), color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(txt)]))
            except Exception:
                pass
            update_train_placeholders(); await safe_update(page)
        except Exception as e:
            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR, color=COLORS.RED), ft.Text(f"Pod create failed: {e}")]))
            train_state["running"] = False
            await safe_update(page)
            return

        # Poll status
        try:
            last_state = None
            while True:
                if cancel_train.get("cancelled"):
                    try:
                        await asyncio.to_thread(rp_pod.delete_pod, api_key, train_state.get("pod_id"))
                        train_timeline.controls.append(ft.Row([ft.Icon(ICONS.CANCEL, color=COLORS.RED), ft.Text("Cancel requested — pod termination sent")]))
                    except Exception as ex:
                        train_timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR, color=COLORS.RED), ft.Text(f"Failed to terminate pod: {ex}")]))
                    break
                pod = await asyncio.to_thread(rp_pod.get_pod, api_key, train_state.get("pod_id"))
                state = (rp_pod.state_of(pod) or "").upper()
                if state != last_state:
                    train_timeline.controls.append(ft.Row([ft.Icon(ICONS.TASK_ALT, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"Pod state: {state}")]))
                    last_state = state
                if state in rp_pod.TERMINAL_STATES:
                    # Set progress to 100% on terminal states and optionally auto-terminate
                    try:
                        train_progress.value = 1.0
                        train_prog_label.value = "Progress: 100%"
                        train_state["progress"] = 1.0
                    except Exception:
                        pass
                    try:
                        await safe_update(page)
                    except Exception:
                        pass
                    if bool(getattr(auto_terminate_cb, "value", False)):
                        try:
                            await asyncio.to_thread(rp_pod.delete_pod, api_key, train_state.get("pod_id"))
                            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.DELETE_FOREVER, color=COLORS.RED), ft.Text("Auto-terminate enabled — pod deleted after training finished")]))
                        except Exception as ex:
                            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR, color=COLORS.RED), ft.Text(f"Failed to auto-delete pod: {ex}")]))
                    break
                # Live log streaming: fetch recent logs and append only new lines
                try:
                    lines = await asyncio.to_thread(rp_pod.get_pod_logs, api_key, train_state.get("pod_id"), 200)
                except Exception:
                    lines = []
                seen = train_state.get("log_seen") or set()
                new_lines = []
                for ln in (lines or []):
                    s = str(ln)
                    if s not in seen:
                        new_lines.append(s)
                        seen.add(s)
                        if len(seen) > 5000:
                            # shrink set to recent size to avoid unbounded growth
                            seen = set(list(seen)[-2000:])
                train_state["log_seen"] = seen
                if new_lines:
                    for s in new_lines:
                        _log_icon = getattr(ICONS, "ARTICLE", getattr(ICONS, "TERMINAL", getattr(ICONS, "DESCRIPTION", ICONS.TASK_ALT)))
                        train_timeline.controls.append(ft.Row([ft.Icon(_log_icon, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(s)]))
                    update_train_placeholders()
                    try:
                        # keep UI list from growing unbounded
                        if len(train_timeline.controls) > 1200:
                            train_timeline.controls[:] = train_timeline.controls[-900:]
                    except Exception:
                        pass
                    try:
                        _update_progress_from_logs(new_lines)
                    except Exception:
                        pass
                await safe_update(page)
                await asyncio.sleep(3.0)

            # Final state
            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text(f"Training finished with state: {last_state}")]))
        except Exception as e:
            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR, color=COLORS.RED), ft.Text(f"Polling error: {e}")]))
        finally:
            train_state["running"] = False
            try:
                start_train_btn.visible = True
                start_train_btn.disabled = False
                stop_train_btn.disabled = True
                refresh_train_btn.disabled = False
                restart_container_btn.disabled = True
                open_runpod_btn.disabled = True
                open_web_terminal_btn.disabled = True
                copy_ssh_btn.disabled = True
            except Exception:
                pass
            await safe_update(page)

    async def on_restart_container():
        try:
            pod_id = (train_state.get("pod_id") or "").strip()
            if not pod_id:
                return
            api_key = (train_state.get("api_key") or "").strip() or (os.environ.get("RUNPOD_API_KEY") or "").strip()
            if not api_key:
                return
            # Honor Configuration mode if a config is loaded
            hp = None
            try:
                mode_val = (config_mode_dd.value or "Normal").lower()
                if mode_val.startswith("config"):
                    cfg = train_state.get("loaded_config") or {}
                    cfg_hp = cfg.get("hp") if isinstance(cfg, dict) else None
                    if isinstance(cfg_hp, dict) and cfg_hp:
                        hp = dict(cfg_hp)
            except Exception:
                pass
            if not isinstance(hp, dict) or not hp:
                hp = _build_hp()
            cmd = rp_pod.build_cmd(hp)
            await asyncio.to_thread(rp_pod.patch_pod_docker_start_cmd, api_key, pod_id, cmd)
            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.RESTART_ALT, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text("Container restarting with new hyper-params…")]))
            update_train_placeholders(); await safe_update(page)
        except Exception as e:
            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR, color=COLORS.RED), ft.Text(f"Restart failed: {e}")]))
            update_train_placeholders(); await safe_update(page)

    def on_open_runpod(_):
        try:
            pod_id = (train_state.get("pod_id") or "").strip()
            if not pod_id:
                return
            url = f"https://www.runpod.io/console/pods/{pod_id}"
            try:
                page.launch_url(url)
            except Exception:
                # Fallback: log the URL
                train_timeline.controls.append(ft.Row([ft.Icon(ICONS.OPEN_IN_NEW, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(url)]))
                update_train_placeholders(); page.update()
        except Exception:
            pass

    def on_open_web_terminal(_):
        """Open the Runpod console for this pod; from there click Connect → Open Web Terminal."""
        try:
            pod_id = (train_state.get("pod_id") or "").strip()
            if not pod_id:
                return
            url = f"https://console.runpod.io/pods/{pod_id}"
            try:
                page.launch_url(url)
            except Exception:
                train_timeline.controls.append(ft.Row([ft.Icon(ICONS.OPEN_IN_NEW, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(url)]))
                update_train_placeholders(); page.update()
        except Exception:
            pass

    async def on_copy_ssh_command(_):
        """Copy an SSH command for this pod to clipboard.
        Prefers public IP + port 22 mapping if available; otherwise falls back to proxy ssh.runpod.io.
        """
        pod_id = (train_state.get("pod_id") or "").strip()
        if not pod_id:
            return
        api_key = (train_state.get("api_key") or os.environ.get("RUNPOD_API_KEY") or "").strip()
        cmd = None
        try:
            info = await asyncio.to_thread(rp_pod.get_pod, api_key, pod_id)
            port_map = info.get("portMappings") or {}
            public_ip = info.get("publicIp") or ""
            ssh_port = None
            if isinstance(port_map, dict):
                ssh_port = port_map.get("22")
            if public_ip and ssh_port:
                cmd = f"ssh root@{public_ip} -p {ssh_port} -i ~/.ssh/id_ed25519"
        except Exception:
            cmd = None
        if cmd:
            try:
                page.set_clipboard(cmd)
                page.snack_bar = ft.SnackBar(ft.Text("SSH command copied to clipboard"))
                page.snack_bar.open = True
                await safe_update(page)
                return
            except Exception:
                # Fallback: print into timeline
                train_timeline.controls.append(ft.Row([ft.Icon(getattr(ICONS, "CONTENT_COPY", ICONS.LINK), color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(cmd)]))
                update_train_placeholders(); await safe_update(page)
                return
        # If we couldn't compute a direct SSH command, open the console SSH tab for accurate proxy command
        url = f"https://console.runpod.io/pods/{pod_id}"
        try:
            page.launch_url(url)
        except Exception:
            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.OPEN_IN_NEW, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(url)]))
        page.snack_bar = ft.SnackBar(ft.Text("Open the pod → Connect → SSH tab to copy the proxy command."))
        page.snack_bar.open = True
        await safe_update(page)

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
    rp_tb_cb = ft.Checkbox(label="Expose TensorBoard (6006)", value=False)
    rp_ssh_cb = ft.Checkbox(label="Expose SSH (22/tcp)", value=False, tooltip="Adds 22/tcp to template ports; requires SSH server in the container.")

    # Info icon for "Public template": clarifies visibility and considerations
    rp_public_info = ft.IconButton(
        icon=getattr(ICONS, "INFO_OUTLINE", getattr(ICONS, "INFO", getattr(ICONS, "HELP_OUTLINE", None))),
        tooltip="Make this template visible to all Runpod users. Be mindful of sensitive env vars.",
        on_click=_mk_help_handler(
            "Public templates are discoverable by other Runpod users. Others can launch pods using this template. If your image is private or requires registry auth, they will need access to run it. Avoid putting sensitive environment variables in the template."
        ),
    )
    rp_public_row = ft.Row([rp_public_cb, rp_public_info], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    rp_infra_busy = ft.ProgressRing(visible=False)
    rp_temp_key_tf = ft.TextField(
        label="API key (optional, temp)",
        password=True,
        can_reveal_password=True,
        width=320,
        tooltip="Temporary key used only here. If a key is saved in Settings, that one takes precedence.",
    )

    async def on_ensure_infra():
        print("EnsureInfra: on_ensure_infra started")
        # Resolve API key: Settings > temp (this tab) > env
        saved_key = ((_runpod_cfg.get("api_key") or "") if isinstance(_runpod_cfg, dict) else "").strip()
        temp_key = (rp_temp_key_tf.value or "").strip()
        key = saved_key or temp_key or (os.environ.get("RUNPOD_API_KEY") or "").strip()
        if not key:
            try:
                train_timeline.controls.append(ft.Row([ft.Icon(ICONS.WARNING, color=COLORS.RED), ft.Text("Runpod API key missing. Set it in Settings → Runpod API Access.")]))
                update_train_placeholders(); await safe_update(page)
            except Exception:
                pass
            return

        # Read params with defaults
        dc = (rp_dc_tf.value or "US-NC-1").strip()
        vol_name = (rp_vol_name_tf.value or "unsloth-volume").strip()
        vol_size_s = (rp_vol_size_tf.value or "50").strip()
        resize = bool(getattr(rp_resize_cb, "value", True))
        tpl_name = (rp_tpl_name_tf.value or "unsloth-trainer-template").strip()
        image = (rp_image_tf.value or "docker.io/sbussiso/unsloth-trainer:latest").strip()
        container_disk_s = (rp_container_disk_tf.value or "30").strip()
        vol_in_gb_s = (rp_volume_in_gb_tf.value or "0").strip()
        mount_path = (rp_mount_path_tf.value or "/data").strip()
        category = (rp_category_tf.value or "NVIDIA").strip()
        is_public = bool(getattr(rp_public_cb, "value", False))
        tb_expose = bool(getattr(rp_tb_cb, "value", False))
        ssh_expose = bool(getattr(rp_ssh_cb, "value", False))

        # Coerce numbers safely
        try:
            vol_size = int(float(vol_size_s))
        except Exception:
            vol_size = 50
        try:
            container_disk = int(float(container_disk_s))
        except Exception:
            container_disk = 30
        try:
            vol_in_gb = int(float(vol_in_gb_s))
        except Exception:
            vol_in_gb = 0

        # Log start
        train_timeline.controls.append(ft.Row([ft.Icon(ICONS.CLOUD, color=ACCENT_COLOR), ft.Text("Ensuring Runpod infrastructure (volume + template)…")]))
        train_timeline.controls.append(ft.Row([ft.Icon(ICONS.SETTINGS, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"DC={dc} • Vol={vol_name} ({vol_size}GB) • Tpl={tpl_name}")]))
        update_train_placeholders(); await safe_update(page)

        # Busy indicator
        try:
            rp_infra_busy.visible = True
            await safe_update(page)
        except Exception:
            pass

        # Call infra helper
        try:
            # Build env vars for the template (inject HF token if available)
            hf_tok = ""
            try:
                hf_tok = (((_hf_cfg.get("token") or "") if isinstance(_hf_cfg, dict) else "").strip())
            except Exception:
                hf_tok = ""
            if not hf_tok:
                try:
                    hf_tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
                except Exception:
                    hf_tok = ""
            if (not hf_tok) and (HfFolder is not None):
                try:
                    hf_tok = getattr(HfFolder, "get_token", lambda: "")() or ""
                except Exception:
                    pass

            tpl_env = {"PYTHONUNBUFFERED": "1"}
            if hf_tok:
                tpl_env["HF_TOKEN"] = hf_tok
                tpl_env["HUGGINGFACE_HUB_TOKEN"] = hf_tok
                # Warn if template is public (exposes env vars)
                if is_public:
                    try:
                        train_timeline.controls.append(ft.Row([
                            ft.Icon(ICONS.WARNING, color=COLORS.ORANGE),
                            ft.Text("Template is Public — environment variables (including HF token) may be visible to others."),
                        ]))
                        update_train_placeholders(); await safe_update(page)
                    except Exception:
                        pass

            def do_call():
                return rp_infra.ensure_infrastructure(
                    api_key=key,
                    datacenter_id=dc,
                    volume_name=vol_name,
                    volume_size_gb=vol_size,
                    resize_if_smaller=resize,
                    template_name=tpl_name,
                    image_name=image,
                    container_disk_gb=container_disk,
                    volume_in_gb=vol_in_gb,
                    volume_mount_path=mount_path,
                    category=category,
                    is_public=is_public,
                    env_vars=tpl_env,
                    ports=( ["6006/http"] if tb_expose else [] ) + ( ["22/tcp"] if ssh_expose else [] ),
                )
            result = await asyncio.to_thread(do_call)
            vol = result.get("volume", {})
            tpl = result.get("template", {})
            # Persist infra and key for training handlers
            try:
                train_state["infra"] = result
                train_state["api_key"] = key
            except Exception:
                pass
            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.DONE_ALL, color=COLORS.GREEN), ft.Text(f"Volume {vol.get('action')} — id={vol.get('id')} size={vol.get('size')}GB")]))
            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.DONE_ALL, color=COLORS.GREEN), ft.Text(f"Template {tpl.get('action')} — id={tpl.get('id')} image={tpl.get('image')}")]))
            # Refresh teardown UI to show new infra immediately
            try:
                await _refresh_teardown_ui()
            except Exception:
                pass
            # Beautiful success message (snack + dialog)
            try:
                success_title = "Runpod infrastructure ready!"
                vol_line = f"Volume {vol.get('action')} • {vol.get('name')} ({vol.get('size')}GB) • DC {vol.get('dc')}"
                tpl_line = f"Template {tpl.get('action')} • {tpl.get('name')} • {tpl.get('image')}"

                # Quick celebratory snackbar
                page.snack_bar = ft.SnackBar(
                    ft.Row([
                        ft.Icon(getattr(ICONS, "CHECK_CIRCLE", getattr(ICONS, "CHECK", getattr(ICONS, "DONE", None))), color=COLORS.GREEN),
                        ft.Text(success_title),
                    ])
                )
                page.snack_bar.open = True
                await safe_update(page)

                # Detailed dialog
                dlg = ft.AlertDialog(
                    modal=True,
                    title=ft.Row([
                        ft.Icon(getattr(ICONS, "CLOUD_DONE", getattr(ICONS, "CLOUD", ICONS.SETTINGS)), color=COLORS.GREEN),
                        ft.Text("Infrastructure ready"),
                    ], alignment=ft.MainAxisAlignment.START),
                    content=ft.Column([
                        ft.Row([ft.Icon(getattr(ICONS, "DONE_ALL", getattr(ICONS, "DONE", getattr(ICONS, "CHECK", None))), color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(vol_line)]),
                        ft.Row([ft.Icon(getattr(ICONS, "DONE_ALL", getattr(ICONS, "DONE", getattr(ICONS, "CHECK", None))), color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(tpl_line)]),
                    ], tight=True, spacing=6),
                    actions=[ft.TextButton("Great!", on_click=lambda e: (setattr(dlg, "open", False), page.update()))],
                )
                page.dialog = dlg
                dlg.open = True
                await safe_update(page)
                try:
                    dataset_section.visible = True
                    train_params_section.visible = True
                    # Enable training controls once infra is ready
                    start_train_btn.disabled = False
                    stop_train_btn.disabled = False
                    refresh_train_btn.disabled = False
                    # Respect Configuration mode visibility
                    _update_mode_visibility()
                except Exception:
                    pass
                await safe_update(page)
            except Exception:
                pass
        except Exception as e:
            msg = str(e)
            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR, color=COLORS.RED), ft.Text(f"Infra setup failed: {msg}")]))
        finally:
            try:
                rp_infra_busy.visible = False
            except Exception:
                pass
            await safe_update(page)

    def on_click_ensure_infra(e):
        # Immediate feedback to confirm click registered and show activity
        try:
            print("EnsureInfra: click handler fired")
            page.snack_bar = ft.SnackBar(ft.Text("Ensuring Runpod infrastructure…"))
            page.snack_bar.open = True
            rp_infra_busy.visible = True
            update_train_placeholders(); page.update()
        except Exception:
            pass
        schedule_task(on_ensure_infra)

    rp_infra_actions = ft.Row([
        ft.ElevatedButton("Ensure Infrastructure", icon=getattr(ICONS, "CLOUD_DONE", getattr(ICONS, "CLOUD", ICONS.SETTINGS)), on_click=on_click_ensure_infra),
        rp_infra_busy,
    ], spacing=10)

    # Compact infra action for Configuration mode (button only)
    rp_infra_compact_row = ft.Row([
        ft.OutlinedButton(
            "Ensure Infrastructure",
            icon=getattr(ICONS, "CLOUD_DONE", getattr(ICONS, "CLOUD", ICONS.SETTINGS)),
            on_click=on_click_ensure_infra,
        ),
    ], spacing=10)

    # Configuration section wrapper to place in Training tab
    config_section = ft.Container(
        content=ft.Column([
            section_title(
                "Configuration",
                getattr(ICONS, "SETTINGS_SUGGEST", ICONS.SETTINGS),
                "Save or load training configs to streamline repeated runs.",
                on_help_click=_mk_help_handler("Save or load training configs to streamline repeated runs."),
            ),
            ft.Row([config_mode_dd], wrap=True),
            config_files_row,
            config_summary_txt,
            rp_infra_compact_row,
            ft.Divider(),
        ], spacing=8),
        visible=True,
    )

    # Group all Runpod infrastructure controls to toggle as one section
    rp_infra_panel = ft.Container(
        content=ft.Column([
            section_title(
                "Runpod Infrastructure",
                getattr(ICONS, "CLOUD", ICONS.SETTINGS),
                "Create or update the required Runpod Network Volume and Template before training.",
                on_help_click=_mk_help_handler("Create or update the required Runpod Network Volume and Template before training."),
            ),
            ft.Text("Defaults are provided; change any value to customize. Key precedence: Settings > Training temp field > environment.", size=12, color=WITH_OPACITY(0.7, BORDER_BASE)),
            ft.Row([rp_dc_tf, rp_vol_name_tf, rp_vol_size_tf, rp_resize_row], wrap=True),
            ft.Row([rp_tpl_name_tf, rp_image_tf], wrap=True),
            ft.Row([rp_container_disk_tf, rp_volume_in_gb_tf, rp_mount_path_tf], wrap=True),
            ft.Row([rp_category_tf, rp_public_row, rp_tb_cb, rp_ssh_cb], wrap=True),
            ft.Row([rp_temp_key_tf], wrap=True),
            rp_infra_actions,
        ], spacing=12),
        visible=True,
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
    train_actions = ft.Row([
        start_train_btn,
        stop_train_btn,
        refresh_train_btn,
        restart_container_btn,
        open_runpod_btn,
        open_web_terminal_btn,
        copy_ssh_btn,
        auto_terminate_cb,
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
        try:
            infra = train_state.get("infra") or {}
        except Exception:
            infra = {}
        tpl = infra.get("template") or {}
        vol = infra.get("volume") or {}
        tpl_id = (str(tpl.get("id") or "").strip())
        vol_id = (str(vol.get("id") or "").strip())
        pod_id = str(train_state.get("pod_id") or "").strip()

        # Reconcile pod presence with Runpod (handles external deletions)
        try:
            key = (((_runpod_cfg.get("api_key") or "") if isinstance(_runpod_cfg, dict) else "").strip()) or (rp_temp_key_tf.value or "").strip() or (os.environ.get("RUNPOD_API_KEY") or "").strip()
        except Exception:
            key = (os.environ.get("RUNPOD_API_KEY") or "").strip()
        if pod_id and key:
            try:
                await asyncio.to_thread(rp_pod.get_pod, key, pod_id)
            except Exception as ex:
                status = None
                try:
                    status = getattr(getattr(ex, "response", None), "status_code", None)
                except Exception:
                    status = None
                if status == 404:
                    # Pod no longer exists; clear and inform timeline
                    try:
                        train_timeline.controls.append(ft.Row([ft.Icon(ICONS.INFO, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"Reconciled: Pod {pod_id} is already deleted on Runpod.")]))
                    except Exception:
                        pass
                    try:
                        train_state["pod_id"] = None
                    except Exception:
                        pass
                    pod_id = ""

        # Template checkbox
        try:
            if tpl_id:
                name = tpl.get("name") or tpl_id
                img = tpl.get("image") or ""
                td_template_cb.label = f"Template: {name} (id={tpl_id}){f' • {img}' if img else ''}"
                td_template_cb.visible = True
            else:
                td_template_cb.visible = False
                td_template_cb.value = False
        except Exception:
            pass

        # Volume checkbox
        try:
            if vol_id:
                vname = vol.get("name") or vol_id
                vsize = vol.get("size") or "?"
                vdc = vol.get("dc") or vol.get("dataCenterId") or ""
                td_volume_cb.label = f"Volume: {vname} ({vsize}GB){f' • DC {vdc}' if vdc else ''} (id={vol_id})"
                td_volume_cb.visible = True
            else:
                td_volume_cb.visible = False
                td_volume_cb.value = False
        except Exception:
            pass

        # Pod checkbox (optional)
        try:
            if pod_id:
                td_pod_cb.label = f"Pod: {pod_id}"
                td_pod_cb.visible = True
            else:
                td_pod_cb.visible = False
                td_pod_cb.value = False
        except Exception:
            pass

        try:
            teardown_section.visible = bool(tpl_id or vol_id or pod_id)
            await safe_update(page)
        except Exception:
            pass

    async def _do_teardown(selected_all: bool = False):
        # Resolve API key with same precedence: Settings > temp > env
        saved_key = (((_runpod_cfg.get("api_key") or "") if isinstance(_runpod_cfg, dict) else "").strip())
        temp_key = (rp_temp_key_tf.value or "").strip()
        key = saved_key or temp_key or (os.environ.get("RUNPOD_API_KEY") or "").strip()
        if not key:
            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.WARNING, color=COLORS.RED), ft.Text("Runpod API key missing. Set it in Settings → Runpod API Access.")]))
            update_train_placeholders(); await safe_update(page)
            return

        infra = train_state.get("infra") or {}
        tpl_id = str(((infra.get("template") or {}).get("id") or "")).strip()
        vol_id = str(((infra.get("volume") or {}).get("id") or "")).strip()
        pod_id = str(train_state.get("pod_id") or "").strip()

        # Reconcile pod existence before proceeding (avoid trying to delete a non-existent pod)
        if pod_id:
            try:
                await asyncio.to_thread(rp_pod.get_pod, key, pod_id)
            except Exception as ex:
                status = getattr(getattr(ex, "response", None), "status_code", None)
                if status == 404:
                    try:
                        train_timeline.controls.append(ft.Row([ft.Icon(ICONS.INFO, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"Pod already absent on Runpod: {pod_id}")]))
                        update_train_placeholders(); await safe_update(page)
                    except Exception:
                        pass
                    try:
                        train_state["pod_id"] = None
                    except Exception:
                        pass
                    pod_id = ""

        sel_tpl = bool(td_template_cb.value) or (selected_all and bool(tpl_id))
        sel_vol = bool(td_volume_cb.value) or (selected_all and bool(vol_id))
        sel_pod = bool(td_pod_cb.value) or (selected_all and bool(pod_id))

        if not (sel_tpl or sel_vol or sel_pod):
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Select at least one item to teardown."))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return

        # Busy on
        try:
            td_busy.visible = True
            await safe_update(page)
        except Exception:
            pass

        # Add timeline log to indicate start and selected targets
        try:
            actions = []
            if sel_pod and pod_id:
                actions.append(f"Pod {pod_id}")
            if sel_tpl and tpl_id:
                actions.append(f"Template {tpl_id}")
            if sel_vol and vol_id:
                actions.append(f"Volume {vol_id}")
            if actions:
                train_timeline.controls.append(ft.Row([
                    ft.Icon(getattr(ICONS, "PLAY_CIRCLE", ICONS.PLAY_ARROW), color=ACCENT_COLOR),
                    ft.Text("Starting teardown: " + ", ".join(actions))
                ]))
                update_train_placeholders(); await safe_update(page)
        except Exception:
            pass

        # Perform deletions. Order: Pod → Template → Volume
        try:
            if sel_pod and pod_id:
                try:
                    # Log intent
                    try:
                        train_timeline.controls.append(ft.Row([ft.Icon(getattr(ICONS, "CLOUD_OFF", getattr(ICONS, "CLOUD", ICONS.CLOSE)), color=WITH_OPACITY(0.9, COLORS.RED)), ft.Text(f"Deleting Pod: {pod_id}...")]))
                        await safe_update(page)
                    except Exception:
                        pass
                    await asyncio.to_thread(rp_pod.delete_pod, key, pod_id)
                    train_timeline.controls.append(ft.Row([ft.Icon(ICONS.DELETE_FOREVER, color=COLORS.RED), ft.Text(f"Pod deleted: {pod_id}")]))
                    try:
                        train_state["pod_id"] = None
                    except Exception:
                        pass
                except Exception as ex:
                    status = getattr(getattr(ex, "response", None), "status_code", None)
                    if status == 404:
                        # Already deleted; treat as success
                        train_timeline.controls.append(ft.Row([ft.Icon(ICONS.INFO, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"Pod already deleted: {pod_id}")]))
                        try:
                            train_state["pod_id"] = None
                        except Exception:
                            pass
                    else:
                        train_timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR, color=COLORS.RED), ft.Text(f"Failed to delete pod: {ex}")]))

            if sel_tpl and tpl_id:
                try:
                    # Log intent
                    try:
                        train_timeline.controls.append(ft.Row([ft.Icon(getattr(ICONS, "DESCRIPTION", ICONS.ARTICLE), color=WITH_OPACITY(0.9, COLORS.RED)), ft.Text(f"Deleting Template: {tpl_id}...")]))
                        await safe_update(page)
                    except Exception:
                        pass
                    await asyncio.to_thread(rp_infra.delete_template, tpl_id, key)
                    train_timeline.controls.append(ft.Row([ft.Icon(ICONS.DELETE_FOREVER, color=COLORS.RED), ft.Text(f"Template deleted: {tpl_id}")]))
                    try:
                        if isinstance(train_state.get("infra"), dict):
                            train_state["infra"]["template"] = {}
                    except Exception:
                        pass
                except Exception as ex:
                    train_timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR, color=COLORS.RED), ft.Text(f"Failed to delete template: {ex}")]))

            if sel_vol and vol_id:
                try:
                    # Log intent
                    try:
                        train_timeline.controls.append(ft.Row([ft.Icon(getattr(ICONS, "STORAGE", ICONS.SAVE), color=WITH_OPACITY(0.9, COLORS.RED)), ft.Text(f"Deleting Volume: {vol_id}...")]))
                        await safe_update(page)
                    except Exception:
                        pass
                    await asyncio.to_thread(rp_infra.delete_volume, vol_id, key)
                    train_timeline.controls.append(ft.Row([ft.Icon(ICONS.DELETE_FOREVER, color=COLORS.RED), ft.Text(f"Volume deleted: {vol_id}")]))
                    try:
                        if isinstance(train_state.get("infra"), dict):
                            train_state["infra"]["volume"] = {}
                    except Exception:
                        pass
                except Exception as ex:
                    train_timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR, color=COLORS.RED), ft.Text(f"Failed to delete volume: {ex}")]))

            # If both infra items are gone, clear infra
            try:
                infra = train_state.get("infra") or {}
                tpl_id2 = str(((infra.get("template") or {}).get("id") or "")).strip()
                vol_id2 = str(((infra.get("volume") or {}).get("id") or "")).strip()
                if not tpl_id2 and not vol_id2:
                    train_state["infra"] = None
            except Exception:
                pass

            # Disable training actions if infra missing
            try:
                has_infra = bool(train_state.get("infra"))
                start_train_btn.disabled = not has_infra
                stop_train_btn.disabled = not has_infra
                refresh_train_btn.disabled = not has_infra
                restart_container_btn.disabled = True
                open_runpod_btn.disabled = True
                open_web_terminal_btn.disabled = True
                copy_ssh_btn.disabled = True
            except Exception:
                pass
            # Completion log and refresh UI
            try:
                train_timeline.controls.append(ft.Row([ft.Icon(getattr(ICONS, "CHECK_CIRCLE", ICONS.CHECK), color=COLORS.GREEN), ft.Text("Teardown complete")]))
            except Exception:
                pass
            update_train_placeholders(); await _refresh_teardown_ui(); await safe_update(page)
        finally:
            try:
                td_busy.visible = False
                await safe_update(page)
            except Exception:
                pass

    async def on_teardown_selected(_=None):
        # Immediate feedback (timeline + snackbar)
        try:
            train_timeline.controls.append(ft.Row([ft.Icon(getattr(ICONS, "WARNING_AMBER", ICONS.WARNING), color=WITH_OPACITY(0.9, COLORS.ORANGE)), ft.Text("Teardown Selected clicked")]))
            update_train_placeholders(); await safe_update(page)
        except Exception:
            pass
        # Immediate feedback
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Preparing teardown confirmation..."))
            page.snack_bar.open = True
            await safe_update(page)
        except Exception:
            pass
        # Build confirmation dialog text
        items = []
        if bool(td_pod_cb.value):
            items.append("Pod")
        if bool(td_template_cb.value):
            items.append("Template")
        if bool(td_volume_cb.value):
            items.append("Volume")
        if not items:
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Select at least one item to teardown."))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return
        msg = "This will delete: " + ", ".join(items) + "."

        confirm_dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(getattr(ICONS, "DELETE_FOREVER", getattr(ICONS, "DELETE", ICONS.CLOSE)), color=COLORS.RED),
                ft.Text("Confirm Teardown"),
            ], alignment=ft.MainAxisAlignment.START),
            content=ft.Text(msg),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: (setattr(confirm_dlg, "open", False), page.update())),
                ft.ElevatedButton("Delete", icon=getattr(ICONS, "CHECK", ICONS.DELETE), on_click=lambda e: (setattr(confirm_dlg, "open", False), page.run_task(_do_teardown, False))),
            ],
        )
        try:
            confirm_dlg.on_dismiss = lambda e: page.update()
        except Exception:
            pass
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

    async def on_teardown_all(_=None):
        # Immediate feedback (timeline + snackbar)
        try:
            train_timeline.controls.append(ft.Row([ft.Icon(getattr(ICONS, "WARNING_AMBER", ICONS.WARNING), color=WITH_OPACITY(0.9, COLORS.ORANGE)), ft.Text("Teardown All clicked")]))
            update_train_placeholders(); await safe_update(page)
        except Exception:
            pass
        # Immediate feedback
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Preparing teardown confirmation..."))
            page.snack_bar.open = True
            await safe_update(page)
        except Exception:
            pass
        confirm_dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(getattr(ICONS, "DELETE_FOREVER", getattr(ICONS, "DELETE", ICONS.CLOSE)), color=COLORS.RED),
                ft.Text("Teardown All infrastructure?"),
            ], alignment=ft.MainAxisAlignment.START),
            content=ft.Text("This will delete the Runpod Template and Network Volume. If a pod exists, it will be deleted first."),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: (setattr(confirm_dlg, "open", False), page.update())),
                ft.ElevatedButton("Delete All", icon=getattr(ICONS, "CHECK", ICONS.DELETE), on_click=lambda e: (setattr(confirm_dlg, "open", False), page.run_task(_do_teardown, True))),
            ],
        )
        try:
            confirm_dlg.on_dismiss = lambda e: page.update()
        except Exception:
            pass
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

    teardown_section = ft.Container(
        content=ft.Column([
            td_title,
            ft.Text("Select items to teardown.", size=12, color=WITH_OPACITY(0.7, BORDER_BASE)),
            ft.Container(
                content=ft.Column([
                    td_pod_cb,
                    td_template_cb,
                    td_volume_cb,
                ], spacing=6),
                padding=8,
                border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                border_radius=8,
            ),
            ft.Row([
                ft.ElevatedButton("Teardown Selected", icon=getattr(ICONS, "DELETE", getattr(ICONS, "DELETE_OUTLINE", ICONS.CLOSE)), on_click=lambda e: page.run_task(on_teardown_selected)),
                ft.OutlinedButton("Teardown All", icon=getattr(ICONS, "DELETE_FOREVER", getattr(ICONS, "DELETE", ICONS.CLOSE)), on_click=lambda e: page.run_task(on_teardown_all)),
                td_busy,
            ], spacing=10),
        ], spacing=8),
        visible=False,
    )

    # Sections hidden until infrastructure is ensured successfully
    dataset_section = ft.Container(
        content=ft.Column([
            section_title(
                "Dataset",
                ICONS.TABLE_VIEW,
                "Select the dataset for training.",
                on_help_click=_mk_help_handler("Select the dataset for training."),
            ),
            ft.Row([train_source, train_hf_repo, train_hf_split, train_hf_config, train_json_path], wrap=True),
            ft.Divider(),
        ], spacing=0),
        visible=False,
    )

    train_params_section = ft.Container(
        content=ft.Column([
            section_title(
                "Training Params",
                ICONS.SETTINGS,
                "Basic hyperparameters and LoRA toggle for training.",
                on_help_click=_mk_help_handler("Basic hyperparameters and LoRA toggle for training."),
            ),
            ft.Row([skill_level, beginner_mode_dd], wrap=True),
            ft.Row([base_model, epochs_tf, lr_tf, batch_tf, grad_acc_tf, max_steps_tf, use_lora_cb, out_dir_tf], wrap=True),
            ft.Row([packing_row, auto_resume_row, push_row, hf_repo_row, resume_from_row], wrap=True),
            advanced_params_section,
            ft.Divider(),
        ], spacing=0),
        visible=False,
    )

    # Update visibility based on mode selection
    def _update_mode_visibility(_=None):
        mode = (config_mode_dd.value or "Normal").lower()
        is_cfg = mode.startswith("config")
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
            rp_infra_panel.visible = (not is_cfg)
            rp_infra_compact_row.visible = is_cfg
        except Exception:
            pass
        try:
            page.update()
        except Exception:
            pass

    # Hook handlers and initialize config list
    config_mode_dd.on_change = _update_mode_visibility
    config_files_dd.on_change = _update_config_buttons_enabled
    config_refresh_btn.on_click = _refresh_config_list
    load_config_btn.on_click = lambda e: page.run_task(on_load_config)
    config_edit_btn.on_click = lambda e: page.run_task(on_edit_config)
    config_rename_btn.on_click = lambda e: page.run_task(on_rename_config)
    config_delete_btn.on_click = lambda e: page.run_task(on_delete_config)
    _refresh_config_list()
    # Ensure initial visibility matches the selected mode
    _update_mode_visibility()

    training_tab = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Container(
                    content=ft.Column([
                        config_section,
                        rp_infra_panel,
                        ft.Divider(),
                        dataset_section,
                        train_params_section,
                        section_title(
                            "Progress & Logs",
                            ICONS.TASK_ALT,
                            "Pod status updates and training logs.",
                            on_help_click=_mk_help_handler("Pod status updates and training logs."),
                        ),
                        ft.Row([train_progress, train_prog_label], spacing=12),
                        ft.Container(
                            ft.Stack([train_timeline, train_timeline_placeholder], expand=True),
                            height=240,
                            width=1000,
                            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                            border_radius=8,
                            padding=10,
                        ),
                        teardown_section,
                        train_actions,
                    ], spacing=12),
                    width=1000,
                )
            ], alignment=ft.MainAxisAlignment.CENTER)
        ], scroll=ft.ScrollMode.AUTO, spacing=0),
        padding=16,
    )

    # ---------- SETTINGS TAB (Proxy config + Ollama) ----------
    settings_tab = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Container(
                    content=ft.Column([
                        section_title(
                            "Proxy Settings",
                            ICONS.SETTINGS,
                            "Override network proxy for requests. Use system env or custom URL.",
                            on_help_click=_mk_help_handler("Override network proxy for requests. Use system env or custom URL."),
                        ),
                        ft.Text(
                            "Configure how network requests route. When enabled, UI settings override environment variables and defaults.",
                            size=12,
                            color=WITH_OPACITY(0.7, BORDER_BASE),
                        ),
                        ft.Divider(),
                        ft.Row([proxy_enable_cb], wrap=True),
                        ft.Row([use_env_cb, proxy_url_tf], wrap=True),
                        ft.Text(
                            "Tip: Tor default is socks5h://127.0.0.1:9050. Leave disabled to use direct connections.",
                            size=11,
                            color=WITH_OPACITY(0.6, BORDER_BASE),
                        ),
                        ft.Divider(),
                        section_title(
                            "Hugging Face Access",
                            getattr(ICONS, "HUB", ICONS.CLOUD),
                            "Save and test your Hugging Face API token. If saved, it's used globally.",
                            on_help_click=_mk_help_handler("Save and test your Hugging Face API token. If saved, it's used globally."),
                        ),
                        ft.Text(
                            "Saved token (if set) is used for Hugging Face Hub operations and dataset downloads.",
                            size=12,
                            color=WITH_OPACITY(0.7, BORDER_BASE),
                        ),
                        ft.Row([hf_token_tf], wrap=True),
                        ft.Row([hf_test_btn, hf_save_btn, hf_remove_btn], spacing=10, wrap=True),
                        hf_status,
                        ft.Divider(),
                        section_title(
                            "Runpod API Access",
                            getattr(ICONS, "VPN_KEY", getattr(ICONS, "KEY", ICONS.SETTINGS)),
                            "Save and test your Runpod API key. Used by Training → Runpod Infrastructure.",
                            on_help_click=_mk_help_handler("Save and test your Runpod API key. Used by Training → Runpod Infrastructure."),
                        ),
                        ft.Text(
                            "Stored locally and applied to RUNPOD_API_KEY when saved. Required for ensuring Runpod Network Volume & Template.",
                            size=12,
                            color=WITH_OPACITY(0.7, BORDER_BASE),
                        ),
                        ft.Row([runpod_key_tf], wrap=True),
                        ft.Row([runpod_test_btn, runpod_save_btn, runpod_remove_btn], spacing=10, wrap=True),
                        runpod_status,
                        ft.Divider(),
                        section_title(
                            "Ollama Connection",
                            getattr(ICONS, "HUB", ICONS.CLOUD),
                            "Configure connection to Ollama server; only stored here.",
                            on_help_click=_mk_help_handler("Configure connection to Ollama server; only stored here."),
                        ),
                        ft.Text(
                            "Connect to a local or remote Ollama server. This is only configuration; other tabs won't use it yet.",
                            size=12,
                            color=WITH_OPACITY(0.7, BORDER_BASE),
                        ),
                        ft.Row([ollama_enable_cb], wrap=True),
                        ft.Row([ollama_base_url_tf, ollama_default_model_tf], wrap=True),
                        ft.Row([ollama_models_dd], wrap=True),
                        ft.Row([ollama_test_btn, ollama_refresh_btn, ollama_save_btn], spacing=10, wrap=True),
                        ollama_status,
                    ], spacing=12),
                    width=1000,
                )
            ], alignment=ft.MainAxisAlignment.CENTER)
        ], scroll=ft.ScrollMode.AUTO, spacing=0),
        padding=16,
    )

    # ---- Mocked Dataset Analysis tab (defined inside main) ----
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
    # Analysis runtime settings (UI only, mocked)
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
        page.snack_bar = ft.SnackBar(ft.Text("Mock analysis executed."))

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
                inn, outn = _guess_cols(names)
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
    overview_block = ft.Column([
        section_title(
            "Overview",
            getattr(ICONS, "DASHBOARD", getattr(ICONS, "INSIGHTS", ICONS.SEARCH)),
            "Key KPIs such as total records, average input/output lengths, and duplicate rate (if enabled).",
            on_help_click=_mk_help_handler("Key KPIs such as total records, average input/output lengths, and duplicate rate (if enabled)."),
        ),
        overview_row,
    ], spacing=6, visible=False)

    sentiment_block = ft.Column([
        section_title(
            "Sentiment",
            getattr(ICONS, "EMOJI_EMOTIONS", getattr(ICONS, "INSERT_EMOTICON", ICONS.SEARCH)),
            "Heuristic sentiment distribution computed over sampled records.",
            on_help_click=_mk_help_handler("Heuristic sentiment distribution computed over sampled records."),
        ),
        sentiment_section,
    ], spacing=6, visible=False)

    class_balance_block = ft.Column([
        section_title(
            "Class balance",
            getattr(ICONS, "DONUT_SMALL", getattr(ICONS, "PIE_CHART", ICONS.SEARCH)),
            "Distribution of labels/classes if present in your dataset.",
            on_help_click=_mk_help_handler("Distribution of labels/classes if present in your dataset."),
        ),
        class_balance_section,
    ], spacing=6, visible=False)

    extra_metrics_block = ft.Column([
        section_title(
            "Extra metrics",
            getattr(ICONS, "INSIGHTS", getattr(ICONS, "ANALYTICS", ICONS.SEARCH)),
            "Lightweight proxies: coverage overlap, leakage check, depth, speaker balance, Q vs statement, readability, NER proxy, toxicity, politeness, dialogue acts, topics, alignment.",
            on_help_click=_mk_help_handler("Lightweight proxies: coverage overlap, leakage check, depth, speaker balance, Q vs statement, readability, NER proxy, toxicity, politeness, dialogue acts, topics, alignment."),
        ),
        extra_metrics_section,
    ], spacing=6, visible=False)

    samples_block = ft.Column([
        section_title(
            "Samples",
            getattr(ICONS, "LIST", getattr(ICONS, "LIST_ALT", ICONS.SEARCH)),
            "Random sample rows for quick spot checks (input/output and lengths).",
            on_help_click=_mk_help_handler("Random sample rows for quick spot checks (input/output and lengths)."),
        ),
        samples_section,
    ], spacing=6, visible=False)

    # Named dividers for each results block (hidden until analysis produces output)
    div_overview = ft.Divider(visible=False)
    div_sentiment = ft.Divider(visible=False)
    div_class = ft.Divider(visible=False)
    div_extra = ft.Divider(visible=False)
    div_samples = ft.Divider(visible=False)

    analysis_tab = ft.Container(
        content=ft.Column([
            ft.Row([
                section_title(
                    "Dataset Analysis",
                    getattr(ICONS, "INSIGHTS", getattr(ICONS, "ANALYTICS", ICONS.SEARCH)),
                    "Run modular analysis on your dataset. Enable modules and click Analyze to compute and reveal results.",
                    on_help_click=_mk_help_handler("Run modular analysis on your dataset. Enable modules and click Analyze to compute and reveal results."),
                ),
                ft.Container(expand=1),
                analyze_btn,
                analysis_busy_ring,
            ], alignment=ft.MainAxisAlignment.START),

            # Dataset chooser row
            ft.Row([
                analysis_source_dd,
                analysis_hf_repo,
                analysis_hf_split,
                analysis_hf_config,
                analysis_json_path,
            ], wrap=True, spacing=10),
            analysis_dataset_hint,
            ft.Divider(),
            section_title(
                "Analysis modules",
                getattr(ICONS, "TUNE", ICONS.SETTINGS),
                "Choose which checks to run. Only enabled modules are computed and displayed.",
                on_help_click=_mk_help_handler("Choose which checks to run. Only enabled modules are computed and displayed."),
            ),
            ft.Row([select_all_modules_cb], wrap=True),
            ft.Container(_build_modules_table(), padding=4, border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)), border_radius=8),

            ft.Divider(),
            section_title(
                "Runtime settings",
                getattr(ICONS, "SETTINGS", getattr(ICONS, "TUNE", ICONS.SETTINGS)),
                "Backend, token (for private HF datasets), and sampling. Sample size limits records analyzed for speed.",
                on_help_click=_mk_help_handler("Backend, token (for private HF datasets), and sampling. Sample size limits records analyzed for speed."),
            ),
            ft.Row([
                analysis_backend_dd,
                analysis_hf_token_tf,
                analysis_sample_size_tf,
            ], wrap=True, spacing=10),

            analysis_overview_note,
            div_overview,
            overview_block,

            div_sentiment,
            sentiment_block,

            div_class,
            class_balance_block,

            div_extra,
            extra_metrics_block,

            div_samples,
            samples_block,
        ], scroll=ft.ScrollMode.AUTO, spacing=12),
        padding=16,
    )

    tabs = ft.Tabs(
        tabs=[
            ft.Tab(text="Scrape", icon=ICONS.SEARCH, content=scrape_tab),
            ft.Tab(text="Build / Publish", icon=ICONS.BUILD_CIRCLE_OUTLINED, content=build_tab),
            ft.Tab(text="Training", icon=getattr(ICONS, "SCIENCE", ICONS.PLAY_CIRCLE), content=training_tab),
            ft.Tab(text="Merge Datasets", icon=getattr(ICONS, "MERGE_TYPE", ICONS.TABLE_VIEW), content=merge_tab),
            ft.Tab(text="Dataset Analysis", icon=getattr(ICONS, "INSIGHTS", ICONS.ANALYTICS), content=analysis_tab),
            ft.Tab(text="Settings", icon=ICONS.SETTINGS, content=settings_tab),
        ],
        expand=1,
    )

    page.add(tabs)
    # Initialize visibility by current source value
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

    # Initialize select-all state based on defaults
    try:
        _sync_select_all_modules()
    except Exception:
        pass


if __name__ == "__main__":
    ft.app(target=main)
