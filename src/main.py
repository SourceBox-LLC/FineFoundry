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
try:
    import save_dataset as sd
except Exception:
    import sys as _sys
    _sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import save_dataset as sd

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
    base_model = ft.Dropdown(
        label="Base model",
        options=[
            ft.dropdown.Option("microsoft/phi-2"),
            ft.dropdown.Option("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            ft.dropdown.Option("google/gemma-2b"),
        ],
        value="microsoft/phi-2",
        width=320,
    )
    epochs_tf = ft.TextField(label="Epochs", value="3", width=120)
    lr_tf = ft.TextField(label="Learning rate", value="2e-4", width=160)
    batch_tf = ft.TextField(label="Per-device batch size", value="2", width=200)
    grad_acc_tf = ft.TextField(label="Grad accum steps", value="4", width=180)
    max_steps_tf = ft.TextField(label="Max steps (mock)", value="200", width=180)
    use_lora_cb = ft.Checkbox(label="Use LoRA", value=True)
    out_dir_tf = ft.TextField(label="Output dir", value="outputs/mock_run", width=260)

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

    # Progress & logs
    train_progress = ft.ProgressBar(value=0.0, width=400)
    train_prog_label = ft.Text("Progress: 0%")
    train_timeline = ft.Column([], spacing=4)
    train_timeline_placeholder = make_empty_placeholder("No training logs yet", getattr(ICONS, "SCIENCE", ICONS.PLAY_CIRCLE))

    def update_train_placeholders():
        try:
            has_logs = len(getattr(train_timeline, "controls", []) or []) > 0
            train_timeline_placeholder.visible = not has_logs
        except Exception:
            pass

    cancel_train = {"cancelled": False}
    train_state = {"running": False}

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
        # Set beginner defaults
        if is_beginner:
            try:
                epochs_tf.value = epochs_tf.value or "1"
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
    train_source.on_change = _update_train_source

    async def on_start_training():
        if train_state.get("running"):
            return
        cancel_train["cancelled"] = False
        train_state["running"] = True

        # Read inputs
        src = train_source.value or "Hugging Face"
        repo = (train_hf_repo.value or "").strip()
        split = (train_hf_split.value or "train").strip()
        jpath = (train_json_path.value or "").strip()
        model = base_model.value or "microsoft/phi-2"
        epochs_s = (epochs_tf.value or "3").strip()
        lr_s = (lr_tf.value or "2e-4").strip()
        bsz_s = (batch_tf.value or "2").strip()
        acc_s = (grad_acc_tf.value or "4").strip()
        max_steps_s = (max_steps_tf.value or "200").strip()
        out_dir = (out_dir_tf.value or "outputs/mock_run").strip()
        try:
            total_steps = max(10, min(2000, int(float(max_steps_s))))
        except Exception:
            total_steps = 200

        # Intro logs
        train_timeline.controls.append(ft.Row([ft.Icon(getattr(ICONS, "SCIENCE", ICONS.PLAY_CIRCLE), color=ACCENT_COLOR), ft.Text("Starting mock training…")]))
        ds_desc = f"HF: {repo} [{split}]" if (src == "Hugging Face") else f"JSON: {jpath}"
        train_timeline.controls.append(ft.Row([ft.Icon(ICONS.TABLE_VIEW, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"Dataset: {ds_desc}")]))
        train_timeline.controls.append(ft.Row([ft.Icon(ICONS.SETTINGS, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"Model={model} • Epochs={epochs_s} • LR={lr_s} • BSZ={bsz_s} • GA={acc_s} • LoRA={'on' if use_lora_cb.value else 'off'}")]))
        train_timeline.controls.append(ft.Row([ft.Icon(ICONS.SAVE_ALT, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"Output dir: {out_dir}")]))
        update_train_placeholders(); await safe_update(page)

        # Simulate steps
        for step in range(1, total_steps + 1):
            if cancel_train.get("cancelled"):
                train_timeline.controls.append(ft.Row([ft.Icon(ICONS.CANCEL, color=COLORS.RED), ft.Text("Training cancelled by user")]))
                train_state["running"] = False
                await safe_update(page)
                return
            await asyncio.sleep(0.035 + random.uniform(0.0, 0.015))
            train_progress.value = step / total_steps
            try:
                pct = int(100 * float(train_progress.value or 0))
                train_prog_label.value = f"Progress: {pct}%"
            except Exception:
                pass
            # Heartbeat logs
            if step == 1 or (step % max(1, total_steps // 20) == 0) or (step % 25 == 0):
                fake_loss = max(0.02, 2.0 / (1.0 + 0.02 * step) + random.uniform(-0.05, 0.05))
                train_timeline.controls.append(ft.Row([ft.Icon(ICONS.FAVORITE, color=WITH_OPACITY(0.9, COLORS.PINK)), ft.Text(f"Step {step}/{total_steps} — loss={fake_loss:.3f}")]))
            await safe_update(page)

        # Completed
        train_progress.value = 1.0
        train_prog_label.value = "Progress: 100%"
        train_timeline.controls.append(ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text("Training finished ✔")]))
        train_state["running"] = False
        await safe_update(page)

    def on_stop_training(_):
        if not train_state.get("running"):
            return
        cancel_train["cancelled"] = True
        try:
            train_timeline.controls.append(ft.Row([ft.Icon(ICONS.CANCEL, color=COLORS.RED), ft.Text("Cancel requested — will stop ASAP")]))
            update_train_placeholders(); page.update()
        except Exception:
            pass

    def on_refresh_training(_):
        if train_state.get("running"):
            return
        try:
            train_timeline.controls.clear()
            train_progress.value = 0.0
            train_prog_label.value = "Progress: 0%"
            update_train_placeholders(); page.update()
        except Exception:
            pass

    train_actions = ft.Row([
        ft.ElevatedButton("Start Training (mock)", icon=getattr(ICONS, "SCIENCE", ICONS.PLAY_CIRCLE), on_click=lambda e: page.run_task(on_start_training)),
        ft.OutlinedButton("Stop", icon=ICONS.STOP_CIRCLE, on_click=on_stop_training),
        ft.TextButton("Refresh", icon=REFRESH_ICON, on_click=on_refresh_training),
    ], spacing=10)

    training_tab = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Container(
                    content=ft.Column([
                        section_title(
                            "Dataset",
                            ICONS.TABLE_VIEW,
                            "Select the dataset used for mock training.",
                            on_help_click=_mk_help_handler("Select the dataset used for mock training."),
                        ),
                        ft.Row([train_source, train_hf_repo, train_hf_split, train_hf_config, train_json_path], wrap=True),
                        ft.Divider(),
                        section_title(
                            "Training Params",
                            ICONS.SETTINGS,
                            "Basic hyperparameters and LoRA toggle. Not executed for real.",
                            on_help_click=_mk_help_handler("Basic hyperparameters and LoRA toggle. Not executed for real."),
                        ),
                        ft.Row([skill_level], wrap=True),
                        ft.Row([base_model, epochs_tf, lr_tf, batch_tf, grad_acc_tf, max_steps_tf, use_lora_cb, out_dir_tf], wrap=True),
                        advanced_params_section,
                        ft.Divider(),
                        section_title(
                            "Progress & Logs",
                            ICONS.TASK_ALT,
                            "Mocked progress bar and log output.",
                            on_help_click=_mk_help_handler("Mocked progress bar and log output."),
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
