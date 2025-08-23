import asyncio
import random
import time
import os
import shutil
from datetime import datetime
from typing import List, Optional
import json
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

import flet as ft
import fourchan_scraper as sc
try:
    import save_dataset as sd
except Exception:
    import sys as _sys
    _sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import save_dataset as sd

import reddit_scraper as rdt
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


def section_title(title: str, icon: str) -> ft.Row:
    return ft.Row([
        ft.Icon(icon, color=ACCENT_COLOR),
        ft.Text(title, size=16, weight=ft.FontWeight.BOLD),
    ])


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
        options=[ft.dropdown.Option("4chan"), ft.dropdown.Option("reddit")],
        width=180,
    )
    reddit_url = ft.TextField(label="Reddit URL (subreddit or post)", value="https://www.reddit.com/r/Conservative/", width=420)
    reddit_max_posts = ft.TextField(label="Max Posts (Reddit)", value="30", width=180, keyboard_type=ft.KeyboardType.NUMBER)
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

    # Toggle visibility between 4chan and Reddit controls
    def update_source_controls():
        use_reddit = (source_dd.value == "reddit")
        # Boards area
        try:
            boards_wrap.visible = not use_reddit
            board_actions.visible = not use_reddit
            board_warning.visible = not use_reddit
        except Exception:
            pass
        # Reddit params
        try:
            reddit_params_row.visible = use_reddit
        except Exception:
            pass
        # Some parameter labels are shared
        max_threads.visible = not use_reddit  # 4chan-specific
        max_pairs.visible = not use_reddit    # 4chan target pairs
        # For Reddit, delay/min_len/output_path and pairing controls still apply
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

    # Actions
    cancel_state = {"cancelled": False}

    # Start button with validation state (default enabled due to defaults)
    start_button = ft.ElevatedButton(
        "Start", icon=ICONS.PLAY_ARROW,
        on_click=lambda e: page.run_task(on_start_scrape),
        disabled=False,
    )

    def update_board_validation():
        # If scraping 4chan, enforce board selection; Reddit doesn't require boards
        if source_dd.value == "reddit":
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
        if source_dd.value != "reddit" and not selected_boards:
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
    reddit_params_row = ft.Row([reddit_url, reddit_max_posts], wrap=True)
    reddit_params_row.visible = False

    scrape_tab = ft.Container(
        content=ft.Column([
            section_title("Source", ICONS.DASHBOARD),
            ft.Row([source_dd], wrap=True),
            section_title("4chan Boards", ICONS.DASHBOARD),
            board_actions,
            boards_wrap,
            board_warning,
            ft.Divider(),
            section_title("Parameters", ICONS.TUNE),
            reddit_params_row,
            ft.Row([max_threads, max_pairs, delay, min_len, output_path], wrap=True),
            ft.Row([pair_mode, strategy_dd, k_field, max_chars_field], wrap=True),
            ft.Row([merge_same_id_cb, require_question_cb], wrap=True),
            scrape_actions,
            ft.Container(height=10),
            section_title("Progress", ICONS.TIMELAPSE),
            ft.Row([scrape_prog, working_ring, ft.Text("Working...")], spacing=16),
            stats_cards,
            ft.Row([threads_label, pairs_label], spacing=20),
            ft.Divider(),
            section_title("Live Log", ICONS.TERMINAL),
            ft.Container(log_area, height=180, border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                         border_radius=8, padding=10),
            section_title("Preview", ICONS.PREVIEW),
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
    data_file = ft.TextField(label="Data file (JSON)", value="scraped_training_data.json", width=360)
    seed = ft.TextField(label="Seed", value="42", width=120, keyboard_type=ft.KeyboardType.NUMBER)
    shuffle = ft.Switch(label="Shuffle", value=True)
    val_slider = ft.Slider(min=0, max=0.2, value=0.01, divisions=20, label="{value}")
    test_slider = ft.Slider(min=0, max=0.2, value=0.0, divisions=20, label="{value}")
    min_len_b = ft.TextField(label="Min Length", value="1", width=120, keyboard_type=ft.KeyboardType.NUMBER)
    save_dir = ft.TextField(label="Save dir", value="hf_dataset", width=240)

    push_toggle = ft.Switch(label="Push to Hub", value=False)
    repo_id = ft.TextField(label="Repo ID", value="username/my-dataset", width=280)
    private = ft.Switch(label="Private", value=True)
    token = ft.TextField(label="HF Token", password=True, can_reveal_password=True, width=320)

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
        token_val = (token.value or "").strip()

        def add_step(text: str, color, icon):
            timeline.controls.append(ft.Row([ft.Icon(icon, color=color), ft.Text(text)]))
            try:
                timeline_placeholder.visible = len(timeline.controls) == 0
            except Exception:
                pass

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
                    token_val = sd.HfFolder.get_token()
                except Exception:
                    token_val = ""
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

                # Build README content with heartbeat (fast, but keep consistent)
                readme_text = ft.Text("Building dataset card (README)")
                timeline.controls.append(ft.Row([ft.Icon(ICONS.ARTICLE, color=COLORS.BLUE), readme_text]))
                await safe_update(page)
                readme = await asyncio.to_thread(sd.build_dataset_card_content, dd, repo)
                readme_text.value = "Built dataset card"
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
        tok = (token.value or "").strip() or getattr(sd.HfFolder, "get_token", lambda: "")()
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
                        section_title("Dataset Params", ICONS.SETTINGS),
                        ft.Row([data_file, seed, shuffle, min_len_b, save_dir], wrap=True),
                        ft.Divider(),
                        section_title("Splits", ICONS.TABLE_VIEW),
                        ft.Row([
                            ft.Column([
                                ft.Text("Validation Fraction"), val_slider,
                                ft.Text("Test Fraction"), test_slider,
                                split_error,
                            ], width=360),
                            ft.Row([split_badges["train"], split_badges["val"], split_badges["test"]], spacing=10),
                        ], wrap=True, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        ft.Divider(),
                        section_title("Push to Hub", ICONS.PUBLIC),
                        ft.Row([push_toggle, repo_id, private, token], wrap=True),
                        build_actions,
                        ft.Divider(),
                        section_title("Status", ICONS.TASK),
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
        ds_id = ft.TextField(label="Dataset repo (e.g., username/dataset)", width=360)
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
        )
        config = ft.TextField(label="Config (optional)", width=180)
        in_col = ft.TextField(label="Input column (optional)", width=200)
        out_col = ft.TextField(label="Output column (optional)", width=200)
        remove_btn = ft.IconButton(ICONS.DELETE)
        row = ft.Row([ds_id, split, config, in_col, out_col, remove_btn], spacing=10, wrap=True)

        # Keep references for later retrieval
        row.data = {
            "ds": ds_id,
            "split": split,
            "config": config,
            "in": in_col,
            "out": out_col,
        }

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

    # Seed with one row initially
    rows_host.controls.append(make_dataset_row())

    # Output settings
    merge_save_dir = ft.TextField(label="Save dir", value="merged_dataset", width=240)

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
            ds_tf = d.get("ds")
            sp_dd = d.get("split")
            cfg_tf = d.get("config")
            in_tf = d.get("in")
            out_tf = d.get("out")
            repo = (getattr(ds_tf, "value", "") or "").strip() if ds_tf else ""
            if repo:
                entries.append({
                    "repo": repo,
                    "split": (getattr(sp_dd, "value", "train") or "train") if sp_dd else "train",
                    "config": (getattr(cfg_tf, "value", "") or "").strip() if cfg_tf else "",
                    "in": (getattr(in_tf, "value", "") or "").strip() if in_tf else "",
                    "out": (getattr(out_tf, "value", "") or "").strip() if out_tf else "",
                })
        if len(entries) < 2:
            merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED), ft.Text("Add at least two datasets")]))
            update_merge_placeholders(); await safe_update(page)
            merge_busy_ring.visible = False
            await safe_update(page)
            return

        out_dir = merge_save_dir.value or "merged_dataset"
        op = merge_op.value or "Concatenate"

        # Load and map each dataset
        prepped_all = []
        for i, ent in enumerate(entries, start=1):
            if merge_cancel.get("cancelled"):
                break
            merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.DOWNLOAD, color=COLORS.BLUE), ft.Text(f"Loading {ent['repo']} [{ent['split']}]…")]))
            update_merge_placeholders(); await safe_update(page)
            try:
                dss = await _load_and_prepare(ent["repo"], ent["split"], ent["config"], ent["in"], ent["out"])
                prepped_all.extend(dss)
                merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text(f"Prepared {ent['repo']}")]))
            except Exception as e:
                merge_timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED), ft.Text(f"Failed {ent['repo']}: {e}")]))
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

        # Merge
        try:
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
            head_n = min(12, len(merged))
            idxs = list(range(head_n))
            head = await asyncio.to_thread(lambda: merged.select(idxs)) if head_n > 0 else None
            pairs = []
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
        """Open a modal dialog showing the merged dataset saved to disk (DatasetDict)."""
        # Immediate feedback
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Opening merged dataset preview..."))
            page.snack_bar.open = True
            await safe_update(page)
        except Exception:
            pass

        # Resolve save dir robustly
        orig_dir = merge_save_dir.value or "merged_dataset"
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
                        section_title("Merge Datasets", ICONS.TABLE_VIEW),
                        ft.Text("Combine multiple Hugging Face datasets. Map columns to a unified input/output schema and merge.", size=12, color=WITH_OPACITY(0.7, BORDER_BASE)),
                        ft.Divider(),
                        section_title("Operation", ICONS.SHUFFLE),
                        ft.Row([merge_op], wrap=True),
                        ft.Divider(),
                        section_title("Datasets", ICONS.TABLE_VIEW),
                        ft.Row([add_row_btn, clear_btn], spacing=8),
                        rows_host,
                        ft.Divider(),
                        section_title("Output", ICONS.SAVE_ALT),
                        ft.Row([merge_save_dir], wrap=True),
                        merge_actions,
                        ft.Divider(),
                        section_title("Preview", ICONS.PREVIEW),
                        ft.Container(ft.Stack([merge_preview_host, merge_preview_placeholder], expand=True),
                                     height=220,
                                     width=1000,
                                     border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                                     border_radius=8,
                                     padding=6,
                        ),
                        ft.Divider(),
                        section_title("Status", ICONS.TASK),
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

    # ---------- SETTINGS TAB (Proxy config) ----------
    settings_tab = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Container(
                    content=ft.Column([
                        section_title("Proxy Settings", ICONS.SETTINGS),
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
                    ], spacing=12),
                    width=1000,
                )
            ], alignment=ft.MainAxisAlignment.CENTER)
        ], scroll=ft.ScrollMode.AUTO, spacing=0),
        padding=16,
    )

    tabs = ft.Tabs(
        tabs=[
            ft.Tab(text="Scrape", icon=ICONS.SEARCH, content=scrape_tab),
            ft.Tab(text="Build / Publish", icon=ICONS.BUILD_CIRCLE_OUTLINED, content=build_tab),
            ft.Tab(text="Merge Datasets", icon=getattr(ICONS, "MERGE_TYPE", ICONS.TABLE_VIEW), content=merge_tab),
            ft.Tab(text="Settings", icon=ICONS.SETTINGS, content=settings_tab),
        ],
        expand=1,
    )

    page.add(tabs)
    # Initialize visibility by current source value
    update_source_controls()


if __name__ == "__main__":
    ft.app(target=main)
