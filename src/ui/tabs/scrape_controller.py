"""Scrape tab controller for FineFoundry.

This module builds the Scrape tab controls and wires up all scraping
handlers, keeping `src/main.py` smaller.

Layout composition still lives in `tab_scrape.py`; this module focuses
on behavior and state wiring.
"""

from __future__ import annotations

from typing import List

import asyncio
import json
import os

import flet as ft

from helpers.boards import load_4chan_boards
from helpers.common import safe_update
from helpers.scrape import (
    run_reddit_scrape as run_reddit_scrape_helper,
    run_real_scrape as run_real_scrape_helper,
    run_stackexchange_scrape as run_stackexchange_scrape_helper,
)
from helpers.synthetic import (
    run_synthetic_generation,
)
from helpers.theme import BORDER_BASE, COLORS, ICONS
from helpers.ui import (
    WITH_OPACITY,
    compute_two_col_flex,
    make_empty_placeholder,
    make_selectable_pill,
    make_wrap,
    pill,
    two_col_header,
    two_col_row,
)
from ui.tabs.tab_scrape import build_scrape_tab


def _schedule_task(page: ft.Page, coro):
    """Robust scheduler helper for async tasks.

    Mirrors the pattern used in `main.py`, preferring `page.run_task`
    when available and falling back to `asyncio.create_task`.
    """
    try:
        if hasattr(page, "run_task") and callable(getattr(page, "run_task")):
            return page.run_task(coro)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        return asyncio.create_task(coro())
    except Exception:  # pragma: no cover - defensive
        # As a last resort, block in a thread
        try:
            loop = asyncio.get_event_loop()
        except Exception:
            return None
        return loop.run_in_executor(None, lambda: asyncio.run(coro()))


def build_scrape_tab_with_logic(
    page: ft.Page,
    *,
    section_title,
    _mk_help_handler,
    proxy_enable_cb: ft.Control,
    use_env_cb: ft.Control,
    proxy_url_tf: ft.TextField,
) -> ft.Control:
    """Build the Scrape tab UI and attach all related handlers.

    This mirrors the previous inline Scrape tab setup from `main.py`, but
    keeps the behavior localized to this module.
    """

    # Boards (dynamic from API with fallback) and multi-select pills
    boards = load_4chan_boards()
    default_sel = {"pol", "b", "x"}
    board_pills: List[ft.Container] = [
        make_selectable_pill(b, selected=b in default_sel, base_color=COLORS.AMBER) for b in boards
    ]
    boards_wrap = make_wrap(board_pills, spacing=6, run_spacing=6)
    board_warning = ft.Text("", color=COLORS.RED)

    # Refs to top-level sections so we can toggle visibility from state helpers
    boards_section_ref: dict = {}
    progress_section_ref: dict = {}
    log_section_ref: dict = {}
    preview_section_ref: dict = {}

    # Inputs
    source_dd = ft.Dropdown(
        label="Source",
        value="4chan",
        options=[
            ft.dropdown.Option("4chan"),
            ft.dropdown.Option("reddit"),
            ft.dropdown.Option("stackexchange"),
            ft.dropdown.Option("synthetic"),
        ],
        width=180,
    )
    reddit_url = ft.TextField(
        label="Reddit URL (subreddit or post)",
        value="https://www.reddit.com/r/LocalLLaMA/",
        width=420,
    )
    reddit_max_posts = ft.TextField(
        label="Max Posts (Reddit)",
        value="30",
        width=180,
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    se_site = ft.TextField(
        label="StackExchange Site",
        value="stackoverflow",
        width=260,
    )

    # Synthetic data controls
    synthetic_files = ft.TextField(
        label="Files/URLs (comma-separated or directory)",
        value="",
        width=500,
        hint_text="e.g., doc.pdf, slides.pptx, https://docs.python.org/",
    )
    synthetic_file_picker = ft.FilePicker()
    synthetic_selected_files: List[str] = []

    def on_synthetic_files_picked(e: ft.FilePickerResultEvent):
        if e.files:
            paths = [f.path for f in e.files if f.path]
            synthetic_selected_files.clear()
            synthetic_selected_files.extend(paths)
            synthetic_files.value = ", ".join(paths)
            update_board_validation()
            page.update()

    synthetic_file_picker.on_result = on_synthetic_files_picked
    page.overlay.append(synthetic_file_picker)

    synthetic_browse_btn = ft.ElevatedButton(
        "Browse",
        icon=ICONS.FOLDER_OPEN if hasattr(ICONS, "FOLDER_OPEN") else ICONS.FOLDER,
        on_click=lambda _: synthetic_file_picker.pick_files(
            allow_multiple=True,
            allowed_extensions=["pdf", "docx", "pptx", "html", "txt"],
        ),
    )
    synthetic_gen_type = ft.Dropdown(
        label="Generation Type",
        value="qa",
        options=[
            ft.dropdown.Option("qa", "Q&A Pairs"),
            ft.dropdown.Option("cot", "Chain of Thought"),
            ft.dropdown.Option("summary", "Summary"),
        ],
        width=180,
    )
    synthetic_num_pairs = ft.TextField(
        label="Pairs per Chunk",
        value="25",
        width=140,
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    synthetic_max_chunks = ft.TextField(
        label="Max Chunks",
        value="10",
        width=140,
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    synthetic_curate_cb = ft.Checkbox(label="Quality Curation", value=False)
    synthetic_threshold = ft.TextField(
        label="Threshold (1-10)",
        value="7.5",
        width=120,
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    synthetic_multimodal_cb = ft.Checkbox(
        label="Multimodal (extract images from PDFs)",
        value=False,
    )
    synthetic_model = ft.TextField(
        label="Model",
        value="unsloth/Llama-3.2-3B-Instruct",
        width=300,
    )

    max_threads = ft.TextField(
        label="Max Threads",
        value="50",
        width=160,
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    max_pairs = ft.TextField(
        label="Max Pairs",
        value="5000",
        width=160,
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    delay = ft.TextField(
        label="Delay (s)",
        value="1.0",
        width=160,
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    min_len = ft.TextField(
        label="Min Length",
        value="3",
        width=160,
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    output_path = ft.TextField(
        label="Output JSON Path",
        value="scraped_training_data.json",
        width=360,
    )
    dataset_format_dd = ft.Dropdown(
        label="Dataset Format",
        options=[ft.dropdown.Option("ChatML"), ft.dropdown.Option("Standard")],
        value="ChatML",
        width=200,
        tooltip=(
            "Select output dataset format: ChatML (multi-turn conversations) or Standard (raw input/output pairs)."
        ),
    )

    # Pairing mode control
    multiturn_sw = ft.Switch(label="Multiturn", value=False)
    strategy_dd = ft.Dropdown(
        label="Context Strategy",
        value="cumulative",
        options=[
            ft.dropdown.Option("cumulative"),
            ft.dropdown.Option("last_k"),
            ft.dropdown.Option("quote_chain"),
        ],
        width=200,
    )
    k_field = ft.TextField(
        label="Last K",
        value="6",
        width=120,
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    max_chars_field = ft.TextField(
        label="Max Input Chars",
        value="",
        width=160,
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    merge_same_id_cb = ft.Checkbox(label="Merge same poster", value=True)
    require_question_cb = ft.Checkbox(
        label="Require question in context",
        value=False,
    )

    def update_context_controls() -> None:
        is_ctx = bool(multiturn_sw.value)
        strategy_dd.visible = is_ctx
        k_field.visible = is_ctx
        max_chars_field.visible = is_ctx
        merge_same_id_cb.visible = is_ctx
        require_question_cb.visible = is_ctx
        page.update()

    multiturn_sw.on_change = lambda e: update_context_controls()
    update_context_controls()

    # Progress + stats
    scrape_prog = ft.ProgressBar(width=400, value=0)
    working_ring = ft.ProgressRing(width=20, height=20, value=None, visible=False)
    threads_label = ft.Text("Threads Visited: 0")
    pairs_label = ft.Text("Pairs Found: 0")
    stats_cards = ft.Row(
        [
            ft.Container(
                pill("Threads Visited: 0", COLORS.BLUE, ICONS.TRAVEL_EXPLORE),
                padding=10,
            ),
            ft.Container(
                pill("Pairs Found: 0", COLORS.GREEN, ICONS.CHAT),
                padding=10,
            ),
        ]
    )

    # Live log
    log_list = ft.ListView(expand=1, auto_scroll=True, spacing=4)
    log_placeholder = make_empty_placeholder("No logs yet", ICONS.TERMINAL)
    log_area = ft.Stack([log_list, log_placeholder], expand=True)

    # Preview host: flex-based two-column grid (ListView of Rows)
    preview_host = ft.ListView(expand=1, auto_scroll=False)
    preview_placeholder = make_empty_placeholder(
        "Preview not available",
        ICONS.PREVIEW,
    )
    preview_area = ft.Stack([preview_host, preview_placeholder], expand=True)

    # Actions
    cancel_state = {"cancelled": False}

    # Start button with validation state (default enabled due to defaults)
    start_button = ft.ElevatedButton(
        "Start",
        icon=ICONS.PLAY_ARROW,
        on_click=lambda e: page.run_task(on_start_scrape),
        disabled=False,
    )

    def update_board_validation() -> None:
        # If scraping 4chan, enforce board selection; others don't require boards
        if source_dd.value in ("reddit", "stackexchange"):
            start_button.disabled = False
            board_warning.value = ""
        elif source_dd.value == "synthetic":
            # Synthetic requires files/URLs
            has_input = bool((synthetic_files.value or "").strip()) or bool(synthetic_selected_files)
            start_button.disabled = not has_input
            board_warning.value = "Add files or URLs to generate from." if not has_input else ""
        else:
            any_selected = any(p.data and p.data.get("selected") for p in board_pills)
            start_button.disabled = not any_selected
            board_warning.value = "Select at least one board to scrape." if not any_selected else ""
        page.update()

    def select_all_boards(_):
        for pill_ctl in board_pills:
            if pill_ctl.data is None:
                pill_ctl.data = {}
            pill_ctl.data["selected"] = True
            try:
                base_color = pill_ctl.data.get("base_color")
            except Exception:
                base_color = None
            try:
                pill_ctl.bgcolor = WITH_OPACITY(0.15, base_color) if base_color is not None else None
            except Exception:
                pass
        page.update()
        update_board_validation()

    def clear_all_boards(_):
        for pill_ctl in board_pills:
            if pill_ctl.data is None:
                pill_ctl.data = {}
            pill_ctl.data["selected"] = False
            pill_ctl.bgcolor = None
        page.update()
        update_board_validation()

    board_actions = ft.Row(
        [
            ft.TextButton("Select All", on_click=select_all_boards),
            ft.TextButton("Clear", on_click=clear_all_boards),
        ],
        spacing=8,
    )

    # Attach change callbacks after start_button exists
    for p in board_pills:
        if p.data is None:
            p.data = {}
        p.data["on_change"] = update_board_validation
    update_board_validation()

    def update_source_controls() -> None:
        src = (source_dd.value or "").strip().lower()
        is_reddit = src == "reddit"
        is_se = src == "stackexchange"
        is_synthetic = src == "synthetic"
        is_4chan = src == "4chan"

        try:
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
        # Synthetic params
        try:
            synthetic_params_row.visible = is_synthetic
        except Exception:
            pass
        # Parameters visibility - hide scraping params for synthetic
        max_threads.visible = is_4chan  # 4chan-specific
        max_pairs.visible = is_4chan or is_se  # used by 4chan and StackExchange
        delay.visible = not is_synthetic  # not needed for synthetic
        min_len.visible = not is_synthetic  # not needed for synthetic
        # Pairing/context controls apply to 4chan and Reddit only
        for ctl in [
            multiturn_sw,
            strategy_dd,
            k_field,
            max_chars_field,
            merge_same_id_cb,
            require_question_cb,
        ]:
            try:
                ctl.visible = is_4chan or is_reddit
            except Exception:
                pass
        page.update()

    source_dd.on_change = lambda e: (update_source_controls(), update_board_validation())

    def update_scrape_placeholders() -> None:
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
                    # Live Log visible while running OR if there are logs to show
                    ctl.visible = bool(working_ring.visible) or has_logs
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

    async def on_start_scrape():
        cancel_state["cancelled"] = False

        # Show immediate snackbar for synthetic (before any processing)
        if source_dd.value == "synthetic":
            page.snack_bar = ft.SnackBar(
                ft.Text("⏳ Loading model... This may take 30-60 seconds on first run"),
                duration=10000,
            )
            page.open(page.snack_bar)
            page.update()  # Force immediate display

        log_list.controls.clear()
        preview_host.controls.clear()
        scrape_prog.value = 0
        threads_label.value = "Sources Processed: 0"
        pairs_label.value = "Pairs Generated: 0"
        working_ring.visible = True
        update_scrape_placeholders()
        # Force UI update immediately so user sees the reset
        await safe_update(page)
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
            ml_val = int(min_len.value or 3)
        except Exception:
            ml_val = 3
        out_path = output_path.value or "scraped_training_data.json"
        # Context params
        multiturn = bool(multiturn_sw.value)
        strat_val = strategy_dd.value or "cumulative"
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
            log_list.controls.append(
                ft.Text(
                    f"Reddit URL: {reddit_url.value} | Max posts: {reddit_max_posts.value}",
                ),
            )
        elif source_dd.value == "stackexchange":
            log_list.controls.append(
                ft.Text(
                    f"StackExchange site: {se_site.value} | Max pairs: {max_pairs.value}",
                ),
            )
        else:
            log_list.controls.append(
                ft.Text(
                    f"Boards: {', '.join(selected_boards[:20])}{' ...' if len(selected_boards) > 20 else ''}",
                ),
            )
        # Log chosen dataset format
        try:
            log_list.controls.append(
                ft.Text(f"Dataset format: {dataset_format_dd.value}"),
            )
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
                    min_len_val=ml_val,
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
                    min_len_val=ml_val,
                    output_path=out_path,
                    ui_proxy_enabled=bool(proxy_enable_cb.value),
                    ui_proxy_url=(proxy_url_tf.value or "").strip(),
                    ui_use_env_proxies=bool(use_env_cb.value),
                    dataset_format=(dataset_format_dd.value or "ChatML"),
                )
            elif source_dd.value == "synthetic":
                # Collect sources from text field or file picker
                sources_list = []
                if synthetic_selected_files:
                    sources_list.extend(synthetic_selected_files)
                if synthetic_files.value:
                    # Parse comma-separated values
                    for item in synthetic_files.value.split(","):
                        item = item.strip()
                        if item and item not in sources_list:
                            sources_list.append(item)

                # Parse synthetic params
                try:
                    syn_num_pairs = int(synthetic_num_pairs.value or 25)
                except Exception:
                    syn_num_pairs = 25
                try:
                    syn_max_chunks = int(synthetic_max_chunks.value or 10)
                except Exception:
                    syn_max_chunks = 10
                try:
                    syn_threshold = float(synthetic_threshold.value or 7.5)
                except Exception:
                    syn_threshold = 7.5

                await run_synthetic_generation(
                    page=page,
                    log_view=log_list,
                    prog=scrape_prog,
                    labels={"threads": threads_label, "pairs": pairs_label},
                    preview_host=preview_host,
                    cancel_flag=cancel_state,
                    sources=sources_list,
                    gen_type=synthetic_gen_type.value or "qa",
                    num_pairs=syn_num_pairs,
                    max_chunks=syn_max_chunks,
                    curate=bool(synthetic_curate_cb.value),
                    curate_threshold=syn_threshold,
                    multimodal=bool(synthetic_multimodal_cb.value),
                    output_path=out_path,
                    dataset_format=(dataset_format_dd.value or "ChatML"),
                    model=synthetic_model.value or "unsloth/Llama-3.2-3B-Instruct",
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
                    min_len_val=ml_val,
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
            make_selectable_pill(b, selected=(b in {"pol", "b", "x"}), base_color=COLORS.AMBER) for b in boards
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
            candidates.extend(
                [
                    orig_path,
                    os.path.abspath(orig_path),
                    os.path.join(os.getcwd(), orig_path),
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), orig_path),
                ]
            )
        # Deduplicate while preserving order
        seen = set()
        resolved_list = []
        for pth in candidates:
            ap = os.path.abspath(pth)
            if ap not in seen:
                seen.add(ap)
                resolved_list.append(ap)
        existing = next((p for p in resolved_list if os.path.exists(p)), None)
        if not existing:
            page.snack_bar = ft.SnackBar(
                ft.Text(
                    "Dataset file not found. Tried:\n" + "\n".join(resolved_list[:4]),
                )
            )
            page.open(page.snack_bar)
            await safe_update(page)
            return

        try:
            data = await asyncio.to_thread(
                lambda: json.load(open(existing, "r", encoding="utf-8")),
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
                return (
                    str(rec.get("input", "") or ""),
                    str(rec.get("output", "") or ""),
                )
            except Exception:
                return ("", "")

        def render_page() -> None:
            start = state["page"] * page_size
            end = min(start + page_size, total)
            grid_list.controls.clear()
            # Compute dynamic flex for current page
            page_samples = [_extract_pair(r if isinstance(r, dict) else {}) for r in data[start:end]]
            lfx, rfx = compute_two_col_flex(page_samples)
            hdr_left = "User" if is_chatml_dataset else "Input"
            hdr_right = "Assistant" if is_chatml_dataset else "Output"
            grid_list.controls.append(
                two_col_header(hdr_left, hdr_right, left_flex=lfx, right_flex=rfx),
            )
            for a, b in page_samples:
                grid_list.controls.append(two_col_row(a, b, lfx, rfx))
            info_text.value = f"Page {state['page'] + 1}/{total_pages} • Showing {start + 1}-{end} of {total}"
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

        controls_bar = ft.Row(
            [prev_btn, next_btn, info_text],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        full_scroll = grid_list

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Dataset Viewer — {len(data)} rows"),
            content=ft.Container(
                width=900,
                height=600,
                content=ft.Column(
                    [
                        controls_bar,
                        ft.Container(full_scroll, expand=True),
                    ],
                    expand=True,
                ),
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
            candidates.extend(
                [
                    orig_path,
                    os.path.abspath(orig_path),
                    os.path.join(os.getcwd(), orig_path),
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), orig_path),
                ]
            )
        seen = set()
        resolved_list = []
        for pth in candidates:
            ap = os.path.abspath(pth)
            if ap not in seen:
                seen.add(ap)
                resolved_list.append(ap)
        existing = next((p for p in resolved_list if os.path.exists(p)), None)
        if not existing:
            page.snack_bar = ft.SnackBar(
                ft.Text(
                    "Dataset file not found. Tried:\n" + "\n".join(resolved_list[:4]),
                )
            )
            page.open(page.snack_bar)
            await safe_update(page)
            return

        # Read raw text content
        try:
            raw_text = await asyncio.to_thread(
                lambda: open(existing, "r", encoding="utf-8").read(),
            )
        except Exception as e:
            page.snack_bar = ft.SnackBar(ft.Text(f"Failed to read {existing}: {e}"))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        # Build a scrollable text area
        text_ctl = ft.Text(
            raw_text,
            size=12,
            no_wrap=False,
            max_lines=None,
            selectable=True,
        )
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

    scrape_actions = ft.Row(
        [
            start_button,
            ft.OutlinedButton("Cancel", icon=ICONS.CANCEL, on_click=on_cancel_scrape),
            ft.TextButton("Reset", icon=ICONS.RESTART_ALT, on_click=on_reset_scrape),
            ft.TextButton("Refresh", icon=ICONS.REFRESH, on_click=on_refresh_scrape),
        ],
        spacing=10,
    )

    def handle_preview_click(_):
        # Immediate feedback that click fired
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Opening dataset preview..."))
            page.snack_bar.open = True
            page.update()
        except Exception:
            pass
        _schedule_task(page, on_preview_dataset)

    def handle_raw_preview_click(_):
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Opening raw dataset..."))
            page.snack_bar.open = True
            page.update()
        except Exception:
            pass
        _schedule_task(page, on_preview_raw_dataset)

    # Rows that are toggled by update_source_controls()
    reddit_params_row = ft.Row([reddit_url, reddit_max_posts], wrap=True, visible=False)
    se_params_row = ft.Row([se_site], wrap=True, visible=False)
    synthetic_params_row = ft.Column(
        [
            ft.Row([synthetic_files, synthetic_browse_btn], wrap=True),
            ft.Row(
                [
                    synthetic_gen_type,
                    synthetic_num_pairs,
                    synthetic_max_chunks,
                ],
                wrap=True,
            ),
            ft.Row(
                [
                    synthetic_curate_cb,
                    synthetic_threshold,
                    synthetic_multimodal_cb,
                ],
                wrap=True,
            ),
            ft.Row([synthetic_model], wrap=True),
        ],
        spacing=8,
        visible=False,
    )

    # Wire up validation when synthetic files change
    synthetic_files.on_change = lambda e: update_board_validation()

    # Initialize source-specific visibility now that rows exist
    try:
        update_source_controls()
    except Exception:
        pass

    # Compose Scrape tab via builder
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
        synthetic_params_row=synthetic_params_row,
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

    return scrape_tab
