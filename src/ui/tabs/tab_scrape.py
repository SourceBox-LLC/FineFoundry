"""Scrape tab builder for FineFoundry.

This module composes the Scrape tab UI using controls created in main.py.
Delegates to per-section builders under `ui.tabs.scrape.sections`.
No behavior changes.
"""

from __future__ import annotations

import flet as ft

from ui.tabs.scrape.sections.source_section import build_source_section
from ui.tabs.scrape.sections.boards_section import build_boards_section
from ui.tabs.scrape.sections.params_section import build_params_section
from ui.tabs.scrape.sections.progress_section import build_progress_section
from ui.tabs.scrape.sections.log_section import build_log_section
from ui.tabs.scrape.sections.preview_section import build_preview_section


def build_scrape_tab(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    source_dd,
    board_actions,
    boards_wrap,
    board_warning,
    reddit_params_row,
    se_params_row,
    synthetic_params_row,
    max_threads,
    max_pairs,
    delay,
    min_len,
    output_path,
    dataset_format_dd,
    multiturn_sw,
    strategy_dd,
    k_field,
    max_chars_field,
    merge_same_id_cb,
    require_question_cb,
    scrape_actions,
    scrape_prog,
    working_ring,
    stats_cards,
    threads_label,
    pairs_label,
    log_area,
    preview_area,
    handle_preview_click,
    handle_raw_preview_click,
    # Optional refs the caller can use to toggle visibility of sections at runtime
    boards_section_ref=None,
    progress_section_ref=None,
    log_section_ref=None,
    preview_section_ref=None,
) -> ft.Container:
    source_section = build_source_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        source_dd=source_dd,
    )

    boards_section = build_boards_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        board_actions=board_actions,
        boards_wrap=boards_wrap,
        board_warning=board_warning,
    )

    params_section = build_params_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
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
        scrape_actions=scrape_actions,
    )

    progress_section = build_progress_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        scrape_prog=scrape_prog,
        working_ring=working_ring,
        stats_cards=stats_cards,
        threads_label=threads_label,
        pairs_label=pairs_label,
    )

    log_section = build_log_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        log_area=log_area,
    )

    preview_section = build_preview_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        preview_area=preview_area,
        handle_preview_click=handle_preview_click,
        handle_raw_preview_click=handle_raw_preview_click,
    )

    # Expose section containers to caller (e.g., to control visibility)
    try:
        if boards_section_ref is not None:
            boards_section_ref["control"] = boards_section
    except Exception:
        pass
    try:
        if progress_section_ref is not None:
            progress_section_ref["control"] = progress_section
    except Exception:
        pass
    try:
        if log_section_ref is not None:
            log_section_ref["control"] = log_section
    except Exception:
        pass
    try:
        if preview_section_ref is not None:
            preview_section_ref["control"] = preview_section
    except Exception:
        pass

    return ft.Container(
        content=ft.Row(
            [
                ft.Container(
                    content=ft.Column(
                        [
                            source_section,
                            boards_section,
                            params_section,
                            progress_section,
                            log_section,
                            preview_section,
                        ],
                        scroll=ft.ScrollMode.AUTO,
                        spacing=12,
                    ),
                    width=1000,
                )
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        padding=16,
    )
