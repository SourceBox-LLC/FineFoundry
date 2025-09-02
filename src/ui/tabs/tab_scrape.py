"""Scrape tab builder for FineFoundry.

This module composes the Scrape tab UI using controls created in main.py.
No logic changes; only layout composition is centralized here.
"""
from __future__ import annotations

import flet as ft


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
    max_threads,
    max_pairs,
    delay,
    min_len,
    output_path,
    pair_mode,
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
) -> ft.Container:
    return ft.Container(
        content=ft.Row([
            ft.Container(
                content=ft.Column([
                    section_title(
                        "Source",
                        ICONS.DASHBOARD,
                        "Choose a data source. Options: 4chan, Reddit, StackExchange.",
                        on_help_click=_mk_help_handler("Choose a data source. Options: 4chan, Reddit, StackExchange."),
                    ),
                    ft.Container(
                        content=ft.Column([
                            ft.Row([source_dd], wrap=True),
                        ], spacing=0),
                        width=1000,
                        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                        border_radius=8,
                        padding=10,
                    ),
                    section_title(
                        "4chan Boards",
                        ICONS.DASHBOARD,
                        "Select which 4chan boards to scrape.",
                        on_help_click=_mk_help_handler("Select which 4chan boards to scrape."),
                    ),
                    ft.Container(
                        content=ft.Column([
                            board_actions,
                            boards_wrap,
                            board_warning,
                        ], spacing=6),
                        width=1000,
                        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                        border_radius=8,
                        padding=10,
                    ),
                    section_title(
                        "Parameters",
                        ICONS.TUNE,
                        "Set scraping limits and pairing behavior. Context options appear when applicable.",
                        on_help_click=_mk_help_handler("Set scraping limits and pairing behavior. Context options appear when applicable."),
                    ),
                    ft.Container(
                        content=ft.Column([
                            reddit_params_row,
                            se_params_row,
                            ft.Row([max_threads, max_pairs, delay, min_len, output_path], wrap=True),
                            ft.Row([pair_mode, strategy_dd, k_field, max_chars_field], wrap=True),
                            ft.Row([merge_same_id_cb, require_question_cb], wrap=True),
                            scrape_actions,
                        ], spacing=8),
                        width=1000,
                        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                        border_radius=8,
                        padding=10,
                    ),
                    section_title(
                        "Progress",
                        ICONS.TIMELAPSE,
                        "Shows current task progress and counters.",
                        on_help_click=_mk_help_handler("Shows current task progress and counters."),
                    ),
                    ft.Container(
                        content=ft.Column([
                            ft.Row([scrape_prog, working_ring, ft.Text("Working...")], spacing=16),
                            stats_cards,
                            ft.Row([threads_label, pairs_label], spacing=20),
                        ], spacing=8),
                        width=1000,
                        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                        border_radius=8,
                        padding=10,
                    ),
                    section_title(
                        "Live Log",
                        ICONS.TERMINAL,
                        "Streaming log of scraping activity.",
                        on_help_click=_mk_help_handler("Streaming log of scraping activity."),
                    ),
                    ft.Container(
                        log_area,
                        height=180,
                        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                        border_radius=8,
                        padding=10,
                        width=1000,
                    ),
                    section_title(
                        "Preview",
                        ICONS.PREVIEW,
                        "Quick sample preview of scraped pairs.",
                        on_help_click=_mk_help_handler("Quick sample preview of scraped pairs."),
                    ),
                    ft.Container(
                        content=ft.Column([
                            ft.Container(
                                preview_area,
                                height=240,
                                border_radius=8,
                                border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                                padding=6,
                            ),
                            ft.Row([
                                ft.ElevatedButton(
                                    "Preview Dataset",
                                    icon=ICONS.PREVIEW,
                                    on_click=handle_preview_click,
                                )
                            ], alignment=ft.MainAxisAlignment.END),
                        ], spacing=8),
                        width=1000,
                        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                        border_radius=8,
                        padding=10,
                    ),
                ], scroll=ft.ScrollMode.AUTO, spacing=12),
                width=1000,
            )
        ], alignment=ft.MainAxisAlignment.CENTER),
        padding=16,
    )
