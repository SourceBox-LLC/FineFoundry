"""Scrape tab: Parameters section builder."""
from __future__ import annotations

import flet as ft


def build_params_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
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
    scrape_actions,
) -> ft.Container:
    return ft.Container(
        content=ft.Column([
            section_title(
                "Parameters",
                getattr(ICONS, "TUNE", ICONS.SETTINGS),
                "Set scraping/generation limits, path, and output format. For synthetic data, select files and generation type.",
                on_help_click=_mk_help_handler("Set scraping/generation limits, path, and output format. For synthetic data, select files and generation type."),
            ),
            ft.Container(
                content=ft.Column([
                    reddit_params_row,
                    se_params_row,
                    synthetic_params_row,
                    ft.Row([max_threads, max_pairs, delay, min_len, output_path, dataset_format_dd], wrap=True),
                    ft.Row([multiturn_sw], wrap=True),
                    scrape_actions,
                ], spacing=8),
                width=1000,
                border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                border_radius=8,
                padding=10,
            ),
        ], spacing=12),
        width=1000,
    )
