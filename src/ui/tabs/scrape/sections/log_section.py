"""Scrape tab: Live Log section builder."""
from __future__ import annotations

import flet as ft


def build_log_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    log_area,
) -> ft.Container:
    return ft.Container(
        content=ft.Column([
            section_title(
                "Live Log",
                getattr(ICONS, "TERMINAL", ICONS.DESCRIPTION),
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
        ], spacing=12),
        width=1000,
    )
