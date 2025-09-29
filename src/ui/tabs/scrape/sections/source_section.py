"""Scrape tab: Source selector section builder."""
from __future__ import annotations

import flet as ft


def build_source_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    source_dd,
) -> ft.Container:
    return ft.Container(
        content=ft.Column([
            section_title(
                "Source",
                getattr(ICONS, "DASHBOARD", ICONS.SETTINGS),
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
        ], spacing=12),
        width=1000,
    )
