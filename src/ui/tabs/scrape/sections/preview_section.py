"""Scrape tab: Preview section builder."""
from __future__ import annotations

import flet as ft


def build_preview_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    preview_area,
    handle_preview_click,
) -> ft.Container:
    return ft.Container(
        content=ft.Column([
            section_title(
                "Preview",
                getattr(ICONS, "PREVIEW", ICONS.SEARCH),
                "Quick sample preview of the selected dataset format. ChatML: first user→assistant turn; Standard: raw input/output pairs.",
                on_help_click=_mk_help_handler("Quick sample preview of the selected dataset format. ChatML: first user→assistant turn; Standard: raw input/output pairs."),
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
                            icon=getattr(ICONS, "PREVIEW", ICONS.SEARCH),
                            on_click=handle_preview_click,
                        )
                    ], alignment=ft.MainAxisAlignment.END),
                ], spacing=8),
                width=1000,
                border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                border_radius=8,
                padding=10,
            ),
        ], spacing=12),
        width=1000,
    )
