"""Merge tab: Preview section builder."""
from __future__ import annotations

import flet as ft


def build_preview_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    merge_preview_host: ft.ListView,
    merge_preview_placeholder: ft.Container,
) -> ft.Container:
    return ft.Container(
        content=ft.Column([
            section_title(
                "Preview",
                getattr(ICONS, "PREVIEW", ICONS.SEARCH),
                "Shows a sample of the merged result.",
                on_help_click=_mk_help_handler("Shows a sample of the merged result."),
            ),
            ft.Container(
                ft.Stack([merge_preview_host, merge_preview_placeholder], expand=True),
                height=220,
                width=1000,
                border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                border_radius=8,
                padding=10,
            ),
        ], spacing=12),
        width=1000,
    )
