"""Merge tab: Status section builder."""

from __future__ import annotations

import flet as ft


def build_status_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    merge_timeline: ft.ListView,
    merge_timeline_placeholder: ft.Container,
    download_button: ft.Control | None = None,
) -> ft.Container:
    controls = [
        section_title(
            "Status",
            getattr(ICONS, "TASK", ICONS.INFO),
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
    ]

    if download_button is not None:
        controls.append(download_button)

    return ft.Container(
        content=ft.Column(controls, spacing=12),
        width=1000,
    )
