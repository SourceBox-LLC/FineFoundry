"""Build tab: Status section builder."""
from __future__ import annotations

import flet as ft


def build_status_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    timeline,
    timeline_placeholder,
) -> ft.Container:
    return ft.Container(
        content=ft.Column([
            section_title(
                "Status",
                ICONS.TASK,
                "Build timeline with step-by-step status.",
                on_help_click=_mk_help_handler("Build timeline with step-by-step status."),
            ),
            ft.Container(
                ft.Stack([timeline, timeline_placeholder], expand=True),
                height=260,
                width=1000,
                border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                border_radius=8,
                padding=10,
            ),
        ], spacing=12),
        width=1000,
    )
