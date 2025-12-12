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
    log_actions: ft.Control | None = None,
) -> ft.Container:
    children: list[ft.Control] = [
        section_title(
            "Live Log",
            getattr(ICONS, "TERMINAL", ICONS.DESCRIPTION),
            "Streaming log of scraping activity.",
            on_help_click=_mk_help_handler("Streaming log of scraping activity."),
        )
    ]

    if log_actions is not None:
        children.append(
            ft.Row(
                [log_actions],
                alignment=ft.MainAxisAlignment.END,
            )
        )

    children.append(
        ft.Container(
            log_area,
            height=180,
            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
            border_radius=8,
            padding=10,
            width=1000,
        )
    )

    return ft.Container(
        content=ft.Column(
            children,
            spacing=12,
        ),
        width=1000,
    )
