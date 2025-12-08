"""Scrape tab: Boards selection section builder."""

from __future__ import annotations

import flet as ft


def build_boards_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    board_actions,
    boards_wrap,
    board_warning,
) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                section_title(
                    "4chan Boards",
                    getattr(ICONS, "DASHBOARD", ICONS.SETTINGS),
                    "Select which 4chan boards to scrape.",
                    on_help_click=_mk_help_handler("Select which 4chan boards to scrape."),
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            board_actions,
                            boards_wrap,
                            board_warning,
                        ],
                        spacing=6,
                    ),
                    width=1000,
                    border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                    border_radius=8,
                    padding=10,
                ),
            ],
            spacing=12,
        ),
        width=1000,
    )
