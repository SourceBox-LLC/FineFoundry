"""Merge tab: Operation selector section builder."""

from __future__ import annotations

import flet as ft


def build_operation_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    merge_op: ft.Dropdown,
) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Operation",
                    getattr(ICONS, "SHUFFLE", ICONS.SETTINGS),
                    "Choose how to merge rows (e.g., concatenate).",
                    on_help_click=_mk_help_handler("Choose how to merge rows (e.g., concatenate)."),
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Row([merge_op], wrap=True),
                        ],
                        spacing=0,
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
