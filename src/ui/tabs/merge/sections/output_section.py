"""Merge tab: Output section builder."""

from __future__ import annotations

import flet as ft


def build_output_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    merge_output_format: ft.Dropdown,
    merge_save_dir: ft.TextField,
    merge_actions: ft.Row,
) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Output",
                    getattr(ICONS, "SAVE_ALT", ICONS.SAVE),
                    "Set output format and save directory.",
                    on_help_click=_mk_help_handler("Set output format and save directory."),
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Row([merge_output_format, merge_save_dir], wrap=True),
                            merge_actions,
                        ],
                        spacing=10,
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
