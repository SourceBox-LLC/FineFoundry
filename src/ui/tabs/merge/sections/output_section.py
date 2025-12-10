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
    merge_session_name: ft.TextField,
    merge_export_path: ft.TextField,
    merge_actions: ft.Row,
) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Output",
                    getattr(ICONS, "SAVE_ALT", ICONS.SAVE),
                    "Save merged dataset to database.",
                    on_help_click=_mk_help_handler(
                        "Merged datasets are saved to the database. Optionally export to JSON."
                    ),
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Row([merge_output_format, merge_session_name], wrap=True),
                            merge_export_path,
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
