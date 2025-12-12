"""Merge tab: Datasets section builder."""

from __future__ import annotations

import flet as ft


def build_datasets_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    add_row_btn: ft.TextButton,
    clear_btn: ft.TextButton,
    rows_container: ft.Container,
) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Datasets",
                    getattr(ICONS, "TABLE_VIEW", ICONS.LIST),
                    "Add datasets from HF or local JSON and map columns.",
                    on_help_click=_mk_help_handler("Add datasets from HF or local JSON and map columns."),
                ),
                ft.Row([add_row_btn, clear_btn], spacing=8),
                rows_container,
                ft.Divider(),
            ],
            spacing=12,
        ),
        width=1000,
    )
