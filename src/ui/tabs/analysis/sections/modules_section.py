"""Analysis tab: Modules selection section builder."""

from __future__ import annotations

import flet as ft


def build_modules_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    select_all_modules_cb: ft.Checkbox,
    _build_modules_table,
) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Analysis modules",
                    getattr(ICONS, "TUNE", ICONS.SETTINGS),
                    "Choose which checks to run. Only enabled modules are computed and displayed.",
                    on_help_click=_mk_help_handler(
                        "Choose which checks to run. Only enabled modules are computed and displayed."
                    ),
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Row([select_all_modules_cb], wrap=True),
                            _build_modules_table(),
                        ],
                        spacing=6,
                    ),
                    width=1000,
                    border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
                    border_radius=8,
                    padding=10,
                ),
            ],
            spacing=12,
        ),
        width=1000,
    )
