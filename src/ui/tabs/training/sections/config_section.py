"""Configuration section builder for FineFoundry Training tab.

Composes the config panel using controls created in main.py. Layout-only.
"""

from __future__ import annotations

import flet as ft


def build_config_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    config_mode_dd: ft.Dropdown,
    config_files_row: ft.Row,
    config_summary_txt: ft.Text,
    rp_infra_compact_row: ft.Row,
) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Configuration",
                    getattr(ICONS, "SETTINGS_SUGGEST", ICONS.SETTINGS),
                    "Save or load training configs to streamline repeated runs.",
                    on_help_click=_mk_help_handler("Save or load training configs to streamline repeated runs."),
                ),
                ft.Row([config_mode_dd], wrap=True),
                config_files_row,
                config_summary_txt,
                rp_infra_compact_row,
                ft.Divider(),
            ],
            spacing=8,
        ),
        visible=True,
    )
