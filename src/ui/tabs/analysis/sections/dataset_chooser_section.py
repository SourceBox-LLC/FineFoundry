"""Analysis tab: Dataset chooser section builder."""

from __future__ import annotations

import flet as ft


def build_dataset_chooser_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    analysis_source_dd: ft.Dropdown,
    analysis_hf_repo: ft.TextField,
    analysis_hf_split: ft.TextField,
    analysis_hf_config: ft.TextField,
    analysis_json_path: ft.TextField,
    analysis_dataset_hint: ft.Text,
) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        analysis_source_dd,
                        analysis_hf_repo,
                        analysis_hf_split,
                        analysis_hf_config,
                        analysis_json_path,
                    ],
                    wrap=True,
                    spacing=10,
                ),
                analysis_dataset_hint,
            ],
            spacing=6,
        ),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
        border_radius=8,
        padding=10,
    )
