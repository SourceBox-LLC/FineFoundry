"""Dataset section builder for FineFoundry Training tab.

Composes the dataset selection block using controls created in main.py.
Layout-only; no logic moved.
"""

from __future__ import annotations

import flet as ft


def build_dataset_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    train_source: ft.Dropdown,
    train_hf_repo: ft.TextField,
    train_hf_split: ft.TextField,
    train_hf_config: ft.TextField,
    train_json_path: ft.TextField,
    train_json_browse_btn: ft.IconButton,
    train_db_session_dd: ft.Dropdown,
    train_db_refresh_btn: ft.IconButton,
    train_db_pair_count: ft.Text,
    visible: bool = False,
) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Dataset",
                    getattr(ICONS, "TABLE_VIEW", getattr(ICONS, "LIST", ICONS.DESCRIPTION)),
                    "Select the dataset for training.",
                    on_help_click=_mk_help_handler("Select the dataset for training."),
                ),
                ft.Row(
                    [
                        train_source,
                        train_hf_repo,
                        train_hf_split,
                        train_hf_config,
                        train_db_session_dd,
                        train_db_refresh_btn,
                        train_db_pair_count,
                        train_json_path,
                        train_json_browse_btn,
                    ],
                    wrap=True,
                ),
                ft.Divider(),
            ],
            spacing=0,
        ),
        visible=visible,
    )
