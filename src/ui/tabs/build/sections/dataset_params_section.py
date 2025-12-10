"""Build tab: Dataset Params + Splits section builder."""
from __future__ import annotations

import flet as ft


def build_dataset_params_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    # Dataset params
    source_mode,
    data_source_dd,
    db_session_dd,
    db_refresh_btn,
    data_file,
    merged_dir,
    seed,
    shuffle,
    min_len_b,
    save_dir,
    # Splits
    val_slider,
    test_slider,
    split_error,
    split_badges: dict,
) -> ft.Container:
    return ft.Container(
        content=ft.Column([
            section_title(
                "Dataset Params",
                ICONS.SETTINGS,
                "Choose input source, preprocessing, and output path. Data is loaded from database by default.",
                on_help_click=_mk_help_handler("Choose input source, preprocessing, and output path. Data is loaded from database by default."),
            ),
            ft.Container(
                content=ft.Column([
                    ft.Row([source_mode, data_source_dd, db_session_dd, db_refresh_btn, data_file, merged_dir], wrap=True),
                    ft.Row([seed, shuffle, min_len_b, save_dir], wrap=True),
                    ft.Divider(),
                    section_title(
                        "Splits",
                        ICONS.TABLE_VIEW,
                        "Configure validation and test fractions; train is the remainder.",
                        on_help_click=_mk_help_handler("Configure validation and test fractions; train is the remainder."),
                    ),
                    ft.Row([
                        ft.Column([
                            ft.Text("Validation Fraction"), val_slider,
                            ft.Text("Test Fraction"), test_slider,
                            split_error,
                        ], width=360),
                        ft.Row([split_badges["train"], split_badges["val"], split_badges["test"]], spacing=10),
                    ], wrap=True, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ], spacing=12),
                width=1000,
                border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                border_radius=8,
                padding=10,
            ),
        ], spacing=12),
        width=1000,
    )
