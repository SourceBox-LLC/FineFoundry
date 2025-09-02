"""Build/Publish tab builder for FineFoundry.

This module composes the Build/Publish tab UI using controls created in main.py.
No logic changes; only layout composition is centralized here.
"""
from __future__ import annotations

import flet as ft


def build_build_tab(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    # Dataset params
    source_mode,
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
    split_badges,
    # Push controls & actions
    push_toggle,
    repo_id,
    private,
    token_val_ui,
    build_actions,
    # Model card creator
    use_custom_card,
    card_preview_switch,
    load_template_btn,
    gen_from_ds_btn,
    gen_with_ollama_btn,
    clear_card_btn,
    ollama_gen_status,
    card_editor,
    card_preview_container,
    # Status
    timeline,
    timeline_placeholder,
) -> ft.Container:
    return ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Container(
                    content=ft.Column([
                        section_title(
                            "Dataset Params",
                            ICONS.SETTINGS,
                            "Choose input source, preprocessing, and output path for building a dataset.",
                            on_help_click=_mk_help_handler("Choose input source, preprocessing, and output path for building a dataset."),
                        ),
                        ft.Container(
                            content=ft.Column([
                                ft.Row([source_mode, data_file, merged_dir, seed, shuffle, min_len_b, save_dir], wrap=True),
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
                        section_title(
                            "Push to Hub",
                            ICONS.PUBLIC,
                            "Optionally upload your dataset to the Hugging Face Hub.",
                            on_help_click=_mk_help_handler("Optionally upload your dataset to the Hugging Face Hub."),
                        ),
                        ft.Container(
                            content=ft.Column([
                                ft.Row([push_toggle, repo_id, private, token_val_ui], wrap=True),
                                build_actions,
                            ], spacing=10),
                            width=1000,
                            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                            border_radius=8,
                            padding=10,
                        ),
                        section_title(
                            "Model Card Creator",
                            ICONS.ARTICLE,
                            "Draft and preview the README dataset card; can generate from template or dataset.",
                            on_help_click=_mk_help_handler("Draft and preview the README dataset card; can generate from template or dataset."),
                        ),
                        ft.Container(
                            content=ft.Column([
                                ft.Row([use_custom_card, card_preview_switch], wrap=True),
                                ft.Row([load_template_btn, gen_from_ds_btn, gen_with_ollama_btn, clear_card_btn], wrap=True),
                                ft.Row([ollama_gen_status], wrap=True),
                                ft.Container(
                                    ft.Column([card_editor], scroll=ft.ScrollMode.AUTO, spacing=0),
                                    height=300,
                                    border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                                    border_radius=8,
                                    padding=8,
                                ),
                                card_preview_container,
                            ], spacing=10),
                            width=1000,
                            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                            border_radius=8,
                            padding=10,
                        ),
                        section_title(
                            "Status",
                            ICONS.TASK,
                            "Build timeline with step-by-step status.",
                            on_help_click=_mk_help_handler("Build timeline with step-by-step status."),
                        ),
                        ft.Container(
                            ft.Stack([timeline, timeline_placeholder], expand=True),
                            height=260,
                            width=1000,
                            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                            border_radius=8,
                            padding=10,
                        ),
                    ], spacing=12),
                    width=1000,
                )
            ], alignment=ft.MainAxisAlignment.CENTER)
        ], scroll=ft.ScrollMode.AUTO, spacing=0),
        padding=16,
    )
