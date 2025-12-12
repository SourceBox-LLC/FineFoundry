"""Inference tab builder for FineFoundry.

This module composes the Inference tab UI using controls created in main.py.
No behavior changes; only layout composition is centralized here.
"""

from __future__ import annotations

import flet as ft


def build_inference_tab(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    infer_status: ft.Text,
    infer_meta: ft.Text,
    infer_base_model_tf: ft.TextField,
    infer_training_run_dd: ft.Dropdown,
    infer_training_run_refresh_btn: ft.IconButton,
    infer_use_latest_btn: ft.Control,
    infer_preset_dd: ft.Dropdown,
    infer_temp_slider: ft.Slider,
    infer_temp_label: ft.Text,
    infer_max_tokens_slider: ft.Slider,
    infer_max_tokens_label: ft.Text,
    infer_rep_penalty_slider: ft.Slider,
    infer_rep_penalty_label: ft.Text,
    infer_prompt_tf: ft.TextField,
    infer_expected_tf: ft.TextField,
    infer_dataset_dd: ft.Dropdown,
    infer_dataset_refresh_btn: ft.IconButton,
    infer_sample_prompts_dd: ft.Dropdown,
    infer_sample_refresh_btn: ft.IconButton,
    infer_generate_btn: ft.ElevatedButton,
    infer_clear_btn: ft.TextButton,
    infer_export_btn: ft.TextButton,
    infer_busy_ring: ft.ProgressRing,
    infer_output: ft.ListView,
    infer_output_placeholder: ft.Text,
    infer_full_chat_btn: ft.Control,
) -> ft.Container:
    model_section = ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Model & adapter",
                    getattr(ICONS, "PSYCHOLOGY", getattr(ICONS, "CHAT", ICONS.PLAY_CIRCLE)),
                    "Select a completed training run to load for inference.",
                    on_help_click=_mk_help_handler(
                        "Choose a completed training run from the database. The adapter and base model "
                        "will be loaded automatically for inference.",
                    ),
                ),
                infer_status,
                infer_meta,
                ft.Row(
                    [infer_training_run_dd, infer_training_run_refresh_btn],
                    spacing=6,
                ),
                ft.Row(
                    [infer_base_model_tf],
                    wrap=True,
                    spacing=10,
                ),
                ft.Row(
                    [infer_use_latest_btn],
                    wrap=True,
                    spacing=10,
                ),
            ],
            spacing=6,
        ),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        padding=10,
    )

    generation_section = ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Prompt & responses",
                    getattr(ICONS, "CHAT", getattr(ICONS, "PSYCHOLOGY", ICONS.PLAY_CIRCLE)),
                    "Enter a prompt and generate responses from the selected model.",
                    on_help_click=_mk_help_handler(
                        "Controls for temperature, max new tokens, and presets let you quickly explore how "
                        "your fine-tuned model behaves.",
                    ),
                ),
                ft.Row(
                    [infer_preset_dd],
                    wrap=True,
                    spacing=10,
                ),
                ft.Row(
                    [infer_dataset_dd, infer_dataset_refresh_btn],
                    spacing=4,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Row(
                    [infer_sample_prompts_dd, infer_sample_refresh_btn],
                    spacing=4,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                infer_prompt_tf,
                infer_expected_tf,
                ft.Row(
                    [
                        ft.Column([infer_temp_label, infer_temp_slider], spacing=2),
                        ft.Column([infer_max_tokens_label, infer_max_tokens_slider], spacing=2),
                        ft.Column([infer_rep_penalty_label, infer_rep_penalty_slider], spacing=2),
                    ],
                    wrap=True,
                    spacing=10,
                ),
                ft.Row(
                    [infer_generate_btn, infer_clear_btn, infer_export_btn, infer_busy_ring],
                    wrap=True,
                    spacing=10,
                ),
                ft.Container(
                    ft.Stack([infer_output, infer_output_placeholder], expand=True),
                    height=260,
                    width=1000,
                    border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                    border_radius=8,
                    padding=10,
                ),
                ft.Row(
                    [infer_full_chat_btn],
                    alignment=ft.MainAxisAlignment.END,
                ),
            ],
            spacing=8,
        ),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        padding=10,
    )

    return ft.Container(
        content=ft.Row(
            [
                ft.Container(
                    content=ft.Column(
                        [model_section, generation_section],
                        spacing=12,
                        scroll=ft.ScrollMode.AUTO,
                    ),
                    width=1000,
                )
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        padding=16,
    )
