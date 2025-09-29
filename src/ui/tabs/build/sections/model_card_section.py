"""Build tab: Model Card Creator section builder."""
from __future__ import annotations

import flet as ft


def build_model_card_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    use_custom_card,
    card_preview_switch,
    load_template_btn,
    gen_from_ds_btn,
    gen_with_ollama_btn,
    clear_card_btn,
    ollama_gen_status,
    card_editor,
    card_preview_container,
) -> ft.Container:
    return ft.Container(
        content=ft.Column([
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
        ], spacing=12),
        width=1000,
    )
