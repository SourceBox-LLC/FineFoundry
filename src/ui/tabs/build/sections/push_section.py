"""Build tab: Push to Hub section builder."""
from __future__ import annotations

import flet as ft


def build_push_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    push_toggle,
    repo_id,
    private,
    token_val_ui,
    build_actions,
) -> ft.Container:
    return ft.Container(
        content=ft.Column([
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
        ], spacing=12),
        width=1000,
    )
