"""Training tab builder for FineFoundry.

This module composes the Training tab UI using controls created in main.py.
No logic changes; only layout composition is centralized here.
"""
from __future__ import annotations

import flet as ft


def build_training_tab(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    train_target_dd: ft.Dropdown,
    pod_content_container: ft.Container,
    local_specs_container: ft.Container,
) -> ft.Container:
    """Assemble the Training tab layout.

    Note: All logic and dynamic visibility handlers remain in main.py. This function
    only arranges already-constructed containers and controls.
    """
    return ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Container(
                    content=ft.Row([train_target_dd], alignment=ft.MainAxisAlignment.CENTER),
                    width=1000,
                    padding=ft.padding.only(top=12),
                )
            ], alignment=ft.MainAxisAlignment.CENTER),
            pod_content_container,
            local_specs_container,
        ], scroll=ft.ScrollMode.AUTO, spacing=0),
        padding=16,
    )
