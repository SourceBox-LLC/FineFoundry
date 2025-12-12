"""Training tab builder for FineFoundry.

This module composes the Training tab UI using controls created in main.py.
No logic changes; only layout composition is centralized here.
"""

from __future__ import annotations

import flet as ft

from ui.tabs.training.sections.header_section import build_target_selector_section


def build_training_tab(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    offline_banner: ft.Container,
    train_target_dd: ft.Dropdown,
    train_target_offline_reason: ft.Text,
    pod_content_container: ft.Container,
    local_specs_container: ft.Container,
) -> ft.Container:
    """Assemble the Training tab layout.

    Note: All logic and dynamic visibility handlers remain in main.py. This function
    only arranges already-constructed containers and controls.
    """
    target_selector = build_target_selector_section(
        train_target_dd=train_target_dd,
        offline_reason=train_target_offline_reason,
    )

    return ft.Container(
        content=ft.Column(
            [
                ft.Row([offline_banner], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([target_selector], alignment=ft.MainAxisAlignment.CENTER),
                pod_content_container,
                local_specs_container,
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=0,
        ),
        padding=16,
    )
