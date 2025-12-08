"""Training tab: Target selector header section builder."""

from __future__ import annotations

import flet as ft


def build_target_selector_section(*, train_target_dd: ft.Dropdown) -> ft.Container:
    return ft.Container(
        content=ft.Row([train_target_dd], alignment=ft.MainAxisAlignment.CENTER),
        width=1000,
        padding=ft.padding.only(top=12),
    )
