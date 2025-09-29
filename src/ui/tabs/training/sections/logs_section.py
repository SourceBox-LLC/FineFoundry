"""Pod logs section builder for FineFoundry.

This module composes the Progress & Logs section using controls created in main.py.
No logic changes; only layout composition is centralized here.
"""
from __future__ import annotations

import flet as ft


def build_pod_logs_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    train_progress: ft.ProgressBar,
    train_prog_label: ft.Text,
    train_timeline: ft.ListView,
    train_timeline_placeholder: ft.Text,
    mk_help_handler,
) -> ft.Container:
    return ft.Container(
        content=ft.Column([
            section_title(
                "Progress & Logs",
                getattr(ICONS, "TASK_ALT", getattr(ICONS, "LIST", ICONS.DESCRIPTION)),
                "Pod status updates and training logs.",
                on_help_click=mk_help_handler("Pod status updates and training logs."),
            ),
            ft.Row([train_progress, train_prog_label], spacing=12),
            ft.Container(
                ft.Stack([train_timeline, train_timeline_placeholder], expand=True),
                height=240,
                width=1000,
                border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                border_radius=8,
                padding=10,
            ),
        ], spacing=12),
        visible=True,
    )
