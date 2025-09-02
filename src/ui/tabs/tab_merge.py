"""Merge Datasets tab builder for FineFoundry.

This module composes the Merge tab UI using controls created in main.py.
No logic changes; only layout composition is centralized here.
"""
from __future__ import annotations

import flet as ft


def build_merge_tab(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    merge_op: ft.Dropdown,
    rows_host: ft.Column,
    add_row_btn: ft.TextButton,
    clear_btn: ft.TextButton,
    merge_output_format: ft.Dropdown,
    merge_save_dir: ft.TextField,
    merge_actions: ft.Row,
    merge_preview_host: ft.ListView,
    merge_preview_placeholder: ft.Container,
    merge_timeline: ft.ListView,
    merge_timeline_placeholder: ft.Container,
) -> ft.Container:
    """Assemble the Merge tab layout using provided controls and containers."""
    return ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Container(
                    content=ft.Column([
                        section_title(
                            "Operation",
                            ICONS.SHUFFLE,
                            "Choose how to merge rows (e.g., concatenate).",
                            on_help_click=_mk_help_handler("Choose how to merge rows (e.g., concatenate)."),
                        ),
                        ft.Container(
                            content=ft.Column([
                                ft.Row([merge_op], wrap=True),
                            ], spacing=0),
                            width=1000,
                            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                            border_radius=8,
                            padding=10,
                        ),
                        section_title(
                            "Datasets",
                            ICONS.TABLE_VIEW,
                            "Add datasets from HF or local JSON and map columns.",
                            on_help_click=_mk_help_handler("Add datasets from HF or local JSON and map columns."),
                        ),
                        ft.Row([
                            add_row_btn, clear_btn
                        ], spacing=8),
                        ft.Container(
                            content=ft.Column([
                                rows_host,
                            ], spacing=10),
                            width=1000,
                            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                            border_radius=8,
                            padding=10,
                        ),
                        section_title(
                            "Output",
                            ICONS.SAVE_ALT,
                            "Set output format and save directory.",
                            on_help_click=_mk_help_handler("Set output format and save directory."),
                        ),
                        ft.Container(
                            content=ft.Column([
                                ft.Row([merge_output_format, merge_save_dir], wrap=True),
                                merge_actions,
                            ], spacing=10),
                            width=1000,
                            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                            border_radius=8,
                            padding=10,
                        ),
                        section_title(
                            "Preview",
                            ICONS.PREVIEW,
                            "Shows a sample of the merged result.",
                            on_help_click=_mk_help_handler("Shows a sample of the merged result."),
                        ),
                        ft.Container(
                            ft.Stack([merge_preview_host, merge_preview_placeholder], expand=True),
                            height=220,
                            width=1000,
                            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                            border_radius=8,
                            padding=10,
                        ),
                        ft.Divider(),
                        section_title(
                            "Status",
                            ICONS.TASK,
                            "Merge timeline and diagnostics.",
                            on_help_click=_mk_help_handler("Merge timeline and diagnostics."),
                        ),
                        ft.Container(
                            ft.Stack([merge_timeline, merge_timeline_placeholder], expand=True),
                            height=200,
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
