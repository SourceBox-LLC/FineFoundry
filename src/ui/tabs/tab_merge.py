"""Merge Datasets tab builder for FineFoundry.

This module composes the Merge tab UI using controls created in main.py.
Delegates to per-section builders under `ui.tabs.merge.sections`.
No behavior changes.
"""
from __future__ import annotations

import flet as ft

from ui.tabs.merge.sections.operation_section import build_operation_section
from ui.tabs.merge.sections.datasets_section import build_datasets_section
from ui.tabs.merge.sections.output_section import build_output_section
from ui.tabs.merge.sections.preview_section import build_preview_section
from ui.tabs.merge.sections.status_section import build_status_section

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
    op_section = build_operation_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        merge_op=merge_op,
    )

    datasets_section = build_datasets_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        add_row_btn=add_row_btn,
        clear_btn=clear_btn,
        rows_host=rows_host,
    )

    output_section = build_output_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        merge_output_format=merge_output_format,
        merge_save_dir=merge_save_dir,
        merge_actions=merge_actions,
    )

    preview_section = build_preview_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        merge_preview_host=merge_preview_host,
        merge_preview_placeholder=merge_preview_placeholder,
    )

    status_section = build_status_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        merge_timeline=merge_timeline,
        merge_timeline_placeholder=merge_timeline_placeholder,
    )

    return ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Container(
                    content=ft.Column([
                        op_section,
                        datasets_section,
                        output_section,
                        preview_section,
                        ft.Divider(),
                        status_section,
                    ], spacing=12),
                    width=1000,
                )
            ], alignment=ft.MainAxisAlignment.CENTER)
        ], scroll=ft.ScrollMode.AUTO, spacing=0),
        padding=16,
    )
