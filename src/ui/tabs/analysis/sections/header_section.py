"""Analysis tab: Header (title + analyze button) section builder."""
from __future__ import annotations

import flet as ft


def build_header_section(
    *,
    section_title,
    ICONS,
    _mk_help_handler,
    analyze_btn: ft.ElevatedButton,
    analysis_busy_ring: ft.ProgressRing,
) -> ft.Container:
    return ft.Container(
        content=ft.Row([
            section_title(
                "Dataset Analysis",
                getattr(ICONS, "INSIGHTS", getattr(ICONS, "ANALYTICS", ICONS.SEARCH)),
                "Run modular analysis on your dataset. Enable modules and click Analyze to compute and reveal results.",
                on_help_click=_mk_help_handler("Run modular analysis on your dataset. Enable modules and click Analyze to compute and reveal results."),
            ),
            ft.Container(expand=1),
            analyze_btn,
            analysis_busy_ring,
        ], alignment=ft.MainAxisAlignment.START),
    )
