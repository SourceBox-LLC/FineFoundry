"""Scrape tab: Progress section builder."""
from __future__ import annotations

import flet as ft


def build_progress_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    scrape_prog,
    working_ring,
    stats_cards,
    threads_label,
    pairs_label,
) -> ft.Container:
    return ft.Container(
        content=ft.Column([
            section_title(
                "Progress",
                getattr(ICONS, "TIMELAPSE", ICONS.INFO),
                "Shows current task progress and counters.",
                on_help_click=_mk_help_handler("Shows current task progress and counters."),
            ),
            ft.Container(
                content=ft.Column([
                    ft.Row([scrape_prog, working_ring, ft.Text("Working...")], spacing=16),
                    stats_cards,
                    ft.Row([threads_label, pairs_label], spacing=20),
                ], spacing=8),
                width=1000,
                border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                border_radius=8,
                padding=10,
            ),
        ], spacing=12),
        width=1000,
    )
