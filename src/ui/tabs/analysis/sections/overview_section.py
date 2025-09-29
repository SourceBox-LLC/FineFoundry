"""Analysis tab: Overview results section builder."""
from __future__ import annotations

import flet as ft


def build_overview_section(
    *,
    analysis_overview_note: ft.Text,
    div_overview: ft.Divider,
    overview_block: ft.Container,
) -> ft.Column:
    return ft.Column([
        analysis_overview_note,
        div_overview,
        overview_block,
    ], spacing=8)
