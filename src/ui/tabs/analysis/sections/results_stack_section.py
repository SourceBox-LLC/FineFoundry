"""Analysis tab: Results stack section builder.

Composes the sequence of result blocks and dividers into a single column.
"""
from __future__ import annotations

import flet as ft


def build_results_stack_section(
    *,
    analysis_overview_note: ft.Text,
    div_overview: ft.Divider,
    overview_block: ft.Container,
    div_sentiment: ft.Divider,
    sentiment_block: ft.Container,
    div_class: ft.Divider,
    class_balance_block: ft.Container,
    div_extra: ft.Divider,
    extra_metrics_block: ft.Container,
    div_samples: ft.Divider,
    samples_block: ft.Container,
) -> ft.Column:
    return ft.Column([
        analysis_overview_note,
        div_overview,
        overview_block,
        div_sentiment,
        sentiment_block,
        div_class,
        class_balance_block,
        div_extra,
        extra_metrics_block,
        div_samples,
        samples_block,
    ], spacing=12)
