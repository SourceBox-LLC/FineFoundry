"""Dataset Analysis tab builder for FineFoundry.

This module composes the Dataset Analysis tab UI using controls created in main.py.
Delegates to per-section builders under `ui.tabs.analysis.sections`.
No behavior changes.
"""

from __future__ import annotations

import flet as ft

from ui.tabs.analysis.sections.header_section import build_header_section
from ui.tabs.analysis.sections.dataset_chooser_section import build_dataset_chooser_section
from ui.tabs.analysis.sections.modules_section import build_modules_section
from ui.tabs.analysis.sections.runtime_section import build_runtime_section
from ui.tabs.analysis.sections.results_stack_section import build_results_stack_section


def build_analysis_tab(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    # Header actions
    analyze_btn: ft.ElevatedButton,
    analysis_busy_ring: ft.ProgressRing,
    # Dataset chooser controls
    analysis_source_dd: ft.Dropdown,
    analysis_hf_repo: ft.TextField,
    analysis_hf_split: ft.TextField,
    analysis_hf_config: ft.TextField,
    analysis_json_path: ft.TextField,
    analysis_dataset_hint: ft.Text,
    # Modules selection
    select_all_modules_cb: ft.Checkbox,
    _build_modules_table,
    # Runtime settings
    analysis_backend_dd: ft.Dropdown,
    analysis_hf_token_tf: ft.TextField,
    analysis_sample_size_tf: ft.TextField,
    # Results blocks & dividers
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
) -> ft.Container:
    header = build_header_section(
        section_title=section_title,
        ICONS=ICONS,
        _mk_help_handler=_mk_help_handler,
        analyze_btn=analyze_btn,
        analysis_busy_ring=analysis_busy_ring,
    )

    dataset_chooser = build_dataset_chooser_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        analysis_source_dd=analysis_source_dd,
        analysis_hf_repo=analysis_hf_repo,
        analysis_hf_split=analysis_hf_split,
        analysis_hf_config=analysis_hf_config,
        analysis_json_path=analysis_json_path,
        analysis_dataset_hint=analysis_dataset_hint,
    )

    modules_section = build_modules_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        select_all_modules_cb=select_all_modules_cb,
        _build_modules_table=_build_modules_table,
    )

    runtime_section = build_runtime_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        analysis_backend_dd=analysis_backend_dd,
        analysis_hf_token_tf=analysis_hf_token_tf,
        analysis_sample_size_tf=analysis_sample_size_tf,
    )

    results_stack = build_results_stack_section(
        analysis_overview_note=analysis_overview_note,
        div_overview=div_overview,
        overview_block=overview_block,
        div_sentiment=div_sentiment,
        sentiment_block=sentiment_block,
        div_class=div_class,
        class_balance_block=class_balance_block,
        div_extra=div_extra,
        extra_metrics_block=extra_metrics_block,
        div_samples=div_samples,
        samples_block=samples_block,
    )

    return ft.Container(
        content=ft.Row(
            [
                ft.Container(
                    content=ft.Column(
                        [
                            header,
                            dataset_chooser,
                            modules_section,
                            runtime_section,
                            results_stack,
                        ],
                        scroll=ft.ScrollMode.AUTO,
                        spacing=12,
                    ),
                    width=1000,
                )
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.START,
            expand=1,
        )
    )
