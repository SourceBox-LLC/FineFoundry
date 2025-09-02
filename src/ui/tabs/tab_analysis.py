"""Dataset Analysis tab builder for FineFoundry.

This module composes the Dataset Analysis tab UI using controls created in main.py.
No logic changes; only layout composition is centralized here.
"""
from __future__ import annotations

import flet as ft


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
    """Assemble the Dataset Analysis tab layout using provided controls and containers.

    Note: All logic, data loading, and dynamic visibility handlers remain in main.py.
    This function only arranges already-constructed containers and controls.
    """
    return ft.Container(
        content=ft.Row([
            ft.Container(
                content=ft.Column([
                    ft.Row([
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

                    # Dataset chooser row
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                analysis_source_dd,
                                analysis_hf_repo,
                                analysis_hf_split,
                                analysis_hf_config,
                                analysis_json_path,
                            ], wrap=True, spacing=10),
                            analysis_dataset_hint,
                        ], spacing=6),
                        width=1000,
                        border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
                        border_radius=8,
                        padding=10,
                    ),
                    ft.Divider(),
                    section_title(
                        "Analysis modules",
                        getattr(ICONS, "TUNE", ICONS.SETTINGS),
                        "Choose which checks to run. Only enabled modules are computed and displayed.",
                        on_help_click=_mk_help_handler("Choose which checks to run. Only enabled modules are computed and displayed."),
                    ),
                    ft.Container(
                        content=ft.Column([
                            ft.Row([select_all_modules_cb], wrap=True),
                            _build_modules_table(),
                        ], spacing=6),
                        width=1000,
                        border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
                        border_radius=8,
                        padding=10,
                    ),

                    ft.Divider(),
                    section_title(
                        "Runtime settings",
                        getattr(ICONS, "SETTINGS", getattr(ICONS, "TUNE", ICONS.SETTINGS)),
                        "Backend, token (for private HF datasets), and sampling. Sample size limits records analyzed for speed.",
                        on_help_click=_mk_help_handler("Backend, token (for private HF datasets), and sampling. Sample size limits records analyzed for speed."),
                    ),
                    ft.Container(
                        content=ft.Row([
                            analysis_backend_dd,
                            analysis_hf_token_tf,
                            analysis_sample_size_tf,
                        ], wrap=True, spacing=10),
                        width=1000,
                        border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
                        border_radius=8,
                        padding=10,
                    ),

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
                ], scroll=ft.ScrollMode.AUTO, spacing=12),
                width=1000,
            )
        ], alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.START, expand=1)
    )
