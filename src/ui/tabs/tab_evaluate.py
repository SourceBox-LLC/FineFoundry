"""Evaluate tab builder for FineFoundry.

This module composes the Evaluate tab UI using controls created in evaluate_controller.py.
Provides systematic model evaluation using lm-evaluation-harness benchmarks.
"""

from __future__ import annotations

import flet as ft


def build_evaluate_tab(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    # Model selection
    eval_status: ft.Text,
    eval_base_model_tf: ft.TextField,
    eval_training_run_dd: ft.Dropdown,
    eval_training_run_refresh_btn: ft.IconButton,
    # Benchmark selection
    eval_benchmark_dd: ft.Dropdown,
    eval_num_samples_tf: ft.TextField,
    eval_batch_size_tf: ft.TextField,
    # Comparison mode
    eval_compare_cb: ft.Checkbox,
    # Action buttons
    eval_run_btn: ft.ElevatedButton,
    eval_stop_btn: ft.ElevatedButton,
    eval_busy_ring: ft.ProgressRing,
    # Progress
    eval_progress: ft.ProgressBar,
    eval_progress_label: ft.Text,
    # Results
    eval_results_container: ft.Container,
    eval_comparison_container: ft.Container,
) -> ft.Container:
    """Build the Evaluate tab layout."""

    # Model selection section
    model_section = ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Model Selection",
                    getattr(ICONS, "PSYCHOLOGY", getattr(ICONS, "SMART_TOY", ICONS.PLAY_CIRCLE)),
                    "Select a fine-tuned model to evaluate against benchmarks.",
                    on_help_click=_mk_help_handler(
                        "Choose a completed training run to evaluate. The base model and LoRA adapter "
                        "will be loaded for benchmark testing. Enable comparison mode to see how your "
                        "fine-tuned model compares to the base model."
                    ),
                ),
                eval_status,
                ft.Row(
                    [eval_training_run_dd, eval_training_run_refresh_btn],
                    spacing=6,
                ),
                ft.Row(
                    [eval_base_model_tf],
                    wrap=True,
                    spacing=10,
                ),
                ft.Row(
                    [eval_compare_cb],
                    spacing=10,
                ),
            ],
            spacing=6,
        ),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        padding=10,
    )

    # Benchmark configuration section
    benchmark_section = ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Benchmark Configuration",
                    getattr(ICONS, "ASSESSMENT", getattr(ICONS, "ANALYTICS", ICONS.INSIGHTS)),
                    "Select which benchmarks to run and configure evaluation parameters.",
                    on_help_click=_mk_help_handler(
                        "Available benchmarks:\n\n"
                        "• TruthfulQA - Tests truthfulness and factual accuracy\n"
                        "• HellaSwag - Tests commonsense reasoning\n"
                        "• ARC (Easy/Challenge) - Science question answering\n"
                        "• MMLU (subset) - Multi-task language understanding\n"
                        "• Winogrande - Pronoun resolution and reasoning\n\n"
                        "Limit samples for faster evaluation during testing."
                    ),
                ),
                ft.Row(
                    [
                        eval_benchmark_dd,
                        eval_num_samples_tf,
                        eval_batch_size_tf,
                    ],
                    spacing=10,
                    wrap=True,
                ),
            ],
            spacing=6,
        ),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        padding=10,
    )

    # Action buttons and progress
    action_section = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        eval_run_btn,
                        eval_stop_btn,
                        eval_busy_ring,
                    ],
                    spacing=10,
                ),
                eval_progress_label,
                eval_progress,
            ],
            spacing=8,
        ),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        padding=10,
    )

    # Results section
    results_section = ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Evaluation Results",
                    getattr(ICONS, "LEADERBOARD", getattr(ICONS, "SCORE", ICONS.STAR)),
                    "Benchmark scores and detailed metrics.",
                    on_help_click=_mk_help_handler(
                        "Results show accuracy/score for each benchmark task. "
                        "Higher scores are better. Compare your fine-tuned model "
                        "against the base model to see if training improved performance."
                    ),
                ),
                eval_results_container,
                eval_comparison_container,
            ],
            spacing=10,
        ),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        padding=10,
    )

    return ft.Container(
        content=ft.Row(
            [
                ft.Container(
                    content=ft.Column(
                        [
                            model_section,
                            benchmark_section,
                            action_section,
                            results_section,
                        ],
                        spacing=12,
                        scroll=ft.ScrollMode.AUTO,
                    ),
                    width=1000,
                )
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        padding=16,
    )
