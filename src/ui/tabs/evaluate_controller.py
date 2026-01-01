"""Evaluate tab controller for FineFoundry.

This module builds the Evaluate tab controls and wires up benchmark evaluation
using lm-evaluation-harness. Provides systematic model evaluation with
base vs fine-tuned comparison.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

import flet as ft

from helpers.common import safe_update
from helpers.logging_config import get_logger
from helpers.theme import ACCENT_COLOR, BORDER_BASE, COLORS, ICONS
from helpers.ui import WITH_OPACITY
from ui.tabs.tab_evaluate import build_evaluate_tab

logger = get_logger(__name__)

# Available benchmarks with descriptions
BENCHMARKS = {
    # Quick & popular (recommended for testing)
    "hellaswag": "⚡ HellaSwag (quick)",
    "truthfulqa_mc2": "⚡ TruthfulQA MC2",
    "arc_easy": "⚡ ARC Easy",
    "winogrande": "⚡ Winogrande",
    "boolq": "⚡ BoolQ",
    # Full benchmarks
    "arc_challenge": "ARC Challenge",
    "mmlu": "MMLU (57 tasks, slow)",
    "mmlu_pro": "MMLU-PRO (10 choices)",
    "gsm8k": "GSM8K (math)",
    # Advanced / Leaderboard v2
    "ifeval": "IFEval (instruction following)",
    "bbh": "BBH (Big Bench Hard)",
    "gpqa": "GPQA (PhD-level)",
    "musr": "MuSR (multistep reasoning)",
    "humaneval": "HumanEval (code gen)",
}

# Metric descriptions for user-friendly display
METRIC_INFO = {
    "acc": "Accuracy - % of correct answers",
    "acc_norm": "Normalized Accuracy - length-normalized scoring",
    "mc1": "MC1 - Single true answer accuracy",
    "mc2": "MC2 - Multiple true answers (weighted)",
    "exact_match": "Exact Match - perfect answer match",
    "f1": "F1 Score - harmonic mean of precision/recall",
}


def build_evaluate_tab_with_logic(
    page: ft.Page,
    *,
    section_title,
    _mk_help_handler,
    train_state: Dict[str, Any],
) -> ft.Control:
    """Build the Evaluate tab and attach all related handlers."""

    # Track evaluation state
    eval_state: Dict[str, Any] = {
        "running": False,
        "process": None,
        "results": {},
        "base_results": {},
    }

    # Status text
    eval_status = ft.Text(
        "Select a completed training run to evaluate.",
        color=WITH_OPACITY(0.6, BORDER_BASE),
    )

    # Model selection
    eval_base_model_tf = ft.TextField(
        label="Base model",
        value="",
        width=520,
        dense=True,
        read_only=True,
        tooltip="Base model will be auto-filled from selected training run",
    )

    eval_training_run_dd = ft.Dropdown(
        label="Training run to evaluate",
        options=[],
        width=520,
        tooltip="Select a completed training run to evaluate",
    )

    eval_training_run_refresh_btn = ft.IconButton(
        icon=ft.Icons.REFRESH,
        tooltip="Refresh training runs",
    )

    # Benchmark selection
    eval_benchmark_dd = ft.Dropdown(
        label="Benchmark",
        options=[ft.dropdown.Option(key=k, text=v) for k, v in BENCHMARKS.items()],
        value="truthfulqa_mc2",
        width=400,
        tooltip="Select which benchmark to run",
    )

    eval_num_samples_tf = ft.TextField(
        label="Max samples",
        value="100",
        width=120,
        keyboard_type=ft.KeyboardType.NUMBER,
        tooltip="Limit samples for faster evaluation (empty = full benchmark)",
    )

    eval_batch_size_tf = ft.TextField(
        label="Batch size",
        value="4",
        width=100,
        keyboard_type=ft.KeyboardType.NUMBER,
        tooltip="Batch size for evaluation (lower if OOM)",
    )

    # Comparison mode (optional - disabled by default to save time/memory)
    eval_compare_cb = ft.Checkbox(
        label="Also evaluate base model for comparison (doubles eval time)",
        value=False,
        tooltip="Optional: Run benchmark on base model too to see if fine-tuning improved performance",
    )

    # Action buttons
    eval_run_btn = ft.ElevatedButton(
        "Run Evaluation",
        icon=ft.Icons.PLAY_ARROW,
        bgcolor=ACCENT_COLOR,
        color=ft.Colors.WHITE,
    )

    eval_stop_btn = ft.ElevatedButton(
        "Stop",
        icon=ft.Icons.STOP,
        disabled=True,
    )

    eval_busy_ring = ft.ProgressRing(width=20, height=20, visible=False)

    # Progress
    eval_progress = ft.ProgressBar(value=0, width=600, visible=False)
    eval_progress_label = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))

    # Results containers
    eval_results_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Metric")),
            ft.DataColumn(ft.Text("Fine-tuned"), numeric=True),
            ft.DataColumn(ft.Text("Base Model"), numeric=True),
            ft.DataColumn(ft.Text("Δ Change"), numeric=True),
        ],
        rows=[],
        visible=False,
    )

    eval_results_placeholder = ft.Text(
        "Run an evaluation to see results here.",
        color=WITH_OPACITY(0.5, BORDER_BASE),
        italic=True,
    )

    eval_results_container = ft.Container(
        content=ft.Column([eval_results_placeholder, eval_results_table]),
        padding=10,
    )

    eval_comparison_chart = ft.Container(visible=False)  # Placeholder for future chart
    eval_comparison_container = ft.Container(
        content=eval_comparison_chart,
    )

    # --- Handlers ---

    def _refresh_training_runs(_=None):
        """Refresh the training runs dropdown with completed runs."""
        try:
            from db.training_runs import list_training_runs

            runs = list_training_runs(status="completed", limit=50)
            options = []
            for r in runs:
                adapter_path = r.get("adapter_path", "")
                if adapter_path and os.path.isdir(adapter_path):
                    label = f"✅ {r['name']} - {r['base_model'].split('/')[-1] if r.get('base_model') else 'unknown'}"
                    options.append(ft.dropdown.Option(key=str(r["id"]), text=label))

            eval_training_run_dd.options = options
            if options:
                eval_status.value = f"Found {len(options)} completed training runs."
                eval_status.color = COLORS.GREEN
            else:
                eval_status.value = "No completed training runs found. Train a model first."
                eval_status.color = COLORS.ORANGE
            page.update()
        except Exception as e:
            logger.error(f"Failed to refresh training runs: {e}")
            eval_status.value = f"Error loading runs: {e}"
            eval_status.color = COLORS.RED
            page.update()

    def _on_training_run_selected(e):
        """When a training run is selected, populate the base model field."""
        try:
            run_id = eval_training_run_dd.value
            if not run_id:
                return

            from db.training_runs import get_training_run

            run = get_training_run(int(run_id))
            if run:
                eval_base_model_tf.value = run.get("base_model", "")
                eval_status.value = f"Selected: {run['name']}"
                eval_status.color = COLORS.GREEN
                page.update()
        except Exception as e:
            logger.error(f"Error selecting run: {e}")

    async def _run_evaluation(_=None):
        """Run the benchmark evaluation."""
        if eval_state["running"]:
            return

        run_id = eval_training_run_dd.value
        if not run_id:
            eval_status.value = "Please select a training run first."
            eval_status.color = COLORS.ORANGE
            await safe_update(page)
            return

        try:
            from db.training_runs import get_training_run

            run = get_training_run(int(run_id))
            if not run:
                eval_status.value = "Training run not found."
                eval_status.color = COLORS.RED
                await safe_update(page)
                return

            adapter_path = run.get("adapter_path", "")
            base_model = run.get("base_model", "")

            if not adapter_path or not os.path.isdir(adapter_path):
                eval_status.value = "Adapter path not found. Training may not have completed."
                eval_status.color = COLORS.RED
                await safe_update(page)
                return

            # Get benchmark config
            benchmark = eval_benchmark_dd.value or "truthfulqa_mc2"
            num_samples = (eval_num_samples_tf.value or "").strip()
            batch_size = (eval_batch_size_tf.value or "4").strip()
            compare_mode = eval_compare_cb.value

            # Update UI state
            eval_state["running"] = True
            eval_state["results"] = {}
            eval_state["base_results"] = {}
            eval_run_btn.disabled = True
            eval_stop_btn.disabled = False
            eval_busy_ring.visible = True
            eval_progress.visible = True
            eval_progress.value = None  # Indeterminate
            eval_results_table.visible = False
            eval_results_placeholder.visible = True
            eval_results_placeholder.value = "Running evaluation..."
            await safe_update(page)

            # Run evaluation on fine-tuned model
            eval_progress_label.value = "Evaluating fine-tuned model..."
            await safe_update(page)

            finetuned_results = await _run_lm_eval(
                base_model=base_model,
                adapter_path=adapter_path,
                benchmark=benchmark,
                num_samples=num_samples,
                batch_size=batch_size,
            )
            eval_state["results"] = finetuned_results

            # Run evaluation on base model if comparison mode
            base_results = {}
            if compare_mode:
                eval_progress_label.value = "Evaluating base model for comparison..."
                await safe_update(page)

                base_results = await _run_lm_eval(
                    base_model=base_model,
                    adapter_path=None,  # No adapter = base model only
                    benchmark=benchmark,
                    num_samples=num_samples,
                    batch_size=batch_size,
                )
                eval_state["base_results"] = base_results

            # Display results
            _display_results(finetuned_results, base_results, benchmark)

            eval_status.value = "Evaluation complete!"
            eval_status.color = COLORS.GREEN
            eval_progress_label.value = ""

        except asyncio.CancelledError:
            eval_status.value = "Evaluation cancelled."
            eval_status.color = COLORS.ORANGE
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            eval_status.value = f"Evaluation failed: {str(e)[:100]}"
            eval_status.color = COLORS.RED
            eval_results_placeholder.value = f"Error: {e}"
        finally:
            eval_state["running"] = False
            eval_run_btn.disabled = False
            eval_stop_btn.disabled = True
            eval_busy_ring.visible = False
            eval_progress.visible = False
            await safe_update(page)

    async def _run_lm_eval(
        *,
        base_model: str,
        adapter_path: Optional[str],
        benchmark: str,
        num_samples: str,
        batch_size: str,
    ) -> Dict[str, Any]:
        """Run lm-evaluation-harness and return results."""
        try:
            # Import lm_eval
            from lm_eval import evaluator

            # Build model args
            model_args = f"pretrained={base_model}"
            if adapter_path:
                model_args += f",peft={adapter_path}"

            # Build task list
            tasks = [benchmark]

            # Build kwargs
            eval_kwargs: Dict[str, Any] = {
                "model": "hf",
                "model_args": model_args,
                "tasks": tasks,
                "batch_size": int(batch_size) if batch_size else 4,
                "device": "cuda" if _has_cuda() else "cpu",
            }

            if num_samples:
                eval_kwargs["limit"] = int(num_samples)

            # Run evaluation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: evaluator.simple_evaluate(**eval_kwargs),
            )

            # Return both results and samples for visualization
            return {
                "metrics": results.get("results", {}),
                "samples": results.get("samples", {}),
                "n_samples": results.get("n-samples", {}),
            }

        except ImportError as e:
            raise RuntimeError(f"lm-eval not installed properly: {e}")
        except Exception as e:
            raise RuntimeError(f"Evaluation error: {e}")

    def _has_cuda() -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def _display_results(
        finetuned: Dict[str, Any],
        base: Dict[str, Any],
        benchmark: str,
    ):
        """Display evaluation results in the table with visualizations."""
        rows = []
        has_comparison = bool(base) and base.get("metrics")

        # Extract metrics from new result structure
        ft_data = finetuned.get("metrics", finetuned)
        base_data = base.get("metrics", base) if base else {}

        # Update table columns based on comparison mode
        if has_comparison:
            eval_results_table.columns = [
                ft.DataColumn(ft.Text("Metric")),
                ft.DataColumn(ft.Text("Fine-tuned"), numeric=True),
                ft.DataColumn(ft.Text("Base Model"), numeric=True),
                ft.DataColumn(ft.Text("Δ Change"), numeric=True),
            ]
        else:
            eval_results_table.columns = [
                ft.DataColumn(ft.Text("Metric")),
                ft.DataColumn(ft.Text("Score"), numeric=True),
            ]

        # Extract metrics from results - try benchmark key first, then direct access
        ft_metrics = ft_data.get(benchmark, ft_data)
        base_metrics = base_data.get(benchmark, base_data) if base_data else {}

        # Get all numeric metrics from the results, excluding stderr and alias
        metric_keys = []
        all_keys = []
        for key, val in ft_metrics.items():
            all_keys.append(f"{key}={val}")
            # Skip non-numeric, private keys, stderr, and alias
            if isinstance(val, (int, float)) and not key.startswith("_"):
                if "stderr" not in key.lower() and key != "alias":
                    metric_keys.append(key)

        # Log all available metrics for debugging
        logger.info(f"Available metrics for {benchmark}: {', '.join(all_keys)}")
        logger.info(f"Displaying metrics: {metric_keys}")

        # Fallback to common metrics if none found
        if not metric_keys:
            metric_keys = ["acc", "acc_norm", "mc1", "mc2", "exact_match", "f1"]

        for key in metric_keys:
            ft_val = ft_metrics.get(key)
            base_val = base_metrics.get(key) if base_metrics else None

            if ft_val is not None:
                ft_display = f"{ft_val * 100:.2f}%" if isinstance(ft_val, float) else str(ft_val)
                # Clean up metric name for display
                display_name = key.replace(",none", "").replace("_", " ").replace(",", " ").title()

                if has_comparison:
                    base_display = f"{base_val * 100:.2f}%" if base_val is not None else "—"

                    # Calculate delta
                    if ft_val is not None and base_val is not None:
                        delta = (ft_val - base_val) * 100
                        delta_display = f"{delta:+.2f}%"
                        delta_color = COLORS.GREEN if delta > 0 else COLORS.RED if delta < 0 else None
                    else:
                        delta_display = "—"
                        delta_color = None

                    rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text(display_name)),
                                ft.DataCell(ft.Text(ft_display, weight=ft.FontWeight.BOLD)),
                                ft.DataCell(ft.Text(base_display)),
                                ft.DataCell(
                                    ft.Text(
                                        delta_display,
                                        color=delta_color,
                                        weight=ft.FontWeight.BOLD if delta_color else None,
                                    )
                                ),
                            ]
                        )
                    )
                else:
                    # Simple view without comparison
                    rows.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text(display_name)),
                                ft.DataCell(ft.Text(ft_display, weight=ft.FontWeight.BOLD)),
                            ]
                        )
                    )

        if rows:
            eval_results_table.rows = rows
            eval_results_table.visible = True
            eval_results_placeholder.visible = False

            # Add visual bars for all metrics
            visual_bars = []

            # Define which metrics to show visually with their colors and labels
            metric_visuals = [
                ("acc,none", "acc", "Accuracy", COLORS.GREEN),
                ("acc_norm,none", "acc_norm", "Normalized Accuracy", ACCENT_COLOR),
                ("mc2,none", "mc2", "MC2 Score", COLORS.GREEN),
            ]

            for key1, key2, label, color in metric_visuals:
                val = None
                if key1 in ft_metrics and isinstance(ft_metrics[key1], (int, float)):
                    val = float(ft_metrics[key1])
                elif key2 in ft_metrics and isinstance(ft_metrics[key2], (int, float)):
                    val = float(ft_metrics[key2])

                if val is not None:
                    bar_width = int(val * 300)  # Scale to 300px max
                    visual_bars.append(
                        ft.Column(
                            [
                                ft.Text(label, size=11, weight=ft.FontWeight.BOLD),
                                ft.Row(
                                    [
                                        ft.Container(
                                            width=bar_width,
                                            height=20,
                                            bgcolor=color,
                                            border_radius=4,
                                        ),
                                        ft.Container(
                                            width=300 - bar_width,
                                            height=20,
                                            bgcolor=WITH_OPACITY(0.2, BORDER_BASE),
                                            border_radius=4,
                                        ),
                                    ],
                                    spacing=0,
                                ),
                                ft.Text(
                                    f"{val * 100:.1f}%",
                                    size=10,
                                    color=WITH_OPACITY(0.7, BORDER_BASE),
                                ),
                            ],
                            spacing=2,
                        )
                    )

            if visual_bars:
                eval_comparison_container.content = ft.Container(
                    content=ft.Column(visual_bars, spacing=12),
                    padding=10,
                )
                eval_comparison_container.visible = True
            else:
                eval_comparison_container.visible = False
        else:
            eval_results_placeholder.value = "No metrics found in results. Check benchmark output."
            eval_results_placeholder.visible = True
            eval_results_table.visible = False
            eval_comparison_container.visible = False

    def _stop_evaluation(_=None):
        """Stop the running evaluation."""
        eval_state["running"] = False
        eval_status.value = "Stopping evaluation..."
        page.update()

    # Wire up handlers
    eval_training_run_refresh_btn.on_click = _refresh_training_runs
    eval_training_run_dd.on_change = _on_training_run_selected
    eval_run_btn.on_click = lambda e: page.run_task(_run_evaluation)
    eval_stop_btn.on_click = _stop_evaluation

    # Initial refresh
    _refresh_training_runs()

    # Build the tab
    return build_evaluate_tab(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        eval_status=eval_status,
        eval_base_model_tf=eval_base_model_tf,
        eval_training_run_dd=eval_training_run_dd,
        eval_training_run_refresh_btn=eval_training_run_refresh_btn,
        eval_benchmark_dd=eval_benchmark_dd,
        eval_num_samples_tf=eval_num_samples_tf,
        eval_batch_size_tf=eval_batch_size_tf,
        eval_compare_cb=eval_compare_cb,
        eval_run_btn=eval_run_btn,
        eval_stop_btn=eval_stop_btn,
        eval_busy_ring=eval_busy_ring,
        eval_progress=eval_progress,
        eval_progress_label=eval_progress_label,
        eval_results_container=eval_results_container,
        eval_comparison_container=eval_comparison_container,
    )
