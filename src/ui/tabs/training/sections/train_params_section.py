"""Training params section builder for FineFoundry Training tab.

Composes the training parameters block using controls created in main.py.
Layout-only; no logic moved.
"""

from __future__ import annotations

import flet as ft


def build_train_params_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    skill_level: ft.Dropdown,
    beginner_mode_dd: ft.Dropdown,
    expert_gpu_dd: ft.Dropdown,
    expert_gpu_busy: ft.ProgressRing,
    expert_spot_cb: ft.Checkbox,
    expert_gpu_refresh_btn: ft.IconButton | ft.TextButton | ft.Control,
    base_model: ft.TextField,
    epochs_tf: ft.TextField,
    lr_tf: ft.TextField,
    batch_tf: ft.TextField,
    grad_acc_tf: ft.TextField,
    max_steps_tf: ft.TextField,
    use_lora_cb: ft.Checkbox,
    lora_r_dd: ft.Dropdown,
    lora_alpha_tf: ft.TextField,
    lora_dropout_tf: ft.TextField,
    use_rslora_cb: ft.Checkbox,
    out_dir_tf: ft.TextField,
    packing_row: ft.Row,
    auto_resume_row: ft.Row,
    push_row: ft.Row,
    hf_repo_row: ft.Row,
    resume_from_row: ft.Row,
    advanced_params_section: ft.Container,
    visible: bool = False,
) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Training Params",
                    getattr(ICONS, "SETTINGS", ICONS.DESCRIPTION),
                    "Basic hyperparameters and LoRA toggle for training.",
                    on_help_click=_mk_help_handler("Basic hyperparameters and LoRA toggle for training."),
                ),
                ft.Row([skill_level, beginner_mode_dd], wrap=True),
                ft.Row([expert_gpu_dd, expert_gpu_busy, expert_spot_cb, expert_gpu_refresh_btn], wrap=True),
                ft.Row(
                    [base_model, epochs_tf, lr_tf, batch_tf, grad_acc_tf, max_steps_tf, out_dir_tf],
                    wrap=True,
                ),
                ft.Row(
                    [use_lora_cb, lora_r_dd, lora_alpha_tf, lora_dropout_tf, use_rslora_cb],
                    wrap=True,
                ),
                ft.Row([packing_row, auto_resume_row, push_row, hf_repo_row, resume_from_row], wrap=True),
                advanced_params_section,
                ft.Divider(),
            ],
            spacing=0,
        ),
        visible=visible,
    )
