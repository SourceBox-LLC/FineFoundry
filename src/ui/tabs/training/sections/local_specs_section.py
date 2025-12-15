"""Local specs and Local Docker training section builder for FineFoundry.

This module composes the Local Training: System Specs + Docker sections using
controls created in main.py. No logic changes; only layout composition is centralized here.
"""

from __future__ import annotations

import flet as ft


def build_local_specs_container(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    REFRESH_ICON,
    # Specs controls
    local_os_txt: ft.Text,
    local_py_txt: ft.Text,
    local_cpu_txt: ft.Text,
    local_ram_txt: ft.Text,
    local_disk_txt: ft.Text,
    local_torch_txt: ft.Text,
    local_cuda_txt: ft.Text,
    local_gpus_txt: ft.Text,
    local_flags_box: ft.Column,
    local_capability_txt: ft.Text,
    # Docker pull controls
    docker_image_tf: ft.TextField,
    docker_pull_btn: ft.ElevatedButton,
    docker_pull_ring: ft.ProgressRing,
    docker_status: ft.Text,
    docker_log_timeline: ft.ListView,
    docker_log_placeholder: ft.Text,
    refresh_specs_click_cb,
    # Local docker run controls - managed training runs
    local_training_run_dd: ft.Dropdown,
    local_training_run_refresh_btn: ft.IconButton,
    local_new_run_name_tf: ft.TextField,
    local_create_run_btn: ft.OutlinedButton,
    local_run_storage_info: ft.Text,
    local_container_name_tf: ft.TextField,
    local_use_gpu_cb: ft.Checkbox,
    local_pass_hf_token_cb: ft.Checkbox,
    local_train_progress: ft.ProgressBar,
    local_train_prog_label: ft.Text,
    local_save_logs_btn: ft.OutlinedButton,
    local_train_timeline: ft.ListView,
    local_train_timeline_placeholder: ft.Text,
    local_start_btn: ft.ElevatedButton,
    local_stop_btn: ft.OutlinedButton,
    local_view_metrics_btn: ft.OutlinedButton,
    local_train_status: ft.Text,
    local_infer_group_container: ft.Container,
    local_save_config_btn: ft.OutlinedButton,
    mk_help_handler,
) -> ft.Container:
    return ft.Container(
        content=ft.Row(
            [
                ft.Container(
                    content=ft.Column(
                        [
                            section_title(
                                "Local Training: System Specs",
                                getattr(ICONS, "COMPUTER", getattr(ICONS, "TERMINAL", ICONS.SETTINGS)),
                                "A quick view of your system for local fine‑tuning feasibility.",
                                on_help_click=mk_help_handler(
                                    "Shows CPU, RAM, disk, GPU, and CUDA status to gauge local fine‑tuning readiness."
                                ),
                            ),
                            ft.Container(
                                content=ft.Column(
                                    [
                                        ft.Row([ft.Text("OS", width=160), local_os_txt]),
                                        ft.Row([ft.Text("Python", width=160), local_py_txt]),
                                        ft.Row([ft.Text("CPU cores", width=160), local_cpu_txt]),
                                        ft.Row([ft.Text("RAM", width=160), local_ram_txt]),
                                        ft.Row([ft.Text("Disk free", width=160), local_disk_txt]),
                                        ft.Row([ft.Text("PyTorch", width=160), local_torch_txt]),
                                        ft.Row([ft.Text("CUDA", width=160), local_cuda_txt]),
                                        ft.Text("GPUs:"),
                                        ft.Container(
                                            local_gpus_txt,
                                            padding=6,
                                            bgcolor=WITH_OPACITY(0.04, BORDER_BASE),
                                            border_radius=6,
                                        ),
                                        ft.Divider(),
                                        local_flags_box,
                                        local_capability_txt,
                                        ft.Divider(),
                                        section_title(
                                            "Docker: Pull Image",
                                            getattr(ICONS, "CLOUD_DOWNLOAD", getattr(ICONS, "DOWNLOAD", ICONS.CLOUD)),
                                            "Pull a Docker image locally for training tasks.",
                                            on_help_click=mk_help_handler(
                                                "Use Docker Desktop. This pulls the image to your machine."
                                            ),
                                        ),
                                        ft.Row(
                                            [docker_image_tf, docker_pull_btn, docker_pull_ring], wrap=True, spacing=10
                                        ),
                                        docker_status,
                                        ft.Container(
                                            ft.Stack([docker_log_timeline, docker_log_placeholder], expand=True),
                                            height=180,
                                            width=1000,
                                            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                                            border_radius=8,
                                            padding=10,
                                        ),
                                        ft.Row(
                                            [
                                                ft.OutlinedButton(
                                                    "Refresh specs", icon=REFRESH_ICON, on_click=refresh_specs_click_cb
                                                ),
                                            ],
                                            wrap=True,
                                        ),
                                        ft.Divider(),
                                        section_title(
                                            "Local Docker: Run Training",
                                            getattr(ICONS, "PLAY_CIRCLE", getattr(ICONS, "TERMINAL", ICONS.PLAY_ARROW)),
                                            "Builds and runs a local Docker container mirroring the Runpod training command.",
                                            on_help_click=mk_help_handler(
                                                "Runs the training script inside the selected Docker image with your dataset mounted at /data. Uses the same command builder as Runpod."
                                            ),
                                        ),
                                        ft.Row(
                                            [local_training_run_dd, local_training_run_refresh_btn],
                                            spacing=6,
                                        ),
                                        ft.Row(
                                            [local_new_run_name_tf, local_create_run_btn],
                                            spacing=10,
                                        ),
                                        local_run_storage_info,
                                        ft.Row(
                                            [local_container_name_tf, local_use_gpu_cb, local_pass_hf_token_cb],
                                            wrap=True,
                                            spacing=10,
                                        ),
                                        ft.Row(
                                            [local_train_progress, local_train_prog_label, local_save_logs_btn],
                                            spacing=12,
                                        ),
                                        ft.Container(
                                            ft.Stack(
                                                [local_train_timeline, local_train_timeline_placeholder], expand=True
                                            ),
                                            height=240,
                                            width=1000,
                                            border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                                            border_radius=8,
                                            padding=10,
                                        ),
                                        ft.Row(
                                            [
                                                local_start_btn,
                                                local_stop_btn,
                                                local_view_metrics_btn,
                                                local_save_config_btn,
                                            ],
                                            spacing=10,
                                            wrap=True,
                                        ),
                                        local_train_status,
                                        local_infer_group_container,
                                    ],
                                    spacing=6,
                                ),
                                width=1000,
                                border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                                border_radius=8,
                                padding=10,
                            ),
                        ],
                        spacing=12,
                    ),
                    width=1000,
                )
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        visible=False,
    )
