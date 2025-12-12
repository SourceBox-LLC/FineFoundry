"""Runpod Training content container builder for FineFoundry.

This module composes the main Runpod training content using pre-built sections
provided by main.py. No logic; layout-only composition.
"""

from __future__ import annotations

import flet as ft


def build_pod_content_container(
    *,
    config_section: ft.Control,
    rp_infra_panel: ft.Control,
    rp_ds_divider: ft.Control,
    ds_tp_group_container: ft.Container,
    pod_logs_section: ft.Container,
    teardown_section: ft.Control,
    train_actions: ft.Control,
) -> ft.Container:
    return ft.Container(
        content=ft.Row(
            [
                ft.Container(
                    content=ft.Column(
                        [
                            config_section,
                            rp_infra_panel,
                            rp_ds_divider,
                            ds_tp_group_container,
                            pod_logs_section,
                            teardown_section,
                            train_actions,
                        ],
                        spacing=12,
                    ),
                    width=1000,
                )
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        visible=True,
    )
