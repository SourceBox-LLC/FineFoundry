"""Runpod Infrastructure section builder for FineFoundry Training tab.

Composes the Runpod infra panel layout using controls created in main.py.
Layout-only; no logic moved.
"""

from __future__ import annotations

import flet as ft


def build_rp_infra_panel(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    rp_dc_tf: ft.TextField,
    rp_vol_name_tf: ft.TextField,
    rp_vol_size_tf: ft.TextField,
    rp_resize_row: ft.Row,
    rp_tpl_name_tf: ft.TextField,
    rp_image_tf: ft.TextField,
    rp_container_disk_tf: ft.TextField,
    rp_volume_in_gb_tf: ft.TextField,
    rp_mount_path_tf: ft.TextField,
    rp_category_tf: ft.TextField,
    rp_public_row: ft.Row,
    rp_temp_key_tf: ft.TextField,
    rp_infra_actions: ft.Row,
) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Runpod Infrastructure",
                    getattr(ICONS, "CLOUD", ICONS.SETTINGS),
                    "Create or update the required Runpod Network Volume and Template before training.",
                    on_help_click=_mk_help_handler(
                        "Create or update the required Runpod Network Volume and Template before training."
                    ),
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                "Defaults are provided; change any value to customize. Key precedence: Settings > Training temp field > environment.",
                                size=12,
                                color=WITH_OPACITY(0.7, BORDER_BASE),
                            ),
                            ft.Row([rp_dc_tf, rp_vol_name_tf, rp_vol_size_tf, rp_resize_row], wrap=True),
                            ft.Row([rp_tpl_name_tf, rp_image_tf], wrap=True),
                            ft.Row([rp_container_disk_tf, rp_volume_in_gb_tf, rp_mount_path_tf], wrap=True),
                            ft.Row([rp_category_tf, rp_public_row], wrap=True),
                            ft.Row([rp_temp_key_tf], wrap=True),
                            rp_infra_actions,
                        ],
                        spacing=12,
                    ),
                    width=1000,
                    border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                    border_radius=8,
                    padding=10,
                ),
            ],
            spacing=12,
        ),
        visible=True,
    )
