"""Analysis tab: Runtime settings section builder."""

from __future__ import annotations

import flet as ft


def build_runtime_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    analysis_backend_dd: ft.Dropdown,
    offline_reason: ft.Text,
    analysis_hf_token_tf: ft.TextField,
    analysis_sample_size_tf: ft.TextField,
) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Runtime settings",
                    getattr(ICONS, "SETTINGS", getattr(ICONS, "TUNE", ICONS.SETTINGS)),
                    "Backend and sampling. Sample size limits records analyzed for speed.",
                    on_help_click=_mk_help_handler(
                        "Backend and sampling. Sample size limits records analyzed for speed."
                    ),
                ),
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Row(
                                [
                                    analysis_backend_dd,
                                    analysis_hf_token_tf,
                                    analysis_sample_size_tf,
                                ],
                                wrap=True,
                                spacing=10,
                            ),
                            offline_reason,
                        ],
                        spacing=0,
                    ),
                    width=1000,
                    border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
                    border_radius=8,
                    padding=10,
                ),
            ],
            spacing=12,
        ),
        width=1000,
    )
