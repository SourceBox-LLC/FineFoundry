"""Teardown section builder for FineFoundry Training tab.

Composes the teardown UI using controls created in main.py. Layout-only.
"""
from __future__ import annotations

import flet as ft


def build_teardown_section(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    td_template_cb: ft.Checkbox,
    td_volume_cb: ft.Checkbox,
    td_pod_cb: ft.Checkbox,
    td_busy: ft.ProgressRing,
    on_teardown_selected_cb,
    on_teardown_all_cb,
) -> ft.Container:
    return ft.Container(
        content=ft.Column([
            section_title(
                "Teardown",
                getattr(ICONS, "DELETE_FOREVER", getattr(ICONS, "DELETE", ICONS.CLOSE)),
                "Select infrastructure items to delete. Teardown All removes all related items.",
                on_help_click=_mk_help_handler("Delete Runpod Template and/or Network Volume. If a pod exists, you can delete it too."),
            ),
            ft.Text("Select items to teardown.", size=12, color=WITH_OPACITY(0.7, BORDER_BASE)),
            ft.Container(
                content=ft.Column([
                    td_pod_cb,
                    td_template_cb,
                    td_volume_cb,
                ], spacing=6),
                padding=8,
                border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                border_radius=8,
            ),
            ft.Row([
                ft.ElevatedButton(
                    "Teardown Selected",
                    icon=getattr(ICONS, "DELETE", getattr(ICONS, "DELETE_OUTLINE", ICONS.CLOSE)),
                    on_click=on_teardown_selected_cb,
                ),
                ft.OutlinedButton(
                    "Teardown All",
                    icon=getattr(ICONS, "DELETE_FOREVER", getattr(ICONS, "DELETE", ICONS.CLOSE)),
                    on_click=on_teardown_all_cb,
                ),
                td_busy,
            ], spacing=10),
        ], spacing=8),
        visible=False,
    )
