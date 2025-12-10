"""Settings tab builder for FineFoundry.

This module builds the Settings tab UI using control instances created in main.py.
No behavior or logic is changed; we only compose the layout here to keep main.py slim.
"""

from __future__ import annotations

import flet as ft


def build_settings_tab(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    proxy_enable_cb,
    use_env_cb,
    proxy_url_tf,
    hf_token_tf,
    hf_status,
    hf_test_btn,
    hf_save_btn,
    hf_remove_btn,
    runpod_key_tf,
    runpod_status,
    runpod_test_btn,
    runpod_save_btn,
    runpod_remove_btn,
    ollama_enable_cb,
    ollama_base_url_tf,
    ollama_default_model_tf,
    ollama_models_dd,
    ollama_test_btn,
    ollama_refresh_btn,
    ollama_save_btn,
    ollama_status,
    REFRESH_ICON,
    system_check_status,
    system_check_btn,
    system_check_log_container,
    system_check_summary_container,
    system_check_download_btn,
) -> ft.Container:
    """Return the Settings tab content container.

    All controls and helpers are provided by the caller (main.py) so
    cross-tab references remain intact and logic is unchanged.
    """
    return ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Container(
                            content=ft.Column(
                                [
                                    # Proxy
                                    section_title(
                                        "Proxy Settings",
                                        ICONS.SETTINGS,
                                        "Override network proxy for requests. Use system env or custom URL.",
                                        on_help_click=_mk_help_handler(
                                            "Override network proxy for requests. Use system env or custom URL."
                                        ),
                                    ),
                                    ft.Text(
                                        "Configure how network requests route. When enabled, UI settings override environment variables and defaults.",
                                        size=12,
                                        color=WITH_OPACITY(0.7, BORDER_BASE),
                                    ),
                                    ft.Divider(),
                                    ft.Row([proxy_enable_cb], wrap=True),
                                    ft.Row([use_env_cb, proxy_url_tf], wrap=True),
                                    ft.Text(
                                        "Tip: Tor default is socks5h://127.0.0.1:9050. Leave disabled to use direct connections.",
                                        size=11,
                                        color=WITH_OPACITY(0.6, BORDER_BASE),
                                    ),
                                    ft.Divider(),
                                    # HF
                                    section_title(
                                        "Hugging Face Access",
                                        getattr(ICONS, "HUB", ICONS.CLOUD),
                                        "Save and test your Hugging Face API token. If saved, it's used globally.",
                                        on_help_click=_mk_help_handler(
                                            "Save and test your Hugging Face API token. If saved, it's used globally."
                                        ),
                                    ),
                                    ft.Text(
                                        "Saved token (if set) is used for Hugging Face Hub operations and dataset downloads.",
                                        size=12,
                                        color=WITH_OPACITY(0.7, BORDER_BASE),
                                    ),
                                    ft.Row([hf_token_tf], wrap=True),
                                    ft.Row([hf_test_btn, hf_save_btn, hf_remove_btn], spacing=10, wrap=True),
                                    hf_status,
                                    ft.Divider(),
                                    # Runpod
                                    section_title(
                                        "Runpod API Access",
                                        getattr(ICONS, "VPN_KEY", getattr(ICONS, "KEY", ICONS.SETTINGS)),
                                        "Save and test your Runpod API key. Used by Training → Runpod Infrastructure.",
                                        on_help_click=_mk_help_handler(
                                            "Save and test your Runpod API key. Used by Training → Runpod Infrastructure."
                                        ),
                                    ),
                                    ft.Text(
                                        "Stored locally and applied to RUNPOD_API_KEY when saved. Required for ensuring Runpod Network Volume & Template.",
                                        size=12,
                                        color=WITH_OPACITY(0.7, BORDER_BASE),
                                    ),
                                    ft.Row([runpod_key_tf], wrap=True),
                                    ft.Row(
                                        [runpod_test_btn, runpod_save_btn, runpod_remove_btn], spacing=10, wrap=True
                                    ),
                                    runpod_status,
                                    ft.Divider(),
                                    # Ollama
                                    section_title(
                                        "Ollama Connection",
                                        getattr(ICONS, "HUB", ICONS.CLOUD),
                                        "Configure connection to Ollama server; only stored here.",
                                        on_help_click=_mk_help_handler(
                                            "Configure connection to Ollama server; only stored here."
                                        ),
                                    ),
                                    ft.Text(
                                        "Connect to a local or remote Ollama server. This is only configuration; other tabs won't use it yet.",
                                        size=12,
                                        color=WITH_OPACITY(0.7, BORDER_BASE),
                                    ),
                                    ft.Row([ollama_enable_cb], wrap=True),
                                    ft.Row([ollama_base_url_tf, ollama_default_model_tf], wrap=True),
                                    ft.Row([ollama_models_dd], wrap=True),
                                    ft.Row(
                                        [ollama_test_btn, ollama_refresh_btn, ollama_save_btn], spacing=10, wrap=True
                                    ),
                                    ollama_status,
                                    ft.Divider(),
                                    section_title(
                                        "System Check",
                                        getattr(ICONS, "CHECK_CIRCLE", ICONS.SETTINGS),
                                        "Run internal diagnostics (pytest-based tests) to verify this installation.",
                                        on_help_click=_mk_help_handler(
                                            "Run internal diagnostics (pytest-based tests) to verify this installation."
                                        ),
                                    ),
                                    ft.Text(
                                        "Runs the internal pytest test suite, coverage, lint, and type checks, and streams output below.",
                                        size=12,
                                        color=WITH_OPACITY(0.7, BORDER_BASE),
                                    ),
                                    ft.Row(
                                        [system_check_btn, system_check_download_btn],
                                        spacing=10,
                                        wrap=True,
                                    ),
                                    system_check_status,
                                    system_check_log_container,
                                    system_check_summary_container,
                                ],
                                spacing=12,
                            ),
                            width=1000,
                        )
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                )
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=0,
        ),
        padding=16,
    )
