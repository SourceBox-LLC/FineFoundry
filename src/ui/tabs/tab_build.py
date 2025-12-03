"""Build/Publish tab builder for FineFoundry.

This module composes the Build/Publish tab UI using controls created in main.py.
Now delegates to per-section builders under `ui.tabs.build.sections` for clarity.
No behavior changes.
"""
from __future__ import annotations

import flet as ft

from ui.tabs.build.sections.dataset_params_section import build_dataset_params_section
from ui.tabs.build.sections.push_section import build_push_section
from ui.tabs.build.sections.model_card_section import build_model_card_section
from ui.tabs.build.sections.status_section import build_status_section

def build_build_tab(
    *,
    section_title,
    ICONS,
    BORDER_BASE,
    WITH_OPACITY,
    _mk_help_handler,
    # Dataset params
    source_mode,
    data_file,
    merged_dir,
    seed,
    shuffle,
    min_len_b,
    save_dir,
    # Splits
    val_slider,
    test_slider,
    split_error,
    split_badges,
    # Push controls & actions
    push_toggle,
    repo_id,
    private,
    token_val_ui,
    build_actions,
    # Model card creator
    use_custom_card,
    card_preview_switch,
    load_template_btn,
    gen_from_ds_btn,
    gen_with_ollama_btn,
    clear_card_btn,
    ollama_gen_status,
    card_editor,
    card_preview_container,
    # Status
    timeline,
    timeline_placeholder,
    # Optional ref to Status section container for visibility control
    status_section_ref=None,
) -> ft.Container:
    dataset_params = build_dataset_params_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        source_mode=source_mode,
        data_file=data_file,
        merged_dir=merged_dir,
        seed=seed,
        shuffle=shuffle,
        min_len_b=min_len_b,
        save_dir=save_dir,
        val_slider=val_slider,
        test_slider=test_slider,
        split_error=split_error,
        split_badges=split_badges,
    )

    push_section = build_push_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        push_toggle=push_toggle,
        repo_id=repo_id,
        private=private,
        token_val_ui=token_val_ui,
        build_actions=build_actions,
    )

    model_card_section = build_model_card_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        use_custom_card=use_custom_card,
        card_preview_switch=card_preview_switch,
        load_template_btn=load_template_btn,
        gen_from_ds_btn=gen_from_ds_btn,
        gen_with_ollama_btn=gen_with_ollama_btn,
        clear_card_btn=clear_card_btn,
        ollama_gen_status=ollama_gen_status,
        card_editor=card_editor,
        card_preview_container=card_preview_container,
    )

    status_section = build_status_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        timeline=timeline,
        timeline_placeholder=timeline_placeholder,
    )

    # Expose status section container to caller (e.g., to control visibility)
    try:
        if status_section_ref is not None:
            status_section_ref["control"] = status_section
    except Exception:
        pass

    return ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Container(
                    content=ft.Column([
                        dataset_params,
                        model_card_section,
                        push_section,
                        status_section,
                    ], spacing=12),
                    width=1000,
                )
            ], alignment=ft.MainAxisAlignment.CENTER)
        ], scroll=ft.ScrollMode.AUTO, spacing=0),
        padding=16,
    )
