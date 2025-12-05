"""Build/Publish tab controller for FineFoundry.

This module builds the Build/Publish tab controls and wires up all build
and push handlers, keeping `src/main.py` slimmer. Layout composition
still lives in `tab_build.py` and its section builders.
"""
from __future__ import annotations

from typing import Any, Dict

import asyncio
import json
import os
import random

import flet as ft

from helpers.common import safe_update
from helpers.theme import BORDER_BASE, COLORS, ICONS, REFRESH_ICON
from helpers.ui import WITH_OPACITY, make_empty_placeholder, pill
from helpers.build import (
    run_build as run_build_helper,
    run_push_async as run_push_async_helper,
)
from helpers.settings_ollama import (
    load_config as load_ollama_config_helper,
    chat as ollama_chat_helper,
)

# save_dataset utilities (local, with PYTHONPATH pointing to project src)
try:  # pragma: no cover - normal path
    import save_dataset as sd
except Exception:  # pragma: no cover - fallback for alternate runtimes
    import sys as _sys

    _sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    import save_dataset as sd

from ui.tabs.tab_build import build_build_tab


def _schedule_task(page: ft.Page, coro):
    """Robust scheduler helper for async tasks.

    Mirrors the pattern used in other controllers, preferring
    ``page.run_task`` when available and falling back to
    ``asyncio.create_task``.
    """

    try:
        if hasattr(page, "run_task") and callable(getattr(page, "run_task")):
            return page.run_task(coro)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        return asyncio.create_task(coro())
    except Exception:  # pragma: no cover - defensive
        return None


def build_build_tab_with_logic(
    page: ft.Page,
    *,
    section_title,
    _mk_help_handler,
    _hf_cfg: Dict[str, Any],
    ollama_enable_cb: ft.Control,
    ollama_models_dd: ft.Dropdown,
) -> ft.Control:
    """Build the Build/Publish tab UI and attach all related handlers.

    This mirrors the previous inline Build tab setup from ``main.py``, but
    keeps the behavior localized to this module.
    """

    # Source selector for dataset preview/processing
    source_mode = ft.Dropdown(
        options=[
            ft.dropdown.Option("JSON file"),
            ft.dropdown.Option("Merged dataset"),
        ],
        value="JSON file",
        width=180,
    )

    # Data source and processing controls
    data_file = ft.TextField(
        label="Data file (JSON)",
        value="scraped_training_data.json",
        width=360,
    )
    merged_dir = ft.TextField(
        label="Merged dataset dir",
        value="merged_dataset",
        width=240,
    )
    seed = ft.TextField(
        label="Seed",
        value="42",
        width=120,
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    shuffle = ft.Switch(label="Shuffle", value=True)
    val_slider = ft.Slider(
        min=0,
        max=0.2,
        value=0.01,
        divisions=20,
        label="{value}",
    )
    test_slider = ft.Slider(
        min=0,
        max=0.2,
        value=0.0,
        divisions=20,
        label="{value}",
    )
    min_len_b = ft.TextField(
        label="Min Length",
        value="1",
        width=120,
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    save_dir = ft.TextField(
        label="Save dir",
        value="hf_dataset",
        width=240,
    )

    push_toggle = ft.Switch(label="Push to Hub", value=False)
    repo_id = ft.TextField(
        label="Repo ID",
        value="username/my-dataset",
        width=280,
    )
    private = ft.Switch(label="Private", value=True)
    token_val_ui = ft.TextField(
        label="HF Token",
        password=True,
        can_reveal_password=True,
        width=320,
    )

    # Validation chip for splits
    split_error = ft.Text("", color=COLORS.RED)

    def on_split_change(_):
        total = (val_slider.value or 0) + (test_slider.value or 0)
        if total >= 0.9:  # generous limit
            split_error.value = f"Warning: val+test too large ({total:.2f})"
        else:
            split_error.value = ""
        page.update()

    val_slider.on_change = on_split_change
    test_slider.on_change = on_split_change

    # Toggle UI fields based on source selection (JSON vs Merged dataset)
    def on_source_change(_):
        mode = (source_mode.value or "JSON file").strip()
        is_json = mode == "JSON file"
        try:
            data_file.visible = is_json
            merged_dir.visible = not is_json
            # Enable JSON-only processing params for JSON mode; disable in merged mode
            for ctl in [seed, shuffle, min_len_b, val_slider, test_slider]:
                try:
                    ctl.disabled = not is_json
                except Exception:
                    pass
        except Exception:
            pass
        page.update()

    source_mode.on_change = on_source_change
    # Initialize visibility/disabled state
    on_source_change(None)

    # Split badges (values updated during build)
    split_badges = {
        "train": pill("Train: 0", COLORS.BLUE, ICONS.STACKED_LINE_CHART),
        "val": pill("Val: 0", COLORS.ORANGE, ICONS.SIGNAL_CELLULAR_ALT),
        "test": pill("Test: 0", COLORS.PURPLE, ICONS.SSID_CHART),
    }
    split_meta = {
        "train": (COLORS.BLUE, ICONS.STACKED_LINE_CHART),
        "val": (COLORS.ORANGE, ICONS.SIGNAL_CELLULAR_ALT),
        "test": (COLORS.PURPLE, ICONS.SSID_CHART),
    }

    # Timeline (scrollable)
    timeline = ft.ListView(expand=1, auto_scroll=True, spacing=6)
    timeline_placeholder = make_empty_placeholder("No status yet", ICONS.TASK)
    status_section_ref: Dict[str, Any] = {}

    cancel_build: Dict[str, Any] = {"cancelled": False}
    dd_ref: Dict[str, Any] = {"dd": None}
    push_state: Dict[str, Any] = {"inflight": False}
    push_ring = ft.ProgressRing(width=18, height=18, value=None, visible=False)
    # Reference to the model card preview container (assigned later)
    card_preview_container: ft.Container | None = None

    # --- Model Card Creator controls/state ---
    # Switch to enable custom model card instead of autogenerated
    use_custom_card = ft.Switch(
        label="Use custom model card (README.md)",
        value=False,
    )

    # Helper to build a simple default template (used when user wants a starting point)
    def _default_card_template(repo: str) -> str:
        rid = (repo or "username/dataset").strip()
        return f"""---
 tags:
   - text-generation
 language:
   - en
 license: other
 pretty_name: {rid}
 ---

 # Dataset Card: {rid}

 ## Dataset Summary
 Provide a concise description of the dataset, its source, and intended purpose.

 ## Data Fields
 - input: description
 - output: description

 ## Source and Collection
 Describe how data was collected and any preprocessing steps.

 ## Splits
 - Train: <num>
 - Validation: <num>
 - Test: <num>

 ## Usage
 ```python
 from datasets import load_dataset
 ds = load_dataset("{rid}")
 print(ds)
 ```

 ## Ethical Considerations and Warnings
 - Content may include offensive or unsafe material depending on source. Use responsibly.

 ## Licensing
 Specify license and any restrictions.

 ## Changelog
 - v1.0: Initial release.
 """

    # Editor and preview
    card_editor = ft.TextField(
        label="Model Card Markdown",
        multiline=True,
        min_lines=12,
        max_lines=32,
        value="",
        width=960,
        disabled=True,
    )

    # Safe Markdown factory for wider Flet compatibility
    def _make_md(value: str):
        try:
            return ft.Markdown(value, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB)
        except Exception:
            try:
                return ft.Markdown(value)
            except Exception:
                # Fallback: plain text if Markdown is unavailable
                return ft.Text(value)

    card_preview_switch = ft.Switch(
        label="Live preview",
        value=False,
        disabled=True,
    )
    card_preview_md = _make_md("")
    try:
        # Some Flet controls don't have 'visible'; guard accordingly
        card_preview_md.visible = False
    except Exception:
        pass

    # Dedicated preview container (hidden until we have content + preview enabled)
    card_preview_container = ft.Container(
        ft.Column(
            [card_preview_md],
            scroll=ft.ScrollMode.AUTO,
            spacing=0,
        ),
        height=300,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        padding=8,
        visible=False,
    )

    def _has_card_content() -> bool:
        try:
            return bool((card_editor.value or "").strip())
        except Exception:
            return False

    def _apply_preview_visibility() -> None:
        # Show preview only when: custom mode enabled + live preview on + content non-empty
        try:
            show = bool(use_custom_card.value) and bool(card_preview_switch.value) and _has_card_content()
            try:
                if hasattr(card_preview_md, "visible"):
                    card_preview_md.visible = show
            except Exception:
                pass
            try:
                if card_preview_container is not None:
                    card_preview_container.visible = show
            except Exception:
                pass
        except Exception:
            pass

    def _update_preview() -> None:
        try:
            if hasattr(card_preview_md, "value"):
                card_preview_md.value = card_editor.value or ""
        except Exception:
            # If preview control is Text (fallback), set .value via content replacement
            try:
                card_preview_md.value = card_editor.value or ""
            except Exception:
                pass
        # Re-evaluate visibility whenever content changes
        _apply_preview_visibility()

    def _on_toggle_custom_card(_):
        enabled = bool(use_custom_card.value)
        try:
            card_editor.disabled = not enabled
            card_preview_switch.disabled = not enabled
            if hasattr(card_preview_md, "visible"):
                card_preview_md.visible = enabled and bool(card_preview_switch.value)
        except Exception:
            pass
        _apply_preview_visibility()
        page.update()

    use_custom_card.on_change = _on_toggle_custom_card

    def _on_editor_change(_):
        if bool(card_preview_switch.value):
            _update_preview()
            page.update()

    try:
        card_editor.on_change = _on_editor_change
    except Exception:
        pass

    def _on_preview_toggle(_):
        try:
            if hasattr(card_preview_md, "visible"):
                card_preview_md.visible = bool(card_preview_switch.value) and bool(
                    use_custom_card.value
                )
        except Exception:
            pass
        _update_preview()
        page.update()

    card_preview_switch.on_change = _on_preview_toggle

    def _on_load_simple_template(_):
        # Turn on custom mode and load a simple template scaffold
        use_custom_card.value = True
        _on_toggle_custom_card(None)
        card_editor.value = _default_card_template((repo_id.value or "username/dataset").strip())
        _update_preview()
        page.update()

    async def _on_generate_from_dataset():
        # Generate using current built dataset (if available)
        dd = dd_ref.get("dd")
        if dd is None:
            page.snack_bar = ft.SnackBar(
                ft.Text("Build the dataset first to generate a default card."),
            )
            page.open(page.snack_bar)
            await safe_update(page)
            return
        rid = (repo_id.value or "").strip()
        if not rid:
            page.snack_bar = ft.SnackBar(
                ft.Text("Enter Repo ID to generate a default card."),
            )
            page.open(page.snack_bar)
            await safe_update(page)
            return
        try:
            use_custom_card.value = True
            _on_toggle_custom_card(None)
            content = await asyncio.to_thread(sd.build_dataset_card_content, dd, rid)
            card_editor.value = content
            _update_preview()
            await safe_update(page)
        except Exception as e:  # pragma: no cover - runtime error path
            page.snack_bar = ft.SnackBar(
                ft.Text(f"Failed to generate card: {e}"),
            )
            page.open(page.snack_bar)
            await safe_update(page)

    ollama_gen_status = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))

    async def _on_generate_with_ollama():
        # Generate using Ollama from the selected data file (JSON list of {input,output})
        try:
            if not bool(ollama_enable_cb.value):
                page.snack_bar = ft.SnackBar(
                    ft.Text("Enable Ollama in Settings first."),
                )
                page.open(page.snack_bar)
                await safe_update(page)
                return
        except Exception:
            pass

        cfg = load_ollama_config_helper()
        base_url = (cfg.get("base_url") or "http://127.0.0.1:11434").strip()
        model_name = (
            (ollama_models_dd.value or "")
            or (cfg.get("selected_model") or "")
            or (cfg.get("default_model") or "")
        ).strip()
        if not model_name:
            page.snack_bar = ft.SnackBar(
                ft.Text("Select an Ollama model in Settings."),
            )
            page.open(page.snack_bar)
            await safe_update(page)
            return

        path = (data_file.value or "scraped_training_data.json").strip()
        try:
            records = await asyncio.to_thread(sd.load_records, path)
        except Exception as e:  # pragma: no cover - runtime error path
            page.snack_bar = ft.SnackBar(
                ft.Text(f"Failed to load data file: {e}"),
            )
            page.open(page.snack_bar)
            await safe_update(page)
            return

        if not isinstance(records, list) or len(records) == 0:
            page.snack_bar = ft.SnackBar(
                ft.Text(
                    "Data file is empty or invalid (expected list of records).",
                ),
            )
            page.open(page.snack_bar)
            await safe_update(page)
            return

        total_n = len(records)
        # Sample a small subset for context
        k = min(8, total_n)
        idxs = (
            random.sample(range(total_n), k)
            if total_n >= k
            else list(range(total_n))
        )
        samples: list[dict[str, str]] = []
        for i in idxs:
            rec = records[i] if isinstance(records[i], dict) else {}
            inp = str(rec.get("input", ""))
            outp = str(rec.get("output", ""))
            try:
                inp = sd._truncate(inp, 400)  # type: ignore[attr-defined]
                outp = sd._truncate(outp, 400)  # type: ignore[attr-defined]
            except Exception:
                if len(inp) > 400:
                    inp = inp[:399] + "…"
                if len(outp) > 400:
                    outp = outp[:399] + "…"
            samples.append({"input": inp, "output": outp})

        # Size category helper
        try:
            size_cat = sd._size_category(total_n)  # type: ignore[attr-defined]
        except Exception:
            size_cat = (
                "n<1K"
                if total_n < 1_000
                else (
                    "1K<n<10K"
                    if total_n < 10_000
                    else (
                        "10K<n<100K"
                        if total_n < 100_000
                        else (
                            "100K<n<1M"
                            if total_n < 1_000_000
                            else "n>1M"
                        )
                    )
                )
            )

        rid = (repo_id.value or "username/dataset").strip()
        user_prompt = (
            f"You are an expert data curator. Create a professional Hugging Face dataset card (README.md) "
            f"in Markdown for the dataset '{rid}'.\n"
            f"Use the provided random samples to infer characteristics. Include a YAML frontmatter header with "
            f"tags, task_categories=text-generation, language=en, license=other, size_categories=[{size_cat}].\n"
            "Then include sections: Dataset Summary, Data Fields, Source and Collection, Splits (estimate if "
            "needed), Usage (datasets code snippet), Ethical Considerations and Warnings, Licensing, Example "
            "Records (re-embed the samples), How to Cite, Changelog.\n"
            "Keep the tone clear and factual. If unsure, state assumptions transparently."
        )
        samples_json = json.dumps(samples, ensure_ascii=False, indent=2)
        user_prompt += f"\n\nSamples (JSON):\n```json\n{samples_json}\n```\nTotal records (approx): {total_n}"

        system_prompt = (
            "You write concise, high-quality dataset cards for Hugging Face. "
            "Output ONLY valid Markdown starting with YAML frontmatter."
        )

        ollama_gen_status.value = (
            f"Generating with Ollama model '{model_name}'…"
        )
        await safe_update(page)
        try:
            md = await ollama_chat_helper(
                base_url,
                model_name,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            use_custom_card.value = True
            _on_toggle_custom_card(None)
            card_editor.value = md
            _update_preview()
            ollama_gen_status.value = "Generated with Ollama ✓"
            await safe_update(page)
        except Exception as e:  # pragma: no cover - runtime error path
            ollama_gen_status.value = f"Ollama generation failed: {e}"
            page.snack_bar = ft.SnackBar(ft.Text(ollama_gen_status.value))
            page.open(page.snack_bar)
            await safe_update(page)

    load_template_btn = ft.TextButton(
        "Load simple template",
        icon=ICONS.ARTICLE,
        on_click=_on_load_simple_template,
    )
    gen_from_ds_btn = ft.TextButton(
        "Generate from built dataset",
        icon=ICONS.BUILD,
        on_click=lambda e: _schedule_task(page, _on_generate_from_dataset),
    )
    gen_with_ollama_btn = ft.ElevatedButton(
        "Generate with Ollama",
        icon=getattr(ICONS, "SMART_TOY", ICONS.HUB),
        on_click=lambda e: _schedule_task(page, _on_generate_with_ollama),
    )
    clear_card_btn = ft.TextButton(
        "Clear",
        icon=ICONS.BACKSPACE,
        on_click=lambda e: (
            setattr(card_editor, "value", ""),
            _update_preview(),
            page.update(),
        ),
    )

    def update_status_placeholder() -> None:
        try:
            has_entries = len(getattr(timeline, "controls", []) or []) > 0
            timeline_placeholder.visible = not has_entries
            try:
                ctl = status_section_ref.get("control")
                if ctl is not None:
                    ctl.visible = has_entries
            except Exception:
                pass
        except Exception:
            pass
        page.update()

    def on_refresh_build(_):
        cancel_build["cancelled"] = False
        timeline.controls.clear()
        for k in split_badges:
            label = {"train": "Train", "val": "Val", "test": "Test"}[k]
            split_badges[k].content = pill(
                f"{label}: 0",
                split_meta[k][0],
                split_meta[k][1],
            ).content
        push_state["inflight"] = False
        push_ring.visible = False
        # Re-enable push button if it was disabled
        for ctl in build_actions.controls:
            if isinstance(ctl, ft.TextButton) and "Push + Upload README" in getattr(
                ctl, "text", ""
            ):
                ctl.disabled = False
        update_status_placeholder()

    async def on_build():
        # Delegate to helper to keep controller slim
        hf_cfg_token = (
            (_hf_cfg.get("token") or "").strip() if isinstance(_hf_cfg, dict) else ""
        )
        return await run_build_helper(
            page=page,
            source_mode=source_mode,
            data_file=data_file,
            merged_dir=merged_dir,
            seed=seed,
            shuffle=shuffle,
            val_slider=val_slider,
            test_slider=test_slider,
            min_len_b=min_len_b,
            save_dir=save_dir,
            push_toggle=push_toggle,
            repo_id=repo_id,
            private=private,
            token_val_ui=token_val_ui,
            timeline=timeline,
            timeline_placeholder=timeline_placeholder,
            split_badges=split_badges,
            split_meta=split_meta,
            dd_ref=dd_ref,
            cancel_build=cancel_build,
            use_custom_card=use_custom_card,
            card_editor=card_editor,
            hf_cfg_token=hf_cfg_token,
            update_status_placeholder=update_status_placeholder,
        )

    async def on_push_async():
        # Delegate to helper to keep controller slim
        hf_cfg_token = (
            (_hf_cfg.get("token") or "").strip() if isinstance(_hf_cfg, dict) else ""
        )
        return await run_push_async_helper(
            page=page,
            repo_id=repo_id,
            token_val_ui=token_val_ui,
            private=private,
            dd_ref=dd_ref,
            push_state=push_state,
            push_ring=push_ring,
            build_actions=build_actions,
            timeline=timeline,
            timeline_placeholder=timeline_placeholder,
            update_status_placeholder=update_status_placeholder,
            use_custom_card=use_custom_card,
            card_editor=card_editor,
            hf_cfg_token=hf_cfg_token,
        )

    def on_cancel_build(_):
        cancel_build["cancelled"] = True
        # Surface immediate feedback in the timeline
        try:
            timeline.controls.append(
                ft.Row(
                    [
                        ft.Icon(ICONS.CANCEL, color=COLORS.RED),
                        ft.Text("Cancel requested — will stop ASAP"),
                    ]
                )
            )
            update_status_placeholder()
        except Exception:
            pass

    build_actions = ft.Row(
        [
            ft.ElevatedButton(
                "Build Dataset",
                icon=ICONS.BUILD,
                on_click=lambda e: _schedule_task(page, on_build),
            ),
            ft.OutlinedButton(
                "Cancel",
                icon=ICONS.CANCEL,
                on_click=on_cancel_build,
            ),
            ft.TextButton(
                "Refresh",
                icon=REFRESH_ICON,
                on_click=on_refresh_build,
            ),
            ft.TextButton(
                "Push + Upload README",
                icon=ICONS.CLOUD_UPLOAD,
                on_click=lambda e: _schedule_task(page, on_push_async),
            ),
            push_ring,
        ],
        spacing=10,
    )

    build_tab = build_build_tab(
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
        push_toggle=push_toggle,
        repo_id=repo_id,
        private=private,
        token_val_ui=token_val_ui,
        build_actions=build_actions,
        use_custom_card=use_custom_card,
        card_preview_switch=card_preview_switch,
        load_template_btn=load_template_btn,
        gen_from_ds_btn=gen_from_ds_btn,
        gen_with_ollama_btn=gen_with_ollama_btn,
        clear_card_btn=clear_card_btn,
        ollama_gen_status=ollama_gen_status,
        card_editor=card_editor,
        card_preview_container=card_preview_container,
        timeline=timeline,
        timeline_placeholder=timeline_placeholder,
        status_section_ref=status_section_ref,
    )
    update_status_placeholder()

    return build_tab
