from __future__ import annotations

import asyncio
import os

import flet as ft

# Local helpers and theming
from helpers.common import safe_update
from helpers.theme import COLORS, ICONS
from helpers.ui import pill

# save_dataset utilities (local, with PYTHONPATH pointing to project src)
try:
    import save_dataset as sd
except Exception:  # pragma: no cover - fallback for alternate runtimes
    import sys as _sys

    _sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import save_dataset as sd

# datasets (optional)
try:  # pragma: no cover - optional dependency
    from datasets import DatasetDict, load_from_disk
except Exception:  # pragma: no cover
    DatasetDict = None  # type: ignore
    load_from_disk = None  # type: ignore


async def run_build(
    *,
    page: ft.Page,
    # UI state/controls
    source_mode: ft.Dropdown,
    data_source_dd: ft.Dropdown,
    db_session_dd: ft.Dropdown,
    data_file: ft.TextField,
    merged_dir: ft.TextField,
    seed: ft.TextField,
    shuffle: ft.Switch,
    val_slider: ft.Slider,
    test_slider: ft.Slider,
    min_len_b: ft.TextField,
    save_dir: ft.TextField,
    push_toggle: ft.Switch,
    repo_id: ft.TextField,
    private: ft.Switch,
    token_val_ui: ft.TextField,
    timeline: ft.ListView,
    timeline_placeholder: ft.Control,
    split_badges: dict,
    split_meta: dict,
    # Shared refs
    dd_ref: dict,
    cancel_build: dict,
    use_custom_card: ft.Control,
    card_editor: ft.TextField,
    # Config
    hf_cfg_token: str,
    update_status_placeholder=None,
) -> None:
    """Build dataset pipeline invoked from Build/Publish tab.
    Mirrors previous in-file logic from main.on_build.
    """

    def add_step(text: str, color, icon):
        timeline.controls.append(ft.Row([ft.Icon(icon, color=color), ft.Text(text)]))
        try:
            if update_status_placeholder is not None:
                update_status_placeholder()
            else:
                timeline_placeholder.visible = len(timeline.controls) == 0
        except Exception:
            pass

    # Reset UI
    cancel_build["cancelled"] = False
    timeline.controls.clear()
    for k in split_badges:
        label = {"train": "Train", "val": "Val", "test": "Test"}[k]
        split_badges[k].content = pill(f"{label}: 0", split_meta[k][0], split_meta[k][1]).content
    await safe_update(page)

    # Parse inputs
    db_session_id = (db_session_dd.value or "").strip()
    source_val = (source_mode.value or "Database").strip()
    merged_path = merged_dir.value or "merged_dataset"
    try:
        seed_val = int(seed.value or 42)
    except Exception:
        seed_val = 42
    shuffle_val = bool(shuffle.value)
    try:
        val_frac = float(val_slider.value or 0.0)
    except Exception:
        val_frac = 0.0
    try:
        test_frac = float(test_slider.value or 0.0)
    except Exception:
        test_frac = 0.0
    try:
        min_len_val = int(min_len_b.value or 1)
    except Exception:
        min_len_val = 1
    out_dir = save_dir.value or "hf_dataset"
    do_push = bool(push_toggle.value)
    repo = (repo_id.value or "").strip()
    is_private = bool(private.value)
    token_val_ui_s = (token_val_ui.value or "").strip()
    saved_tok = (hf_cfg_token or "").strip()
    token_val = saved_tok or token_val_ui_s

    # If using locally merged dataset, skip JSON pipeline and load from disk
    if source_val == "Merged dataset":
        add_step(f"Loading merged dataset from: {merged_path}", COLORS.BLUE, ICONS.UPLOAD_FILE)
        await safe_update(page)
        try:
            if load_from_disk is None:
                raise RuntimeError("datasets.load_from_disk unavailable")
            loaded = await asyncio.to_thread(load_from_disk, merged_path)
        except Exception as e:
            add_step(f"Failed loading merged dataset: {e}", COLORS.RED, ICONS.ERROR_OUTLINE)
            page.snack_bar = ft.SnackBar(ft.Text(f"Merged dataset not found or invalid at '{merged_path}'."))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        # Coerce to DatasetDict if needed
        try:
            is_dd = DatasetDict is not None and isinstance(loaded, DatasetDict)
        except Exception:
            is_dd = False
        if not is_dd:
            try:
                # Treat as single-train dataset
                loaded = DatasetDict({"train": loaded}) if DatasetDict is not None else loaded
            except Exception:
                add_step("Loaded object is not a datasets.Dataset or DatasetDict", COLORS.RED, ICONS.ERROR_OUTLINE)
                await safe_update(page)
                return

        dd = loaded

        # Update split badges with counts
        train_n = len(dd.get("train", []))
        val_n = len(dd.get("validation", [])) if "validation" in dd else 0
        test_n = len(dd.get("test", [])) if "test" in dd else 0
        split_badges["train"].content = pill(
            f"Train: {train_n}", split_meta["train"][0], split_meta["train"][1]
        ).content
        split_badges["val"].content = pill(f"Val: {val_n}", split_meta["val"][0], split_meta["val"][1]).content
        split_badges["test"].content = pill(f"Test: {test_n}", split_meta["test"][0], split_meta["test"][1]).content
        await safe_update(page)

        # Save to disk with heartbeat
        save_text = ft.Text(f"Saving dataset to {out_dir}")
        timeline.controls.append(ft.Row([ft.Icon(ICONS.SAVE_ALT, color=COLORS.BLUE), save_text]))
        await safe_update(page)
        save_task = asyncio.create_task(
            asyncio.to_thread(lambda: (os.makedirs(out_dir, exist_ok=True), dd.save_to_disk(out_dir)))
        )
        i = 0
        while not save_task.done():
            if cancel_build["cancelled"]:
                save_text.value = "Cancel requested — waiting for current save to finish…"
            else:
                save_text.value = f"Saving dataset{'.' * (i % 4)}"
            i += 1
            await safe_update(page)
            await asyncio.sleep(0.4)
        try:
            await save_task
        except Exception as e:
            save_text.value = f"Save failed: {e}"
            timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED), ft.Text("Save failed")]))
            await safe_update(page)
            return
        save_text.value = "Saved dataset"
        timeline.controls.append(ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text("Save complete")]))
        await safe_update(page)
        dd_ref["dd"] = dd

        # Optional push
        if do_push:
            if not repo:
                add_step("Missing Repo ID for push", COLORS.RED, ICONS.ERROR_OUTLINE)
                await safe_update(page)
                return
            if not token_val:
                try:
                    token_val = os.environ.get("HF_TOKEN") or sd.HfFolder.get_token()
                except Exception:
                    token_val = ""
            if not token_val:
                add_step("No HF token found — cannot push", COLORS.RED, ICONS.ERROR_OUTLINE)
                await safe_update(page)
                return
            add_step(f"Pushing to Hub: {repo}", COLORS.BLUE, ICONS.CLOUD_UPLOAD)
            await safe_update(page)
            try:
                push_text = ft.Text(f"Pushing to Hub: {repo}")
                timeline.controls.append(ft.Row([ft.Icon(ICONS.CLOUD_UPLOAD, color=COLORS.BLUE), push_text]))
                await safe_update(page)
                push_task = asyncio.create_task(asyncio.to_thread(sd.push_to_hub, dd, repo, is_private, token_val))
                j = 0
                while not push_task.done():
                    if cancel_build["cancelled"]:
                        push_text.value = "Cancel requested — waiting for current upload to finish…"
                    else:
                        push_text.value = f"Uploading{'.' * (j % 4)}"
                    j += 1
                    await safe_update(page)
                    await asyncio.sleep(0.6)
                await push_task

                if cancel_build["cancelled"]:
                    timeline.controls.append(
                        ft.Row([ft.Icon(ICONS.CANCEL, color=COLORS.RED), ft.Text("Build cancelled by user")])
                    )
                    await safe_update(page)
                    return

                # Prepare README (custom or autogenerated) and upload
                readme_text = ft.Text("Preparing dataset card (README)")
                timeline.controls.append(ft.Row([ft.Icon(ICONS.ARTICLE, color=COLORS.BLUE), readme_text]))
                await safe_update(page)
                _custom_enabled = bool(getattr(use_custom_card, "value", False))
                _custom_text = (getattr(card_editor, "value", "") or "").strip()
                if _custom_enabled and _custom_text:
                    readme = _custom_text
                else:
                    readme = await asyncio.to_thread(sd.build_dataset_card_content, dd, repo)
                readme_text.value = "Prepared dataset card"
                await safe_update(page)

                up_text = ft.Text("Uploading README.md")
                timeline.controls.append(ft.Row([ft.Icon(ICONS.ARTICLE, color=COLORS.BLUE), up_text]))
                await safe_update(page)
                up_task = asyncio.create_task(asyncio.to_thread(sd.upload_readme, repo, token_val, readme))
                k = 0
                while not up_task.done():
                    if cancel_build["cancelled"]:
                        up_text.value = "Cancel requested — waiting for current upload to finish…"
                    else:
                        up_text.value = f"Uploading README{'.' * (k % 4)}"
                    k += 1
                    await safe_update(page)
                    await asyncio.sleep(0.6)
                await up_task

                _url = f"https://huggingface.co/datasets/{repo}"
                timeline.controls.append(
                    ft.Row(
                        [
                            ft.Icon(ICONS.OPEN_IN_NEW, color=COLORS.BLUE),
                            ft.TextButton("Open on Hugging Face", on_click=lambda e, u=_url: page.launch_url(u)),
                        ]
                    )
                )
                add_step("Push complete!", COLORS.GREEN, ICONS.CHECK_CIRCLE)
                page.snack_bar = ft.SnackBar(ft.Text("Pushed to Hub 🚀"))
                page.open(page.snack_bar)
                await safe_update(page)
            except Exception as e:
                add_step(f"Push failed: {e}", COLORS.RED, ICONS.ERROR_OUTLINE)
                await safe_update(page)

        return

    # Validate splits
    if val_frac + test_frac >= 1.0:
        add_step("Invalid split: val+test must be < 1.0", COLORS.RED, ICONS.ERROR_OUTLINE)
        page.snack_bar = ft.SnackBar(ft.Text("Invalid split: val+test must be < 1.0"))
        page.open(page.snack_bar)
        await safe_update(page)
        return

    # Step 1: Load
    add_step("Loading data", COLORS.BLUE, ICONS.UPLOAD_FILE)
    await safe_update(page)
    try:
        if source_val == "Database" and db_session_id:
            # Load from database session
            from db.scraped_data import get_pairs_for_session, get_scrape_session

            session = get_scrape_session(int(db_session_id))
            if not session:
                add_step(f"Database session {db_session_id} not found", COLORS.RED, ICONS.ERROR_OUTLINE)
                await safe_update(page)
                return
            pairs = await asyncio.to_thread(lambda: get_pairs_for_session(int(db_session_id)))
            if not pairs:
                add_step(f"No pairs found in session {db_session_id}", COLORS.RED, ICONS.ERROR_OUTLINE)
                await safe_update(page)
                return
            records = pairs
            add_step(f"Loaded {len(records)} pairs from database session", COLORS.GREEN, ICONS.CHECK_CIRCLE)
        else:
            add_step("No database session selected", COLORS.RED, ICONS.ERROR_OUTLINE)
            await safe_update(page)
            return
    except Exception as e:
        add_step(f"Failed loading data: {e}", COLORS.RED, ICONS.ERROR_OUTLINE)
        await safe_update(page)
        return
    if cancel_build["cancelled"]:
        add_step("Build cancelled by user", COLORS.RED, ICONS.CANCEL)
        await safe_update(page)
        return

    # Step 2: Normalize
    add_step("Normalizing records", COLORS.BLUE, ICONS.SHUFFLE)
    await safe_update(page)
    try:
        examples = await asyncio.to_thread(sd.normalize_records, records, min_len_val)
    except Exception as e:
        add_step(f"Normalization failed: {e}", COLORS.RED, ICONS.ERROR_OUTLINE)
        await safe_update(page)
        return
    if not examples:
        add_step("No valid examples after normalization", COLORS.RED, ICONS.ERROR_OUTLINE)
        page.snack_bar = ft.SnackBar(ft.Text("No valid examples after normalization."))
        page.open(page.snack_bar)
        await safe_update(page)
        return
    if cancel_build["cancelled"]:
        add_step("Build cancelled by user", COLORS.RED, ICONS.CANCEL)
        await safe_update(page)
        return

    # Step 3: Splits
    add_step("Creating splits (train/val/test)", COLORS.BLUE, ICONS.CALENDAR_VIEW_MONTH)
    await safe_update(page)
    try:
        dd = await asyncio.to_thread(
            sd.build_dataset_dict,
            examples,
            val_frac,
            test_frac,
            shuffle_val,
            seed_val,
        )
    except Exception as e:
        add_step(f"Split creation failed: {e}", COLORS.RED, ICONS.ERROR_OUTLINE)
        await safe_update(page)
        return

    # Update split badges with real counts
    train_n = len(dd.get("train", []))
    val_n = len(dd.get("validation", [])) if "validation" in dd else 0
    test_n = len(dd.get("test", [])) if "test" in dd else 0
    split_badges["train"].content = pill(f"Train: {train_n}", split_meta["train"][0], split_meta["train"][1]).content
    split_badges["val"].content = pill(f"Val: {val_n}", split_meta["val"][0], split_meta["val"][1]).content
    split_badges["test"].content = pill(f"Test: {test_n}", split_meta["test"][0], split_meta["test"][1]).content
    await safe_update(page)
    if cancel_build["cancelled"]:
        add_step("Build cancelled by user", COLORS.RED, ICONS.CANCEL)
        await safe_update(page)
        return

    # Step 4: Save to disk (with heartbeat + cancel-aware)
    save_text = ft.Text(f"Saving dataset to {out_dir}")
    timeline.controls.append(ft.Row([ft.Icon(ICONS.SAVE_ALT, color=COLORS.BLUE), save_text]))
    await safe_update(page)
    save_task = asyncio.create_task(
        asyncio.to_thread(lambda: (os.makedirs(out_dir, exist_ok=True), dd.save_to_disk(out_dir)))
    )
    i = 0
    while not save_task.done():
        if cancel_build["cancelled"]:
            save_text.value = "Cancel requested — waiting for current save to finish…"
        else:
            save_text.value = f"Saving dataset{'.' * (i % 4)}"
        i += 1
        await safe_update(page)
        await asyncio.sleep(0.4)
    try:
        await save_task
    except Exception as e:
        save_text.value = f"Save failed: {e}"
        timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED), ft.Text("Save failed")]))
        await safe_update(page)
        return
    save_text.value = "Saved dataset"
    timeline.controls.append(ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text("Save complete")]))
    await safe_update(page)
    dd_ref["dd"] = dd
    if cancel_build["cancelled"]:
        timeline.controls.append(ft.Row([ft.Icon(ICONS.CANCEL, color=COLORS.RED), ft.Text("Build cancelled by user")]))
        await safe_update(page)
        return

    # Optional push
    if do_push:
        if not repo:
            add_step("Missing Repo ID for push", COLORS.RED, ICONS.ERROR_OUTLINE)
            await safe_update(page)
            return
        if not token_val:
            try:
                token_val = os.environ.get("HF_TOKEN") or sd.HfFolder.get_token()
            except Exception:
                token_val = os.environ.get("HF_TOKEN") or ""
        if not token_val:
            add_step("No HF token found — cannot push", COLORS.RED, ICONS.ERROR_OUTLINE)
            await safe_update(page)
            return
        add_step(f"Pushing to Hub: {repo}", COLORS.BLUE, ICONS.CLOUD_UPLOAD)
        await safe_update(page)
        try:
            # Push with heartbeat
            push_text = ft.Text(f"Pushing to Hub: {repo}")
            timeline.controls.append(ft.Row([ft.Icon(ICONS.CLOUD_UPLOAD, color=COLORS.BLUE), push_text]))
            await safe_update(page)
            push_task = asyncio.create_task(asyncio.to_thread(sd.push_to_hub, dd, repo, is_private, token_val))
            j = 0
            while not push_task.done():
                if cancel_build["cancelled"]:
                    push_text.value = "Cancel requested — waiting for current upload to finish…"
                else:
                    push_text.value = f"Uploading{'.' * (j % 4)}"
                j += 1
                await safe_update(page)
                await asyncio.sleep(0.6)
            await push_task

            if cancel_build["cancelled"]:
                timeline.controls.append(
                    ft.Row([ft.Icon(ICONS.CANCEL, color=COLORS.RED), ft.Text("Build cancelled by user")])
                )
                await safe_update(page)
                return

            # Prepare README content (custom or autogenerated)
            readme_text = ft.Text("Preparing dataset card (README)")
            timeline.controls.append(ft.Row([ft.Icon(ICONS.ARTICLE, color=COLORS.BLUE), readme_text]))
            await safe_update(page)
            _custom_enabled = bool(getattr(use_custom_card, "value", False))
            _custom_text = (getattr(card_editor, "value", "") or "").strip()
            if _custom_enabled and _custom_text:
                readme = _custom_text
            else:
                readme = await asyncio.to_thread(sd.build_dataset_card_content, dd, repo)
            readme_text.value = "Prepared dataset card"
            await safe_update(page)

            # Upload README with heartbeat
            up_text = ft.Text("Uploading README.md")
            timeline.controls.append(ft.Row([ft.Icon(ICONS.ARTICLE, color=COLORS.BLUE), up_text]))
            await safe_update(page)
            up_task = asyncio.create_task(asyncio.to_thread(sd.upload_readme, repo, token_val, readme))
            k = 0
            while not up_task.done():
                if cancel_build["cancelled"]:
                    up_text.value = "Cancel requested — waiting for current upload to finish…"
                else:
                    up_text.value = f"Uploading README{'.' * (k % 4)}"
                k += 1
                await safe_update(page)
                await asyncio.sleep(0.6)
            await up_task

            _url = f"https://huggingface.co/datasets/{repo}"
            timeline.controls.append(
                ft.Row(
                    [
                        ft.Icon(ICONS.OPEN_IN_NEW, color=COLORS.BLUE),
                        ft.TextButton("Open on Hugging Face", on_click=lambda e, u=_url: page.launch_url(u)),
                    ]
                )
            )
            add_step("Push complete!", COLORS.GREEN, ICONS.CHECK_CIRCLE)
            page.snack_bar = ft.SnackBar(ft.Text("Pushed to Hub 🚀"))
            page.open(page.snack_bar)
            await safe_update(page)
        except Exception as e:
            add_step(f"Push failed: {e}", COLORS.RED, ICONS.ERROR_OUTLINE)
            await safe_update(page)


async def run_push_async(
    *,
    page: ft.Page,
    repo_id: ft.TextField,
    token_val_ui: ft.TextField,
    private: ft.Switch,
    dd_ref: dict,
    push_state: dict,
    push_ring: ft.ProgressRing,
    build_actions: ft.Row,
    timeline: ft.ListView,
    timeline_placeholder: ft.Control,
    update_status_placeholder,
    use_custom_card: ft.Control,
    card_editor: ft.TextField,
    hf_cfg_token: str,
) -> None:
    """Push the most recent built dataset and upload README, with UI heartbeat."""
    if push_state.get("inflight"):
        return
    dd = dd_ref.get("dd")
    if dd is None:
        page.snack_bar = ft.SnackBar(ft.Text("Build the dataset first."))
        page.open(page.snack_bar)
        await safe_update(page)
        return

    repo = (repo_id.value or "").strip()
    tok = (token_val_ui.value or "").strip() or (hf_cfg_token or "").strip()
    if not tok:
        try:
            tok = os.environ.get("HF_TOKEN") or getattr(sd.HfFolder, "get_token", lambda: "")()
        except Exception:
            tok = ""
    if not repo or not tok:
        page.snack_bar = ft.SnackBar(ft.Text("Repo ID and a valid HF token are required."))
        page.open(page.snack_bar)
        await safe_update(page)
        return

    # UI: show inflight state
    push_state["inflight"] = True
    push_ring.visible = True
    # Disable the Push button while pushing
    for ctl in build_actions.controls:
        if isinstance(ctl, ft.TextButton) and "Push + Upload README" in getattr(ctl, "text", ""):
            ctl.disabled = True
    timeline.controls.append(
        ft.Row([ft.Icon(ICONS.CLOUD_UPLOAD, color=COLORS.BLUE), ft.Text(f"Pushing to Hub: {repo}")])
    )
    await safe_update(page)
    update_status_placeholder()

    try:
        await asyncio.to_thread(sd.push_to_hub, dd, repo, bool(private.value), tok)
        timeline.controls.append(
            ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text("Dataset pushed to Hub")])
        )
        await safe_update(page)

        _custom_enabled = bool(getattr(use_custom_card, "value", False))
        _custom_text = (getattr(card_editor, "value", "") or "").strip()
        if _custom_enabled and _custom_text:
            readme = _custom_text
        else:
            readme = await asyncio.to_thread(sd.build_dataset_card_content, dd, repo)
        await asyncio.to_thread(sd.upload_readme, repo, tok, readme)
        timeline.controls.append(
            ft.Row([ft.Icon(ICONS.ARTICLE, color=COLORS.GREEN), ft.Text("Uploaded dataset card (README)")])
        )
        # Add link to dataset on Hub
        _url = f"https://huggingface.co/datasets/{repo}"
        timeline.controls.append(
            ft.Row(
                [
                    ft.Icon(ICONS.OPEN_IN_NEW, color=COLORS.BLUE),
                    ft.TextButton("Open on Hugging Face", on_click=lambda e, u=_url: page.launch_url(u)),
                ]
            )
        )
        timeline.controls.append(ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text("Push complete!")]))
        page.snack_bar = ft.SnackBar(ft.Text("Pushed to Hub 🚀"))
        page.open(page.snack_bar)
        await safe_update(page)
    except Exception as e:
        timeline.controls.append(ft.Row([ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED), ft.Text(f"Push failed: {e}")]))
        await safe_update(page)
    finally:
        push_state["inflight"] = False
        push_ring.visible = False
        # Re-enable Push button
        for ctl in build_actions.controls:
            if isinstance(ctl, ft.TextButton) and "Push + Upload README" in getattr(ctl, "text", ""):
                ctl.disabled = False
        update_status_placeholder()
        await safe_update(page)
