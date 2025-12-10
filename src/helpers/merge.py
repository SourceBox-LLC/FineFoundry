from __future__ import annotations

import asyncio
import json
import os
from typing import Optional, List

import flet as ft

from helpers.common import safe_update
from helpers.logging_config import get_logger
from helpers.theme import COLORS, ICONS, ACCENT_COLOR
from helpers.ui import WITH_OPACITY, compute_two_col_flex, two_col_header, two_col_row
from helpers.datasets import guess_input_output_columns

# Initialize logger
logger = get_logger(__name__)

try:
    import save_dataset as sd
except Exception:
    import sys as _sys

    _sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import save_dataset as sd

# Optional datasets dependency
try:
    from datasets import load_dataset, Dataset, DatasetDict, get_dataset_config_names, load_from_disk
except Exception:
    load_dataset = None  # type: ignore
    Dataset = None  # type: ignore
    DatasetDict = None  # type: ignore
    get_dataset_config_names = None  # type: ignore
    load_from_disk = None  # type: ignore


async def _load_and_prepare(
    repo: str,
    split: str,
    config: Optional[str],
    in_col: Optional[str],
    out_col: Optional[str],
    *,
    timeline: ft.ListView | None = None,
    page: ft.Page | None = None,
) -> List:
    if load_dataset is None:
        raise RuntimeError("datasets library not available — cannot load from Hub")

    def do_load():
        if split == "all":
            dd = load_dataset(repo, name=(config or None))
            return dd
        return load_dataset(repo, split=split, name=(config or None))

    try:
        obj = await asyncio.to_thread(do_load)
    except Exception as e:
        msg = str(e).lower()
        auto_loaded = False
        if (get_dataset_config_names is not None) and (
            "config name is missing" in msg or "config name is required" in msg
        ):
            try:
                cfgs = await asyncio.to_thread(lambda: get_dataset_config_names(repo))
            except Exception:
                cfgs = []
            pick = None
            for pref in ("main", "default", "socratic"):
                if pref in cfgs:
                    pick = pref
                    break
            if not pick and cfgs:
                pick = cfgs[0]
            if pick:
                try:
                    if timeline is not None:
                        timeline.controls.append(
                            ft.Row(
                                [
                                    ft.Icon(ICONS.INFO, color=WITH_OPACITY(0.8, ACCENT_COLOR)),
                                    ft.Text(f"'{repo}' requires a config; using '{pick}' automatically"),
                                ]
                            )
                        )
                    if page is not None:
                        await safe_update(page)
                except Exception:
                    pass

                def do_load_cfg():
                    if split == "all":
                        return load_dataset(repo, name=pick)
                    return load_dataset(repo, split=split, name=pick)

                obj = await asyncio.to_thread(do_load_cfg)
                auto_loaded = True
        if not auto_loaded:
            raise

    ds_list: list = []
    if isinstance(obj, dict) or (DatasetDict is not None and isinstance(obj, DatasetDict)):
        for k in ["train", "validation", "test"]:
            try:
                if k in obj:
                    ds_list.append(obj[k])
            except Exception:
                pass
        try:
            for k in getattr(obj, "keys", lambda: [])():
                if k not in {"train", "validation", "test"}:
                    ds_list.append(obj[k])
        except Exception:
            pass
    else:
        ds_list = [obj]

    prepped = []
    for ds in ds_list:
        try:
            names = list(getattr(ds, "column_names", []) or [])
        except Exception:
            names = []
        inn = (in_col or "").strip() or None
        outn = (out_col or "").strip() or None
        if not inn or inn not in names or not outn or outn not in names:
            gi, go = guess_input_output_columns(names)
            inn = inn if (inn and inn in names) else gi
            outn = outn if (outn and outn in names) else go
        if not inn or not outn:
            raise RuntimeError(f"Could not resolve input/output columns for {repo} (have: {', '.join(names)})")

        def mapper(batch):
            src = batch.get(inn, [])
            tgt = batch.get(outn, [])
            return {
                "input": ["" if v is None else str(v).strip() for v in src],
                "output": ["" if v is None else str(v).strip() for v in tgt],
            }

        try:
            mapped = await asyncio.to_thread(
                lambda: ds.map(mapper, batched=True, remove_columns=list(getattr(ds, "column_names", []) or []))
            )
        except Exception:
            try:
                to_list = [
                    {
                        "input": "" if r.get(inn) is None else str(r.get(inn)).strip(),
                        "output": "" if r.get(outn) is None else str(r.get(outn)).strip(),
                    }
                    for r in ds
                ]
                if Dataset is None:
                    raise RuntimeError("datasets.Dataset unavailable to construct from list")
                mapped = await asyncio.to_thread(lambda: Dataset.from_list(to_list))
            except Exception as e:
                raise RuntimeError(f"Failed to map columns for {repo}: {e}")

        try:
            mapped = await asyncio.to_thread(
                lambda: mapped.filter(
                    lambda r: (len(r.get("input", "") or "") > 0 and len(r.get("output", "") or "") > 0)
                )
            )
        except Exception:
            pass

        prepped.append(mapped)
    return prepped


async def run_merge(
    *,
    page: ft.Page,
    rows_host: ft.Column,
    merge_op: ft.Dropdown,
    merge_output_format: ft.Dropdown,
    merge_session_name: ft.TextField,
    merge_export_path: ft.TextField | None = None,
    merge_timeline: ft.ListView,
    merge_timeline_placeholder: ft.Control,
    merge_preview_host: ft.ListView,
    merge_preview_placeholder: ft.Control,
    merge_cancel: dict,
    merge_busy_ring: ft.ProgressRing,
    download_button: ft.Control | None = None,
    update_merge_placeholders=None,
) -> None:
    logger.info("Starting merge operation")
    merge_cancel["cancelled"] = False
    merge_timeline.controls.clear()
    merge_preview_host.controls.clear()
    merge_busy_ring.visible = True
    try:
        merge_timeline_placeholder.visible = True
        merge_preview_placeholder.visible = True
    except Exception:
        pass
    try:
        if update_merge_placeholders is not None:
            update_merge_placeholders()
    except Exception:
        pass
    await safe_update(page)

    # Validate inputs
    rows = [r for r in list(getattr(rows_host, "controls", []) or []) if isinstance(r, ft.Row)]
    entries = []
    for r in rows:
        d = getattr(r, "data", None) or {}
        src_dd = d.get("source")
        db_session_dd = d.get("db_session")
        ds_tf = d.get("ds")
        sp_dd = d.get("split")
        cfg_tf = d.get("config")
        in_tf = d.get("in")
        out_tf = d.get("out")
        src = (getattr(src_dd, "value", "Database") or "Database") if src_dd else "Database"
        repo = (getattr(ds_tf, "value", "") or "").strip() if ds_tf else ""
        db_session_id = (getattr(db_session_dd, "value", "") or "").strip() if db_session_dd else ""
        if (src == "Database" and db_session_id) or (src == "Hugging Face" and repo):
            entries.append(
                {
                    "source": src,
                    "db_session_id": db_session_id,
                    "repo": repo,
                    "split": (getattr(sp_dd, "value", "train") or "train") if sp_dd else "train",
                    "config": (getattr(cfg_tf, "value", "") or "").strip() if cfg_tf else "",
                    "in": (getattr(in_tf, "value", "") or "").strip() if in_tf else "",
                    "out": (getattr(out_tf, "value", "") or "").strip() if out_tf else "",
                }
            )
    if len(entries) < 2:
        logger.warning(f"Merge validation failed: Only {len(entries)} dataset(s) provided")
        merge_timeline.controls.append(
            ft.Row([ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED), ft.Text("Add at least two datasets")])
        )
        try:
            merge_timeline_placeholder.visible = len(merge_timeline.controls) == 0
        except Exception:
            pass
        try:
            if update_merge_placeholders is not None:
                update_merge_placeholders()
        except Exception:
            pass
        await safe_update(page)
        merge_busy_ring.visible = False
        await safe_update(page)
        return

    session_name = (merge_session_name.value or "merged_dataset").strip()
    export_path = (merge_export_path.value if merge_export_path else "").strip() or None
    op = merge_op.value or "Concatenate"
    fmt = (merge_output_format.value or "Database").lower()
    wants_export = "export" in fmt

    logger.info(f"Merge config - Operation: {op}, Session: {session_name}, Format: {fmt}, Datasets: {len(entries)}")

    # Load and map each dataset
    hf_prepped = []  # list[Dataset]
    json_sources: List[List[dict]] = []
    for i, ent in enumerate(entries, start=1):
        if merge_cancel.get("cancelled"):
            break
        src = ent.get("source", "Database")
        if src == "Database":
            label = f"DB session {ent.get('db_session_id', '?')}"
        elif src == "Hugging Face":
            label = ent["repo"]
        else:
            label = ent.get("json", "(json)")
        split_lbl = ent["split"] if src == "Hugging Face" else "-"
        merge_timeline.controls.append(
            ft.Row([ft.Icon(ICONS.DOWNLOAD, color=COLORS.BLUE), ft.Text(f"Loading {label} [{split_lbl}]…")])
        )
        try:
            await safe_update(page)
        except Exception:
            pass
        try:
            if src == "Database":
                # Load from database session
                db_session_id = ent.get("db_session_id")
                if not db_session_id:
                    raise RuntimeError("Database session ID required")
                from db.scraped_data import get_pairs_for_session, get_scrape_session

                session = get_scrape_session(int(db_session_id))
                if not session:
                    raise RuntimeError(f"Session {db_session_id} not found")
                records = await asyncio.to_thread(lambda: get_pairs_for_session(int(db_session_id)))
                if not records:
                    raise RuntimeError(f"No pairs in session {db_session_id}")
                data = [{"input": r["input"], "output": r["output"]} for r in records]
                # Always use JSON format for merging
                json_sources.append(data)
                merge_timeline.controls.append(
                    ft.Row(
                        [
                            ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN),
                            ft.Text(f"Prepared {len(data)} pairs from DB session"),
                        ]
                    )
                )
            elif src == "Hugging Face":
                dss = await _load_and_prepare(
                    ent["repo"], ent["split"], ent["config"], ent["in"], ent["out"], timeline=merge_timeline, page=page
                )
                hf_prepped.extend(dss)
                merge_timeline.controls.append(
                    ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text(f"Prepared {ent['repo']}")])
                )
            else:
                path = ent.get("json")
                if not path:
                    raise RuntimeError("JSON path required")
                try:
                    records = await asyncio.to_thread(sd.load_records, path)
                except Exception as e:
                    raise RuntimeError(f"Failed to read JSON: {e}")
                try:
                    data = await asyncio.to_thread(sd.normalize_records, records, 1)
                except Exception:
                    data = []
                    for r in records or []:
                        if isinstance(r, dict):
                            a = str((r.get("input") or "")).strip()
                            b = str((r.get("output") or "")).strip()
                            if a and b:
                                data.append({"input": a, "output": b})
                # Always use JSON format for merging
                json_sources.append(data)
                merge_timeline.controls.append(
                    ft.Row(
                        [ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text(f"Prepared {os.path.basename(path)}")]
                    )
                )
        except Exception as e:
            merge_timeline.controls.append(
                ft.Row([ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED), ft.Text(f"Failed {label}: {e}")])
            )
            await safe_update(page)
            merge_busy_ring.visible = False
            try:
                merge_timeline_placeholder.visible = len(merge_timeline.controls) == 0
                merge_preview_placeholder.visible = len(merge_preview_host.controls) == 0
            except Exception:
                pass
            try:
                if update_merge_placeholders is not None:
                    update_merge_placeholders()
            except Exception:
                pass
            await safe_update(page)
            return
        await safe_update(page)

    if merge_cancel.get("cancelled"):
        merge_timeline.controls.append(
            ft.Row([ft.Icon(ICONS.CANCEL, color=COLORS.RED), ft.Text("Merge cancelled by user")])
        )
        merge_busy_ring.visible = False
        try:
            merge_timeline_placeholder.visible = len(merge_timeline.controls) == 0
            merge_preview_placeholder.visible = len(merge_preview_host.controls) == 0
        except Exception:
            pass
        try:
            if update_merge_placeholders is not None:
                update_merge_placeholders()
        except Exception:
            pass
        await safe_update(page)
        return

    # Convert HF prepped to JSON if output is JSON
    # Convert HF datasets to JSON format for merging
    if hf_prepped:
        for ds in hf_prepped:
            try:
                exs = []
                for rec in ds:
                    exs.append({"input": (rec.get("input", "") or ""), "output": (rec.get("output", "") or "")})
                json_sources.append(exs)
            except Exception:
                pass

    # Merge - collect all examples into a single list
    try:
        if not json_sources:
            raise RuntimeError("No datasets to merge after preparation")

        merged_examples: List[dict] = []
        if op == "Interleave" and len(json_sources) > 1:
            indices = [0] * len(json_sources)
            total = sum(len(s) for s in json_sources)
            while len(merged_examples) < total:
                for i, s in enumerate(json_sources):
                    if indices[i] < len(s):
                        merged_examples.append(s[indices[i]])
                        indices[i] += 1
        else:
            for s in json_sources:
                merged_examples.extend(s)

        # Save to database
        from db.scraped_data import create_scrape_session, add_scraped_pairs

        # Create a new scrape session for the merged data
        session = await asyncio.to_thread(
            create_scrape_session,
            source="Merged",
            source_details=session_name,
            dataset_format="standard",
            metadata={"operation": op, "source_count": len(entries)},
        )
        session_id = session["id"]

        # Add all merged pairs to the session
        pairs_to_add = [{"input": rec.get("input", ""), "output": rec.get("output", "")} for rec in merged_examples]
        await asyncio.to_thread(add_scraped_pairs, session_id, pairs_to_add)

        logger.info(f"Saved {len(merged_examples)} merged records to database session {session_id}")
        merge_timeline.controls.append(
            ft.Row(
                [
                    ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN),
                    ft.Text(f"Saved {len(merged_examples)} pairs to database as '{session_name}'"),
                ]
            )
        )
        await safe_update(page)

        # Optionally export to JSON file
        if wants_export and export_path:
            out_abs = os.path.abspath(export_path)
            os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
            await asyncio.to_thread(
                lambda: open(out_abs, "w", encoding="utf-8").write(
                    json.dumps(merged_examples, ensure_ascii=False, indent=4)
                )
            )
            logger.info(f"Exported merged dataset to JSON: {out_abs}")
            merge_timeline.controls.append(
                ft.Row([ft.Icon(ICONS.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text(f"Exported to {out_abs}")])
            )
            await safe_update(page)

        # Populate inline preview with a small sample
        try:
            pairs = [
                (str(rec.get("input", "") or ""), str(rec.get("output", "") or ""))
                for rec in (merged_examples[:100] if merged_examples else [])
            ]
            merge_preview_host.controls.clear()
            if pairs:
                lfx, rfx = compute_two_col_flex(pairs)
                merge_preview_host.controls.append(two_col_header(left_flex=lfx, right_flex=rfx))
                for a, b in pairs:
                    merge_preview_host.controls.append(two_col_row(a, b, lfx, rfx))
            try:
                merge_preview_placeholder.visible = len(merge_preview_host.controls) == 0
            except Exception:
                pass
            try:
                if update_merge_placeholders is not None:
                    update_merge_placeholders()
            except Exception:
                pass
            await safe_update(page)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Merge operation failed: {str(e)}", exc_info=True)
        merge_timeline.controls.append(
            ft.Row([ft.Icon(ICONS.ERROR_OUTLINE, color=COLORS.RED), ft.Text(f"Merge failed: {e}")])
        )
        merge_busy_ring.visible = False
        try:
            merge_timeline_placeholder.visible = len(merge_timeline.controls) == 0
        except Exception:
            pass
        try:
            if update_merge_placeholders is not None:
                update_merge_placeholders()
        except Exception:
            pass
        await safe_update(page)
        return

    logger.info("Merge operation completed successfully")
    merge_busy_ring.visible = False
    try:
        merge_timeline_placeholder.visible = len(merge_timeline.controls) == 0
    except Exception:
        pass
    try:
        if update_merge_placeholders is not None:
            update_merge_placeholders()
    except Exception:
        pass
    # Show download button after successful merge
    if download_button is not None:
        try:
            download_button.visible = True
            logger.debug("Download button made visible after successful merge")
        except Exception:
            pass
    page.snack_bar = ft.SnackBar(ft.Text("Merge complete!"))
    page.open(page.snack_bar)
    await safe_update(page)


async def preview_merged(
    *,
    page: ft.Page,
    merge_output_format: ft.Dropdown,
    merge_save_dir: ft.TextField,
) -> None:
    try:
        page.snack_bar = ft.SnackBar(ft.Text("Opening merged dataset preview..."))
        page.open(page.snack_bar)
        await safe_update(page)
    except Exception:
        pass

    orig_dir = merge_save_dir.value or "merged_dataset"
    fmt_now = (merge_output_format.value or "").lower()
    wants_json = ("json" in fmt_now) or (str(orig_dir).lower().endswith(".json"))
    candidates = []
    if os.path.isabs(orig_dir):
        candidates.append(orig_dir)
    else:
        candidates.extend(
            [
                orig_dir,
                os.path.abspath(orig_dir),
                os.path.join(os.getcwd(), orig_dir),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), orig_dir),
            ]
        )
    seen = set()
    resolved_list: List[str] = []
    for pth in candidates:
        ap = os.path.abspath(pth)
        if ap not in seen:
            seen.add(ap)
            resolved_list.append(ap)
    existing = next((p for p in resolved_list if os.path.exists(p)), None)
    if not existing:
        page.snack_bar = ft.SnackBar(ft.Text("Merged dataset not found. Tried:\n" + "\n".join(resolved_list[:4])))
        page.open(page.snack_bar)
        await safe_update(page)
        return

    if wants_json:
        try:
            data = await asyncio.to_thread(sd.load_records, existing)
        except Exception as e:
            page.snack_bar = ft.SnackBar(ft.Text(f"Failed to read JSON: {e}"))
            page.open(page.snack_bar)
            await safe_update(page)
            return
        try:
            data = await asyncio.to_thread(sd.normalize_records, data, 1)
        except Exception:
            pass
        total = len(data or [])
        page_size = 100
        total_pages = max(1, (total + page_size - 1) // page_size)
        state = {"page": 0}

        grid_list = ft.ListView(expand=1, auto_scroll=False)
        info_text = ft.Text("")
        prev_btn = ft.TextButton("Prev")
        next_btn = ft.TextButton("Next")

        def render_page_json():
            start = state["page"] * page_size
            end = min(start + page_size, total)
            grid_list.controls.clear()
            pairs = []
            for rec in (data or [])[start:end]:
                pairs.append(((rec.get("input", "") or ""), (rec.get("output", "") or "")))
            lfx, rfx = compute_two_col_flex(pairs)
            grid_list.controls.append(two_col_header(left_flex=lfx, right_flex=rfx))
            for a, b in pairs:
                grid_list.controls.append(two_col_row(a, b, lfx, rfx))
            info_text.value = f"Page {state['page'] + 1}/{total_pages} • Showing {start + 1}-{end} of {total}"
            prev_btn.disabled = state["page"] <= 0
            next_btn.disabled = state["page"] >= (total_pages - 1)
            page.update()

        def on_prev_json(_):
            if state["page"] > 0:
                state["page"] -= 1
                render_page_json()

        def on_next_json(_):
            if state["page"] < (total_pages - 1):
                state["page"] += 1
                render_page_json()

        prev_btn.on_click = on_prev_json
        next_btn.on_click = on_next_json

        controls_bar = ft.Row(
            [
                prev_btn,
                next_btn,
                info_text,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Merged Dataset Viewer — {total} rows"),
            content=ft.Container(
                width=900,
                height=600,
                content=ft.Column(
                    [
                        controls_bar,
                        ft.Container(grid_list, expand=True),
                    ],
                    expand=True,
                ),
            ),
            actions=[],
        )

        def close_dlg_json(_):
            dlg.open = False
            page.update()

        dlg.actions = [ft.TextButton("Close", on_click=close_dlg_json)]
        try:
            dlg.on_dismiss = lambda e: page.update()
        except Exception:
            pass
        render_page_json()
        opened = False
        try:
            if hasattr(page, "open") and callable(getattr(page, "open")):
                page.open(dlg)
                opened = True
        except Exception:
            opened = False
        if not opened:
            page.dialog = dlg
            dlg.open = True
        await safe_update(page)
        return

    if load_from_disk is None:
        page.snack_bar = ft.SnackBar(ft.Text("datasets.load_from_disk unavailable — cannot open preview"))
        page.open(page.snack_bar)
        await safe_update(page)
        return

    try:
        obj = await asyncio.to_thread(lambda: load_from_disk(existing))
    except Exception as e:
        page.snack_bar = ft.SnackBar(ft.Text(f"Failed to load dataset from {existing}: {e}"))
        page.open(page.snack_bar)
        await safe_update(page)
        return

    ds = None
    try:
        if DatasetDict is not None and isinstance(obj, DatasetDict):
            for k in ["train", "validation", "test"]:
                if k in obj:
                    ds = obj[k]
                    break
            if ds is None:
                for k in getattr(obj, "keys", lambda: [])():
                    ds = obj[k]
                    break
        else:
            ds = obj
    except Exception:
        ds = obj
    if ds is None:
        page.snack_bar = ft.SnackBar(ft.Text("No split found to preview"))
        page.open(page.snack_bar)
        await safe_update(page)
        return

    try:
        total = len(ds)
    except Exception:
        try:
            total = int(getattr(ds, "num_rows", 0))
        except Exception:
            total = 0
    page_size = 100
    total_pages = max(1, (total + page_size - 1) // page_size)
    state = {"page": 0}

    grid_list = ft.ListView(expand=1, auto_scroll=False)
    info_text = ft.Text("")
    prev_btn = ft.TextButton("Prev")
    next_btn = ft.TextButton("Next")

    def render_page():
        start = state["page"] * page_size
        end = min(start + page_size, total)
        grid_list.controls.clear()
        try:
            idxs = list(range(start, end))
            page_ds = ds.select(idxs)
        except Exception:
            page_ds = None
        pairs = []
        try:
            if page_ds is not None:
                for rec in page_ds:
                    pairs.append((rec.get("input", "") or "", rec.get("output", "") or ""))
        except Exception:
            pass
        lfx, rfx = compute_two_col_flex(pairs)
        grid_list.controls.append(two_col_header(left_flex=lfx, right_flex=rfx))
        for a, b in pairs:
            grid_list.controls.append(two_col_row(a, b, lfx, rfx))
        info_text.value = f"Page {state['page'] + 1}/{total_pages} • Showing {start + 1}-{end} of {total}"
        prev_btn.disabled = state["page"] <= 0
        next_btn.disabled = state["page"] >= (total_pages - 1)
        page.update()

    def on_prev(_):
        if state["page"] > 0:
            state["page"] -= 1
            render_page()

    def on_next(_):
        if state["page"] < (total_pages - 1):
            state["page"] += 1
            render_page()

    prev_btn.on_click = on_prev
    next_btn.on_click = on_next

    controls_bar = ft.Row(
        [
            prev_btn,
            next_btn,
            info_text,
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
    )

    dlg = ft.AlertDialog(
        modal=True,
        title=ft.Text(f"Merged Dataset Viewer — {total} rows"),
        content=ft.Container(
            width=900,
            height=600,
            content=ft.Column(
                [
                    controls_bar,
                    ft.Container(grid_list, expand=True),
                ],
                expand=True,
            ),
        ),
        actions=[],
    )

    def close_dlg(_):
        dlg.open = False
        page.update()

    dlg.actions = [ft.TextButton("Close", on_click=close_dlg)]
    try:
        dlg.on_dismiss = lambda e: page.update()
    except Exception:
        pass
    render_page()
    opened = False
    try:
        if hasattr(page, "open") and callable(getattr(page, "open")):
            page.open(dlg)
            opened = True
    except Exception:
        opened = False
    if not opened:
        page.dialog = dlg
        dlg.open = True
    await safe_update(page)
