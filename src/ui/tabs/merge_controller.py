"""Merge Datasets tab controller for FineFoundry.

This module builds the Merge tab controls and wires up all merge and
preview handlers, keeping `src/main.py` slimmer. Layout composition
still lives in `tab_merge.py` and its per-section builders.
"""

from __future__ import annotations

from typing import Any, Dict, List

import asyncio
import os

import flet as ft

from helpers.common import safe_update
from helpers.logging_config import get_logger
from helpers.theme import BORDER_BASE, COLORS, ICONS, REFRESH_ICON
from helpers.ui import WITH_OPACITY, make_empty_placeholder, build_offline_banner, offline_reason_text
from helpers.merge import (
    run_merge as run_merge_helper,
    preview_merged as preview_merged_helper,
)
from ui.tabs.tab_merge import build_merge_tab


logger = get_logger(__name__)


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
        # As a last resort, run in a background thread
        try:
            loop = asyncio.get_event_loop()
        except Exception:
            return None
        return loop.run_in_executor(None, lambda: asyncio.run(coro()))


def build_merge_tab_with_logic(
    page: ft.Page,
    *,
    section_title,
    _mk_help_handler,
    offline_mode_sw: ft.Switch,
) -> ft.Control:
    """Build the Merge Datasets tab UI and attach all related handlers.

    This mirrors the previous inline Merge tab setup from ``main.py``, but
    keeps the behavior localized to this module.
    """

    # Operation selector
    merge_op = ft.Dropdown(
        label="Operation",
        options=[
            ft.dropdown.Option("Concatenate"),
            ft.dropdown.Option("Interleave"),
        ],
        value="Concatenate",
        width=220,
    )

    # Dynamic dataset rows
    rows_host = ft.Column(spacing=8)

    rows_container = ft.Container(
        content=ft.Column(
            [
                rows_host,
            ],
            spacing=10,
        ),
        width=1000,
        border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
        border_radius=8,
        padding=10,
        visible=False,
    )

    offline_click_state: Dict[str, Any] = {"shown": False}

    def _show_offline_snack_once(msg: str) -> None:
        try:
            if bool(offline_click_state.get("shown")):
                return
            offline_click_state["shown"] = True
        except Exception:
            return
        try:
            page.snack_bar = ft.SnackBar(ft.Text(msg))
            page.open(page.snack_bar)
            page.update()
        except Exception:
            pass

    def update_rows_container_visibility() -> None:
        try:
            rows_container.visible = len(getattr(rows_host, "controls", []) or []) > 0
            page.update()
        except Exception:
            pass

    def make_dataset_row() -> ft.Row:
        source_dd = ft.Dropdown(
            label="Source",
            options=[
                ft.dropdown.Option("Database"),
                ft.dropdown.Option("Hugging Face"),
            ],
            value="Database",
            width=160,
        )

        row_offline_reason = offline_reason_text("Offline Mode: Hugging Face datasets are disabled.")

        def on_source_offline_click(_=None):
            try:
                if not bool(getattr(offline_mode_sw, "value", False)):
                    return
            except Exception:
                return
            _show_offline_snack_once("Offline Mode: Hugging Face datasets are disabled.")

        source_cluster = ft.Container(
            content=ft.Column(
                [
                    source_dd,
                    ft.Container(
                        content=row_offline_reason,
                        on_click=on_source_offline_click,
                    ),
                ],
                spacing=0,
            ),
        )
        # Database session selector
        db_session_dd = ft.Dropdown(
            label="Scrape session",
            options=[],
            width=360,
            visible=True,
        )
        ds_id = ft.TextField(
            label="Dataset repo (e.g., username/dataset)",
            width=360,
            visible=False,
        )
        split = ft.Dropdown(
            label="Split",
            options=[
                ft.dropdown.Option("train"),
                ft.dropdown.Option("validation"),
                ft.dropdown.Option("test"),
                ft.dropdown.Option("all"),
            ],
            value="train",
            width=160,
            visible=False,
        )
        config = ft.TextField(
            label="Config (optional)",
            width=180,
            visible=False,
        )
        in_col = ft.TextField(
            label="Input column (optional)",
            width=200,
            visible=False,
        )
        out_col = ft.TextField(
            label="Output column (optional)",
            width=200,
            visible=False,
        )
        # JSON path kept for internal use only (not user-facing)
        json_path = ft.TextField(
            label="JSON path",
            width=360,
            visible=False,
        )
        remove_btn = ft.IconButton(ICONS.DELETE)
        row = ft.Row(
            [
                source_cluster,
                db_session_dd,
                ds_id,
                split,
                config,
                in_col,
                out_col,
                remove_btn,
            ],
            spacing=10,
            wrap=True,
        )

        # Keep references for later retrieval
        row.data = {
            "source": source_dd,
            "offline_reason": row_offline_reason,
            "db_session": db_session_dd,
            "ds": ds_id,
            "split": split,
            "config": config,
            "in": in_col,
            "out": out_col,
            "json": json_path,
        }

        def refresh_db_sessions():
            try:
                from db.scraped_data import list_scrape_sessions

                sessions = list_scrape_sessions(limit=50)
                options = []
                for s in sessions:
                    # Prefer custom name if set, otherwise fallback to auto-generated label
                    if s.get("name"):
                        label = f"{s['name']} ({s['pair_count']} pairs)"
                    elif s.get("source_details"):
                        label = f"{s['source']}: {s['source_details'][:30]} - {s['pair_count']} pairs"
                    else:
                        label = f"{s['source']} - {s['pair_count']} pairs ({s['created_at'][:10]})"
                    options.append(ft.dropdown.Option(key=str(s["id"]), text=label))
                db_session_dd.options = options
                if options and not db_session_dd.value:
                    db_session_dd.value = options[0].key
            except Exception:
                pass

        def on_source_change(_):
            src = getattr(source_dd, "value", "Database") or "Database"
            is_db = src == "Database"
            is_hf = src == "Hugging Face"
            db_session_dd.visible = is_db
            ds_id.visible = is_hf
            split.visible = is_hf
            config.visible = is_hf
            in_col.visible = is_hf
            out_col.visible = is_hf
            json_path.visible = False  # Always hidden - JSON removed as user option
            if is_db:
                refresh_db_sessions()
            try:
                page.update()
            except Exception:
                pass

        try:
            source_dd.on_change = on_source_change
            # Initialize DB sessions
            refresh_db_sessions()
        except Exception:
            pass

        def remove_row(_):
            try:
                rows_host.controls.remove(row)
                update_rows_container_visibility()
            except Exception:
                pass

        remove_btn.on_click = remove_row
        return row

    def add_row(_=None):
        rows_host.controls.append(make_dataset_row())
        update_rows_container_visibility()
        # If Offline Mode is currently enabled, immediately re-apply the
        # offline gating so newly added rows have Hugging Face greyed out.
        try:
            apply_offline_mode_to_merge()
        except Exception:
            pass

    add_row_btn = ft.TextButton(
        "Add Dataset",
        icon=ICONS.ADD,
        on_click=add_row,
    )
    clear_btn = ft.TextButton(
        "Clear",
        icon=ICONS.BACKSPACE,
        on_click=lambda e: (rows_host.controls.clear(), update_rows_container_visibility()),
    )

    # Output settings - Database is primary, export options for external use
    merge_output_format = ft.Dropdown(
        label="Output format",
        options=[
            ft.dropdown.Option("Database"),
            ft.dropdown.Option("Database + Export JSON"),
        ],
        value="Database",
        width=220,
    )
    merge_session_name = ft.TextField(
        label="Session name",
        value="merged_dataset",
        width=240,
        hint_text="Name for the merged session in database",
    )
    # Hidden export path for JSON export option
    merge_export_path = ft.TextField(
        label="Export path",
        value="merged.json",
        width=240,
        visible=False,
    )

    def update_output_controls(_=None):
        fmt = (merge_output_format.value or "").lower()
        wants_export = "export" in fmt
        merge_export_path.visible = wants_export
        try:
            page.update()
        except Exception:
            pass

    try:
        merge_output_format.on_change = update_output_controls
    except Exception:
        pass

    # Status & preview
    merge_timeline = ft.ListView(expand=1, auto_scroll=True, spacing=6)
    merge_timeline_placeholder = make_empty_placeholder(
        "No status yet",
        ICONS.TASK,
    )
    merge_preview_host = ft.ListView(expand=1, auto_scroll=False)
    merge_preview_placeholder = make_empty_placeholder(
        "Preview not available",
        ICONS.PREVIEW,
    )
    merge_status_section_ref: Dict[str, Any] = {}
    merge_preview_section_ref: Dict[str, Any] = {}

    merge_cancel: Dict[str, Any] = {"cancelled": False}
    merge_busy_ring = ft.ProgressRing(width=18, height=18, value=None, visible=False)

    def update_merge_placeholders() -> None:
        try:
            has_status = len(getattr(merge_timeline, "controls", []) or []) > 0
            has_preview = len(getattr(merge_preview_host, "controls", []) or []) > 0
            # Stack-level placeholders
            merge_timeline_placeholder.visible = not has_status
            merge_preview_placeholder.visible = not has_preview
            # Section-level visibility via refs from tab_merge
            try:
                ctl = merge_status_section_ref.get("control")
                if ctl is not None:
                    ctl.visible = has_status
            except Exception:
                pass
            try:
                ctl = merge_preview_section_ref.get("control")
                if ctl is not None:
                    ctl.visible = has_preview
            except Exception:
                pass
        except Exception:
            pass
        page.update()

    async def on_merge():
        """Delegate merge operation to helpers.merge.run_merge.

        When Offline Mode is enabled, prevent merging from Hugging Face
        sources to avoid remote downloads.
        """

        try:
            is_offline = bool(getattr(offline_mode_sw, "value", False))
        except Exception:
            is_offline = False
        if is_offline:
            # If any row is currently set to Hugging Face, block the action.
            try:
                for row in rows_host.controls or []:
                    try:
                        sd = (getattr(row, "data", {}) or {}).get("source")
                        if sd is not None and (sd.value or "Database") == "Hugging Face":
                            page.snack_bar = ft.SnackBar(
                                ft.Text("Offline mode is enabled; merging Hugging Face datasets is disabled."),
                            )
                            page.open(page.snack_bar)
                            await safe_update(page)
                            return
                    except Exception:
                        continue
            except Exception:
                pass

        return await run_merge_helper(
            page=page,
            rows_host=rows_host,
            merge_op=merge_op,
            merge_output_format=merge_output_format,
            merge_session_name=merge_session_name,
            merge_export_path=merge_export_path,
            merge_timeline=merge_timeline,
            merge_timeline_placeholder=merge_timeline_placeholder,
            merge_preview_host=merge_preview_host,
            merge_preview_placeholder=merge_preview_placeholder,
            merge_cancel=merge_cancel,
            merge_busy_ring=merge_busy_ring,
            download_button=download_merged_button,
            update_merge_placeholders=update_merge_placeholders,
        )

    def on_cancel_merge(_):
        merge_cancel["cancelled"] = True
        try:
            merge_timeline.controls.append(
                ft.Row(
                    [
                        ft.Icon(ICONS.CANCEL, color=COLORS.RED),
                        ft.Text("Cancel requested — will stop ASAP"),
                    ]
                )
            )
            update_merge_placeholders()
            page.update()
        except Exception:
            pass

    def on_refresh_merge(_):
        merge_cancel["cancelled"] = False
        merge_timeline.controls.clear()
        merge_preview_host.controls.clear()
        merge_busy_ring.visible = False
        download_merged_button.visible = False
        update_merge_placeholders()
        page.update()

    async def on_preview_merged():
        """Delegate merged dataset preview to helpers.merge.preview_merged.

        When Offline Mode is enabled, prevent previewing Hugging Face-backed
        merges that would require remote dataset fetches.
        """

        try:
            is_offline = bool(getattr(offline_mode_sw, "value", False))
        except Exception:
            is_offline = False
        if is_offline:
            try:
                for row in rows_host.controls or []:
                    try:
                        sd = (getattr(row, "data", {}) or {}).get("source")
                        if sd is not None and (sd.value or "Database") == "Hugging Face":
                            page.snack_bar = ft.SnackBar(
                                ft.Text("Offline mode is enabled; previewing Hugging Face datasets is disabled."),
                            )
                            page.open(page.snack_bar)
                            await safe_update(page)
                            return
                    except Exception:
                        continue
            except Exception:
                pass

        return await preview_merged_helper(
            page=page,
            merge_output_format=merge_output_format,
            merge_session_name=merge_session_name,
        )

    def handle_merge_preview_click(_):
        try:
            page.snack_bar = ft.SnackBar(
                ft.Text("Opening merged dataset preview..."),
            )
            page.open(page.snack_bar)
            try:
                page.update()
            except Exception:
                pass
        except Exception:
            pass
        _schedule_task(page, on_preview_merged)

    async def on_download_merged(e: ft.FilePickerResultEvent):
        """Handle downloading the merged dataset to a user-selected location."""

        logger.info("Download merged dataset called with destination: %s", e.path)

        if e.path is None:
            logger.warning("Download cancelled: No destination selected")
            page.snack_bar = ft.SnackBar(ft.Text("No destination selected"))
            page.snack_bar.open = True
            await safe_update(page)
            return

        dest_dir = e.path
        session_name = merge_session_name.value or "merged_dataset"
        export_path = merge_export_path.value or "merged.json"
        fmt_now = (merge_output_format.value or "").lower()
        wants_json = "export" in fmt_now

        logger.debug(
            "Download params - dest_dir: %s, session_name: %s, export_path: %s, wants_json: %s",
            dest_dir,
            session_name,
            export_path,
            wants_json,
        )

        # For JSON export, find the exported file
        if wants_json and export_path:
            candidates: List[str] = []
            if os.path.isabs(export_path):
                candidates.append(export_path)
            else:
                candidates.extend(
                    [
                        export_path,
                        os.path.abspath(export_path),
                        os.path.join(os.getcwd(), export_path),
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

            logger.debug("Source search candidates: %s", candidates)
            logger.debug("Found existing source: %s", existing)

            if not existing:
                logger.error("Exported JSON not found. Searched: %s", export_path)
                page.snack_bar = ft.SnackBar(
                    ft.Text(f"Exported JSON not found. Searched: {export_path}"),
                )
                page.open(page.snack_bar)
                await safe_update(page)
                return

            source_basename = os.path.basename(export_path)
            dest_path = os.path.join(dest_dir, source_basename)
        else:
            # No JSON export - data is in database only
            logger.info("No JSON export available - data is in database session: %s", session_name)
            page.snack_bar = ft.SnackBar(
                ft.Text(f"Data is in database session '{session_name}'. Enable 'Export JSON' to download."),
            )
            page.open(page.snack_bar)
            await safe_update(page)
            return

        logger.info("Copying %s from %s to %s", source_basename, existing, dest_path)

        try:
            import shutil

            if wants_json or os.path.isfile(existing):
                # Copy single file
                logger.debug("Starting file copy operation")
                await asyncio.to_thread(shutil.copy2, existing, dest_path)
                msg = f"Downloaded to {dest_path}"
                logger.info("File copy successful: %s", dest_path)
            else:
                # Copy directory
                logger.debug("Starting directory copy operation")
                await asyncio.to_thread(
                    shutil.copytree,
                    existing,
                    dest_path,
                    dirs_exist_ok=True,
                )
                msg = f"Downloaded to {dest_path}"
                logger.info("Directory copy successful: %s", dest_path)

            page.snack_bar = ft.SnackBar(ft.Text(msg))
            page.open(page.snack_bar)
            await safe_update(page)
        except Exception as ex:  # pragma: no cover - runtime error path
            logger.error(
                "Download failed - Source: %s, Dest: %s",
                existing,
                dest_path,
                exc_info=True,
            )
            error_details = f"Download failed: {str(ex)}"
            page.snack_bar = ft.SnackBar(ft.Text(error_details))
            page.open(page.snack_bar)
            await safe_update(page)

    async def handle_download_result(e: ft.FilePickerResultEvent):
        await on_download_merged(e)

    download_file_picker = ft.FilePicker(on_result=handle_download_result)
    try:
        page.overlay.append(download_file_picker)
    except Exception:
        pass

    download_merged_button = ft.ElevatedButton(
        "Download Merged Dataset",
        icon=getattr(ICONS, "DOWNLOAD", ICONS.ARROW_DOWNWARD),
        on_click=lambda _: download_file_picker.get_directory_path(
            dialog_title="Select Download Location",
        ),
        visible=False,
    )

    merge_actions = ft.Row(
        [
            ft.ElevatedButton(
                "Merge Datasets",
                icon=ICONS.TABLE_VIEW,
                on_click=lambda e: _schedule_task(page, on_merge),
            ),
            ft.OutlinedButton(
                "Cancel",
                icon=ICONS.CANCEL,
                on_click=on_cancel_merge,
            ),
            ft.TextButton(
                "Refresh",
                icon=REFRESH_ICON,
                on_click=on_refresh_merge,
            ),
            ft.TextButton(
                "Preview Merged",
                icon=ICONS.PREVIEW,
                on_click=handle_merge_preview_click,
            ),
            merge_busy_ring,
        ],
        spacing=10,
    )

    offline_banner = build_offline_banner(
        [
            "Hugging Face datasets are disabled.",
            "Only Database sources can be selected.",
        ]
    )

    merge_tab = build_merge_tab(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        offline_banner=offline_banner,
        merge_op=merge_op,
        rows_container=rows_container,
        add_row_btn=add_row_btn,
        clear_btn=clear_btn,
        merge_output_format=merge_output_format,
        merge_session_name=merge_session_name,
        merge_export_path=merge_export_path,
        merge_actions=merge_actions,
        merge_preview_host=merge_preview_host,
        merge_preview_placeholder=merge_preview_placeholder,
        merge_timeline=merge_timeline,
        merge_timeline_placeholder=merge_timeline_placeholder,
        download_button=download_merged_button,
        preview_section_ref=merge_preview_section_ref,
        status_section_ref=merge_status_section_ref,
    )
    update_merge_placeholders()

    # Hook: respond to Offline Mode changes (disable Hugging Face sources when offline)
    def apply_offline_mode_to_merge(_=None):
        try:
            is_offline = bool(getattr(offline_mode_sw, "value", False))
        except Exception:
            is_offline = False

        try:
            offline_banner.visible = is_offline
        except Exception:
            pass

        try:
            for row in rows_host.controls or []:
                data = getattr(row, "data", {}) or {}
                src_dd: ft.Dropdown = data.get("source")  # type: ignore[assignment]
                if src_dd is None:
                    continue

                try:
                    rr = data.get("offline_reason")
                    if rr is not None:
                        rr.visible = is_offline
                except Exception:
                    pass

                opts = list(getattr(src_dd, "options", []) or [])
                for opt in opts:
                    try:
                        label = str(getattr(opt, "key", getattr(opt, "text", "")) or "").strip()
                    except Exception:
                        label = ""
                    try:
                        if label == "Hugging Face":
                            opt.disabled = is_offline
                        else:
                            opt.disabled = False
                    except Exception:
                        pass
                try:
                    src_dd.options = opts
                except Exception:
                    pass

                # If currently pointing at Hugging Face while offline, reset to Database
                try:
                    if is_offline and (src_dd.value or "Database") == "Hugging Face":
                        src_dd.value = "Database"
                        # Re-apply visibility rules for controls
                        is_db = True
                        is_hf = False
                        db_dd = data.get("db_session")
                        ds_id = data.get("ds")
                        split = data.get("split")
                        config = data.get("config")
                        in_col = data.get("in")
                        out_col = data.get("out")
                        json_path = data.get("json")
                        if db_dd is not None:
                            db_dd.visible = is_db
                        if ds_id is not None:
                            ds_id.visible = is_hf
                        if split is not None:
                            split.visible = is_hf
                        if config is not None:
                            config.visible = is_hf
                        if in_col is not None:
                            in_col.visible = is_hf
                        if out_col is not None:
                            out_col.visible = is_hf
                        if json_path is not None:
                            json_path.visible = False
                except Exception:
                    pass
        except Exception:
            pass

        try:
            page.update()
        except Exception:
            pass

    try:
        hooks = getattr(offline_mode_sw, "data", None)
        if hooks is None:
            hooks = {}
        hooks["merge_tab_offline"] = lambda e=None: apply_offline_mode_to_merge(e)
        offline_mode_sw.data = hooks
    except Exception:
        pass

    # Apply current offline state on first load
    apply_offline_mode_to_merge()

    return merge_tab
