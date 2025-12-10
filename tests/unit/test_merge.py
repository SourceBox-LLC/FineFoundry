"""Tests for helpers.merge run_merge orchestration.

These tests focus on control flow and wiring using lightweight dummy
objects and monkeypatching. No real network, HF datasets, or DB I/O
are performed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List

import flet as ft

import helpers.merge as merge


# ---------------------------------------------------------------------------
# Dummy UI helpers
# ---------------------------------------------------------------------------


@dataclass
class DummyLabel:
    value: str = ""


class DummyPage:
    def __init__(self) -> None:
        self.snack_bar = None
        self.open_calls: List[Any] = []

    def open(self, snack_bar):  # type: ignore[override]
        self.open_calls.append(snack_bar)


class DummyListView:
    def __init__(self) -> None:
        self.controls: List[Any] = []


class DummyBusyRing:
    def __init__(self) -> None:
        self.visible: bool = False


class DummyPlaceholder:
    def __init__(self) -> None:
        self.visible: bool = True


async def _noop_safe_update(page: Any) -> None:  # noqa: ARG001
    """Async no-op replacement for helpers.common.safe_update."""
    return None


def _make_db_row(session_id: int) -> ft.Row:
    """Create a Row whose .data mimics a Database merge row."""

    @dataclass
    class Ctl:
        value: str

    data = {
        "source": Ctl("Database"),
        "db_session": Ctl(str(session_id)),
        "ds": Ctl(""),
        "split": Ctl("train"),
        "config": Ctl(""),
        "in": Ctl(""),
        "out": Ctl(""),
    }
    return ft.Row([], data=data)


def _make_merge_controls(single_row: bool = False):
    page = DummyPage()
    rows_host = ft.Column()
    rows_host.controls.append(_make_db_row(1))
    if not single_row:
        rows_host.controls.append(_make_db_row(2))

    merge_op = type("DD", (), {"value": "Concatenate"})()
    merge_output_format = type("DD", (), {"value": "Database"})()
    merge_session_name = type("TF", (), {"value": "merged_session"})()
    merge_export_path = type("TF", (), {"value": ""})()

    merge_timeline = DummyListView()
    merge_timeline_placeholder = DummyPlaceholder()
    merge_preview_host = DummyListView()
    merge_preview_placeholder = DummyPlaceholder()
    merge_cancel = {"cancelled": False}
    merge_busy_ring = DummyBusyRing()
    download_button = type("BTN", (), {"visible": False})()

    return (
        page,
        rows_host,
        merge_op,
        merge_output_format,
        merge_session_name,
        merge_export_path,
        merge_timeline,
        merge_timeline_placeholder,
        merge_preview_host,
        merge_preview_placeholder,
        merge_cancel,
        merge_busy_ring,
        download_button,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_merge_requires_at_least_two_entries(monkeypatch):
    """Validation: with <2 entries, run_merge logs error and returns early."""

    (
        page,
        rows_host,
        merge_op,
        merge_output_format,
        merge_session_name,
        merge_export_path,
        merge_timeline,
        merge_timeline_placeholder,
        merge_preview_host,
        merge_preview_placeholder,
        merge_cancel,
        merge_busy_ring,
        download_button,
    ) = _make_merge_controls(single_row=True)

    # Patch helper functions to no-op
    monkeypatch.setattr(merge, "safe_update", _noop_safe_update)

    # Run
    asyncio.run(
        merge.run_merge(
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
            download_button=download_button,
            update_merge_placeholders=lambda: None,
        )
    )

    # Should have an error row about needing at least two datasets
    assert any(
        "Add at least two datasets" in getattr(row.controls[1], "value", "")
        for row in merge_timeline.controls
        if isinstance(row, ft.Row) and len(getattr(row, "controls", [])) >= 2
    )
    # Busy ring should be hidden after validation failure
    assert merge_busy_ring.visible is False


def test_run_merge_concatenate_two_db_sessions(monkeypatch):
    """Two Database rows should be concatenated and saved to a new merged session."""

    (
        page,
        rows_host,
        merge_op,
        merge_output_format,
        merge_session_name,
        merge_export_path,
        merge_timeline,
        merge_timeline_placeholder,
        merge_preview_host,
        merge_preview_placeholder,
        merge_cancel,
        merge_busy_ring,
        download_button,
    ) = _make_merge_controls(single_row=False)

    # Patch helper functions to deterministic, side-effect-free versions
    monkeypatch.setattr(merge, "safe_update", _noop_safe_update)

    def fake_compute_two_col_flex(pairs):  # noqa: ARG001
        return 1, 1

    def fake_two_col_header(*args, **kwargs):  # noqa: ARG001
        return ("HEADER", args, kwargs)

    def fake_two_col_row(a, b, *args, **kwargs):  # noqa: ARG001
        return ("ROW", a, b)

    monkeypatch.setattr(merge, "compute_two_col_flex", fake_compute_two_col_flex)
    monkeypatch.setattr(merge, "two_col_header", fake_two_col_header)
    monkeypatch.setattr(merge, "two_col_row", fake_two_col_row)

    # Patch DB accessors used inside run_merge
    import db.scraped_data as sdb

    session_pairs = {
        1: [{"input": "q1", "output": "a1"}],
        2: [{"input": "q2", "output": "a2"}, {"input": "q3", "output": "a3"}],
    }

    def fake_get_scrape_session(session_id: int):  # type: ignore[override]
        return {"id": session_id}

    def fake_get_pairs_for_session(session_id: int):  # type: ignore[override]
        return session_pairs[int(session_id)]

    created_sessions: List[dict] = []

    def fake_create_scrape_session(**kwargs):  # type: ignore[override]
        created_sessions.append(kwargs)
        return {"id": 999}

    added_pairs: List[tuple[int, list[dict]]] = []

    def fake_add_scraped_pairs(session_id: int, pairs: list[dict]):  # type: ignore[override]
        added_pairs.append((session_id, pairs))

    monkeypatch.setattr(sdb, "get_scrape_session", fake_get_scrape_session)
    monkeypatch.setattr(sdb, "get_pairs_for_session", fake_get_pairs_for_session)
    monkeypatch.setattr(sdb, "create_scrape_session", fake_create_scrape_session)
    monkeypatch.setattr(sdb, "add_scraped_pairs", fake_add_scraped_pairs)

    # Run merge
    asyncio.run(
        merge.run_merge(
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
            download_button=download_button,
            update_merge_placeholders=lambda: None,
        )
    )

    # DB session should be created once with expected metadata
    assert len(created_sessions) == 1
    meta = created_sessions[0]["metadata"]
    assert created_sessions[0]["source"] == "Merged"
    assert created_sessions[0]["source_details"] == "merged_session"
    assert meta["operation"] == "Concatenate"
    assert meta["source_count"] == 2

    # All pairs from both sessions should be added to the merged session
    assert added_pairs == [
        (
            999,
            [
                {"input": "q1", "output": "a1"},
                {"input": "q2", "output": "a2"},
                {"input": "q3", "output": "a3"},
            ],
        )
    ]

    # Preview host should contain a header and at least one row tuple from fake_two_col_row
    assert any(isinstance(c, tuple) and c[0] == "HEADER" for c in merge_preview_host.controls)
    assert any(isinstance(c, tuple) and c[0] == "ROW" for c in merge_preview_host.controls)

    # Download button is shown after successful merge
    assert download_button.visible is True
