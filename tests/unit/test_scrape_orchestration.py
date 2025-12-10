"""Tests for helpers.scrape orchestration logic.

These tests focus on wiring and control flow, using lightweight dummy
objects and monkeypatching to avoid real network, database, or UI work.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List

import helpers.scrape as scrape


@dataclass
class DummyLabel:
    value: str = ""


class DummyPage:
    """Minimal stand‑in for ft.Page used in scrape helpers."""

    def __init__(self) -> None:
        self.snack_bar = None
        self.open_calls: List[Any] = []

    def open(self, snack_bar):  # type: ignore[override]
        self.open_calls.append(snack_bar)


class DummyListView:
    def __init__(self) -> None:
        self.controls: List[Any] = []


class DummyProgressBar:
    def __init__(self) -> None:
        self.value: float | None = None


async def _noop_safe_update(page: Any) -> None:  # noqa: ARG001
    """Async no‑op replacement for helpers.common.safe_update."""
    return None


def _make_basic_ui_state():
    page = DummyPage()
    log_view = DummyListView()
    prog = DummyProgressBar()
    labels = {"pairs": DummyLabel(), "threads": DummyLabel()}
    preview_host = DummyListView()
    cancel_flag = {"cancelled": False}
    return page, log_view, prog, labels, preview_host, cancel_flag


def test_run_real_scrape_standard_pairs_saves_to_db_and_builds_preview(monkeypatch):
    """run_real_scrape should call 4chan scraper and save pairs to DB in standard mode.

    This test exercises the non‑ChatML path (multiturn=False, dataset_format="Standard").
    """

    page, log_view, prog, labels, preview_host, cancel_flag = _make_basic_ui_state()

    # Patch safe_update to a no‑op and compute/two‑col helpers to simple stand‑ins
    monkeypatch.setattr(scrape, "safe_update", _noop_safe_update)

    def fake_compute_two_col_flex(pairs):  # noqa: ARG001
        return 1, 1

    def fake_two_col_header(left, right, left_flex=1, right_flex=1):  # noqa: ARG001
        return ("HEADER", left, right)

    def fake_two_col_row(a, b, left_flex=1, right_flex=1):  # noqa: ARG001
        return ("ROW", a, b)

    monkeypatch.setattr(scrape, "compute_two_col_flex", fake_compute_two_col_flex)
    monkeypatch.setattr(scrape, "two_col_header", fake_two_col_header)
    monkeypatch.setattr(scrape, "two_col_row", fake_two_col_row)

    # Avoid touching real proxy or scrapers / DB
    monkeypatch.setattr(scrape, "apply_proxy_from_ui", lambda *a, **k: "proxy ok")

    scraped_pairs = [
        {"input": "q1", "output": "a1"},
        {"input": "q2", "output": "a2"},
    ]

    def fake_sc_scrape(**kwargs):  # type: ignore[override]
        # Sanity‑check some arguments from wiring
        assert kwargs["board"] == "pol"
        assert kwargs["max_pairs"] == 10
        return scraped_pairs

    monkeypatch.setattr(scrape, "sc", type("SC", (), {"scrape": staticmethod(fake_sc_scrape)})())

    saved_calls: list[dict] = []

    def fake_save_scrape_to_db(**kwargs):  # type: ignore[override]
        saved_calls.append(kwargs)

    monkeypatch.setattr(scrape, "save_scrape_to_db", fake_save_scrape_to_db)
    monkeypatch.setattr(scrape, "save_chatml_to_db", lambda **_: None)

    # Run
    asyncio.run(
        scrape.run_real_scrape(
            page=page,
            log_view=log_view,
            prog=prog,
            labels=labels,
            preview_host=preview_host,
            cancel_flag=cancel_flag,
            boards=["pol"],
            max_threads=5,
            max_pairs_total=10,
            delay=0.0,
            min_len_val=1,
            multiturn=False,  # chatml_enabled False
            ctx_strategy="cumulative",
            ctx_k=4,
            ctx_max_chars=None,
            merge_same_id=False,
            require_question=False,
            ui_proxy_enabled=False,
            ui_proxy_url=None,
            ui_use_env_proxies=False,
            dataset_format="Standard",
        )
    )

    # DB save was called once with expected metadata
    assert len(saved_calls) == 1
    call = saved_calls[0]
    assert call["source"] == "4chan"
    assert call["dataset_format"] == "Standard"
    assert call["pairs"] == scraped_pairs
    assert "boards=pol" in call["source_details"]

    # Labels updated to reflect pair count
    assert labels["pairs"].value == "Pairs Found: 2"

    # Preview host should contain a header and one row per pair
    # using our fake two_col_header/row helpers.
    assert preview_host.controls[0][0] == "HEADER"
    rows = [c for c in preview_host.controls[1:] if c[0] == "ROW"]
    assert len(rows) == 2


def test_run_reddit_scrape_handles_missing_module(monkeypatch):
    """If reddit scraper module is None, run_reddit_scrape should show a snackbar and return."""

    page, log_view, prog, labels, preview_host, cancel_flag = _make_basic_ui_state()

    # Patch safe_update to a no‑op
    monkeypatch.setattr(scrape, "safe_update", _noop_safe_update)

    # Force rdt to None for this test
    monkeypatch.setattr(scrape, "rdt", None)

    # Call function; it should early‑return without raising
    asyncio.run(
        scrape.run_reddit_scrape(
            page=page,
            log_view=log_view,
            prog=prog,
            labels=labels,
            preview_host=preview_host,
            cancel_flag=cancel_flag,
            url="https://www.reddit.com/r/test/",
            max_posts=10,
            delay=0.1,
            min_len_val=1,
            multiturn=False,
            ctx_k=2,
            ctx_max_chars=None,
            merge_same_id=False,
            require_question=False,
            ui_proxy_enabled=False,
            ui_proxy_url=None,
            ui_use_env_proxies=False,
            dataset_format="Standard",
        )
    )

    # Should have log entry about module not available and a snackbar set
    assert any("Reddit scraper module not available" in str(c.value) for c in log_view.controls)
    assert page.snack_bar is not None
