"""Tests for helpers.build.

Focus on split validation and basic push preconditions using mocks.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

import flet as ft

import helpers.build as build


class Page:
    def __init__(self) -> None:
        self.snack_bar = None
        self.open_calls: List[Any] = []

    def open(self, sb):  # type: ignore[override]
        self.open_calls.append(sb)

    def launch_url(self, url: str):  # pragma: no cover
        self.last_url = url


class LV:
    def __init__(self) -> None:
        self.controls: List[Any] = []


class Placeholder:
    def __init__(self) -> None:
        self.visible = True


async def _noop_safe_update(page: Any) -> None:  # noqa: ARG001
    return None


def _common_build_args() -> dict[str, Any]:
    page = Page()
    args = dict(
        page=page,
        source_mode=ft.Dropdown(value="Database"),
        data_source_dd=ft.Dropdown(value="Database"),
        db_session_dd=ft.Dropdown(value="1"),
        data_file=ft.TextField(value=""),
        merged_dir=ft.TextField(value="merged_dataset"),
        seed=ft.TextField(value="123"),
        shuffle=ft.Switch(value=True),
        val_slider=ft.Slider(value=0.0),
        test_slider=ft.Slider(value=0.0),
        min_len_b=ft.TextField(value="1"),
        save_dir=ft.TextField(value="out_dir"),
        push_toggle=ft.Switch(value=False),
        repo_id=ft.TextField(value="username/ds"),
        private=ft.Switch(value=False),
        token_val_ui=ft.TextField(value=""),
        timeline=LV(),
        timeline_placeholder=Placeholder(),
        split_badges={k: ft.Text() for k in ("train", "val", "test")},
        split_meta={k: ("bg", "fg") for k in ("train", "val", "test")},
        dd_ref={},
        cancel_build={"cancelled": False},
        use_custom_card=ft.Switch(value=False),
        card_editor=ft.TextField(value=""),
        hf_cfg_token="",
    )
    return args


def test_run_build_invalid_splits(monkeypatch):
    """val+test >= 1.0 -> validation error and early return."""

    c = _common_build_args()
    c["val_slider"].value = 0.6
    c["test_slider"].value = 0.5

    monkeypatch.setattr(build, "safe_update", _noop_safe_update)

    asyncio.run(
        build.run_build(
            page=c["page"],
            source_mode=c["source_mode"],
            data_source_dd=c["data_source_dd"],
            db_session_dd=c["db_session_dd"],
            data_file=c["data_file"],
            merged_dir=c["merged_dir"],
            seed=c["seed"],
            shuffle=c["shuffle"],
            val_slider=c["val_slider"],
            test_slider=c["test_slider"],
            min_len_b=c["min_len_b"],
            save_dir=c["save_dir"],
            push_toggle=c["push_toggle"],
            repo_id=c["repo_id"],
            private=c["private"],
            token_val_ui=c["token_val_ui"],
            timeline=c["timeline"],
            timeline_placeholder=c["timeline_placeholder"],
            split_badges=c["split_badges"],
            split_meta=c["split_meta"],
            dd_ref=c["dd_ref"],
            cancel_build=c["cancel_build"],
            use_custom_card=c["use_custom_card"],
            card_editor=c["card_editor"],
            hf_cfg_token=c["hf_cfg_token"],
            update_status_placeholder=None,
        )
    )

    assert any(
        "Invalid split: val+test must be < 1.0" in getattr(row.controls[1], "value", "")
        for row in c["timeline"].controls
        if isinstance(row, ft.Row) and len(row.controls) >= 2
    )
    assert c["page"].snack_bar is not None


def test_run_push_async_requires_built_dataset(monkeypatch):
    """run_push_async errors when dd_ref['dd'] is missing."""

    page = Page()
    repo_id = ft.TextField(value="username/ds")
    token_val_ui = ft.TextField(value="token")
    private = ft.Switch(value=False)
    dd_ref: dict[str, Any] = {}
    push_state = {"inflight": False}
    push_ring = ft.ProgressRing(visible=False)
    build_actions = ft.Row(controls=[ft.TextButton("Push + Upload README")])
    timeline = LV()
    placeholder = Placeholder()

    monkeypatch.setattr(build, "safe_update", _noop_safe_update)

    asyncio.run(
        build.run_push_async(
            page=page,
            repo_id=repo_id,
            token_val_ui=token_val_ui,
            private=private,
            dd_ref=dd_ref,
            push_state=push_state,
            push_ring=push_ring,
            build_actions=build_actions,
            timeline=timeline,
            timeline_placeholder=placeholder,
            update_status_placeholder=lambda: None,
            use_custom_card=ft.Switch(value=False),
            card_editor=ft.TextField(value=""),
            hf_cfg_token="",
        )
    )

    assert page.snack_bar is not None
    assert push_state["inflight"] is False
