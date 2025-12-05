import json

import pytest
import flet as ft

from helpers import merge


class DummyPage:
    def __init__(self):
        self.snack_bar = None
        self.opened = []

    def update(self):  # safe_update will call this synchronously
        return None

    def open(self, ctl):
        self.snack_bar = ctl
        self.opened.append(ctl)


def _make_json_row(path: str) -> ft.Row:
    src_dd = ft.Dropdown(value="JSON file")
    ds_tf = ft.TextField(value="")
    sp_dd = ft.Dropdown(value="train")
    cfg_tf = ft.TextField(value="")
    in_tf = ft.TextField(value="")
    out_tf = ft.TextField(value="")
    json_tf = ft.TextField(value=path)

    row = ft.Row()
    row.data = {
        "source": src_dd,
        "ds": ds_tf,
        "split": sp_dd,
        "config": cfg_tf,
        "in": in_tf,
        "out": out_tf,
        "json": json_tf,
    }
    return row


@pytest.mark.integration
@pytest.mark.anyio
async def test_run_merge_json_interleave_two_sources(tmp_path):
    # Two small JSON datasets to be merged
    data1 = [
        {"input": "a1", "output": "b1"},
        {"input": "a2", "output": "b2"},
    ]
    data2 = [
        {"input": "c1", "output": "d1"},
    ]
    p1 = tmp_path / "ds1.json"
    p2 = tmp_path / "ds2.json"
    p1.write_text(json.dumps(data1), encoding="utf-8")
    p2.write_text(json.dumps(data2), encoding="utf-8")

    rows_host = ft.Column()
    rows_host.controls = [
        _make_json_row(str(p1)),
        _make_json_row(str(p2)),
    ]

    merge_op = ft.Dropdown(value="Interleave")
    merge_output_format = ft.Dropdown(value="JSON file")
    out_path = tmp_path / "merged.json"
    merge_save_dir = ft.TextField(value=str(out_path))

    merge_timeline = ft.ListView(controls=[])
    merge_timeline_placeholder = ft.Container()
    merge_preview_host = ft.ListView(controls=[])
    merge_preview_placeholder = ft.Container()
    merge_cancel = {}
    merge_busy_ring = ft.ProgressRing()
    download_button = ft.TextButton("Download")
    download_button.visible = False

    page = DummyPage()

    await merge.run_merge(
        page=page,
        rows_host=rows_host,
        merge_op=merge_op,
        merge_output_format=merge_output_format,
        merge_save_dir=merge_save_dir,
        merge_timeline=merge_timeline,
        merge_timeline_placeholder=merge_timeline_placeholder,
        merge_preview_host=merge_preview_host,
        merge_preview_placeholder=merge_preview_placeholder,
        merge_cancel=merge_cancel,
        merge_busy_ring=merge_busy_ring,
        download_button=download_button,
        update_merge_placeholders=lambda: None,
    )

    merged_path = out_path
    data = json.loads(merged_path.read_text(encoding="utf-8"))

    # Expect all records from both sources, interleaved
    assert [r["input"] for r in data] == ["a1", "c1", "a2"]
    assert [r["output"] for r in data] == ["b1", "d1", "b2"]

    # Download button should have been made visible after successful merge
    assert download_button.visible is True
    # And a completion snack bar was opened
    assert page.snack_bar is not None
