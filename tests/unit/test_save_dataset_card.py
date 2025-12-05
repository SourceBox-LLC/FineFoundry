import json

import pytest

import save_dataset as sd

pytest.importorskip("datasets")
from datasets import Dataset, DatasetDict  # type: ignore  # noqa: E402


def test_size_category_boundaries():
    assert sd._size_category(0) == "n<1K"
    assert sd._size_category(999) == "n<1K"
    assert sd._size_category(1000) == "1K<n<10K"
    assert sd._size_category(9999) == "1K<n<10K"
    assert sd._size_category(10_000) == "10K<n<100K"
    assert sd._size_category(99_999) == "10K<n<100K"
    assert sd._size_category(100_000) == "100K<n<1M"
    assert sd._size_category(999_999) == "100K<n<1M"
    assert sd._size_category(1_000_000) == "n>1M"


def test_truncate_handles_newlines_and_length():
    txt = "line1\nline2\rmore"
    out = sd._truncate(txt, max_len=50)

    # Newlines become spaces / escaped newlines; no raw CR/LF remain
    assert "\n" not in out
    assert "\r" not in out

    long = "x" * 100
    out2 = sd._truncate(long, max_len=10)
    assert len(out2) == 10
    assert out2.endswith("…")


def test_build_dataset_card_content_includes_counts_and_example():
    # Single-record train set so sampling is deterministic
    train = Dataset.from_list([
        {"input": "hello", "output": "world"},
    ])
    dd = DatasetDict({"train": train})

    md = sd.build_dataset_card_content(dd, "user/ds")

    # Split counts and size bucket
    assert "Train: 1" in md
    assert "Validation: 0" in md
    assert "Test: 0" in md
    assert "Total: 1 (n<1K)" in md

    # Example snippet from the single record
    ex_json = json.dumps({"input": "hello", "output": "world"}, ensure_ascii=False, indent=2)
    assert ex_json in md
