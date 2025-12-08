
import pytest

import save_dataset as sd


def test_normalize_records_standard_pairs_basic():
    records = [
        {"input": " hello ", "output": "world"},
        {"input": "foo", "output": " "},  # output too short when stripped
        "not-a-dict",  # ignored
    ]

    out = sd.normalize_records(records, min_len=1)

    assert out == [{"input": "hello", "output": "world"}]


def test_normalize_records_chatml_user_assistant_pair():
    records = [
        {
            "messages": [
                {"role": "user", "content": " question "},
                {"role": "assistant", "content": " answer "},
            ]
        }
    ]

    out = sd.normalize_records(records, min_len=1)

    assert out == [{"input": "question", "output": "answer"}]


@pytest.mark.parametrize("val_size,test_size", [(0.2, 0.1), (0.0, 0.0)])
def test_build_dataset_dict_splits_cover_all_examples(val_size: float, test_size: float):
    examples = [{"input": f"in-{i}", "output": f"out-{i}"} for i in range(10)]

    # Use deterministic seed and no shuffle by default so counts are stable
    dd = sd.build_dataset_dict(
        examples=examples,
        val_size=val_size,
        test_size=test_size,
        shuffle=False,
        seed=123,
    )

    # All splits together should cover all examples exactly once
    total = sum(len(split_ds) for split_ds in dd.values())
    assert total == len(examples)

    if val_size == 0.0 and test_size == 0.0:
        # Only train split when no val/test requested
        assert set(dd.keys()) == {"train"}
    else:
        # When both requested, expect three splits present
        assert {"train", "validation", "test"}.issuperset(dd.keys())
        assert "train" in dd
