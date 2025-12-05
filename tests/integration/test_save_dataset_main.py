import json

import pytest

import save_dataset as sd


@pytest.mark.integration
def test_save_dataset_main_end_to_end(tmp_path, monkeypatch):
    datasets = pytest.importorskip("datasets")

    # Prepare a tiny mixed-format dataset: one standard pair and one ChatML conversation
    records = [
        {"input": "q1", "output": "a1"},
        {
            "messages": [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ]
        },
    ]
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(records), encoding="utf-8")

    out_dir = tmp_path / "hf_dataset"

    # Patch module-level config so main() uses our temp paths and does not push to Hub
    monkeypatch.setattr(sd, "DATA_FILE", str(data_path))
    monkeypatch.setattr(sd, "SAVE_DIR", str(out_dir))
    monkeypatch.setattr(sd, "PUSH_TO_HUB", False)
    monkeypatch.setattr(sd, "VAL_SIZE", 0.0)
    monkeypatch.setattr(sd, "TEST_SIZE", 0.0)
    monkeypatch.setattr(sd, "SHUFFLE", False)
    monkeypatch.setattr(sd, "SEED", 123)

    # Run the full pipeline: load -> normalize -> split -> save_to_disk
    sd.main()

    assert out_dir.exists()

    # Load back with datasets to verify structure
    dd = datasets.load_from_disk(str(out_dir))
    assert "train" in dd
    train = dd["train"]
    assert len(train) == 2

    first = train[0]
    assert "input" in first and "output" in first
    assert isinstance(first["input"], str)
    assert isinstance(first["output"], str)
