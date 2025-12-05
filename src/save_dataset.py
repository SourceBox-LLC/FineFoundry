#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =========================
# Configuration (edit me)
# =========================
DATA_FILE = "scraped_training_data.json"  # path to JSON list of {"input","output"}
SAVE_DIR = "hf_dataset"                   # output directory for save_to_disk

SEED = 42
SHUFFLE = True

VAL_SIZE = 0.01                           # fraction for validation split
TEST_SIZE = 0.0                           # fraction for test split
MIN_LEN = 1                               # min length for input/output after strip

# Hugging Face Hub settings
PUSH_TO_HUB = True                       # set True to push to Hub
REPO_ID = "sbussiso/4chan-pairs"                              # e.g. "username/my-4chan-pairs"
PRIVATE = True                            # create/keep private repo if pushing
HF_TOKEN = None                         # if None, will use env HF_TOKEN or cached login
# =========================


import json
import os
import io
import random
import textwrap
from typing import List, Dict, Any, Optional

from datasets import Dataset, DatasetDict  # pip install datasets
from huggingface_hub import HfApi, HfFolder  # pip install huggingface_hub


def load_records(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected JSON root to be a list of records.")
    return data


def normalize_records(
    records: List[Dict[str, Any]],
    min_len: int = 1,
) -> List[Dict[str, str]]:
    """Normalize a heterogeneous list of records into simple input/output pairs.

    Supported formats:
    - Standard: {"input": str, "output": str}
    - ChatML: {"messages": [{"role": "user"|"assistant", "content": str}, ...]}
      We emit one pair per user→assistant exchange, falling back to first two
      non-empty messages if roles are missing or misordered.
    """
    out: List[Dict[str, str]] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        # 1) Standard pairs
        if "input" in r or "output" in r:
            inp = r.get("input", "")
            outp = r.get("output", "")
            if inp is None:
                inp = ""
            if outp is None:
                outp = ""
            a = str(inp).strip()
            b = str(outp).strip()
            if len(a) >= min_len and len(b) >= min_len:
                out.append({"input": a, "output": b})
            continue

        # 2) ChatML-style conversations
        msgs = r.get("messages")
        if isinstance(msgs, list) and msgs:
            # Try to extract sequential pairs of user -> assistant
            pending_user: Optional[str] = None
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = (m.get("role") or "").strip().lower()
                text = str(m.get("content") or "").strip()
                if not text:
                    continue
                if role == "user":
                    # Start/replace pending user turn
                    pending_user = text
                elif role == "assistant" and pending_user is not None:
                    if len(pending_user) >= min_len and len(text) >= min_len:
                        out.append({"input": pending_user, "output": text})
                    pending_user = None
            # Fallback: if no pairs were produced, try first two non-empty messages
            if all(k not in r for k in ("input", "output")):
                if not any(True for _ in out):
                    texts = [str((m.get("content") if isinstance(m, dict) else "") or "").strip() for m in msgs]
                    texts = [t for t in texts if t]
                    if len(texts) >= 2:
                        a, b = texts[0], texts[1]
                        if len(a) >= min_len and len(b) >= min_len:
                            out.append({"input": a, "output": b})
            continue

        # 3) Unknown format -> skip
        continue

    return out


def build_dataset_dict(
    examples: List[Dict[str, str]],
    val_size: float,
    test_size: float,
    shuffle: bool,
    seed: int,
) -> DatasetDict:
    if not 0.0 <= val_size < 1.0:
        raise ValueError("VAL_SIZE must be in [0,1)")
    if not 0.0 <= test_size < 1.0:
        raise ValueError("TEST_SIZE must be in [0,1)")
    if val_size + test_size >= 1.0:
        raise ValueError("VAL_SIZE + TEST_SIZE must be < 1.0")

    ds = Dataset.from_list(examples)
    if shuffle:
        ds = ds.shuffle(seed=seed)

    # No splits requested
    if val_size == 0.0 and test_size == 0.0:
        return DatasetDict({"train": ds})

    # First split off test
    if test_size > 0.0:
        split_1 = ds.train_test_split(test_size=test_size, seed=seed)
        ds_train = split_1["train"]
        ds_test = split_1["test"]
    else:
        ds_train = ds
        ds_test = None

    # Then split off validation from the remaining train
    if val_size > 0.0:
        denom = 1.0 - test_size
        adjusted_val = val_size / denom if denom > 0 else 0.0
        split_2 = ds_train.train_test_split(test_size=adjusted_val, seed=seed)
        ds_train_final = split_2["train"]
        ds_val = split_2["test"]
    else:
        ds_train_final = ds_train
        ds_val = None

    dd = {"train": ds_train_final}
    if ds_val is not None:
        dd["validation"] = ds_val
    if ds_test is not None:
        dd["test"] = ds_test
    return DatasetDict(dd)


def ensure_repo(repo_id: str, private: bool, token: str) -> None:
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )


def push_to_hub(dd: DatasetDict, repo_id: str, private: bool, token: str) -> None:
    ensure_repo(repo_id, private=private, token=token)
    dd.push_to_hub(repo_id=repo_id, token=token)


def _truncate(txt: str, max_len: int = 280) -> str:
    txt = txt.replace("\r", " ").replace("\n", "\\n")
    return txt if len(txt) <= max_len else txt[: max_len - 1] + "…"


def _size_category(total: int) -> str:
    # Align with HF size_categories buckets
    if total < 1_000:
        return "n<1K"
    if total < 10_000:
        return "1K<n<10K"
    if total < 100_000:
        return "10K<n<100K"
    if total < 1_000_000:
        return "100K<n<1M"
    return "n>1M"


def build_dataset_card_content(dd: DatasetDict, repo_id: str) -> str:
    train_n = len(dd.get("train", []))
    val_n = len(dd.get("validation", [])) if "validation" in dd else 0
    test_n = len(dd.get("test", [])) if "test" in dd else 0
    total_n = train_n + val_n + test_n
    size_cat = _size_category(total_n)

    # Sample up to 3 examples from train for preview
    examples = []
    if "train" in dd and train_n > 0:
        k = min(3, train_n)
        idxs = random.sample(range(train_n), k)
        batch = dd["train"].select(idxs)
        for rec in batch:
            examples.append(
                {
                    "input": _truncate(str(rec.get("input", ""))),
                    "output": _truncate(str(rec.get("output", ""))),
                }
            )

    yaml_header = f"""---
tags:
  - 4chan
  - scraped
  - conversation
  - nsfw
  - toxic
task_categories:
  - text-generation
language:
  - en
license: other
size_categories:
  - {size_cat}
pretty_name: 4chan Conversational Pairs
---
"""

    # Build examples section
    ex_lines: List[str] = []
    for i, ex in enumerate(examples, 1):
        ex_lines.append(
            "\n".join(
                [
                    f"Example {i}",
                    "",
                    "```json",
                    json.dumps(ex, ensure_ascii=False, indent=2),
                    "```",
                ]
            )
        )
    examples_md = "\n\n".join(ex_lines) if ex_lines else "No preview examples shown."

    body = f"""
# Dataset Card: {repo_id}

## Dataset Summary
Pairs of short-form conversational text scraped from public 4chan threads and transformed into input/output examples suitable for text-to-text training and dialog-style fine-tuning. This dataset may contain NSFW content and toxic language. It is provided strictly for research purposes and requires careful filtering and alignment before use in production.

## Data Fields
- `input`: Source utterance.
- `output`: Target response.

## Source and Collection
- Source: Public threads on 4chan across a selection of boards.
- Collection Method: Programmatic scraping of thread catalogs and posts, pairing adjacent messages within threads and cleaning markup/URLs.

## Splits
- Train: {train_n}
- Validation: {val_n}
- Test: {test_n}
- Total: {total_n} ({size_cat})

## Usage
Python (datasets):
```python
from datasets import load_dataset
ds = load_dataset("{repo_id}")
print(ds)
print(ds["train"][0])
```

## Ethical Considerations and Warnings
- Content Warning: This dataset may include explicit, offensive, or otherwise harmful content reflective of its source forum.
- Alignment: Users should apply filtering, detoxification, and safety alignment techniques prior to training or deployment.
- Privacy: Only public forum content was scraped; nevertheless, review and respect platform policies and applicable laws.

## Licensing
The original posts are user-generated content from 4chan whose licensing status is not explicitly defined. As a result, this dataset is distributed under `other` with research intent only. Users are responsible for ensuring compliance with applicable laws and platform policies.

## Example Records
{examples_md}

## How to Cite
If you use this dataset, please include a citation to this repository and note the source as 4chan public threads.

## Changelog
- v1.0: Initial release.
"""

    return yaml_header + textwrap.dedent(body).lstrip()


def upload_readme(repo_id: str, token: str, content: str) -> None:
    api = HfApi()
    fileobj = io.BytesIO(content.encode("utf-8"))
    api.upload_file(
        path_or_fileobj=fileobj,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )


def main():
    # Load + normalize
    records = load_records(DATA_FILE)
    examples = normalize_records(records, min_len=MIN_LEN)
    if not examples:
        raise RuntimeError("No valid examples after normalization.")

    # Build dataset splits
    dd = build_dataset_dict(
        examples=examples,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        shuffle=SHUFFLE,
        seed=SEED,
    )

    # Save locally
    os.makedirs(SAVE_DIR, exist_ok=True)
    dd.save_to_disk(SAVE_DIR)
    print(f"Saved dataset to: {SAVE_DIR}")
    for split, ds in dd.items():
        print(f"  {split}: {len(ds)} examples")

    # Optionally push to Hub
    if PUSH_TO_HUB:
        if not REPO_ID:
            raise ValueError("PUSH_TO_HUB=True requires REPO_ID (e.g. 'username/repo').")
        token = HF_TOKEN or os.environ.get("HF_TOKEN") or HfFolder.get_token()
        if not token:
            raise ValueError("No HF token found. Set HF_TOKEN constant, env var, or run `huggingface-cli login`.")
        push_to_hub(dd, repo_id=REPO_ID, private=PRIVATE, token=token)
        print(f"Pushed to Hub: https://huggingface.co/datasets/{REPO_ID}")
        # Build and upload a professional dataset card
        readme = build_dataset_card_content(dd, REPO_ID)
        upload_readme(REPO_ID, token, readme)
        print("Uploaded dataset card (README.md)")


if __name__ == "__main__":
    main()