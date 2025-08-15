import re
import json
from typing import List, Dict

import pandas as pd
from datasets import load_dataset



def format_data(rejected_text: str, min_len: int = 1) -> List[Dict[str, str]]:
    """Split one rejected conversation transcript into input/output pairs.

    - Input is the text following each "Human:" marker up to the next "Assistant:".
    - Output is the text following that "Assistant:" up to the next "Human:" or end.

    Returns a list of dicts: {"input": human_text, "output": assistant_text}.
    """
    if not isinstance(rejected_text, str) or not rejected_text.strip():
        return []

    pattern = re.compile(
        r"Human:\s*(.*?)\s*Assistant:\s*(.*?)(?=(?:\r?\n+\s*Human:|\Z))",
        flags=re.DOTALL | re.IGNORECASE,
    )

    pairs: List[Dict[str, str]] = []
    for human, assistant in pattern.findall(rejected_text):
        inp = human.strip()
        out = assistant.strip()
        if len(inp) >= min_len and len(out) >= min_len:
            pairs.append({"input": inp, "output": out})
    return pairs


def format_data_pd(rejected_text: str, min_len: int = 1) -> pd.DataFrame:
    """Pandas-friendly wrapper: returns a DataFrame with columns [input, output]."""
    pairs = format_data(rejected_text, min_len=min_len)
    if not pairs:
        return pd.DataFrame(columns=["input", "output"])
    return pd.DataFrame(pairs, columns=["input", "output"])


def build_rejected_df(split, min_len: int = 1, limit: int | None = None) -> pd.DataFrame:
    """Build a DataFrame of all pairs from a Hugging Face split (e.g., ds["train"]).

    - split: a datasets.Dataset (e.g., ds["train"]).
    - limit: if provided, only process the first N rejected entries.
    """
    texts = split["rejected"]
    if limit is not None:
        texts = texts[:limit]
    dfs = [format_data_pd(t, min_len=min_len) for t in texts]
    if not dfs:
        return pd.DataFrame(columns=["input", "output"])
    return pd.concat(dfs, ignore_index=True)


def export_rejected_pairs_json(
    output_path: str = "scraped_training_data.json",
    split_name: str = "train",
    min_len: int = 1,
    limit: int | None = None,
) -> int:
    """Export rejected-only pairs to a JSON list [{input, output}, ...] using pandas.

    Returns the number of pairs written.
    """
    ds = load_dataset("Anthropic/hh-rlhf")
    df = build_rejected_df(ds[split_name], min_len=min_len, limit=limit)
    records = df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return len(records)


if __name__ == "__main__":
    ds = load_dataset("Anthropic/hh-rlhf")
    sample_rejected = ds["train"]["rejected"][0]
    examples = format_data(sample_rejected)
    print(f"Extracted {len(examples)} pairs from first rejected example")
    for ex in examples[:3]:
        print("- input:", ex["input"][:120].replace("\n", " ") + ("…" if len(ex["input"]) > 120 else ""))
        print("  output:", ex["output"][:120].replace("\n", " ") + ("…" if len(ex["output"]) > 120 else ""))
