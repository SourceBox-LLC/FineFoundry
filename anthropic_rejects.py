import re
import json
import argparse
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


def parse_turns(text: str) -> List[tuple]:
    """Parse a transcript into a list of (role, content) turns.

    Role markers are "Human:" and "Assistant:" (case-insensitive). Works with or
    without newlines between turns.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    rx = re.compile(r"(Human|Assistant)\s*:\s*", flags=re.IGNORECASE)
    turns: List[tuple] = []
    last_role = None
    last_end = 0
    for m in rx.finditer(text):
        role = m.group(1).title()
        if last_role is not None:
            content = text[last_end:m.start()].strip()
            turns.append((last_role, content))
        last_role = role
        last_end = m.end()
    if last_role is not None:
        content = text[last_end:].strip()
        turns.append((last_role, content))
    return turns


def format_data_with_history(
    rejected_text: str,
    *,
    min_len: int = 1,
    strategy: str = "cumulative",  # or "last_k"
    k: int = 6,
    max_chars: int | None = None,
) -> List[Dict[str, str]]:
    """Create input/output pairs that include multi-turn context.

    For each Assistant turn, build an input that includes prior turns and ends
    with a bare "Assistant:" cue. The output is that turn's assistant content.

    - strategy="cumulative": include all prior turns.
    - strategy="last_k": include only the last k prior turns.
    - max_chars: if provided, left-trim earlier turns until input <= max_chars.
    """
    turns = parse_turns(rejected_text)
    if not turns:
        return []

    results: List[Dict[str, str]] = []
    for i, (role, content) in enumerate(turns):
        if role != "Assistant":
            continue
        prior = turns[:i]
        if strategy == "last_k" and k > 0:
            prior = prior[-k:]

        lines = [f"{r}: {c}".strip() for r, c in prior]
        input_text = ("\n".join(lines + ["Assistant:"])) if lines else "Assistant:"

        if max_chars is not None and max_chars > 0 and len(input_text) > max_chars:
            # Left-trim earliest turns until within limit
            # Operate at turn granularity for stability
            j = 0
            while len(input_text) > max_chars and j < len(prior):
                trimmed = prior[j+1:]
                lines = [f"{r}: {c}".strip() for r, c in trimmed]
                input_text = ("\n".join(lines + ["Assistant:"])) if lines else "Assistant:"
                j += 1

        output_text = (content or "").strip()
        if len(input_text.strip()) >= min_len and len(output_text) >= min_len:
            results.append({"input": input_text, "output": output_text})
    return results


def build_rejected_df(
    split,
    min_len: int = 1,
    limit: int | None = None,
    *,
    with_history: bool = True,
    strategy: str = "cumulative",
    k: int = 6,
    max_chars: int | None = None,
    column: str = "rejected",
) -> pd.DataFrame:
    """Build a DataFrame of pairs from a Hugging Face split (e.g., ds["train"]).

    - with_history: if True, include multi-turn context per assistant turn.
    - strategy: "cumulative" or "last_k" (only used when with_history=True)
    - k: number of prior turns to keep when strategy="last_k"
    - max_chars: left-trim context to keep input under this length
    - limit: if provided, only process the first N rejected entries.
    - column: which text column to read from the split (default: "rejected").
    """
    # Guard: ensure requested column exists
    try:
        col_names = getattr(split, "column_names", None)
        if col_names is not None and column not in col_names:
            raise KeyError(f"Column '{column}' not in dataset columns: {col_names}")
        texts = split[column]
    except Exception as e:
        raise ValueError(f"Unable to read column '{column}' from split: {e}")
    if limit is not None:
        texts = texts[:limit]
    all_pairs: List[Dict[str, str]] = []
    for t in texts:
        if with_history:
            all_pairs.extend(
                format_data_with_history(t, min_len=min_len, strategy=strategy, k=k, max_chars=max_chars)
            )
        else:
            all_pairs.extend(format_data(t, min_len=min_len))
    if not all_pairs:
        return pd.DataFrame(columns=["input", "output"])
    return pd.DataFrame(all_pairs, columns=["input", "output"])


def export_rejected_pairs_json(
    output_path: str = "scraped_training_data.json",
    split_name: str = "train",
    min_len: int = 1,
    limit: int | None = None,
    *,
    with_history: bool = True,
    strategy: str = "cumulative",
    k: int = 6,
    max_chars: int | None = None,
    column: str = "rejected",
) -> int:
    """Export rejected-only pairs to a JSON list [{input, output}, ...].

    Returns the number of pairs written.
    """
    ds = load_dataset("Anthropic/hh-rlhf")
    df = build_rejected_df(
        ds[split_name],
        min_len=min_len,
        limit=limit,
        with_history=with_history,
        strategy=strategy,
        k=k,
        max_chars=max_chars,
        column=column,
    )
    records = df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Anthropic HH-RLHF pairs to JSON (rejected by default)")
    parser.add_argument("--output", default="anthropic_rejects_history.json", help="Output JSON path")
    parser.add_argument("--split", default="train", help="Split name: train|test|validation if present")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows to process")
    parser.add_argument("--no-history", action="store_true", help="Disable multi-turn history (use adjacent-only pairs)")
    parser.add_argument("--strategy", choices=["cumulative", "last_k"], default="cumulative", help="History strategy")
    parser.add_argument("--k", type=int, default=6, help="Last-K turns to keep when strategy=last_k")
    parser.add_argument("--max-chars", dest="max_chars", type=int, default=8000, help="Max input length; trims earliest turns if exceeded")
    parser.add_argument("--column", default="rejected", help="Dataset column to read (default: rejected)")
    args = parser.parse_args()

    with_history = not args.no_history
    n = export_rejected_pairs_json(
        output_path=args.output,
        split_name=args.split,
        min_len=1,
        limit=args.limit,
        with_history=with_history,
        strategy=args.strategy,
        k=args.k,
        max_chars=args.max_chars,
        column=args.column,
    )
    print(
        f"Wrote {n} pairs to {args.output} "
        f"(split={args.split}, column={args.column}, history={'on' if with_history else 'off'}, "
        f"strategy={args.strategy}, k={args.k}, max_chars={args.max_chars}, limit={args.limit})"
    )
