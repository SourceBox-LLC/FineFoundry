from __future__ import annotations

from typing import List, Optional, Tuple


def guess_input_output_columns(names: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Guess likely input/output column names from a list of dataset column names.

    Preference order covers common patterns found in instruction-tuning and Q/A datasets.
    Returns a tuple (input_col, output_col), any of which may be None if not found.
    """
    low = {n.lower(): n for n in (names or [])}
    in_cands = [
        "input",
        "prompt",
        "question",
        "instruction",
        "source",
        "text",
        "query",
        "context",
        "post",
    ]
    out_cands = [
        "output",
        "response",
        "answer",
        "completion",
        "target",
        "label",
        "reply",
    ]
    inn = next((low[x] for x in in_cands if x in low), None)
    outn = next((low[x] for x in out_cands if x in low), None)
    if inn is None and outn is None:
        if "question" in low and "answer" in low:
            return low["question"], low["answer"]
    return inn, outn
