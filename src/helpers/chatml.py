"""ChatML conversation builders.

This module provides utilities to convert 4chan threads (list of post dicts)
into multi-turn ChatML-style conversations compatible with the app's dataset format.

Output example per conversation:
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]
}
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Reuse cleaners and reference extractors from the existing 4chan scraper
from scrapers import fourchan_scraper as sc


def pair_to_chatml(inp: str, out: str, add_system: Optional[str] = None) -> Dict[str, Any]:
    """Wrap a single input/output pair into a ChatML conversation object.

    Returns a dict: {"messages": [...]}
    """
    messages: List[Dict[str, str]] = []
    if add_system:
        messages.append({"role": "system", "content": add_system})
    messages.append({"role": "user", "content": str(inp or "").strip()})
    messages.append({"role": "assistant", "content": str(out or "").strip()})
    return {"messages": messages}


def pairs_to_chatml(pairs: List[Dict[str, Any]], add_system: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convert a list of {input, output} dicts into ChatML conversations."""
    out: List[Dict[str, Any]] = []
    for d in pairs or []:
        inp = d.get("input", "") if isinstance(d, dict) else ""
        outp = d.get("output", "") if isinstance(d, dict) else ""
        if (inp is not None and str(inp).strip() != "") and (outp is not None and str(outp).strip() != ""):
            out.append(pair_to_chatml(str(inp), str(outp), add_system=add_system))
    return out


def _merge_same_author_chunks(ctx_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge adjacent context chunks with the same 4chan poster id into a single chunk."""
    out: List[Dict[str, Any]] = []
    for ch in ctx_chunks:
        if out:
            prev_id = out[-1].get("id")
            cur_id = ch.get("id")
            if prev_id is not None and prev_id == cur_id:
                out[-1]["text"] = (out[-1]["text"] + "\n\n" + ch["text"]).strip()
                continue
        out.append(dict(ch))
    return out


def thread_to_chatml_conversations(
    posts: List[Dict[str, Any]],
    *,
    min_len: int = 3,
    k: int = 6,
    max_rounds_per_conv: int = 6,
    max_chars: Optional[int] = None,
    merge_same_id: bool = True,
    add_system: Optional[str] = None,
    ban_pattern: Optional[re.Pattern] = None,
) -> List[Dict[str, Any]]:
    """Build multi-turn ChatML conversations from a single thread's posts.

    This follows reply quote-chains when available, else falls back to last-k/adjacent
    to build a sequence of messages, and ensures alternation ends with an assistant turn.
    """
    # Map post number -> index
    id_to_idx: Dict[int, int] = {}
    for idx, p in enumerate(posts):
        no = p.get("no")
        if isinstance(no, int):
            id_to_idx[no] = idx

    cleaned = [sc.clean_text(p.get("com")) for p in posts]
    conversations: List[Dict[str, Any]] = []

    for i in range(1, len(posts)):
        cur_raw = posts[i].get("com")
        cur_txt = cleaned[i]
        if len(cur_txt) < min_len:
            continue

        # Build chain of parents via last quotelink
        chain_indices: List[int] = []
        visited: set[int] = set()
        parent_idx: Optional[int] = None

        refs = sc.extract_refs(cur_raw)
        for ref in reversed(refs):
            idx = id_to_idx.get(ref)
            if idx is not None and idx < i:
                parent_idx = idx
                break

        steps = 0
        while parent_idx is not None and steps < k and parent_idx not in visited:
            visited.add(parent_idx)
            chain_indices.append(parent_idx)
            parent_raw = posts[parent_idx].get("com")
            parent_refs = sc.extract_refs(parent_raw)
            next_parent = None
            for ref in reversed(parent_refs):
                cand = id_to_idx.get(ref)
                if cand is not None and cand < parent_idx:
                    next_parent = cand
                    break
            parent_idx = next_parent
            steps += 1

        # Assemble context chunks oldest -> newest
        ctx_chunks: List[Dict[str, Any]] = []
        for j in reversed(chain_indices):
            t = cleaned[j]
            if len(t) >= min_len:
                ctx_chunks.append({"text": t, "id": posts[j].get("id")})

        # Fallbacks
        if not ctx_chunks:
            start = max(0, i - k)
            for j in range(start, i):
                t = cleaned[j]
                if len(t) >= min_len:
                    ctx_chunks.append({"text": t, "id": posts[j].get("id")})

        if not ctx_chunks:
            prev = cleaned[i - 1]
            if len(prev) >= min_len:
                ctx_chunks.append({"text": prev, "id": posts[i - 1].get("id")})
            else:
                continue

        if merge_same_id:
            ctx_chunks = _merge_same_author_chunks(ctx_chunks)

        # Optionally trim message lengths
        def trim_msg(s: str) -> str:
            if max_chars is not None and max_chars > 0 and len(s) > max_chars:
                return s[-max_chars:]
            return s

        # Append current post, merging with previous if same author
        cur_chunk = {"text": cur_txt, "id": posts[i].get("id")}
        if merge_same_id and ctx_chunks:
            prev_id = ctx_chunks[-1].get("id")
            cur_id = cur_chunk.get("id")
            if prev_id is not None and prev_id == cur_id:
                ctx_chunks[-1]["text"] = (ctx_chunks[-1]["text"] + "\n\n" + cur_chunk["text"]).strip()
            else:
                ctx_chunks.append(cur_chunk)
        else:
            ctx_chunks.append(cur_chunk)

        # Convert to list of message texts
        msgs = [trim_msg(ch["text"]).strip() for ch in ctx_chunks if len(ch.get("text", "").strip()) >= min_len]
        # Enforce at least 2 messages
        if len(msgs) < 2:
            continue

        # Keep only the last 2*max_rounds_per_conv messages
        keep = max(2, 2 * max(1, int(max_rounds_per_conv or 1)))
        if len(msgs) > keep:
            msgs = msgs[-keep:]

        # Ensure final role is assistant by having even count; drop earliest if needed
        if len(msgs) % 2 != 0:
            msgs = msgs[1:]
        if len(msgs) < 2:
            continue

        # Drop by banned pattern if any message matches
        if ban_pattern and any(ban_pattern.search(m) for m in msgs):
            continue

        messages: List[Dict[str, str]] = []
        if add_system:
            messages.append({"role": "system", "content": add_system})
        # Alternate roles starting with user, ensuring last is assistant
        for idx, text in enumerate(msgs):
            role = "user" if (idx % 2 == 0) else "assistant"
            messages.append({"role": role, "content": text})

        # Only keep conversations that end with assistant
        if messages and messages[-1].get("role") == "assistant":
            conversations.append({"messages": messages})

    return conversations
