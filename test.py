#!/usr/bin/env python3
"""
Scrape 4chan and build multi-turn ChatML conversations.

- Fetches threads from a board using scrapers.fourchan_scraper.
- Extracts reply chains and adjacent context to form multi-turn dialogues.
- Exports a list of objects:
  { "messages": [ {"role":"user","content":"..."}, {"role":"assistant","content":"..."}, ... ] }

This script is standalone and does not modify the main app. Proxies are always disabled here.
"""

import os
import sys
import re
import json
import time
import argparse
from typing import List, Dict, Any, Optional

# Ensure local src/ is importable
ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Import the 4chan scraper
from scrapers import fourchan_scraper as sc  # noqa: E402


def merge_same_author_chunks(ctx_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    We follow reply quote-chains when available, else fall back to last-k/adjacent
    to build a sequence of messages, and ensure alternation ends with an assistant turn.
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
            ctx_chunks = merge_same_author_chunks(ctx_chunks)

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


def save_dataset(objs: List[Dict[str, Any]], out_path: str, jsonl: bool, pretty_jsonl: bool = False) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if jsonl:
        with open(out_path, "w", encoding="utf-8") as f:
            for obj in objs:
                if pretty_jsonl:
                    # Note: multi-line JSON objects break strict one-object-per-line JSONL readers.
                    # Use only for human readability.
                    f.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
                else:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(objs, f, ensure_ascii=False, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description="Scrape 4chan and export multi-turn ChatML conversations.")
    p.add_argument("--board", type=str, default="pol", help="Board to scrape.")
    p.add_argument("--max-threads", type=int, default=10, help="Max threads to sample.")
    p.add_argument("--delay", type=float, default=0.5, help="Delay between thread fetches (seconds).")
    p.add_argument("--min-len", type=int, default=3, help="Minimum length per text block.")
    p.add_argument("--k", type=int, default=6, help="Max steps to walk the reply chain (or last-k fallback).")
    p.add_argument("--max-rounds-per-conv", type=int, default=6, help="Max user+assistant rounds per conversation.")
    p.add_argument("--max-chars", type=int, default=None, help="Truncate each message to last N chars (if set).")
    p.add_argument("--merge-same-id", action="store_true", default=True, help="Merge consecutive same-id chunks.")
    p.add_argument("--no-merge-same-id", dest="merge_same_id", action="store_false", help="Disable merging.")
    p.add_argument("--ban-regex", type=str, default=None, help="Regex to drop conversations if any message matches.")
    p.add_argument("--add-system", type=str, default=None, help="Optional system message to prepend.")
    p.add_argument("--jsonl", action="store_true", help="Write JSONL instead of JSON.")
    p.add_argument(
        "--pretty-jsonl",
        action="store_true",
        help="Pretty-print JSON objects for readability when using --jsonl (multi-line per record).",
    )
    p.add_argument("--output", type=str, required=True, help="Output path (.json or .jsonl).")

    args = p.parse_args()

    # Disable proxies for this experimental script
    sc.USE_ENV_PROXIES = False
    sc.PROXY_URL = None
    sc.apply_session_config()

    ban_patt = None
    if args.ban_regex:
        try:
            ban_patt = re.compile(args.ban_regex, re.I)
        except re.error as e:
            raise SystemExit(f"Invalid --ban-regex: {e}") from e

    # Choose threads round-robin across catalog pages to diversify sampling
    pages = sc.fetch_catalog_pages(args.board)
    thread_ids: List[int] = []
    idx = [0] * len(pages)
    while len(thread_ids) < args.max_threads and any(i < len(pages[p]) for p, i in enumerate(idx)):
        for p_i in range(len(pages)):
            if len(thread_ids) >= args.max_threads:
                break
            i = idx[p_i]
            if i < len(pages[p_i]):
                thread_ids.append(pages[p_i][i])
                idx[p_i] += 1

    print(f"[i] Selected {len(thread_ids)} threads from /{args.board}/ to scrape.")

    conversations: List[Dict[str, Any]] = []
    for tid in thread_ids:
        posts = sc.fetch_thread(args.board, tid)
        if not posts:
            continue
        convs = thread_to_chatml_conversations(
            posts,
            min_len=args.min_len,
            k=args.k,
            max_rounds_per_conv=args.max_rounds_per_conv,
            max_chars=args.max_chars,
            merge_same_id=args.merge_same_id,
            add_system=args.add_system,
            ban_pattern=ban_patt,
        )
        if convs:
            conversations.extend(convs)
        time.sleep(args.delay)

    print(f"[i] Built {len(conversations)} ChatML conversations.")
    save_dataset(conversations, args.output, jsonl=args.jsonl, pretty_jsonl=args.pretty_jsonl)
    print(f"[ok] Wrote {args.output}")


if __name__ == "__main__":
    main()