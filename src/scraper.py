import re
import time
from html import unescape
from typing import List, Dict, Any, Optional

import requests

BASE_URL = "https://a.4cdn.org"
USER_AGENT = "Mozilla/5.0 (compatible; QwenFineTuneScraper/1.0)"

# Static default allowlist (not used by the UI flow; kept for convenience)
ALLOWLIST_DEFAULT: List[str] = [
    "pol", "b", "r9k", "s4s", "soc",
    "gif", "h", "hc", "d", "s", "aco",
    "trash", "t", "hr", "bant", "news",
    "v", "biz", "adv", "x"
]


def fetch_catalog(board: str) -> List[int]:
    """Return a list of thread IDs for a board using the catalog endpoint."""
    url = f"{BASE_URL}/{board}/catalog.json"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    pages = resp.json()
    thread_ids: List[int] = []
    for page in pages:
        for t in page.get("threads", []):
            if "no" in t:
                thread_ids.append(t["no"])
    return thread_ids


def fetch_catalog_pages(board: str) -> List[List[int]]:
    """Return a list of pages, each page is a list of thread IDs (preserves page grouping)."""
    url = f"{BASE_URL}/{board}/catalog.json"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    pages = resp.json()
    page_threads: List[List[int]] = []
    for page in pages:
        ids: List[int] = []
        for t in page.get("threads", []):
            if "no" in t:
                ids.append(t["no"])
        if ids:
            page_threads.append(ids)
    return page_threads


def fetch_thread(board: str, thread_id: int) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/{board}/thread/{thread_id}.json"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    data = resp.json()
    return data.get("posts", [])


TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+")
QUOTE_REF_RE = re.compile(r">>\d+")
MULTI_WS_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = TAG_RE.sub("", text)
    text = unescape(text)
    return text


def clean_text(raw: Optional[str]) -> str:
    if not raw:
        return ""
    txt = strip_html(raw)
    # Remove greentext quote lines and quote refs
    lines = [ln for ln in txt.splitlines() if not ln.strip().startswith(">")]
    txt = "\n".join(lines)
    txt = QUOTE_REF_RE.sub("", txt)
    # Remove URLs
    txt = URL_RE.sub("", txt)
    # Collapse whitespace
    txt = MULTI_WS_RE.sub(" ", txt).strip()
    return txt


def build_pairs_adjacent(posts: List[Dict[str, Any]], min_len: int = 3) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    cleaned = [clean_text(p.get("com")) for p in posts]
    # Pair sequentially within a thread
    for i in range(1, len(cleaned)):
        a, b = cleaned[i - 1], cleaned[i]
        if len(a) >= min_len and len(b) >= min_len:
            pairs.append({"input": a, "output": b})
    return pairs


def drop_banned(pairs: List[Dict[str, str]], banned_pattern: Optional[re.Pattern]) -> List[Dict[str, str]]:
    if not banned_pattern:
        return pairs
    out: List[Dict[str, str]] = []
    for ex in pairs:
        s = f"{ex['input']}\n{ex['output']}"
        if banned_pattern.search(s):
            continue
        out.append(ex)
    return out


def scrape(
    board: str,
    max_threads: int,
    max_pairs: int,
    delay: float,
    min_len: int,
) -> List[Dict[str, str]]:
    """Scrape one board and return up to max_pairs input/output examples."""
    # Select threads evenly across catalog pages (round-robin) to diversify sampling
    pages = fetch_catalog_pages(board)
    thread_ids: List[int] = []
    idx = [0] * len(pages)
    while len(thread_ids) < max_threads and any(i < len(pages[p]) for p, i in enumerate(idx)):
        for p in range(len(pages)):
            if len(thread_ids) >= max_threads:
                break
            i = idx[p]
            if i < len(pages[p]):
                thread_ids.append(pages[p][i])
                idx[p] += 1
    pairs: List[Dict[str, str]] = []
    for tid in thread_ids:
        posts = fetch_thread(board, tid)
        if not posts:
            continue
        thread_pairs = build_pairs_adjacent(posts, min_len=min_len)
        pairs.extend(thread_pairs)
        if len(pairs) >= max_pairs:
            break
        time.sleep(delay)
    return pairs[:max_pairs]


__all__ = [
    "ALLOWLIST_DEFAULT",
    "fetch_catalog",
    "fetch_catalog_pages",
    "fetch_thread",
    "strip_html",
    "clean_text",
    "build_pairs_adjacent",
    "drop_banned",
    "scrape",
]
