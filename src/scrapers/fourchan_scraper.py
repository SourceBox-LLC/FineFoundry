import re
import time
from html import unescape
from typing import List, Dict, Any, Optional

import requests

from scrapers.utils import get_rate_limiter, make_request_with_retry

BASE_URL = "https://a.4cdn.org"
USER_AGENT = "Mozilla/5.0 (compatible; QwenFineTuneScraper/1.0)"

# Proxy + shared session (default to Tor via SOCKS5 for anonymity)
# Change PROXY_URL to disable or set a different proxy as needed
PROXY_URL: Optional[str] = "socks5h://127.0.0.1:9050"
USE_ENV_PROXIES: bool = False  # If True, let requests use environment proxies

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def apply_session_config() -> None:
    """Apply current proxy configuration to SESSION.

    - If USE_ENV_PROXIES is True, trust environment variables (HTTP(S)_PROXY, etc.).
    - Else, disable env proxies and use PROXY_URL if set.
    """
    # Environment proxies
    if USE_ENV_PROXIES:
        SESSION.trust_env = True
        return
    # Explicit config
    SESSION.trust_env = False
    if PROXY_URL:
        SESSION.proxies.update({"http": PROXY_URL, "https": PROXY_URL})
    else:
        SESSION.proxies.clear()


# Apply default proxy configuration at import time
apply_session_config()

# Static default allowlist (not used by the UI flow; kept for convenience)
ALLOWLIST_DEFAULT: List[str] = [
    "pol",
    "b",
    "r9k",
    "s4s",
    "soc",
    "gif",
    "h",
    "hc",
    "d",
    "s",
    "aco",
    "trash",
    "t",
    "hr",
    "bant",
    "news",
    "v",
    "biz",
    "adv",
    "x",
]


def fetch_catalog(board: str, max_retries: int = 3) -> List[int]:
    """Return a list of thread IDs for a board using the catalog endpoint.
    
    Args:
        board: Board code (e.g., 'pol', 'b').
        max_retries: Maximum retry attempts on failure.
    
    Returns:
        List of thread IDs.
    """
    url = f"{BASE_URL}/{board}/catalog.json"
    rate_limiter = get_rate_limiter("4chan")
    resp = make_request_with_retry(
        SESSION, "GET", url,
        max_retries=max_retries,
        rate_limiter=rate_limiter,
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    resp.raise_for_status()
    pages = resp.json()
    thread_ids: List[int] = []
    for page in pages:
        for t in page.get("threads", []):
            if "no" in t:
                thread_ids.append(t["no"])
    return thread_ids


def fetch_catalog_pages(board: str, max_retries: int = 3) -> List[List[int]]:
    """Return a list of pages, each page is a list of thread IDs (preserves page grouping).
    
    Args:
        board: Board code (e.g., 'pol', 'b').
        max_retries: Maximum retry attempts on failure.
    
    Returns:
        List of pages, each containing thread IDs.
    """
    url = f"{BASE_URL}/{board}/catalog.json"
    rate_limiter = get_rate_limiter("4chan")
    resp = make_request_with_retry(
        SESSION, "GET", url,
        max_retries=max_retries,
        rate_limiter=rate_limiter,
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
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


def fetch_thread(board: str, thread_id: int, max_retries: int = 3) -> List[Dict[str, Any]]:
    """Fetch all posts from a thread.
    
    Args:
        board: Board code (e.g., 'pol', 'b').
        thread_id: Thread number.
        max_retries: Maximum retry attempts on failure.
    
    Returns:
        List of post dictionaries, or empty list if thread not found.
    """
    url = f"{BASE_URL}/{board}/thread/{thread_id}.json"
    rate_limiter = get_rate_limiter("4chan")
    try:
        resp = make_request_with_retry(
            SESSION, "GET", url,
            max_retries=max_retries,
            rate_limiter=rate_limiter,
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("posts", [])
    except requests.exceptions.HTTPError as e:
        if hasattr(e, "response") and e.response is not None and e.response.status_code == 404:
            return []
        raise


TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+")
QUOTE_REF_RE = re.compile(r">>\d+")
MULTI_WS_RE = re.compile(r"\s+")
QUOTELINK_RX = re.compile(r"(?:>>|&gt;&gt;)(\d+)")
QUESTION_RX = re.compile(r"\b(who|what|why|how|where|when|does|do|did|can|could|should|would|is|are|am)\b", re.I)


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


def extract_refs(raw_html: Optional[str]) -> List[int]:
    """Extract referenced post numbers from a post's raw HTML (quotelinks).
    Handles both plain ">>123" and HTML-escaped "&gt;&gt;123" variants.
    """
    if not raw_html:
        return []
    try:
        txt = unescape(raw_html)
    except Exception:
        txt = raw_html
    refs = [int(m) for m in QUOTELINK_RX.findall(txt)]
    return refs


def _merge_same_id_chunks(ctx_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ch in ctx_chunks:
        if out and out[-1].get("id") == ch.get("id"):
            out[-1]["text"] = (out[-1]["text"] + "\n\n" + ch["text"]).strip()
        else:
            out.append(dict(ch))
    return out


def _looks_like_question(text: str) -> bool:
    return ("?" in text) or bool(QUESTION_RX.search(text))


def build_pairs_adjacent(posts: List[Dict[str, Any]], min_len: int = 3) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    cleaned = [clean_text(p.get("com")) for p in posts]
    # Pair sequentially within a thread
    for i in range(1, len(cleaned)):
        a, b = cleaned[i - 1], cleaned[i]
        if len(a) >= min_len and len(b) >= min_len:
            pairs.append({"input": a, "output": b})
    return pairs


def build_pairs_contextual(
    posts: List[Dict[str, Any]],
    *,
    min_len: int = 3,
    strategy: str = "cumulative",
    k: int = 6,
    max_chars: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Build context-aware input/output pairs from a thread.

    For each post i (i>=1), the input is composed from prior posts according
    to the chosen strategy and the output is the current post text.

    - strategy="cumulative": input = join(posts[:i])
    - strategy="last_k":    input = join(posts[max(0, i-k):i])

    If max_chars is provided, the input will be truncated to the last
    max_chars characters to preserve most-recent context.
    """
    cleaned = [clean_text(p.get("com")) for p in posts]
    out: List[Dict[str, str]] = []
    for i in range(1, len(cleaned)):
        cur = cleaned[i]
        if len(cur) < min_len:
            continue
        if strategy == "last_k":
            start = max(0, i - k)
            ctx_list = cleaned[start:i]
        else:
            ctx_list = cleaned[:i]
        # Drop very short/empty context entries to reduce noise
        ctx_list = [c for c in ctx_list if len(c) >= min_len]
        if not ctx_list:
            # no meaningful context; fall back to adjacent
            prev = cleaned[i - 1]
            if len(prev) >= min_len:
                out.append({"input": prev, "output": cur})
            continue
        ctx = "\n\n".join(ctx_list).strip()
        if not ctx:
            continue
        if max_chars is not None and max_chars > 0 and len(ctx) > max_chars:
            ctx = ctx[-max_chars:]
        if len(ctx) >= min_len:
            out.append({"input": ctx, "output": cur})
    return out


def build_pairs_quote_contextual(
    posts: List[Dict[str, Any]],
    *,
    min_len: int = 3,
    k: int = 6,
    max_chars: Optional[int] = None,
    merge_same_id: bool = True,
    require_question: bool = False,
) -> List[Dict[str, str]]:
    """Build pairs using the reply quote-chain as the primary context signal.

    For each post i, we walk the last quotelink (>>no) backwards up to k steps,
    assembling context from the referenced chain. We fall back to last-k prior
    posts if the chain is empty, and finally to adjacent pairing.
    """
    # Map post number -> index for fast lookup
    id_to_idx: Dict[int, int] = {}
    for idx, p in enumerate(posts):
        no = p.get("no")
        if isinstance(no, int):
            id_to_idx[no] = idx

    cleaned = [clean_text(p.get("com")) for p in posts]
    out: List[Dict[str, str]] = []

    for i in range(1, len(posts)):
        cur_raw = posts[i].get("com")
        cur_txt = cleaned[i]
        if len(cur_txt) < min_len:
            continue

        # Build chain following last valid quoted parent repeatedly
        chain_indices: List[int] = []
        visited: set[int] = set()
        parent_idx: Optional[int] = None

        # Choose the last quotelink that points to an earlier post
        refs = extract_refs(cur_raw)
        for ref in reversed(refs):
            idx = id_to_idx.get(ref)
            if idx is not None and idx < i:
                parent_idx = idx
                break

        steps = 0
        while parent_idx is not None and steps < k and parent_idx not in visited:
            visited.add(parent_idx)
            chain_indices.append(parent_idx)
            # Walk further using the parent's last quotelink
            parent_raw = posts[parent_idx].get("com")
            parent_refs = extract_refs(parent_raw)
            next_parent = None
            for ref in reversed(parent_refs):
                cand = id_to_idx.get(ref)
                if cand is not None and cand < parent_idx:
                    next_parent = cand
                    break
            parent_idx = next_parent
            steps += 1

        # Assemble context text from chain (oldest -> newest)
        ctx_chunks: List[Dict[str, Any]] = []
        for j in reversed(chain_indices):
            t = cleaned[j]
            if len(t) >= min_len:
                ctx_chunks.append({"text": t, "id": posts[j].get("id")})

        # Fallbacks if chain too short
        if not ctx_chunks:
            # last_k prior posts
            start = max(0, i - k)
            for j in range(start, i):
                t = cleaned[j]
                if len(t) >= min_len:
                    ctx_chunks.append({"text": t, "id": posts[j].get("id")})

        if not ctx_chunks:
            # adjacent fallback
            prev = cleaned[i - 1]
            if len(prev) >= min_len:
                out.append({"input": prev, "output": cur_txt})
            continue

        if merge_same_id:
            ctx_chunks = _merge_same_id_chunks(ctx_chunks)

        ctx = "\n\n".join(ch["text"] for ch in ctx_chunks).strip()
        if not ctx:
            continue
        if max_chars is not None and max_chars > 0 and len(ctx) > max_chars:
            ctx = ctx[-max_chars:]
        if len(ctx) < min_len:
            continue
        if require_question and not _looks_like_question(ctx):
            continue

        out.append({"input": ctx, "output": cur_txt})
    return out


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
    *,
    mode: str = "normal",  # "normal" (adjacent) or "contextual"
    strategy: str = "cumulative",
    k: int = 6,
    max_chars: Optional[int] = None,
    merge_same_id: bool = True,
    require_question: bool = False,
) -> List[Dict[str, str]]:
    """Scrape one board and return up to max_pairs input/output examples."""
    # Ensure session reflects latest proxy config
    apply_session_config()
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
        if mode == "contextual":
            if strategy == "quote_chain":
                thread_pairs = build_pairs_quote_contextual(
                    posts,
                    min_len=min_len,
                    k=k,
                    max_chars=max_chars,
                    merge_same_id=merge_same_id,
                    require_question=require_question,
                )
            else:
                thread_pairs = build_pairs_contextual(
                    posts,
                    min_len=min_len,
                    strategy=strategy,
                    k=k,
                    max_chars=max_chars,
                )
        else:
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
    "build_pairs_contextual",
    "build_pairs_quote_contextual",
    "drop_banned",
    "scrape",
    "apply_session_config",
]
