#!/usr/bin/env python3
"""
reddit_crawl_full.py — Crawl a subreddit and save full post threads (including expanded "more" comments).

Run with:
    python reddit_crawl_full.py

No CLI args needed. Configure defaults below.
"""

import json
import argparse
import random
import shutil
import tempfile
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# =========================
# CONFIG — CHANGE THESE
# =========================
DEFAULT_URL = "https://www.reddit.com/r/LocalLLaMA/"  # Subreddit front page or a specific post URL
MAX_POSTS = 100                 # Max posts to harvest from the subreddit
REQUEST_DELAY = 1.0             # Base delay between requests (seconds)
UA = "Mozilla/5.0 (compatible; reddit-scraper/2.0; +https://example.com/contact)"
OUTPUT_DIR = None               # None -> auto folder per run; or set to a fixed path string
EXPAND_MORE_COMMENTS = True     # Use /api/morechildren to expand all "more" stubs
BATCH_SIZE_MORECHILDREN = 100   # morechildren allows up to 100 child IDs per call

# Dataset building options
BUILD_DATASET = True
PAIRING_MODE = "parent_child"   # "parent_child" or "contextual"
CONTEXT_K = 4                    # Max ancestors to include in contextual mode
MAX_INPUT_CHARS: Optional[int] = 2000
REQUIRE_QUESTION = False
MERGE_SAME_AUTHOR = True
MIN_LEN = 1                      # Minimum non-whitespace chars per side
EXCLUDE_AUTOMOD = True           # Skip AutoModerator comments

# Rate limiting / safety caps
REQUEST_JITTER_FRAC = 0.5        # Sleep in [REQUEST_DELAY*(1-J), REQUEST_DELAY*(1+J)]
MAX_REQUESTS_TOTAL: Optional[int] = 1000  # None to disable
STOP_AFTER_SECONDS: Optional[int] = None  # e.g., 300 to stop after 5 minutes

# Dump location controls
USE_TEMP_DUMP = True            # If True, write dump into a system temp folder and remove after run

# =========================
# Session
# =========================
# Proxy configuration (default to Tor via SOCKS5 for anonymity)
# Change PROXY_URL to disable or set a different proxy as needed
PROXY_URL: Optional[str] = "socks5h://127.0.0.1:9050"
USE_ENV_PROXIES: bool = False  # If True, let requests use environment proxies

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

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

# Global counters
START_TS = time.time()
REQUESTS_MADE = 0

# Simple cleaners and detection
URL_RE = re.compile(r"https?://\S+")
MULTI_WS_RE = re.compile(r"\s+")
QUESTION_RX = re.compile(r"\b(who|what|why|how|where|when|does|do|did|can|could|should|would|is|are|am|\?)\b", re.I)

def sleep(s: float):
    time.sleep(max(0.0, s))

def sleep_with_jitter(base: float):
    j = max(0.0, float(REQUEST_JITTER_FRAC))
    lo = max(0.0, base * (1.0 - j))
    hi = base * (1.0 + j)
    time.sleep(random.uniform(lo, hi))

def iso(ts: Optional[float]) -> Optional[str]:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None

def safe_filename(text: str, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text)[:limit]

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def log(msg: str) -> None:
    # Simple timestamped logger
    try:
        ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
    except Exception:
        ts = "--:--:--"
    print(f"[{ts}] {msg}", flush=True)

def log_config() -> None:
    # Print a concise configuration summary at startup
    log("=== Config Summary ===")
    log(f"URL: {DEFAULT_URL}")
    log(f"Mode: {'post' if is_post_url(DEFAULT_URL) else 'subreddit'} | max_posts={MAX_POSTS}")
    log(f"Delay: base={REQUEST_DELAY:.2f}s, jitter={REQUEST_JITTER_FRAC:.2f}")
    log(f"Caps: max_requests={MAX_REQUESTS_TOTAL or 0} (0=off), stop_after={STOP_AFTER_SECONDS or 0}s (0=off)")
    log(f"Dump: {'temp' if USE_TEMP_DUMP else 'project'} | output_dir={OUTPUT_DIR or '(auto)'}")
    log(f"Expand 'more' comments: {EXPAND_MORE_COMMENTS}")
    log(f"Dataset: build={BUILD_DATASET} | pairing={PAIRING_MODE} | k={CONTEXT_K} | max_input_chars={MAX_INPUT_CHARS or 0} | min_len={MIN_LEN}")
    log(f"Filter: exclude_automod={EXCLUDE_AUTOMOD} | require_question={REQUIRE_QUESTION}")
    log("=======================")

def get_json(url: str, method="GET", data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None, retries: int = 3) -> Any:
    """GET/POST with simple retry + backoff for 429/5xx."""
    attempt = 0
    while True:
        try:
            log(f"HTTP {method} {url} (attempt {attempt + 1})")
            # Hard caps: time and request budget
            global REQUESTS_MADE
            if STOP_AFTER_SECONDS is not None and (time.time() - START_TS) > float(STOP_AFTER_SECONDS):
                log("Stop time exceeded; halting crawl")
                raise RuntimeError("Stop time exceeded; halting crawl")
            if MAX_REQUESTS_TOTAL is not None and REQUESTS_MADE >= int(MAX_REQUESTS_TOTAL):
                log("Max requests exceeded; halting crawl")
                raise RuntimeError("Max requests exceeded; halting crawl")

            REQUESTS_MADE += 1
            if method == "GET":
                r = SESSION.get(url, params=params, timeout=30)
            else:
                r = SESSION.post(url, data=data, params=params, timeout=30)
            if r.status_code == 429:
                # rate limit — backoff
                wait = (attempt + 1) * 3
                log(f"429 rate limit on {url}; backing off {wait:.1f}s")
                sleep_with_jitter(wait)
                attempt += 1
                if attempt > retries:
                    r.raise_for_status()
                continue
            r.raise_for_status()
            log(f"HTTP OK {method} {url}")
            return r.json()
        except requests.RequestException as e:
            log(f"HTTP error on {url}: {e} (attempt {attempt + 1})")
            attempt += 1
            if attempt > retries:
                raise
            sleep_with_jitter(2 * attempt)

def to_json_url(url: str) -> str:
    return (url.rstrip("/") + "/") + ".json"

def is_post_url(url: str) -> bool:
    return "/comments/" in url

# =========================
# Parsing helpers
# =========================
def parse_subreddit_listing(listing: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    posts = []
    children = listing.get("data", {}).get("children", [])
    for ch in children:
        d = ch.get("data", {})
        posts.append({
            "id": d.get("id"),
            "fullname": d.get("name"),  # t3_xxxxx
            "subreddit": d.get("subreddit"),
            "title": d.get("title"),
            "author": d.get("author"),
            "created_utc": iso(d.get("created_utc")),
            "permalink": d.get("permalink"),
            "url": "https://www.reddit.com" + (d.get("permalink") or ""),
            "score": d.get("score"),
            "num_comments": d.get("num_comments"),
            "is_self": d.get("is_self"),
        })
    after = listing.get("data", {}).get("after")
    return posts, after

def parse_post_header(json_data: List[Any]) -> Dict[str, Any]:
    post_raw = json_data[0]["data"]["children"][0]["data"]
    return {
        "id": post_raw.get("id"),
        "fullname": post_raw.get("name"),  # t3_xxxxx
        "subreddit": post_raw.get("subreddit"),
        "title": post_raw.get("title"),
        "author": post_raw.get("author"),
        "created_utc": iso(post_raw.get("created_utc")),
        "url": "https://www.reddit.com" + post_raw.get("permalink", ""),
        "score": post_raw.get("score"),
        "num_comments": post_raw.get("num_comments"),
        "is_self": post_raw.get("is_self"),
        "selftext": (post_raw.get("selftext") or "").strip(),
        "external_url": post_raw.get("url") if not post_raw.get("is_self") else None,
        "preview_images": [
            i.get("source", {}).get("url")
            for i in (post_raw.get("preview", {}).get("images") or [])
        ],
    }

def flatten_initial_comments(json_data: List[Any]) -> Tuple[List[Dict[str, Any]], List[str], str]:
    """
    Returns (flat_comments, pending_more_ids, link_fullname)
    """
    link_fullname = json_data[0]["data"]["children"][0]["data"]["name"]  # t3_xxxxx
    comments_root = json_data[1]["data"]["children"]
    flat: List[Dict[str, Any]] = []
    more_ids: List[str] = []

    def walk(node: Dict[str, Any], depth: int):
        kind = node.get("kind")
        data = node.get("data", {})
        if kind == "t1":
            flat.append({
                "kind": "t1",
                "id": data.get("id"),
                "author": data.get("author"),
                "created_utc": iso(data.get("created_utc")),
                "score": data.get("score"),
                "depth": depth,
                "body": (data.get("body") or "").strip(),
                "parent_id": data.get("parent_id"),
                "permalink": data.get("permalink"),
            })
            replies = data.get("replies")
            if isinstance(replies, dict):
                for child in replies.get("data", {}).get("children", []):
                    walk(child, depth + 1)
        elif kind == "more":
            more_ids.extend(data.get("children", []) or [])

    for c in comments_root:
        walk(c, depth=0)

    return flat, more_ids, link_fullname

def fetch_more_children(link_fullname: str, children_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Calls /api/morechildren.json to expand 'more' stubs.
    Returns a list of normalized comments (like the t1 items above).
    """
    out: List[Dict[str, Any]] = []
    # Batch up to BATCH_SIZE_MORECHILDREN
    total_batches = max(1, math.ceil(len(children_ids) / max(1, BATCH_SIZE_MORECHILDREN)))
    for i in range(0, len(children_ids), BATCH_SIZE_MORECHILDREN):
        batch = children_ids[i: i + BATCH_SIZE_MORECHILDREN]
        batch_idx = (i // BATCH_SIZE_MORECHILDREN) + 1
        log(f"Expanding 'more' comments batch {batch_idx}/{total_batches} (size={len(batch)})")
        params = {
            "api_type": "json",
            "link_id": link_fullname,  # e.g., t3_abc123
            "children": ",".join(batch),
            "sort": "best",
        }
        sleep_with_jitter(REQUEST_DELAY)
        resp = get_json("https://www.reddit.com/api/morechildren.json", method="POST", data=params)
        # Structure: { "json": { "data": { "things": [ {...}, ... ] } } }
        things = (((resp or {}).get("json", {}) or {}).get("data", {}) or {}).get("things", [])
        for t in things:
            if t.get("kind") == "t1":
                d = t.get("data", {})
                out.append({
                    "kind": "t1",
                    "id": d.get("id"),
                    "author": d.get("author"),
                    "created_utc": iso(d.get("created_utc")),
                    "score": d.get("score"),
                    "depth": d.get("depth", 0),  # Reddit includes computed depth here
                    "body": (d.get("body") or "").strip(),
                    "parent_id": d.get("parent_id"),
                    "permalink": d.get("permalink"),
                })
            elif t.get("kind") == "more":
                # Rare: nested 'more' comes back — queue again
                morekids = (t.get("data", {}) or {}).get("children", []) or []
                if morekids:
                    out.extend(fetch_more_children(link_fullname, morekids))
    return out

# =========================
# Markdown rendering
# =========================
def to_markdown(thread: Dict[str, Any]) -> str:
    p = thread["post"]
    comments = thread["comments"]
    lines = [
        f"# {p['title']}",
        f"*Subreddit:* r/{p['subreddit']}  ",
        f"*Author:* u/{p['author']}  ",
        f"*Created:* {p['created_utc']}  ",
        f"*Score:* {p['score']} | *Comments:* {p['num_comments']}  ",
        f"*URL:* {p['url']}",
        "",
    ]
    if p.get("selftext"):
        lines += ["## Post Body", p["selftext"], ""]
    if p.get("external_url"):
        lines += [f"**Link:** {p['external_url']}", ""]
    if comments:
        lines.append("## Conversation")
        # sort by depth then by created to keep a stable order
        comments_sorted = sorted(comments, key=lambda c: (c.get("depth", 0), c.get("created_utc") or ""))
        for c in comments_sorted:
            indent = "  " * int(c.get("depth", 0))
            author = c.get("author") or "unknown"
            created = c.get("created_utc") or ""
            body = c.get("body") or ""
            lines.append(f"{indent}- **u/{author}** ({created}) — {body}")
    return "\n".join(lines)

# =========================
# Pair builders (dataset)
# =========================
def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    t = URL_RE.sub("", text)
    t = t.replace("\r", "\n")
    t = MULTI_WS_RE.sub(" ", t)
    return t.strip()

def looks_like_question(text: str) -> bool:
    if not text:
        return False
    # Check in the last 300 chars for a question mark or question word
    tail = text[-300:]
    return bool(QUESTION_RX.search(tail)) or ("?" in tail)

def _post_text(post: Dict[str, Any]) -> str:
    title = post.get("title") or ""
    body = post.get("selftext") or ""
    if body:
        return clean_text(f"{title}\n\n{body}")
    return clean_text(title)

def _comment_text(c: Dict[str, Any]) -> str:
    return clean_text(c.get("body") or "")

def _author(c: Optional[Dict[str, Any]]) -> str:
    return (c.get("author") if c else None) or ""

def build_pairs_parent_child(thread: Dict[str, Any]) -> List[Dict[str, str]]:
    post = thread["post"]
    comments: List[Dict[str, Any]] = thread.get("comments", [])
    id_map = {c.get("id"): c for c in comments}
    pairs: List[Dict[str, str]] = []
    post_txt = _post_text(post)
    for c in comments:
        if EXCLUDE_AUTOMOD and (_author(c).lower() == "automoderator"):
            continue
        child_txt = _comment_text(c)
        if len(child_txt.strip()) < MIN_LEN:
            continue
        pid = c.get("parent_id") or ""
        if pid.startswith("t1_"):
            parent = id_map.get(pid[3:])
            if not parent:
                continue
            if EXCLUDE_AUTOMOD and (_author(parent).lower() == "automoderator"):
                continue
            parent_txt = _comment_text(parent)
        elif pid.startswith("t3_"):
            parent_txt = post_txt
        else:
            continue
        if len(parent_txt.strip()) < MIN_LEN:
            continue
        pairs.append({"input": parent_txt, "output": child_txt})
    return pairs

def build_pairs_contextual(thread: Dict[str, Any], k: int = CONTEXT_K, merge_same_author: bool = MERGE_SAME_AUTHOR,
                           require_question: bool = REQUIRE_QUESTION, max_chars: Optional[int] = MAX_INPUT_CHARS) -> List[Dict[str, str]]:
    post = thread["post"]
    comments: List[Dict[str, Any]] = thread.get("comments", [])
    id_map = {c.get("id"): c for c in comments}
    pairs: List[Dict[str, str]] = []
    post_txt = _post_text(post)

    for c in comments:
        if EXCLUDE_AUTOMOD and (_author(c).lower() == "automoderator"):
            continue
        child_txt = _comment_text(c)
        if len(child_txt.strip()) < MIN_LEN:
            continue

        # Walk up ancestors from immediate parent
        chain_nodes: List[Dict[str, Any]] = []
        cur_parent_id = c.get("parent_id") or ""
        steps = 0
        while steps < max(0, int(k)) and cur_parent_id:
            if cur_parent_id.startswith("t1_"):
                parent = id_map.get(cur_parent_id[3:])
                if not parent:
                    break
                chain_nodes.append(parent)
                cur_parent_id = parent.get("parent_id") or ""
                steps += 1
            elif cur_parent_id.startswith("t3_"):
                # Root is the post
                chain_nodes.append({"__type": "post", "text": post_txt, "author": post.get("author") or ""})
                break
            else:
                break

        if not chain_nodes:
            # If no ancestors found (shouldn't happen), fallback to post
            chain_nodes.append({"__type": "post", "text": post_txt, "author": post.get("author") or ""})

        # Make chronological: oldest -> newest
        chain_nodes = list(reversed(chain_nodes))

        # Convert to texts and optionally merge by author
        pieces: List[Tuple[str, str]] = []  # (author, text)
        for node in chain_nodes:
            if node.get("__type") == "post":
                a = node.get("author") or ""
                t = clean_text(node.get("text") or "")
            else:
                a = _author(node)
                t = _comment_text(node)
            if len(t.strip()) < MIN_LEN:
                continue
            if merge_same_author and pieces and pieces[-1][0] == (a or ""):
                pieces[-1] = (pieces[-1][0], (pieces[-1][1] + "\n\n" + t).strip())
            else:
                pieces.append((a or "", t))

        context = "\n\n".join([t for (_a, t) in pieces]).strip()
        if not context:
            continue
        if require_question and not looks_like_question(context):
            continue
        if max_chars and len(context) > max_chars:
            context = context[-max_chars:]
        pairs.append({"input": context, "output": child_txt})

    return pairs

def build_pairs_for_thread(thread: Dict[str, Any]) -> List[Dict[str, str]]:
    if PAIRING_MODE == "contextual":
        return build_pairs_contextual(thread)
    return build_pairs_parent_child(thread)

# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reddit scraper -> threads + conversational pairs")
    p.add_argument("--url", default=DEFAULT_URL, help="Subreddit front page or a specific post URL")
    p.add_argument("--max-posts", type=int, default=MAX_POSTS, help="Max posts to harvest from subreddit")
    p.add_argument("--request-delay", type=float, default=REQUEST_DELAY, help="Base delay between requests (s)")
    p.add_argument("--request-jitter-frac", type=float, default=REQUEST_JITTER_FRAC, help="Fractional jitter for delays (0.0-1.0)")
    p.add_argument("--max-requests", type=int, default=(MAX_REQUESTS_TOTAL or 0), help="Stop after this many HTTP requests (0=disabled)")
    p.add_argument("--stop-after-seconds", type=int, default=(STOP_AFTER_SECONDS or 0), help="Stop after this many seconds (0=disabled)")
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory (default: auto timestamped)")
    p.add_argument("--use-temp-dump", action="store_true", default=USE_TEMP_DUMP, help="Write the dump into a temporary directory and remove it after run")
    p.add_argument("--no-expand-more", action="store_true", help="Do not expand 'more' comments via API")

    # Pairing options
    p.add_argument("--build-dataset", action="store_true", default=BUILD_DATASET, help="Build conversational pairs JSON")
    p.add_argument("--mode", choices=["parent_child", "contextual"], default=PAIRING_MODE, help="Pairing mode")
    p.add_argument("--k", type=int, default=CONTEXT_K, help="Contextual: number of ancestors to include")
    p.add_argument("--max-input-chars", type=int, default=(MAX_INPUT_CHARS or 0), help="Truncate input context to this many chars (0=disabled)")
    p.add_argument("--require-question", action="store_true", default=REQUIRE_QUESTION, help="Keep only pairs whose context looks like a question")
    p.add_argument("--no-merge-same-author", action="store_true", help="Do not merge consecutive context nodes with same author")
    p.add_argument("--min-len", type=int, default=MIN_LEN, help="Minimum chars for input/output after cleaning")
    p.add_argument("--include-automod", action="store_true", help="Include AutoModerator comments")
    p.add_argument("--pairs-path", default=None, help="If set, also write pairs to this path in addition to dump/pairs")
    p.add_argument("--cleanup", action="store_true", help="Delete the reddit_dump_* folder after run (use with --pairs_path to preserve pairs)")

    return p.parse_args()

# =========================
# Harvest logic
# =========================
def harvest_subreddit(sub_url: str, max_posts: int) -> List[Dict[str, Any]]:
    harvested: List[Dict[str, Any]] = []
    after = None
    remaining = max_posts
    while remaining > 0:
        limit = min(100, remaining)
        params = {"limit": str(limit)}
        if after:
            params["after"] = after
        log(f"Listing subreddit page (limit={limit}, remaining={remaining}, after={after})")
        sleep_with_jitter(REQUEST_DELAY)
        listing = get_json(to_json_url(sub_url), params=params)
        posts, after = parse_subreddit_listing(listing)
        harvested.extend(posts)
        log(f"Fetched {len(posts)} posts from listing; next after={after}")
        remaining -= len(posts)
        if not after or len(posts) == 0:
            log("No further pages in subreddit listing; stopping")
            break
    return harvested[:max_posts]

def harvest_post(permalink_or_url: str) -> Dict[str, Any]:
    thread_url = permalink_or_url if permalink_or_url.startswith("http") else "https://www.reddit.com" + permalink_or_url
    log(f"Fetching post thread: {thread_url}")
    sleep_with_jitter(REQUEST_DELAY)
    j = get_json(to_json_url(thread_url))
    post = parse_post_header(j)
    flat, more_ids, link_fullname = flatten_initial_comments(j)
    log(f"Initial comments flattened: {len(flat)}; pending more IDs: {len(more_ids)}")

    if EXPAND_MORE_COMMENTS and more_ids:
        # de-dup and expand
        uniq = sorted(set(more_ids))
        log(f"Expanding {len(uniq)} 'more' IDs in batches of {BATCH_SIZE_MORECHILDREN}")
        expanded = fetch_more_children(link_fullname, uniq)
        flat.extend(expanded)
        log(f"Expanded comments total: +{len(expanded)} -> {len(flat)} items")
    elif not EXPAND_MORE_COMMENTS and more_ids:
        log("Skipping 'more' expansion (disabled)")

    # Final normalization: only keep comments (kind=t1)
    comments = [c for c in flat if c.get("kind") == "t1"]
    thread = {"type": "post", "post": post, "comments": comments}
    return thread

def run() -> Tuple[Path, Optional[Path]]:
    log("Starting run")
    if USE_TEMP_DUMP:
        # Use the mkdtemp directory directly as the dump folder to ensure cleanup removes it fully
        base_out = Path(tempfile.mkdtemp(prefix=f"reddit_dump_{safe_filename(DEFAULT_URL.strip('/'))}_"))
        print(f"[INFO] Using temporary dump directory: {base_out}")
    else:
        base_out = Path(OUTPUT_DIR) if OUTPUT_DIR else Path(
            f"reddit_dump_{safe_filename(DEFAULT_URL.strip('/'))}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        print(f"[INFO] Using project dump directory: {base_out}")
    ensure_dir(base_out)
    posts_dir = ensure_dir(base_out / "posts")

    index: Dict[str, Any] = {
        "source": DEFAULT_URL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "post_count": 0,
        "posts": [],
    }

    all_pairs: List[Dict[str, str]] = []
    pairs_path: Optional[Path] = None

    if is_post_url(DEFAULT_URL):
        # Single thread mode
        log("Mode: single post")
        thread = harvest_post(DEFAULT_URL)
        pid = thread["post"]["id"]
        stem = safe_filename(pid or "post")
        # write files
        with (posts_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
            json.dump(thread, f, indent=2, ensure_ascii=False)
        with (posts_dir / f"{stem}.md").open("w", encoding="utf-8") as f:
            f.write(to_markdown(thread))
        log(f"Wrote files for post {stem}")
        index["post_count"] = 1
        index["posts"].append({
            "id": pid,
            "title": thread["post"]["title"],
            "url": thread["post"]["url"],
            "json": f"posts/{stem}.json",
            "md": f"posts/{stem}.md",
            "num_comments_scraped": len(thread["comments"]),
        })
        if BUILD_DATASET:
            all_pairs.extend(build_pairs_for_thread(thread))
            log(f"Pairs built so far: {len(all_pairs)}")
    else:
        # Subreddit crawl mode
        log(f"Mode: subreddit crawl -> {DEFAULT_URL} (max_posts={MAX_POSTS})")
        posts = harvest_subreddit(DEFAULT_URL, MAX_POSTS)
        log(f"Total posts to process: {len(posts)}")
        for idx, p in enumerate(posts, start=1):
            try:
                log(f"[{idx}/{len(posts)}] Processing post: {p.get('id')} — {safe_filename(p.get('title') or '')[:80]}")
                thread = harvest_post(p["url"])
                pid = thread["post"]["id"]
                stem = safe_filename(pid or p["id"] or "post")
                with (posts_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
                    json.dump(thread, f, indent=2, ensure_ascii=False)
                with (posts_dir / f"{stem}.md").open("w", encoding="utf-8") as f:
                    f.write(to_markdown(thread))
                log(f"Wrote files for post {stem}")
                index["posts"].append({
                    "id": pid,
                    "title": thread["post"]["title"],
                    "url": thread["post"]["url"],
                    "json": f"posts/{stem}.json",
                    "md": f"posts/{stem}.md",
                    "num_comments_scraped": len(thread["comments"]),
                })
                index["post_count"] += 1
                if BUILD_DATASET:
                    all_pairs.extend(build_pairs_for_thread(thread))
                    if index["post_count"] % 5 == 0:
                        log(f"Processed {index['post_count']} posts; pairs so far: {len(all_pairs)}")
            except Exception as e:
                log(f"Error processing post {p.get('id')}: {e}")
                # keep going on failures
                index["posts"].append({
                    "id": p.get("id"),
                    "title": p.get("title"),
                    "url": p.get("url"),
                    "error": str(e),
                })
                continue

    log("Writing index.json")
    with (base_out / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    log(f"Index written with {index['post_count']} posts")

    if BUILD_DATASET:
        log("Writing pairs JSON")
        pairs_path = base_out / "reddit_pairs.json"
        with pairs_path.open("w", encoding="utf-8") as f:
            json.dump(all_pairs, f, indent=2, ensure_ascii=False)
        print(f"[OK] Built {len(all_pairs)} pairs -> {pairs_path}")

    print(f"[OK] Saved {index['post_count']} threads into: {base_out.resolve()}")
    print("First few items:")
    for item in index["posts"][:min(5, len(index['posts']))]:
        print(" -", item.get("title"), "->", item.get("json"))
    # Summary
    elapsed = time.time() - START_TS
    log(f"Run complete: requests={REQUESTS_MADE}, elapsed={elapsed:.1f}s")
    return base_out, pairs_path

if __name__ == "__main__":
    args = parse_args()
    # Override module-level config with CLI
    DEFAULT_URL = args.url
    MAX_POSTS = args.max_posts
    REQUEST_DELAY = args.request_delay
    REQUEST_JITTER_FRAC = max(0.0, float(args.request_jitter_frac))
    MAX_REQUESTS_TOTAL = None if (args.max_requests is None or int(args.max_requests) <= 0) else int(args.max_requests)
    STOP_AFTER_SECONDS = None if (args.stop_after_seconds is None or int(args.stop_after_seconds) <= 0) else int(args.stop_after_seconds)
    OUTPUT_DIR = args.output_dir
    USE_TEMP_DUMP = bool(args.use_temp_dump)
    EXPAND_MORE_COMMENTS = not args.no_expand_more

    BUILD_DATASET = bool(args.build_dataset)
    PAIRING_MODE = args.mode
    CONTEXT_K = args.k
    MAX_INPUT_CHARS = None if (args.max_input_chars is None or int(args.max_input_chars) <= 0) else int(args.max_input_chars)
    REQUIRE_QUESTION = bool(args.require_question)
    MERGE_SAME_AUTHOR = not bool(args.no_merge_same_author)
    MIN_LEN = max(0, int(args.min_len))
    EXCLUDE_AUTOMOD = not bool(args.include_automod)

    # Show configuration before starting
    log_config()

    dump_dir, pairs_src = run()

    # Copy pairs to destination
    if BUILD_DATASET and pairs_src and pairs_src.exists():
        try:
            if args.pairs_path:
                dst = Path(args.pairs_path)
            else:
                # Default: write to project root as reddit_pairs.json
                dst = Path.cwd() / "reddit_pairs.json"
            ensure_dir(dst.parent)
            dst.write_text(pairs_src.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[OK] Copied pairs to: {dst.resolve()}")
        except Exception as e:
            print(f"[WARN] Could not copy pairs to destination: {e}")

    # Optional cleanup of dump directory
    if args.cleanup or USE_TEMP_DUMP:
        try:
            shutil.rmtree(dump_dir, ignore_errors=True)
            print(f"[OK] Cleanup complete. Removed: {dump_dir.resolve()}")
        except Exception as e:
            print(f"[WARN] Cleanup failed for {dump_dir}: {e}")
