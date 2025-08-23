import os
import time
import re
import json
from html import unescape
from typing import Callable, Dict, List, Optional

import requests

# Public configuration (aligned with other scrapers)
PROXY_URL: Optional[str] = None
USE_ENV_PROXIES: bool = False
STACKAPPS_KEY: str = os.getenv("STACKAPPS_KEY", "")
API_BASE = "https://api.stackexchange.com/2.3"
USER_AGENT = "dataset-studio/stackexchange/1.0"

# Internal session (configured via apply_session_config)
SESSION: Optional[requests.Session] = None


def apply_session_config() -> None:
    """Configure a requests Session using current proxy settings.
    - If USE_ENV_PROXIES is True, allow trust_env and don't override session.proxies
    - Else, disable env proxies and, if PROXY_URL set, route http/https via it
    """
    global SESSION
    s = SESSION or requests.Session()
    # trust_env controls whether requests picks up HTTP(S)_PROXY
    s.trust_env = bool(USE_ENV_PROXIES)
    # Reset proxies each time
    s.proxies = {}
    if not USE_ENV_PROXIES and PROXY_URL:
        s.proxies = {
            "http": PROXY_URL,
            "https": PROXY_URL,
        }
    s.headers.update({"User-Agent": USER_AGENT})
    SESSION = s


def _get_session() -> requests.Session:
    global SESSION
    if SESSION is None:
        apply_session_config()
    return SESSION  # type: ignore[return-value]


def html_to_text(html: str) -> str:
    """Minimal HTML→text cleaner."""
    s = unescape(html or "")
    s = re.sub(r"<pre><code>(.*?)</code></pre>", lambda m: f"\n```\n{m.group(1).strip()}\n```\n",
               s, flags=re.DOTALL | re.IGNORECASE)
    s = s.replace("<br>", "\n").replace("<br/>", "\n").replace("<p>", "\n\n")
    s = re.sub(r"<.*?>", "", s, flags=re.DOTALL)
    lines = [ln.rstrip() for ln in s.splitlines()]
    out: List[str] = []
    last_blank = False
    for ln in lines:
        blank = (ln.strip() == "")
        if blank and last_blank:
            continue
        out.append(ln)
        last_blank = blank
    return "\n".join(out).strip()


def fetch_questions(site: str, page: int, pagesize: int) -> Dict:
    params = {
        "order": "desc",
        "sort": "activity",
        "site": site,
        "filter": "withbody",
        "pagesize": pagesize,
        "page": page,
    }
    if STACKAPPS_KEY:
        params["key"] = STACKAPPS_KEY
    r = _get_session().get(f"{API_BASE}/questions", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_answers(site: str, ids: List[int]) -> Dict[int, Dict]:
    if not ids:
        return {}
    answers: Dict[int, Dict] = {}
    sess = _get_session()
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        ids_str = ";".join(str(x) for x in chunk)
        params = {"order": "desc", "sort": "activity", "site": site, "filter": "withbody"}
        if STACKAPPS_KEY:
            params["key"] = STACKAPPS_KEY
        r = sess.get(f"{API_BASE}/answers/{ids_str}", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for itm in data.get("items", []) or []:
            try:
                answers[int(itm["answer_id"])] = itm
            except Exception:
                pass
        if data.get("backoff"):
            time.sleep(int(data["backoff"]))
    return answers


def scrape(
    site: str = "stackoverflow",
    max_pairs: int = 100,
    delay: float = 0.2,
    min_len: int = 0,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> List[Dict[str, str]]:
    """Fetch Stack Exchange Q/A pairs.

    Returns a list of {"input": question_text, "output": accepted_answer_text}.
    - site: Stack Exchange site (e.g., "stackoverflow", "superuser")
    - max_pairs: number of Q/A pairs to collect
    - delay: polite delay between API calls
    - min_len: optional minimum character length (applied to both input and output)
    - cancel_cb: optional callable that returns True if the caller requested cancellation
    """
    results: List[Dict[str, str]] = []
    written = 0
    seen: set[int] = set()
    page = 1

    while written < max_pairs:
        if cancel_cb and cancel_cb():
            break
        data = fetch_questions(site, page, 100)
        items = data.get("items", []) or []
        if not items:
            break

        accepted = [q for q in items if q.get("is_answered") and q.get("accepted_answer_id")]
        answer_ids = [int(q["accepted_answer_id"]) for q in accepted]
        answers = fetch_answers(site, answer_ids)

        for q in accepted:
            if written >= max_pairs:
                break
            if cancel_cb and cancel_cb():
                break
            try:
                qid = int(q.get("question_id"))
            except Exception:
                qid = None  # type: ignore[assignment]
            if qid in seen:
                continue
            aid = q.get("accepted_answer_id")
            ans = answers.get(int(aid)) if aid is not None else None
            if not ans:
                continue

            q_text = (str(q.get("title", "")).strip() + "\n\n" + html_to_text(q.get("body", ""))).strip()
            a_text = html_to_text(ans.get("body", ""))
            if min_len and (len(q_text) < min_len or len(a_text) < min_len):
                continue

            results.append({
                "input": q_text,
                "output": a_text,
            })
            written += 1
            if qid is not None:
                seen.add(qid)

        if data.get("backoff"):
            time.sleep(int(data["backoff"]))
        time.sleep(max(0.0, float(delay)))
        if not data.get("has_more"):
            break
        page += 1

    return results
