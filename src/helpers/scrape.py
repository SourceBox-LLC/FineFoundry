from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from datetime import datetime
from typing import List, Optional, Tuple

import flet as ft

from helpers.common import safe_update
from helpers.ui import compute_two_col_flex, two_col_header, two_col_row
from helpers.chatml import (
    thread_to_chatml_conversations,
    pairs_to_chatml,
    reddit_thread_to_chatml_conversations,
)
from helpers.proxy import apply_proxy_from_ui
from scrapers import stackexchange_scraper as sx
from scrapers import fourchan_scraper as sc
from scrapers import reddit_scraper as rdt


async def run_stackexchange_scrape(
    page: ft.Page,
    log_view: ft.ListView,
    prog: ft.ProgressBar,
    labels: dict,
    preview_host: ft.ListView,
    cancel_flag: dict,
    site: str,
    max_pairs: int,
    delay: float,
    min_len_val: int,
    output_path: str,
    ui_proxy_enabled: bool,
    ui_proxy_url: Optional[str],
    ui_use_env_proxies: bool,
    dataset_format: str,
) -> None:
    """Run the Stack Exchange scraper in a worker thread and integrate results into the UI."""
    def log(msg: str):
        log_view.controls.append(ft.Text(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"))
    await safe_update(page)

    # Apply proxy settings from UI (overrides env/defaults)
    try:
        pmsg = apply_proxy_from_ui(bool(ui_proxy_enabled), ui_proxy_url, bool(ui_use_env_proxies))
        log(pmsg)
    except Exception:
        pass

    prog.value = 0
    labels.get("threads").value = "Pages processed: 0"
    pairs_output = (str(dataset_format or "ChatML").strip().lower() == "standard")
    labels.get("pairs").value = ("Pairs Found: 0" if pairs_output else "Conversations Found: 0")
    await safe_update(page)

    log(f"Starting StackExchange scrape (site={site}, max_pairs={max_pairs})...")
    await safe_update(page)

    # Kick off the blocking scraper in a background thread with cancellation support
    fut = asyncio.create_task(asyncio.to_thread(
        sx.scrape,
        site=site or "stackoverflow",
        max_pairs=int(max_pairs),
        delay=float(delay),
        min_len=int(min_len_val),
        cancel_cb=lambda: bool(cancel_flag.get("cancelled")),
    ))

    # Progress pulse and cooperative cancellation monitor
    async def pulse_and_watch():
        try:
            while not fut.done():
                # Nothing to force-cancel; cancel_cb will be polled in the worker
                cur = (prog.value or 0.0)
                cur = 0.4 if cur >= 0.9 else (cur + 0.04)
                prog.value = cur
                await safe_update(page)
                await asyncio.sleep(0.35)
        except Exception:
            # Ignore UI pulse errors
            pass

    pulse_task = asyncio.create_task(pulse_and_watch())

    try:
        results = await fut
    except Exception as e:
        if cancel_flag.get("cancelled"):
            log("Scrape cancelled by user.")
            page.snack_bar = ft.SnackBar(ft.Text("Scrape cancelled "))
            page.open(page.snack_bar)
        else:
            log(f"StackExchange scrape failed: {e}")
            page.snack_bar = ft.SnackBar(ft.Text(f"StackExchange scrape failed: {e}"))
            page.open(page.snack_bar)
        await safe_update(page)
        try:
            pulse_task.cancel()
        except Exception:
            pass
        return
    finally:
        try:
            pulse_task.cancel()
        except Exception:
            pass

    # Completed
    prog.value = 1.0
    await safe_update(page)

    # Write output according to dataset format and build preview
    count = 0
    preview_pairs: List[Tuple[str, str]] = []
    try:
        if pairs_output:
            payload = results or []
        else:
            payload = pairs_to_chatml(results or [])
        await asyncio.to_thread(
            lambda: open(output_path, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=4))
        )
        if pairs_output:
            log(f"Wrote {len(payload)} pairs to {output_path}")
            count = len(payload)
            head = payload[:10]
            preview_pairs = [((ex.get("input", "") or ""), (ex.get("output", "") or "")) for ex in head]
        else:
            log(f"Wrote {len(payload)} conversations to {output_path}")
            count = len(payload)
            head = payload[:10]
            for c in head:
                try:
                    msgs = c.get("messages", []) or []
                    # Prefer user→assistant; fallback to first two non-empty messages
                    user_text = None
                    assistant_text = None
                    for m in msgs:
                        if not isinstance(m, dict):
                            continue
                        role = m.get("role")
                        text = (m.get("content") or "").strip()
                        if not text:
                            continue
                        if role == "user" and user_text is None:
                            user_text = text
                        elif role == "assistant" and user_text is not None:
                            assistant_text = text
                            break
                    if not (user_text and assistant_text):
                        texts = [(m.get("content") or "").strip() for m in msgs if isinstance(m, dict) and (m.get("content") or "").strip()]
                        if len(texts) >= 2:
                            user_text, assistant_text = texts[0], texts[1]
                        elif len(texts) == 1:
                            user_text, assistant_text = texts[0], ""
                    if user_text or assistant_text:
                        preview_pairs.append((user_text or "", assistant_text or ""))
                except Exception:
                    pass
        if not preview_pairs:
            # Fallback: read from saved file and derive a few pairs
            try:
                loaded = await asyncio.to_thread(lambda: json.load(open(output_path, "r", encoding="utf-8")))
                if isinstance(loaded, list) and loaded:
                    tmp: List[tuple[str, str]] = []
                    if pairs_output:
                        for ex in loaded[:10]:
                            if isinstance(ex, dict):
                                tmp.append(((ex.get("input", "") or ""), (ex.get("output", "") or "")))
                    else:
                        for rec in loaded[:10]:
                            if not isinstance(rec, dict):
                                continue
                            msgs = rec.get("messages", []) or []
                            u = None
                            a = None
                            for m in msgs:
                                if not isinstance(m, dict):
                                    continue
                                role = m.get("role")
                                text = (m.get("content") or "").strip()
                                if not text:
                                    continue
                                if role == "user" and u is None:
                                    u = text
                                elif role == "assistant" and u is not None:
                                    a = text
                                    break
                            if not (u and a):
                                texts = [(m.get("content") or "").strip() for m in msgs if isinstance(m, dict) and (m.get("content") or "").strip()]
                                if len(texts) >= 2:
                                    u, a = texts[0], texts[1]
                                elif len(texts) == 1:
                                    u, a = texts[0], ""
                            if u or a:
                                tmp.append((u or "", a or ""))
                    preview_pairs = tmp or preview_pairs
            except Exception:
                pass
        if not preview_pairs:
            preview_pairs = [("(no preview)", "")]
    except Exception as e:
        log(f"Failed to write results: {e}")
    await safe_update(page)

    labels.get("pairs").value = (f"Pairs Found: {count}" if pairs_output else f"Conversations Found: {count}")
    labels.get("threads").value = f"Questions processed: {count}"

    # Populate preview grid
    try:
        preview_host.controls.clear()
        lfx, rfx = compute_two_col_flex(preview_pairs)
        # Header labels depend on dataset format
        hdr_left = "Input" if pairs_output else "User"
        hdr_right = "Output" if pairs_output else "Assistant"
        preview_host.controls.append(two_col_header(hdr_left, hdr_right, left_flex=lfx, right_flex=rfx))
        for a, b in preview_pairs:
            preview_host.controls.append(two_col_row(a, b, lfx, rfx))
    except Exception as e:
        log(f"Failed to render preview: {e}")
    page.snack_bar = ft.SnackBar(ft.Text("Scrape complete! "))
    page.open(page.snack_bar)
    await safe_update(page)


async def run_reddit_scrape(
    page: ft.Page,
    log_view: ft.ListView,
    prog: ft.ProgressBar,
    labels: dict,
    preview_host: ft.ListView,
    cancel_flag: dict,
    url: str,
    max_posts: int,
    delay: float,
    min_len_val: int,
    output_path: str,
    multiturn: bool,
    ctx_k: int,
    ctx_max_chars: Optional[int],
    merge_same_id: bool,
    require_question: bool,
    ui_proxy_enabled: bool,
    ui_proxy_url: Optional[str],
    ui_use_env_proxies: bool,
    dataset_format: str,
) -> None:
    """Run the Reddit scraper in a worker thread and integrate results into the UI."""
    def log(msg: str):
        log_view.controls.append(ft.Text(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"))
    await safe_update(page)

    if rdt is None:
        log("Reddit scraper module not available.")
        page.snack_bar = ft.SnackBar(ft.Text("Reddit scraper module not available — please install or enable it."))
        page.open(page.snack_bar)
        await safe_update(page)
        return

    # Configure reddit module from UI inputs
    try:
        # Reset run-scoped counters to make stop caps reliable
        rdt.START_TS = time.time()
        rdt.REQUESTS_MADE = 0
        rdt.MAX_REQUESTS_TOTAL = None
        rdt.STOP_AFTER_SECONDS = None

        rdt.DEFAULT_URL = (url or rdt.DEFAULT_URL).strip()
        rdt.MAX_POSTS = int(max_posts)
        rdt.REQUEST_DELAY = max(0.0, float(delay))
        # Always build dataset; we'll copy pairs to desired path
        rdt.BUILD_DATASET = True
        rdt.PAIRING_MODE = "contextual" if bool(multiturn) else "parent_child"
        rdt.CONTEXT_K = int(ctx_k)
        rdt.MAX_INPUT_CHARS = None if (ctx_max_chars is None) else int(ctx_max_chars)
        rdt.MERGE_SAME_AUTHOR = bool(merge_same_id)
        rdt.REQUIRE_QUESTION = bool(require_question)
        rdt.MIN_LEN = max(0, int(min_len_val))
        # Keep dumps temp so we can clean after copying pairs
        rdt.OUTPUT_DIR = None
        rdt.USE_TEMP_DUMP = True
    except Exception as e:
        log(f"Invalid Reddit configuration: {e}")
        page.snack_bar = ft.SnackBar(ft.Text(f"Invalid Reddit configuration: {e}"))
        page.open(page.snack_bar)
        await safe_update(page)
        return

    # Apply proxy settings from UI (overrides env/defaults)
    try:
        pmsg = apply_proxy_from_ui(bool(ui_proxy_enabled), ui_proxy_url, bool(ui_use_env_proxies))
        log(pmsg)
    except Exception:
        pass

    prog.value = 0
    labels.get("threads").value = "Threads Visited: 0"
    pairs_output = (str(dataset_format or "ChatML").strip().lower() == "standard")
    labels.get("pairs").value = ("Pairs Found: 0" if pairs_output else "Conversations Found: 0")
    await safe_update(page)

    log("Starting Reddit scrape...")
    await safe_update(page)

    # Kick off the blocking scraper in a background thread
    fut = asyncio.create_task(asyncio.to_thread(rdt.run))

    # A soft progress pulse and cooperative cancellation monitor
    async def pulse_and_watch():
        try:
            while not fut.done():
                if cancel_flag.get("cancelled"):
                    # Force the worker to abort on next HTTP call
                    rdt.MAX_REQUESTS_TOTAL = 0
                    rdt.STOP_AFTER_SECONDS = 0
                # Pulse progress to indicate activity
                cur = (prog.value or 0.0)
                cur = 0.4 if cur >= 0.9 else (cur + 0.04)
                prog.value = cur
                await safe_update(page)
                await asyncio.sleep(0.35)
        except Exception:
            # Ignore UI pulse errors
            pass

    pulse_task = asyncio.create_task(pulse_and_watch())

    base_out = None
    pairs_src = None
    try:
        base_out, pairs_src = await fut
    except Exception as e:
        if cancel_flag.get("cancelled"):
            log("Scrape cancelled by user.")
            page.snack_bar = ft.SnackBar(ft.Text("Scrape cancelled "))
            page.open(page.snack_bar)
        else:
            log(f"Reddit scrape failed: {e}")
            page.snack_bar = ft.SnackBar(ft.Text(f"Reddit scrape failed: {e}"))
            page.open(page.snack_bar)
        await safe_update(page)
        try:
            pulse_task.cancel()
        except Exception:
            pass
        return
    finally:
        try:
            pulse_task.cancel()
        except Exception:
            pass

    # Completed
    prog.value = 1.0
    await safe_update(page)

    # Build output (ChatML conversations or Standard pairs), write to output, and build preview
    conv_count = 0
    pairs_count = 0
    chatml_convs: List[dict] = []
    standard_pairs: List[dict] = []
    sample_pairs: List[Tuple[str, str]] = []
    try:
        if bool(multiturn):
            # Build multi-turn conversations directly from scraped Reddit threads
            idx_path = os.path.join(str(base_out), "index.json") if base_out else None
            threads_count = 0
            if idx_path and os.path.exists(idx_path):
                idx = await asyncio.to_thread(lambda: json.load(open(idx_path, "r", encoding="utf-8")))
                for item in (idx.get("posts") or []):
                    rel_json = item.get("json")
                    if not rel_json:
                        continue
                    th_path = os.path.join(str(base_out), rel_json)
                    if not os.path.exists(th_path):
                        continue
                    # Load thread and convert to multi-turn ChatML
                    thread = await asyncio.to_thread(lambda: json.load(open(th_path, "r", encoding="utf-8")))
                    convs = reddit_thread_to_chatml_conversations(
                        thread,
                        min_len=max(0, int(min_len_val)),
                        k=int(ctx_k),
                        max_rounds_per_conv=6,
                        max_chars=(int(ctx_max_chars) if (ctx_max_chars is not None) else None),
                        merge_same_author=bool(merge_same_id),
                    )
                    if convs:
                        chatml_convs.extend(convs)
                        threads_count += 1
            conv_count = len(chatml_convs)
            if threads_count == 0 and conv_count == 0:
                log("No conversations constructed from threads; falling back to pairs conversion if available.")
                # Optional fallback to pairs if present
                if pairs_src is not None and os.path.exists(str(pairs_src)):
                    data = await asyncio.to_thread(lambda: json.load(open(str(pairs_src), "r", encoding="utf-8")))
                    if isinstance(data, list):
                        if pairs_output:
                            standard_pairs = data
                            pairs_count = len(standard_pairs)
                        else:
                            chatml_convs = pairs_to_chatml(data)
                            conv_count = len(chatml_convs)
        else:
            # Single-turn mode
            if pairs_src is not None and os.path.exists(str(pairs_src)):
                data = await asyncio.to_thread(lambda: json.load(open(str(pairs_src), "r", encoding="utf-8")))
                if isinstance(data, list):
                    if pairs_output:
                        standard_pairs = data
                        pairs_count = len(standard_pairs)
                    else:
                        chatml_convs = pairs_to_chatml(data)
                        conv_count = len(chatml_convs)
            else:
                log("No pairs JSON produced (pairs_src missing).")

        # If Standard selected and we have conversations, convert to first user→assistant pairs
        if pairs_output and not standard_pairs and chatml_convs:
            for conv in chatml_convs:
                try:
                    msgs = conv.get("messages", []) or []
                    user_text = None
                    assistant_text = None
                    for m in msgs:
                        role = m.get("role")
                        text = m.get("content") or ""
                        if role == "user" and user_text is None and text:
                            user_text = text
                        elif role == "assistant" and user_text is not None and text:
                            assistant_text = text
                            break
                    if user_text and assistant_text:
                        standard_pairs.append({"input": user_text, "output": assistant_text})
                except Exception:
                    pass
            pairs_count = len(standard_pairs)

        # Write to desired output path (ChatML or Standard)
        dest = output_path or "scraped_training_data.json"
        dest_abs = os.path.abspath(dest)
        os.makedirs(os.path.dirname(dest_abs) or ".", exist_ok=True)
        payload = chatml_convs if not pairs_output else standard_pairs
        await asyncio.to_thread(
            lambda: open(dest_abs, "w", encoding="utf-8").write(
                json.dumps(payload, ensure_ascii=False, indent=4)
            )
        )
        if pairs_output:
            log(f"Wrote {pairs_count} pairs to: {dest_abs}")
        else:
            log(f"Wrote {conv_count} conversations to: {dest_abs}")

        # Build preview
        if pairs_output:
            for ex in (standard_pairs or [])[:10]:
                a = (ex.get("input", "") or "") if isinstance(ex, dict) else ""
                b = (ex.get("output", "") or "") if isinstance(ex, dict) else ""
                sample_pairs.append((a, b))
        else:
            # From ChatML: prefer user→assistant; fallback to first two non-empty messages
            for conv in chatml_convs[:10]:
                try:
                    msgs = conv.get("messages", []) or []
                    user_text = None
                    assistant_text = None
                    for m in msgs:
                        if not isinstance(m, dict):
                            continue
                        role = m.get("role")
                        text = (m.get("content") or "").strip()
                        if not text:
                            continue
                        if role == "user" and user_text is None:
                            user_text = text
                        elif role == "assistant" and user_text is not None:
                            assistant_text = text
                            break
                    if not (user_text and assistant_text):
                        texts = [(m.get("content") or "").strip() for m in msgs if isinstance(m, dict) and (m.get("content") or "").strip()]
                        if len(texts) >= 2:
                            user_text, assistant_text = texts[0], texts[1]
                        elif len(texts) == 1:
                            user_text, assistant_text = texts[0], ""
                    if user_text or assistant_text:
                        sample_pairs.append((user_text or "", assistant_text or ""))
                except Exception:
                    pass
    except Exception as e:
        log(f"Failed to build output/write file: {e}")
    await safe_update(page)

    # Read index.json for post count (threads label)
    try:
        idx_path = os.path.join(str(base_out), "index.json") if base_out else None
        if idx_path and os.path.exists(idx_path):
            idx = await asyncio.to_thread(lambda: json.load(open(idx_path, "r", encoding="utf-8")))
            pc = int(idx.get("post_count") or 0)
            labels.get("threads").value = f"Posts processed: {pc}"
    except Exception:
        pass

    labels.get("pairs").value = (
        f"Pairs Found: {pairs_count}" if pairs_output else f"Conversations Found: {conv_count}"
    )

    # Populate preview grid from ChatML-derived sample pairs
    try:
        preview_host.controls.clear()
        if not sample_pairs:
            # Fallback: read from saved file
            try:
                loaded = await asyncio.to_thread(lambda: json.load(open(dest_abs, "r", encoding="utf-8")))
                if isinstance(loaded, list) and loaded:
                    tmp: List[tuple[str, str]] = []
                    if pairs_output:
                        for ex in loaded[:10]:
                            if isinstance(ex, dict):
                                tmp.append(((ex.get("input", "") or ""), (ex.get("output", "") or "")))
                    else:
                        for rec in loaded[:10]:
                            if not isinstance(rec, dict):
                                continue
                            msgs = rec.get("messages", []) or []
                            u = None
                            a = None
                            for m in msgs:
                                if not isinstance(m, dict):
                                    continue
                                role = m.get("role")
                                text = (m.get("content") or "").strip()
                                if not text:
                                    continue
                                if role == "user" and u is None:
                                    u = text
                                elif role == "assistant" and u is not None:
                                    a = text
                                    break
                            if not (u and a):
                                texts = [(m.get("content") or "").strip() for m in msgs if isinstance(m, dict) and (m.get("content") or "").strip()]
                                if len(texts) >= 2:
                                    u, a = texts[0], texts[1]
                                elif len(texts) == 1:
                                    u, a = texts[0], ""
                            if u or a:
                                tmp.append((u or "", a or ""))
                    sample_pairs = tmp or sample_pairs
            except Exception:
                pass
        if not sample_pairs:
            sample_pairs = [("(no preview)", "")]  # graceful empty state
        lfx, rfx = compute_two_col_flex(sample_pairs)
        hdr_left = "Input" if pairs_output else "User"
        hdr_right = "Output" if pairs_output else "Assistant"
        preview_host.controls.append(two_col_header(hdr_left, hdr_right, left_flex=lfx, right_flex=rfx))
        for a, b in sample_pairs:
            preview_host.controls.append(two_col_row(a, b, lfx, rfx))
    except Exception as e:
        log(f"Failed to render preview: {e}")
    await safe_update(page)

    # Cleanup temporary dump folder
    try:
        if base_out and os.path.isdir(str(base_out)) and getattr(rdt, "USE_TEMP_DUMP", False):
            shutil.rmtree(str(base_out), ignore_errors=True)
            log(f"Cleaned up temp dump: {base_out}")
    except Exception as e:
        log(f"Cleanup warning: {e}")
    page.snack_bar = ft.SnackBar(ft.Text("Scrape complete! "))
    page.open(page.snack_bar)
    await safe_update(page)


async def run_real_scrape(
    page: ft.Page,
    log_view: ft.ListView,
    prog: ft.ProgressBar,
    labels: dict,
    preview_host: ft.ListView,
    cancel_flag: dict,
    boards: List[str],
    max_threads: int,
    max_pairs_total: int,
    delay: float,
    min_len_val: int,
    output_path: str,
    multiturn: bool,
    ctx_strategy: str,
    ctx_k: int,
    ctx_max_chars: Optional[int],
    merge_same_id: bool,
    require_question: bool,
    ui_proxy_enabled: bool,
    ui_proxy_url: Optional[str],
    ui_use_env_proxies: bool,
    dataset_format: str,
) -> None:
    def log(msg: str):
        log_view.controls.append(ft.Text(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"))
    await safe_update(page)

    total_boards = len(boards)
    if total_boards == 0:
        return
    prog.value = 0
    pairs_accum: List[dict] = []
    conversations_accum: List[dict] = []
    chatml_enabled = bool(multiturn)
    pairs_output = (str(dataset_format or "ChatML").strip().lower() == "standard")
    # Initialize counters label
    labels.get("pairs").value = ("Pairs Found: 0" if pairs_output else "Conversations Found: 0")

    # Apply proxy settings from UI (overrides env/defaults)
    try:
        pmsg = apply_proxy_from_ui(bool(ui_proxy_enabled), ui_proxy_url, bool(ui_use_env_proxies))
        log(pmsg)
    except Exception:
        pass

    for idx, b in enumerate(boards, start=1):
        if cancel_flag.get("cancelled"):
            log("Scrape cancelled by user.")
            page.snack_bar = ft.SnackBar(ft.Text("Scrape cancelled "))
            page.open(page.snack_bar)
            await safe_update(page)
            return

        remaining = (max_pairs_total - (len(conversations_accum) if chatml_enabled else len(pairs_accum)))
        if remaining <= 0:
            break

        if chatml_enabled:
            # Build ChatML conversations by sampling threads round-robin across catalog pages
            mode_str = "chatml"
            log(f"Scraping /{b}/ (up to {remaining} conversations) — mode={mode_str}")
            try:
                pages = await asyncio.to_thread(sc.fetch_catalog_pages, b)
            except Exception as e:
                log(f"Error fetching catalog for /{b}/: {e}")
                await safe_update(page)
                continue

            # Round-robin selection of thread IDs up to max_threads
            thread_ids: List[int] = []
            rr_idx = [0] * len(pages)
            try:
                while len(thread_ids) < max_threads and any(i < len(pages[p]) for p, i in enumerate(rr_idx)):
                    for p_i in range(len(pages)):
                        if len(thread_ids) >= max_threads:
                            break
                        i2 = rr_idx[p_i]
                        if i2 < len(pages[p_i]):
                            thread_ids.append(pages[p_i][i2])
                            rr_idx[p_i] += 1
            except Exception:
                pass

            board_new = 0
            for tid in thread_ids:
                if (max_pairs_total - len(conversations_accum)) <= 0:
                    break
                try:
                    posts = await asyncio.to_thread(sc.fetch_thread, b, tid)
                except Exception:
                    posts = []
                if not posts:
                    continue
                # Build conversations for this thread
                convs = thread_to_chatml_conversations(
                    posts,
                    min_len=min_len_val,
                    k=ctx_k,
                    max_rounds_per_conv=6,
                    max_chars=ctx_max_chars,
                    merge_same_id=merge_same_id,
                    add_system=None,
                    ban_pattern=None,
                )
                if not convs:
                    await asyncio.sleep(delay)
                    continue
                # Respect remaining budget
                rem = max_pairs_total - len(conversations_accum)
                if rem <= 0:
                    break
                if len(convs) > rem:
                    convs = convs[:rem]
                conversations_accum.extend(convs)
                board_new += len(convs)
                labels.get("pairs").value = f"Conversations Found: {len(conversations_accum)}"
                await safe_update(page)
                await asyncio.sleep(delay)

            labels.get("threads").value = f"Boards processed: {idx}/{total_boards}"
            prog.value = idx / total_boards
            log(f"/{b}/ -> {board_new} conversations (total {len(conversations_accum)})")
            await safe_update(page)
        else:
            mode_str = "contextual" if bool(multiturn) else "normal"
            log(f"Scraping /{b}/ (up to {remaining} pairs) — mode={mode_str}")
            try:
                data = await asyncio.to_thread(
                    sc.scrape,
                    board=b,
                    max_threads=max_threads,
                    max_pairs=remaining,
                    delay=delay,
                    min_len=min_len_val,
                    mode=mode_str,
                    strategy=ctx_strategy,
                    k=ctx_k,
                    max_chars=ctx_max_chars,
                    merge_same_id=merge_same_id,
                    require_question=require_question,
                )
            except Exception as e:
                log(f"Error scraping /{b}/: {e}")
                await safe_update(page)
                continue

            pairs_accum.extend(data)
            labels.get("pairs").value = (
                f"Pairs Found: {len(pairs_accum)}" if pairs_output else f"Conversations Found: {len(pairs_accum)}"
            )
            labels.get("threads").value = f"Boards processed: {idx}/{total_boards}"
            prog.value = idx / total_boards
            log(f"/{b}/ -> {len(data)} pairs (total {len(pairs_accum)})")
            await safe_update(page)

    # Write JSON (ChatML conversations or Standard pairs)
    try:
        if pairs_output:
            if chatml_enabled:
                # Convert conversations to first user→assistant pairs
                std_pairs: List[dict] = []
                for conv in conversations_accum:
                    try:
                        msgs = conv.get("messages", []) or []
                        user_text = None
                        assistant_text = None
                        for m in msgs:
                            role = m.get("role")
                            text = m.get("content") or ""
                            if role == "user" and user_text is None and text:
                                user_text = text
                            elif role == "assistant" and user_text is not None and text:
                                assistant_text = text
                                break
                        if user_text and assistant_text:
                            std_pairs.append({"input": user_text, "output": assistant_text})
                    except Exception:
                        pass
                payload = std_pairs
            else:
                payload = pairs_accum
        else:
            if chatml_enabled:
                payload = conversations_accum
            else:
                payload = pairs_to_chatml(pairs_accum)
        await asyncio.to_thread(
            lambda: open(output_path, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=4))
        )
        log(
            (f"Wrote {len(payload)} pairs to {output_path}" if pairs_output else f"Wrote {len(payload)} conversations to {output_path}")
        )
    except Exception as e:
        log(f"Failed to write {output_path}: {e}")
        await safe_update(page)
        return

    # Populate preview with flex grid and scrollable cells
    preview_host.controls.clear()
    if pairs_output:
        # Standard: preview raw pairs
        head = []
        if chatml_enabled:
            # Convert conversations to first pairs just for preview
            for conv in (conversations_accum or [])[:10]:
                try:
                    msgs = conv.get("messages", []) or []
                    user_text = None
                    assistant_text = None
                    for m in msgs:
                        role = m.get("role")
                        text = m.get("content") or ""
                        if role == "user" and user_text is None and text:
                            user_text = text
                        elif role == "assistant" and user_text is not None and text:
                            assistant_text = text
                            break
                    if user_text and assistant_text:
                        head.append({"input": user_text, "output": assistant_text})
                except Exception:
                    pass
        else:
            head = pairs_accum[:10]
        sample_pairs = [(ex.get("input", "") or "", ex.get("output", "") or "") for ex in head]
        if not sample_pairs:
            # Fallback: read from saved file
            try:
                loaded = await asyncio.to_thread(lambda: json.load(open(output_path, "r", encoding="utf-8")))
                if isinstance(loaded, list) and loaded:
                    tmp: List[tuple[str, str]] = []
                    if pairs_output:
                        for ex in loaded[:10]:
                            if isinstance(ex, dict):
                                tmp.append(((ex.get("input", "") or ""), (ex.get("output", "") or "")))
                    else:
                        for rec in loaded[:10]:
                            if not isinstance(rec, dict):
                                continue
                            msgs = rec.get("messages", []) or []
                            u = None
                            a = None
                            for m in msgs:
                                if not isinstance(m, dict):
                                    continue
                                role = m.get("role")
                                text = (m.get("content") or "").strip()
                                if not text:
                                    continue
                                if role == "user" and u is None:
                                    u = text
                                elif role == "assistant" and u is not None:
                                    a = text
                                    break
                            if not (u and a):
                                texts = [(m.get("content") or "").strip() for m in msgs if isinstance(m, dict) and (m.get("content") or "").strip()]
                                if len(texts) >= 2:
                                    u, a = texts[0], texts[1]
                                elif len(texts) == 1:
                                    u, a = texts[0], ""
                            if u or a:
                                tmp.append((u or "", a or ""))
                    sample_pairs = tmp or sample_pairs
            except Exception:
                pass
        if not sample_pairs:
            sample_pairs = [("(no preview)", "")]
    else:
        # ChatML selected: derive preview from conversations.
        # If Multiturn was off, convert the accumulated pairs to single-turn ChatML first.
        conv_src = conversations_accum
        if not conv_src and pairs_accum:
            try:
                conv_src = pairs_to_chatml(pairs_accum)
            except Exception:
                conv_src = []
        sample_pairs = []
        for conv in (conv_src or [])[:10]:
            try:
                msgs = conv.get("messages", []) or []
                user_text = None
                assistant_text = None
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    role = m.get("role")
                    text = (m.get("content") or "").strip()
                    if not text:
                        continue
                    if role == "user" and user_text is None:
                        user_text = text
                    elif role == "assistant" and user_text is not None:
                        assistant_text = text
                        break
                if not (user_text and assistant_text):
                    texts = [(m.get("content") or "").strip() for m in msgs if isinstance(m, dict) and (m.get("content") or "").strip()]
                    if len(texts) >= 2:
                        user_text, assistant_text = texts[0], texts[1]
                    elif len(texts) == 1:
                        user_text, assistant_text = texts[0], ""
                if user_text or assistant_text:
                    sample_pairs.append((user_text or "", assistant_text or ""))
            except Exception:
                pass
        if not sample_pairs:
            sample_pairs = [("(no preview)", "")]

    lfx, rfx = compute_two_col_flex(sample_pairs)
    hdr_left = "Input" if pairs_output else "User"
    hdr_right = "Output" if pairs_output else "Assistant"
    preview_host.controls.append(two_col_header(hdr_left, hdr_right, left_flex=lfx, right_flex=rfx))
    for a, b in sample_pairs:
        preview_host.controls.append(two_col_row(a, b, lfx, rfx))
    page.snack_bar = ft.SnackBar(ft.Text("Scrape complete! 🎉"))
    page.open(page.snack_bar)
    await safe_update(page)
