# Scrape Tab

The Scrape tab lets you collect conversational training data from sources like **4chan**, and prepare it as `input` / `output` pairs for later building, analysis, and training.

Use this tab to:

- Configure which boards to scrape and with what parameters
- Choose **normal** vs **contextual** pairing strategies
- Monitor scraping progress and logs in real time
- Quickly preview the resulting JSON dataset

![Scrape Tab](../../img/ff_scrape_tab.png)

---

## Overview

Typical workflow:

1. Select one or more **boards** to scrape.
2. Tune **Max Threads**, **Max Pairs**, **Delay**, and other parameters.
3. Choose a pairing **mode** (normal vs contextual) and strategy (e.g. quote_chain).
4. Click **Start Scrape** and watch the progress / logs.
5. When finished, use **Preview Dataset** to inspect the pairs.
6. Feed the resulting JSON into the **Build / Publish** and **Training** tabs.

The output is a JSON file like:

```json
[
  {"input": "...", "output": "..."}
]
```

---

## Layout at a Glance

### 1. Source & Boards

- **Boards list**
  - Multi-select chips for 4chan boards (e.g., `pol`, `b`, `x`).
  - Includes convenience actions like **Select All** / **Clear**.
- **Target JSON file**
  - Path to the output JSON file (default is usually `scraped_training_data.json`).

### 2. Parameters

Core scraping / pairing parameters:

- **Max Threads** – Number of threads per board to sample; higher values increase coverage.
- **Max Pairs** – Upper bound on how many `input` / `output` pairs to extract.
- **Delay (s)** – Polite delay between HTTP requests.
- **Min Length** – Minimum character count per side for a pair to be kept.
- **Mode** – `normal` (adjacent posts) or `contextual`.
- **Strategy** (contextual only):
  - `quote_chain`, `cumulative`, or `last_k`.
- **K** – Context depth for contextual mode.
- **Max Input Chars** – Optional truncation of long contexts.

### 3. Progress & Logs

- **Status line** – High-level state (idle, scraping, completed, error).
- **Progress bar / counters** – Approximate thread/pair progress.
- **Logs panel** – Streaming log messages from the scraper.

### 4. Output & Preview

- **Output path** – Where the JSON file will be written.
- **Preview button** – Opens a two-column preview of the first N `input` / `output` pairs so you can inspect data before building or training.

---

## Usage Examples

### Example 1: Quick sanity-check scrape

1. Select `pol` and `b`.
2. Set:
   - Max Threads: `50`
   - Max Pairs: `500`
   - Delay: `0.5`
   - Min Length: `10`
3. Leave **Mode** as `normal` (adjacent pairs).
4. Click **Start Scrape**.
5. When done, click **Preview Dataset** to inspect a handful of pairs.

### Example 2: Contextual quote-chain dataset

1. Select a conversation-heavy board (e.g., `pol`).
2. Set:
   - Mode: `contextual`
   - Strategy: `quote_chain`
   - K: `6`
   - Max Input Chars: `2000`
3. Enable any "require question" / "merge same author" options as desired.
4. Start the scrape and inspect the preview to confirm that `input` contains multi-turn context.

### Example 3: Longer, slower crawl

- Increase **Max Threads** and **Max Pairs** for deeper coverage.
- Increase **Delay** slightly to be respectful of API rate limits.
- Leave the job running while you work elsewhere, then return to preview the final dataset.

---

## Tips & Best Practices

- Start with **smaller** runs to validate configuration; increase Max Threads / Max Pairs later.
- Use **contextual** mode when you care about conversational context rather than single-turn QA.
- Watch the **logs panel** for network issues, rate limiting, or parsing errors.
- Use **Min Length** and mode settings to reduce low-signal / spammy pairs.
- After scraping, pass the JSON file through the **Build / Publish** and **Dataset Analysis** tabs before training.

---

## Related Topics

- [Build & Publish Tab](build-publish-tab.md) – turn raw JSON into train/val/test splits.
- [Merge Datasets Tab](merge-tab.md) – combine multiple scraped datasets.
- [Analysis Tab](analysis-tab.md) – inspect dataset quality before training.
- [Training Tab](training-tab.md) – fine-tune models on your prepared dataset.
- [Quick Start Guide](quick-start.md) – end-to-end overview.

---

**Next**: [Build & Publish Tab](build-publish-tab.md) | **Previous**: [Quick Start Guide](quick-start.md) | [Back to Documentation Index](../README.md)
