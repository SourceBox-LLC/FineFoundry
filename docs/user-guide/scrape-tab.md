# Scrape Tab

The Scrape tab lets you collect conversational training data from multiple sources and prepare it as `input` / `output` pairs for later building, analysis, and training.

**Supported data sources:**

- **4chan** — Scrape boards with adjacent or contextual pairing strategies
- **Reddit** — Scrape subreddits or single posts with comment expansion
- **Stack Exchange** — Q&A pairs from Stack Overflow and other sites
- **Synthetic** — Generate training data from your own documents using local LLMs

Use this tab to:

- Select a data source and configure parameters
- Choose pairing strategies (for network sources) or generation types (for synthetic)
- Monitor progress and logs in real time
- Preview the resulting dataset

All scraped data is automatically saved to the database.

![Scrape Tab](../../img/ff_scrape_tab.png)

______________________________________________________________________

## Overview

Typical workflow:

1. Select a **data source** from the dropdown (4chan, Reddit, Stack Exchange, or Synthetic).
1. Configure source-specific parameters.
1. Click **Start** and watch the progress / logs.
1. When finished, preview the dataset in the two-column grid.
1. Use the resulting database session in the **Build / Publish**, **Merge**, and **Training** tabs.

Data is stored in the database using the standard schema:

```json
[
  {"input": "...", "output": "..."}
]
```

Or in ChatML format:

```json
[
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
]
```

You can export data to JSON via the database helpers if needed for external tools.

______________________________________________________________________

## Layout at a Glance

### 1. Source Selection

- **Source dropdown** — Choose between 4chan, Reddit, Stack Exchange, or Synthetic
- **Source-specific controls** — Each source shows relevant configuration options

### 2. Source-Specific Configuration

#### 4chan

- **Boards list** — Multi-select chips (e.g., `pol`, `b`, `x`) with Select All / Clear
- **Mode** — `normal` (adjacent posts) or `contextual`
- **Strategy** (contextual only) — `quote_chain`, `cumulative`, or `last_k`

#### Reddit

- **URL** — Subreddit or single post URL
- **Max Posts** — Number of posts to scrape

#### Stack Exchange

- **Site** — Stack Overflow, Super User, etc.
- **Max Pairs** — Target number of Q&A pairs

#### Synthetic

- **Files/URLs** — Add PDFs, DOCX, TXT, HTML files or URLs
- **Model** — Local LLM to use (default: `unsloth/Llama-3.2-3B-Instruct`)
- **Generation Type** — `qa` (Q&A pairs), `cot` (chain-of-thought), or `summary`
- **Num Pairs** — Target examples per chunk
- **Max Chunks** — Maximum document chunks to process
- **Curate** — Enable quality filtering with threshold

### 3. Common Parameters

- **Dataset Format** — Standard (input/output) or ChatML (messages array)
- **Delay (s)** — Polite delay between HTTP requests (for network sources)
- **Min Length** — Minimum character count per side for a pair to be kept

All scraped data is automatically saved to the database as a new session.

### 4. Progress & Logs

- **Status line** — High-level state (idle, scraping, completed, error)
- **Progress bar / counters** — Approximate thread/pair progress
- **Logs panel** — Streaming log messages from the selected data source

### 5. Output & Preview

- **Preview grid** — Two-column preview of the first N `input` / `output` pairs so you can inspect data before building or training
- All data is automatically saved to the database as a new scrape session

______________________________________________________________________

## Usage Examples

### Example 1: Quick 4chan scrape

1. Select **4chan** as the source.
1. Select boards like `pol` and `b`.
1. Set:
   - Max Threads: `50`
   - Max Pairs: `500`
   - Delay: `0.5`
   - Min Length: `10`
1. Leave **Mode** as `normal` (adjacent pairs).
1. Click **Start** and preview the results when done.

### Example 2: Contextual quote-chain dataset

1. Select **4chan** and a conversation-heavy board (e.g., `pol`).
1. Set:
   - Mode: `contextual`
   - Strategy: `quote_chain`
   - K: `6`
   - Max Input Chars: `2000`
1. Enable "require question" / "merge same author" options as desired.
1. Start the scrape and inspect the preview to confirm multi-turn context.

### Example 3: Reddit subreddit scrape

1. Select **Reddit** as the source.
1. Enter a subreddit URL (e.g., `https://www.reddit.com/r/LocalLLaMA/`).
1. Set Max Posts to `50`.
1. Click **Start** and watch the logs as posts and comments are fetched.

### Example 4: Synthetic data from PDF

1. Select **Synthetic** as the source.
1. Click **Browse** and select a PDF document (research paper, manual, etc.).
1. Configure:
   - Model: `unsloth/Llama-3.2-3B-Instruct` (default)
   - Generation Type: `qa`
   - Num Pairs: `25`
   - Max Chunks: `10`
1. Click **Start** — a snackbar appears immediately while the model loads.
1. Watch live progress as the document is chunked and Q&A pairs are generated.
1. Preview the generated pairs in the two-column grid.

**Note**: First run takes 30-60 seconds for model loading. Subsequent runs are faster.

______________________________________________________________________

## Tips & Best Practices

- **Start small** — Validate configuration with smaller runs before scaling up.
- **Use contextual mode** — For conversational context rather than single-turn QA (4chan/Reddit).
- **Watch the logs** — Monitor for network issues, rate limiting, or parsing errors.
- **Min Length filter** — Reduce low-signal or spammy pairs.
- **Synthetic generation** — Use high-quality source documents for better results.
- **Model choice** — Larger models produce better synthetic data but require more VRAM.
- **After generation** — Use **Build / Publish** and **Dataset Analysis** to prepare your data before training.

______________________________________________________________________

## Offline Mode

When **Offline Mode** is enabled:

- The Source dropdown remains visible, but **all non-synthetic sources** are disabled.
- The current source is forced to **Synthetic**.
- If you try to start a scrape while Offline Mode is enabled and the source is not Synthetic, the app blocks the run and shows a snackbar:
  - "Offline mode is enabled. Only the Synthetic data source can be used."

The UI also shows an Offline banner at the top of the tab and an inline helper text under the Source dropdown explaining the restriction.

______________________________________________________________________

## Related Topics

- [Build & Publish Tab](build-publish-tab.md) – turn scraped data into train/val/test splits.
- [Merge Datasets Tab](merge-tab.md) – combine multiple scraped datasets.
- [Analysis Tab](analysis-tab.md) – inspect dataset quality before training.
- [Training Tab](training-tab.md) – fine-tune models on your prepared dataset.
- [Quick Start Guide](quick-start.md) – end-to-end overview.

______________________________________________________________________

**Next**: [Build & Publish Tab](build-publish-tab.md) | **Previous**: [Quick Start Guide](quick-start.md) | [Back to Documentation Index](../README.md)
