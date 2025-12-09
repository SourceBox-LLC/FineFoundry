# CLI Usage

FineFoundry includes command‑line tools in addition to the GUI. The CLI is useful when you want to:

- Automate dataset builds in scripts or CI.
- Run scrapers on a schedule.
- Reproduce a known‑good configuration without clicking through the UI.

This page focuses on three common CLI entry points:

- `src/save_dataset.py` – build and (optionally) push a dataset to the Hugging Face Hub.
- `src/scrapers/reddit_scraper.py` – crawl Reddit and build conversational pairs.
- `src/synthetic_cli.py` – generate synthetic training data from documents and URLs.

For general installation and a first end‑to‑end run, start with the **[Quick Start Guide](quick-start.md)** and GUI tab guides.

______________________________________________________________________

## Prerequisites

Before using the CLI tools, make sure you have:

- Python **3.10+**
- This repository cloned and installed (see **[Installation](installation.md)**)
- Optional: a Hugging Face token for pushing datasets (see **[Authentication](authentication.md)**)

All examples assume you are running commands from the project root.

______________________________________________________________________

## Dataset build & push: `src/save_dataset.py`

`src/save_dataset.py` is a scriptable path for turning a JSON file of pairs into a Hugging Face dataset, with optional push and dataset card generation.

There are **no CLI flags** – you configure a small block of constants in the file header, then run the script.

### 1. Prepare your JSON file

You should have a JSON file with a list of input/output pairs, for example:

```json
[
  {"input": "...", "output": "..."}
]
```

The default flow in FineFoundry saves scraped data to `scraped_training_data.json` in this format.

### 2. Configure `src/save_dataset.py`

Open `src/save_dataset.py` and edit the configuration block near the top:

```python
DATA_FILE = "scraped_training_data.json"
SAVE_DIR = "hf_dataset"
SEED = 42
SHUFFLE = True
VAL_SIZE = 0.01
TEST_SIZE = 0.0
MIN_LEN = 1
PUSH_TO_HUB = True
REPO_ID = "username/my-dataset"
PRIVATE = True
HF_TOKEN = None  # if None, uses env HF_TOKEN or cached login
```

Key points:

- `DATA_FILE` points to your JSON file.
- `SAVE_DIR` is where the `datasets.DatasetDict` will be saved locally.
- `VAL_SIZE` / `TEST_SIZE` control the split fractions.
- `PUSH_TO_HUB`, `REPO_ID`, `PRIVATE`, and `HF_TOKEN` control whether/how you push to the Hub.
- If `HF_TOKEN` is `None`, FineFoundry will fall back to environment variables or cached CLI login (see **[Authentication](authentication.md)**).

### 3. Run the script

From the project root:

```bash
python src/save_dataset.py
```

This will:

- Read `DATA_FILE`.
- Build a `datasets.DatasetDict` with the configured splits.
- Save the dataset to `SAVE_DIR`.
- Optionally push to `REPO_ID` on the Hugging Face Hub and upload a dataset card as `README.md`.

If you intend to use this in CI, wrap the command in your workflow and provide the token via environment variables or a secure secret store.

______________________________________________________________________

## Reddit scraper CLI: `src/scrapers/reddit_scraper.py`

The Reddit scraper provides a CLI for crawling subreddits or individual posts and building conversation pairs.

### Basic crawl example

```bash
python src/scrapers/reddit_scraper.py \
  --url https://www.reddit.com/r/AskReddit/ \
  --max-posts 50 \
  --mode contextual --k 4 --max-input-chars 2000 \
  --pairs-path reddit_pairs.json --cleanup
```

This will:

- Crawl up to 50 posts from `r/AskReddit`.
- Build contextual pairs (using the last `k` posts as context, up to `max-input-chars`).
- Save pairs to `reddit_pairs.json`.
- Clean up the intermediate dump directory when finished.

### Important options (general crawl)

- `--url` – subreddit or post URL to crawl.
- `--max-posts` – maximum posts to process.
- `--request-delay` – delay between requests (seconds).
- `--request-jitter-frac` – random jitter fraction applied to the delay.
- `--max-requests` – hard cap on total HTTP requests (`0` = off).
- `--stop-after-seconds` – wall‑clock time limit (`0` = off).
- `--output-dir` – where to store the raw dump (`reddit_dump_<slug>/` by default).
- `--use-temp-dump` – use a temporary dump directory.
- `--no-expand-more` – disable expansion of “more comments”.

### Important options (dataset build)

- `--build-dataset` – enable dataset building (on by default).
- `--mode {parent_child,contextual}` – pairing strategy.
- `--k` – context depth for contextual mode.
- `--max-input-chars` – truncate context to this many characters (`0` = off).
- `--require-question` – keep only pairs where context looks like a question.
- `--no-merge-same-author` – disable merging consecutive messages from the same author.
- `--min-len` – minimum characters per side.
- `--include-automod` – include AutoModerator posts.
- `--pairs-path` – stable path to copy the final pairs JSON.
- `--cleanup` – delete the dump folder after copying pairs.

Proxy behavior is controlled in `src/scrapers/reddit_scraper.py` via `PROXY_URL` and `USE_ENV_PROXIES`. For an overview of proxy configuration, see **[Proxy Setup](../deployment/proxy-setup.md)**.

### Single-post example

To build pairs from a single Reddit post:

```bash
python src/scrapers/reddit_scraper.py \
  --url https://www.reddit.com/r/AskReddit/comments/abc123/example_post/ \
  --mode parent_child --pairs-path reddit_pairs.json
```

This ignores `--max-posts` and focuses on the given post/thread.

______________________________________________________________________

## When to use CLI vs GUI

Use the **GUI** when you want to:

- Explore boards, parameters, and datasets interactively.
- Iterate on scraping and dataset settings with visual feedback.
- Manage training runs and inference from a single desktop app.

Use the **CLI** when you want to:

- Schedule scrapes or dataset builds as cron jobs.
- Integrate scraping and dataset building into larger Python workflows.
- Reproduce the same configuration across machines or CI.

For a complete overview of all tabs and features, see the **[GUI Overview](gui-overview.md)** and the tab‑specific guides.

______________________________________________________________________

## Synthetic data generation: `src/synthetic_cli.py`

The synthetic CLI generates training data from documents (PDF, DOCX, TXT, HTML) and URLs using Unsloth's SyntheticDataKit.

### Basic example

```bash
python src/synthetic_cli.py --source document.pdf --output qa_pairs.json
```

This will:

- Load the specified model (default: `unsloth/Llama-3.2-3B-Instruct`).
- Ingest and chunk the document.
- Generate Q&A pairs from each chunk.
- Save the combined dataset to `qa_pairs.json`.

### Multiple sources

```bash
python src/synthetic_cli.py \
  --source paper.pdf \
  --source notes.txt \
  --source https://example.com/article \
  --output combined_data.json
```

### Generation types

- `--type qa` – Generate question-answer pairs (default).
- `--type cot` – Generate chain-of-thought reasoning examples.
- `--type summary` – Generate summaries.

```bash
python src/synthetic_cli.py --source paper.pdf --type cot --output cot_data.json
```

### Quality curation

Enable Llama-as-judge curation to filter low-quality pairs:

```bash
python src/synthetic_cli.py \
  --source document.pdf \
  --curate --threshold 8.0 \
  --output curated_pairs.json
```

### Important options

- `--source`, `-s` – Source file or URL (can be specified multiple times).
- `--output`, `-o` – Output JSON file path (default: `synthetic_data.json`).
- `--type`, `-t` – Generation type: `qa`, `cot`, or `summary` (default: `qa`).
- `--num-pairs`, `-n` – Pairs to generate per chunk (default: 25).
- `--max-chunks` – Maximum chunks to process per source (default: 10).
- `--model`, `-m` – Model to use (default: `unsloth/Llama-3.2-3B-Instruct`).
- `--curate` – Enable quality curation.
- `--threshold` – Curation quality threshold 1-10 (default: 7.5).
- `--format`, `-f` – Output format: `chatml` or `standard` (default: `chatml`).
- `--multimodal` – Enable multimodal processing for images.
- `--no-db` – Don't save results to the FineFoundry database.
- `--quiet`, `-q` – Quiet mode (JSON summary output only).
- `--verbose`, `-v` – Verbose mode (detailed debug output).
- `--config` – Load options from a YAML config file.
- `--keep-server` – Keep vLLM server running after generation (faster for batch runs).

### Config file

Use a YAML config file to avoid repeating options:

```yaml
# synthetic_config.yaml
sources:
  - document.pdf
  - https://example.com/article
output: synthetic_data.json
type: qa
num_pairs: 25
max_chunks: 10
model: unsloth/Llama-3.2-3B-Instruct
curate: false
threshold: 7.5
format: chatml
```

```bash
uv run python src/synthetic_cli.py --config synthetic_config.yaml
```

CLI arguments override config file values.

### Batch processing with model caching

For processing multiple batches, use `--keep-server` to avoid reloading the model:

```bash
# First run - loads model and keeps server running
uv run python src/synthetic_cli.py --source batch1.pdf --output batch1.json --keep-server

# Subsequent runs reuse the running server (much faster)
uv run python src/synthetic_cli.py --source batch2.pdf --output batch2.json --keep-server

# Final run - let it clean up
uv run python src/synthetic_cli.py --source batch3.pdf --output batch3.json
```

### Quiet mode for scripting

Use `--quiet` for machine-readable output:

```bash
uv run python src/synthetic_cli.py --source doc.pdf --quiet
# Output: {"success": true, "output": "synthetic_data.json", "pairs": 125, ...}
```

### Verbose mode for debugging

Use `--verbose` to see detailed processing information:

```bash
uv run python src/synthetic_cli.py --source doc.pdf --verbose
```
