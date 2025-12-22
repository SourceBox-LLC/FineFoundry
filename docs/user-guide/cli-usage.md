# CLI Usage

While most users interact with FineFoundry through the desktop app, everything also works from the command line. The CLI is especially useful when you want to automate dataset builds, schedule scrapes, or integrate FineFoundry into larger workflows and CI pipelines.

This guide covers three main CLI tools: dataset building, Reddit scraping, and synthetic data generation.

## Before You Start

Make sure you have Python 3.10+ installed and the repository cloned and set up (see the [Installation Guide](installation.md)). If you're pushing to Hugging Face, you'll need a token—see [Authentication](authentication.md) for setup options.

All commands below assume you're running from the project root.

______________________________________________________________________

## Building and Pushing Datasets

The `src/save_dataset.py` script turns a JSON file of input/output pairs into a proper Hugging Face dataset, with optional push to the Hub.

### Preparing Your Data

FineFoundry stores scrape results in its SQLite database. To use them with this script, export a session to JSON first:

```bash
uv run python -c "import sys; sys.path.append('src'); from db.scraped_data import list_scrape_sessions; print(list_scrape_sessions(limit=10))"

uv run python -c "import sys; sys.path.append('src'); from db.scraped_data import export_session_to_json; export_session_to_json(SESSION_ID, 'scraped_training_data.json')"
```

Your JSON should look like this:

```json
[
  {"input": "...", "output": "..."}
]
```

### Configuring the Script

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
HF_TOKEN = None  # uses env HF_TOKEN or cached login if None
```

Then run it:

```bash
python src/save_dataset.py
```

The script reads your JSON, builds train/validation/test splits, saves locally, and optionally pushes to the Hub with a dataset card. For CI, provide your token via environment variables.

______________________________________________________________________

## Reddit Scraping

The Reddit scraper CLI crawls subreddits or individual posts and builds conversation pairs.

### Basic Example

```bash
python src/scrapers/reddit_scraper.py \
  --url https://www.reddit.com/r/AskReddit/ \
  --max-posts 50 \
  --mode contextual --k 4 --max-input-chars 2000 \
  --pairs-path reddit_pairs.json --cleanup
```

This crawls up to 50 posts from r/AskReddit, builds contextual pairs (using the last 4 posts as context, truncated to 2000 chars), saves the result to `reddit_pairs.json`, and cleans up intermediate files.

### Key Options

For crawling: `--url` sets the target, `--max-posts` limits how many posts to process, `--request-delay` controls pacing (respect rate limits!), and `--stop-after-seconds` sets a time limit.

For dataset building: `--mode` chooses between `parent_child` (simple reply pairs) or `contextual` (conversation context), `--k` sets context depth, `--require-question` filters for Q&A-style content, and `--min-len` sets minimum character length.

### Single Post

To focus on one thread:

```bash
python src/scrapers/reddit_scraper.py \
  --url https://www.reddit.com/r/AskReddit/comments/abc123/example_post/ \
  --mode parent_child --pairs-path reddit_pairs.json
```

Proxy settings are configured in the script file itself. See [Proxy Setup](../deployment/proxy-setup.md) for details.

______________________________________________________________________

## CLI vs GUI

Use the GUI when you want to explore interactively, iterate on settings with visual feedback, or manage the full workflow from scraping through training and inference in one place.

Use the CLI when you're automating—cron jobs, CI pipelines, batch processing, or reproducing exact configurations across machines. The CLI gives you the same functionality with scriptability.

______________________________________________________________________

## Synthetic Data Generation

The synthetic CLI generates training data from your documents and web pages. Point it at PDFs, Word docs, text files, or URLs, and it produces Q&A pairs, chain-of-thought examples, or summaries.

### Basic Usage

```bash
python src/synthetic_cli.py --source document.pdf --output qa_pairs.json
```

This loads a model, chunks your document, generates Q&A pairs from each chunk, and saves everything to JSON. Results also go to the FineFoundry database by default.

You can process multiple sources at once:

```bash
python src/synthetic_cli.py \
  --source paper.pdf \
  --source notes.txt \
  --source https://example.com/article \
  --output combined_data.json
```

### Generation Types

Use `--type` to choose what kind of data to generate:

- `qa` (default) — question-answer pairs
- `cot` — chain-of-thought reasoning examples
- `summary` — summaries

```bash
python src/synthetic_cli.py --source paper.pdf --type cot --output cot_data.json
```

### Quality Curation

Enable `--curate` to filter out low-quality pairs using Llama-as-judge:

```bash
python src/synthetic_cli.py --source document.pdf --curate --threshold 8.0 --output curated_pairs.json
```

Higher thresholds (up to 10) are stricter.

### Config Files

For repeatable setups, use a YAML config:

```yaml
# synthetic_config.yaml
sources:
  - document.pdf
  - https://example.com/article
output: synthetic_data.json
type: qa
num_pairs: 25
max_chunks: 10
curate: false
```

```bash
uv run python src/synthetic_cli.py --config synthetic_config.yaml
```

CLI arguments override config values.

### Batch Processing

When processing multiple files, use `--keep-server` to avoid reloading the model each time:

```bash
uv run python src/synthetic_cli.py --source batch1.pdf --output batch1.json --keep-server
uv run python src/synthetic_cli.py --source batch2.pdf --output batch2.json --keep-server
uv run python src/synthetic_cli.py --source batch3.pdf --output batch3.json  # last run cleans up
```

### Resuming and Deduplication

If generation gets interrupted, use `--resume` to continue from where you left off. Progress saves after each chunk. Use `--dedupe` to remove duplicate entries based on input text.

### Output Formats

Besides JSON, you can export directly to Hugging Face datasets format or Parquet:

```bash
uv run python src/synthetic_cli.py --source doc.pdf --output-type hf --output my_dataset
uv run python src/synthetic_cli.py --source doc.pdf --output-type parquet --output data.parquet
```

### Pushing to Hub

Push directly to Hugging Face:

```bash
uv run python src/synthetic_cli.py --source doc.pdf --push-to-hub username/my-dataset --private
```

Requires authentication via `huggingface-cli login` or `HF_TOKEN`.

### Scripting and Debugging

Use `--quiet` for machine-readable JSON output, or `--verbose` for detailed logs. Add `--stats` to see token counts and length distributions after generation.
