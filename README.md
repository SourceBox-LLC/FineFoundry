# FineFoundry

A desktop studio to curate datasets and fine-tune models. Scrape, merge, analyze, build/publish, and train on Runpod or locally — then ship to the Hugging Face Hub.

Built with Flet for a fast, native-like UI. Includes:

- 4chan scraper with adjacent and contextual pairing (quote‑chain or cumulative) and robust cleaning.
- Reddit scraper CLI for subreddits or single posts; expands “more” comments; builds pairs (parent→child or contextual).
- Stack Exchange Q/A scraper (programmatic) for accepted answers.
- Dataset builder for train/val/test splits with Hugging Face `datasets`, with optional push + dataset card.
- Dataset analysis with togglable modules (sentiment, class balance, extra proxy metrics).
- Training via Runpod (managed pods, network volume at /data) or local Docker, with LoRA, packing, auto‑resume, and Hub push.

## Contents

- `src/main.py` — Flet desktop app (Scrape, Build/Publish, Training, Merge, Analysis, Settings)
- `src/fourchan_scraper.py` — 4chan scraper and text cleaners (library)
- `src/reddit_scraper.py` — Reddit scraper CLI + conversational pair builder
- `src/stackexchange_scraper.py` — Stack Exchange Q/A scraper (programmatic)
- `src/save_dataset.py` — CLI dataset builder and Hub pusher
- `src/ensure_infra.py` — Runpod infrastructure automation (network volume + template)
- `src/runpod_pod.py` — Runpod pod helper (create/run, patch command, logs)
- `requirements.txt` — pinned dependencies

## Prerequisites

- Python 3.10+ on Windows/macOS/Linux
- Git (optional)

Optional (for pushing to Hugging Face):

- A Hugging Face account and an access token with write permissions: <https://huggingface.co/settings/tokens>

## Quick Start (GUI)

1. Create and activate a virtual environment

   ```bash
   # Windows (PowerShell)
   python -m venv venv
   ./venv/Scripts/Activate.ps1

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run the desktop app

   ```bash
   # Either of the following works
   python src/main.py
   # or
   flet run src/main.py
   ```

The app opens a desktop window. Use the tabs: Scrape, Build / Publish, Training, Merge Datasets, Dataset Analysis, and Settings.

## Using the App

### Scrape tab

- **Boards**: Multi-select chips with Select All / Clear.
- **Parameters**: `Max Threads`, `Max Pairs`, `Delay (s)`, `Min Length`, `Output JSON Path`.
- Click **Start** to scrape. Live progress, stats, and logs are shown.
- **Preview Dataset** to inspect the first rows in a two-column grid.

Default output is `scraped_training_data.json` in the project root. Schema is a list of objects:

```json
[
  {"input": "...", "output": "..."}
]
```

### Build / Publish tab

- Point to your `Data file (JSON)` (defaults to `scraped_training_data.json`).
- Configure `Seed`, `Shuffle`, `Min Length`, `Save dir`.
- Set split fractions with sliders (`Validation`, `Test`).
- Click **Build Dataset** to create `datasets.DatasetDict` and save to `Save dir`.
- To publish, enable **Push to Hub**, set:
  - `Repo ID` (e.g., `username/my-dataset`)
  - `Private` (toggle)
  - `HF Token` (can be left blank if you logged in via CLI or have `HF_TOKEN` env var)
- Click **Push + Upload README** to upload the dataset and a generated dataset card.

### Training tab

- **Dataset source**: select Hugging Face repo and split for training data.
- **Hyperparameters**: Base model (default `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`), Epochs, Learning rate, Per-device batch size, Grad accum steps, Max steps.
- **Options**: Use LoRA, Packing, Auto-resume.
- **Output**: `Output dir` defaults to `/data/outputs/runpod_run`. Optional `Resume from (path)` to continue from a checkpoint.
- **Push to Hub**: toggle to upload trained weights/adapters; set HF repo id and ensure authentication.
- **Run on Runpod**:
  - First, ensure infrastructure: create/reuse a Network Volume and a Pod Template. The default mount path is `/data`. Avoid `/workspace` to prevent hiding `train.py` inside the image.
  - Start training on a pod from the template; checkpoints will appear under `/data/outputs/runpod_run`.
- **Run locally via Docker**:
  - Choose a host data directory to mount as `/data`, select image and GPU, then launch. Outputs land in the mounted folder under `outputs/runpod_run`.

### Merge Datasets tab

- Combine multiple JSON files and/or Hugging Face datasets into a single dataset.
- Automatically maps input/output columns and filters empty rows.
- Optionally normalize and save merged results to a JSON path or build a `datasets.DatasetDict`.

### Dataset Analysis tab

- Select dataset source (Hugging Face or JSON) and click Analyze.
- Toggles gate computation and visibility:
  - Sentiment
  - Class balance (length buckets)
  - Extra metrics (lightweight proxy metrics): Coverage overlap, Data leakage, Conversation depth, Speaker balance, Question vs Statement, Readability, NER proxy, Toxicity, Politeness, Dialogue Acts, Topics, Alignment
- A summary lists all active modules after each run.

### Settings tab

- Hugging Face Access: save token used by Build/Publish and Training push.
- Proxies: per-scraper defaults and env-proxy usage.
- Runpod: save API key for infrastructure and pod control.

## Hugging Face Authentication

You can authenticate in one of three ways:

- Paste your token into the `HF Token` field in the UI.
- Set an environment variable before launching the app:

   ```bash
   # Windows (PowerShell)
   setx HF_TOKEN "hf_xxx"
   # New shells will inherit it
   ```

- Log in via CLI (persisted cache):

  ```bash
  huggingface-cli login
  ```

The app tries the field value first, then `HF_TOKEN`, then the cached login.

## Proxy Configuration

- All scrapers support proxies via module variables.
  - 4chan (`src/fourchan_scraper.py`): `PROXY_URL = "socks5h://127.0.0.1:9050"`, `USE_ENV_PROXIES = False` (default Tor SOCKS5)
  - Reddit (`src/reddit_scraper.py`): `PROXY_URL = "socks5h://127.0.0.1:9050"`, `USE_ENV_PROXIES = False` (default Tor SOCKS5)
  - Stack Exchange (`src/stackexchange_scraper.py`): `PROXY_URL = None`, `USE_ENV_PROXIES = False` (no proxy by default)
- To use system env proxies, set `USE_ENV_PROXIES = True` and define `HTTP_PROXY`/`HTTPS_PROXY` before launching.
- Programmatic runtime change:

  ```python
  # 4chan
  from src.fourchan_scraper import PROXY_URL, USE_ENV_PROXIES, apply_session_config
  PROXY_URL = "http://127.0.0.1:8080"  # or None to disable
  USE_ENV_PROXIES = False
  apply_session_config()

  # Stack Exchange
  from src import stackexchange_scraper as se
  se.PROXY_URL = "socks5h://127.0.0.1:9050"
  se.USE_ENV_PROXIES = False
  se.apply_session_config()
  ```

## CLI: Build and Push Without the GUI

`src/save_dataset.py` provides a fully scriptable path.
Note: there are no CLI flags; configure constants in the file header, then run it.

1. Ensure you have a JSON file like `scraped_training_data.json` (schema above).
2. Open `src/save_dataset.py` and edit the configuration block at the top:

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

3. Run the script

   ```bash
   python src/save_dataset.py
   ```

This saves to `SAVE_DIR` and optionally pushes to `REPO_ID`. A dataset card is generated and uploaded as `README.md` in the repo.

## CLI: Reddit Scraper

Run the crawler and build pairs from a subreddit or a single post:

```bash
python src/reddit_scraper.py \
  --url https://www.reddit.com/r/AskReddit/ \
  --max-posts 50 \
  --mode contextual --k 4 --max-input-chars 2000 \
  --pairs-path reddit_pairs.json --cleanup
```

- General crawl:
  - `--url` (subreddit or post URL), `--max-posts`, `--request-delay`, `--request-jitter-frac`
  - `--max-requests` (0=off), `--stop-after-seconds` (0=off)
  - `--output-dir`, `--use-temp-dump`, `--no-expand-more`
- Dataset build:
  - `--build-dataset`, `--mode {parent_child,contextual}`, `--k`, `--max-input-chars` (0=off)
  - `--require-question`, `--no-merge-same-author`, `--min-len`, `--include-automod`
  - `--pairs-path` (copy pairs to a stable path), `--cleanup` (remove dump)
- Proxy is configured via `PROXY_URL`/`USE_ENV_PROXIES` in `src/reddit_scraper.py` (see Proxy Configuration).

- Output:
  - Writes dump under an auto folder, e.g., `reddit_dump_<slug>/` (or `--output-dir`).
  - Saves pairs to `reddit_dump_*/reddit_pairs.json`, then copies to `--pairs-path` or `./reddit_pairs.json` by default.
  - With `--cleanup` (or when `--use-temp-dump` is on), the dump folder is removed after copying pairs.

- Defaults (from `src/reddit_scraper.py`):
  - `--url`: <https://www.reddit.com/r/Conservative/>
  - `--max-posts`: 100
  - `--request-delay`: 1.0 (s)
  - `--request-jitter-frac`: 0.5
  - `--max-requests`: 1000 (0=off)
  - `--stop-after-seconds`: 0 (off)
  - `--output-dir`: auto
  - `--use-temp-dump`: true
  - expand-more: on (disable with `--no-expand-more`)
  - `--build-dataset`: true
  - `--mode`: parent_child (alt: contextual)
  - `--k`: 4
  - `--max-input-chars`: 2000 (0=off)
  - `--require-question`: false
  - merge-same-author: on (disable with `--no-merge-same-author`)
  - `--min-len`: 1
  - exclude AutoModerator: on (enable with `--include-automod`)

- Single-post example (ignores `--max-posts`):

  ```bash
  python src/reddit_scraper.py \
    --url https://www.reddit.com/r/AskReddit/comments/abc123/example_post/ \
    --mode parent_child --pairs-path reddit_pairs.json
  ```

## Programmatic: 4chan Scraper

Use the library API from `src/fourchan_scraper.py`.

```python
from src.fourchan_scraper import scrape

pairs = []
for board in ["pol", "b"]:
    pairs += scrape(
        board=board,
        max_threads=150,
        max_pairs=5000,
        delay=0.5,
        min_len=3,
        mode="contextual",            # "normal" or "contextual"
        strategy="quote_chain",       # contextual: "quote_chain", "cumulative", or "last_k"
        k=6,
        max_chars=2000,
        merge_same_id=True,
        require_question=False,
    )
```

- Parameters (defaults in code):
  - `board` (e.g., `pol`, `b`); see `ALLOWLIST_DEFAULT` for a sample list.
  - `max_threads`, `max_pairs`, `delay`, `min_len`
  - `mode`: `normal` (adjacent) or `contextual`
  - `strategy`: `cumulative` (default), `last_k`, or `quote_chain`
  - `k`: context depth (default 6)
  - `max_chars`: truncate context from the end (None/off by default)
  - `merge_same_id`: merge consecutive chunks from the same poster (default True)
  - `require_question`: keep only pairs with question-like context (default False)
- Proxy: defaults to Tor via `PROXY_URL = "socks5h://127.0.0.1:9050"`; set `USE_ENV_PROXIES=True` to use `HTTP(S)_PROXY`. If you change these at runtime, call `apply_session_config()` before `scrape()`.

## Programmatic: Stack Exchange Scraper

Scrape accepted Q/A pairs via `src/stackexchange_scraper.py`.

```python
from src.stackexchange_scraper import scrape

pairs = scrape(site="stackoverflow", max_pairs=300, delay=0.3, min_len=10)
```

Enable a proxy programmatically:

```python
from src import stackexchange_scraper as se
se.PROXY_URL = "socks5h://127.0.0.1:9050"  # or HTTP proxy
se.USE_ENV_PROXIES = False  # or True to use HTTP(S)_PROXY env
se.apply_session_config()
pairs = se.scrape(site="superuser", max_pairs=50)
```

- Parameters (defaults in code):
  - `site`: e.g., `stackoverflow`, `superuser` (default `stackoverflow`)
  - `max_pairs`: default 100
  - `delay`: polite delay between calls (default 0.2s)
  - `min_len`: minimum characters per side (default 0)
- Env var: set `STACKAPPS_KEY` to use a Stack Apps API key and reduce throttling.
  - Windows (PowerShell): `setx STACKAPPS_KEY "your_key"`
  - macOS/Linux: `export STACKAPPS_KEY="your_key"`
- Backoff: the API may return a `backoff` value; the scraper honors it automatically.
- No CLI entrypoint for Stack Exchange; use the programmatic API or the GUI.

## How Scraping Works (4chan)

- Uses the 4chan public JSON API (`a.4cdn.org`).
- Samples threads across catalog pages to diversify coverage.
- Two pairing modes:
  - **Normal**: pairs sequential posts within threads (`input` = previous post, `output` = next post).
  - **Contextual**: builds `input` context using either:
    - **Quote-chain**: walk backwards via the most recent quotelink on each post up to `K`, merging same-poster chunks and optionally requiring a question in context; honors `Max Input Chars`.
    - **Cumulative**: use the last `K` sequential posts as context.
- Cleans HTML, removes greentext quote lines, quote references, URLs, and collapses whitespace.
- Minimum length filter is applied per post.

Key modules: `src/fourchan_scraper.py` (e.g., `fetch_catalog_pages()`, `fetch_thread()`, `build_pairs_*()`).
Reddit and Stack Exchange details are documented in their CLI/usage sections below.

## Dataset Artifacts

- On build, a `datasets.DatasetDict` is created with keys among `train`, `validation`, `test`.
- Saved via `DatasetDict.save_to_disk(SAVE_DIR)`.
- Pushed datasets are versioned on the Hub. The app also uploads a dataset card with:
  - Summary, fields, size bucket, example records, warnings, usage snippet.

## Troubleshooting

- **No data found**: Verify boards are selected and `Max Pairs` > 0; check network access to `a.4cdn.org`.
- **Push fails (401/403)**: Ensure your token has write scope and is correctly provided.
- **Large previews feel slow**: Use the dataset preview dialog (paginated) or open the saved dataset with Python.
- **Windows SmartScreen warnings**: This is a Python app; run within your venv.
- **SSL/Cert errors**: Update `certifi` or your system certificates.
- **Too few pairs when using contextual**: try reducing `Last K`, unchecking "Require question", or increasing `Max Threads`/`Max Pairs`.

## Ethical and Legal Notes

- Content may be NSFW, offensive, or harmful. For research purposes only.
- Respect platform policies and applicable laws in your jurisdiction.
- Before any production use, apply filtering, detoxification, and alignment techniques.

## Development

- UI built with Flet (`flet`, `flet-desktop`), pinned in `requirements.txt`.
- Run formatting/linting as desired. Contributions welcome via PR.

## License

MIT