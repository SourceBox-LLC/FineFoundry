# FineFoundry

A desktop studio to curate datasets and fine-tune models. Scrape, merge, analyze, build/publish, and train on Runpod or locally — then ship to the Hugging Face Hub.

<p align="center">
  <img src="img/FineForge-logo.png" alt="FineFoundry logo" width="420" />
  <br/>
  <br/>
  <a href="https://www.python.org/">
    <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  </a>
  <a href="https://flet.dev/">
    <img alt="Flet 0.28.3" src="https://img.shields.io/badge/Flet-0.28.3-03A9F4?style=for-the-badge&logo=flutter&logoColor=white" />
  </a>
  <a href="https://huggingface.co/docs/datasets">
    <img alt="datasets 4.0.0" src="https://img.shields.io/badge/datasets-4.0.0-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white" />
  </a>
  <img alt="OS" src="https://img.shields.io/badge/OS-Windows%20|%20macOS%20|%20Linux-2E3440?style=for-the-badge&logo=windows&logoColor=white" />
  <a href="#license">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-4CAF50?style=for-the-badge" />
  </a>
</p>

<p align="center">
  <a href="#quick-start-gui">Quick Start</a> ·
  <a href="docs/README.md">📚 Full Documentation</a> ·
  <a href="docs/user-guide/troubleshooting.md">Troubleshooting</a> ·
  <a href="docs/development/logging.md">Logging</a> ·
  <a href="#license">License</a>
</p>

<hr/>

<details>
  <summary><b>Table of Contents</b></summary>

  - <a href="#quick-start-gui">Quick Start</a>
  - <a href="#using-the-app">Using the App</a>
    - <a href="#scrape-tab">Scrape tab</a>
    - <a href="#build--publish-tab">Build / Publish tab</a>
    - <a href="#training-tab">Training tab</a>
    - <a href="#merge-datasets-tab">Merge Datasets tab</a>
    - <a href="#dataset-analysis-tab">Dataset Analysis tab</a>
    - <a href="#settings-tab">Settings tab</a>
  - <a href="#hugging-face-authentication">Hugging Face Authentication</a>
  - <a href="#proxy-configuration">Proxy Configuration</a>
  - <a href="#cli-build-and-push-without-the-gui">CLI: Build &amp; Push</a>
  - <a href="#cli-reddit-scraper">CLI: Reddit Scraper</a>
  - <a href="#programmatic-4chan-scraper">Programmatic: 4chan</a>
  - <a href="#programmatic-stack-exchange-scraper">Programmatic: Stack Exchange</a>
  - <a href="#how-scraping-works-4chan">How Scraping Works (4chan)</a>
  - <a href="#dataset-artifacts">Dataset Artifacts</a>
  - <a href="#troubleshooting">Troubleshooting</a>
  - <a href="#ethical-and-legal-notes">Ethical &amp; Legal</a>
  - <a href="#development">Development</a>
  - <a href="#license">License</a>
</details>

Built with Flet for a fast, native-like UI. Includes:

- 4chan scraper with adjacent and contextual pairing (quote‑chain or cumulative) and robust cleaning.
- Reddit scraper CLI for subreddits or single posts; expands “more” comments; builds pairs (parent→child or contextual).
- Stack Exchange Q/A scraper (programmatic) for accepted answers.
- Dataset builder for train/val/test splits with Hugging Face `datasets`, with optional push + dataset card.
- Dataset analysis with togglable modules (sentiment, class balance, extra proxy metrics).
- Training via Runpod (managed pods, network volume at /data) or local Docker, with LoRA, packing, auto‑resume, Quick Local Inference, and reusable training configurations.

<a id="contents"></a>
## 🧭 Contents

- `src/main.py` — Flet desktop app (Scrape, Build/Publish, Training, Merge, Analysis, Settings)
- `src/scrapers/fourchan_scraper.py` — 4chan scraper and text cleaners (library)
- `src/scrapers/reddit_scraper.py` — Reddit scraper CLI + conversational pair builder
- `src/scrapers/stackexchange_scraper.py` — Stack Exchange Q/A scraper (programmatic)
- `src/save_dataset.py` — CLI dataset builder and Hub pusher
- `src/runpod/ensure_infra.py` — Runpod infrastructure automation (network volume + template)
- `src/runpod/runpod_pod.py` — Runpod pod helper (create/run, patch command, logs)
- `requirements.txt` — pinned dependencies

<a id="prerequisites"></a>
## 🧰 Prerequisites

- Python 3.10+ on Windows/macOS/Linux
- Git (optional)

Optional (for pushing to Hugging Face):

- A Hugging Face account and an access token with write permissions: <https://huggingface.co/settings/tokens>

<a id="quick-start-gui"></a>
## 🚀 Quick Start (GUI)

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

## 📚 Documentation

For comprehensive guides, tutorials, and API documentation, visit the **[docs/](docs/README.md)** directory:

- **[Quick Start Guide](docs/user-guide/quick-start.md)** - Get started in minutes
- **[User Guides](docs/README.md#-user-guides)** - Complete GUI and CLI documentation
- **[Merge Datasets](docs/user-guide/merge-tab.md)** - Detailed merge tab guide with download feature
- **[Troubleshooting](docs/user-guide/troubleshooting.md)** - Solutions to common issues
- **[Logging System](docs/development/logging.md)** - Professional logging and debugging
- **[Development Guides](docs/README.md#-development)** - For contributors
- **[API Reference](docs/README.md#-api-reference)** - Programmatic usage

<a id="using-the-app"></a>
## 🖥️ Using the App (Quick Reference)

<a id="scrape-tab"></a>
### 🧭 Scrape tab

![Scrape tab](img/ff_scrape_tab.png)

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

<a id="build--publish-tab"></a>
### 🏗️ Build / Publish tab

![Build / Publish tab](img/ff_buld_publish.png)

- Point to your `Data file (JSON)` (defaults to `scraped_training_data.json`).
- Configure `Seed`, `Shuffle`, `Min Length`, `Save dir`.
- Set split fractions with sliders (`Validation`, `Test`).
- Click **Build Dataset** to create `datasets.DatasetDict` and save to `Save dir`.
- To publish, enable **Push to Hub**, set:
  - `Repo ID` (e.g., `username/my-dataset`)
  - `Private` (toggle)
  - `HF Token` (can be left blank if you logged in via CLI or have `HF_TOKEN` env var)
- Click **Push + Upload README** to upload the dataset and a generated dataset card.

<a id="training-tab"></a>
### 🧠 Training tab

![Training tab](img/ff_training.png)

- **Training target**: choose **Runpod - Pod** (remote GPU pod) or **local** (Docker on this machine).
- **Dataset source**: select Hugging Face repo and split, or point to a JSON file.
- **Hyperparameters**: Base model (default `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`), Epochs, LR, Per-device batch size, Grad accum steps, Max steps, Packing, Auto-resume.
- **Output**: `Output dir` is used inside the container (typically under `/data/outputs/...`) and mapped back into your host folder when using Runpod or local Docker.
- **Push to Hub**: toggle to upload trained weights/adapters; set HF repo id and ensure authentication (HF token in Settings or env).
- **Skill level & Beginner presets**: pick Beginner or Expert. In Beginner mode, presets adapt to the training target:
  - **Runpod - Pod**: `Fastest (Runpod)` vs `Cheapest (Runpod)`.
  - **local**: `Quick local test` vs `Auto Set (local)`, where Auto Set uses your detected GPU VRAM to choose batch size, grad accumulation, and max steps that push the GPU without being overly aggressive.
- **Run on Runpod**:
  - First, ensure infrastructure: create/reuse a Network Volume and a Pod Template. The default mount path is `/data` (avoid `/workspace` to prevent hiding `train.py` baked into the image).
  - Start training on a pod from the template; checkpoints will appear under `/data/outputs/...` on the volume.
- **Run locally via Docker**:
  - Choose a host data directory to mount as `/data`, select image and GPU, optionally pass the HF token into the container, then launch.
  - Outputs and the trained adapter are written under your mounted host folder (e.g. `.../outputs/local_run`).
- **Quick Local Inference**:
  - After a successful local run, a Quick Local Inference panel appears.
  - Test the trained adapter with prompt input, temperature / max token sliders, presets (Deterministic / Balanced / Creative), and a clear-history button.
  - When you click **Run Inference**, the button is disabled and a small progress ring plus status text indicate that the fine-tuned model is loading and generating; the response appears in the panel once it is ready.
- **Training configurations**:
  - Save the current setup (dataset, hyperparameters, training target, Runpod infra or local Docker settings) as a JSON config.
  - Load configs from the **Configuration** section to quickly restore full training setups.
  - Config files live under `src/saved_configs/`, and the last used config auto-loads on startup.

#### 💡 Training notes & tips

- **Base model choice**: The default 4-bit instruct-tuned Llama 3.1 8B reduces VRAM needs and pairs well with LoRA.
- **LoRA**: Enables parameter-efficient fine-tuning. Use this on consumer GPUs; pushes typically upload adapters rather than full weights.
- **Packing**: Concatenates multiple short examples up to the model context window to improve throughput. Enable when most samples are short.
- **Grad accumulation**: If you hit out-of-memory, lower per-device batch size and increase gradient accumulation to keep effective batch similar.
- **Resume**: Use Auto-resume for interrupted sessions or provide a specific checkpoint path in “Resume from (path)”.
- **Outputs & checkpoints**: Intermediate checkpoints and the final artifact are saved under `Output dir`. When training remotely on Runpod, this is under `/data/...`.
- **Push to Hub**: Requires valid auth (see Hugging Face Authentication). Ensure the repo (and organization, if any) exists or your token has permissions to create it.

#### ⚙️ Alternative: AutoTrain Web (no-code)

- Export/push your dataset to the Hub from the Build/Publish tab and then use Hugging Face AutoTrain Web.
- Recommended base model: `Meta-Llama-3.1-8B-Instruct`.
- This is independent of the Training tab; settings here won’t affect AutoTrain jobs.

<a id="merge-datasets-tab"></a>
### 🔀 Merge Datasets tab

![Merge Datasets tab](img/ff_merge.png)

- Combine multiple JSON files and/or Hugging Face datasets into a single dataset.
- Automatically maps input/output columns and filters empty rows.
- Optionally normalize and save merged results to a JSON path or build a `datasets.DatasetDict`.

<a id="dataset-analysis-tab"></a>
### 📊 Dataset Analysis tab

![Dataset Analysis tab](img/ff_dataset_analysis.png)

- Select dataset source (Hugging Face or JSON) and click Analyze dataset.
- Use "Select all" to toggle all modules at once.
- Toggles gate computation and visibility:
  - Basic Stats
  - Duplicates & Similarity
  - Sentiment
  - Class balance (length buckets)
  - Extra metrics (lightweight proxy metrics): Coverage overlap, Data leakage, Conversation depth, Speaker balance, Question vs Statement, Readability, NER proxy, Toxicity, Politeness, Dialogue Acts, Topics, Alignment
- A summary lists all active modules after each run.

#### What each module shows

- **Basic Stats**: Record count and average input/output lengths.
- **Duplicates & Similarity**: Approximate duplicate rate using hashed input/output pairs.
- **Sentiment**: Distribution of sentiment polarity across samples to gauge tone and potential bias.
- **Class balance (length)**: Buckets by text length to reveal whether training targets are mostly short or long.
- **Coverage overlap**: Proxy overlap between input and output to catch trivial copies or potential leakage.
- **Data leakage**: Flags potential target text appearing in inputs (input/output inclusion proxy).
- **Conversation depth**: Proxy measure of turns/context size for conversational datasets.
- **Speaker balance**: Approximate input share vs output share (length-based proxy).
- **Question vs Statement**: Heuristic detection of question-like prompts vs declaratives.
- **Readability**: Proxy readability score/band to understand complexity.
- **NER proxy**: Frequency of entity-like tokens (names, orgs, locations) by simple proxies.
- **Toxicity/Politeness**: Lightweight proxies to surface potentially problematic content.
- **Dialogue Acts / Topics / Alignment**: Keyword-based proxies to surface common intents/themes and simple alignment signals.

#### Use the insights

- **Cleaning**: High toxicity or low readability? Consider filtering or normalizing before training.
- **Packing**: Mostly short outputs? Enable packing in Training to improve throughput.
- **Splits**: Any leakage flags? Rebuild with different shuffle/seed or adjust split sizes.
- **Curriculum**: Skewed length distribution? Consider mixing datasets or weighting sampling.

<a id="settings-tab"></a>
### ⚙️ Settings tab

- Hugging Face Access: save token used by Build/Publish and Training push.
- Proxies: per-scraper defaults and env-proxy usage.
- Runpod: save API key for infrastructure and pod control.
- Ollama: enable connection, set base URL, list/select models, and save settings. Used for dataset card generation.

<a id="hugging-face-authentication"></a>
## 🔐 Hugging Face Authentication

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

<a id="proxy-configuration"></a>
## 🌐 Proxy Configuration

- All scrapers support proxies via module variables.
  - 4chan (`src/scrapers/fourchan_scraper.py`): `PROXY_URL = "socks5h://127.0.0.1:9050"`, `USE_ENV_PROXIES = False` (default Tor SOCKS5)
  - Reddit (`src/scrapers/reddit_scraper.py`): `PROXY_URL = "socks5h://127.0.0.1:9050"`, `USE_ENV_PROXIES = False` (default Tor SOCKS5)
  - Stack Exchange (`src/scrapers/stackexchange_scraper.py`): `PROXY_URL = None`, `USE_ENV_PROXIES = False` (no proxy by default)
- To use system env proxies, set `USE_ENV_PROXIES = True` and define `HTTP_PROXY`/`HTTPS_PROXY` before launching.
- Programmatic runtime change:

  ```python
  # 4chan
  from scrapers.fourchan_scraper import PROXY_URL, USE_ENV_PROXIES, apply_session_config
  PROXY_URL = "http://127.0.0.1:8080"  # or None to disable
  USE_ENV_PROXIES = False
  apply_session_config()

  # Stack Exchange
  from scrapers import stackexchange_scraper as se
  se.PROXY_URL = "socks5h://127.0.0.1:9050"
  se.USE_ENV_PROXIES = False
  se.apply_session_config()
  ```

<a id="cli-build-and-push-without-the-gui"></a>
## 🛠️ CLI: Build and Push Without the GUI

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

<a id="cli-reddit-scraper"></a>
## 🧭 CLI: Reddit Scraper

Run the crawler and build pairs from a subreddit or a single post:

```bash
python src/scrapers/reddit_scraper.py \
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
- Proxy is configured via `PROXY_URL`/`USE_ENV_PROXIES` in `src/scrapers/reddit_scraper.py` (see Proxy Configuration).

- Output:
  - Writes dump under an auto folder, e.g., `reddit_dump_<slug>/` (or `--output-dir`).
  - Saves pairs to `reddit_dump_*/reddit_pairs.json`, then copies to `--pairs-path` or `./reddit_pairs.json` by default.
  - With `--cleanup` (or when `--use-temp-dump` is on), the dump folder is removed after copying pairs.

- Defaults (from `src/scrapers/reddit_scraper.py`):
  - `--url`: <https://www.reddit.com/r/LocalLLaMA/>
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
  python src/scrapers/reddit_scraper.py \
    --url https://www.reddit.com/r/AskReddit/comments/abc123/example_post/ \
    --mode parent_child --pairs-path reddit_pairs.json
  ```

<a id="programmatic-4chan-scraper"></a>
## 🧩 Programmatic: 4chan Scraper

Use the library API from `src/scrapers/fourchan_scraper.py`.

Tip: Ensure Python can find the `src/` directory for imports, e.g. set `PYTHONPATH=src` or add `sys.path.append("src")` before importing from `scrapers.*`.

```python
from scrapers.fourchan_scraper import scrape

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

<a id="programmatic-stack-exchange-scraper"></a>
## 🧩 Programmatic: Stack Exchange Scraper

Scrape accepted Q/A pairs via `src/scrapers/stackexchange_scraper.py`.

```python
from scrapers.stackexchange_scraper import scrape

pairs = scrape(site="stackoverflow", max_pairs=300, delay=0.3, min_len=10)
```

Enable a proxy programmatically:

```python
from scrapers import stackexchange_scraper as se
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

<a id="how-scraping-works-4chan"></a>
## ⚙️ How Scraping Works (4chan)

- Uses the 4chan public JSON API (`a.4cdn.org`).
- Samples threads across catalog pages to diversify coverage.
- Two pairing modes:
  - **Normal**: pairs sequential posts within threads (`input` = previous post, `output` = next post).
  - **Contextual**: builds `input` context using either:
    - **Quote-chain**: walk backwards via the most recent quotelink on each post up to `K`, merging same-poster chunks and optionally requiring a question in context; honors `Max Input Chars`.
    - **Cumulative**: use the last `K` sequential posts as context.
- Cleans HTML, removes greentext quote lines, quote references, URLs, and collapses whitespace.
- Minimum length filter is applied per post.

Key modules: `src/scrapers/fourchan_scraper.py` (e.g., `fetch_catalog_pages()`, `fetch_thread()`, `build_pairs_*()`).
Reddit and Stack Exchange details are documented in their CLI/usage sections below.

<a id="dataset-artifacts"></a>
## 🗃️ Dataset Artifacts

- On build, a `datasets.DatasetDict` is created with keys among `train`, `validation`, `test`.
- Saved via `DatasetDict.save_to_disk(SAVE_DIR)`.
- Pushed datasets are versioned on the Hub. The app also uploads a dataset card with:
  - Summary, fields, size bucket, example records, warnings, usage snippet.

<a id="troubleshooting"></a>
## 🩺 Troubleshooting

For comprehensive troubleshooting, see the **[Troubleshooting Guide](docs/user-guide/troubleshooting.md)**.

Quick fixes:
- **No data found**: Verify boards are selected and `Max Pairs` > 0; check network access to `a.4cdn.org`.
- **Push fails (401/403)**: Ensure your token has write scope and is correctly provided.
- **Large previews feel slow**: Use the dataset preview dialog (paginated) or open the saved dataset with Python.
- **SSL/Cert errors**: Update `certifi` or your system certificates.
- **Debug logging**: Set `FINEFOUNDRY_DEBUG=1` environment variable - see [Logging Guide](docs/development/logging.md)

<a id="ethical-and-legal-notes"></a>
## ⚖️ Ethical and Legal Notes

- Content may be NSFW, offensive, or harmful. For research purposes only.
- Respect platform policies and applicable laws in your jurisdiction.
- Before any production use, apply filtering, detoxification, and alignment techniques.

<a id="development"></a>
## 🧑‍💻 Development

- UI built with Flet (`flet`, `flet-desktop`), pinned in `requirements.txt`.
- Run formatting/linting as desired. Contributions welcome via PR.

<a id="license"></a>
## 📄 License

MIT