# FineFoundry

**Create your own custom AI models — no coding required.**

FineFoundry is a desktop app that lets you collect training data, teach AI models, and test the results. Whether you want to build a chatbot, create a specialized assistant, or just experiment with AI, FineFoundry makes it accessible to everyone.

<p align="center">
  <img src="img/FineForge-logo.png" alt="FineFoundry logo" width="420" />
</p>

## ✨ What Can You Do?

- **Collect data** from Reddit, 4chan, Stack Exchange, or your own documents
- **Train AI models** on your computer or in the cloud
- **Test your models** by chatting with them
- **Share your work** on Hugging Face

<p align="center">
  <a href="#quick-start-gui">🚀 Quick Start</a> ·
  <a href="docs/README.md">📚 Full Documentation</a> ·
  <a href="docs/user-guide/troubleshooting.md">🔧 Troubleshooting</a>
</p>

<hr/>

<p align="center">
  <a href="https://www.python.org/">
    <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  </a>
  <img alt="OS" src="https://img.shields.io/badge/OS-Windows%20|%20macOS%20|%20Linux-2E3440?style=for-the-badge&logo=windows&logoColor=white" />
  <a href="#license">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-4CAF50?style=for-the-badge" />
  </a>
</p>

<details>
  <summary><b>📋 Full Table of Contents</b></summary>

- <a href="#quick-start-gui">Quick Start</a>
- <a href="#using-the-app">Using the App</a>
  - <a href="#scrape-tab">Scrape tab</a>
  - <a href="#build--publish-tab">Publish tab</a>
  - <a href="#training-tab">Training tab</a>
  - <a href="#inference-tab">Inference tab</a>
  - <a href="#evaluate-tab">Evaluate tab</a>
  - <a href="#merge-datasets-tab">Merge Datasets tab</a>
  - <a href="#dataset-analysis-tab">Dataset Analysis tab</a>
  - <a href="#settings-tab">Settings tab</a>
- <a href="#hugging-face-authentication">Hugging Face Authentication</a>
- <a href="#proxy-configuration">Proxy Configuration</a>
- <a href="#cli-build-and-push-without-the-gui">CLI: Build & Push</a>
- <a href="#cli-reddit-scraper">CLI: Reddit Scraper</a>
- <a href="#programmatic-4chan-scraper">Programmatic: 4chan</a>
- <a href="#programmatic-stack-exchange-scraper">Programmatic: Stack Exchange</a>
- <a href="#how-scraping-works-4chan">How Scraping Works (4chan)</a>
- <a href="#data-storage">Data Storage</a>
- <a href="#dataset-artifacts">Dataset Artifacts</a>
- <a href="#troubleshooting">Troubleshooting</a>
- <a href="#ethical-and-legal-notes">Ethical & Legal</a>
- <a href="#development">Development</a>
- <a href="#license">License</a>

</details>

<details>
  <summary><b>🔧 Technical Features (for developers)</b></summary>

- 4chan scraper with adjacent and contextual pairing (quote‑chain or cumulative) and robust cleaning
- Reddit scraper for subreddits or single posts with comment expansion
- Stack Exchange Q/A scraper for accepted answers
- Resilient HTTP scrapers with proxy support, rate limiting, and automatic retries
- Synthetic data generation using Unsloth's SyntheticDataKit
- Dataset builder with train/val/test splits and Hugging Face Hub integration
- Dataset analysis with sentiment, duplicates, and quality metrics
- Training via Runpod or locally with Unsloth‑based LoRA fine‑tuning
- Local inference with Transformers + PEFT + bitsandbytes
- Model evaluation with EleutherAI lm-evaluation-harness (14 benchmarks including MMLU, HellaSwag, TruthfulQA)

</details>

<a id="contents"></a>

## 🧭 Contents

- `src/main.py` — Flet desktop app (Scrape, Build/Publish, Training, Merge, Analysis, Settings)
- `src/db/` — SQLite database module (sole storage mechanism)
- `src/helpers/` — Business logic helpers (settings, training, scraping, synthetic)
- `src/scrapers/` — Data scrapers (4chan, Reddit, Stack Exchange)
- `src/ui/` — UI components and tab controllers
- `src/runpod/` — Runpod infrastructure automation
- `src/save_dataset.py` — CLI dataset builder and Hub pusher
- `finefoundry.db` — SQLite database (auto-created on first run)
- `training_outputs/` — Model artifacts (checkpoints, adapters)

<a id="prerequisites"></a>

## 🧰 Prerequisites

- Python 3.10+ on Windows/macOS/Linux
- Git (optional)

Optional (for pushing to Hugging Face):

- A Hugging Face account and an access token with write permissions: <https://huggingface.co/settings/tokens>

<a id="quick-start-gui"></a>

## 🚀 Quick Start (GUI)

### Option 1: uv (recommended)

```bash
git clone https://github.com/SourceBox-LLC/FineFoundry.git FineFoundry-Core
cd FineFoundry-Core

# Install uv if needed
pip install uv

# Run the app (creates an isolated env and installs deps automatically)
# One-time (macOS/Linux): allow executing the launcher script
chmod +x run_finefoundry.sh
./run_finefoundry.sh

# Alternative (without the launcher script)
uv run src/main.py
```

### Option 2: venv + pip

```bash
git clone https://github.com/SourceBox-LLC/FineFoundry.git FineFoundry-Core
cd FineFoundry-Core

# Windows (PowerShell)
python -m venv venv
./venv/Scripts/Activate.ps1

# macOS/Linux
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Run the desktop app
python src/main.py
# or
flet run src/main.py
```

The app opens a desktop window. Use the tabs: Scrape, Publish, Training, Inference, Evaluate, Merge Datasets, Dataset Analysis, and Settings.

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

![Scrape tab](img/new/ff_data_sources.png)

Choose from multiple data sources:

- **4chan**: Multi-select board chips with Select All / Clear.
- **Reddit**: Scrape subreddits or single posts with comment expansion.
- **Stack Exchange**: Q&A pairs from Stack Overflow and other sites.
- **Synthetic**: Generate training data from documents using local LLMs (see below).

**Parameters**: `Max Threads`, `Max Pairs`, `Delay (s)`, `Min Length`.

Click **Start** to scrape. Live progress, stats, and logs are shown. **Preview Dataset** to inspect the first rows in a two-column grid. All scraped data is automatically saved to the database.

#### Synthetic Data Generation

Generate Q&A pairs, chain-of-thought reasoning, or summaries from your own documents:

1. Select **Synthetic** as the source
1. Add files (PDF, DOCX, TXT, HTML) or URLs
1. Configure:
   - **Model**: Local LLM to use (default: `unsloth/Llama-3.2-3B-Instruct`)
   - **Generation Type**: `qa` (Q&A pairs), `cot` (chain-of-thought), or `summary`
   - **Num Pairs**: Target number of examples per chunk
   - **Max Chunks**: Maximum document chunks to process
1. Click **Start** — a snackbar appears immediately while the model loads (~30-60s on first run)
1. Watch live progress as chunks are processed and pairs generated
1. Results are automatically saved to the database

Generated data uses the standard schema:

```json
[
  {"input": "...", "output": "..."}
]
```

You can export data to JSON via the database helpers if needed for external tools.

<a id="build--publish-tab"></a>

### 🏗️ Publish tab

![Publish tab](img/new/ff_build_publish.png)

- Select a **Database Session** from your scrape history.
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

![Training tab](img/new/ff_training.png)

- **Training target**: choose **Runpod - Pod** (remote GPU pod) or **Local** (native training on this machine).
- **Dataset source**: select a **Database Session** from your scrape history, or a **Hugging Face** repo and split.
- **Hyperparameters**: Base model (default `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`), Epochs, LR, Per-device batch size, Grad accum steps, Max steps, Packing, Auto-resume.
- **Output**: `Output dir` is where checkpoints and adapters are saved. On Runpod this is under `/data/outputs/...` on the network volume; locally it's a path on your filesystem.
- **Push to Hub**: toggle to upload trained weights/adapters; set HF repo id and ensure authentication (HF token in Settings or env).
- **Skill level & Beginner presets**: pick Beginner or Expert. In Beginner mode, presets adapt to the training target:
  - **Runpod - Pod**: `Fastest (Runpod)` vs `Cheapest (Runpod)`.
  - **local**: `Quick local test` vs `Auto Set (local)`, where Auto Set uses your detected GPU VRAM to choose batch size, grad accumulation, and max steps that push the GPU without being overly aggressive.
- **Run on Runpod**:
  - First, ensure infrastructure: create/reuse a Network Volume and a Pod Template. The default mount path is `/data` (avoid `/workspace` to prevent hiding `train.py` baked into the image).
  - Start training on a pod from the template; checkpoints will appear under `/data/outputs/...` on the volume.
- **Run locally (native)**:
  - Enable GPU if you have a CUDA-capable card, optionally pass your HF token for Hub push, then launch.
  - Outputs and the trained adapter are written to your configured output directory (e.g. `.../outputs/local_run`).
- **Quick Local Inference**:
  - After a successful local run, a Quick Local Inference panel appears.
  - Test the trained adapter with prompt input, temperature / max token sliders, presets (Deterministic / Balanced / Creative), and a clear-history button.
  - When you click **Run Inference**, the button is disabled and a small progress ring plus status text indicate that the fine-tuned model is loading and generating; the response appears in the panel once it is ready.
- **Training configurations**:
  - Save the current setup (dataset, hyperparameters, training target, Runpod infra or local settings) as a named configuration.
  - Load configs from the **Configuration** section to quickly restore full training setups.
  - Configurations are stored in the database and the last used config auto-loads on startup.

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

![Merge Datasets tab](img/new/ff_merge_datasets.png)

- Combine multiple **Database Sessions** and/or **Hugging Face** datasets into a single dataset.
- Automatically maps input/output columns and filters empty rows.
- Merged results are saved to the database as a new session, with optional JSON export.

<a id="dataset-analysis-tab"></a>

### 📊 Dataset Analysis tab

![Dataset Analysis tab](img/new/ff_data_analysis.png)

- Select dataset source (Database Session or Hugging Face) and click Analyze dataset.
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
<a id="inference-tab"></a>

### 💬 Inference tab

![Inference tab](img/new/ff_inference.png)

- Pick a **base model** and a **completed training run**.
- The run's adapter directory is **validated immediately**; invalid adapters show a red status message and a snackbar, and the prompt controls remain locked.
- Once validated, the **Prompt & responses** section unlocks, with:
  - Prompt box, **Preset** dropdown (Deterministic / Balanced / Creative), Temperature and Max new tokens sliders.
  - **Generate** button that uses the same local inference helpers as Quick Local Inference, with a small progress ring and status text while the model loads and responds.
  - **Clear history** button to reset the shared conversation.
- **Full Chat View** button opens a focused chat dialog where you can have multi‑turn conversations with the same adapter/base model pair.
  - The main Prompt & responses history and Full Chat View share a single conversation history, so you can switch back and forth without losing context.

<a id="evaluate-tab"></a>

### 📊 Evaluate tab

![Evaluate tab](img/new/ff_evaluate.png)

Systematically benchmark your fine-tuned models using [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the same framework that powers the HuggingFace Open LLM Leaderboard.

- **Model Selection**: Pick a completed training run to evaluate. The base model and adapter path are auto-detected.
- **Benchmark Selection**: Choose from 14 benchmarks across different categories:
  - **Quick tests** (⚡): HellaSwag, TruthfulQA, ARC Easy, Winogrande, BoolQ — fast to run, great for quick validation
  - **Full benchmarks**: ARC Challenge, MMLU (57 tasks), MMLU-PRO, GSM8K (math)
  - **Advanced**: IFEval (instruction following), BBH (Big Bench Hard), GPQA (PhD-level), MuSR (multistep reasoning), HumanEval (code)
- **Configuration**: Set max samples (limit dataset size for faster runs) and batch size.
- **Comparison Mode**: Optionally evaluate the base model too and see the delta (Δ) showing improvement or regression.
- **Visual Results**: 
  - Metrics table showing accuracy percentages
  - Visual bar charts for each metric (accuracy, normalized accuracy)
  - Color-coded delta values (green = improvement, red = regression)

#### Evaluate workflow

1. Train a model in the **Training** tab
2. Test it manually in the **Inference** tab ("does it feel right?")
3. **Evaluate** it systematically ("here are the numbers proving it works")
4. **Publish** to Hugging Face with confidence

#### Tips

- Start with **quick benchmarks** (⚡) to get fast feedback
- Use **100 samples** for testing, increase for final evaluation
- **Comparison mode** doubles eval time but shows if fine-tuning actually helped
- GPU memory is automatically cleared between evaluations

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

   The GUI stores scrape results in the SQLite database (`finefoundry.db`). To create a JSON file from a scrape session, export it from the database:

   ```bash
   uv run python -c "import sys; sys.path.append('src'); from db.scraped_data import export_session_to_json; export_session_to_json(SESSION_ID, 'scraped_training_data.json')"
   ```

1. Open `src/save_dataset.py` and edit the configuration block at the top:

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

1. Run the script

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

<a id="data-storage"></a>

## 💾 Data Storage

FineFoundry uses SQLite (`finefoundry.db`) as the **sole storage mechanism** for all application data:

- **Settings** — HF token, RunPod API key, Ollama config, proxy settings
- **Training Configs** — Saved hyperparameter configurations
- **Scrape Sessions** — History of all scrape runs with metadata (4chan, Reddit, Stack Exchange, Synthetic)
- **Scraped Pairs** — All input/output pairs from scraping and synthetic generation
- **Training Runs** — Managed training runs with logs, adapters, checkpoints
- **App Logs** — Application logs (database-backed, not file-based)

The database is auto-created on first run. The only file-based storage is `training_outputs/` for binary model artifacts (checkpoints, adapters).

For detailed schema and API documentation, see [Database Architecture](docs/development/database.md).

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

- UI built with Flet (`flet[all]`), pinned in `pyproject.toml`.
- Run formatting/linting as desired. Contributions welcome via PR.

### Controller pattern

- Each tab's behavior and event wiring lives in `ui/tabs/*_controller.py` (Scrape, Build / Publish, Merge Datasets, Dataset Analysis, Training, Inference).
- Layout-only builders live in `ui/tabs/tab_*.py` and per-tab section modules under `ui/tabs/<tab>/sections/`.
- `src/main.py` is intentionally kept slim: it builds the global shell and delegates tab construction to controller functions such as `build_scrape_tab_with_logic(...)`, `build_build_tab_with_logic(...)`, `build_merge_tab_with_logic(...)`, and `build_analysis_tab_with_logic(...)`.

#### Adding a new tab (high level)

- Pick a short tab key (for example, `mytab`).
- Create a layout builder `ui/tabs/tab_mytab.py` and any sections under `ui/tabs/mytab/sections/`.
- Create a controller `ui/tabs/mytab_controller.py` that exposes `build_mytab_tab_with_logic(page, *, section_title, _mk_help_handler, ...)`.
- Import that controller function in `src/main.py` and add a new `ft.Tab(...)` entry whose `content` is the result of `build_mytab_tab_with_logic(...)`.

### Testing

Tests live under `tests/` and are discovered by `pytest` (configured via `pyproject.toml`). The main groups are:

- **Unit tests (`tests/unit/`)**

  - Fast, isolated tests for helper modules and the `save_dataset` CLI (normalization, splits, logging, local specs, proxy logic, Ollama settings, ChatML conversion, hyperparameter builder, etc.).

  - Run locally:

    ```bash
    uv run pytest tests/unit
    ```

- **Integration tests (`tests/integration/`)**

  - End-to-end flows such as `save_dataset.main` (data → normalized pairs → on-disk HF dataset), synthetic data generation, and tab-controller smoke tests (Scrape/Build/Merge/Analysis/Training/Inference controllers building successfully on a dummy `Page`).

  - Marked with `@pytest.mark.integration` so you can select or skip them:

    ```bash
    # Only integration tests
    uv run pytest -m "integration"

    # Everything except integration tests (fast inner loop)
    uv run pytest -m "not integration"
    ```

- **Full suite**

  ```bash
  uv run pytest
  ```

#### Coverage

- Locally, you can measure coverage with `coverage.py` (install once with `uv pip install coverage`):

  ```bash
  uv run coverage run --source=src -m pytest --ignore=proxy_test.py
  uv run coverage report -m
  ```

- In CI, tests are run under coverage for each Python version in the matrix. A summary is printed in the logs, an XML report is produced per version,
  and a minimum coverage threshold is enforced with `coverage report -m --fail-under=30` (see `.github/workflows/ci.yml`).

### CI/CD workflows

- **CI (GitHub Actions)** — `.github/workflows/ci.yml`

  - `lint` (py311): sets up Python 3.11 with `uv sync --frozen`, then runs `ruff` against `src/` using `ruff.toml`.
  - `test` (py310/py311/py312): matrix over Python 3.10, 3.11, 3.12; each job uses `uv sync --frozen`, installs `pytest` and `coverage`, runs `coverage run --source=src -m pytest --ignore=proxy_test.py`, prints `coverage report -m --fail-under=30`, and uploads `coverage.xml` as an artifact (treats "no tests collected" as success).
  - `typecheck` (py311): runs `mypy` against `src/helpers` and `src/save_dataset.py` using the configuration in `pyproject.toml`.
  - `security` (py311): runs `pip-audit` via `uv` against the synced environment, ignoring a small, explicit set of tracked CVEs while failing on any new vulnerabilities.
  - `docs` (py311): checks documentation quality by running `mdformat` (formatting), `codespell` (spelling), and `lychee` (link checker) over `README.md` and `docs/`.
  - `build` (py311): depends on `lint`, `test`, and `typecheck`; uses `uv` to run `compileall` on `src` and `scripts` and performs an import smoke test for `helpers`, `scrapers`, `runpod`, and `ui`.

- **Release (CD)** — `.github/workflows/release.yml`

  - Triggers on tags matching `v*` (for example, `v0.1.0`) or via manual `workflow_dispatch`.
  - Uses `uv build` to create wheel/sdist into `dist/`, then uploads them as workflow artifacts.
  - If the `PYPI_API_TOKEN` GitHub secret is set, publishes the built artifacts to PyPI via `pypa/gh-action-pypi-publish`.

<a id="license"></a>

## 📄 License

MIT
