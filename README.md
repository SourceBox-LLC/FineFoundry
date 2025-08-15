# Dataset Studio

A desktop GUI and CLI toolkit to scrape conversational pairs from 4chan and publish a clean, versioned dataset to the Hugging Face Hub.

Built with Flet for a fast, native-like UI. Ships with:
- Scraper that pairs adjacent replies from 4chan threads and cleans markup/URLs.
- Dataset builder that creates train/val/test splits and saves with `datasets`.
- One-click push to the Hugging Face Hub, including an auto-generated dataset card (README.md).


## Contents
- `src/main.py` — Flet desktop app (Scrape + Build/Publish tabs)
- `src/scraper.py` — 4chan scraper and text cleaners
- `src/save_dataset.py` — CLI dataset builder and Hub pusher
- `requirements.txt` — pinned dependencies


## Prerequisites
- Python 3.10+ on Windows/macOS/Linux
- Git (optional)

Optional (for pushing to Hugging Face):
- A Hugging Face account and an access token with write permissions: https://huggingface.co/settings/tokens


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

The app opens a desktop window. Use the two tabs: Scrape and Build / Publish.


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


## CLI: Build and Push Without the GUI
`src/save_dataset.py` provides a fully scriptable path.

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


## How Scraping Works
- Uses the 4chan public JSON API (`a.4cdn.org`).
- Samples threads across catalog pages to diversify coverage.
- Pairs sequential posts within threads (`input` = previous post, `output` = next post).
- Cleans HTML, removes greentext quote lines, quote references, URLs, and collapses whitespace.
- Minimum length filter is applied per post.

Key module: `src/scraper.py` (functions such as `fetch_catalog_pages()`, `fetch_thread()`, `build_pairs_adjacent()`).


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


## Ethical and Legal Notes
- Content may be NSFW, offensive, or harmful. For research purposes only.
- Respect platform policies and applicable laws in your jurisdiction.
- Before any production use, apply filtering, detoxification, and alignment techniques.


## Development
- UI built with Flet (`flet`, `flet-desktop`), pinned in `requirements.txt`.
- Run formatting/linting as desired. Contributions welcome via PR.


## License
MIT
