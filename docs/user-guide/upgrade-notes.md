# Upgrade Notes

This page highlights important behavior changes across recent FineFoundry releases. If you are returning to FineFoundry after some time, read this before following older guides or tutorials.

______________________________________________________________________

## Summary of major changes

### 1. Database-first workflows (SQLite)

FineFoundry now uses a SQLite database (`finefoundry.db`) as the primary system of record for:

- Scrape sessions and scraped pairs
- Training configurations
- Training runs and run metadata
- Logs
- App settings

**What changed:** Many workflows that used to default to JSON files now save to the database by default.

### 2. Scrape tab: "data sources" UX

The Scrape tab is organized around a **Source** selector (4chan / Reddit / Stack Exchange / Synthetic).

**What changed:** Documentation and UI now consistently refer to a "data source" rather than selecting a "scraper".

### 3. Build & Publish: build from database sessions

The Build & Publish tab builds datasets from **database scrape sessions**.

**What changed:** The GUI no longer presents a workflow where you build a dataset directly from arbitrary JSON files or Hugging Face datasets.

### 4. Inference: select completed training runs

The Inference tab runs inference against adapters from **completed training runs**.

**What changed:** The GUI no longer asks you to browse to an adapter directory manually. FineFoundry loads the adapter path tracked by the selected training run and validates it automatically.

### 5. Offline Mode gating

When **Offline Mode** is enabled, FineFoundry disables actions that require external services.

**Examples:**

- Hugging Face Hub push actions are disabled.
- Hugging Face dataset sources are disabled.
- Runpod cloud training is disabled.
- Scrape tab network sources are disabled (Synthetic remains available).

The UI keeps controls visible where helpful, but disables them and shows inline explanations.

### 6. Dependency management: `uv` + `pyproject.toml`

FineFoundry uses `uv` and `pyproject.toml` for dependency management.

**What changed:** `requirements.txt` should be considered deprecated for this repo.

______________________________________________________________________

## Action checklist when upgrading

- If you used to rely on JSON outputs from scraping, plan to use the database instead.
- If you need a JSON file for external tools:
  - Export a scrape session from the database explicitly, then use it as input to your external workflow.
- If you previously used manual adapter directory selection for inference:
  - Switch to selecting a completed training run in the Inference tab.
- If you are offline or working in an air-gapped environment:
  - Enable Offline Mode and expect Hub/Runpod actions to be disabled.

______________________________________________________________________

## Related topics

- [Quick Start Guide](quick-start.md)
- [Scrape Tab](scrape-tab.md)
- [Build & Publish Tab](build-publish-tab.md)
- [Training Tab](training-tab.md)
- [Inference Tab](inference-tab.md)
- [CLI Usage](cli-usage.md)
