# Testing Guide

How to run and understand the tests for FineFoundry.

## Test Layout

Tests live under the top-level `tests/` directory:

```text
tests/
├── unit/          # Fast unit tests for helpers and core scripts
├── integration/   # End-to-end flows and UI/controller smoke tests
└── fixtures/      # Optional shared test data (add as needed)
```

Pytest discovery is configured via `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
markers = [
  "integration: marks tests as integration (deselect with '-m "not integration"')",
]
```

## Unit Tests (`tests/unit/`)

Unit tests cover focused, deterministic behavior without external services:

- `test_save_dataset.py` / `test_save_dataset_card.py` – normalization, split logic, size buckets, truncation, and dataset card content.
- `test_datasets.py` – column guessing for HF datasets.
- `test_proxy.py` – proxy env parsing and application to scrapers.
- `test_local_specs.py` – local hardware/specs helpers.
- `test_settings_ollama.py` / `test_settings_ollama_http.py` – Ollama settings and HTTP client behavior.
- `test_chatml.py` – ChatML conversation builders.
- `test_training.py` – hyperparameter extraction from Training tab controls.
- `test_common.py` – safe UI update and terminal title helpers.
- `test_logging_config.py` / `test_logging_config_levels.py` – logger setup and global log-level control.
- `test_synthetic.py` – synthetic data generation helpers (URL detection, config creation, format conversion).
- `test_scraped_data.py` – database CRUD for scrape sessions and pairs.
- `test_scrape_db.py` – high-level scrape database helpers.
- `test_fourchan_scraper.py` – 4chan scraper text cleaning and pair building strategies.
- `test_reddit_scraper.py` – Reddit scraper utilities (URL handling, text cleaning, question detection).
- `test_stackexchange_scraper.py` – Stack Exchange HTML-to-text conversion and session config.
- `test_chatml_builders.py` – ChatML conversation builders for 4chan and Reddit threads.
- `test_migrate.py` – JSON to SQLite migration and export functions.
- `test_db_settings.py` – database settings CRUD and convenience functions.
- `test_synthetic_cli.py` – synthetic data CLI argument parsing and validation.
- `test_boards.py` – 4chan board list and API loading.
- `test_ui_helpers.py` – UI utility functions (opacity, two-column layout, cell text).
- `test_scraper_utils.py` – shared HTTP retry / rate limiting utilities for scrapers.
- `test_local_inference.py` – local inference prompt shaping, chat templates, and repetition penalty wiring.
- `test_scrape_orchestration.py` – Scrape-tab orchestration for 4chan/Reddit, including DB-save flows.
- `test_merge.py` – dataset merge orchestration (DB/HF/JSON sources, interleave/concatenate) and DB writes.
- `test_build.py` – Publish split validation, DB-source pipeline, and push preconditions.
- `test_training_config.py` – database-backed training config helpers (list/save/delete/rename/last-used/validate).
- `test_training_controller_local_infer.py` – UI-level wiring for Quick Local Inference (prompts, sliders, chat export) using mocks.

Run unit tests only:

```bash
uv run pytest tests/unit
```

## Integration Tests (`tests/integration/`)

Integration tests exercise realistic flows and minimal UI wiring. All are marked with `@pytest.mark.integration`.

Current integration tests include:

- `test_save_dataset_main.py`

  - Runs `save_dataset.main()` end-to-end against a tiny JSON file (standard pairs + ChatML).
  - Asserts an HF `DatasetDict` is saved to disk and has the expected structure.

- `test_synthetic_e2e.py`

  - End-to-end tests for synthetic data generation flows.
  - Tests dedup stats, resume flow, standard format, and empty data handling.

- `test_tab_controllers_smoke.py`

  - Instantiates Scrape, Build, Merge, Analysis, Training, and Inference tab controllers with a minimal `DummyPage`.
  - Verifies each controller can build its tab content and, for Training/Inference, that a shared `train_state` dict is returned.

- `test_synthetic_cli_integration.py`

  - Tests the synthetic CLI with mocked model loading.
  - Verifies error handling for CUDA, OOM, and missing dependencies.
  - Tests config file loading and format conversion.

Run only integration tests:

```bash
uv run pytest -m "integration"
```

Run all tests except integration (fast inner loop):

```bash
uv run pytest -m "not integration"
```

Run the full suite:

```bash
uv run pytest
```

## Async Tests and AnyIO

Some tests are async and use `pytest-anyio` via the `anyio` plugin. A local `anyio_backend` fixture in `tests/conftest.py` restricts tests to the `asyncio` backend so no extra async libraries (like `trio`) are required.

## Coverage

FineFoundry uses `coverage.py` to measure test coverage, especially in CI.

### Local Coverage

Install coverage once into your environment:

```bash
uv pip install coverage
```

Then run tests under coverage (measuring only project code under `src/`) and show a summary:

```bash
uv run coverage run --source=src -m pytest
uv run coverage report -m
```

You can also generate an XML report (for IDEs or external tools):

```bash
uv run coverage xml -o coverage.xml
```

### CI Coverage and Quality Gates

The `test` job in `.github/workflows/ci.yml` runs `pytest` under coverage for Python 3.10, 3.11, and 3.12. For each matrix entry it:

- Installs `pytest` and `coverage` via `uv`.
- Runs `coverage run -m pytest --ignore=proxy_test.py`.
- Enforces a minimum coverage threshold with `coverage report -m --fail-under=30`.
  This threshold acts as a **floor that prevents regressions** while leaving room to
  gradually increase it as more tests are added.
- Exports `coverage.xml` and uploads it as a GitHub Actions artifact.

Additional quality jobs in CI include:

- **Typecheck (`typecheck` job)** – Runs `mypy` against `src/helpers` and `src/save_dataset.py` using the configuration in `pyproject.toml`.

- **Security audit (`security` job)** – Uses `pip-audit` (via `uv`) to scan the synced environment for known dependency vulnerabilities. The job is configured to
  ignore a small, explicit set of CVEs/GHSAs that currently come only from transitive dependencies and are tracked separately:

  - `CVE-2025-6176` (affecting `brotli` 1.1.0; fixed in 1.2.0)
  - `CVE-2025-62727` (affecting `starlette` 0.48.0; fixed in 0.49.1)
  - `CVE-2025-66418` (affecting `urllib3` 2.5.0; fixed in 2.6.0)
  - `CVE-2025-66471` (affecting `urllib3` 2.5.0; fixed in 2.6.0)
  - `GHSA-f83h-ghpp-7wcc` (affecting `pdfminer-six` ≤20251107; no fix yet, local privesc only)

  All other vulnerabilities reported by `pip-audit` will still cause the `security` job to fail.

- **Docs quality (`docs` job)** – Uses `mdformat` to enforce consistent Markdown formatting, `codespell` to catch common spelling mistakes, and `lychee` to
  check links across `README.md` and `docs/**/*.md`. These checks keep documentation clean and prevent broken links from creeping into the repository.

You can download coverage artifacts from the CI run to inspect coverage in detail or feed them into external reporting tools.

### GUI System Check (Settings Tab)

For non-developers (or for a quick visual check), the **Settings → System Check** panel in the GUI wraps a subset of the commands from this guide:

- Runs targeted `pytest` groups for key feature areas:
  - Scraping utilities and scrape-tab orchestration.
  - Merge and publish pipelines.
  - Training config + local training helpers.
  - Quick Local Inference and its UI wiring.
- Runs the full test suite: `pytest tests`.
- Runs coverage:
  - `coverage run --source=src -m pytest`
  - `coverage report -m`

The panel streams live output into a log view and then shows a grouped **System Health Summary** (Data Collection, Dataset Build, Training & Inference, Overall Health).

Under the hood it uses the same Python environment as your CLI. If you are debugging issues in the System Check, you can reproduce its behavior by running the equivalent `pytest` and `coverage` commands directly in a shell.

## Adding New Tests

When adding new features:

- Prefer unit tests for pure helpers and small functions.
- Add integration tests for end-to-end flows (e.g., new CLI, multi-step UI flows) where it adds real signal.
- Keep UI/controller tests as **smoke tests**: instantiate controllers with a simple dummy page, trigger key callbacks, and assert no exceptions and basic state changes.

This keeps the test suite fast while providing high confidence in critical data and training flows.
