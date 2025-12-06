# FineFoundry FAQ

This FAQ focuses on **where to find things in the docs** and how to quickly answer the most common questions about installing, running, testing, and maintaining FineFoundry.

If you are unsure where to start, skim this page first and follow the links that match what you are trying to do.

---

## Getting Started

### I just discovered FineFoundry. Where should I start?

Start with the **Quick Start Guide**:

- [`user-guide/quick-start.md`](quick-start.md)

It walks you through installation, launching the app, and running your first simple flow end-to-end.

For a bird’s-eye view of the interface, also see:

- [`user-guide/gui-overview.md`](gui-overview.md)

### Where are installation instructions and system requirements?

See:

- **Installation** – [`user-guide/installation.md`](installation.md) (OS, Python, dependencies)
- **Quick Start** – [`user-guide/quick-start.md`](quick-start.md) (practical “do this, then this” setup)

If you’re developing locally, also glance at:

- **Project Structure** – [`development/project-structure.md`](../development/project-structure.md)

### What Python version does FineFoundry target?

FineFoundry is developed and tested against **Python 3.10+**.

- For installation details and any version-specific notes, see [`user-guide/installation.md`](installation.md).
- CI runs tests and checks on a matrix of Python versions; see **CI Coverage and Quality Gates** in [`development/testing.md`](../development/testing.md).

---

## Using the Application

### Where can I find documentation for each tab in the GUI?

Start with the high-level overview:

- [`user-guide/gui-overview.md`](gui-overview.md)

Then drill into the tab-specific guides:

- **Scrape Tab** – [`user-guide/scrape-tab.md`](scrape-tab.md)
- **Build & Publish Tab** – [`user-guide/build-publish-tab.md`](build-publish-tab.md)
- **Training Tab** – [`user-guide/training-tab.md`](training-tab.md)
- **Inference Tab** – [`user-guide/inference-tab.md`](inference-tab.md)
- **Merge Datasets Tab** – [`user-guide/merge-tab.md`](merge-tab.md)
- **Analysis Tab** – [`user-guide/analysis-tab.md`](analysis-tab.md)
- **Settings Tab** – [`user-guide/settings-tab.md`](settings-tab.md)

Each tab guide explains the main concepts, key controls, and typical workflows.

### Is there a command-line interface (CLI)?

Yes. CLI usage is documented separately from the GUI:

- [`user-guide/cli-usage.md`](cli-usage.md)

Use that guide when you want to automate dataset building, integrations, or CI workflows without opening the GUI.

### How do I handle authentication for Hugging Face and Runpod?

See:

- **Authentication** – [`user-guide/authentication.md`](authentication.md)
- **Runpod Setup** – [`deployment/runpod.md`](../deployment/runpod.md)

These cover token management, environment variables, and best practices for storing credentials.

---

## Data, Scraping, and Proxies

### Where is scraping behavior documented?

There are two complementary places:

- **GUI flows** – [`user-guide/scrape-tab.md`](scrape-tab.md)
- **Programmatic API** – [`api/scrapers.md`](../api/scrapers.md) and per-source pages:
  - [`api/fourchan-scraper.md`](../api/fourchan-scraper.md)
  - [`api/reddit-scraper.md`](../api/reddit-scraper.md)
  - [`api/stackexchange-scraper.md`](../api/stackexchange-scraper.md)

### How do I configure proxies or Tor for scraping?

See the deployment docs:

- **Proxy Configuration** – [`deployment/proxy-setup.md`](../deployment/proxy-setup.md)

This explains SOCKS proxies, Tor integration, and environment variables used by the helpers.

---

## Training, Inference, and Runpod

### Where can I learn about training models with FineFoundry?

For the GUI-based training flow:

- **Training Tab** – [`user-guide/training-tab.md`](training-tab.md)

For programmatic usage and advanced configuration:

- **Training API** – [`api/training.md`](../api/training.md)

### How do I set up or use Runpod for training?

See:

- **Runpod Setup** – [`deployment/runpod.md`](../deployment/runpod.md)

It explains how the Runpod integration works, how infrastructure is provisioned, and where to configure related settings.

### Where is inference documented?

Use:

- **Inference Tab** – [`user-guide/inference-tab.md`](inference-tab.md)

for the GUI side, and refer to the Training API docs if you need to wire inference into scripts.

---

## Testing, CI, and Code Quality

### How do I run tests locally?

See the Testing guide:

- [`development/testing.md`](../development/testing.md)

In short:

- **Unit tests** live under `tests/unit/`.
- **Integration tests** live under `tests/integration/`.
- Tests are collected by `pytest` (configured in `pyproject.toml`).

The Testing guide shows commands like:

- `uv run pytest` – full suite
- `uv run pytest tests/unit` – only unit tests
- `uv run pytest -m "integration"` – only integration tests
- `uv run pytest -m "not integration"` – fast inner loop without integration tests

### How is coverage measured, and what is the CI threshold?

Coverage is handled with `coverage.py`:

- The Testing guide documents the recommended local command:
  - `uv run coverage run --source=src -m pytest --ignore=proxy_test.py`
  - `uv run coverage report -m`
- In CI, the `test` job runs under coverage and enforces an initial minimum threshold via:
  - `coverage report -m --fail-under=20`

Details are under **CI Coverage and Quality Gates** in:

- [`development/testing.md`](../development/testing.md)

### Where is mypy/type-checking documented?

Type checking is configured via `mypy` and described in the same Testing guide:

- The **Typecheck (`typecheck` job)** section in [`development/testing.md`](../development/testing.md)
  explains which paths are checked (e.g., `src/helpers`, `src/save_dataset.py`) and how it is wired into CI.

You can run it locally with:

- `uv run mypy`

### What security checks run in CI?

The **Security audit (`security` job)** section in
[`development/testing.md`](../development/testing.md) covers this.

In short:

- CI installs and runs `pip-audit` against the synced environment.
- Two specific CVEs in transitive dependencies are currently ignored in CI and tracked separately:
  - `CVE-2025-6176` (affecting `brotli` 1.1.0; fixed in 1.2.0)
  - `CVE-2025-62727` (affecting `starlette` 0.48.0; fixed in 0.49.1)
- Any **other** vulnerability reported by `pip-audit` will still fail the `security` job.

See **CI Coverage and Quality Gates** in
[`development/testing.md`](../development/testing.md) for the full matrix of CI jobs: `lint`, `test`, `typecheck`, `security`, and `build`.

---

## Development & Internals

### How is the codebase organized?

See:

- **Project Structure** – [`development/project-structure.md`](../development/project-structure.md)

It explains the layout of `src/`, helpers, UI tabs, tests, and deployment-related modules.

### Where can I learn about logging and where logs go?

Use:

- **Logging System** – [`development/logging.md`](../development/logging.md)

This describes how logs are structured, where they are written, and how to use them for troubleshooting.

### Is there a coding style or contribution guide?

Yes:

- **Code Style** – [`development/code-style.md`](../development/code-style.md)
- **Contributing Guide** – [`development/contributing.md`](../development/contributing.md)

These cover formatting, naming conventions, and how to submit changes (including tests and documentation updates).

---

## Troubleshooting & Support

### Where should I look first when something goes wrong?

Two main documents:

- **Troubleshooting** – [`user-guide/troubleshooting.md`](troubleshooting.md)
- **Logging System** – [`development/logging.md`](../development/logging.md)

The troubleshooting guide covers common runtime issues, while the logging guide shows how to gather more detail.

### Where can I ask questions or report bugs?

See the main docs index:

- [`docs/README.md`](../README.md)

for current links to GitHub Issues / Discussions and any other support channels. When filing an issue, include:

- Your OS and Python version
- The FineFoundry version
- Steps to reproduce
- Relevant logs and configuration snippets

---

## How to Use This FAQ

- If you are **a new user**, follow the links in **Getting Started** and **Using the Application**.
- If you are **a contributor**, focus on **Testing, CI, and Code Quality** plus **Development & Internals**.
- If you are **debugging**, jump straight to **Troubleshooting & Support**.

Whenever you add new documentation, consider whether a short pointer here would help future readers discover it quickly.
