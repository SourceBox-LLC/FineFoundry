# Testing Guide
+
+How to run and understand the tests for FineFoundry.
+
+## Test Layout
+
+Tests live under the top-level `tests/` directory:
+
+```text
+tests/
+├── unit/          # Fast unit tests for helpers and core scripts
+├── integration/   # End-to-end flows and UI/controller smoke tests
+└── fixtures/      # Optional shared test data (add as needed)
+```
+
+Pytest discovery is configured via `pyproject.toml`:
+
+```toml
+[tool.pytest.ini_options]
+testpaths = ["tests"]
+pythonpath = ["src"]
+markers = [
+  "integration: marks tests as integration (deselect with '-m \"not integration\"')",
+]
+```
+
+## Unit Tests (`tests/unit/`)
+
+Unit tests cover focused, deterministic behavior without external services:
+
+- `test_save_dataset.py` / `test_save_dataset_card.py` – normalization, split logic, size buckets, truncation, and dataset card content.
+- `test_datasets.py` – column guessing for HF datasets.
+- `test_proxy.py` – proxy env parsing and application to scrapers.
+- `test_local_specs.py` – local hardware/specs helpers.
+- `test_settings_ollama.py` / `test_settings_ollama_http.py` – Ollama settings and HTTP client behavior.
+- `test_chatml.py` – ChatML conversation builders.
+- `test_training.py` – hyperparameter extraction from Training tab controls.
+- `test_common.py` – safe UI update and terminal title helpers.
+- `test_logging_config.py` / `test_logging_config_levels.py` – logger setup and global log-level control.
+
+Run unit tests only:
+
+```bash
+uv run pytest tests/unit
+```
+
+## Integration Tests (`tests/integration/`)
+
+Integration tests exercise realistic flows and minimal UI wiring. All are marked with `@pytest.mark.integration`.
+
+Current integration tests include:
+
+- `test_save_dataset_main.py`
+  - Runs `save_dataset.main()` end-to-end against a tiny JSON file (standard pairs + ChatML).
+  - Asserts an HF `DatasetDict` is saved to disk and has the expected structure.
+
+- `test_merge_json_interleave.py`
+  - Uses the merge helpers to interleave two small JSON datasets into a single JSON output.
+  - Verifies the interleave order and that the download button and snackbar state look correct.
+
+- `test_tab_controllers_smoke.py`
+  - Instantiates Scrape, Build, Merge, Analysis, Training, and Inference tab controllers with a minimal `DummyPage`.
+  - Verifies each controller can build its tab content and, for Training/Infernce, that a shared `train_state` dict is returned.
+
+Run only integration tests:
+
+```bash
+uv run pytest -m "integration"
+```
+
+Run all tests except integration (fast inner loop):
+
+```bash
+uv run pytest -m "not integration"
+```
+
+Run the full suite:
+
+```bash
+uv run pytest
+```
+
+## Async Tests and AnyIO
+
+Some tests are async and use `pytest-anyio` via the `anyio` plugin. A local `anyio_backend` fixture in `tests/conftest.py` restricts tests to the `asyncio` backend so no extra async libraries (like `trio`) are required.
+
+## Coverage
+
+FineFoundry uses `coverage.py` to measure test coverage, especially in CI.
+
+### Local Coverage
+
+Install coverage once into your environment:
+
+```bash
+uv pip install coverage
+```
+
+Then run tests under coverage (measuring only project code under `src/`) and show a summary:
+
+```bash
+uv run coverage run --source=src -m pytest
+uv run coverage report -m
+```
+
+You can also generate an XML report (for IDEs or external tools):
+
+```bash
+uv run coverage xml -o coverage.xml
+```
+
+### CI Coverage and Quality Gates

The `test` job in `.github/workflows/ci.yml` runs `pytest` under coverage for Python 3.10, 3.11, and 3.12. For each matrix entry it:

- Installs `pytest` and `coverage` via `uv`.
- Runs `coverage run -m pytest --ignore=proxy_test.py`.
- Enforces an initial minimum coverage threshold with `coverage report -m --fail-under=20`.
  This threshold is intentionally set close to the current overall coverage so it acts as a
  **floor that prevents regressions** while leaving room to gradually increase it as more
  tests are added.
- Exports `coverage.xml` and uploads it as a GitHub Actions artifact.

Additional quality jobs in CI include:

- **Typecheck (`typecheck` job)** – Runs `mypy` against `src/helpers` and `src/save_dataset.py` using the configuration in `pyproject.toml`.
- **Security audit (`security` job)** – Uses `pip-audit` (via `uv`) to scan the synced environment for known dependency vulnerabilities. The job is configured to
  ignore a small, explicit set of CVEs that currently come only from transitive dependencies and are tracked separately:
  - `CVE-2025-6176` (affecting `brotli` 1.1.0; fixed in 1.2.0)
  - `CVE-2025-62727` (affecting `starlette` 0.48.0; fixed in 0.49.1)
  - `CVE-2025-66418` (affecting `urllib3` 2.5.0; fixed in 2.6.0)
  - `CVE-2025-66471` (affecting `urllib3` 2.5.0; fixed in 2.6.0)

  All other vulnerabilities reported by `pip-audit` will still cause the `security` job to fail.

You can download coverage artifacts from the CI run to inspect coverage in detail or feed them into external reporting tools.
+
+## Adding New Tests
+
+When adding new features:
+
+- Prefer unit tests for pure helpers and small functions.
+- Add integration tests for end-to-end flows (e.g., new CLI, multi-step UI flows) where it adds real signal.
+- Keep UI/controller tests as **smoke tests**: instantiate controllers with a simple dummy page, trigger key callbacks, and assert no exceptions and basic state changes.
+
+This keeps the test suite fast while providing high confidence in critical data and training flows.
