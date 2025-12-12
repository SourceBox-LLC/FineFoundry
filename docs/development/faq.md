# FineFoundry Developer FAQ

This FAQ is for **contributors and advanced users** working on the FineFoundry codebase itself: tests, CI, typing, security checks, and internals.

If you just want to use the app, start with the **User FAQ** instead:

- [`user-guide/faq.md`](../user-guide/faq.md)

______________________________________________________________________

## Getting Started as a Contributor

### I want to understand the codebase layout. Where should I look?

See:

- **Project Structure** – [`development/project-structure.md`](project-structure.md)

It explains where helpers, UI tabs, scrapers, tests, and deployment-related code live under `src/`.

### How do I set up a development environment?

Key points:

- FineFoundry targets **Python 3.10+**.
- Dependencies and tooling are managed with **`uv`** and `pyproject.toml`.
- CI runs a matrix of checks (lint, tests with coverage, mypy, pip-audit, build).

For a quick orientation, combine:

- [`development/project-structure.md`](project-structure.md)
- [`development/testing.md`](testing.md)
- The main CI workflow: [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)

______________________________________________________________________

## Tests and Test Layout

### Where are the tests, and how are they organized?

See the Testing guide:

- [`development/testing.md`](testing.md)

In short:

- **Unit tests** live in `tests/unit/` and cover helpers, `save_dataset`, and other core logic.
- **Integration tests** live in `tests/integration/` and cover end-to-end flows and UI/controller smoke tests.

### How do I run tests locally?

Common commands (using `uv`):

- Full suite:
  - `uv run pytest`
- Only unit tests:
  - `uv run pytest tests/unit`
- Only integration tests:
  - `uv run pytest -m "integration"`
- Everything except integration tests (fast inner loop):
  - `uv run pytest -m "not integration"`

The Testing guide also covers async testing (AnyIO/asyncio) and how UI/controller smoke tests are structured.

### How is coverage measured?

Coverage is handled with `coverage.py` and integrated with `pytest`:

- Recommended local command:
  - `uv run coverage run --source=src -m pytest --ignore=proxy_test.py`
  - `uv run coverage report -m`

In CI:

- Each `test` job run executes the suite under coverage and enforces a minimum threshold via:
  - `coverage report -m --fail-under=30`
- XML reports (`coverage.xml`) are produced per Python version for artifacts or external tools.

Details live under **CI Coverage and Quality Gates** in:

- [`development/testing.md`](testing.md)

______________________________________________________________________

## Linting and Code Style

### What linter is used, and how do I run it?

FineFoundry uses **Ruff** for linting.

- Local command:
  - `uv run ruff check src`

The CI `lint` job:

- Sets up Python 3.11.
- Runs `uv sync --frozen` to install dependencies.
- Runs `ruff` against the `src/` tree according to `ruff.toml`.

### How strict is the code style?

In practice:

- Imports and unused symbols are enforced via Ruff.
- New code should follow the existing patterns in helpers and UI controllers.
- Prefer small, testable helpers and explicit error handling/logging.

When in doubt, copy the style of nearby modules (especially in `src/helpers/`).

______________________________________________________________________

## Type Checking (mypy)

### How is mypy configured, and what does it check?

Configuration lives in `pyproject.toml` under `[tool.mypy]`.

Key points:

- `python_version` is set to **3.10** and `mypy_path` to `src`.
- `files` focuses on core helpers and `src/save_dataset.py`.
- `ignore_missing_imports = true` is used so 3rd-party libs without stubs don’t break type checking.
- Some heavy runtime-oriented modules (scrapers and certain helpers) are explicitly marked with `ignore_errors` overrides so they do not block CI while type coverage is being improved.

### How do I run mypy locally?

Use:

- `uv run mypy`

This uses the configuration in `pyproject.toml` and should match what CI runs in the `typecheck` job.

If you add new helper modules or significantly change types, consider tightening annotations and running mypy as part of your inner loop.

______________________________________________________________________

## Security and Dependency Auditing

### What security checks are run in CI?

CI uses **`pip-audit`** to scan the synced environment for known vulnerabilities.

- The `security` job installs `pip-audit` via `uv` and runs it against the installed packages.
- Several specific CVEs/GHSAs in transitive dependencies are currently ignored (and tracked separately):
  - `CVE-2025-6176` (affecting `brotli` 1.1.0; fixed in 1.2.0)
  - `CVE-2025-62727` (affecting `starlette` 0.48.0; fixed in 0.49.1)
  - `CVE-2025-66418` (affecting `urllib3` 2.5.0; fixed in 2.6.0)
  - `CVE-2025-66471` (affecting `urllib3` 2.5.0; fixed in 2.6.0)
  - `GHSA-f83h-ghpp-7wcc` (affecting `pdfminer-six` ≤20251107; no fix yet, local privesc only)
- Any **other** vulnerabilities reported by `pip-audit` will fail the `security` job.

The exact command and ignore list are maintained in:

- [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
- Described in the **Security audit (`security` job)** section of [`development/testing.md`](testing.md)

### How should I handle new vulnerabilities that pip-audit finds?

General guidance:

1. **Prefer upgrading** the affected dependency (or its parent package) when possible.
1. If a vulnerability only affects an optional feature you do not use, document that and consider an ignore rule.
1. Avoid piling on ignore rules—treat each one as a conscious decision that should be documented.

Open a PR that:

- Updates dependencies or adds a targeted ignore, and
- Explains the rationale in the PR description.

______________________________________________________________________

## CI Jobs Overview

### What jobs run in CI?

The main jobs (as configured in `.github/workflows/ci.yml`) are:

- **`lint`** – Runs Ruff on `src/`.
- **`test`** – Matrix over Python 3.10/3.11/3.12; runs `pytest` under coverage with a minimum coverage threshold and uploads `coverage.xml`.
- **`typecheck`** – Runs `mypy` with the configuration in `pyproject.toml`.
- **`security`** – Runs `pip-audit` with a small, explicit ignore list for known transitive issues.
- **`build`** – Depends on other jobs and performs a compile/import smoke test of the codebase.

See the Testing guide for how these jobs relate to local commands:

- [`development/testing.md`](testing.md)

______________________________________________________________________

## Troubleshooting as a Developer

### Where should I look when something fails in CI?

1. Open the failing job in GitHub Actions and inspect the logs for the specific step.
1. Cross-check the corresponding section in:
   - [`development/testing.md`](testing.md)
   - [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
1. Try to reproduce locally using the same `uv` command shown in the workflow.

### Where can I get more detail via logs?

See:

- **Logging System** – [`development/logging.md`](logging.md)

for how logs are emitted and where to find them. This is especially useful when debugging scrapers, training flows, or background processes.

______________________________________________________________________

## How to Use This FAQ

- If you are **fixing CI**, focus on sections for **Tests and Test Layout**, **Type Checking**, **Security**, and **CI Jobs Overview**.
- If you are **adding features**, read **Getting Started as a Contributor** and **Tests and Test Layout** so new code ships with tests.
- If you are **triaging bugs**, combine the advice here with the **Logging System** docs.

When you add new developer-oriented docs, consider adding a short Q&A here pointing readers to them.
