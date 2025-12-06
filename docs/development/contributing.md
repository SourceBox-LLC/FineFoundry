# Contributing Guide

Thank you for your interest in contributing to FineFoundry!

This guide summarizes how to set up a development environment, how to structure changes, and which checks to run before opening a pull request.

For deeper background on tests and CI, see:

- [Testing Guide](testing.md)
- [Developer FAQ](faq.md)

______________________________________________________________________

## Getting Started

- Use **Python 3.10+**.
- Use **`uv`** for dependency management and commands (recommended).

Clone the repository:

```bash
git clone https://github.com/SourceBox-LLC/FineFoundry.git FineFoundry-Core
cd FineFoundry-Core
```

Install dependencies using `uv`:

```bash
uv sync
```

or, if you prefer, follow the user **[Installation](../user-guide/installation.md)** guide for a `venv` + `pip` setup.

To run the app during development:

```bash
uv run src/main.py
```

______________________________________________________________________

## Branches and Workflow

- Base your work on the current default branch (usually `master`).
- Create a feature branch for each logical change (for example, `feat/new-scraper`, `fix/training-oom`, `docs/update-user-guide`).
- Keep changes focused:
  - Small, incremental PRs are easier to review and debug.
  - Avoid mixing large refactors with behavior changes.

When you open a pull request:

- Clearly describe **what** you changed and **why**.
- Call out any behavioral changes, UI changes, or new external dependencies.
- Mention any follow‑up work you intentionally left for a later PR.

______________________________________________________________________

## Local Checks Before a PR

Run these checks locally before pushing. This keeps CI green and shortens review time.

### 1. Lint (Ruff)

```bash
uv run ruff check src
```

Fix issues as needed. You can often auto‑fix many of them:

```bash
uv run ruff check src --fix
```

See `ruff.toml` for the current configuration (Python 3.10 target, 120‑column line length, and a small set of ignored rules).

### 2. Tests (Pytest)

Run the unit tests for a fast inner loop:

```bash
uv run pytest tests/unit
```

When changing core flows, also run integration tests:

```bash
uv run pytest -m "integration"          # only integration tests
uv run pytest -m "not integration"      # everything except integration tests
```

For a full run (what CI effectively does):

```bash
uv run pytest
```

See the [Testing Guide](testing.md) for details about the test layout and markers.

### 3. Coverage

Coverage is enforced in CI with a minimum threshold. To check coverage locally:

```bash
uv pip install coverage
uv run coverage run --source=src -m pytest --ignore=proxy_test.py
uv run coverage report -m
```

If coverage drops significantly for critical modules, consider adding tests before submitting the PR.

### 4. Type Checking (mypy)

FineFoundry uses `mypy` with configuration in `pyproject.toml` (currently focused on `src/helpers` and `src/save_dataset.py`, with some overrides for runtime‑heavy modules).

Run type checking locally:

```bash
uv run mypy
```

When touching typed modules, prefer tightening annotations rather than adding broad `ignore` rules. If you must add `# type: ignore[...]`, include a brief comment or pick the most specific code to ignore.

### 5. Docs (formatting, spelling, links)

Documentation is validated in CI by a dedicated `docs` job that runs `mdformat`, `codespell`, and `lychee`.

Locally, you can run:

```bash
uv run mdformat README.md docs
uv run codespell README.md docs
```

- `mdformat` enforces consistent Markdown formatting.
- `codespell` catches common spelling mistakes.

CI also runs **lychee** to validate links in `README.md` and `docs/**/*.md`. You typically do not need to run lychee locally unless you are working heavily on docs or debugging link issues.

______________________________________________________________________

## Adding Features or Fixes

When adding a new feature or fixing a bug:

- **Add or update tests**:
  - Unit tests for helpers and logic in `src/helpers/`.
  - Integration tests for end‑to‑end flows (for example, new CLI entrypoints or multi‑step GUI flows using controllers).
  - See "Adding New Tests" in the [Testing Guide](testing.md) for guidance.
- **Update documentation**:
  - User‑facing changes: update relevant pages under `docs/user-guide/`.
  - Developer changes: update `docs/development/` and `docs/api/` as needed.

Try to keep code, tests, and docs changes in the same PR so they stay in sync.

______________________________________________________________________

## Style and Architecture

A few high‑level conventions:

- Follow the existing patterns in `src/helpers/` and `src/ui/`.
- Prefer clear, explicit names over clever abbreviations.
- Log meaningful events via the central logging configuration instead of using `print`.
- For the GUI, keep **layout** code and **controller/logic** code separated, following the tab controller pattern described in the main `README.md` and UI docs.

For more details, see the **[Code Style](code-style.md)** page.

______________________________________________________________________

## Questions and Support

If you are unsure about how to implement something or how strict a rule is meant to be:

- Check the **[Developer FAQ](faq.md)**.
- Open a draft PR early and ask questions in the description.
- Use GitHub Issues or Discussions on the main repository to discuss larger changes before investing a lot of time.

Thoughtful, well‑tested, and well‑documented contributions are very welcome.
