# Contributing Guide

Thanks for wanting to contribute to FineFoundry! This guide covers setting up your development environment, making changes, and getting your PR ready for review.

For more on tests and CI, see the [Testing Guide](testing.md) and [Developer FAQ](faq.md).

______________________________________________________________________

## Getting Started

You'll need Python 3.10+ and we recommend using `uv` for dependency management.

```bash
git clone https://github.com/SourceBox-LLC/FineFoundry.git FineFoundry-Core
cd FineFoundry-Core
uv sync
```

If you prefer pip and venv, see the [Installation Guide](../user-guide/installation.md).

To run the app during development:

```bash
uv run src/main.py
```

Or use the launcher script (`./run_finefoundry.sh` on macOS/Linux).

______________________________________________________________________

## Branches and Workflow

Base your work on the default branch (usually `master`) and create a feature branch for each change—something like `feat/new-scraper`, `fix/training-oom`, or `docs/update-user-guide`.

Keep your PRs focused. Small, incremental changes are easier to review and debug. Avoid mixing large refactors with behavior changes in the same PR.

When you open a pull request, clearly describe what you changed and why. Call out behavioral changes, UI changes, or new dependencies. Mention any follow-up work you're leaving for a later PR.

______________________________________________________________________

## Local Checks Before a PR

Run these checks locally before pushing—it keeps CI green and speeds up review.

### Linting

```bash
uv run ruff check src
uv run ruff check src --fix  # auto-fix many issues
```

### Tests

Run unit tests for a fast feedback loop, integration tests when changing core flows, or the full suite:

```bash
uv run pytest tests/unit                # unit tests only
uv run pytest -m "integration"          # integration tests only
uv run pytest                           # full suite
```

See the [Testing Guide](testing.md) for more on test structure and markers.

### Coverage

CI enforces a minimum coverage threshold. Check locally with:

```bash
uv run coverage run --source=src -m pytest
uv run coverage report -m
```

### Type Checking

```bash
uv run mypy
```

When adding type annotations, prefer precise types over `Any`. If you need `# type: ignore`, keep it narrow and use a specific error code.

### Docs

CI validates documentation formatting, spelling, and links:

```bash
uv run mdformat README.md docs
uv run codespell README.md docs
```

______________________________________________________________________

## Adding Features or Fixes

When you add something new or fix a bug, include tests—unit tests for helper logic, integration tests for end-to-end flows. Update the relevant docs too, whether that's user-facing guides in `docs/user-guide/` or developer docs in `docs/development/`.

Keep code, tests, and docs in the same PR so they stay in sync.

______________________________________________________________________

## Style and Architecture

Follow the existing patterns in `src/helpers/` and `src/ui/`. Use clear, explicit names. Log with the central logging system instead of `print`. For GUI code, keep layout and controller logic separated.

See the [Code Style Guide](code-style.md) for more details.

______________________________________________________________________

## Questions?

If you're not sure how to approach something, check the [Developer FAQ](faq.md), open a draft PR early and ask in the description, or start a discussion in GitHub Issues.

We welcome thoughtful, well-tested contributions!
