# Code Style

This guide covers the coding conventions we use in FineFoundry. The goal is consistency and readability while staying pragmatic for a GUI-heavy application.

______________________________________________________________________

## Basics

We target Python 3.10+ with a 120-character line length (configured in `ruff.toml`). Use modern Python idioms—f-strings, type hints, context managers—and let Ruff handle formatting:

```bash
uv run ruff check src --fix
```

## Imports

Keep imports at the top of the file, grouped logically: standard library first, then third-party packages, then local imports. Avoid wildcard imports. When in doubt, mirror existing modules in `src/helpers/`.

## Naming and Error Handling

Use descriptive names like `scrape_threads`, `build_dataset`, `merge_pairs`—not cryptic abbreviations. Fail fast with clear exceptions. When catching errors, log enough context to diagnose the problem without leaking sensitive data.

## Logging

Use the central logging system instead of `print`. Get a logger with `get_logger(__name__)` so logs go to the database. Use appropriate levels (debug, info, warning, error) and avoid spamming logs in tight loops. See the [Logging Guide](logging.md) for more.

## Type Hints

New or heavily edited code should be type-annotated. Prefer precise types (`list[str]`, `dict[str, Any]`) over bare `list`/`dict`. Avoid `Any` unless necessary. Run `uv run mypy` before submitting.

If you need `# type: ignore`, keep it narrow and use a specific error code.

## GUI Patterns

The Flet-based GUI uses a controller pattern. Layout code lives in `ui/tabs/tab_*.py` and section modules. Controllers in `ui/tabs/*_controller.py` handle events, state, and business logic. Keep layout and logic separated, prefer small composable functions, and reuse helpers from `src/helpers/`.

## Before a PR

Run Ruff, tests, coverage, and mypy as described in the [Contributing Guide](contributing.md). Update docs for user-facing changes. Keep changes consistent with nearby code.
