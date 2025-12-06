# Code Style

This page describes the main coding standards and conventions used in FineFoundry.

The goal is to keep the codebase consistent, easy to read, and easy to reason about, while staying pragmatic for a GUI‑heavy application.

______________________________________________________________________

## Python Version and Formatting

- Target **Python 3.10+**.
- The default line length is **120 characters** (configured in `ruff.toml`). Avoid excessively long lines, but do not contort code purely to hit 79 characters.
- Prefer standard modern Python idioms (f‑strings, type hints, context managers, etc.).

Formatting is primarily enforced by **Ruff** and the configured rules. You can auto‑fix many issues via:

```bash
uv run ruff check src --fix
```

______________________________________________________________________

## Imports and Module Layout

- Keep imports at the **top of the file** unless there is a clear, documented reason (for example, optional heavy dependencies in CLI tools).
- Group imports logically:
  - Standard library
  - Third‑party packages
  - Local imports from `helpers`, `scrapers`, `runpod`, `ui`, etc.
- Avoid wildcard imports (`from module import *`). Import exactly what you need.
- For helpers that may be reused, keep modules focused and small rather than creating large "grab‑bag" files.

When in doubt, mirror the structure of existing modules in `src/helpers/`.

______________________________________________________________________

## Naming, Errors, and Logging

- Use descriptive names for functions, classes, and variables.
- Prefer explicit names like `scrape_threads`, `build_dataset`, `merge_pairs` over very short or ambiguous names.

Error handling:

- Fail fast with clear exceptions where appropriate.
- When catching exceptions, log enough context to diagnose the problem (parameters, external services involved) without leaking sensitive information.

Logging:

- Use the central logging configuration instead of `print`.
- Log via the logger obtained from the logging helpers so logs appear in the correct files under `logs/`.
- Use appropriate log levels (`debug`, `info`, `warning`, `error`) and avoid spamming logs inside tight loops.

For more detail on logging, see the **[Logging Guide](logging.md)**.

______________________________________________________________________

## Type Hints and mypy

FineFoundry uses **type hints** and **mypy** to catch errors early, with configuration in `pyproject.toml`.

- New or heavily edited code should be type‑annotated.
- Prefer precise types (`list[str]`, `dict[str, Any]`, custom `TypedDict` or dataclasses) over bare `list`/`dict`.
- Avoid `Any` unless strictly necessary; if you must use it, keep it localized.
- Use `Optional[...]` (or `| None` syntax where appropriate) to make `None`‑able values explicit.

Run mypy locally:

```bash
uv run mypy
```

The configuration currently focuses on `src/helpers` and `src/save_dataset.py`, with explicit `ignore_errors` overrides for some runtime‑heavy modules and scrapers. Over time, new code should aim to **reduce** the need for ignores rather than add more.

If you need a `# type: ignore[...]` comment, keep it as narrow as possible and use a concrete error code where you can.

______________________________________________________________________

## Ruff Configuration

Ruff is configured in `ruff.toml` to:

- Target `py310`.
- Use a 120‑column line length.
- Ignore a small set of rules by default, such as:
  - `E501` (line too long) to avoid noisy reports while the codebase is still being aligned.
  - `E402` (module level import not at top of file) where some CLI scripts intentionally import after argument parsing.

As you touch code, it’s fine to gradually make it cleaner:

- Fix style issues reported by Ruff where it improves readability.
- Avoid introducing new long lines without a good reason.

Run Ruff locally before opening a PR:

```bash
uv run ruff check src
```

______________________________________________________________________

## GUI and Controller Patterns

The Flet‑based GUI follows a **controller pattern**:

- Layout‑only builders live in `ui/tabs/tab_*.py` and per‑tab section modules under `ui/tabs/<tab>/sections/`.
- Controllers live in `ui/tabs/*_controller.py` and are responsible for wiring events, state, and helpers.
- `src/main.py` should remain thin and delegate to controller functions such as `build_scrape_tab_with_logic(...)`.

When adding or modifying UI code:

- Keep UI layout and business logic separated.
- Prefer small, composable functions over very large controller methods.
- Reuse shared helper functions from `src/helpers/` instead of duplicating logic in controllers.

For a high‑level overview, see the **Controller pattern** section in the main `README.md` and the tab‑specific docs under `docs/user-guide/`.

______________________________________________________________________

## Before Opening a PR

Before you submit a pull request:

- Run Ruff, tests, coverage, and mypy as described in the **[Contributing Guide](contributing.md)**.
- Ensure any user‑facing changes have corresponding updates in `docs/user-guide/`.
- Keep changes consistent with nearby code; when in doubt, follow the style of existing helpers and controllers.

Consistent style and clear structure make it easier for everyone to understand and extend FineFoundry.
