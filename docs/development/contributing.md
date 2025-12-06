# Contributing Guide

Thank you for your interest in contributing to FineFoundry!

This stub exists so that documentation links resolve and link checks pass. A more complete contributing guide can be added over time.

## Basics

- Use Python 3.10+.
- Use `uv` for dependency management and commands.
- Run the test and quality suite before opening a pull request:
  - `uv run ruff check src`
  - `uv run pytest`
  - `uv run coverage run --source=src -m pytest --ignore=proxy_test.py && uv run coverage report -m`
  - `uv run mypy`

For documentation about the test layout and CI, see:

- [Testing Guide](testing.md)
- [Developer FAQ](faq.md)
