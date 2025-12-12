# Dataset Builder API

FineFoundry can build datasets from JSON pairs and save them as Hugging Face `datasets.DatasetDict` objects, with optional push to the Hub.

This page provides a brief overview of the programmatic pieces involved. For now, it is intentionally lightweight and points you to the most important entry points.

______________________________________________________________________

## Main entry points

- `src/save_dataset.py` – scriptable dataset builder that:
  - Reads a JSON file of `{ "input": ..., "output": ... }` pairs.
  - Builds a `DatasetDict` with train/validation/test splits.
  - Saves the dataset locally.
  - Optionally pushes to the Hugging Face Hub and uploads a dataset card.
- Helper functions in `src/helpers/` (for example, normalization and split logic) support this script.

For end‑to‑end CLI usage, see:

- **[CLI Usage](../user-guide/cli-usage.md)** – includes a concrete `save_dataset.py` example.

______________________________________________________________________

## When to use the API vs the GUI

- Use the **GUI** (Build & Publish tab) when you want an interactive workflow.
  - The GUI builds datasets from **database scrape sessions**.
- Use the **CLI/API** when you want repeatable, scripted dataset builds in automation or CI.
  - The CLI entry point (`src/save_dataset.py`) builds datasets from a **JSON file**.
  - If your data lives in the database, export a scrape session to JSON first, then run `src/save_dataset.py`.

Future iterations of this page may add detailed function references and configuration objects, but the recommended starting point today is to:

- Copy the `src/save_dataset.py` pattern, and
- Adapt it to your own JSON input and split requirements.
