# Training API

This page provides a brief overview of FineFoundry's programmatic training interfaces and where to look in the codebase.

The primary implementation lives under:

- `src/helpers/training.py`
- `src/helpers/training_pod.py`
- `src/runpod/ensure_infra.py`
- `src/runpod/runpod_pod.py`

These modules support both **local Docker training** and **Runpod‑based training** as exposed in the GUI Training tab.

Today, the best way to understand the training API is to:

- Read the **[Training Tab (GUI)](../user-guide/training-tab.md)** guide for high‑level behavior.
- Inspect the helper modules above to see how configuration objects are built and how jobs are launched and monitored.

Future iterations of this page may add concrete examples and function‑level references, but it is already safe to treat the helpers as a stable entry point for advanced, programmatic usage.
