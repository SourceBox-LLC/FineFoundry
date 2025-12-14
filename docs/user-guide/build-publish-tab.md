# Publish Tab

The Publish tab is your hub for publishing artifacts.

In Phase 1, it supports:

- Publishing datasets to the Hugging Face Hub
- Publishing LoRA adapters (from completed training runs) to the Hugging Face Hub

In Phase 2 (planned), it will add full model publishing (merged weights) and more platform targets.

![Publish Tab](../../img/new/ff_build_publish.png)

______________________________________________________________________

## Overview

Typical workflow:

1. Select a **Scrape session** from your database history.
1. Configure **splits**, **min length**, and shuffling.
1. Click **Build Dataset** to create a `DatasetDict` on disk.
1. Optionally enable **Push to Hub**, fill in repo details and token.
1. Click **Push + Upload README** to publish the dataset.

If you also have completed training runs, you can publish an adapter:

1. Select a **completed training run**.
1. Set the **Model repo ID**, privacy, and token.
1. Click **Publish adapter**.

______________________________________________________________________

## Layout at a Glance

### 1. Data Source

- **Database (Scrape session)**
  - Select a scrape session from your database history.
  - The dataset is built from the stored `input`/`output` pairs.

### 2. Split & Filtering Parameters

- **Seed** – Controls shuffling deterministically.
- **Shuffle** – Whether to shuffle before splitting.
- **Validation / Test split sliders** – Fractions for validation and test sets (remainder becomes train).
- **Min Length** – Minimum number of characters for `input` and/or `output` to be kept.

### 3. Output Settings

- **Save dir** – Directory where the built `DatasetDict` is saved.
  - Uses `datasets.DatasetDict.save_to_disk(SAVE_DIR)`.

### 4. Build Actions

- **Build Dataset** button
  - Loads data from the selected source, applies filtering and splitting, and writes the dataset to `Save dir`.
  - Shows logs / status for success or errors.

### 5. Push to Hub

- **Push to Hub** toggle
  - Enables pushing the dataset to the Hugging Face Hub.
- **Repo ID** – e.g., `username/my-first-dataset`.
- **Private** – Whether to create a private dataset repo.
- **HF Token** – Optional if you already authenticated via CLI or `HF_TOKEN` env var.
- **Push + Upload README** button
  - Uploads the dataset and a generated dataset card (`README.md` in the repo).

#### Dataset card (README.md)

If you enable the dataset card editor:

- Use **Load simple template** to start from a clean scaffold.
- Use **Generate with Ollama** to draft an intelligent card from your selected database session.

When you generate with Ollama, the UI shows a small loading spinner and snackbars for start/success/failure.

### 6. Publish model (adapter)

- **Training run** dropdown
  - Selects from completed runs.
  - The adapter folder is uploaded from the run's saved adapter output.
- **Model repo ID** – e.g., `username/my-adapter`.
- **Private** – Whether to create a private model repo.
- **HF Token** – Optional if you already authenticated via CLI or `HF_TOKEN` env var.
- **Publish adapter** button
  - Uploads the adapter folder (LoRA) to the Hugging Face model repository.
  - This is adapter-only publishing (Phase 1).

#### Model card (README.md)

If you enable the model card editor:

- Use **Load simple template** to start from a clean scaffold.
- Use **Generate with Ollama** to draft an intelligent card using the selected training run metadata.

When you generate with Ollama, the UI shows a small loading spinner and snackbars for start/success/failure.

______________________________________________________________________

## Usage Examples

### Example 1: Local splits only

1. Select your **Database Session** from the dropdown.
1. Set **Validation** to `0.05` and **Test** to `0.0`.
1. Enable **Shuffle** and choose a **Seed** (e.g. 42).
1. Set **Save dir** to `hf_dataset`.
1. Click **Build Dataset**.

Result: a `DatasetDict` saved under `hf_dataset/` with `train` and `validation` splits.

### Example 2: Prepare a dataset for training + Hub

1. Select your scrape session from the dropdown.
1. Choose validation/test fractions (e.g. 5% validation, 5% test).
1. Set **Save dir** to something like `my_dataset`.
1. Enable **Push to Hub** and provide:
   - **Repo ID**: `username/my-first-dataset`
   - **Private**: as desired
   - **HF Token**: if not already set in Settings/env.
1. Click **Build Dataset** and then **Push + Upload README**.

Result: a local dataset plus a remote dataset repo with a generated card.

______________________________________________________________________

## Tips & Best Practices

- Keep **Min Length** modest at first (e.g. 1–10 chars) to avoid over-filtering; tighten later.
- Use a consistent **Seed** across experiments so training runs are comparable.
- Double-check **Repo ID** and permissions before pushing to avoid cluttering your account.
- Use the **Dataset Analysis** tab after building to understand distribution, duplicates, sentiment, etc.

## Offline Mode

When **Offline Mode** is enabled:

- Hugging Face Hub actions are disabled.
- Push-related controls are disabled, and the UI shows an inline explanation:
  - "Offline Mode: Hugging Face Hub actions are disabled."

This applies to:

- Dataset pushing ("Push to Hub")
- Model adapter publishing ("Publish adapter")

FineFoundry keeps these sections visible so you can see what is available, but blocks the network action.

## Adapter vs full model publishing

- **Adapter (Phase 1)**
  - Publishes the LoRA adapter weights produced by training.
  - This is lightweight and the recommended default for sharing fine-tunes.
  - Consumers load it with the base model.
- **Full model (Phase 2, planned)**
  - Publishes a merged set of model weights.
  - Larger upload and more storage cost, but simpler for consumers.

______________________________________________________________________

## Related Topics

- [Data Sources Tab](scrape-tab.md) – create the initial dataset.
- [Merge Datasets Tab](merge-tab.md) – combine multiple datasets before building.
- [Analysis Tab](analysis-tab.md) – analyze the built dataset.
- [Training Tab](training-tab.md) – fine-tune models on your dataset.
- [Authentication](authentication.md) – managing your Hugging Face token.

______________________________________________________________________

**Next**: [Training Tab](training-tab.md) | **Previous**: [Data Sources Tab](scrape-tab.md) | [Back to Documentation Index](../README.md)
