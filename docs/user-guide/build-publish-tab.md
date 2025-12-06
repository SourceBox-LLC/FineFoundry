# Build & Publish Tab

The Build & Publish tab converts raw JSON data into a structured Hugging Face dataset and optionally pushes it to the Hub with a generated dataset card.

Use this tab to:

- Load a scraped JSON file (e.g. from the Scrape tab)
- Create train/validation/test splits with reproducible settings
- Save a `datasets.DatasetDict` to disk
- Push datasets to the Hugging Face Hub with a basic README card

![Build & Publish Tab](../../img/ff_buld_publish.png)

______________________________________________________________________

## Overview

Typical workflow:

1. Select a **Data file (JSON)** produced by scraping or other sources.
1. Configure **splits**, **min length**, and shuffling.
1. Click **Build Dataset** to create a `DatasetDict` on disk.
1. Optionally enable **Push to Hub**, fill in repo details and token.
1. Click **Push + Upload README** to publish the dataset.

______________________________________________________________________

## Layout at a Glance

### 1. Data Source

- **Data file (JSON)**
  - Path to your dataset (commonly `scraped_training_data.json`).
  - Must have schema like `[{"input": "...", "output": "..."}, ...]`.

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
  - Reads JSON, applies filtering and splitting, and writes the dataset to `Save dir`.
  - Shows logs / status for success or errors.

### 5. Push to Hub

- **Push to Hub** toggle
  - Enables pushing the dataset to the Hugging Face Hub.
- **Repo ID** – e.g., `username/my-first-dataset`.
- **Private** – Whether to create a private dataset repo.
- **HF Token** – Optional if you already authenticated via CLI or `HF_TOKEN` env var.
- **Push + Upload README** button
  - Uploads the dataset and a generated dataset card (`README.md` in the repo).

______________________________________________________________________

## Usage Examples

### Example 1: Local splits only

1. Set **Data file (JSON)** to `scraped_training_data.json`.
1. Set **Validation** to `0.05` and **Test** to `0.0`.
1. Enable **Shuffle** and choose a **Seed** (e.g. 42).
1. Set **Save dir** to `hf_dataset`.
1. Click **Build Dataset**.

Result: a `DatasetDict` saved under `hf_dataset/` with `train` and `validation` splits.

### Example 2: Prepare a dataset for training + Hub

1. Point to your scraped JSON.
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

______________________________________________________________________

## Related Topics

- [Scrape Tab](scrape-tab.md) – create the initial JSON dataset.
- [Merge Datasets Tab](merge-tab.md) – combine multiple datasets before building.
- [Analysis Tab](analysis-tab.md) – analyze the built dataset.
- [Training Tab](training-tab.md) – fine-tune models on your dataset.
- [Authentication](authentication.md) – managing your Hugging Face token.

______________________________________________________________________

**Next**: [Training Tab](training-tab.md) | **Previous**: [Scrape Tab](scrape-tab.md) | [Back to Documentation Index](../README.md)
