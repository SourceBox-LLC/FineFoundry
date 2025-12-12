# Dataset Analysis Tab

The Dataset Analysis tab provides interactive insights into your datasets so you can assess quality, diversity, and potential issues **before** training.

Use this tab to:

- Load a database session (or optionally a Hugging Face dataset when online)
- Run a configurable set of analysis modules (stats, duplicates, sentiment, etc.)
- Visualize distributions and proxy metrics for data quality

![Dataset Analysis Tab](../../img/ff_dataset_analysis.png)

______________________________________________________________________

## Overview

Typical workflow:

1. Select a **dataset source** (Database Session, or Hugging Face repo when online).
1. Enable the analysis **modules** you care about.
1. Click **Analyze dataset**.
1. Review summary stats and module outputs.
1. Adjust scraping/building/merging parameters if needed.

______________________________________________________________________

## Layout at a Glance

### 1. Dataset Source

- **Source selector** – Choose Database Session or Hugging Face dataset (when online).
- **Session / repo fields** – Select a scrape session or provide repo id + split.

### 2. Analysis Modules

Toggle modules on/off:

- **Basic Stats** – record counts, mean input/output lengths.
- **Duplicates & Similarity** – approximate duplicate rate via hashed pairs.
- **Sentiment** – polarity distribution across samples.
- **Class Balance (length)** – short/medium/long buckets for input length.
- **Extra metrics** (proxy signals):
  - Coverage overlap
  - Data leakage
  - Conversation depth
  - Speaker balance
  - Question vs statement
  - Readability
  - NER proxy
  - Toxicity / Politeness
  - Dialogue acts / Topics / Alignment

A summary section lists all active modules after each run.

### 3. Results & Visualizations

- **KPI tiles / metrics** – e.g., total records, avg lengths.
- **Progress bars / charts** – visualize distributions (sentiment, length buckets, etc.).
- **Tables / samples** – preview representative rows and metrics.

______________________________________________________________________

## Usage Examples

### Example 1: Quick sanity check

1. Load your just-built dataset from the **Build & Publish** tab.
1. Enable **Basic Stats**, **Duplicates & Similarity**, and **Sentiment**.
1. Click **Analyze dataset**.
1. Confirm record counts and check for excessive duplicates or extreme sentiment imbalance.

### Example 2: Curriculum planning

1. Load a merged dataset.
1. Enable **Class Balance (length)**, **Conversation depth**, and **Topics**.
1. Use the outputs to decide whether you need to:
   - Rebalance short vs long samples.
   - Add or remove certain topic sources.

### Example 3: Safety scanning

1. Enable **Toxicity / Politeness** and **Data leakage** proxies.
1. Identify whether your dataset has significant toxic content or potential leakage issues.
1. Adjust scraping filters or dataset cleaning steps accordingly.

______________________________________________________________________

## Tips & Best Practices

- Run analysis **before** committing to long training runs.
- Use **Duplicates & Similarity** to spot unintentional dataset duplication.
- Use **Sentiment** and **Toxicity** to gauge whether additional filtering is needed for your use case.
- Re-run analysis after major changes to scraping, merging, or cleaning logic.

______________________________________________________________________

## Offline Mode

When **Offline Mode** is enabled:

- **Hugging Face dataset source** remains visible but is disabled.
  - If you were previously set to Hugging Face, the selector resets to **Database**.
- **HF Inference API backend** remains visible but is disabled.
  - If you were previously using HF Inference API, the backend resets to **Local (Transformers)**.

The UI shows an Offline banner at the top of the tab and inline helper text under key controls explaining why they are disabled.

______________________________________________________________________

## Related Topics

- [Scrape Tab](scrape-tab.md) – collect raw conversational data.
- [Build & Publish Tab](build-publish-tab.md) – build train/val/test splits from a database session and optionally push to the Hub.
- [Merge Datasets Tab](merge-tab.md) – combine multiple datasets.
- [Training Tab](training-tab.md) – fine-tune models on analyzed datasets.

______________________________________________________________________

**Next**: [Settings Tab](settings-tab.md) | **Previous**: [Merge Datasets Tab](merge-tab.md) | [Back to Documentation Index](../README.md)
