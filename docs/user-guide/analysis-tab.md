# Dataset Analysis Tab

The Dataset Analysis tab provides interactive insights into your datasets so you can assess quality, diversity, and potential issues **before** training.

Use this tab to:

- Load a JSON or Hugging Face dataset
- Run a configurable set of analysis modules (stats, duplicates, sentiment, etc.)
- Visualize distributions and proxy metrics for data quality

![Dataset Analysis Tab](../../img/ff_dataset_analysis.png)

---

## Overview

Typical workflow:

1. Select a **dataset source** (JSON file or Hugging Face repo).
2. Enable the analysis **modules** you care about.
3. Click **Analyze dataset**.
4. Review summary stats and module outputs.
5. Adjust scraping/building/merging parameters if needed.

---

## Layout at a Glance

### 1. Dataset Source

- **Source selector** – Choose JSON file or Hugging Face dataset.
- **Path / repo fields** – Provide the file path or repo id + split.

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

---

## Usage Examples

### Example 1: Quick sanity check

1. Load your just-built dataset from the **Build & Publish** tab.
2. Enable **Basic Stats**, **Duplicates & Similarity**, and **Sentiment**.
3. Click **Analyze dataset**.
4. Confirm record counts and check for excessive duplicates or extreme sentiment imbalance.

### Example 2: Curriculum planning

1. Load a merged dataset.
2. Enable **Class Balance (length)**, **Conversation depth**, and **Topics**.
3. Use the outputs to decide whether you need to:
   - Rebalance short vs long samples.
   - Add or remove certain topic sources.

### Example 3: Safety scanning

1. Enable **Toxicity / Politeness** and **Data leakage** proxies.
2. Identify whether your dataset has significant toxic content or potential leakage issues.
3. Adjust scraping filters or dataset cleaning steps accordingly.

---

## Tips & Best Practices

- Run analysis **before** committing to long training runs.
- Use **Duplicates & Similarity** to spot unintentional dataset duplication.
- Use **Sentiment** and **Toxicity** to gauge whether additional filtering is needed for your use case.
- Re-run analysis after major changes to scraping, merging, or cleaning logic.

---

## Related Topics

- [Scrape Tab](scrape-tab.md) – collect raw conversational data.
- [Build & Publish Tab](build-publish-tab.md) – convert JSON into a structured dataset.
- [Merge Datasets Tab](merge-tab.md) – combine multiple datasets.
- [Training Tab](training-tab.md) – fine-tune models on analyzed datasets.

---

**Next**: [Settings Tab](settings-tab.md) | **Previous**: [Merge Datasets Tab](merge-tab.md) | [Back to Documentation Index](../README.md)
