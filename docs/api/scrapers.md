# Scrapers API

High-level reference for FineFoundry's data collection interfaces.

This page provides an overview of the scraper and generator modules and links to more detailed per-source pages:

## Network Scrapers

- [4chan Scraper](fourchan-scraper.md) — `src/scrapers/fourchan_scraper.py`
- [Reddit Scraper](reddit-scraper.md) — `src/scrapers/reddit_scraper.py`
- [Stack Exchange Scraper](stackexchange-scraper.md) — `src/scrapers/stackexchange_scraper.py`

## Synthetic Data Generation

- **Synthetic Generator** — `src/helpers/synthetic.py`

Generate training data from your own documents using local LLMs powered by Unsloth's SyntheticDataKit.

### Supported Input Formats

- PDF documents
- DOCX (Word documents)
- PPTX (PowerPoint)
- HTML/HTM web pages
- TXT plain text
- URLs (fetched and parsed)

### Generation Types

- **qa** — Question-answer pairs from document content
- **cot** — Chain-of-thought reasoning examples
- **summary** — Document summaries

### Basic Usage (via UI)

1. Select **Synthetic** in the Scrape tab
1. Add files or URLs
1. Configure model, generation type, and parameters
1. Click **Start**

### Programmatic Usage

```python
from helpers.synthetic import run_synthetic_generation

# Called internally by the UI - async function
await run_synthetic_generation(
    page=page,
    log_view=log_list,
    prog=progress_bar,
    labels={"threads": threads_label, "pairs": pairs_label},
    preview_host=preview_host,
    cancel_flag=cancel_state,
    sources=["document.pdf", "https://example.com/article"],
    gen_type="qa",
    num_pairs=25,
    max_chunks=10,
    curate=False,
    curate_threshold=7.5,
    multimodal=False,
    dataset_format="ChatML",
    model="unsloth/Llama-3.2-3B-Instruct",
)
```

Synthetic generation results are saved to the SQLite database (`finefoundry.db`).

### Requirements

- GPU with sufficient VRAM (8GB+ recommended)
- Unsloth package with SyntheticDataKit
- vLLM for local model serving

Future iterations can add full parameter tables, examples, and CLI usage for each source.
