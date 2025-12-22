# Frequently Asked Questions

Not sure where to start? This FAQ points you to the right documentation for common questions.

______________________________________________________________________

## Getting Started

**Where should I start?**  
The [Quick Start Guide](quick-start.md) walks you through installation and your first end-to-end workflow. For a tour of the interface, see the [GUI Overview](gui-overview.md).

**What are the system requirements?**  
Python 3.10+ on Linux, macOS, or Windows. Full details in the [Installation Guide](installation.md).

**How do I install FineFoundry?**  
Clone the repo and run `uv sync` (or use pip). The [Installation Guide](installation.md) covers both methods.

______________________________________________________________________

## Using the App

**Where's the documentation for each tab?**  
Start with the [GUI Overview](gui-overview.md), then see individual guides: [Data Sources](scrape-tab.md), [Publish](build-publish-tab.md), [Training](training-tab.md), [Inference](inference-tab.md), [Merge Datasets](merge-tab.md), [Analysis](analysis-tab.md), and [Settings](settings-tab.md).

**Is there a CLI?**  
Yes—see [CLI Usage](cli-usage.md) for automating dataset builds, scraping, and synthetic data generation from the command line.

**How do I set up authentication?**  
The [Authentication Guide](authentication.md) covers Hugging Face tokens and Runpod API keys. For cloud training specifics, see [Runpod Setup](../deployment/runpod.md).

______________________________________________________________________

## Data and Scraping

**Where's the scraping documentation?**  
For the GUI, see [Data Sources Tab](scrape-tab.md). For programmatic use, see the [Scrapers API](../api/scrapers.md) and individual scraper docs for [4chan](../api/fourchan-scraper.md), [Reddit](../api/reddit-scraper.md), and [Stack Exchange](../api/stackexchange-scraper.md).

**How do I use a proxy or Tor?**  
See the [Proxy Setup Guide](../deployment/proxy-setup.md) for SOCKS proxy configuration and Tor integration.

______________________________________________________________________

## Training and Inference

**How do I train a model?**  
The [Training Tab](training-tab.md) covers the GUI workflow. For programmatic training, see the [Training API](../api/training.md).

**How do I use Runpod?**  
The [Runpod Setup Guide](../deployment/runpod.md) explains cloud training infrastructure.

**Where's the inference documentation?**  
See the [Inference Tab](inference-tab.md) for testing trained adapters.

______________________________________________________________________

## Testing and CI

**How do I run tests?**  
Use `uv run pytest` for the full suite. Unit tests are in `tests/unit/`, integration tests in `tests/integration/`. The [Testing Guide](../development/testing.md) has full details.

**How does coverage work?**  
Coverage is measured with `coverage.py`. CI enforces a minimum threshold. See the Testing Guide for commands and thresholds.

**What about type checking?**  
Run `uv run mypy` locally. The Testing Guide documents which paths are checked.

______________________________________________________________________

## Development

**How is the code organized?**  
The [Project Structure](../development/project-structure.md) guide explains the layout.

**Where do logs go?**  
See the [Logging Guide](../development/logging.md) for log locations and how to use them for debugging.

**How do I contribute?**  
The [Contributing Guide](../development/contributing.md) and [Code Style](../development/code-style.md) cover conventions and submission process.

______________________________________________________________________

## Troubleshooting

**Something's broken—where do I start?**  
The [Troubleshooting Guide](troubleshooting.md) covers common issues. For more detail, enable debug logging as described in the [Logging Guide](../development/logging.md).

**How do I report a bug?**  
Check [GitHub Issues](https://github.com/SourceBox-LLC/FineFoundry/issues) first to see if it's known. If not, create an issue with your OS, Python version, steps to reproduce, and relevant logs.
