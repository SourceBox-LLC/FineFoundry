# Quick Start Guide

This guide will take you from a freshly cloned repo to a running desktop app with your first dataset in just a few minutes. By the end, you'll have scraped some data, built a dataset, and be ready to start training.

## Before You Begin

You'll need Python 3.10 or newer installed on your machine—Windows, macOS, and Linux are all supported. Git is helpful for cloning the repository but not strictly required. For package management, we recommend using `uv` (it's fast and handles everything automatically), though pip works fine too.

If you plan to publish datasets to Hugging Face, you'll also want a [Hugging Face account](https://huggingface.co/) with an [access token](https://huggingface.co/settings/tokens) that has write permissions.

## Installation

The fastest way to get started is with `uv`. Clone the repository and run the launcher script—it handles the virtual environment and dependencies for you:

```bash
git clone https://github.com/SourceBox-LLC/FineFoundry.git FineFoundry-Core
cd FineFoundry-Core

# One-time setup on macOS/Linux: make the script executable
chmod +x run_finefoundry.sh
./run_finefoundry.sh
```

If you don't have `uv` installed, you can install it with `pip install uv`, or you can use the alternative without the launcher: `uv run src/main.py`.

If you prefer the traditional pip workflow, create a virtual environment and install the package:

```bash
git clone https://github.com/SourceBox-LLC/FineFoundry.git
cd FineFoundry-Core

python -m venv venv
source venv/bin/activate  # On Windows: ./venv/Scripts/Activate.ps1

pip install -e .
python src/main.py
```

## What You'll See

When FineFoundry launches, a desktop window opens with tabs across the top. The **Data Sources** tab is where you collect training data from 4chan, Reddit, Stack Exchange, or generate synthetic data from your own documents. **Dataset Analysis** helps you understand your data quality, while **Merge Datasets** lets you combine data from different sources.

The **Training** tab is where the magic happens—fine-tune models on Runpod's cloud GPUs or locally via Docker. After training, the **Inference** tab gives you a playground to chat with your fine-tuned model and verify it learned what you intended. **Publish** handles pushing your datasets and adapters to Hugging Face Hub, and **Settings** is where you configure authentication and preferences.

## Your First Dataset

Let's walk through creating a simple dataset to see how everything fits together.

### Scraping Some Data

Head to the **Data Sources** tab and select a few boards to scrape (like `pol`, `b`, or `x`). For a quick test, set Max Threads to 50, Max Pairs to 500, Delay to 0.5 seconds, and Min Length to 10 characters. Hit **Start** and watch the progress bar fill up as data comes in. The logs panel shows you exactly what's happening in real time.

When the scrape finishes, click **Preview Dataset** to see what you collected. You'll see a two-column view of input/output pairs—this is the raw material for training. Your data is automatically saved to the database, so you won't lose it.

### Building a Dataset (Optional)

If you want to create proper train/validation/test splits, go to the **Publish** tab. Select your scrape session from the dropdown, adjust the split percentages with the sliders, and click **Build Dataset**. 

Want to share your dataset with the world? Enable **Push to Hub**, set your Repo ID (something like `username/my-first-dataset`), add your HF Token, and click **Push + Upload README**. FineFoundry generates a dataset card for you automatically.

### Checking Data Quality (Optional)

Before training, it's worth understanding what you're working with. The **Dataset Analysis** tab lets you run various analysis modules—sentiment distribution, duplicate detection, readability scores, and more. Select your dataset, enable the modules you're curious about, and click **Analyze Dataset**. The insights can help you decide whether to filter, clean, or augment your data before training.

## Where to Go From Here

Now that you've got the basics down, you're ready to explore further.

For training, check out the [Training Tab Guide](training-tab.md)—it covers everything from beginner-friendly presets that auto-configure based on your GPU to expert-level hyperparameter control. The same training script runs on Runpod pods or your local machine, so you can iterate quickly.

After a successful local training run, the Quick Local Inference panel appears right in the Training tab. It's a quick way to smoke-test your adapter—just type a prompt and see how your model responds. For more extensive testing, the [Inference Tab](inference-tab.md) offers a dedicated playground with prompt history and a full chat view.

Training configs are saved to the database, so you can snapshot a working setup and reload it later. The last config you used auto-loads on startup, making it easy to pick up where you left off.

For a complete tour of the interface, read the [GUI Overview](gui-overview.md). If you want to automate things, the [CLI Usage](cli-usage.md) guide shows you how. And if you need to combine datasets from different sources, the [Merge Datasets Tab](merge-tab.md) has you covered.

## If Something Goes Wrong

Check the [Troubleshooting Guide](troubleshooting.md) first—it covers the most common issues. The [FAQ](faq.md) has answers to frequently asked questions. For deeper debugging, enable debug mode to get detailed logs (see the [Logging Guide](../development/logging.md)).

If you're still stuck, open an issue on [GitHub](https://github.com/SourceBox-LLC/FineFoundry/issues).

______________________________________________________________________

**Next**: [GUI Overview](gui-overview.md) | [Back to Documentation Index](../README.md)
