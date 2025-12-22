# Installation

This guide walks you through installing FineFoundry on your machine. If you want the quickest possible path to a working app with your first dataset, the [Quick Start Guide](quick-start.md) is more concise—but this page covers all the details if you need them.

## What You'll Need

FineFoundry runs on Python 3.10 or newer across Windows, macOS, and Linux. While not required, having a NVIDIA GPU with recent drivers is recommended if you plan to do training or run 4-bit quantized models locally.

You'll also need Git if you want to clone the repository (though you can download a zip instead), and internet access for installing dependencies. Later, if you want to publish datasets or train on cloud GPUs, you'll configure a Hugging Face account and token (see [Authentication](authentication.md)), and optionally a Runpod account for remote training.

## The Easy Way: Using uv

We recommend `uv` because it handles virtual environments and dependencies automatically—no manual setup required. If you don't already have it, install it with `pip install uv`.

Clone the repository and run the launcher script:

```bash
git clone https://github.com/SourceBox-LLC/FineFoundry.git FineFoundry-Core
cd FineFoundry-Core
```

On macOS/Linux, make the script executable (you only need to do this once):

```bash
chmod +x run_finefoundry.sh
./run_finefoundry.sh
```

That's it. The script creates an isolated environment, resolves dependencies, and launches the app. The first run takes a bit longer while everything gets set up, but subsequent launches are fast.

If you prefer not to use the launcher script, you can run `uv run src/main.py` directly. And if you want to pre-download dependencies (say, before going offline), run `uv sync` first.

## The Traditional Way: pip and venv

If you'd rather manage your own virtual environment, the classic approach works just as well.

Start by cloning the repository:

```bash
git clone https://github.com/SourceBox-LLC/FineFoundry.git
cd FineFoundry-Core
```

Create and activate a virtual environment:

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows (PowerShell):
./venv/Scripts/Activate.ps1
```

If your system uses `py` instead of `python`, run `py -3.10 -m venv venv` instead.

With the environment active, install the package:

```bash
pip install -e .
```

Then launch the app:

```bash
python src/main.py
```

## Checking That It Worked

When FineFoundry starts, you should see a desktop window with tabs across the top: Data Sources, Dataset Analysis, Merge Datasets, Training, Inference, Publish, and Settings. If you see import errors in the terminal, try re-running `uv sync` or `pip install -e . --upgrade`.

To really verify everything is working, try the "Your First Dataset" section in the [Quick Start Guide](quick-start.md)—scrape a small sample, build a dataset, and optionally run a quick analysis. If that all works, you're good to go.

## If Something Goes Wrong

Installation problems are usually one of a few common issues. "Python command not found" or confusion between `python` and `py` is common on Windows. "uv command not found" means you need to install uv first. "Module not found" errors after launch typically mean dependencies didn't install correctly—re-run the installation step.

The [Troubleshooting Guide](troubleshooting.md) has detailed solutions for these and other issues. If you're still stuck, open a GitHub issue with your OS, Python version, the command you ran, and the full error message.

## What's Next

With FineFoundry installed and running, head to the [Quick Start Guide](quick-start.md) to create your first dataset. The [GUI Overview](gui-overview.md) gives you a tour of all the tabs, and [Authentication](authentication.md) covers setting up your Hugging Face and Runpod credentials.

From there, you can dive into the detailed guides for each tab and start building datasets and training models.
