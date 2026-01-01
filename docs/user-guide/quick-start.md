# Quick Start Guide

Get your first dataset ready in about 10 minutes! This guide walks you through installing FineFoundry and collecting your first batch of training data.

## What You'll Need

- **A computer** running Windows, macOS, or Linux
- **Python 3.10 or newer** — [Download Python here](https://www.python.org/downloads/) if you don't have it
- **An internet connection** for downloading and collecting data

**Optional** (for sharing your work online):

- A free [Hugging Face account](https://huggingface.co/join)

## Step 1: Download and Install

### Option A: The Easy Way (Recommended)

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run these commands:

```bash
git clone https://github.com/SourceBox-LLC/FineFoundry.git FineFoundry-Core
cd FineFoundry-Core
pip install uv
```

Then start the app:

**On Mac/Linux:**

```bash
chmod +x run_finefoundry.sh
./run_finefoundry.sh
```

**On Windows:**

```bash
uv run src/main.py
```

### Option B: Traditional Installation

If the above doesn't work, try this instead:

```bash
git clone https://github.com/SourceBox-LLC/FineFoundry.git FineFoundry-Core
cd FineFoundry-Core
python -m venv venv
```

Activate your virtual environment:

- **Mac/Linux:** `source venv/bin/activate`
- **Windows:** `.\venv\Scripts\Activate.ps1`

Then install and run:

```bash
pip install -e .
python src/main.py
```

## Step 2: Tour the App

When FineFoundry opens, you'll see a window with tabs across the top:

- **Data Sources** — Where you collect data from websites or documents
- **Publish** — Prepare and share your datasets
- **Training** — Teach AI models with your data
- **Inference** — Test your trained models by chatting with them
- **Merge Datasets** — Combine multiple data collections
- **Analysis** — Check your data quality
- **Settings** — Set up accounts and preferences

## Step 3: Collect Your First Data

Let's grab some data to work with:

1. **Click the "Data Sources" tab**
1. **Choose a source** — For this example, select "4chan"
1. **Pick some boards** — Click a few board chips like `b`, `pol`, or `x`
1. **Set your limits:**
   - Max Threads: `50`
   - Max Pairs: `500`
   - Delay: `0.5`
   - Min Length: `10`
1. **Click "Start"**

Watch the progress bar and logs as data flows in. This usually takes 1-3 minutes.

## Step 4: Preview Your Data

When the collection finishes:

1. Click **"Preview Dataset"**
1. You'll see a two-column view showing conversation pairs

Each row shows an "input" (like a question or prompt) and an "output" (the response). This is what the AI will learn from!

**Your data is automatically saved**, so you won't lose it if you close the app.

## What's Next?

You've just collected your first dataset! Here's what you can do now:

### Want to train a model?

Go to the [Training Tab Guide](training-tab.md) to learn how to teach an AI using your data.

### Want to share your dataset?

Go to the [Publish Tab](build-publish-tab.md) to upload it to Hugging Face.

### Want more data?

- Try different sources (Reddit, Stack Exchange)
- Collect from multiple boards
- Generate synthetic data from your own documents

### Want to combine datasets?

Use the [Merge Datasets Tab](merge-tab.md) to mix data from different collections.

## Having Problems?

- **App won't start?** Make sure Python 3.10+ is installed
- **No data collected?** Check your internet connection and try different boards
- **Other issues?** See the [Troubleshooting Guide](troubleshooting.md)

Still stuck? Ask for help in [GitHub Discussions](https://github.com/SourceBox-LLC/FineFoundry/discussions).

______________________________________________________________________

**Next**: [Data Sources Tab](scrape-tab.md) | [Back to Documentation Index](../README.md)
