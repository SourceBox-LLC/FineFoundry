# Quick Start Guide

Use this guide to install FineFoundry and get from "cloned repo" to a running desktop app and your first small dataset in just a few minutes.

## Prerequisites

- **Python 3.10+** (Windows, macOS, or Linux)
- **Git** (optional, for cloning the repository)
- **uv** (recommended) or pip for package management

Optional for publishing:

- [Hugging Face account](https://huggingface.co/) with an [access token](https://huggingface.co/settings/tokens)

## Installation

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/SourceBox-LLC/FineFoundry.git
cd FineFoundry-Core

# Run the application (uv handles dependencies automatically)
uv run src/main.py
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/SourceBox-LLC/FineFoundry.git
cd FineFoundry-Core

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell)
./venv/Scripts/Activate.ps1

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python src/main.py
```

## First Launch

When you first launch FineFoundry, you'll see a desktop application with several tabs:

1. **Scrape** - Collect training data from 4chan, Reddit, or Stack Exchange
1. **Build / Publish** - Create datasets and publish to Hugging Face
1. **Training** - Fine-tune models on Runpod or locally
1. **Inference** - Run inference against fine-tuned adapters with prompt history and Full Chat View
1. **Merge Datasets** - Combine multiple datasets
1. **Dataset Analysis** - Analyze dataset quality
1. **Settings** - Configure authentication and preferences

## Your First Dataset

Let's create a simple dataset from 4chan:

### Step 1: Scrape Data

1. Navigate to the **Scrape** tab
1. Select a few boards (e.g., `pol`, `b`, `x`)
1. Configure parameters:
   - **Max Threads**: 50 (for a quick test)
   - **Max Pairs**: 500
   - **Delay**: 0.5 seconds
   - **Min Length**: 10 characters
1. Click **Start Scrape**
1. Watch the progress bar and logs
1. When complete, click **Preview Dataset** to inspect the data

Your data is saved to `scraped_training_data.json` by default.

### Step 2: Build & Publish (Optional)

1. Navigate to the **Build / Publish** tab
1. The data file should already be set to `scraped_training_data.json`
1. Configure splits:
   - Adjust validation/test split percentages with sliders
   - Set seed for reproducibility
1. Click **Build Dataset** to create train/val/test splits
1. (Optional) To publish to Hugging Face:
   - Enable **Push to Hub**
   - Set **Repo ID** (e.g., `username/my-first-dataset`)
   - Add your **HF Token** (or configure in Settings)
   - Click **Push + Upload README**

### Step 3: Analyze Your Dataset (Optional)

1. Navigate to the **Dataset Analysis** tab
1. Select your dataset source (JSON file or Hugging Face)
1. Enable analysis modules you're interested in
1. Click **Analyze Dataset**
1. Review the insights to understand your data quality

## Next Steps

Now that you have the basics:

- **🧠 Training (Runpod or local Docker)**: Start [fine-tuning models](training-tab.md). Run the same training script on Runpod pods or your local machine and iterate quickly on new adapters.
- **⚙️ Reusable training configs**: Use the Training tab's **Configuration** section or "Save current setup" buttons to snapshot full training setups (dataset, hyperparameters, target, and infra). Configs are stored under `src/saved_configs/` and the last one auto-loads on startup.
- **🧪 Quick Local Inference**: After a successful local run, use the Quick Local Inference panel as a **one-click smoke test** for your latest local adapter with temperature/max token sliders and presets. When you click **Run Inference**, the app shows a short loading state (button disabled + spinner + status text) while the fine-tuned model is loaded and a response is generated.
- **💬 Inference tab**: Use the [Inference Tab](inference-tab.md) as your **dedicated playground**: select any adapter directory, validate it immediately, and chat with your fine-tuned model using either the Prompt & responses view or the Full Chat View dialog.
- **📖 Learn More**: Read the [complete GUI guide](gui-overview.md)
- **🔧 CLI Usage**: Automate workflows with [CLI tools](cli-usage.md)
- **🔀 Merge**: [Combine multiple datasets](merge-tab.md)
- **🔐 Authentication**: Set up [Hugging Face and Runpod access](authentication.md)

## Getting Help

- **Troubleshooting**: Check the [troubleshooting guide](troubleshooting.md)
- **FAQ**: Browse [frequently asked questions](faq.md)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/SourceBox-LLC/FineFoundry/issues)
- **Logging**: Enable debug mode for detailed logs (see [Logging Guide](../development/logging.md))

______________________________________________________________________

**Next**: [GUI Overview](gui-overview.md) | [Back to Documentation Index](../README.md)
