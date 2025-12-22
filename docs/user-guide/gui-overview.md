# GUI Overview

When you launch FineFoundry, you'll see a desktop window with tabs arranged across the top. Each tab handles a different part of the dataset and training workflow, and they're designed to flow naturally from one to the next.

## The Tabs

**Data Sources** is where most projects begin. Here you collect training data by scraping 4chan, Reddit, or Stack Exchange, or by generating synthetic data from your own documents using local LLMs. The data you collect gets saved to the database automatically, ready for the next steps.

**Dataset Analysis** helps you understand what you've collected before you commit to training. Run analysis modules to check sentiment distribution, find duplicates, measure readability, and spot potential issues. It's a quick sanity check that can save you time later.

**Merge Datasets** lets you combine data from multiple sources into a single dataset. If you've scraped from several places or want to mix in some Hugging Face datasets, this is where you bring it all together.

**Training** is where the fine-tuning happens. Choose between running on Runpod's cloud GPUs or locally via Docker, configure your hyperparameters (or use the beginner-friendly presets), and kick off a training run. The same training script works in both environments, so you can develop locally and scale up to the cloud when you're ready.

**Inference** gives you a playground to chat with your fine-tuned models. Select a completed training run, and the tab validates that the adapter is ready to use. Then you can test prompts, adjust temperature and other settings, and have multi-turn conversations to see how your model behaves.

**Publish** handles getting your work out into the world. Create train/validation/test splits from your scraped data, then push datasets and LoRA adapters to Hugging Face Hub. FineFoundry generates dataset cards automatically.

**Settings** is where you configure authentication (Hugging Face tokens, Runpod API keys), proxy settings, and Ollama integration for local LLM features.

## Detailed Guides

Each tab has its own in-depth guide with screenshots, layout explanations, example workflows, and tips. Jump to whichever one you need: [Data Sources](scrape-tab.md), [Publish](build-publish-tab.md), [Training](training-tab.md), [Inference](inference-tab.md), [Merge Datasets](merge-tab.md), [Analysis](analysis-tab.md), or [Settings](settings-tab.md).

For a hands-on walkthrough that touches all the key tabs, see the [Quick Start Guide](quick-start.md).

______________________________________________________________________

[Back to Documentation Index](../README.md)
