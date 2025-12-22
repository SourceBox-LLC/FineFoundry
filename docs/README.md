# FineFoundry Documentation

Welcome! This is your home base for learning everything about FineFoundry—from getting your first dataset built in minutes to diving deep into advanced training workflows and contributing to the project.

Whether you're a first-time user looking to scrape some data and fine-tune a model, or an experienced developer wanting to extend the platform, you'll find what you need here.

## Getting Started

New to FineFoundry? Start with the [Quick Start Guide](user-guide/quick-start.md) to go from zero to a working dataset in just a few minutes. If you need more detailed setup instructions, the [Installation Guide](user-guide/installation.md) walks you through every step, and [Upgrade Notes](user-guide/upgrade-notes.md) covers important changes if you're updating from an earlier version.

## User Guides

Once you're up and running, the [GUI Overview](user-guide/gui-overview.md) gives you a bird's-eye view of the desktop app. From there, each tab has its own in-depth guide:

The [Data Sources Tab](user-guide/scrape-tab.md) is where you collect training data—scrape from 4chan, Reddit, or Stack Exchange, or generate synthetic data from your own documents using local LLMs. When your data is ready, the [Publish Tab](user-guide/build-publish-tab.md) helps you create train/val/test splits and push datasets to Hugging Face Hub.

For training, the [Training Tab](user-guide/training-tab.md) covers everything from beginner-friendly presets to expert-level hyperparameter tuning, whether you're running on Runpod or locally via Docker. After training, use the [Inference Tab](user-guide/inference-tab.md) to chat with your fine-tuned model and verify it learned what you intended.

Need to combine datasets from different sources? The [Merge Datasets Tab](user-guide/merge-tab.md) handles that. Want to understand your data quality before training? The [Analysis Tab](user-guide/analysis-tab.md) provides insights into sentiment, duplicates, readability, and more.

Finally, the [Settings Tab](user-guide/settings-tab.md) is where you configure authentication and preferences, while [CLI Usage](user-guide/cli-usage.md) and [Authentication](user-guide/authentication.md) cover scripting and token setup.

## For Developers

If you want to understand or contribute to the codebase, start with [Project Structure](development/project-structure.md) to see how everything is organized. The [Database Architecture](development/database.md) doc explains the SQLite storage system, and the [Logging System](development/logging.md) shows you how to debug issues effectively.

Ready to contribute? The [Contributing Guide](development/contributing.md) explains the workflow, [Testing](development/testing.md) covers how to run and write tests, and [Code Style](development/code-style.md) describes the conventions we follow.

## API Reference

For programmatic usage, the [Scrapers API](api/scrapers.md) documents how to use the data collection tools from Python, including the [4chan](api/fourchan-scraper.md), [Reddit](api/reddit-scraper.md), and [Stack Exchange](api/stackexchange-scraper.md) scrapers, plus synthetic data generation. The [Dataset Builder API](api/dataset-builder.md) and [Training API](api/training.md) cover programmatic dataset creation and training workflows.

## Deployment

Running FineFoundry in production or on cloud infrastructure? The [Proxy Configuration](deployment/proxy-setup.md) guide covers Tor and custom proxy setup, [Docker Deployment](deployment/docker.md) explains containerized workflows, and [Runpod Setup](deployment/runpod.md) walks through cloud training infrastructure.

## Getting Help

If something isn't working, check the [Troubleshooting Guide](user-guide/troubleshooting.md) first—it covers the most common issues and their solutions. The [User FAQ](user-guide/faq.md) answers frequently asked questions, while the [Developer FAQ](development/faq.md) is aimed at contributors and power users.

For legal and ethical considerations around the data you collect, see [Ethical & Legal](user-guide/ethical-legal.md).

## Contributing to the Docs

Found a typo or want to improve something? Documentation contributions are always welcome. See the [Contributing Guide](development/contributing.md) for how to submit improvements.

## Support

If you can't find what you need here, open an issue on [GitHub Issues](https://github.com/SourceBox-LLC/FineFoundry/issues) or start a discussion in [GitHub Discussions](https://github.com/SourceBox-LLC/FineFoundry/discussions).

______________________________________________________________________

**Last Updated**: December 2025
**Version**: See `pyproject.toml` for the current version.
