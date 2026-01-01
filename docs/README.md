# FineFoundry Documentation

Welcome to FineFoundry! This app helps you create custom AI models trained on your own data. Whether you want to build a chatbot that sounds like your favorite community, create a specialized assistant, or just experiment with AI training, you're in the right place.

**No coding experience required.** The app handles all the technical stuff for you.

## What Can You Do With FineFoundry?

- **Collect training data** from websites like Reddit, 4chan, and Stack Exchange—or create your own from documents
- **Train AI models** on your data, either on your computer or using cloud GPUs
- **Test your models** to see how well they learned
- **Share your work** by publishing to Hugging Face (a popular AI community)

## Getting Started

**New here?** Start with the [Quick Start Guide](user-guide/quick-start.md)—you'll have your first dataset ready in about 10 minutes.

Need help installing? The [Installation Guide](user-guide/installation.md) walks you through every step with screenshots.

## How to Use Each Tab

FineFoundry has several tabs, each with a specific job:

| Tab | What It Does |
|-----|-------------|
| [Data Sources](user-guide/scrape-tab.md) | Collect training data from websites or your own documents |
| [Publish](user-guide/build-publish-tab.md) | Prepare your data and share it online |
| [Training](user-guide/training-tab.md) | Teach an AI model using your data |
| [Inference](user-guide/inference-tab.md) | Chat with your trained model to test it |
| [Merge Datasets](user-guide/merge-tab.md) | Combine data from different sources |
| [Analysis](user-guide/analysis-tab.md) | Check your data quality before training |
| [Settings](user-guide/settings-tab.md) | Set up your accounts and preferences |

## Common Questions

**Do I need a powerful computer?**
For collecting data and basic tasks, any modern computer works fine. For training models, you'll either need a gaming-grade graphics card (GPU) or you can use cloud GPUs through Runpod (costs a few dollars per hour).

**Is this free?**
FineFoundry itself is completely free. Cloud training on Runpod costs money, but you can also train on your own computer for free if you have a compatible GPU.

**Do I need to know how to code?**
Nope! Everything works through the visual interface. Just click buttons and fill in fields.

## Need Help?

- **Something not working?** Check the [Troubleshooting Guide](user-guide/troubleshooting.md)
- **Quick answers:** [Frequently Asked Questions](user-guide/faq.md)
- **Still stuck?** Ask in [GitHub Discussions](https://github.com/SourceBox-LLC/FineFoundry/discussions)

## For Developers

If you want to contribute code or understand how FineFoundry works under the hood:

- [Project Structure](development/project-structure.md) — How the code is organized
- [Contributing Guide](development/contributing.md) — How to submit improvements
- [API Reference](api/scrapers.md) — Using FineFoundry from Python scripts

## Legal & Ethics

Collecting data from the internet comes with responsibilities. Please read [Ethical & Legal Considerations](user-guide/ethical-legal.md) before scraping.

______________________________________________________________________

**Last Updated**: December 2025
