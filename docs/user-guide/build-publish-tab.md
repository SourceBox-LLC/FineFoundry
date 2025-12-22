# Publish Tab

The Publish tab is where you get your work out into the world. Here you can build proper train/validation/test splits from your scraped data and push datasets to Hugging Face Hub. You can also publish LoRA adapters from completed training runs. Full model publishing (merged weights) is planned for a future release.

![Publish Tab](../../img/new/ff_build_publish.png)

## Building Datasets

Start by selecting a scrape session from your database history. Then configure how you want to split the data—use the sliders to set validation and test percentages, with the remainder going to the training split. Enable shuffling and set a seed for reproducibility. The min length filter removes pairs where the input or output is too short.

Click Build Dataset to create a `DatasetDict` on disk. The logs show you what's happening, and when it's done you'll have a proper dataset saved to your chosen directory.

## Publishing to Hugging Face

If you want to share your dataset, enable Push to Hub and fill in your repo ID (like `username/my-first-dataset`). Set whether it should be private, and provide your HF token if it's not already configured in Settings or as an environment variable. Click Push + Upload README to upload everything, including a generated dataset card.

You can customize the dataset card before uploading. Load a simple template to start from scratch, or use Generate with Ollama to draft an intelligent card based on your data. The UI shows loading states and success/failure snackbars so you know what's happening.

## Publishing Adapters

If you've completed training runs, you can publish the resulting LoRA adapters. Select a training run from the dropdown, set your model repo ID and privacy preferences, and click Publish adapter. The adapter folder gets uploaded to Hugging Face as a model repository.

Like datasets, you can customize the model card before uploading—either start from a template or generate one with Ollama using your training run's metadata.

## Examples

### Local Dataset Only

Select your database session, set validation to 5% and test to 0%, enable shuffle with seed 42, and set a save directory. Click Build Dataset. You'll get a local `DatasetDict` with train and validation splits.

### Dataset + Hub Publishing

Select your session, configure your splits, set the save directory, enable Push to Hub with your repo ID and token, then click Build Dataset followed by Push + Upload README. You get both a local dataset and a published repo with a generated card.

## Tips

Keep min length modest at first (1-10 characters) to avoid over-filtering—you can always tighten it later. Use a consistent seed across experiments so your training runs are comparable. Double-check your repo ID before pushing to avoid cluttering your Hugging Face account. After building, use the Dataset Analysis tab to understand your data before training.

## Offline Mode

When Offline Mode is enabled, all Hub actions are disabled. The push controls remain visible so you can see what's available, but they won't work until you go back online. A banner explains this.

## Adapter vs Full Model

Currently, FineFoundry publishes adapters—the lightweight LoRA weights that consumers load alongside the base model. This is efficient and recommended for most use cases. Full model publishing (merged weights) is coming in a future release for situations where you want a simpler experience for consumers.

______________________________________________________________________

## Related Guides

Before you get here, you'll typically use the [Data Sources Tab](scrape-tab.md) to collect data or the [Merge Datasets Tab](merge-tab.md) to combine datasets. After building, the [Analysis Tab](analysis-tab.md) helps you understand your data quality. Then the [Training Tab](training-tab.md) is where you fine-tune. For token setup, see [Authentication](authentication.md).

______________________________________________________________________

**Next**: [Training Tab](training-tab.md) | **Previous**: [Data Sources Tab](scrape-tab.md) | [Back to Documentation Index](../README.md)
