# Publish Tab

This is where you prepare your data for training and optionally share it with the world on Hugging Face (a popular AI community site).

![Publish Tab](../../img/new/ff_build_publish.png)

## What Can You Do Here?

1. **Build a dataset** — Organize your collected data into proper training format
1. **Share your dataset** — Upload to Hugging Face for others to use
1. **Share your trained models** — Upload your trained AI adapters

______________________________________________________________________

## Building a Dataset

After collecting data in the Data Sources tab, you need to "build" it before training. This organizes your data properly.

### How to Build

1. **Select your data** — Pick a collection from the dropdown
1. **Set the split** (optional):
   - Training data: What the AI learns from (usually 90-95%)
   - Validation data: Used to check progress (usually 5-10%)
1. **Click "Build Dataset"**

That's it! Your data is now ready for training.

### What Are Splits?

Think of it like studying for a test:

- **Training data** = The textbook you study from
- **Validation data** = Practice quizzes to check if you're learning

Most people use 95% training, 5% validation. The default settings work fine for beginners.

______________________________________________________________________

## Sharing Your Dataset (Optional)

Want others to use your dataset? You can upload it to Hugging Face.

### What You'll Need

- A free [Hugging Face account](https://huggingface.co/join)
- A "write" access token (create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

### How to Share

1. **Build your dataset first** (see above)
1. **Enable "Push to Hub"**
1. **Enter your repo name** — Something like `yourusername/my-dataset`
1. **Add your token** (or set it up in Settings first)
1. **Click "Push + Upload README"**

FineFoundry automatically creates a nice description page for your dataset!

______________________________________________________________________

## Sharing Your Trained Models (Optional)

After training an AI model, you can share the results on Hugging Face.

### How to Share a Model

1. **Select a completed training run** from the dropdown
1. **Enter your model repo name** — Like `yourusername/my-chatbot`
1. **Click "Publish adapter"**

**Note:** FineFoundry uploads "adapters" (small files that modify a base model) rather than full models. This is more efficient and is the standard practice.

______________________________________________________________________

## Settings Explained

| Setting | What It Does |
|---------|--------------|
| **Validation %** | How much data to set aside for checking progress |
| **Shuffle** | Mix up your data randomly (recommended) |
| **Seed** | A number that makes shuffling reproducible |
| **Min Length** | Skip very short pairs (filters out junk) |
| **Private** | Keep your upload hidden from public view |

______________________________________________________________________

## Tips

- **Start with defaults** — The default settings work well for most cases
- **Check before uploading** — Preview your data first to make sure it looks right
- **Use descriptive names** — Name your repos something memorable like `reddit-coding-help` instead of `dataset1`

______________________________________________________________________

## Common Questions

**Do I have to share my data?**
No! Sharing is completely optional. You can build and train locally without ever uploading anything.

**Is sharing free?**
Yes, Hugging Face is free for public datasets and models. Private repos are also free up to a limit.

**Can I delete something after uploading?**
Yes, you can delete repos from your Hugging Face account settings.

______________________________________________________________________

**Next**: [Training Tab](training-tab.md) | **Previous**: [Data Sources Tab](scrape-tab.md) | [Back to Documentation Index](../README.md)
