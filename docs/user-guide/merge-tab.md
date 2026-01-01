# Merge Datasets Tab

This is where you combine data from different sources into one big dataset. Great for when you've collected from multiple places and want to train on everything together.

![Merge Datasets Tab](../../img/new/ff_merge_datasets.png)

## Why Merge?

- **Variety** — Mix Reddit + Stack Exchange + 4chan for diverse training data
- **Size** — Combine small collections into one larger dataset
- **Experimentation** — Add external datasets from Hugging Face to your own

______________________________________________________________________

## How to Merge (Step by Step)

1. **Click "Add Dataset"** — Creates a row for each source you want to combine
1. **Select your sources** — Pick from your collections or enter a Hugging Face dataset
1. **Name your merged dataset** — Something descriptive like "combined_reddit_stackoverflow"
1. **Click "Merge Datasets"**

That's it! Your combined data appears as a new collection you can use for training.

______________________________________________________________________

## Merge Methods

You have two options for how to combine:

### Concatenate (Default)

Stacks all data together in order.

- Dataset A rows, then Dataset B rows, then Dataset C rows...
- **Use when:** Order doesn't matter, you just want everything together

### Interleave

Alternates between sources.

- Row from A, row from B, row from A, row from B...
- **Use when:** You want variety throughout (helps training see diverse examples)

______________________________________________________________________

## Adding Different Sources

### From Your Collections

1. Set source type to "Database Session"
1. Pick a collection from the dropdown
1. Columns are mapped automatically

### From Hugging Face

1. Set source type to "Hugging Face"
1. Enter the dataset name (like `databricks/dolly-15k`)
1. Pick the split (usually "train")
1. FineFoundry auto-detects the columns (or you can specify them manually)

______________________________________________________________________

## Examples

### Combining Two Collections

1. Add Dataset → Select your Reddit collection
1. Add Dataset → Select your Stack Exchange collection
1. Name it "reddit_stackoverflow_mix"
1. Click Merge

### Adding an External Dataset

1. Add Dataset → Select your local collection
1. Add Dataset → Set to Hugging Face, enter `OpenAssistant/oasst1`
1. Name it "my_data_plus_oasst"
1. Click Merge

______________________________________________________________________

## After Merging

When the merge finishes:

- **Preview** — Click to see the first 100 rows
- **Download** — Save a copy to your Downloads folder
- **Use it** — The merged dataset appears in Training tab dropdowns

______________________________________________________________________

## Tips

- **Check before merging** — Preview each source first to make sure it looks right
- **Use descriptive names** — You'll thank yourself later when you have many merged datasets
- **Start small** — Test with smaller datasets before merging huge ones

______________________________________________________________________

## Common Questions

**Can I merge more than two datasets?**
Yes! Click "Add Dataset" as many times as you need.

**What if my datasets have different column names?**
FineFoundry auto-detects common patterns (input/output, prompt/response, question/answer). If it can't figure it out, you can manually specify the columns.

**Why is merging slow?**
Large datasets take time. The status section shows progress for each source.

**Can I undo a merge?**
The merge creates a new dataset—your original sources are untouched. Just delete the merged one if you don't want it.

______________________________________________________________________

**Next**: [Analysis Tab](analysis-tab.md) | **Previous**: [Training Tab](training-tab.md) | [Back to Documentation Index](../README.md)
