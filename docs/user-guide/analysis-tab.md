# Dataset Analysis Tab

The Dataset Analysis tab helps you understand your data before committing to a long training run. You can check for duplicates, analyze sentiment distribution, spot potential data leakage, and assess overall quality—all without writing any code.

![Dataset Analysis Tab](../../img/new/ff_data_analysis.png)

## How It Works

Select a dataset (either a database session from your scrape history or a Hugging Face dataset when online), enable the analysis modules you're interested in, and click Analyze dataset. The results show up as metrics, charts, and sample previews so you can quickly assess whether your data is ready for training.

## Analysis Modules

The tab offers several analysis modules you can toggle on or off depending on what you want to check:

**Basic Stats** gives you record counts and average input/output lengths—a quick sanity check that your dataset has what you expect. **Duplicates & Similarity** uses hashing to estimate how many duplicate pairs you have, which can hurt training if too high. **Sentiment** shows the distribution of positive, negative, and neutral content across your samples.

**Class Balance** buckets your data by length (short, medium, long) so you can see if you're skewed toward one type. The **Extra metrics** section provides lightweight proxy signals for things like coverage overlap, potential data leakage, conversation depth, readability, toxicity, and more. These aren't perfect measures, but they surface potential issues worth investigating.

## When to Run Analysis

Run analysis before committing to a long training run—it's much easier to fix data problems now than to debug a model later. Run it again after major changes to your scraping, merging, or cleaning logic to verify you didn't introduce issues.

## Examples

### Quick Sanity Check

After building a dataset in the Publish tab, load it here and enable Basic Stats, Duplicates, and Sentiment. Confirm your record count looks right, duplicates are low, and sentiment distribution makes sense for your use case.

### Curriculum Planning

If you're planning to train in stages or weight certain examples, check Class Balance and Conversation depth. See if you need to rebalance short vs long samples or adjust which sources you're pulling from.

### Safety Review

Enable Toxicity/Politeness and Data leakage proxies before training. High toxicity might mean you need additional filtering. Leakage flags suggest your input and output columns might share too much content, which could lead to trivial learning.

## Tips

Use analysis iteratively—check your raw scrape, then check again after merging, and once more after filtering. Each step can introduce or fix issues. The Duplicates module is especially useful after merging multiple datasets, since overlapping sources can inflate duplicate rates.

Don't over-rely on proxy metrics. They're heuristics, not ground truth. Use them to surface things worth a closer look, not as definitive judgments.

## Offline Mode

When Offline Mode is enabled, Hugging Face dataset sources are disabled (the UI resets to Database if you were using HF). The HF Inference API backend is also disabled, falling back to local processing. A banner explains what's unavailable.

______________________________________________________________________

## Related Guides

Collect data in the [Data Sources Tab](scrape-tab.md), build splits in the [Publish Tab](build-publish-tab.md), combine datasets in the [Merge Datasets Tab](merge-tab.md), then train in the [Training Tab](training-tab.md).

______________________________________________________________________

**Next**: [Settings Tab](settings-tab.md) | **Previous**: [Merge Datasets Tab](merge-tab.md) | [Back to Documentation Index](../README.md)
