# Analysis Tab

This is where you check your data quality before training. Think of it as a health checkup for your dataset—it can spot problems before they waste hours of training time.

![Dataset Analysis Tab](../../img/new/ff_data_analysis.png)

## Why Analyze?

Training on bad data = bad results. Analysis helps you catch issues like:

- Too many duplicate entries
- Unbalanced content (all short responses, or all negative sentiment)
- Low-quality or gibberish text

**5 minutes of analysis can save hours of wasted training.**

______________________________________________________________________

## How to Analyze

1. **Select a dataset** from the dropdown
1. **Choose what to check** — Toggle on the modules you want
1. **Click "Analyze Dataset"**
1. **Review the results** — Charts and numbers show what's in your data

______________________________________________________________________

## What Can You Check?

### Basic Stats

Quick overview of your data:

- How many entries you have
- Average length of inputs and outputs
- **Look for:** Too few entries (< 100), very short texts

### Duplicates

How much repeated content is in your data:

- **Low duplicates (< 5%)** — Good!
- **High duplicates (> 20%)** — Consider cleaning your data

### Sentiment

The emotional tone of your content:

- Positive, negative, or neutral distribution
- **Look for:** Unexpected skew (all negative when you expected balanced)

### Length Balance

Distribution of short vs. medium vs. long entries:

- **Look for:** Heavy skew toward one length (may affect training)

### Extra Checks

Additional quality signals:

- **Toxicity** — Potentially offensive content
- **Readability** — How complex the text is
- **Data leakage** — When input and output are too similar

______________________________________________________________________

## When Should I Analyze?

- **After collecting data** — Before doing anything else
- **After merging** — Combining sources can introduce duplicates
- **Before training** — Final check that everything looks good

______________________________________________________________________

## Reading the Results

### Good Signs

- Duplicate rate under 10%
- Balanced sentiment (unless you want a specific tone)
- Mix of short, medium, and long entries
- Low toxicity (unless that's intentional)

### Warning Signs

- Duplicate rate over 25%
- Extremely short average lengths (< 50 characters)
- All entries clustering in one category
- High data leakage score

______________________________________________________________________

## What to Do About Problems

**Too many duplicates?**

- Go back to Data Sources and collect from different boards/subreddits
- Or filter your data manually

**Unbalanced sentiment?**

- Collect from different sources
- This might be fine depending on your goal

**Very short entries?**

- Increase the "Min Length" setting when collecting
- Collect from sources with longer discussions

**High toxicity?**

- May be expected for some sources (like 4chan)
- Consider if this matches your intended use case

______________________________________________________________________

## Tips

- **Don't obsess over perfect numbers** — These are guidelines, not rules
- **Context matters** — A 4chan dataset will look different from a Stack Overflow one
- **Run analysis multiple times** — Before and after each processing step

______________________________________________________________________

**Next**: [Settings Tab](settings-tab.md) | **Previous**: [Merge Datasets Tab](merge-tab.md) | [Back to Documentation Index](../README.md)
