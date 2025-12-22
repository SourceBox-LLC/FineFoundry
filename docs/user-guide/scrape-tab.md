# Data Sources Tab

The Data Sources tab is where you collect the raw material for training—conversational data organized as input/output pairs. You can scrape from 4chan, Reddit, or Stack Exchange, or generate synthetic training data from your own documents using local LLMs.

![Data Sources Tab](../../img/new/ff_data_sources.png)

## How It Works

The workflow is straightforward: pick a data source, configure a few parameters, and hit Start. The tab shows you progress and logs in real time as data comes in. When it's done, you can preview what you collected in a two-column grid before moving on to publishing, merging, or training.

Everything you collect gets saved to the database automatically, so you won't lose your work. The data uses a simple schema—either standard input/output pairs or ChatML format with a messages array. If you need to export for external tools, the database helpers can dump to JSON.

```json
{"input": "...", "output": "..."}
```

Or in ChatML format:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

______________________________________________________________________

## The Data Sources

### 4chan

For 4chan scraping, you select which boards to scrape using the chip selector (with Select All and Clear buttons for convenience). The scraper supports two modes: **normal** creates pairs from adjacent posts in a thread, while **contextual** builds richer context by looking at quote chains, cumulative history, or the last K posts.

### Reddit

Reddit scraping works with either subreddit URLs or individual post URLs. Set how many posts to fetch, and the scraper handles comment expansion automatically. Like 4chan, you can choose between simple parent-child pairing or contextual mode for richer conversations.

### Stack Exchange

Stack Exchange pulls Q&A pairs from sites like Stack Overflow, Super User, and others. Just pick a site and set how many pairs you want. The scraper focuses on questions with accepted answers to ensure quality.

### Synthetic

Synthetic generation is different—instead of scraping the web, you feed it your own documents (PDFs, Word docs, text files, HTML, or URLs) and it uses a local LLM to generate training pairs. You can create Q&A pairs, chain-of-thought reasoning examples, or summaries. The Curate option adds quality filtering to keep only the best generations.

### Common Settings

Regardless of which source you choose, you can set the output format (standard input/output or ChatML), a polite delay between requests (for network sources), and a minimum character length to filter out low-quality pairs.

______________________________________________________________________

## Examples

### Quick 4chan Scrape

Select 4chan, pick a couple boards like `pol` and `b`, set Max Threads to 50, Max Pairs to 500, Delay to 0.5, and Min Length to 10. Leave Mode as normal for simple adjacent pairs. Hit Start and watch the progress, then preview what you got.

### Building Contextual Conversations

For richer multi-turn data, use contextual mode with the quote_chain strategy. Set K to 6 for up to 6 turns of context, and Max Input Chars to 2000 to keep inputs manageable. Enable "require question" if you only want pairs where the context contains a question.

### Reddit Subreddit Crawl

Select Reddit, paste a subreddit URL like `https://www.reddit.com/r/LocalLLaMA/`, set Max Posts to 50, and start. The logs show you each post and comment thread as they're fetched.

### Generating Synthetic Data

Select Synthetic, browse for a PDF (a research paper, manual, or any document), and configure the generation. The default model works well for most cases. Set Generation Type to `qa` for question-answer pairs, Num Pairs to 25 per chunk, and Max Chunks to 10. The first run takes 30-60 seconds while the model loads, but subsequent runs are faster.

______________________________________________________________________

## Tips

Start with smaller runs to validate your configuration before scaling up. Watch the logs for network issues, rate limiting, or parsing errors. The min length filter is useful for cutting out low-signal or spammy content.

For synthetic generation, the quality of your output depends heavily on the quality of your input documents. Larger models produce better results but need more VRAM. After any scrape or generation, use the Publish tab to create proper splits and the Analysis tab to check data quality before training.

______________________________________________________________________

## Offline Mode

When Offline Mode is enabled, only the Synthetic source is available—all network-based sources are disabled. The UI shows a banner explaining this, and if you try to start a network scrape, you'll get a snackbar telling you to switch to Synthetic.

______________________________________________________________________

## Related Guides

Once you've collected data, head to the [Publish Tab](build-publish-tab.md) to create train/val/test splits, or the [Merge Datasets Tab](merge-tab.md) if you want to combine multiple sessions. The [Analysis Tab](analysis-tab.md) helps you understand your data quality before training, and the [Training Tab](training-tab.md) is where you fine-tune models on your prepared dataset.

______________________________________________________________________

**Next**: [Publish Tab](build-publish-tab.md) | **Previous**: [Quick Start Guide](quick-start.md) | [Back to Documentation Index](../README.md)
