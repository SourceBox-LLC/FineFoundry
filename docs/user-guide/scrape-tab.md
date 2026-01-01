# Data Sources Tab

This is where you collect data to train your AI. You can grab conversations from websites like Reddit and 4chan, or create training data from your own documents.

![Data Sources Tab](../../img/new/ff_data_sources.png)

## How It Works

1. **Pick a source** — Choose where to get your data from
2. **Set your limits** — How much data do you want?
3. **Click Start** — Watch as data flows in
4. **Preview results** — See what you collected

All your data is **automatically saved**, so you won't lose anything if you close the app.

______________________________________________________________________

## Where Can I Get Data From?

### 4chan
Collects conversations from 4chan boards. Good for casual, unfiltered dialogue.

**How to use:**
1. Click the board chips you want (like `b`, `pol`, `x`)
2. Set how many threads and pairs to collect
3. Click Start

### Reddit
Collects conversations from Reddit posts and comments. Great for topic-specific data.

**How to use:**
1. Paste a subreddit URL (like `https://www.reddit.com/r/LocalLLaMA/`)
2. Or paste a specific post URL
3. Set how many posts to fetch
4. Click Start

### Stack Exchange
Collects question-and-answer pairs from sites like Stack Overflow. Perfect for technical/factual training data.

**How to use:**
1. Pick a Stack Exchange site from the dropdown
2. Set how many Q&A pairs you want
3. Click Start

### Synthetic (From Your Own Documents)
Creates training data from your own files—PDFs, Word docs, or text files. An AI reads your documents and generates question-answer pairs from them.

**How to use:**
1. Browse for your files (or paste URLs)
2. Choose what to generate: Q&A pairs, reasoning examples, or summaries
3. Click Start

**Note:** The first time takes 30-60 seconds to load the AI model. After that, it's faster.

______________________________________________________________________

## Settings Explained

| Setting | What It Does |
|---------|--------------|
| **Max Threads/Posts** | How many pages to collect from |
| **Max Pairs** | Stop after collecting this many conversation pairs |
| **Delay** | Pause between requests (be polite to websites!) |
| **Min Length** | Skip pairs shorter than this (filters out junk) |

**Tip:** Start with small numbers (like 50 threads, 500 pairs) to test, then scale up.

______________________________________________________________________

## Examples

### Quick Test Run
- Source: 4chan
- Boards: Pick 2-3 boards
- Max Threads: 50
- Max Pairs: 500
- Delay: 0.5
- Click Start

This takes about 1-3 minutes and gives you enough data to test the whole workflow.

### Building a Reddit Dataset
- Source: Reddit
- URL: Paste your favorite subreddit
- Max Posts: 100
- Click Start

Good for creating topic-specific training data.

### Creating Data from Your Documents
- Source: Synthetic
- Add your PDF, Word doc, or text file
- Generation Type: `qa` (question-answer pairs)
- Num Pairs: 25
- Click Start

Perfect when you want the AI to learn from specific information.

______________________________________________________________________

## Checking Your Results

After collection finishes, click **"Preview Dataset"** to see what you got. You'll see two columns:

- **Input** — The prompt or question
- **Output** — The response or answer

This is exactly what the AI will learn from during training.

______________________________________________________________________

## Tips for Better Data

- **Quality matters more than quantity** — 500 good pairs often beat 5000 bad ones
- **Check your preview** — Make sure the data looks right before training
- **Use the Analysis tab** — It can spot problems in your data
- **Try different sources** — Mix Reddit + Stack Exchange for variety

______________________________________________________________________

## Common Questions

**Why is collection slow?**
The delay setting adds pauses between requests to avoid overwhelming websites. This is intentional and polite.

**Why did I get fewer pairs than expected?**
Some threads might be empty, or the min length filter removed low-quality pairs. This is usually fine.

**Can I collect from multiple sources?**
Yes! Collect from each source separately, then use the Merge tab to combine them.

______________________________________________________________________

**Next**: [Publish Tab](build-publish-tab.md) | **Previous**: [Quick Start Guide](quick-start.md) | [Back to Documentation Index](../README.md)
