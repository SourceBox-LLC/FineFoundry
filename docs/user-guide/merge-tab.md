# Merge Datasets Tab

The Merge Datasets tab lets you combine data from multiple sources into a single unified dataset. This is useful when you've scraped from several places, want to mix in existing Hugging Face datasets, or need to consolidate data from different sessions into one training set.

![Merge Datasets Tab](../../img/new/ff_merge_datasets.png)

## How Merging Works

Add the datasets you want to combine—these can be database sessions from your scrape history or datasets from Hugging Face. FineFoundry automatically handles column mapping (it recognizes common patterns like input/output, prompt/response, question/answer), filters out empty rows, and combines everything using your chosen operation.

You can either concatenate datasets (stack them sequentially) or interleave them (alternate records from each source for better distribution). The result gets saved as a new database session, with an optional JSON export if you need a portable file.

## Adding Datasets

Click "Add Dataset" to create a row for each source you want to include. For database sessions, just select from your scrape history—columns are mapped automatically. For Hugging Face datasets, provide the repo ID, split, and optionally specify which columns contain your input and output (though auto-detection usually works).

Use "Clear All" to start over if you need to reconfigure.

## Output Options

Choose whether to save just to the database or also export a JSON file. Give your merged session a descriptive name so you can find it later. After the merge completes, you can preview the first 100 records in a two-column view, and a download button appears if you want to copy the result elsewhere.

## Examples

### Merging Two Scrape Sessions

Add two dataset rows, set each to Database Session, select your sessions from the dropdowns, name the output something like `combined_dataset`, and click Merge Datasets.

### Mixing Hugging Face with Local Data

Add a Hugging Face dataset (repo ID, split) and a database session. Choose whether to concatenate or interleave, set the output format to Database + Export JSON if you want a file, and merge.

### Interleaving for Better Distribution

When merging datasets of different types or sizes, interleaving alternates records from each source. This can help with training by ensuring the model sees varied examples throughout. Add your datasets, set Operation to Interleave, and merge.

## Column Mapping

FineFoundry recognizes common column naming patterns: input/output, prompt/response, question/answer, instruction/completion, and text/label. Everything gets normalized to input/output internally. If auto-detection fails for a Hugging Face dataset, you can manually specify the column names. Rows with empty input or output are automatically filtered out.

## Tips

Preview your datasets individually before merging to catch any issues early. Use the Analysis Tab to check quality before and after. For large merges (over 50k records), expect it to take a few minutes—the status section shows progress for each dataset.

Use descriptive session names like `reddit_and_stackoverflow_merged` so you remember what went into each merge. Keep your source sessions around in case you need to debug or redo the merge differently.

## Downloading Results

After a successful merge, a Download Merged Dataset button appears. Click it to copy the result to another location (like your Downloads folder). The original stays in your project directory.

## Offline Mode

When Offline Mode is enabled, only database sessions are available—Hugging Face sources are disabled with an explanation in the UI.

## Troubleshooting

If you see "Merged dataset not found," the merge may not have completed successfully—check the status section for errors. Column mapping failures usually mean you need to manually specify column names for HF datasets. Out-of-memory errors can be worked around by merging in smaller batches.

Empty results typically mean your source datasets were empty or column mapping failed. Check that your sources contain data and that column names are correctly specified.

______________________________________________________________________

## Related Guides

Build datasets from individual sessions in the [Publish Tab](build-publish-tab.md). Check merged dataset quality in the [Analysis Tab](analysis-tab.md). For debugging, see the [Logging Guide](../development/logging.md). Automate merging with the [CLI](cli-usage.md).

______________________________________________________________________

**Next**: [Analysis Tab](analysis-tab.md) | **Previous**: [Training Tab](training-tab.md) | [Back to Documentation Index](../README.md)
