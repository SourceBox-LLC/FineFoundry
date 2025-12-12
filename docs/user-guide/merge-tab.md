# Merge Datasets Tab

The Merge Datasets tab allows you to combine multiple datasets from different sources into a single unified dataset. This is useful for:

- Combining data from multiple scraping sessions
- Merging Hugging Face datasets with database sessions
- Creating larger, more diverse training datasets
- Consolidating datasets from different sources

![Merge Datasets Tab](../../img/ff_merge.png)

## Overview

The merge process:

1. Loads datasets from multiple sources (database sessions or Hugging Face repos)
1. Automatically maps input/output columns
1. Filters empty rows
1. Combines datasets using your chosen operation
1. Saves to the database as a new session (with optional JSON export)
1. Provides inline preview of merged results

## Interface Sections

### 1. Operation

Choose how to combine your datasets:

- **Concatenate** (default): Stack all datasets sequentially
- **Interleave**: Alternate records from each dataset for better distribution

### 2. Datasets

Add multiple data sources to merge:

**For each dataset row, configure:**

- **Source**: Choose between:
  - **Database Session**: Load from a previous scrape session
  - **Hugging Face**: Load from Hugging Face Hub

**Database Session options:**

- **Session**: Select from your scrape history
- Input/output columns are automatically mapped

**Hugging Face options:**

- **Dataset repo**: e.g., `username/dataset-name`
- **Split**: train, validation, or test
- **Config**: (optional) dataset configuration name
- **Input column**: Column name for inputs (auto-detected if empty)
- **Output column**: Column name for outputs (auto-detected if empty)

**Actions:**

- **➕ Add Dataset**: Add another dataset to merge
- **Clear All**: Remove all dataset rows and start over

### 3. Output

Configure where and how to save the merged result:

- **Output Format**:
  - **Database**: Save as a new database session
  - **Database + Export JSON**: Save to database and export to JSON file
- **Session Name**: Name for the new merged session
- **Export Path** (optional): Path for JSON export if selected

**Actions:**

- **Merge Datasets**: Start the merge operation
- **Cancel**: Stop an in-progress merge
- **Refresh**: Clear all results and reset the UI
- **Preview Merged**: View the merged dataset (after merge completes)

### 4. Preview

Shows a two-column preview of the first 100 records from the merged dataset:

- **Input** (left column): Context or questions
- **Output** (right column): Responses or answers

This preview updates automatically after a successful merge.

### 5. Status

Displays real-time progress during the merge operation:

- Loading status for each dataset
- Dataset preparation messages
- Success/error messages
- Final save location

**Download Button** (appears after successful merge):

- **Download Merged Dataset**: Copy the merged dataset to another location
  - Click the button
  - Select a destination folder
  - The dataset is copied with its original filename

## Usage Examples

### Example 1: Merge Two Database Sessions

1. Click **➕ Add Dataset** twice to create two rows
1. For each row:
   - **Source**: Database Session
   - **Session**: Select from your scrape history
1. **Output Format**: Database
1. **Session Name**: `combined_dataset`
1. Click **Merge Datasets**

### Example 2: Combine Hugging Face Dataset with Database Session

1. **First row**:
   - **Source**: Hugging Face
   - **Dataset repo**: `username/existing-dataset`
   - **Split**: train
1. **Second row**:
   - **Source**: Database Session
   - **Session**: Select your scrape session
1. **Output Format**: Database + Export JSON
1. **Session Name**: `enhanced_dataset`
1. **Export Path**: `enhanced_dataset.json`
1. **Operation**: Concatenate
1. Click **Merge Datasets**

### Example 3: Interleave Multiple Datasets

For better data distribution when merging datasets of different types:

1. Add 3+ datasets (any combination of HF and Database Sessions)
1. **Operation**: Interleave
1. **Output Format**: Database
1. **Session Name**: `interleaved_dataset`
1. Click **Merge Datasets**

Result: Records will be alternated from each source dataset.

## Column Mapping

FineFoundry automatically handles column mapping:

- **Auto-detection**: If input/output columns are not specified, the system tries to guess them
- **Common patterns recognized**:
  - `input` / `output`
  - `prompt` / `response`
  - `question` / `answer`
  - `instruction` / `completion`
  - `text` / `label`
- **Normalization**: All datasets are converted to `input`/`output` format internally
- **Filtering**: Rows with empty input or output are automatically removed

## Tips & Best Practices

### Data Quality

- Preview your datasets individually before merging
- Use the [Analysis Tab](analysis-tab.md) to check quality before merging
- Filter or clean problematic data before merging

### Output Format Choice

- **Use Database only** when:
  - You want to keep everything in the database
  - You're merging small to medium datasets (< 100k records)
  - You plan to use the merged data in other tabs
- **Use Database + Export JSON** when:
  - You need a portable file for external tools
  - You want to share the merged data
  - You need a backup of the merged data

### Performance

- Large merges (> 50k records) may take a few minutes
- The status section shows progress for each dataset
- Use **Cancel** if you need to stop a long-running operation

### Organization

- Use descriptive session names: `topic1_and_topic2_merged`
- Keep source sessions separate for debugging
- Document your merge operations (which sessions, what operation)

## Download Merged Dataset

After a successful merge, the **Download Merged Dataset** button appears at the bottom of the Status section.

**To download:**

1. Click **Download Merged Dataset**
1. Select a destination folder (e.g., Downloads, backup directory)
1. The dataset is copied to your chosen location
1. You'll see a confirmation message with the full path

**Notes:**

- The original merged dataset remains in the project directory
- The downloaded copy uses the same filename you specified in Output
- Works for exported JSON files

## Offline Mode

When **Offline Mode** is enabled:

- Hugging Face datasets are disabled.
- Only **Database** sources can be selected.
- Each dataset row shows an inline explanation under the **Source** dropdown indicating that Hugging Face datasets are disabled.

## Troubleshooting

### "Merged dataset not found"

- Ensure the merge completed successfully (check Status section)
- Verify the output path in the Output section
- The dataset must exist before downloading

### Column mapping fails

- Manually specify input/output column names for HF datasets
- Ensure JSON files follow the expected schema:
  ```json
  [
    {"input": "...", "output": "..."},
    ...
  ]
  ```

### Merge fails with error

- Check the Status section for specific error messages
- Verify all dataset paths are correct
- Ensure you have write permissions to the output directory
- Check [Logging Guide](../development/logging.md) for detailed error traces

### Out of memory

- Merge smaller batches of datasets
- Close other applications to free up RAM
- Consider merging in stages (merge 2-3 at a time)

### Empty merged dataset

- Check that source datasets actually contain data
- Verify column names are correctly mapped
- Check minimum length filters in source datasets

## Related Topics

- [Build & Publish Tab](build-publish-tab.md) - Create datasets from scratch
- [Analysis Tab](analysis-tab.md) - Analyze merged dataset quality
- [Logging Guide](../development/logging.md) - Debug merge issues
- [CLI Usage](cli-usage.md) - Automate merging with scripts

______________________________________________________________________

**Next**: [Analysis Tab](analysis-tab.md) | **Previous**: [Training Tab](training-tab.md) | [Back to Documentation Index](../README.md)
