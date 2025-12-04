# Merge Datasets Tab

The Merge Datasets tab allows you to combine multiple datasets from different sources into a single unified dataset. This is useful for:

- Combining data from multiple scraping sessions
- Merging Hugging Face datasets with local JSON files
- Creating larger, more diverse training datasets
- Consolidating datasets from different sources

![Merge Datasets Tab](../../img/ff_merge.png)

## Overview

The merge process:
1. Loads datasets from multiple sources (JSON files or Hugging Face repos)
2. Automatically maps input/output columns
3. Filters empty rows
4. Combines datasets using your chosen operation
5. Saves to JSON or Hugging Face dataset format
6. Provides inline preview of merged results

## Interface Sections

### 1. Operation

Choose how to combine your datasets:

- **Concatenate** (default): Stack all datasets sequentially
- **Interleave**: Alternate records from each dataset for better distribution

### 2. Datasets

Add multiple data sources to merge:

**For each dataset row, configure:**

- **Source**: Choose between:
  - **Hugging Face**: Load from Hugging Face Hub
  - **JSON file**: Load from local JSON file

**Hugging Face options:**
- **Dataset repo**: e.g., `username/dataset-name`
- **Split**: train, validation, or test
- **Config**: (optional) dataset configuration name
- **Input column**: Column name for inputs (auto-detected if empty)
- **Output column**: Column name for outputs (auto-detected if empty)

**JSON file options:**
- **JSON path**: Path to your local JSON file
- Input/output columns are automatically mapped from the file structure

**Actions:**
- **➕ Add Dataset**: Add another dataset to merge
- **Clear All**: Remove all dataset rows and start over

### 3. Output

Configure where and how to save the merged result:

- **Output Format**:
  - **JSON file**: Save as a single JSON file
  - **HF dataset dir**: Save as a Hugging Face dataset directory
- **Save directory/filename**: Where to save the merged dataset
  - For JSON: e.g., `merged_dataset.json`
  - For HF: e.g., `merged_dataset` (directory name)

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

### Example 1: Merge Two JSON Files

1. Click **➕ Add Dataset** twice to create two rows
2. For each row:
   - **Source**: JSON file
   - **JSON path**: Path to your file (e.g., `scraped_data_1.json`, `scraped_data_2.json`)
3. **Output Format**: JSON file
4. **Save directory**: `combined_dataset.json`
5. Click **Merge Datasets**

### Example 2: Combine Hugging Face Dataset with Local Data

1. **First row**:
   - **Source**: Hugging Face
   - **Dataset repo**: `username/existing-dataset`
   - **Split**: train
2. **Second row**:
   - **Source**: JSON file
   - **JSON path**: `my_new_data.json`
3. **Output Format**: HF dataset dir
4. **Save directory**: `enhanced_dataset`
5. **Operation**: Concatenate
6. Click **Merge Datasets**

### Example 3: Interleave Multiple Datasets

For better data distribution when merging datasets of different types:

1. Add 3+ datasets (any combination of HF and JSON)
2. **Operation**: Interleave
3. **Output Format**: JSON file
4. **Save directory**: `interleaved_dataset.json`
5. Click **Merge Datasets**

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
- **Use JSON** when:
  - You need a simple, portable format
  - You're merging small to medium datasets (< 100k records)
  - You want to quickly inspect the data
- **Use HF dataset dir** when:
  - You have large datasets (> 100k records)
  - You plan to train with Hugging Face `datasets`
  - You need memory-efficient data loading

### Performance
- Large merges (> 50k records) may take a few minutes
- The status section shows progress for each dataset
- Use **Cancel** if you need to stop a long-running operation

### Organization
- Use descriptive filenames: `topic1_and_topic2_merged.json`
- Keep source datasets separate for debugging
- Document your merge operations (which datasets, what operation)

## Download Merged Dataset

After a successful merge, the **Download Merged Dataset** button appears at the bottom of the Status section.

**To download:**
1. Click **Download Merged Dataset**
2. Select a destination folder (e.g., Downloads, backup directory)
3. The dataset is copied to your chosen location
4. You'll see a confirmation message with the full path

**Notes:**
- The original merged dataset remains in the project directory
- The downloaded copy uses the same filename you specified in Output
- Works for both JSON files and HF dataset directories

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
- Use HF dataset dir format instead of JSON
- Close other applications to free up RAM

### Empty merged dataset
- Check that source datasets actually contain data
- Verify column names are correctly mapped
- Check minimum length filters in source datasets

## Related Topics

- [Build & Publish Tab](build-publish-tab.md) - Create datasets from scratch
- [Analysis Tab](analysis-tab.md) - Analyze merged dataset quality
- [Logging Guide](../development/logging.md) - Debug merge issues
- [CLI Usage](cli-usage.md) - Automate merging with scripts

---

**Next**: [Analysis Tab](analysis-tab.md) | **Previous**: [Training Tab](training-tab.md) | [Back to Documentation Index](../README.md)
