# Troubleshooting Guide

Common issues and their solutions for FineFoundry.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Scraping Issues](#scraping-issues)
- [Synthetic Data Generation Issues](#synthetic-data-generation-issues)
- [Dataset Building Issues](#dataset-building-issues)
- [Training Issues](#training-issues)
- [Merge Issues](#merge-issues)
- [Authentication Issues](#authentication-issues)
- [Performance Issues](#performance-issues)
- [Logging & Debugging](#logging--debugging)

______________________________________________________________________

## Installation Issues

### "Python command not found"

**Problem**: `python` or `python3` not recognized

**Solution**:

- Ensure Python 3.10+ is installed
- On Windows, use `py` instead of `python`
- Add Python to your PATH environment variable

### "uv command not found"

**Problem**: `uv` is not installed

**Solution**:

```bash
# Install uv
pip install uv

# Or use pip directly
pip install -r requirements.txt
python src/main.py
```

### "Module not found" errors

**Problem**: Missing dependencies

**Solution**:

```bash
# With uv
uv sync

# With pip
pip install -r requirements.txt --upgrade
```

### SSL/Certificate Errors

**Problem**: `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution**:

```bash
# Update certifi
pip install --upgrade certifi

# Or update all SSL-related packages
pip install --upgrade certifi urllib3 requests
```

______________________________________________________________________

## Scraping Issues

### "No data found" after scraping

**Possible causes**:

1. No boards selected
1. Max Pairs set to 0
1. Network connectivity issues
1. Boards are currently slow/empty

**Solutions**:

- Verify boards are actually selected (highlighted)
- Set Max Pairs > 0 (try 500 for testing)
- Check your internet connection
- Try different boards
- Disable proxy if enabled and test
- Check logs for specific errors

### Scraping is very slow

**Solutions**:

- Decrease Delay (but respect rate limits - minimum 0.3s recommended)
- Increase Max Threads (but more threads = more API requests)
- Check your network speed
- If using proxy (Tor), it may be slow - try without proxy
- Some boards are naturally slower than others

### "Too few pairs" with contextual mode

**Solutions**:

- Reduce "Last K" value (try 3-4 instead of 6)
- Uncheck "Require question" option
- Increase Max Threads
- Increase Max Pairs
- Try different boards (some have more conversation depth)
- Switch strategy from "quote_chain" to "cumulative" or "last_k"

### Proxy connection fails

**Problem**: Tor or custom proxy not working

**Solution**:

1. Verify Tor is running:
   ```bash
   # Check if Tor is listening on port 9050
   netstat -an | grep 9050
   ```
1. Try without proxy first to isolate the issue
1. Verify proxy URL format: `socks5h://127.0.0.1:9050`
1. Check proxy settings in [Settings Tab](settings-tab.md)
1. See [Proxy Setup Guide](../deployment/proxy-setup.md)

______________________________________________________________________

## Synthetic Data Generation Issues

### Model loading takes a long time

**Problem**: First synthetic generation run takes 30-60+ seconds before any progress

**Solution**:

- This is expected behavior — the model must be downloaded and loaded on first run
- A snackbar notification appears immediately when you click Start
- Subsequent runs are faster as the model is cached
- Ensure you have sufficient disk space for model weights (~3-6 GB)

### "No pairs generated" after synthetic generation

**Possible causes**:

1. Document is too short or has no extractable content
1. Max Chunks set too low
1. PDF/document parsing failed

**Solutions**:

- Try a different document format (TXT often works best)
- Increase Max Chunks parameter
- Check logs for parsing errors
- Verify the document has actual text content (not just images)

### Synthetic generation fails with model errors

**Problem**: Errors related to model loading or CUDA

**Solutions**:

- Ensure you have sufficient GPU VRAM (4GB+ recommended)
- Try a smaller model if available
- Check that PyTorch and CUDA are properly installed
- Close other GPU-intensive applications

### Generated pairs are low quality

**Solutions**:

- Enable **Curate** option with a higher threshold (e.g., 8.0)
- Try a different generation type (qa, cot, summary)
- Use higher quality source documents
- Increase Num Pairs to generate more candidates for curation

______________________________________________________________________

## Dataset Building Issues

### "Failed to load JSON file"

**Problem**: JSON file is corrupted or wrong format

**Solution**:

1. Validate JSON syntax:
   ```bash
   python -m json.tool your_file.json
   ```
1. Check file encoding (should be UTF-8)
1. Verify schema matches expected format:
   ```json
   [
     {"input": "text", "output": "text"}
   ]
   ```
1. Re-scrape if file is corrupted

### "No records after filtering"

**Problem**: All records filtered out due to min length or empty fields

**Solution**:

- Reduce minimum length requirement
- Check that scraped data has actual content
- Preview raw JSON to verify data quality
- Check logs for specific filtering reasons

### Push to Hub fails (401/403)

**Problem**: Authentication error

**Solution**:

1. Verify HF token has write permissions
1. Check token in Settings tab or environment variable
1. Try logging in via CLI:
   ```bash
   huggingface-cli login
   ```
1. Verify repo ID format: `username/repo-name`
1. Check if repo already exists and you have access
1. See [Authentication Guide](authentication.md)

______________________________________________________________________

## Training Issues

### "CUDA out of memory"

**Problem**: GPU doesn't have enough VRAM

**Solution**:

- Enable LoRA (reduces memory significantly)
- Reduce per-device batch size
- Increase gradient accumulation steps
- Use a smaller base model
- Use 4-bit quantized models (e.g., models with `bnb-4bit`)
- Disable packing if enabled

On a single 12 GB GPU (e.g., RTX 3060), a conservative starting point for 8B 4-bit models is:

- Batch size per device: 1–2
- Grad accumulation: 2–4
- Max steps: 50–200 for quick experiments

If you still hit OOM, also reduce sequence length and/or disable evaluation during training.

For local Docker training, the Training tab's Beginner preset **Auto Set (local)** automatically picks conservative batch size, grad accumulation, and max steps based on your detected GPU VRAM. Use this preset for first runs on consumer GPUs if you're unsure what values to choose.

### Inference tab: "Adapter directory doesn't look like a valid LoRA adapter"

**Problem**: The Inference tab refuses to validate an adapter directory and shows a red error like:

- *"Adapter directory is missing or invalid"*, or
- *"Adapter directory doesn't look like a valid LoRA adapter"*.

**Causes**:

- The path points to the **parent run folder** instead of the adapter subfolder.
- The directory is empty or missing key adapter files.

**Solution**:

1. Make sure you're selecting the **adapter subdirectory** from a completed fine-tuning run (often named `adapter/`).
1. Verify the folder contains at least one of:
   - `adapter_config.json`
   - LoRA weight files such as `*.safetensors` or `*.bin`
1. If you're unsure, click **Use latest local training** in the Inference tab to automatically populate the adapter path and base model from your most recent local run.
1. After fixing the path, wait for the Inference tab to re-run validation (spinner + snackbar) before trying to generate.

### Training pod won't start

**Problem**: Runpod pod fails to start or reach ready state

**Solution**:

1. Check Runpod dashboard for pod status
1. Verify GPU availability in selected region
1. Check if you have sufficient credits
1. Try a different GPU type
1. Verify network volume is accessible
1. Check pod logs in Runpod dashboard

### "Network volume not found"

**Problem**: Training can't find /data mount

**Solution**:

1. Verify network volume exists in Runpod
1. Check volume is attached to template
1. Ensure mount path is `/data` (not `/workspace`)
1. Click "Ensure Infrastructure" in Training tab
1. Wait for volume to sync (can take a few minutes)

### Training stops/fails unexpectedly

**Solutions**:

- Enable "Auto-resume" to recover from interruptions
- Check logs for specific error messages
- Verify dataset is accessible
- Check disk space on network volume
- Monitor pod status in Runpod dashboard

### Local training: container exits with code 137

**Problem**: Local Docker training exits with status code 137.

**Cause**: This usually means the container was killed by the OS (often an OOM killer) or manually stopped.

**Solution**:

- Treat it like a CUDA OOM: reduce per-device batch size, increase grad accumulation, or use a smaller model.
- Make sure other heavy GPU/CPU workloads are closed while training.
- Check local logs (Training → Local Docker section) for additional error messages.

### Local training: Hugging Face auth errors inside container

**Problem**: Training completes but fails at the end with a `huggingface_hub` error (e.g., around `api.whoami()`), especially when `--push` is enabled.

**Causes**:

- HF token not present inside the Docker container.
- Token lacks required write permissions.

**Solutions**:

1. In **Settings → Hugging Face**, save a valid token with write access, or export `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` before launching the app.
1. In the **Local Docker: Run Training** section, enable **"Pass HF token to container"** so the token is forwarded as env vars.
1. If you don't need to push to the Hub for a given run, disable **Push to Hub** to avoid calling Hub APIs at the end.

### Config‑related mistakes

**Problem**: Training fails or behaves unexpectedly after manual changes to many fields.

**Solution**:

- Use the Training tab's **Save current setup** buttons (or Configuration mode) to snapshot known‑good setups.
- Re‑load a saved config instead of re‑entering values manually, especially when switching between Runpod and local runs.
- Config JSON files live under `src/saved_configs/`, and the last used config auto-loads on startup.
- Enable DEBUG logging (see [Logging Guide](../development/logging.md))

______________________________________________________________________

## Merge Issues

### "Merged dataset not found"

**Problem**: Can't find merged dataset to download

**Solution**:

- Verify merge completed successfully (check Status section)
- Check the output path you specified
- Look in project root directory
- Check logs for actual save location
- Try Preview Merged button to verify it exists

### Column mapping fails

**Problem**: Input/output columns not detected

**Solution**:

- Manually specify column names for HF datasets
- For JSON files, ensure schema matches:
  ```json
  [{"input": "...", "output": "..."}]
  ```
- Check for typos in column names
- Verify dataset actually has the columns you specified

### Out of memory during merge

**Problem**: Large datasets cause memory issues

**Solution**:

- Merge fewer datasets at once
- Use HF dataset dir format instead of JSON
- Close other applications
- Merge in batches
- Upgrade system RAM if consistently hitting limits

______________________________________________________________________

## Authentication Issues

### Hugging Face token not working

**Problem**: Token rejected or not recognized

**Solution**:

1. Generate a new token with write permissions: https://huggingface.co/settings/tokens
1. Copy token exactly (no extra spaces)
1. Paste in Settings tab HF Token field
1. Or set environment variable:
   ```bash
   # Linux/macOS
   export HF_TOKEN="hf_xxxxxxxxxxxxx"

   # Windows PowerShell
   $env:HF_TOKEN="hf_xxxxxxxxxxxxx"

   # Windows CMD
   set HF_TOKEN=hf_xxxxxxxxxxxxx
   ```
1. Restart the application after setting env variable

### Runpod API key issues

**Problem**: Can't access Runpod features

**Solution**:

1. Get API key from Runpod Settings: https://runpod.io/console/user/settings
1. Paste in Settings tab
1. Click "Save Runpod Settings"
1. Verify key has correct permissions
1. Try refreshing the page/app

______________________________________________________________________

## Performance Issues

### Application is slow/laggy

**Solutions**:

- Close unused tabs in the application
- Reduce preview size in dataset views
- Clear logs periodically
- Close other resource-intensive applications
- Check system resources (CPU, RAM)
- Disable unnecessary analysis modules

### Large dataset previews feel slow

**Solutions**:

- Use paginated preview dialogs instead of inline previews
- Open datasets in external tools for large datasets
- Use HF dataset dir format for better performance
- Consider dataset subsampling for preview purposes

### Logs filling up disk space

**Solutions**:

- Log files auto-rotate at 10MB
- Old backups (`.log.1` through `.log.5`) can be safely deleted
- See [Logging Guide](../development/logging.md) for details
- Disable DEBUG mode if enabled

______________________________________________________________________

## Logging & Debugging

### Enable debug logging

To see detailed diagnostic information:

```bash
# Linux/macOS
export FINEFOUNDRY_DEBUG=1
uv run src/main.py

# Windows PowerShell
$env:FINEFOUNDRY_DEBUG=1
uv run src/main.py
```

### View logs

```bash
# Real-time monitoring
tail -f logs/__main__.log
tail -f logs/helpers_merge.log

# Search for errors
grep "ERROR" logs/*.log

# View recent activity
tail -100 logs/__main__.log
```

### Log locations

- `logs/__main__.log` - Main application logs
- `logs/helpers_merge.log` - Merge operation logs
- Additional module logs as needed

See the complete [Logging Guide](../development/logging.md) for more details.

______________________________________________________________________

## Still Having Issues?

### Check the logs

Enable DEBUG mode and check log files for detailed error information.

### Search existing issues

Check [GitHub Issues](https://github.com/SourceBox-LLC/FineFoundry/issues) to see if others have encountered the same problem.

### Report a bug

If you've found a bug:

1. Enable DEBUG logging
1. Reproduce the issue
1. Collect relevant log files
1. Create a GitHub issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Log excerpts (remove sensitive info)
   - System information (OS, Python version)
   - Screenshots if applicable

### Ask for help

- [GitHub Discussions](https://github.com/SourceBox-LLC/FineFoundry/discussions)
- Include relevant context and logs
- Be specific about what you tried

______________________________________________________________________

**Related Documentation**:

- [Logging Guide](../development/logging.md)
- [Authentication](authentication.md)
- [FAQ](faq.md)
- [Proxy Setup](../deployment/proxy-setup.md)

______________________________________________________________________

**Back to**: [Documentation Index](../README.md)
