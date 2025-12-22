# Troubleshooting Guide

When something goes wrong, this guide helps you figure out what happened and how to fix it. The sections below cover the most common issues, organized by what you were trying to do.

______________________________________________________________________

## Installation Issues

### Python Not Found

If your terminal doesn't recognize `python` or `python3`, make sure Python 3.10+ is installed and added to your PATH. On Windows, try `py` instead of `python`.

### Missing uv

If `uv` isn't installed, you can get it with `pip install uv`. Or skip uv entirely and use pip directly:

```bash
pip install -e .
python src/main.py
```

### Missing Dependencies

If you see "Module not found" errors, your dependencies are out of sync. Run `uv sync` (or `pip install -e . --upgrade` if you're using pip) to fix it.

### SSL Certificate Errors

Certificate errors usually mean your SSL packages are outdated. Update them:

```bash
pip install --upgrade certifi urllib3 requests
```

______________________________________________________________________

## Scraping Issues

### No Data Found

If scraping finishes but you don't have any data, check a few things: Are boards actually selected (highlighted)? Is Max Pairs set to something greater than zero? Try 500 for testing. Check your internet connection and try different boards—some may be slow or empty at any given time.

If you're using a proxy, try disabling it temporarily to see if that's the issue. Check the logs for specific error messages.

### Scraping Is Slow

Some slowness is expected due to rate limiting, but if it's unusually slow: check your network speed, try reducing the delay (but keep it at least 0.3s to respect rate limits), and increase Max Threads if your connection can handle it.

If you're using Tor, that adds latency—try without the proxy to see the difference. Some sources are naturally slower than others.

### Too Few Pairs with Contextual Mode

Contextual mode is pickier about what it captures. Try reducing the "Last K" value to 3-4 instead of 6. Uncheck "Require question" if you're not specifically looking for Q&A. Try different boards with more conversation depth, or switch from "quote_chain" to "cumulative" or "last_k" strategy.

### Proxy Connection Fails

First, verify your proxy is actually running. For Tor, check if it's listening on port 9050:

```bash
netstat -an | grep 9050
```

Try without the proxy to isolate the issue. Make sure the URL format is correct—for Tor it should be `socks5h://127.0.0.1:9050`. Check your proxy settings in the [Settings Tab](settings-tab.md) and see the [Proxy Setup Guide](../deployment/proxy-setup.md) for more details.

______________________________________________________________________

## Synthetic Data Generation Issues

### Model Loading Is Slow

The first run takes 30-60+ seconds because the model has to download and load. This is normal—a snackbar appears immediately to let you know it's working. Subsequent runs are much faster since the model is cached. Make sure you have enough disk space (3-6 GB for model weights).

### No Pairs Generated

If generation completes but you get no output, the document might be too short, have no extractable text, or parsing might have failed. Try a different format (TXT usually works best), increase the Max Chunks parameter, and check the logs for parsing errors. Make sure your document has actual text, not just images.

### CUDA or Model Errors

GPU issues usually come down to insufficient VRAM—you need at least 4GB. Try closing other GPU-intensive applications, use a smaller model if available, and verify PyTorch and CUDA are properly installed.

### Low Quality Output

Enable the Curate option with a higher threshold (try 8.0) to filter out weak pairs. Try different generation types—sometimes `cot` or `summary` work better than `qa` depending on your source material. Higher quality source documents produce better results.

______________________________________________________________________

## Dataset Building Issues

### Failed to Load Database Session

The session might not exist or could be corrupted. You can verify it exists by checking the database directly, or simply try re-scraping to create a fresh session.

### No Records After Filtering

If all your records get filtered out, your min length setting is probably too aggressive. Lower it and try again. Also check that your scraped data actually has content—preview the session to verify data quality before building.

### Push to Hub Fails (401/403)

This is an authentication error. Make sure your HF token has write permissions, not just read. Check that it's saved correctly in the Settings tab or set as an environment variable. Verify your repo ID format is `username/repo-name`.

If the repo already exists, make sure you have access to it. Try `huggingface-cli login` to refresh your credentials. See the [Authentication Guide](authentication.md) for more details.

______________________________________________________________________

## Training Issues

### CUDA Out of Memory

Your GPU doesn't have enough VRAM. Try reducing per-device batch size (start with 1-2 for an 8B 4-bit model on a 12GB GPU), increasing gradient accumulation (2-4), using a smaller base model, or disabling packing.

The Beginner preset "Auto Set (local)" automatically configures conservative settings based on your GPU. Use it if you're unsure what values to pick.

### Invalid Adapter Errors in Inference Tab

If the Inference tab won't validate your adapter, the training run either didn't complete or didn't produce valid adapter files. Select a completed run, check that the adapter directory contains `adapter_config.json` and weight files (`.safetensors` or `.bin`), and wait for validation to finish before trying to generate.

### Runpod Pod Won't Start

Check the Runpod dashboard for status. Common causes: GPU unavailable in your region, insufficient credits, or network volume not accessible. Try a different GPU type and verify your network volume exists and is attached to the template.

### Network Volume Not Found

Make sure the volume exists in Runpod and is attached to your template with mount path `/data` (not `/workspace`). Click "Ensure Infrastructure" in the Training tab and give it a few minutes to sync.

### Training Stops Unexpectedly

Enable Auto-resume to recover from interruptions. Check logs for error messages, verify your dataset is accessible, and make sure there's enough disk space on the network volume.

### Container Exits with Code 137

This means the container was killed, usually by the OS OOM killer. Reduce batch size, increase gradient accumulation, use a smaller model, and close other heavy workloads while training.

### HF Auth Errors Inside Container

If push-to-hub fails at the end of training, your HF token isn't reaching the container. In Settings, save a valid token with write access, then enable "Pass HF token to container" in the Local Docker section. Or just disable Push to Hub if you don't need it for this run.

### Config Mistakes

If training behaves unexpectedly after lots of manual changes, use "Save current setup" to snapshot known-good configurations and reload them instead of re-entering everything. The last config auto-loads on startup.

______________________________________________________________________

## Merge Issues

### Merged Dataset Not Found

Check the Status section to confirm the merge actually completed. Look at the output path you specified and check the project root directory. Use the Preview Merged button to verify the result exists.

### Column Mapping Fails

For Hugging Face datasets, manually specify the input/output column names—auto-detection doesn't always work. Database sessions use the standard schema automatically. Double-check for typos in column names.

### Out of Memory During Merge

Large merges can exhaust RAM. Merge fewer datasets at once, close other applications, or merge in batches of 2-3 at a time.

______________________________________________________________________

## Authentication Issues

### Hugging Face Token Not Working

Generate a new token with write permissions at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Copy it exactly (no extra spaces) and paste it in the Settings tab. Alternatively, set the `HF_TOKEN` environment variable and restart the app.

### Runpod API Key Issues

Get your key from the [Runpod console](https://runpod.io/console/user/settings), paste it in Settings, and save. Verify it has the right permissions. If features still don't work, try refreshing the app.

______________________________________________________________________

## Performance Issues

### Application Is Slow

Close unused tabs, reduce preview sizes, and clear old logs periodically. Check system resources—if you're low on RAM or CPU, close other applications. Disable analysis modules you're not using.

### Large Dataset Previews Lag

Use paginated previews instead of inline ones. For very large datasets, open them in external tools or subsample for preview purposes.

### Database Growing Too Large

Clear old logs and delete scrape sessions you no longer need. Disable DEBUG mode if it's enabled. See the [Database Guide](../development/database.md) for maintenance details.

______________________________________________________________________

## Logging & Debugging

### Enabling Debug Mode

For detailed diagnostic information, set the debug flag before launching:

```bash
export FINEFOUNDRY_DEBUG=1
uv run src/main.py
```

On Windows PowerShell, use `$env:FINEFOUNDRY_DEBUG=1`.

### Viewing Logs

Logs go to the SQLite database and print to the console in real time. You can query them directly:

```bash
sqlite3 finefoundry.db "SELECT * FROM app_logs WHERE level='ERROR' ORDER BY timestamp DESC LIMIT 10"
```

See the [Logging Guide](../development/logging.md) for more details on accessing and filtering logs.

______________________________________________________________________

## Still Stuck?

### Check Existing Issues

Someone else may have hit the same problem. Check [GitHub Issues](https://github.com/SourceBox-LLC/FineFoundry/issues) to see if there's already a solution.

### Report a Bug

If you've found something new: enable DEBUG logging, reproduce the issue, and create a GitHub issue with a clear description, steps to reproduce, relevant log excerpts (remove sensitive info), and your system information (OS, Python version).

### Ask for Help

For questions and discussion, use [GitHub Discussions](https://github.com/SourceBox-LLC/FineFoundry/discussions). Include context and logs, and be specific about what you've already tried.

______________________________________________________________________

## Related Guides

- [Logging Guide](../development/logging.md) — detailed logging and debugging
- [Authentication](authentication.md) — credential setup
- [FAQ](faq.md) — quick answers to common questions
- [Proxy Setup](../deployment/proxy-setup.md) — network configuration

______________________________________________________________________

**Back to**: [Documentation Index](../README.md)
