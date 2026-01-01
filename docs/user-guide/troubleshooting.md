# Troubleshooting Guide

Something not working? Find your problem below and follow the fix.

______________________________________________________________________

## App Won't Start

**"Python not found"**

- Make sure Python 3.10+ is installed
- On Windows, try `py` instead of `python`
- Download Python: [python.org/downloads](https://www.python.org/downloads/)

**"Module not found" error**

- Your packages need updating
- Run: `pip install -e . --upgrade`

**App crashes immediately**

- Try running from terminal to see error messages
- Make sure you're in the FineFoundry folder

______________________________________________________________________

## Data Collection Problems

**No data collected**

- Make sure you selected boards/sources (they should be highlighted)
- Check your internet connection
- Try different boards—some may be empty
- Increase Max Pairs (try 500)

**Collection is very slow**

- This is normal—FineFoundry is polite to websites
- Using Tor? That adds extra delay
- Some sources are naturally slower than others

**Got fewer pairs than expected**

- Some content gets filtered for quality
- Try lowering the Min Length setting
- Some threads may be empty

______________________________________________________________________

## Training Problems

**"Out of memory" or "CUDA OOM"**

Your graphics card ran out of space. Try these fixes:

1. **Use "Quick local test" preset** — Uses less memory
1. **In Expert mode, reduce batch size** — Try 1 or 2
1. **Close other programs** — Games, browsers with lots of tabs
1. **Use Runpod instead** — Cloud GPUs have more memory

**Training seems stuck**

- Check the logs—it might just be slow
- Large models can take hours
- Look for error messages in red

**Training finished but model doesn't work**

- Make sure training actually completed (check status)
- The adapter folder should contain files
- Try training for more steps

______________________________________________________________________

## Cloud Training (Runpod) Problems

**Pod won't start**

- Check your Runpod credits/balance
- Try a different GPU type
- Check Runpod's status page for outages

**"Network volume not found"**

- Click "Ensure Infrastructure" in Training tab
- Wait a minute and try again
- Check Runpod dashboard that volume exists

**Training stops unexpectedly**

- Enable "Auto-resume" to recover
- Check if you ran out of credits
- Look at logs for error messages

______________________________________________________________________

## Sharing Problems (Hugging Face)

**"401" or "403" error when pushing**

- Your token might be wrong or expired
- Make sure token has "write" permission
- Go to Settings and re-enter your token

**"Repository not found"**

- Check your repo name format: `username/repo-name`
- Make sure you typed it correctly

______________________________________________________________________

## Testing Problems (Inference)

**"Validating" never finishes**

- The training might not have completed successfully
- Check that adapter files exist in the output folder

**First response is very slow**

- Normal! The model needs to load first (30-60 seconds)
- After that, responses are faster

**Responses don't make sense**

- Your training data might need work
- Try training for more steps
- Check data quality in Analysis tab

______________________________________________________________________

## Merge Problems

**Merge seems to fail silently**

- Check the status section for errors
- Make sure source datasets aren't empty

**Wrong columns merged**

- For Hugging Face datasets, manually specify column names
- Check that column names are spelled correctly

______________________________________________________________________

## Quick Fixes to Try First

Before diving deep, try these:

1. **Restart the app** — Fixes many temporary issues
1. **Check the logs** — Look for red error messages
1. **Try smaller settings** — Fewer threads, smaller batch size
1. **Test your connection** — Can you browse the web normally?

______________________________________________________________________

## Getting More Help

**Still stuck?**

1. **Search existing issues:** [GitHub Issues](https://github.com/SourceBox-LLC/FineFoundry/issues)
1. **Ask the community:** [GitHub Discussions](https://github.com/SourceBox-LLC/FineFoundry/discussions)
1. **Report a bug:** Create a new issue with:
   - What you were trying to do
   - What happened instead
   - Any error messages you saw

______________________________________________________________________

**Back to**: [Documentation Index](../README.md)
