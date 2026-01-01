# Authentication

FineFoundry connects to Hugging Face for dataset and model hosting, and optionally to Runpod for cloud GPU training. This guide explains how to set up credentials for both.

## Hugging Face

You'll need a Hugging Face token with write permissions to push datasets or adapters. Create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

FineFoundry looks for your token in three places, in order:

1. The HF Token field in the Settings tab
1. The `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` environment variable
1. A cached login from `huggingface-cli login`

Use whichever method fits your workflow—you don't need all three.

### Storing in Settings (Easiest)

Open the Settings tab, paste your token in the HF Token field, click Test to verify it works, then Save. That's it—the Publish tab will use this token when pushing to the Hub.

### Using Environment Variables

If you'd rather not store tokens in the app, set them in your shell before launching:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxx"
uv run src/main.py
```

On Windows PowerShell, use `$env:HF_TOKEN="hf_xxxxxxxxxxxxx"`. FineFoundry picks up the environment variable automatically.

### Using the Hugging Face CLI

If you've already logged in with `huggingface-cli login`, FineFoundry will use that cached credential as a fallback when no token is set in Settings or the environment.

### Verifying It Works

Click Test in the Settings tab to check connectivity. For a real test, try pushing a small dataset from the Publish tab and check your Hugging Face account to see if it appeared.

### Security Tips

Treat tokens like passwords. Never commit them to Git or share them in logs. Environment variables are generally more secure than pasting into the UI, especially on shared machines.

## Runpod

For cloud training, you'll need a Runpod API key. Get it from [runpod.io/console/user/settings](https://runpod.io/console/user/settings).

Paste it in the Settings tab under Runpod Settings, click Test, then Save. The Training tab uses this key to create network volumes, pod templates, and manage training pods.

### Verifying It Works

Click Test in Settings. Then try the Ensure Infrastructure controls in the Training tab—if it can create a network volume and template, you're set.

### Security Tips

Keep your API key private and revoke it from the Runpod console when you no longer need it. Consider separate keys for experimentation vs production workloads.

## Offline Mode

When Offline Mode is enabled, all Hub and Runpod features are disabled regardless of whether you have valid credentials configured.

______________________________________________________________________

## Quick Summary

Set up credentials in the Settings tab, verify with the Test buttons, and try a small push or pod creation to confirm everything works. If something fails, check the [Troubleshooting Guide](troubleshooting.md#authentication-issues).
