# Authentication

FineFoundry talks to a few external services:

- **Hugging Face Hub** – for hosting datasets and (optionally) models.
- **Runpod** – for remote GPU training via pods and a shared network volume.

This page shows how to provide credentials for those services and how to verify that everything is wired up correctly.

For where these settings live in the UI, see the **[Settings Tab](settings-tab.md)** guide.

______________________________________________________________________

## Hugging Face authentication

You need a Hugging Face account and an access token with **write** permissions to push datasets or models.

Create or manage tokens here:

- <https://huggingface.co/settings/tokens>

FineFoundry can discover your token in three ways, in this order:

1. The **HF Token field** in the Settings tab
1. Environment variables (`HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`)
1. A cached login from the `huggingface-cli` tool

You can use whichever method fits your workflow; you do **not** need all three.

**Offline Mode note**: When Offline Mode is enabled, Hugging Face dataset access and Hub push actions are disabled throughout the app, even if you have a valid token configured.

### Option A: Paste a token in the Settings tab (recommended for most users)

1. Open the **Settings** tab.
1. In the **Hugging Face Settings** section, paste your token into the **HF Token** field.
1. Click **Test** to verify that the token works.
1. Click **Save** to persist it.

This token will be used by:

- The **Build & Publish** tab when `Push to Hub` is enabled.
- The **Training** tab when training jobs are configured to push adapters/weights.

### Option B: Use environment variables

If you prefer not to store tokens in the app, you can provide them via environment variables before launching FineFoundry.

On **Linux/macOS**:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxx"
uv run src/main.py
```

On **Windows PowerShell**:

```powershell
$env:HF_TOKEN="hf_xxxxxxxxxxxxx"
uv run src/main.py
```

On **Windows CMD**:

```cmd
set HF_TOKEN=hf_xxxxxxxxxxxxx
python src\main.py
```

FineFoundry will use the `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`, if set) even if you leave the HF Token field blank in Settings.

### Option C: Log in via the Hugging Face CLI

If you already use the Hugging Face CLI on this machine, you can log in once and let FineFoundry reuse that cached auth.

```bash
huggingface-cli login
```

After logging in, start FineFoundry from the same user account. The app will first try the Settings tab token and environment variables; if those are not present, it will fall back to the cached CLI login.

### Verifying Hugging Face authentication

To confirm that auth is working:

1. In the **Settings** tab, click **Test** in the Hugging Face section.
1. In the **Build & Publish** tab, enable **Push to Hub**, set a test `Repo ID` (such as `username/test-dataset`), and build a tiny dataset.
1. Check your Hugging Face account to see whether the dataset repo was created or updated.

If authentication fails, see **Authentication Issues** in the **[Troubleshooting Guide](troubleshooting.md#authentication-issues)**.

### Security best practices

- Treat Hugging Face tokens like **passwords**.
- Never commit tokens to Git, screenshots, or shared logs.
- Prefer environment variables or the local Settings store over hard‑coding tokens into scripts.
- Use different tokens (or scopes) for development vs production where appropriate.

______________________________________________________________________

## Runpod authentication

If you use Runpod for remote training, FineFoundry needs a **Runpod API key**.

**Offline Mode note**: When Offline Mode is enabled, Runpod infrastructure helpers and Runpod training are disabled.

Create or view your key from the Runpod console (you may be redirected between domains):

- <https://runpod.io/console/user/settings>

### Provide your Runpod API key to FineFoundry

1. Open the **Settings** tab.
1. In the **Runpod Settings** section, paste your API key into the **Runpod API Key** field.
1. Click **Test** to verify that the key can reach Runpod APIs.
1. Click **Save** to persist it.

This key is used by the **Training** tab when you select **Runpod – Pod** as the training target. It allows FineFoundry to:

- Create and reuse **Network Volumes** (usually mounted at `/data`).
- Create and reuse **Pod Templates** for training jobs.
- Launch pods, monitor status, and fetch logs.

### Verifying Runpod authentication

To confirm that Runpod auth is working:

1. Click **Test** in the Runpod section of the Settings tab.
1. In the **Training** tab, choose **Runpod – Pod** and use the **Ensure Infrastructure** controls (Network Volume and Template).
1. Check the Runpod console to see whether the volume/template exist and whether pods can be created from the template.

If Runpod operations fail, see the **Runpod API key issues** and related sections in the **[Troubleshooting Guide](troubleshooting.md#authentication-issues)** and the **[Training Tab](training-tab.md)** documentation.

### Security best practices for Runpod

- Keep your API key private and never commit it to Git.
- Revoke keys that are no longer needed from the Runpod console.
- Consider using separate keys or accounts for experimentation vs more critical workloads.

______________________________________________________________________

## Summary

- Use the **Settings** tab as the primary place to paste and manage credentials.
- Environment variables and CLI logins are supported for Hugging Face if you prefer not to store tokens in the UI.
- Always verify auth using the **Test** buttons and a small end‑to‑end action (such as pushing a test dataset or launching a small training job).
