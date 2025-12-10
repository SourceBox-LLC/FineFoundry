# Settings Tab

The Settings tab centralizes configuration for **authentication**, **proxies**, and optional **Ollama** integration used by other parts of the app.

Use this tab to:

- Configure and persist **Hugging Face** and **Runpod** credentials
- Set up **proxy behavior** for scrapers
- Configure **Ollama** connection for optional dataset card generation
- Run a **System Check** to verify that core scraping, build, training, and inference flows work end-to-end

______________________________________________________________________

## Overview

Typical workflow:

1. Enter and save your **Hugging Face** token.
1. (Optional) Enter and save your **Runpod** API key.
1. Configure proxy behavior if you use Tor or HTTP proxies.
1. (Optional) Point the app at an **Ollama** instance and test connectivity.

These settings are used by the Scrape, Build & Publish, Training, and tooling features throughout the app.

______________________________________________________________________

## Layout at a Glance

### 1. Hugging Face Settings

- **HF Token** field
  - Paste a token with the required read/write permissions.
- **Test** button
  - Verifies connectivity to Hugging Face using the provided token or environment variables.
- **Save / Remove** buttons
  - Persist or clear the stored token.

The token is used by:

- Build & Publish tab (Push to Hub)
- Training tab (push adapters/weights)

### 2. Runpod Settings

- **Runpod API Key** field
  - Paste your key from the Runpod console.
- **Test** button
  - Verifies that the key can reach Runpod APIs.
- **Save / Remove** buttons
  - Persist or clear the stored key.

The key is used by:

- Training tab (Runpod infrastructure & pod management)

### 3. Proxy Settings

Controls proxy behavior for scrapers:

- **Enable proxy / Use env proxies** toggles
- **Proxy URL** field
  - Example: `socks5h://127.0.0.1:9050` for Tor.

These settings map onto the underlying helper modules so Scrape and other features respect your proxy configuration.

### 4. Ollama Settings (Optional)

If you use **Ollama** for dataset card drafting or other features:

- **Enable Ollama** toggle
- **Base URL** (e.g. `http://localhost:11434`)
- **Default model** field
- **Models dropdown + Refresh button** – list and select models from the Ollama instance.
- **Test / Save** buttons

### 5. System Check (Diagnostics)

The **System Check** panel lives at the bottom of the Settings tab and provides a one-click health check for your environment.

- **Run system diagnostics** button

  - Executes a diagnostics pipeline using your current Python environment.
  - Runs focused `pytest` groups for key feature areas:
    - **Data Collection** – scraping utilities and scrape-tab orchestration.
    - **Dataset Build** – merge and build/publish pipelines.
    - **Training & Inference** – training config + local training infra, and Quick Local Inference wiring.
  - Then runs the **full test suite** (`pytest tests`) and a **coverage run + report**:
    - `coverage run --source=src -m pytest`
    - `coverage report -m`
  - All output is streamed live into a log view so you can see exactly what passed or failed.

- **Live log viewer**

  - Initially hidden and shown automatically when diagnostics start.
  - Displays detailed stdout/stderr from each step (pytest and coverage commands).
  - Useful for debugging environment issues without leaving the app.

- **System Health Summary**

  - After the run completes, a summary appears **beneath** the log.
  - Results are grouped into card-like sections:
    - **Data Collection** – scraping and scrape orchestration tests.
    - **Dataset Build** – merge and build pipeline tests.
    - **Training & Inference** – training config, local Docker training helpers, and Quick Local Inference UI wiring.
    - **Overall Health** – full test suite and coverage steps.
  - Each card lists its component checks with icons and a clear **Passed / Failed** label and exit code.
  - The overall status line above the log tells you whether everything passed or if some areas failed.

- **Download diagnostics log**

  - A **Download log** button opens a file-save dialog.
  - Saves the full textual log to a `.txt` file so you can:
    - Attach it to bug reports.
    - Share it with collaborators.
    - Keep a local record of a given run.

The System Check is optional but recommended when:

- You first install FineFoundry on a new machine.
- You upgrade dependencies or Python versions.
- You want a quick, visual confirmation that scraping, build, training, and inference are all wired up correctly.

______________________________________________________________________

## Tips & Best Practices

- Prefer storing long‑lived tokens in **environment variables** when possible, then confirm them via the Settings tab.
- Use the **Test** buttons after changing tokens or API keys.
- If scraping over Tor or an HTTP proxy, configure proxy settings here and verify scrapers behave as expected.

______________________________________________________________________

## Related Topics

- [Authentication](authentication.md) – Hugging Face and Runpod auth in more detail.
- [Proxy Setup](../deployment/proxy-setup.md) – Tor and custom proxy configuration.
- [Training Tab](training-tab.md) – uses Hugging Face and Runpod settings.
- [Build & Publish Tab](build-publish-tab.md) – uses Hugging Face token for pushing.

______________________________________________________________________

**Previous**: [Analysis Tab](analysis-tab.md) | [Back to Documentation Index](../README.md)
