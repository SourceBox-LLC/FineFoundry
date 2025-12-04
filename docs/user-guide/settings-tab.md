# Settings Tab

The Settings tab centralizes configuration for **authentication**, **proxies**, and optional **Ollama** integration used by other parts of the app.

Use this tab to:

- Configure and persist **Hugging Face** and **Runpod** credentials
- Set up **proxy behavior** for scrapers
- Configure **Ollama** connection for optional dataset card generation

---

## Overview

Typical workflow:

1. Enter and save your **Hugging Face** token.
2. (Optional) Enter and save your **Runpod** API key.
3. Configure proxy behavior if you use Tor or HTTP proxies.
4. (Optional) Point the app at an **Ollama** instance and test connectivity.

These settings are used by the Scrape, Build & Publish, Training, and tooling features throughout the app.

---

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

---

## Tips & Best Practices

- Prefer storing long‑lived tokens in **environment variables** when possible, then confirm them via the Settings tab.
- Use the **Test** buttons after changing tokens or API keys.
- If scraping over Tor or an HTTP proxy, configure proxy settings here and verify scrapers behave as expected.

---

## Related Topics

- [Authentication](authentication.md) – Hugging Face and Runpod auth in more detail.
- [Proxy Setup](../deployment/proxy-setup.md) – Tor and custom proxy configuration.
- [Training Tab](training-tab.md) – uses Hugging Face and Runpod settings.
- [Build & Publish Tab](build-publish-tab.md) – uses Hugging Face token for pushing.

---

**Previous**: [Analysis Tab](analysis-tab.md) | [Back to Documentation Index](../README.md)
