# Settings Tab

The Settings tab is your control center for credentials, proxy configuration, and optional integrations. Everything you configure here flows through to the other tabs—your Hugging Face token enables publishing, your Runpod key enables cloud training, and your proxy settings affect how scrapers reach the web.

## Hugging Face

Paste your Hugging Face token here if you want to push datasets or adapters to the Hub. You'll need a token with write permissions, which you can create at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Click Test to verify the token works, then Save to persist it. The Publish tab uses this token when pushing datasets, and Training uses it when you enable push-to-hub for adapters.

If you prefer not to store tokens in the app, you can set the `HF_TOKEN` environment variable instead—FineFoundry will pick it up automatically.

## Runpod

If you're using Runpod for cloud GPU training, paste your API key here. Get it from the [Runpod console settings](https://runpod.io/console/user/settings). Click Test to verify connectivity, then Save.

This key lets the Training tab create and manage network volumes, pod templates, and training pods on your behalf.

## Proxy Settings

If you're scraping through Tor or a custom proxy, configure it here. Toggle on proxy support, enter your proxy URL (like `socks5h://127.0.0.1:9050` for Tor), and the scrapers will route their requests through it.

You can also enable "Use env proxies" to pick up `HTTP_PROXY` and `HTTPS_PROXY` from your environment instead of specifying a URL directly.

## Ollama (Optional)

If you have Ollama running locally, you can connect FineFoundry to it for generating dataset and model cards. Enable Ollama, set the base URL (usually `http://localhost:11434`), and pick a model from the dropdown. Click Test to verify the connection works.

This is entirely optional—you can always write cards manually or use the simple template instead.

## System Check

The System Check panel at the bottom runs diagnostics to verify your environment is set up correctly. It executes the test suite in focused groups (data collection, dataset build, training and inference) and then runs full coverage.

Results appear in a live log view, and after completion you get a summary showing which areas passed or failed. This is useful when you first install FineFoundry, after upgrading dependencies, or whenever something seems off and you want a quick sanity check.

You can download the full diagnostics log to attach to bug reports or share with collaborators.

## Tips

Store long-lived tokens in environment variables when possible—it's more secure than pasting them into the UI, and they persist across sessions. Always click Test after changing credentials to catch typos early. If scraping seems slow or blocked, check your proxy settings here first.

______________________________________________________________________

## Related Guides

For more on authentication, see the [Authentication Guide](authentication.md). Proxy details are in [Proxy Setup](../deployment/proxy-setup.md). The [Training Tab](training-tab.md) and [Publish Tab](build-publish-tab.md) are the main consumers of these settings.

______________________________________________________________________

**Previous**: [Analysis Tab](analysis-tab.md) | [Back to Documentation Index](../README.md)
