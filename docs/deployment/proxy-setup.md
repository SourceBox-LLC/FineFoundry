# Proxy Configuration

This page explains how to run FineFoundry behind HTTP/SOCKS proxies and Tor, and how those settings interact with the scrapers.

If you only need a high‑level view of scraping behavior, see the **[Data Sources Tab](../user-guide/scrape-tab.md)** guide. This page dives a bit deeper into configuration.

______________________________________________________________________

## Where proxy settings are used

FineFoundry can route scraper traffic through proxies in two main ways:

1. **Application‑level settings** via the **[Settings Tab](../user-guide/settings-tab.md)**
1. **Module‑level settings** in the scraper helpers under `src/scrapers/`

Under the hood, the scrapers use two key variables:

- `PROXY_URL` – proxy URL, for example `socks5h://127.0.0.1:9050` for Tor.
- `USE_ENV_PROXIES` – whether to respect standard `HTTP_PROXY` / `HTTPS_PROXY` environment variables.

These are defined per scraper, for example:

- 4chan: `src/scrapers/fourchan_scraper.py`
- Reddit: `src/scrapers/reddit_scraper.py`
- Stack Exchange: `src/scrapers/stackexchange_scraper.py`

The Settings tab provides a user‑friendly way to control these without editing code.

______________________________________________________________________

## Using Tor (SOCKS5 proxy)

A common setup is to route scraper traffic through Tor using a local SOCKS5 proxy.

### 1. Start Tor locally

Make sure Tor is running and listening on a port such as `9050`. For example:

```bash
# Check that something is listening on port 9050
netstat -an | grep 9050
```

### 2. Configure the proxy in FineFoundry

In the **Settings** tab:

1. Open the **Proxy Settings** section.
1. Set the **Proxy URL** to `socks5h://127.0.0.1:9050`.
1. Enable the relevant toggle to route scraper traffic via this proxy.

This will adjust the underlying `PROXY_URL` values used by the scrapers so that Scrape and related operations go through Tor.

### 3. Verify behavior

- Run a small scrape (for example, a few threads) and check that it succeeds.
- If it fails, test without the proxy enabled to confirm whether the issue is specific to Tor.

For more troubleshooting steps, see **Proxy connection fails** in the **[Troubleshooting Guide](../user-guide/troubleshooting.md#scraping-issues)**.

______________________________________________________________________

## Using HTTP/HTTPS proxies via environment variables

If you already rely on system‑level HTTP or HTTPS proxies, you can have the scrapers respect those environment variables.

1. Set `HTTP_PROXY` and/or `HTTPS_PROXY` in your shell before launching FineFoundry.
1. Ensure `USE_ENV_PROXIES` is enabled for the relevant scrapers (either via the Settings tab or by editing the helper modules directly for advanced use cases).

Example on Linux/macOS:

```bash
export HTTP_PROXY="http://127.0.0.1:8080"
export HTTPS_PROXY="http://127.0.0.1:8080"
uv run src/main.py
```

Refer to your proxy documentation for the correct URL, authentication options, and security considerations.

______________________________________________________________________

## Performance and reliability notes

- **Latency**: Tor and some HTTP proxies can be significantly slower than direct connections. Expect longer scrape times.
- **Rate limits**: Proxies do not eliminate platform rate limits. Be respectful of target sites and their policies.
- **Failure modes**: Many scrape issues with proxies come from misconfigured URLs or unreachable proxy servers. Always test without a proxy to isolate the problem.

For more detail on how scrapers behave and what configuration knobs exist, see:

- **[Data Sources Tab](../user-guide/scrape-tab.md)** – user‑level scraping configuration.
- **[Proxy Configuration](../../README.md#proxy-configuration)** in the main README – programmatic examples with `PROXY_URL`/`USE_ENV_PROXIES`.
