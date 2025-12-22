# Proxy Configuration

FineFoundry can route scraper traffic through HTTP, HTTPS, or SOCKS proxies. This is useful if you want to scrape through Tor or need to work behind a corporate proxy.

For basic scraping setup, see the [Data Sources Tab](../user-guide/scrape-tab.md). This page covers proxy configuration in more detail.

## How It Works

Configure proxies in the [Settings Tab](../user-guide/settings-tab.md). The scrapers use two key settings: a proxy URL (like `socks5h://127.0.0.1:9050` for Tor) and an option to respect system environment variables (`HTTP_PROXY`/`HTTPS_PROXY`).

## Using Tor

Make sure Tor is running and listening on port 9050:

```bash
netstat -an | grep 9050
```

In the Settings tab, set the proxy URL to `socks5h://127.0.0.1:9050` and enable the proxy toggle. Run a small test scrape to verify it works. If it fails, try without the proxy to isolate the issue.

## Using Environment Proxies

If you already have `HTTP_PROXY` or `HTTPS_PROXY` set in your environment, enable "Use env proxies" in Settings and FineFoundry will respect those.

```bash
export HTTP_PROXY="http://127.0.0.1:8080"
export HTTPS_PROXY="http://127.0.0.1:8080"
uv run src/main.py
```

## Things to Know

Proxies add latency—expect slower scrapes, especially with Tor. Proxies don't bypass rate limits, so be respectful of target sites. Many proxy issues come from misconfigured URLs or unreachable servers. Always test without the proxy first to isolate problems.

For troubleshooting, see [Proxy connection fails](../user-guide/troubleshooting.md#scraping-issues) in the Troubleshooting Guide.
