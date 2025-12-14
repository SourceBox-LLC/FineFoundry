# Ethical and Legal Notes

FineFoundry makes it easy to scrape and curate datasets from sources like 4chan, Reddit, and Stack Exchange. That flexibility comes with important ethical and legal responsibilities.

This page summarizes key considerations and points you to related documentation.

______________________________________________________________________

## Content warnings

- Scraped content may be **NSFW, offensive, or harmful**.
- Datasets built with FineFoundry can include hate speech, harassment, or other sensitive material.
- Treat outputs as research artifacts, not production‑ready systems.

If you work with sensitive or user‑generated data, ensure that you:

- Comply with the terms of service of the platforms you scrape.
- Follow local laws and regulations in your jurisdiction.
- Handle personal data carefully and in accordance with privacy regulations.

______________________________________________________________________

## Responsible use

Before using datasets or models derived from FineFoundry in production, you should:

- Apply **filtering and detoxification** techniques.
- Consider **alignment and safety** fine‑tuning.
- Document known limitations and risks in any downstream project.

Be transparent about how your datasets were created and what sources they include.

______________________________________________________________________

## Licensing and attribution

- Respect the licenses and terms of the original content sources.
- Where appropriate, provide attribution or links back to original platforms.
- Review the **[License](../../README.md#license)** section in the main README for FineFoundry's own licensing.

If you are unsure whether a particular use is allowed, seek legal advice.

______________________________________________________________________

## Related documentation

- [Data Sources Tab](scrape-tab.md) – how scraping is configured in the GUI.
- [Scrapers API](../api/scrapers.md) – programmatic scraping interfaces.
- [Troubleshooting](troubleshooting.md) – includes logging guidance to help you understand what the scrapers are doing.

FineFoundry is a powerful tool; please use it thoughtfully and responsibly.
