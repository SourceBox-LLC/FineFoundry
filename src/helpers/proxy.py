import os
from typing import Optional

from scrapers import fourchan_scraper as sc
try:
    from scrapers import reddit_scraper as rdt
except Exception:
    rdt = None
from scrapers import stackexchange_scraper as sx


def _env_truthy(val: Optional[str]) -> bool:
    """Interpret common truthy strings from environment variables."""
    try:
        return str(val).strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        return False


def apply_proxy_from_env() -> str:
    """Apply proxy settings from environment to both scrapers.

    Env vars:
    - TOR_PROXY / PROXY_URL: e.g., socks5h://127.0.0.1:9050
    - USE_ENV_PROXIES: if truthy, allow requests to use HTTP(S)_PROXY from env
    Returns a short status string for logging.
    """
    raw_proxy = os.getenv("TOR_PROXY") or os.getenv("PROXY_URL")
    raw_use_env = os.getenv("USE_ENV_PROXIES")
    use_env = _env_truthy(raw_use_env) if raw_use_env is not None else None

    # 4chan scraper
    try:
        if raw_proxy is not None and hasattr(sc, "PROXY_URL"):
            sc.PROXY_URL = raw_proxy
        if use_env is not None and hasattr(sc, "USE_ENV_PROXIES"):
            sc.USE_ENV_PROXIES = bool(use_env)
        if hasattr(sc, "apply_session_config"):
            sc.apply_session_config()
    except Exception:
        pass

    # Reddit scraper (optional)
    try:
        if rdt is not None:
            if raw_proxy is not None and hasattr(rdt, "PROXY_URL"):
                rdt.PROXY_URL = raw_proxy
            if use_env is not None and hasattr(rdt, "USE_ENV_PROXIES"):
                rdt.USE_ENV_PROXIES = bool(use_env)
            if hasattr(rdt, "apply_session_config"):
                rdt.apply_session_config()
    except Exception:
        pass

    # StackExchange scraper (optional)
    try:
        if sx is not None:
            if raw_proxy is not None and hasattr(sx, "PROXY_URL"):
                sx.PROXY_URL = raw_proxy
            if use_env is not None and hasattr(sx, "USE_ENV_PROXIES"):
                sx.USE_ENV_PROXIES = bool(use_env)
            if hasattr(sx, "apply_session_config"):
                sx.apply_session_config()
    except Exception:
        pass

    # Determine effective configuration for logging
    if use_env is True:
        return "Proxy: using environment proxies (USE_ENV_PROXIES=on)"
    if raw_proxy:
        return f"Proxy: routing via {raw_proxy}"

    # No env overrides provided; report module defaults
    try:
        eff_env = bool(getattr(sc, "USE_ENV_PROXIES", False)) or bool(getattr(rdt, "USE_ENV_PROXIES", False) if rdt is not None else False)
    except Exception:
        eff_env = False
    if eff_env:
        return "Proxy: using environment proxies (module default)"

    eff_proxy = None
    try:
        eff_proxy = getattr(rdt, "PROXY_URL", None) if rdt is not None else None
    except Exception:
        eff_proxy = None
    if not eff_proxy:
        try:
            eff_proxy = getattr(sc, "PROXY_URL", None)
        except Exception:
            eff_proxy = None
    if eff_proxy:
        return f"Proxy: routing via {eff_proxy} (module default)"
    return "Proxy: disabled (no proxy configured)"


def apply_proxy_from_ui(enabled: bool, proxy_url: Optional[str], use_env: bool) -> str:
    """Apply proxy settings based on UI controls.

    Priority:
    - If not enabled: disable all proxies for both scrapers.
    - If enabled and use_env: allow requests to use environment proxies.
    - If enabled and not use_env: route via explicit proxy_url (if provided), else disable.
    Returns a status string for logging.
    """
    try:
        # 4chan
        if hasattr(sc, "USE_ENV_PROXIES"):
            sc.USE_ENV_PROXIES = bool(use_env) if enabled else False
        if hasattr(sc, "PROXY_URL"):
            sc.PROXY_URL = (proxy_url or None) if (enabled and not use_env) else None
        if hasattr(sc, "apply_session_config"):
            sc.apply_session_config()
    except Exception:
        pass

    try:
        # Reddit (optional)
        if rdt is not None:
            if hasattr(rdt, "USE_ENV_PROXIES"):
                rdt.USE_ENV_PROXIES = bool(use_env) if enabled else False
            if hasattr(rdt, "PROXY_URL"):
                rdt.PROXY_URL = (proxy_url or None) if (enabled and not use_env) else None
            if hasattr(rdt, "apply_session_config"):
                rdt.apply_session_config()
    except Exception:
        pass

    try:
        # StackExchange
        if hasattr(sx, "USE_ENV_PROXIES"):
            sx.USE_ENV_PROXIES = bool(use_env) if enabled else False
        if hasattr(sx, "PROXY_URL"):
            sx.PROXY_URL = (proxy_url or None) if (enabled and not use_env) else None
        if hasattr(sx, "apply_session_config"):
            sx.apply_session_config()
    except Exception:
        pass

    if not enabled:
        return "Proxy: disabled via UI"
    if use_env:
        return "Proxy: using environment proxies (UI)"
    if proxy_url:
        return f"Proxy: routing via {proxy_url} (UI)"
    return "Proxy: disabled (no proxy URL provided)"

__all__ = [
    "_env_truthy",
    "apply_proxy_from_env",
    "apply_proxy_from_ui",
]
