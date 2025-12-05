from types import SimpleNamespace

import helpers.proxy as px


def test_env_truthy_handles_common_values():
    assert px._env_truthy("1") is True
    assert px._env_truthy("true") is True
    assert px._env_truthy("Yes") is True
    assert px._env_truthy("on") is True
    assert px._env_truthy("0") is False
    assert px._env_truthy("false") is False
    assert px._env_truthy("") is False
    assert px._env_truthy(None) is False


def _make_dummy_scraper():
    ns = SimpleNamespace()
    ns.PROXY_URL = None
    ns.USE_ENV_PROXIES = False

    def apply_session_config():
        ns.applied = True

    ns.apply_session_config = apply_session_config
    return ns


def test_apply_proxy_from_env_sets_proxy_and_env(monkeypatch):
    dummy_sc = _make_dummy_scraper()
    dummy_rdt = _make_dummy_scraper()
    dummy_sx = _make_dummy_scraper()
    monkeypatch.setattr(px, "sc", dummy_sc)
    monkeypatch.setattr(px, "rdt", dummy_rdt)
    monkeypatch.setattr(px, "sx", dummy_sx)
    monkeypatch.setenv("TOR_PROXY", "socks5h://127.0.0.1:9050")
    monkeypatch.setenv("USE_ENV_PROXIES", "1")

    status = px.apply_proxy_from_env()

    assert "proxy:" in status.lower()
    assert dummy_sc.PROXY_URL == "socks5h://127.0.0.1:9050"
    assert dummy_sc.USE_ENV_PROXIES is True
    assert dummy_rdt.PROXY_URL == "socks5h://127.0.0.1:9050"
    assert dummy_sx.PROXY_URL == "socks5h://127.0.0.1:9050"


def test_apply_proxy_from_ui_disabled_clears_all(monkeypatch):
    dummy_sc = _make_dummy_scraper()
    dummy_rdt = _make_dummy_scraper()
    dummy_sx = _make_dummy_scraper()
    monkeypatch.setattr(px, "sc", dummy_sc)
    monkeypatch.setattr(px, "rdt", dummy_rdt)
    monkeypatch.setattr(px, "sx", dummy_sx)

    status = px.apply_proxy_from_ui(False, "http://proxy", True)

    assert "disabled" in status.lower()
    assert dummy_sc.USE_ENV_PROXIES is False
    assert dummy_sc.PROXY_URL is None
