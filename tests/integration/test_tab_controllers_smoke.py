import flet as ft
import pytest

from helpers.ui import section_title
from ui.tabs.build_controller import build_build_tab_with_logic
from ui.tabs.merge_controller import build_merge_tab_with_logic
from ui.tabs.analysis_controller import build_analysis_tab_with_logic
from ui.tabs.scrape_controller import build_scrape_tab_with_logic
from ui.tabs.training_controller import build_training_tab_with_logic
from ui.tabs.inference_controller import build_inference_tab_with_logic


class DummyPage:
    """Minimal stand-in for ft.Page used in controller smoke tests.

    It implements just enough of the interface that controllers expect:
    - controls / overlay collections
    - snack_bar / dialog attributes
    - update() / update_async()
    - open() (for dialogs and snack bars)
    - run_task() (records scheduled coroutines but does not execute them)
    """

    def __init__(self):
        self.controls = []
        self.overlay = []
        self.snack_bar = None
        self.dialog = None
        self.floating_action_button = None
        self.updated = False
        self.scheduled_tasks = []

    def add(self, *controls):
        self.controls.extend(controls)

    def update(self):  # called by many controllers
        self.updated = True

    async def update_async(self):  # used by safe_update in some paths
        self.update()

    def open(self, ctl):  # used for dialogs and snack bars
        self.snack_bar = ctl
        self.dialog = getattr(ctl, "dialog", None)

    def run_task(self, coro):  # controllers prefer page.run_task for async handlers
        # Accept either an async function or a coroutine object and simply record it.
        self.scheduled_tasks.append(coro)
        return None


def _mk_help_handler(_msg: str):
    """Return a no-op help handler; controllers wire this into help icons."""

    def handler(_=None):  # noqa: ARG001
        return None

    return handler


@pytest.mark.integration
def test_build_scrape_tab_smoke(offline_mode_sw):
    page = DummyPage()
    proxy_enable_cb = ft.Checkbox()
    use_env_cb = ft.Checkbox()
    proxy_url_tf = ft.TextField()

    tab = build_scrape_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
        offline_mode_sw=offline_mode_sw,
        proxy_enable_cb=proxy_enable_cb,
        use_env_cb=use_env_cb,
        proxy_url_tf=proxy_url_tf,
    )

    assert isinstance(tab, ft.Control)
    assert isinstance(getattr(offline_mode_sw, "data", None), dict)
    assert "apply_offline_mode_to_sources" in offline_mode_sw.data


@pytest.mark.integration
def test_build_build_tab_smoke(offline_mode_sw):
    page = DummyPage()
    ollama_enable_cb = ft.Checkbox()
    ollama_models_dd = ft.Dropdown()

    tab = build_build_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
        offline_mode_sw=offline_mode_sw,
        _hf_cfg={},
        ollama_enable_cb=ollama_enable_cb,
        ollama_models_dd=ollama_models_dd,
    )

    assert isinstance(tab, ft.Control)
    assert isinstance(getattr(offline_mode_sw, "data", None), dict)
    assert "build_tab_offline" in offline_mode_sw.data


@pytest.mark.integration
def test_build_merge_tab_smoke(offline_mode_sw):
    page = DummyPage()

    tab = build_merge_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
        offline_mode_sw=offline_mode_sw,
    )

    assert isinstance(tab, ft.Control)
    assert isinstance(getattr(offline_mode_sw, "data", None), dict)
    assert "merge_tab_offline" in offline_mode_sw.data


@pytest.mark.integration
def test_build_analysis_tab_smoke(offline_mode_sw):
    page = DummyPage()

    tab = build_analysis_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
        offline_mode_sw=offline_mode_sw,
    )

    assert isinstance(tab, ft.Control)
    assert isinstance(getattr(offline_mode_sw, "data", None), dict)
    assert "analysis_tab_offline" in offline_mode_sw.data


@pytest.mark.integration
def test_build_training_and_inference_tabs_smoke(offline_mode_sw):
    page = DummyPage()

    hf_token_tf = ft.TextField()
    proxy_enable_cb = ft.Checkbox()
    use_env_cb = ft.Checkbox()
    proxy_url_tf = ft.TextField()

    training_tab, train_state = build_training_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
        offline_mode_sw=offline_mode_sw,
        _hf_cfg={},
        _runpod_cfg={},
        hf_token_tf=hf_token_tf,
        proxy_enable_cb=proxy_enable_cb,
        use_env_cb=use_env_cb,
        proxy_url_tf=proxy_url_tf,
    )

    assert isinstance(training_tab, ft.Control)
    assert isinstance(train_state, dict)
    assert "running" in train_state
    assert isinstance(getattr(offline_mode_sw, "data", None), dict)
    assert "training_tab_offline" in offline_mode_sw.data

    # Inference tab uses the shared train_state (e.g., for latest local run info).
    inference_tab = build_inference_tab_with_logic(
        page,
        section_title=section_title,
        _mk_help_handler=_mk_help_handler,
        train_state=train_state,
    )

    assert isinstance(inference_tab, ft.Control)
