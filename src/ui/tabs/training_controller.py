"""Training tab controller for FineFoundry.

This module will encapsulate all Training tab controls, handlers, and
state wiring, keeping `src/main.py` slimmer. Layout composition remains
in `tab_training.py` and the per-section builders under
`ui/tabs/training/sections/`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import asyncio
import json
import os
import re
import time
from datetime import datetime

import flet as ft

from helpers.common import safe_update
from helpers.logging_config import get_logger
from helpers.theme import ACCENT_COLOR, BORDER_BASE, COLORS, ICONS, REFRESH_ICON
from helpers.ui import WITH_OPACITY, make_empty_placeholder, build_offline_banner, offline_reason_text
from helpers.local_inference import generate_text as local_infer_generate_text_helper
from helpers.training import (
    build_hp_from_controls as build_hp_from_controls_helper,
    run_local_training as run_local_training_helper,
    stop_local_training as stop_local_training_helper,
)
from helpers.training_pod import (
    confirm_teardown_all as confirm_teardown_all_helper,
    confirm_teardown_selected as confirm_teardown_selected_helper,
    copy_ssh_command as copy_ssh_command_helper,
    do_teardown as do_teardown_helper,
    ensure_infrastructure as ensure_infrastructure_helper,
    open_runpod as open_runpod_helper,
    open_web_terminal as open_web_terminal_helper,
    refresh_expert_gpus as refresh_expert_gpus_helper,
    refresh_teardown_ui as refresh_teardown_ui_helper,
    restart_pod_container as restart_pod_container_helper,
    run_pod_training as run_pod_training_helper,
)
from helpers.training_config import (
    get_last_used_config_name as get_last_used_config_name_helper,
    list_saved_configs as list_saved_configs_helper,
    read_json_file as read_json_file_helper,
    saved_configs_dir as saved_configs_dir_helper,
    set_last_used_config_name as set_last_used_config_name_helper,
    validate_config as validate_config_helper,
)
from helpers.local_specs import (
    gather_local_specs as gather_local_specs_helper,
    refresh_local_gpus as refresh_local_gpus_helper,
)
from ui.tabs.tab_training import build_training_tab
from ui.tabs.training.sections.config_section import build_config_section
from ui.tabs.training.sections.dataset_section import build_dataset_section
from ui.tabs.training.sections.local_specs_section import build_local_specs_container
from ui.tabs.training.sections.logs_section import build_pod_logs_section
from ui.tabs.training.sections.pod_content import build_pod_content_container
from ui.tabs.training.sections.rp_infra_section import build_rp_infra_panel
from ui.tabs.training.sections.teardown_section import build_teardown_section
from ui.tabs.training.sections.train_params_section import build_train_params_section


# Runpod modules (pod lifecycle and infra helpers) — mirrored from main.py
try:
    from runpod import runpod_pod as rp_pod
    from runpod import ensure_infra as rp_infra
except Exception:  # pragma: no cover - defensive
    import sys as __sys2

    __sys2.path.append(os.path.dirname(__file__))
    from runpod import runpod_pod as rp_pod

    try:
        from runpod import ensure_infra as rp_infra
    except Exception:  # pragma: no cover - defensive
        rp_infra = None


logger = get_logger(__name__)


def _find_first_file(root_dir: str, filename: str) -> str:
    try:
        if not root_dir:
            return ""
        direct = os.path.join(root_dir, filename)
        if os.path.isfile(direct):
            return direct
        for base, _dirs, files in os.walk(root_dir):
            if filename in files:
                return os.path.join(base, filename)
    except Exception:
        return ""
    return ""


def _load_trainer_metrics(
    output_dir: str,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Dict[str, Any]]:
    train_pts: List[Tuple[float, float]] = []
    eval_pts: List[Tuple[float, float]] = []
    stats: Dict[str, Any] = {}
    try:
        p = _find_first_file(output_dir, "trainer_state.json")
        if not p:
            return train_pts, eval_pts, stats
        with open(p, "r", encoding="utf-8") as f:
            state = json.load(f)
        hist = state.get("log_history") or []
        if not isinstance(hist, list):
            return train_pts, eval_pts, stats
        for row in hist:
            if not isinstance(row, dict):
                continue
            try:
                step = row.get("step")
                if step is None:
                    continue
                x = float(step)
            except Exception:
                continue
            if "loss" in row:
                try:
                    y = float(row.get("loss"))
                    if y == y:
                        train_pts.append((x, y))
                except Exception:
                    pass
            if "eval_loss" in row:
                try:
                    y = float(row.get("eval_loss"))
                    if y == y:
                        eval_pts.append((x, y))
                except Exception:
                    pass
        train_pts = sorted(set(train_pts), key=lambda t: t[0])
        eval_pts = sorted(set(eval_pts), key=lambda t: t[0])
        if train_pts:
            stats["final_train_loss"] = train_pts[-1][1]
            stats["min_train_loss"] = min(y for _x, y in train_pts)
        if eval_pts:
            stats["best_eval_loss"] = min(y for _x, y in eval_pts)
            stats["final_eval_loss"] = eval_pts[-1][1]
        if train_pts or eval_pts:
            stats["max_step"] = max([train_pts[-1][0] if train_pts else 0, eval_pts[-1][0] if eval_pts else 0])
    except Exception:
        pass
    return train_pts, eval_pts, stats


def _schedule_task(page: ft.Page, coro):
    """Robust scheduler helper for async tasks.

    Mirrors the pattern used in `main.py`, preferring `page.run_task` when
    available and falling back to `asyncio.create_task`.
    """
    try:
        if hasattr(page, "run_task") and callable(getattr(page, "run_task")):
            return page.run_task(coro)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        import asyncio

        return asyncio.create_task(coro())
    except Exception:  # pragma: no cover - defensive
        return None


def build_training_tab_with_logic(
    page: ft.Page,
    *,
    section_title,
    _mk_help_handler,
    _hf_cfg: Dict[str, Any],
    _runpod_cfg: Dict[str, Any],
    hf_token_tf: ft.TextField,
    proxy_enable_cb: ft.Control,
    use_env_cb: ft.Control,
    proxy_url_tf: ft.TextField,
    offline_mode_sw: ft.Switch,
) -> Tuple[ft.Control, Dict[str, Any]]:
    """Build the Training tab UI and attach all related handlers.

    This constructs all Training controls (dataset source, hyperparameters,
    Runpod and local targets, configs, teardown, logs, and Quick Local
    Inference), wires their behavior, and returns the composed Training tab
    plus the shared ``train_state`` dictionary.
    """

    # Shared training state; extended later with per-target sub-dicts.
    train_state: Dict[str, Any] = {
        "running": False,
        "pod_id": None,
        "infra": None,
        "api_key": "",
        "loaded_config": None,
        "suppress_skill_defaults": False,
    }

    # Dataset source
    train_source = ft.Dropdown(
        label="Dataset source",
        options=[
            ft.dropdown.Option("Database"),
            ft.dropdown.Option("Hugging Face"),
        ],
        value="Database",
        width=180,
    )
    train_hf_repo = ft.TextField(
        label="Dataset repo (e.g., username/dataset)",
        width=360,
        visible=False,  # Hidden by default since Database is default source
    )
    train_hf_split = ft.Dropdown(
        label="Split",
        options=[
            ft.dropdown.Option("train"),
            ft.dropdown.Option("validation"),
            ft.dropdown.Option("test"),
        ],
        value="train",
        width=140,
        visible=False,  # Hidden by default
    )
    train_hf_config = ft.TextField(label="Config (optional)", width=180, visible=False)  # Hidden by default
    # JSON path kept for internal use only (not user-facing)
    train_json_path = ft.TextField(label="JSON path", width=300, visible=False)

    # Database session selector (visible by default since Database is default source)
    train_db_session_dd = ft.Dropdown(
        label="Scrape session",
        options=[],
        width=400,
        visible=True,
        tooltip="Select a scrape session from the database",
    )
    train_db_refresh_btn = ft.IconButton(
        icon=ft.Icons.REFRESH,
        tooltip="Refresh sessions",
        visible=True,
    )
    train_db_pair_count = ft.Text("", visible=True)

    # File picker for JSON dataset
    def _on_json_file_picked(e: ft.FilePickerResultEvent):
        if e.files and len(e.files) > 0:
            train_json_path.value = e.files[0].path
            try:
                page.update()
            except Exception:
                pass

    json_file_picker = ft.FilePicker(on_result=_on_json_file_picked)
    page.overlay.append(json_file_picker)

    def _pick_json_file(_=None):
        json_file_picker.pick_files(
            dialog_title="Select JSON Dataset",
            allowed_extensions=["json", "jsonl"],
            allow_multiple=False,
        )

    train_json_browse_btn = ft.IconButton(
        icon=ft.Icons.FOLDER_OPEN,
        tooltip="Browse for JSON file",
        visible=False,
        on_click=_pick_json_file,
    )

    def _refresh_db_sessions(_=None):
        """Refresh the database session dropdown with available scrape sessions."""
        try:
            from db.scraped_data import list_scrape_sessions

            sessions = list_scrape_sessions(limit=50)
            options = []
            for s in sessions:
                # Prefer custom name if set, otherwise fallback to auto-generated label
                if s.get("name"):
                    label = f"{s['name']} ({s['pair_count']} pairs)"
                elif s.get("source_details"):
                    label = f"{s['source']}: {s['source_details'][:30]} - {s['pair_count']} pairs"
                else:
                    label = f"{s['source']} - {s['pair_count']} pairs ({s['created_at'][:10]})"
                options.append(ft.dropdown.Option(key=str(s["id"]), text=label))
            train_db_session_dd.options = options
            if options and not train_db_session_dd.value:
                train_db_session_dd.value = options[0].key
                _update_db_pair_count(None)
        except Exception as e:
            train_db_session_dd.options = [ft.dropdown.Option(key="", text=f"Error: {e}")]
        try:
            page.update()
        except Exception:
            pass

    def _update_db_pair_count(_=None):
        """Update the pair count display when session changes."""
        try:
            session_id = train_db_session_dd.value
            if session_id:
                from db.scraped_data import get_scrape_session

                session = get_scrape_session(int(session_id))
                if session:
                    train_db_pair_count.value = f"📊 {session['pair_count']} pairs available"
                else:
                    train_db_pair_count.value = ""
            else:
                train_db_pair_count.value = ""
        except Exception:
            train_db_pair_count.value = ""
        try:
            page.update()
        except Exception:
            pass

    train_db_session_dd.on_change = _update_db_pair_count
    train_db_refresh_btn.on_click = _refresh_db_sessions

    def _update_train_source(_=None):
        src = getattr(train_source, "value", "Database") or "Database"
        is_hf = src == "Hugging Face"
        is_db = src == "Database"
        # HuggingFace controls
        train_hf_repo.visible = is_hf
        train_hf_split.visible = is_hf
        train_hf_config.visible = is_hf
        # Database controls
        train_db_session_dd.visible = is_db
        train_db_refresh_btn.visible = is_db
        train_db_pair_count.visible = is_db
        if is_db:
            _refresh_db_sessions()
        # JSON controls always hidden (internal use only)
        train_json_path.visible = False
        train_json_browse_btn.visible = False
        try:
            page.update()
        except Exception:  # pragma: no cover - UI best-effort
            pass

    # Training parameters
    skill_level = ft.Dropdown(
        label="Skill level",
        options=[ft.dropdown.Option("Beginner"), ft.dropdown.Option("Expert")],
        value="Beginner",
        width=160,
    )
    # Beginner presets use stable keys ("fastest" / "cheapest"); labels change
    # depending on Training target (Runpod vs local) via
    # _update_beginner_mode_labels_for_target in the full wiring.
    beginner_mode_dd = ft.Dropdown(
        label="Beginner preset",
        options=[
            ft.dropdown.Option(text="Fastest (Runpod)", key="fastest"),
            ft.dropdown.Option(text="Cheapest (Runpod)", key="cheapest"),
            ft.dropdown.Option(text="Simple custom", key="simple"),
        ],
        value="fastest",
        width=220,
        visible=True,
        tooltip=(
            "Beginner presets. For Runpod, Fastest favors throughput and Cheapest favors lower cost. "
            "For local training, labels change to Quick local test / Longer local run."
        ),
    )

    simple_duration_dd = ft.Dropdown(
        label="Training duration",
        options=[
            ft.dropdown.Option(key="very_short", text="Very short (smoke test)"),
            ft.dropdown.Option(key="short", text="Short"),
            ft.dropdown.Option(key="medium", text="Medium"),
            ft.dropdown.Option(key="long", text="Long"),
        ],
        value="short",
        width=240,
        tooltip="How long to train. This maps to Max steps (safe, capped).",
    )
    simple_memory_dd = ft.Dropdown(
        label="Memory / stability",
        options=[
            ft.dropdown.Option(key="safe", text="Safe (avoid OOM)"),
            ft.dropdown.Option(key="normal", text="Normal"),
            ft.dropdown.Option(key="aggressive", text="Aggressive (may OOM)"),
        ],
        value="safe",
        width=220,
        tooltip="Controls batch size and gradient accumulation based on detected GPU VRAM.",
    )
    simple_quality_dd = ft.Dropdown(
        label="Speed vs quality",
        options=[
            ft.dropdown.Option(key="speed", text="Speed"),
            ft.dropdown.Option(key="balanced", text="Balanced"),
            ft.dropdown.Option(key="quality", text="Quality"),
        ],
        value="balanced",
        width=200,
        tooltip="Speed favors packing + a slightly higher learning rate; Quality is slightly more conservative.",
    )
    simple_summary_txt = ft.Text("", size=11, color=WITH_OPACITY(0.7, BORDER_BASE))

    beginner_simple_custom_panel = ft.Container(
        content=ft.Column(
            [
                ft.Text("Simple custom", size=12, weight=ft.FontWeight.W_600),
                ft.Row([simple_duration_dd, simple_memory_dd, simple_quality_dd], wrap=True),
                simple_summary_txt,
            ],
            spacing=6,
        ),
        padding=ft.padding.only(top=4, bottom=4),
        visible=False,
    )

    # Expert-mode GPU picker (hidden by default)
    expert_gpu_dd = ft.Dropdown(
        label="GPU (Expert)",
        options=[ft.dropdown.Option("AUTO")],
        value="AUTO",
        width=260,
        visible=False,
        tooltip=(
            "Pick a GPU type available in the selected datacenter. 'AUTO' will pick the best available secure GPU."
        ),
    )
    expert_spot_cb = ft.Checkbox(
        label="Use Spot (interruptible)",
        value=False,
        visible=False,
        tooltip="When enabled and available, a spot/interruptible pod is used.",
    )
    expert_gpu_refresh_btn = ft.IconButton(
        icon=getattr(
            ICONS,
            "REFRESH",
            getattr(
                ICONS,
                "AUTORENEW",
                getattr(ICONS, "UPDATE", getattr(ICONS, "SYNC", getattr(ICONS, "CACHED", ICONS.REFRESH))),
            ),
        ),
        tooltip="Refresh available GPUs from Runpod",
        visible=False,
    )
    expert_gpu_busy = ft.ProgressRing(width=18, height=18, value=None, visible=False)
    # Map gpu_id -> availability flags to drive spot toggle enabling
    expert_gpu_avail: Dict[str, Any] = {}

    def _update_expert_spot_enabled(_=None):
        try:
            gid = expert_gpu_dd.value or "AUTO"
            flags = expert_gpu_avail.get(gid) or {}
            sec_ok = bool(flags.get("secureAvailable"))
            spot_ok = bool(flags.get("spotAvailable"))
            # Only enable checkbox if any mode is available; constrain value when not supported
            expert_spot_cb.disabled = not (spot_ok or sec_ok)
            if not spot_ok and bool(getattr(expert_spot_cb, "value", False)):
                expert_spot_cb.value = False
            expert_spot_cb.tooltip = f"Spot available: {spot_ok} • Secure available: {sec_ok}"
        except Exception:  # pragma: no cover - defensive UI
            pass
        try:
            page.update()
        except Exception:
            pass

    expert_gpu_dd.on_change = _update_expert_spot_enabled

    base_model = ft.Dropdown(
        label="Base model",
        options=[
            ft.dropdown.Option("unsloth/Meta-Llama-3.1-8B-bnb-4bit"),
            ft.dropdown.Option("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"),
            ft.dropdown.Option("unsloth/Meta-Llama-3.1-70B-bnb-4bit"),
            ft.dropdown.Option("unsloth/Meta-Llama-3.1-405B-bnb-4bit"),
            ft.dropdown.Option("unsloth/Mistral-Nemo-Base-2407-bnb-4bit"),
            ft.dropdown.Option("unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit"),
            ft.dropdown.Option("unsloth/mistral-7b-v0.3-bnb-4bit"),
            ft.dropdown.Option("unsloth/mistral-7b-instruct-v0.3-bnb-4bit"),
            ft.dropdown.Option("unsloth/Phi-3.5-mini-instruct"),
            ft.dropdown.Option("unsloth/Phi-3-medium-4k-instruct"),
            ft.dropdown.Option("unsloth/gemma-2-9b-bnb-4bit"),
            ft.dropdown.Option("unsloth/gemma-2-27b-bnb-4bit"),
        ],
        value="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        width=320,
    )
    epochs_tf = ft.TextField(
        label="Epochs", value="1", width=120, tooltip="Recommended: 1-3 epochs. More can cause overfitting."
    )
    lr_tf = ft.TextField(label="Learning rate", value="2e-4", width=160)
    batch_tf = ft.TextField(label="Per-device batch size", value="2", width=200)
    grad_acc_tf = ft.TextField(label="Grad accum steps", value="4", width=180)
    max_steps_tf = ft.TextField(
        label="Max steps", value="-1", width=180, tooltip="-1 = train full epochs. Set positive value to limit steps."
    )
    use_lora_cb = ft.Checkbox(label="Use LoRA", value=True)
    lora_r_dd = ft.Dropdown(
        label="LoRA Rank",
        value="16",
        options=[ft.dropdown.Option(x) for x in ["8", "16", "32", "64", "128"]],
        width=100,
        tooltip="Higher rank = more capacity but slower. 16-32 recommended.",
    )
    lora_alpha_tf = ft.TextField(
        label="LoRA Alpha",
        value="",
        width=100,
        hint_text="2×r",
        tooltip="Scaling factor. Default: 2×rank for aggressive learning. Leave empty for auto.",
    )
    lora_dropout_tf = ft.TextField(
        label="LoRA Dropout",
        value="0",
        width=100,
        tooltip="Regularization. 0 = Unsloth optimized, 0.05-0.1 if overfitting.",
    )
    use_rslora_cb = ft.Checkbox(
        label="RSLoRA",
        value=False,
        tooltip="Rank-stabilized LoRA. Better for high ranks (64+).",
    )
    lora_hint_caption = ft.Text(
        "LoRA rank/alpha/dropout controls are available in Expert mode.",
        size=11,
        color=WITH_OPACITY(0.7, BORDER_BASE),
    )
    out_dir_tf = ft.TextField(label="Output dir", value="/data/outputs/runpod_run", width=260)

    # New HP toggles/fields
    packing_cb = ft.Checkbox(
        label="Packing",
        value=True,
        tooltip="Pack multiple samples into a sequence for higher utilization (if trainer supports).",
    )
    auto_resume_cb = ft.Checkbox(
        label="Auto-resume",
        value=True,
        tooltip="Resume from latest checkpoint if container restarts.",
    )
    push_cb = ft.Checkbox(
        label="Push to HF Hub",
        value=False,
        tooltip=(
            "When enabled, the trainer will attempt to push the final model/adapters to the "
            "Hugging Face Hub. Requires a valid HF token with write access."
        ),
    )
    hf_repo_id_tf = ft.TextField(
        label="HF repo id (for push)",
        value="",
        width=280,
        hint_text="username/model-name",
        tooltip=(
            "Model repository on Hugging Face to push to (e.g., username/my-lora-model). "
            "You must own the repo or have write access and be authenticated."
        ),
    )
    resume_from_tf = ft.TextField(
        label="Resume from (path)",
        value="",
        width=320,
        hint_text="/data/outputs/runpod_run/checkpoint-500",
        tooltip=(
            "Optional explicit checkpoint directory to resume from, inside the mounted volume "
            "(e.g., /data/outputs/runpod_run/checkpoint-500)."
        ),
    )

    # Info icons next to toggles/fields
    _info_icon = getattr(ICONS, "INFO_OUTLINE", getattr(ICONS, "INFO", getattr(ICONS, "HELP_OUTLINE", None)))

    packing_info = ft.IconButton(
        icon=_info_icon,
        tooltip=(
            "Packing combines multiple short samples into a fixed-length sequence to better utilize tokens "
            "(only if the training script supports it)."
        ),
        on_click=_mk_help_handler(
            "Packing: When enabled, the trainer may pack several shorter samples into a fixed-length "
            "training sequence to improve GPU utilization and throughput.\n\nWhen to use: If your "
            "training script supports packing and you have many short samples. If unsupported, leave it off."
        ),
    )
    auto_resume_info = ft.IconButton(
        icon=_info_icon,
        tooltip="Try to continue from the latest checkpoint in Output dir if the container restarts.",
        on_click=_mk_help_handler(
            "Auto-resume: On container restarts, the trainer looks for the latest checkpoint in your Output dir "
            "and continues training from it.\n\nRequirements: Keep Output dir on the persistent Runpod Network "
            "Volume and reuse the same Output dir for the same run."
        ),
    )
    push_info = ft.IconButton(
        icon=_info_icon,
        tooltip="Push final model/adapters to the Hugging Face Hub at the end of training.",
        on_click=_mk_help_handler(
            "Push to HF Hub: If enabled, the trainer will attempt to upload the resulting model (or LoRA adapters) "
            "to a Hugging Face model repository.\n\nProvide: • A valid HF token with write scope "
            "(Settings → Hugging Face Access) • The repo id as username/model-name.\nNote: Create the repo on "
            "the Hub first to ensure permissions."
        ),
    )
    hf_repo_info = ft.IconButton(
        icon=_info_icon,
        tooltip="Hugging Face model repo id, e.g., username/my-lora-model.",
        on_click=_mk_help_handler(
            "HF repo id (for push): The target model repository on Hugging Face to push your trained "
            "weights/adapters to.\n\nFormat: username/model-name (e.g., sbussiso/my-lora-phi3). You must own the "
            "repo or have collaborator write access. Authenticate via Settings → Hugging Face Access "
            "or HF_TOKEN env."
        ),
    )
    resume_info = ft.IconButton(
        icon=_info_icon,
        tooltip="Explicit checkpoint path inside /data, e.g., /data/outputs/runpod_run/checkpoint-500",
        on_click=_mk_help_handler(
            "Resume from (path): Force the trainer to resume from a specific checkpoint directory.\n\nExample: "
            "/data/outputs/runpod_run/checkpoint-500 or /data/outputs/runpod_run/last. Must exist on the "
            "mounted volume. Leave blank to let Auto-resume find the latest checkpoint automatically (if supported)."
        ),
    )

    # Group each control with its info icon
    packing_row = ft.Row([packing_cb, packing_info], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)
    auto_resume_row = ft.Row(
        [auto_resume_cb, auto_resume_info], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER
    )
    push_row = ft.Column(
        [
            ft.Row([push_cb, push_info], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Text(
                "HF repo field appears once enabled.",
                size=11,
                color=WITH_OPACITY(0.7, BORDER_BASE),
            ),
        ],
        spacing=2,
    )
    hf_repo_row = ft.Row(
        [hf_repo_id_tf, hf_repo_info],
        spacing=6,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        visible=False,
    )
    resume_from_row = ft.Row([resume_from_tf, resume_info], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    try:
        push_cb.disabled = True
        push_cb.value = False
    except Exception:
        pass
    try:
        push_row.visible = False
    except Exception:
        pass
    try:
        hf_repo_row.visible = False
    except Exception:
        pass

    def _update_hf_repo_visibility(_=None):
        """Show HF repo id only when 'Push to HF Hub' is active."""
        try:
            is_enabled = bool(getattr(push_cb, "value", False)) and not bool(getattr(push_cb, "disabled", False))
        except Exception:
            is_enabled = False
        try:
            hf_repo_row.visible = is_enabled
        except Exception:
            pass
        try:
            page.update()
        except Exception:
            pass

    try:
        push_cb.on_change = _update_hf_repo_visibility
    except Exception:
        pass

    # Initialize HF repo visibility based on current push state
    try:
        _update_hf_repo_visibility()
    except Exception:
        pass

    # Advanced parameters (Expert mode)
    warmup_steps_tf = ft.TextField(label="Warmup steps", value="10", width=140)
    weight_decay_tf = ft.TextField(label="Weight decay", value="0.01", width=140)
    lr_sched_dd = ft.Dropdown(
        label="LR scheduler",
        options=[
            ft.dropdown.Option("linear"),
            ft.dropdown.Option("cosine"),
            ft.dropdown.Option("constant"),
        ],
        value="cosine",
        width=160,
        tooltip="Cosine often gives better convergence than linear.",
    )
    optim_dd = ft.Dropdown(
        label="Optimizer",
        options=[ft.dropdown.Option("adamw_8bit"), ft.dropdown.Option("adamw_torch")],
        value="adamw_8bit",
        width=180,
    )
    logging_steps_tf = ft.TextField(label="Logging steps", value="25", width=160)
    logging_first_step_cb = ft.Checkbox(label="Log first step", value=True)
    disable_tqdm_cb = ft.Checkbox(label="Disable tqdm", value=False)
    seed_tf = ft.TextField(label="Seed", value="3407", width=140)
    save_strategy_dd = ft.Dropdown(
        label="Save strategy",
        options=[
            ft.dropdown.Option("epoch"),
            ft.dropdown.Option("steps"),
            ft.dropdown.Option("no"),
        ],
        value="epoch",
        width=160,
    )
    save_total_limit_tf = ft.TextField(label="Save total limit", value="2", width=180)
    pin_memory_cb = ft.Checkbox(label="Pin dataloader memory", value=False)
    report_to_dd = ft.Dropdown(
        label="Report to",
        options=[ft.dropdown.Option("none"), ft.dropdown.Option("wandb")],
        value="none",
        width=160,
    )
    fp16_cb = ft.Checkbox(label="Use FP16", value=False, tooltip="Auto-selected by trainer based on GPU support.")
    bf16_cb = ft.Checkbox(
        label="Use BF16 (if supported)", value=False, tooltip="Auto-selected by trainer based on GPU support."
    )

    advanced_params_section = ft.Column(
        [
            ft.Row([warmup_steps_tf, weight_decay_tf, lr_sched_dd, optim_dd], wrap=True),
            ft.Row([logging_steps_tf, logging_first_step_cb, disable_tqdm_cb, seed_tf], wrap=True),
            ft.Row([save_strategy_dd, save_total_limit_tf, pin_memory_cb, report_to_dd], wrap=True),
            ft.Row([fp16_cb, bf16_cb], wrap=True),
        ],
        spacing=8,
    )

    # Progress & logs
    train_progress = ft.ProgressBar(value=0.0, width=400)
    train_prog_label = ft.Text("Progress: 0%")
    train_timeline = ft.ListView([], spacing=4, auto_scroll=True, expand=True)
    train_timeline_placeholder = make_empty_placeholder(
        "No training logs yet",
        getattr(ICONS, "SCIENCE", ICONS.PLAY_CIRCLE),
    )

    # High-level Training status banner (Runpod lifecycle)
    train_status_label = ft.Text(
        "Status: Idle",
        size=12,
        color=WITH_OPACITY(0.8, BORDER_BASE),
    )
    train_status_chip = ft.Container(
        content=train_status_label,
        padding=ft.padding.symmetric(horizontal=8, vertical=4),
        border_radius=16,
        bgcolor=getattr(COLORS, "BLUE_50", getattr(COLORS, "BLUE", "#e3f2fd")),
    )
    train_status_row = ft.Row(
        [train_status_chip],
        alignment=ft.MainAxisAlignment.START,
    )

    def _set_runpod_status(status: str) -> None:
        """Update high-level Runpod training status banner and store in train_state."""
        try:
            key = (status or "").strip().lower() or "idle"
        except Exception:
            key = "idle"
        text_map = {
            "idle": "Status: Idle",
            "starting": "Status: Starting Runpod training…",
            "running": "Status: Running on Runpod",
            "completed": "Status: Completed",
            "failed": "Status: Failed",
            "cancelled": "Status: Cancelled",
        }
        color_map = {
            "idle": getattr(COLORS, "BLUE_50", getattr(COLORS, "BLUE", "#e3f2fd")),
            "starting": getattr(COLORS, "AMBER_100", getattr(COLORS, "AMBER", "#fff8e1")),
            "running": getattr(COLORS, "LIGHT_GREEN_100", getattr(COLORS, "GREEN", "#c8e6c9")),
            "completed": getattr(COLORS, "GREEN_100", getattr(COLORS, "GREEN", "#c8e6c9")),
            "failed": getattr(COLORS, "RED_100", getattr(COLORS, "RED", "#ffcdd2")),
            "cancelled": getattr(COLORS, "GREY_200", getattr(COLORS, "GREY", "#eeeeee")),
        }
        try:
            train_status_label.value = text_map.get(key, text_map["idle"])
            train_status_chip.bgcolor = color_map.get(key, color_map["idle"])
        except Exception:
            pass
        try:
            train_state["runpod_status"] = key
        except Exception:
            pass
        try:
            page.update()
        except Exception:
            pass

    def _update_progress_from_logs(new_lines: List[str]) -> None:
        """Parse progress from recent log lines and update the progress bar and label.

        Supported patterns:
        - Percent like "12%"
        - Step/total like "Step 100/1000", "global_step 1234/5000", "Step 10 of 100"
        - Epoch progress like "Epoch 1/3" or "Epoch [1/3]"
        If no numeric progress found, show the most recent log line (truncated).
        """
        try:
            last_val = float(train_state.get("progress") or 0.0)
        except Exception:
            last_val = 0.0

        if not new_lines:
            return

        pct_found: Optional[int] = None
        for line in reversed(list(new_lines)):
            s = str(line)
            # 1) Percentage patterns
            m = re.search(r"(\d{1,3})\s?%", s)
            if m:
                try:
                    pct = int(m.group(1))
                    if 0 <= pct <= 100:
                        pct_found = pct
                        break
                except Exception:
                    pass

            # 2) Step/total patterns (Step 100/1000, global_step 12/500, Iteration 5 of 20)
            m = re.search(
                r"(?:global[_ ]?step|steps?|iter(?:ation)?|it(?:er)?)\s*[:=]?\s*(\d+)\s*(?:/|of)\s*(\d+)",
                s,
                re.IGNORECASE,
            )
            if m:
                try:
                    cur = int(m.group(1))
                    tot = int(m.group(2))
                    if tot > 0:
                        pct_found = max(0, min(100, int(cur * 100 / tot)))
                        break
                except Exception:
                    pass

            # 3) Epoch progress patterns (Epoch 1/3, Epoch [1/3])
            m = re.search(r"epoch\s*[:#]?\s*(\d+)\s*/\s*(\d+)", s, re.IGNORECASE)
            if not m:
                m = re.search(r"epoch\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]", s, re.IGNORECASE)
            if m:
                try:
                    cur = int(m.group(1))
                    tot = int(m.group(2))
                    if tot > 0:
                        # Use (cur-1)/tot as coarse progress into epochs
                        pct_found = max(0, min(100, int(((cur - 1) / tot) * 100)))
                        break
                except Exception:
                    pass

        try:
            if pct_found is not None:
                v = max(last_val, min(1.0, pct_found / 100.0))
                train_progress.value = v
                train_prog_label.value = f"Progress: {int(v * 100)}%"
                try:
                    train_state["progress"] = v
                except Exception:
                    pass
            else:
                # No numeric progress found; show latest line text
                latest = str(new_lines[-1]) if new_lines else ""
                if latest:
                    max_len = 120
                    disp = latest if len(latest) <= max_len else (latest[: max_len - 1] + "…")
                    train_prog_label.value = disp
        except Exception:
            # Never let UI updates crash the loop
            pass

    def update_train_placeholders():
        try:
            has_logs = len(getattr(train_timeline, "controls", []) or []) > 0
            train_timeline_placeholder.visible = not has_logs
        except Exception:
            pass

    cancel_train: Dict[str, Any] = {"cancelled": False}

    def _update_skill_controls(_=None):
        level = (skill_level.value or "Beginner").lower()
        is_beginner = level == "beginner"
        # Hide some tweak knobs for beginners
        for ctl in [lr_tf, batch_tf, grad_acc_tf, max_steps_tf]:
            try:
                ctl.visible = not is_beginner
            except Exception:
                pass
        # Advanced block
        try:
            advanced_params_section.visible = not is_beginner
        except Exception:
            pass
        # Checkpoints & resume: hide explicit resume path for beginners
        try:
            resume_from_row.visible = not is_beginner
        except Exception:
            pass
        # LoRA details: keep the main toggle visible, hide rank/alpha/dropout/RS-LoRA in Beginner mode
        try:
            use_lora_cb.visible = True
            lora_r_dd.visible = not is_beginner
            lora_alpha_tf.visible = not is_beginner
            lora_dropout_tf.visible = not is_beginner
            use_rslora_cb.visible = not is_beginner
            lora_hint_caption.visible = is_beginner
        except Exception:
            pass
        # Beginner target control visibility
        try:
            beginner_mode_dd.visible = is_beginner
        except Exception:
            pass
        # Simple custom panel is only meaningful for Beginner + local target + preset=simple
        try:
            tgt_sc = (train_target_dd.value or "Runpod - Pod").lower()
            is_local_sc = not tgt_sc.startswith("runpod - pod")
            mode_sc = (beginner_mode_dd.value or "fastest").strip().lower()
            beginner_simple_custom_panel.visible = bool(is_beginner and is_local_sc and mode_sc == "simple")
        except Exception:
            pass
        # Expert GPU picker visibility
        try:
            expert_gpu_dd.visible = not is_beginner
            # Spot is only meaningful for Runpod target
            tgt = (train_target_dd.value or "Runpod - Pod").lower()
            if tgt.startswith("runpod - pod"):
                expert_spot_cb.visible = not is_beginner
                expert_spot_cb.disabled = False
            else:
                expert_spot_cb.value = False
                expert_spot_cb.disabled = True
                expert_spot_cb.visible = False
            expert_gpu_refresh_btn.visible = not is_beginner
            if is_beginner:
                expert_gpu_busy.visible = False
        except Exception:
            pass
        # Local-mode beginner uses simple GPU checkbox; expert uses dropdown
        try:
            tgt2 = (train_target_dd.value or "Runpod - Pod").lower()
            if not tgt2.startswith("runpod - pod"):
                local_use_gpu_cb.visible = is_beginner
        except Exception:
            pass
        suppress = bool(train_state.get("suppress_skill_defaults"))
        # Set beginner defaults (depend on beginner mode and training target)
        if is_beginner and (not suppress):
            try:
                mode = (beginner_mode_dd.value or "Fastest").lower()
                tgt3 = (train_target_dd.value or "Runpod - Pod").lower()
                is_runpod_target = tgt3.startswith("runpod - pod")
                epochs_tf.value = epochs_tf.value or "1"
                if is_runpod_target:
                    # Existing Runpod-oriented presets
                    if mode == "fastest":
                        lr_tf.value = "2e-4"
                        batch_tf.value = "4"
                        grad_acc_tf.value = "1"
                        max_steps_tf.value = max_steps_tf.value or "200"
                    else:
                        lr_tf.value = "2e-5"
                        batch_tf.value = "2"
                        grad_acc_tf.value = "4"
                        max_steps_tf.value = "200"
                else:
                    # Local presets
                    if mode == "fastest":
                        # Quick local test: short run, small-ish batch for a fast sanity check
                        lr_tf.value = "2e-4"
                        batch_tf.value = "2"
                        grad_acc_tf.value = "1"
                        max_steps_tf.value = max_steps_tf.value or "50"
                    elif mode == "simple":
                        # Simple custom: map a few safe knobs onto the hidden HP fields
                        try:
                            data = gather_local_specs_helper()
                        except Exception:
                            data = {}
                        try:
                            gpus = list(data.get("gpus") or [])
                            max_vram = 0.0
                            for g in gpus:
                                try:
                                    v = g.get("vram_gb")
                                    if isinstance(v, (int, float)) and v is not None:
                                        max_vram = max(max_vram, float(v))
                                except Exception:
                                    pass
                        except Exception:
                            max_vram = 0.0

                        dur = (simple_duration_dd.value or "short").strip().lower()
                        mem = (simple_memory_dd.value or "safe").strip().lower()
                        qual = (simple_quality_dd.value or "balanced").strip().lower()

                        # Duration -> max_steps
                        steps_map = {
                            "very_short": "50",
                            "short": "200",
                            "medium": "600",
                            "long": "1200",
                        }
                        max_steps_tf.value = steps_map.get(dur, "200")

                        # Memory/stability -> (batch, grad_acc) based on VRAM tier
                        # Conservative defaults if unknown / CPU-only
                        if max_vram >= 24:
                            tiers = {
                                "safe": ("2", "4"),
                                "normal": ("4", "2"),
                                "aggressive": ("4", "1"),
                            }
                        elif max_vram >= 16:
                            tiers = {
                                "safe": ("2", "4"),
                                "normal": ("2", "2"),
                                "aggressive": ("4", "2"),
                            }
                        elif max_vram >= 11.5:
                            tiers = {
                                "safe": ("1", "4"),
                                "normal": ("2", "4"),
                                "aggressive": ("2", "2"),
                            }
                        elif max_vram >= 8:
                            tiers = {
                                "safe": ("1", "4"),
                                "normal": ("1", "2"),
                                "aggressive": ("2", "4"),
                            }
                        else:
                            tiers = {
                                "safe": ("1", "4"),
                                "normal": ("1", "4"),
                                "aggressive": ("1", "2"),
                            }
                        bsz, ga = tiers.get(mem, tiers.get("safe"))
                        batch_tf.value = bsz
                        grad_acc_tf.value = ga

                        # Speed vs quality: packing + bounded LR
                        if qual == "speed":
                            packing_cb.value = True
                            lr_tf.value = "2e-4" if max_vram >= 8 else "1e-4"
                        elif qual == "quality":
                            packing_cb.value = False
                            lr_tf.value = "1e-4"
                        else:
                            packing_cb.value = True
                            lr_tf.value = "1e-4" if max_vram < 8 else "2e-4"

                        try:
                            simple_summary_txt.value = (
                                f"Will run ~{max_steps_tf.value} steps • batch {batch_tf.value} • grad accum {grad_acc_tf.value} "
                                f"• packing {'on' if bool(getattr(packing_cb, 'value', False)) else 'off'}"
                            )
                        except Exception:
                            pass
                    else:
                        # Auto Set (local): use detected GPU VRAM to choose a config that
                        # pushes the system reasonably hard without being reckless.
                        try:
                            data = gather_local_specs_helper()
                        except Exception:
                            data = {}
                        try:
                            gpus = list(data.get("gpus") or [])
                            max_vram = 0.0
                            for g in gpus:
                                try:
                                    v = g.get("vram_gb")
                                    if isinstance(v, (int, float)) and v is not None:
                                        max_vram = max(max_vram, float(v))
                                except Exception:
                                    pass
                        except Exception:
                            max_vram = 0.0
                        # Aggressive heuristic tiers for 4-bit 7B/8B LoRA-style training.
                        # Goal: fly high without crashing -> prioritize per-device batch, keep grad_acc low.
                        # Always overwrite hidden values so switching presets is predictable.
                        try:
                            packing_cb.value = True
                        except Exception:
                            pass
                        if max_vram >= 48:
                            lr_tf.value = "2e-4"
                            batch_tf.value = "8"
                            grad_acc_tf.value = "1"
                            max_steps_tf.value = "800"
                        elif max_vram >= 24:
                            lr_tf.value = "2e-4"
                            batch_tf.value = "4"
                            grad_acc_tf.value = "1"
                            max_steps_tf.value = "600"
                        elif max_vram >= 16:
                            lr_tf.value = "2e-4"
                            batch_tf.value = "4"
                            grad_acc_tf.value = "1"
                            max_steps_tf.value = "400"
                        elif max_vram >= 11.5:
                            lr_tf.value = "2e-4"
                            batch_tf.value = "2"
                            grad_acc_tf.value = "1"
                            max_steps_tf.value = "300"
                        elif max_vram >= 8:
                            lr_tf.value = "1e-4"
                            batch_tf.value = "1"
                            grad_acc_tf.value = "2"
                            max_steps_tf.value = "200"
                        elif max_vram > 0:
                            lr_tf.value = "1e-4"
                            batch_tf.value = "1"
                            grad_acc_tf.value = "4"
                            max_steps_tf.value = "100"
                        else:
                            # CPU-only or unknown GPU: keep this functional but still short.
                            lr_tf.value = "1e-4"
                            batch_tf.value = "1"
                            grad_acc_tf.value = "4"
                            max_steps_tf.value = "50"
            except Exception:
                pass

        # Apply visibility and default changes immediately
        try:
            page.update()
        except Exception:
            pass

    # Wire up skill-level and dataset source handlers
    skill_level.on_change = _update_skill_controls
    beginner_mode_dd.on_change = _update_skill_controls
    simple_duration_dd.on_change = _update_skill_controls
    simple_memory_dd.on_change = _update_skill_controls
    simple_quality_dd.on_change = _update_skill_controls
    train_source.on_change = _update_train_source
    # Initialize skill-level dependent visibility once
    try:
        _update_skill_controls()
    except Exception:
        pass
    # Initialize database sessions dropdown (since Database is default source)
    try:
        _refresh_db_sessions()
    except Exception:
        pass

    def _build_hp() -> dict:
        """Build train.py flags via helper (delegated)."""
        hp = build_hp_from_controls_helper(
            train_source=train_source,
            train_hf_repo=train_hf_repo,
            train_hf_split=train_hf_split,
            train_hf_config=train_hf_config,
            train_json_path=train_json_path,
            train_db_session_dd=train_db_session_dd,
            base_model=base_model,
            out_dir_tf=out_dir_tf,
            epochs_tf=epochs_tf,
            lr_tf=lr_tf,
            batch_tf=batch_tf,
            grad_acc_tf=grad_acc_tf,
            max_steps_tf=max_steps_tf,
            use_lora_cb=use_lora_cb,
            lora_r_dd=lora_r_dd,
            lora_alpha_tf=lora_alpha_tf,
            lora_dropout_tf=lora_dropout_tf,
            use_rslora_cb=use_rslora_cb,
            packing_cb=packing_cb,
            auto_resume_cb=auto_resume_cb,
            push_cb=push_cb,
            hf_repo_id_tf=hf_repo_id_tf,
            resume_from_tf=resume_from_tf,
            warmup_steps_tf=warmup_steps_tf,
            weight_decay_tf=weight_decay_tf,
            lr_sched_dd=lr_sched_dd,
            optim_dd=optim_dd,
            logging_steps_tf=logging_steps_tf,
            logging_first_step_cb=logging_first_step_cb,
            disable_tqdm_cb=disable_tqdm_cb,
            seed_tf=seed_tf,
            save_strategy_dd=save_strategy_dd,
            save_total_limit_tf=save_total_limit_tf,
            pin_memory_cb=pin_memory_cb,
            report_to_dd=report_to_dd,
            fp16_cb=fp16_cb,
            bf16_cb=bf16_cb,
        )

        try:
            is_offline = bool(getattr(offline_mode_sw, "value", False))
        except Exception:
            is_offline = False
        if is_offline and isinstance(hp, dict):
            try:
                hp.pop("hf_dataset_id", None)
                hp.pop("hf_dataset_split", None)
                hp.pop("hf_dataset_config", None)
            except Exception:
                pass
        return hp

    async def _on_pod_created(data: dict):
        try:
            hp = (data or {}).get("hp") or {}
            chosen_gpu_type_id = (data or {}).get("chosen_gpu_type_id")
            chosen_interruptible = bool((data or {}).get("chosen_interruptible"))
            default_name = (
                f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
                f"{str(hp.get('base_model', 'model')).replace('/', '_')}"
            )
            name_tf = ft.TextField(label="Save as", value=default_name, width=420)

            def _collect_infra_ui_state() -> dict:
                return {
                    "dc": (rp_dc_tf.value or "US-NC-1"),
                    "vol_name": (rp_vol_name_tf.value or "unsloth-volume"),
                    "vol_size": int(float(rp_vol_size_tf.value or "50")),
                    "resize_if_smaller": bool(getattr(rp_resize_cb, "value", True)),
                    "tpl_name": (rp_tpl_name_tf.value or "unsloth-trainer-template"),
                    "image": (rp_image_tf.value or "docker.io/sbussiso/unsloth-trainer:latest"),
                    "container_disk": int(float(rp_container_disk_tf.value or "30")),
                    "pod_volume_gb": int(float(rp_volume_in_gb_tf.value or "0")),
                    "mount_path": (rp_mount_path_tf.value or "/data"),
                    "category": (rp_category_tf.value or "NVIDIA"),
                    "public": bool(getattr(rp_public_cb, "value", False)),
                }

            payload = {
                "hp": hp,
                "infra_ui": _collect_infra_ui_state(),
                "meta": {
                    "skill_level": skill_level.value,
                    "beginner_mode": (beginner_mode_dd.value if (skill_level.value or "") == "Beginner" else ""),
                },
                "pod": {
                    "gpu_type_id": chosen_gpu_type_id,
                    "interruptible": bool(chosen_interruptible),
                },
            }

            def _do_save(_=None):
                name = (name_tf.value or default_name).strip()
                if not name:
                    return
                # Remove .json extension if present (database stores by name only)
                if name.lower().endswith(".json"):
                    name = name[:-5]
                try:
                    from helpers.training_config import save_config

                    save_config(name, payload)
                    page.snack_bar = ft.SnackBar(ft.Text(f"Saved config: {name}"))
                    page.snack_bar.open = True
                    _refresh_config_list()
                except Exception as ex:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Failed to save config: {ex}"))
                    page.snack_bar.open = True
                try:
                    dlg.open = False
                    page.update()
                except Exception:
                    pass

            dlg = ft.AlertDialog(
                modal=True,
                title=ft.Row(
                    [
                        ft.Icon(
                            getattr(ICONS, "SAVE_ALT", ICONS.SAVE),
                            color=ACCENT_COLOR,
                        ),
                        ft.Text("Save this training setup?"),
                    ],
                    alignment=ft.MainAxisAlignment.START,
                ),
                content=ft.Column(
                    [
                        ft.Text(
                            "You can reuse this configuration later via Training a Configuration mode.",
                        ),
                        name_tf,
                    ],
                    tight=True,
                    spacing=6,
                ),
                actions=[
                    ft.TextButton(
                        "Skip",
                        on_click=lambda e: (
                            setattr(dlg, "open", False),
                            page.update(),
                        ),
                    ),
                    ft.ElevatedButton(
                        "Save",
                        icon=getattr(ICONS, "SAVE", ICONS.CHECK),
                        on_click=_do_save,
                    ),
                ],
            )
            try:
                dlg.on_dismiss = lambda e: page.update()
            except Exception:
                pass
            opened = False
            try:
                if hasattr(page, "open") and callable(getattr(page, "open")):
                    page.open(dlg)
                    opened = True
            except Exception:
                opened = False
            if not opened:
                page.dialog = dlg
                dlg.open = True
            train_timeline.controls.append(
                ft.Row(
                    [
                        ft.Icon(
                            getattr(ICONS, "SAVE", ICONS.SAVE_ALT),
                            color=WITH_OPACITY(0.9, COLORS.BLUE),
                        ),
                        ft.Text("Opened Save Configuration dialog"),
                    ]
                )
            )
            update_train_placeholders()
            await safe_update(page)
            try:
                await asyncio.sleep(0.05)
            except Exception:
                pass
        except Exception:
            pass

    async def on_start_training():
        # Respect Offline Mode: block Runpod training when offline
        try:
            if bool(getattr(offline_mode_sw, "value", False)):
                try:
                    page.snack_bar = ft.SnackBar(
                        ft.Text(
                            "Offline mode is enabled; Runpod training is disabled. Switch target to 'local' or disable Offline mode."
                        ),
                    )
                    page.open(page.snack_bar)
                    await safe_update(page)
                except Exception:
                    pass
                return
        except Exception:
            pass

        # Mark status as starting before handing off to Runpod helper
        try:
            _set_runpod_status("starting")
        except Exception:
            pass

        return await run_pod_training_helper(
            page=page,
            rp_pod_module=rp_pod,
            train_state=train_state,
            cancel_train=cancel_train,
            _runpod_cfg=_runpod_cfg,
            rp_temp_key_tf=rp_temp_key_tf,
            config_mode_dd=config_mode_dd,
            config_files_dd=config_files_dd,
            skill_level=skill_level,
            beginner_mode_dd=beginner_mode_dd,
            expert_gpu_dd=expert_gpu_dd,
            expert_spot_cb=expert_spot_cb,
            base_model=base_model,
            train_source=train_source,
            train_hf_repo=train_hf_repo,
            train_hf_split=train_hf_split,
            train_json_path=train_json_path,
            train_db_session_dd=train_db_session_dd,
            train_timeline=train_timeline,
            train_progress=train_progress,
            train_prog_label=train_prog_label,
            start_train_btn=start_train_btn,
            stop_train_btn=stop_train_btn,
            refresh_train_btn=refresh_train_btn,
            restart_container_btn=restart_container_btn,
            open_runpod_btn=open_runpod_btn,
            open_web_terminal_btn=open_web_terminal_btn,
            copy_ssh_btn=copy_ssh_btn,
            auto_terminate_cb=auto_terminate_cb,
            update_train_placeholders=update_train_placeholders,
            refresh_teardown_ui_fn=_refresh_teardown_ui,
            update_progress_from_logs=_update_progress_from_logs,
            build_hp_fn=_build_hp,
            on_pod_created=_on_pod_created,
            set_status_fn=_set_runpod_status,
        )

    async def on_restart_container():
        return await restart_pod_container_helper(
            page=page,
            rp_pod_module=rp_pod,
            train_state=train_state,
            config_mode_dd=config_mode_dd,
            config_files_dd=config_files_dd,
            train_timeline=train_timeline,
            update_train_placeholders=update_train_placeholders,
            build_hp_fn=_build_hp,
        )

    def on_open_runpod(_):
        if bool(getattr(offline_mode_sw, "value", False)):
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Offline mode is enabled; opening Runpod dashboard is disabled."))
                page.open(page.snack_bar)
            except Exception:
                pass
            return
        return open_runpod_helper(page, train_state, train_timeline)

    def on_open_web_terminal(_):
        if bool(getattr(offline_mode_sw, "value", False)):
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Offline mode is enabled; remote web terminals are disabled."))
                page.open(page.snack_bar)
            except Exception:
                pass
            return
        return open_web_terminal_helper(page, train_state, train_timeline)

    async def on_copy_ssh_command(_):
        if bool(getattr(offline_mode_sw, "value", False)):
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Offline mode is enabled; remote SSH commands are disabled."))
                page.open(page.snack_bar)
                await safe_update(page)
            except Exception:
                pass
            return

        return await copy_ssh_command_helper(
            page=page,
            rp_pod_module=rp_pod,
            train_state=train_state,
            train_timeline=train_timeline,
        )

    def on_stop_training(_):
        if not train_state.get("running"):
            pod_id = train_state.get("pod_id")
            if pod_id:
                try:

                    async def _terminate():
                        try:
                            await asyncio.to_thread(
                                rp_pod.delete_pod,
                                (train_state.get("api_key") or os.environ.get("RUNPOD_API_KEY") or "").strip(),
                                pod_id,
                            )
                        finally:
                            try:
                                train_timeline.controls.append(
                                    ft.Row(
                                        [
                                            ft.Icon(ICONS.CANCEL, color=COLORS.RED),
                                            ft.Text(
                                                "Termination requested for existing pod",
                                            ),
                                        ]
                                    )
                                )
                                start_train_btn.visible = True
                                start_train_btn.disabled = False
                                stop_train_btn.disabled = True
                                refresh_train_btn.disabled = False
                                train_state["pod_id"] = None
                                update_train_placeholders()
                                await safe_update(page)
                            except Exception:
                                pass

                    _schedule_task(page, _terminate)
                except Exception:
                    pass
            return
        cancel_train["cancelled"] = True
        try:
            train_timeline.controls.append(
                ft.Row(
                    [
                        ft.Icon(ICONS.CANCEL, color=COLORS.RED),
                        ft.Text("Cancel requested — training will stop ASAP"),
                    ]
                )
            )
            stop_train_btn.disabled = True
            update_train_placeholders()
            try:
                _set_runpod_status("cancelled")
            except Exception:
                pass
            page.update()
        except Exception:
            pass

    def on_refresh_training(_):
        """Clear Runpod training logs and reset progress placeholders."""
        try:
            train_timeline.controls.clear()
            try:
                train_state["log_seen"] = set()
            except Exception:
                pass
            update_train_placeholders()
            page.update()
        except Exception:
            pass

    # Training target selector (top of Training tab)
    train_target_dd = ft.Dropdown(
        label="Training target",
        options=[
            ft.dropdown.Option("Runpod - Pod"),
            ft.dropdown.Option(key="local", text="Local"),
        ],
        value="Runpod - Pod",
        width=420,
        tooltip=(
            "Choose where training runs. 'Runpod - Pod' uses the Runpod workflow; "
            "'Local' runs directly on this machine (no Docker)."
        ),
    )

    train_target_offline_reason = offline_reason_text("Offline Mode: cloud training is disabled (Local only).")

    # Configuration mode controls (basic UI; wiring migrated later)
    config_mode_dd = ft.Dropdown(
        label="Mode",
        options=[ft.dropdown.Option("Normal"), ft.dropdown.Option("Configuration")],
        value="Normal",
        width=180,
        tooltip="Configuration mode: load a saved config and run with minimal inputs.",
    )
    config_files_dd = ft.Dropdown(label="Saved config", options=[], width=420)
    config_refresh_btn = ft.TextButton("Refresh", icon=REFRESH_ICON)
    load_config_btn = ft.ElevatedButton(
        "Load Config",
        icon=getattr(ICONS, "FILE_OPEN", getattr(ICONS, "FOLDER_OPEN", ICONS.UPLOAD)),
    )
    config_save_current_btn = ft.ElevatedButton(
        "Save Current",
        icon=getattr(ICONS, "SAVE", ICONS.CHECK),
        tooltip="Save the current training setup as a reusable config",
    )
    config_edit_btn = ft.TextButton(
        "Edit",
        icon=getattr(ICONS, "EDIT", getattr(ICONS, "MODE_EDIT", ICONS.SETTINGS)),
        tooltip="Edit selected config file",
        disabled=True,
    )

    local_view_metrics_btn = ft.OutlinedButton(
        "View training metrics",
        icon=getattr(
            ICONS,
            "SHOW_CHART",
            getattr(ICONS, "QUERY_STATS", getattr(ICONS, "INSIGHTS", getattr(ICONS, "INFO", None))),
        ),
        disabled=True,
    )
    config_rename_btn = ft.TextButton(
        "Rename",
        icon=getattr(ICONS, "DRIVE_FILE_RENAME_OUTLINE", getattr(ICONS, "EDIT", ICONS.SETTINGS)),
        tooltip="Rename selected config file",
        disabled=True,
    )
    config_delete_btn = ft.TextButton(
        "Delete",
        icon=getattr(ICONS, "DELETE", getattr(ICONS, "DELETE_FOREVER", ICONS.CLOSE)),
        tooltip="Delete selected config file",
        disabled=True,
    )
    config_summary_txt = ft.Text("", size=12, color=WITH_OPACITY(0.7, BORDER_BASE))
    config_files_row = ft.Row(
        [
            config_files_dd,
            config_refresh_btn,
            load_config_btn,
            config_save_current_btn,
            config_edit_btn,
            config_rename_btn,
            config_delete_btn,
        ],
        wrap=True,
    )

    # Basic config helpers (full wiring and calls migrated in later steps)
    def _saved_configs_dir() -> str:
        return saved_configs_dir_helper()

    def _list_saved_configs() -> List[str]:
        return list_saved_configs_helper(_saved_configs_dir())

    def _update_config_buttons_enabled(_=None):
        try:
            has_sel = bool((config_files_dd.value or "").strip())
            config_edit_btn.disabled = not has_sel
            config_rename_btn.disabled = not has_sel
            config_delete_btn.disabled = not has_sel
        except Exception:
            pass
        try:
            page.update()
        except Exception:
            pass

    def _refresh_config_list(_=None):
        """Refresh saved config dropdown, filtered by current training target.

        Configs without an explicit meta.train_target are shown for all targets
        (backward compatibility).
        """
        try:
            all_files = _list_saved_configs()
        except Exception:
            all_files = []
        files: List[str] = []
        try:
            cur_target = (train_target_dd.value or "Runpod - Pod").strip().lower()
        except Exception:
            cur_target = "runpod - pod"
        for name in all_files:
            tgt_val = ""
            try:
                # Remove .json extension if present
                config_name = name[:-5] if name.lower().endswith(".json") else name
                conf = _read_json_file(config_name) or {}
                meta = conf.get("meta") or {}
                tgt_val = str(meta.get("train_target") or "").strip().lower()
            except Exception:
                tgt_val = ""
            # No target recorded: show for all targets
            if not tgt_val:
                files.append(name)
                continue
            # Target-aware filtering
            if cur_target.startswith("runpod - pod"):
                if tgt_val.startswith("runpod - pod") or tgt_val == "runpod":
                    files.append(name)
            elif cur_target.startswith("local"):
                if tgt_val.startswith("local"):
                    files.append(name)
            else:
                # Unknown target label; do not hide config
                files.append(name)
        try:
            config_files_dd.options = [ft.dropdown.Option(f) for f in files]
            cur_val = (config_files_dd.value or "").strip()
            if cur_val and (cur_val not in files):
                config_files_dd.value = files[0] if files else None
            if files and not config_files_dd.value:
                config_files_dd.value = files[0]
            if not files:
                config_files_dd.value = None
        except Exception:
            pass
        _update_config_buttons_enabled()

    def _read_json_file(path: str) -> Optional[dict]:
        return read_json_file_helper(path)

    def _validate_config(conf: dict) -> Tuple[bool, str]:
        return validate_config_helper(conf)

    # ---------- Runpod Infrastructure (Ensure volume + template) ----------
    rp_dc_tf = ft.TextField(label="Datacenter ID", value="US-NC-1", width=140)
    rp_vol_name_tf = ft.TextField(label="Volume name", value="unsloth-volume", width=220)
    rp_vol_size_tf = ft.TextField(label="Volume size (GB)", value="50", width=180)
    rp_resize_cb = ft.Checkbox(label="Resize if smaller", value=True)

    # Info icon for "Resize if smaller": explains that existing smaller volumes will be expanded (never shrunk)
    rp_resize_info = ft.IconButton(
        icon=getattr(ICONS, "INFO_OUTLINE", getattr(ICONS, "INFO", getattr(ICONS, "HELP_OUTLINE", None))),
        tooltip=(
            "If enabled, an existing volume smaller than the requested size will be expanded. It never shrinks volumes."
        ),
        on_click=_mk_help_handler(
            "When ensuring the Runpod Network Volume: if a volume with this name already exists and its size "
            "is smaller than the size you specify, it will be automatically increased to match your requested "
            "size. Existing volumes are never shrunk.",
        ),
    )

    # Keep the info icon on the right of the checkbox by grouping them together
    rp_resize_row = ft.Row([rp_resize_cb, rp_resize_info], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    rp_tpl_name_tf = ft.TextField(label="Template name", value="unsloth-trainer-template", width=260)
    rp_image_tf = ft.TextField(
        label="Image name",
        value="docker.io/sbussiso/unsloth-trainer:latest",
        width=360,
    )
    rp_container_disk_tf = ft.TextField(label="Container disk (GB)", value="30", width=200)
    rp_volume_in_gb_tf = ft.TextField(
        label="Pod volume (GB)",
        value="0",
        width=180,
        tooltip="Optional pod-local disk, not the network volume",
    )
    rp_mount_path_tf = ft.TextField(
        label="Mount path",
        value="/data",
        width=220,
        tooltip=("Avoid mounting at /workspace to prevent hiding train.py inside the image. /data is recommended."),
    )
    rp_category_tf = ft.TextField(label="Category", value="NVIDIA", width=160)
    rp_public_cb = ft.Checkbox(label="Public template", value=False)

    # Temporary API key input (overrides Settings key for this session)
    rp_temp_key_tf = ft.TextField(
        label="Runpod API key (temp)",
        password=True,
        can_reveal_password=True,
        width=420,
        tooltip=("Optional. Overrides Settings key for this run. You can also set RUNPOD_API_KEY env var."),
    )
    try:
        rp_temp_key_tf.visible = False
    except Exception:
        pass

    # Info icon for "Public template": clarifies visibility and considerations
    rp_public_info = ft.IconButton(
        icon=getattr(ICONS, "INFO_OUTLINE", getattr(ICONS, "INFO", getattr(ICONS, "HELP_OUTLINE", None))),
        tooltip="Make this template visible to all Runpod users. Be mindful of sensitive env vars.",
        on_click=_mk_help_handler(
            "Public templates are discoverable by other Runpod users. Others can launch pods using this "
            "template. If your image is private or requires registry auth, they will need access to run it. "
            "Avoid putting sensitive environment variables in the template.",
        ),
    )
    rp_public_row = ft.Row([rp_public_cb, rp_public_info], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    # (SSH exposure option removed by request)
    rp_infra_busy = ft.ProgressRing(visible=False)

    async def on_ensure_infra():
        # Respect Offline Mode: Runpod infra API is a cloud operation
        try:
            if bool(getattr(offline_mode_sw, "value", False)):
                try:
                    page.snack_bar = ft.SnackBar(
                        ft.Text("Offline mode is enabled; ensuring Runpod infrastructure is disabled."),
                    )
                    page.snack_bar.open = True
                    await safe_update(page)
                except Exception:
                    pass
                return
        except Exception:
            pass

        return await ensure_infrastructure_helper(
            page=page,
            rp_infra_module=rp_infra,
            _hf_cfg=_hf_cfg,
            _runpod_cfg=_runpod_cfg,
            rp_dc_tf=rp_dc_tf,
            rp_vol_name_tf=rp_vol_name_tf,
            rp_vol_size_tf=rp_vol_size_tf,
            rp_resize_cb=rp_resize_cb,
            rp_tpl_name_tf=rp_tpl_name_tf,
            rp_image_tf=rp_image_tf,
            rp_container_disk_tf=rp_container_disk_tf,
            rp_volume_in_gb_tf=rp_volume_in_gb_tf,
            rp_mount_path_tf=rp_mount_path_tf,
            rp_category_tf=rp_category_tf,
            rp_public_cb=rp_public_cb,
            rp_temp_key_tf=rp_temp_key_tf,
            rp_infra_busy=rp_infra_busy,
            train_timeline=train_timeline,
            refresh_expert_gpus_fn=lambda: page.run_task(refresh_expert_gpus),
            refresh_teardown_ui_fn=_refresh_teardown_ui,
            train_state=train_state,
            dataset_section=dataset_section,
            train_params_section=train_params_section,
            start_train_btn=start_train_btn,
            stop_train_btn=stop_train_btn,
            refresh_train_btn=refresh_train_btn,
            _update_mode_visibility=_update_mode_visibility,
        )

    def on_click_ensure_infra(e):
        # Immediate feedback to confirm click registered and show activity
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Ensuring Runpod infrastructure…"))
            page.snack_bar.open = True
            rp_infra_busy.visible = True
            update_train_placeholders()
            page.update()
        except Exception:
            pass
        _schedule_task(page, on_ensure_infra)

    ensure_infra_btn = ft.ElevatedButton(
        "Ensure Infrastructure",
        icon=getattr(ICONS, "CLOUD_DONE", getattr(ICONS, "CLOUD", ICONS.SETTINGS)),
        on_click=on_click_ensure_infra,
    )
    try:
        if rp_infra is None:
            ensure_infra_btn.disabled = True
            ensure_infra_btn.tooltip = "Runpod infra helper unavailable. Install/enable runpod ensure_infra."
    except Exception:
        pass
    rp_infra_actions = ft.Row(
        [
            ensure_infra_btn,
            rp_infra_busy,
        ],
        spacing=10,
    )

    # Populate Expert GPU dropdown from Runpod based on datacenter (delegated)
    async def refresh_expert_gpus(_=None):
        return await refresh_expert_gpus_helper(
            page=page,
            rp_pod_module=rp_pod,
            train_state=train_state,
            _runpod_cfg=_runpod_cfg,
            rp_temp_key_tf=rp_temp_key_tf,
            rp_dc_tf=rp_dc_tf,
            expert_gpu_busy=expert_gpu_busy,
            expert_gpu_dd=expert_gpu_dd,
            expert_spot_cb=expert_spot_cb,
            expert_gpu_avail=expert_gpu_avail,
            _update_expert_spot_enabled=_update_expert_spot_enabled,
        )

    # Populate Expert GPU dropdown from LOCAL system GPUs (delegated)
    async def refresh_local_gpus(_=None):
        return await refresh_local_gpus_helper(
            page=page,
            expert_gpu_busy=expert_gpu_busy,
            expert_gpu_dd=expert_gpu_dd,
            expert_spot_cb=expert_spot_cb,
            expert_gpu_avail=expert_gpu_avail,
        )

    # Refresh local system specs (CPU/RAM/disk/GPU) and update the Local Training: System Specs UI.
    async def on_refresh_local_specs(_=None):
        try:
            data = gather_local_specs_helper()
        except Exception:
            data = {}

        try:
            local_os_txt.value = str(data.get("os") or "")
            local_py_txt.value = str(data.get("python") or "")
            local_cpu_txt.value = str(data.get("cpu_cores") or "")

            ram_gb = data.get("ram_gb")
            local_ram_txt.value = f"{ram_gb} GB" if isinstance(ram_gb, (int, float)) else "?"

            disk_gb = data.get("disk_free_gb")
            local_disk_txt.value = f"{disk_gb} GB free" if isinstance(disk_gb, (int, float)) else "?"

            torch_ok = bool(data.get("torch_installed"))
            cuda_ok = bool(data.get("cuda_available"))
            local_torch_txt.value = "Installed" if torch_ok else "Not installed"
            local_cuda_txt.value = "Available" if cuda_ok else "Not available"

            gpus = list(data.get("gpus") or [])
            gpu_lines: List[str] = []
            for g in gpus:
                try:
                    idx = g.get("index")
                    name = str(g.get("name") or f"GPU {idx}")
                    vram = g.get("vram_gb")
                    if isinstance(vram, (int, float)):
                        gpu_lines.append(f"GPU {idx}: {name} ({vram} GB)")
                    else:
                        gpu_lines.append(f"GPU {idx}: {name}")
                except Exception:
                    continue
            local_gpus_txt.value = "\n".join(gpu_lines) if gpu_lines else "No NVIDIA GPUs detected."

            local_capability_txt.value = str(data.get("capability") or "Unknown")

            # Red flags list
            flags = list(data.get("red_flags") or [])
            local_flags_col.controls.clear()
            if flags:
                for msg in flags:
                    try:
                        local_flags_col.controls.append(ft.Text(str(msg)))
                    except Exception:
                        continue
                local_flags_box.visible = True
            else:
                local_flags_box.visible = False

            await safe_update(page)
        except Exception:
            # Best-effort; avoid breaking the UI if specs gathering fails
            try:
                await safe_update(page)
            except Exception:
                pass

    # Wire refresh button (dispatch based on training target, but block cloud when offline)
    def on_click_expert_gpu_refresh(e):
        try:
            is_offline = bool(getattr(offline_mode_sw, "value", False))
        except Exception:
            is_offline = False

        try:
            tgt = (train_target_dd.value or "Runpod - Pod").lower()
        except Exception:
            tgt = "runpod - pod"

        if is_offline and tgt.startswith("runpod - pod"):
            try:
                page.snack_bar = ft.SnackBar(
                    ft.Text("Offline mode is enabled; refreshing Runpod GPU catalog is disabled."),
                )
                page.open(page.snack_bar)
            except Exception:
                pass
            return

        try:
            if tgt.startswith("runpod - pod"):
                page.run_task(refresh_expert_gpus)
            else:
                page.run_task(refresh_local_gpus)
        except Exception:
            pass

    try:
        expert_gpu_refresh_btn.on_click = on_click_expert_gpu_refresh
    except Exception:
        pass

    rp_infra_compact_row = ft.Row(
        [
            ft.OutlinedButton(
                "Ensure Infrastructure",
                icon=getattr(ICONS, "CLOUD_DONE", getattr(ICONS, "CLOUD", ICONS.SETTINGS)),
                on_click=on_click_ensure_infra,
            ),
        ],
        spacing=10,
    )

    config_section = build_config_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        config_mode_dd=config_mode_dd,
        config_files_row=config_files_row,
        config_summary_txt=config_summary_txt,
        rp_infra_compact_row=rp_infra_compact_row,
    )

    rp_infra_panel = build_rp_infra_panel(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        rp_dc_tf=rp_dc_tf,
        rp_vol_name_tf=rp_vol_name_tf,
        rp_vol_size_tf=rp_vol_size_tf,
        rp_resize_row=rp_resize_row,
        rp_tpl_name_tf=rp_tpl_name_tf,
        rp_image_tf=rp_image_tf,
        rp_container_disk_tf=rp_container_disk_tf,
        rp_volume_in_gb_tf=rp_volume_in_gb_tf,
        rp_mount_path_tf=rp_mount_path_tf,
        rp_category_tf=rp_category_tf,
        rp_public_row=rp_public_row,
        rp_temp_key_tf=rp_temp_key_tf,
        rp_infra_actions=rp_infra_actions,
    )

    rp_ds_divider = ft.Divider()

    start_train_btn = ft.ElevatedButton(
        "Start Training",
        icon=getattr(ICONS, "SCIENCE", ICONS.PLAY_CIRCLE),
        on_click=lambda e: page.run_task(on_start_training),
        disabled=True,
    )
    stop_train_btn = ft.OutlinedButton(
        "Stop",
        icon=ICONS.STOP_CIRCLE,
        on_click=on_stop_training,
        disabled=True,
    )
    refresh_train_btn = ft.TextButton(
        "Refresh",
        icon=REFRESH_ICON,
        on_click=on_refresh_training,
        disabled=True,
    )
    restart_container_btn = ft.OutlinedButton(
        "Restart Container",
        icon=getattr(ICONS, "RESTART_ALT", getattr(ICONS, "REFRESH", ICONS.SETTINGS)),
        on_click=lambda e: page.run_task(on_restart_container),
        disabled=True,
    )
    auto_terminate_cb = ft.Checkbox(
        label="Auto-terminate on finish",
        value=True,
        tooltip="Delete pod automatically when training reaches a terminal state.",
    )
    open_runpod_btn = ft.TextButton(
        "Open in Runpod",
        icon=getattr(ICONS, "OPEN_IN_NEW", getattr(ICONS, "LINK", ICONS.SETTINGS)),
        on_click=on_open_runpod,
        disabled=True,
    )
    open_web_terminal_btn = ft.TextButton(
        "Open Web Terminal",
        icon=getattr(ICONS, "TERMINAL", getattr(ICONS, "CODE", ICONS.OPEN_IN_NEW)),
        on_click=on_open_web_terminal,
        disabled=True,
        tooltip="Opens the pod page; then click Connect a Open Web Terminal",
    )
    copy_ssh_btn = ft.TextButton(
        "Copy SSH Command",
        icon=getattr(
            ICONS,
            "CONTENT_COPY",
            getattr(ICONS, "COPY", ICONS.LINK),
        ),
        on_click=lambda e: page.run_task(on_copy_ssh_command),
        disabled=True,
        tooltip="Copies an SSH command for this pod to your clipboard.",
    )
    save_config_bottom_btn = ft.TextButton(
        "Save current setup",
        icon=getattr(ICONS, "SAVE", ICONS.CHECK),
        tooltip=("Save the current training setup (dataset, hyperparameters, target, and infra) as a reusable config."),
        on_click=lambda e: page.run_task(on_save_current_config),
    )
    train_actions = ft.Row(
        [
            start_train_btn,
            stop_train_btn,
            refresh_train_btn,
            restart_container_btn,
            open_runpod_btn,
            open_web_terminal_btn,
            copy_ssh_btn,
            auto_terminate_cb,
            save_config_bottom_btn,
        ],
        spacing=10,
    )

    section_title(
        "Teardown",
        getattr(ICONS, "DELETE_FOREVER", getattr(ICONS, "DELETE", ICONS.CLOSE)),
        "Select infrastructure items to delete. Teardown All removes all related items.",
        on_help_click=_mk_help_handler(
            "Delete Runpod Template and/or Network Volume. If a pod exists, you can delete it too.",
        ),
    )
    td_template_cb = ft.Checkbox(
        label="Template: (none)",
        value=False,
        visible=False,
    )
    td_volume_cb = ft.Checkbox(
        label="Volume: (none)",
        value=False,
        visible=False,
    )
    td_pod_cb = ft.Checkbox(
        label="Pod: (none)",
        value=False,
        visible=False,
    )
    td_busy = ft.ProgressRing(visible=False)

    async def _refresh_teardown_ui(_=None):
        return await refresh_teardown_ui_helper(
            page=page,
            rp_pod_module=rp_pod,
            train_state=train_state,
            td_template_cb=td_template_cb,
            td_volume_cb=td_volume_cb,
            td_pod_cb=td_pod_cb,
            teardown_section=teardown_section,
            _runpod_cfg=_runpod_cfg,
            rp_temp_key_tf=rp_temp_key_tf,
        )

    async def _do_teardown(selected_all: bool = False):
        return await do_teardown_helper(
            page=page,
            rp_pod_module=rp_pod,
            rp_infra_module=rp_infra,
            train_state=train_state,
            td_template_cb=td_template_cb,
            td_volume_cb=td_volume_cb,
            td_pod_cb=td_pod_cb,
            td_busy=td_busy,
            train_timeline=train_timeline,
            update_train_placeholders=update_train_placeholders,
            _runpod_cfg=_runpod_cfg,
            rp_temp_key_tf=rp_temp_key_tf,
            refresh_teardown_ui_fn=_refresh_teardown_ui,
            selected_all=selected_all,
        )

    async def on_teardown_selected(_=None):
        return await confirm_teardown_selected_helper(
            page=page,
            td_template_cb=td_template_cb,
            td_volume_cb=td_volume_cb,
            td_pod_cb=td_pod_cb,
            td_busy=td_busy,
            train_timeline=train_timeline,
            update_train_placeholders=update_train_placeholders,
            _runpod_cfg=_runpod_cfg,
            rp_temp_key_tf=rp_temp_key_tf,
            rp_pod_module=rp_pod,
            rp_infra_module=rp_infra,
            train_state=train_state,
            refresh_teardown_ui_fn=_refresh_teardown_ui,
        )

    async def on_teardown_all(_=None):
        return await confirm_teardown_all_helper(
            page=page,
            td_template_cb=td_template_cb,
            td_volume_cb=td_volume_cb,
            td_pod_cb=td_pod_cb,
            td_busy=td_busy,
            train_timeline=train_timeline,
            update_train_placeholders=update_train_placeholders,
            _runpod_cfg=_runpod_cfg,
            rp_temp_key_tf=rp_temp_key_tf,
            rp_pod_module=rp_pod,
            rp_infra_module=rp_infra,
            train_state=train_state,
            refresh_teardown_ui_fn=_refresh_teardown_ui,
        )

    teardown_section = build_teardown_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        td_template_cb=td_template_cb,
        td_volume_cb=td_volume_cb,
        td_pod_cb=td_pod_cb,
        td_busy=td_busy,
        on_teardown_selected_cb=lambda e: page.run_task(on_teardown_selected),
        on_teardown_all_cb=lambda e: page.run_task(on_teardown_all),
    )

    train_source_offline_reason = offline_reason_text("Offline Mode: Hugging Face datasets are disabled.")
    dataset_section = build_dataset_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        train_source=train_source,
        offline_reason=train_source_offline_reason,
        train_hf_repo=train_hf_repo,
        train_hf_split=train_hf_split,
        train_hf_config=train_hf_config,
        train_json_path=train_json_path,
        train_json_browse_btn=train_json_browse_btn,
        train_db_session_dd=train_db_session_dd,
        train_db_refresh_btn=train_db_refresh_btn,
        train_db_pair_count=train_db_pair_count,
        visible=False,
    )

    train_params_section = build_train_params_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        _mk_help_handler=_mk_help_handler,
        skill_level=skill_level,
        beginner_mode_dd=beginner_mode_dd,
        beginner_simple_custom_panel=beginner_simple_custom_panel,
        expert_gpu_dd=expert_gpu_dd,
        expert_gpu_busy=expert_gpu_busy,
        expert_spot_cb=expert_spot_cb,
        expert_gpu_refresh_btn=expert_gpu_refresh_btn,
        base_model=base_model,
        epochs_tf=epochs_tf,
        lr_tf=lr_tf,
        batch_tf=batch_tf,
        grad_acc_tf=grad_acc_tf,
        max_steps_tf=max_steps_tf,
        use_lora_cb=use_lora_cb,
        lora_r_dd=lora_r_dd,
        lora_alpha_tf=lora_alpha_tf,
        lora_dropout_tf=lora_dropout_tf,
        use_rslora_cb=use_rslora_cb,
        lora_hint_caption=lora_hint_caption,
        out_dir_tf=out_dir_tf,
        packing_row=packing_row,
        auto_resume_row=auto_resume_row,
        push_row=push_row,
        hf_repo_row=hf_repo_row,
        resume_from_row=resume_from_row,
        advanced_params_section=advanced_params_section,
        visible=False,
    )

    ds_tp_group_container = ft.Container(
        content=ft.Column(
            [
                dataset_section,
                train_params_section,
            ],
            spacing=12,
        ),
    )

    def _update_mode_visibility(_=None):
        mode = (config_mode_dd.value or "Normal").lower()
        is_cfg = mode.startswith("config")
        try:
            tgt_val = (train_target_dd.value or "Runpod - Pod").lower()
            is_pod_target = tgt_val.startswith("runpod - pod")
        except Exception:
            is_pod_target = True
        try:
            config_files_row.visible = is_cfg
            _update_config_buttons_enabled()
            update_train_placeholders()
            page.update()
        except Exception:
            pass
        try:
            dataset_section.visible = not is_cfg
            train_params_section.visible = not is_cfg
            rp_infra_panel.visible = (not is_cfg) and is_pod_target
            rp_infra_compact_row.visible = is_cfg and is_pod_target
            try:
                ds_tp_group_container.visible = not is_cfg
            except Exception:
                pass
        except Exception:
            pass
        try:
            page.update()
        except Exception:
            pass

    # -------- LOCAL TRAINING: System Specs --------
    # Delegated to helpers.local_specs (gather_local_specs, refresh_local_gpus)

    # Controls to show specs
    local_os_txt = ft.Text("", selectable=True)
    local_py_txt = ft.Text("")
    local_cpu_txt = ft.Text("")
    local_ram_txt = ft.Text("")
    local_disk_txt = ft.Text("")
    local_torch_txt = ft.Text("")
    local_cuda_txt = ft.Text("")
    local_gpus_txt = ft.Text("", selectable=True)
    local_capability_txt = ft.Text("", weight=ft.FontWeight.W_600)
    # Red flags UI (hidden when none)
    local_flags_col = ft.Column([], spacing=2)
    local_flags_box = ft.Column(
        [
            ft.Row(
                [
                    ft.Icon(ICONS.ERROR_OUTLINE, color=getattr(COLORS, "RED_400", COLORS.ERROR), size=18),
                    ft.Text(
                        "Potential issues",
                        weight=ft.FontWeight.W_600,
                        color=getattr(COLORS, "RED_400", COLORS.ERROR),
                    ),
                ],
                spacing=6,
            ),
            local_flags_col,
        ],
        spacing=6,
        visible=False,
    )

    # ---------- LOCAL: Run Training ----------
    # Training run selector (managed storage)
    local_training_run_dd = ft.Dropdown(
        label="Training run",
        options=[],
        width=400,
        tooltip="Select or create a training run for managed storage",
    )
    local_training_run_refresh_btn = ft.IconButton(
        icon=ft.Icons.REFRESH,
        tooltip="Refresh training runs",
    )
    local_new_run_name_tf = ft.TextField(
        label="New run name",
        width=280,
        dense=True,
        hint_text="Name for new training run",
    )
    local_create_run_btn = ft.OutlinedButton(
        "Create Run",
        icon=getattr(ICONS, "ADD", ICONS.CHECK),
    )
    local_run_storage_info = ft.Text(
        "",
        size=11,
        color=WITH_OPACITY(0.7, BORDER_BASE),
    )

    def _refresh_training_runs(_=None):
        """Refresh the training runs dropdown."""
        try:
            from db.training_runs import list_training_runs

            runs = list_training_runs(limit=50)
            options = []
            for r in runs:
                status_icon = {"pending": "⏳", "running": "🔄", "completed": "✅", "failed": "❌"}.get(r["status"], "")
                label = f"{status_icon} {r['name']} ({r['status']}) - {r['created_at'][:10]}"
                options.append(ft.dropdown.Option(key=str(r["id"]), text=label))
            local_training_run_dd.options = options
            if options and not local_training_run_dd.value:
                local_training_run_dd.value = options[0].key
                _update_run_storage_info()
        except Exception as e:
            local_training_run_dd.options = [ft.dropdown.Option(key="", text=f"Error: {e}")]
        try:
            page.update()
        except Exception:
            pass

    def _update_run_storage_info(_=None):
        """Update the storage info text for selected run."""
        try:
            run_id = local_training_run_dd.value
            if run_id:
                from db.training_runs import get_run_storage_paths

                paths = get_run_storage_paths(int(run_id))
                if paths:
                    local_run_storage_info.value = f"Storage: {paths['root']}"
                else:
                    local_run_storage_info.value = ""
            else:
                local_run_storage_info.value = ""
            page.update()
        except Exception:
            pass

    def _create_training_run(_=None):
        """Create a new training run."""
        try:
            name = (local_new_run_name_tf.value or "").strip()
            if not name:
                name = f"run_{int(time.time())}"

            # Get current dataset info
            src = train_source.value or "Database"
            dataset_id = ""
            if src == "Database":
                dataset_id = train_db_session_dd.value or ""
            elif src == "Hugging Face":
                dataset_id = train_hf_repo.value or ""

            model = base_model.value or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

            from db.training_runs import create_training_run

            run = create_training_run(
                name=name,
                base_model=model,
                dataset_source=src,
                dataset_id=dataset_id,
            )

            # Refresh and select the new run
            _refresh_training_runs()
            local_training_run_dd.value = str(run["id"])
            _update_run_storage_info()
            local_new_run_name_tf.value = ""
            page.update()
        except Exception as e:
            local_run_storage_info.value = f"Error creating run: {e}"
            page.update()

    local_training_run_refresh_btn.on_click = _refresh_training_runs
    local_training_run_dd.on_change = _update_run_storage_info
    local_create_run_btn.on_click = _create_training_run

    # Initialize training runs on load
    try:
        _refresh_training_runs()
    except Exception:
        pass

    local_use_gpu_cb = ft.Checkbox(
        label="Use NVIDIA GPU",
        value=False,
    )
    local_pass_hf_token_cb = ft.Checkbox(
        label="Pass HF token to trainer (HF_TOKEN / HUGGINGFACE_HUB_TOKEN)",
    )
    local_train_status = ft.Text("")
    local_save_config_btn = ft.OutlinedButton(
        "Save current setup",
        icon=getattr(ICONS, "SAVE", ICONS.CHECK),
        tooltip=(
            "Save the current training setup (dataset, hyperparameters, target, and local settings) as a reusable config."
        ),
        on_click=lambda e: page.run_task(on_save_current_config),
    )
    local_train_progress = ft.ProgressBar(value=0.0, width=280)
    local_train_prog_label = ft.Text("Idle")
    local_train_timeline = ft.ListView(expand=True, spacing=4, auto_scroll=True)
    local_train_timeline_placeholder = ft.Text(
        "Logs will appear here after starting local training.",
        color=WITH_OPACITY(0.5, BORDER_BASE),
    )

    # Quick local inference controls (initially hidden until a local adapter is available)
    local_infer_status = ft.Text(
        "Quick local inference will be enabled after a successful local training run.",
        color=WITH_OPACITY(0.6, BORDER_BASE),
    )
    local_infer_meta = ft.Text("", size=12, color=WITH_OPACITY(0.65, BORDER_BASE))
    local_infer_prompt_tf = ft.TextField(
        label="Quick local inference prompt",
        multiline=True,
        min_lines=3,
        max_lines=6,
        width=1000,
        dense=True,
    )
    local_infer_expected_tf = ft.TextField(
        label="Expected answer (from dataset)",
        multiline=True,
        min_lines=3,
        max_lines=6,
        width=1000,
        dense=True,
        read_only=True,
        visible=False,
    )
    local_infer_preset_dd = ft.Dropdown(
        label="Preset",
        options=[
            ft.dropdown.Option("Deterministic"),
            ft.dropdown.Option("Balanced"),
            ft.dropdown.Option("Creative"),
        ],
        value="Balanced",
        width=220,
    )
    # Sample prompts dropdown - populated from training dataset
    local_infer_sample_prompts_dd = ft.Dropdown(
        label="Sample prompts from dataset (optional)",
        options=[],
        width=700,
        hint_text="Select a sample prompt or enter your own below",
    )
    local_infer_refresh_samples_btn = ft.IconButton(
        icon=getattr(ICONS, "REFRESH", ICONS.REPLAY),
        tooltip="Get new random samples",
    )
    local_infer_temp_slider = ft.Slider(
        min=0.1,
        max=1.2,
        divisions=11,
        value=0.7,
        width=280,
    )
    local_infer_temp_label = ft.Text("Temperature: 0.7", size=12)
    local_infer_max_tokens_slider = ft.Slider(
        min=64,
        max=512,
        divisions=14,
        value=256,
        width=280,
    )
    local_infer_max_tokens_label = ft.Text("Max tokens: 256", size=12)
    local_infer_rep_penalty_slider = ft.Slider(
        min=1.0,
        max=1.5,
        divisions=10,
        value=1.15,
        width=280,
    )
    local_infer_rep_penalty_label = ft.Text("Rep. penalty: 1.15", size=12)

    def _update_local_infer_slider_labels(e=None):
        try:
            local_infer_temp_label.value = f"Temperature: {local_infer_temp_slider.value:.1f}"
            local_infer_max_tokens_label.value = f"Max tokens: {int(local_infer_max_tokens_slider.value)}"
            local_infer_rep_penalty_label.value = f"Rep. penalty: {local_infer_rep_penalty_slider.value:.2f}"
            page.update()
        except Exception:
            pass

    local_infer_temp_slider.on_change = _update_local_infer_slider_labels
    local_infer_max_tokens_slider.on_change = _update_local_infer_slider_labels
    local_infer_rep_penalty_slider.on_change = _update_local_infer_slider_labels

    local_infer_output = ft.ListView(expand=True, spacing=4, auto_scroll=True)
    local_infer_output_placeholder = ft.Text(
        "Responses will appear here after running inference.",
        color=WITH_OPACITY(0.5, BORDER_BASE),
    )
    local_infer_btn = ft.ElevatedButton(
        "Run Inference",
        icon=getattr(ICONS, "PLAY_CIRCLE", ICONS.PLAY_ARROW),
    )
    local_infer_clear_btn = ft.TextButton(
        "Clear history",
        icon=getattr(ICONS, "DELETE_SWEEP", getattr(ICONS, "DELETE", ICONS.CLOSE)),
    )
    local_infer_export_btn = ft.TextButton(
        "Export chats",
        icon=getattr(ICONS, "DOWNLOAD", getattr(ICONS, "SAVE_ALT", ICONS.SAVE)),
    )
    local_infer_busy_ring = ft.ProgressRing(visible=False, width=24, height=24, stroke_width=3)
    # Buffer for storing chat history (prompt/response pairs)
    local_infer_chat_buffer: List[Dict[str, str]] = []
    local_infer_group_container = ft.Container(
        content=ft.Column(
            [
                section_title(
                    "Quick Local Inference",
                    getattr(ICONS, "PSYCHOLOGY", getattr(ICONS, "CHAT", ICONS.PLAY_CIRCLE)),
                    "Test the latest local adapter with a prompt to verify its behavior.",
                    on_help_click=_mk_help_handler(
                        "Runs the fine-tuned adapter locally so you can sanity-check training results.",
                    ),
                ),
                local_infer_status,
                local_infer_meta,
                local_infer_preset_dd,
                ft.Row(
                    [local_infer_sample_prompts_dd, local_infer_refresh_samples_btn],
                    spacing=4,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                local_infer_prompt_tf,
                local_infer_expected_tf,
                ft.Row(
                    [
                        ft.Column([local_infer_temp_label, local_infer_temp_slider], spacing=2),
                        ft.Column([local_infer_max_tokens_label, local_infer_max_tokens_slider], spacing=2),
                        ft.Column([local_infer_rep_penalty_label, local_infer_rep_penalty_slider], spacing=2),
                    ],
                    wrap=True,
                    spacing=10,
                ),
                ft.Row(
                    [local_infer_btn, local_infer_clear_btn, local_infer_export_btn, local_infer_busy_ring],
                    wrap=True,
                    spacing=10,
                ),
                ft.Container(
                    ft.Stack(
                        [local_infer_output, local_infer_output_placeholder],
                        expand=True,
                    ),
                    height=220,
                    width=1000,
                    border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                    border_radius=8,
                    padding=10,
                ),
            ],
            spacing=8,
        ),
        visible=False,
    )

    # Buffer for saving logs
    local_log_buffer: List[str] = []

    def on_local_infer_clear(e=None):
        try:
            local_infer_output.controls.clear()
            local_infer_chat_buffer.clear()
            try:
                local_infer_output_placeholder.visible = True
            except Exception:
                pass
            local_infer_status.value = "Idle  history cleared."
            page.update()
        except Exception:
            pass

    def on_local_infer_preset_change(e=None):
        try:
            name = (local_infer_preset_dd.value or "Balanced").lower()
        except Exception:
            name = "balanced"
        if name.startswith("deterministic"):
            t = 0.2
            n = 128
            r = 1.2  # Higher penalty for more focused output
        elif name.startswith("creative"):
            t = 1.0
            n = 512
            r = 1.1  # Lower penalty for more variety
        else:
            t = 0.7
            n = 256
            r = 1.15  # Balanced default
        try:
            local_infer_temp_slider.value = t
            local_infer_max_tokens_slider.value = n
            local_infer_rep_penalty_slider.value = r
            # Update labels
            local_infer_temp_label.value = f"Temperature: {t:.1f}"
            local_infer_max_tokens_label.value = f"Max tokens: {int(n)}"
            local_infer_rep_penalty_label.value = f"Rep. penalty: {r:.2f}"
            page.update()
        except Exception:
            pass

    def _refresh_sample_prompts(e=None):
        """Refresh the sample prompts dropdown with random prompts from the training dataset."""
        try:
            local_infer_info = train_state.get("local_infer", {})

            # Check if HuggingFace dataset was used - use cached HF samples if available
            hf_dataset_id = local_infer_info.get("hf_dataset_id")
            if hf_dataset_id:
                hf_samples = local_infer_info.get("hf_samples", [])
                if hf_samples:
                    # Use cached HF samples
                    import random

                    samples_to_show = list(hf_samples)
                    random.shuffle(samples_to_show)
                    samples_to_show = samples_to_show[:5]

                    expected_by_prompt: Dict[str, str] = {}
                    options = []
                    for i, (prompt, expected) in enumerate(samples_to_show):
                        expected_by_prompt[prompt] = expected or ""
                        display_text = prompt[:80] + "..." if len(prompt) > 80 else prompt
                        display_text = display_text.replace("\n", " ")
                        options.append(ft.dropdown.Option(key=prompt, text=f"{i + 1}. {display_text}"))

                    local_infer_sample_prompts_dd.options = options
                    local_infer_sample_prompts_dd.value = None

                    # Store mapping for selection -> expected answer
                    try:
                        li = train_state.get("local_infer")
                        if not isinstance(li, dict):
                            li = {}
                            train_state["local_infer"] = li
                        li["expected_by_prompt"] = expected_by_prompt
                    except Exception:
                        pass

                    try:
                        local_infer_expected_tf.value = ""
                        local_infer_expected_tf.visible = False
                    except Exception:
                        pass
                    page.update()
                    return
                else:
                    # No cached samples available
                    local_infer_sample_prompts_dd.options = [
                        ft.dropdown.Option(key="", text=f"(HF dataset: {hf_dataset_id} - no samples cached)")
                    ]
                    local_infer_sample_prompts_dd.value = None
                    try:
                        local_infer_expected_tf.value = ""
                        local_infer_expected_tf.visible = False
                    except Exception:
                        pass
                    page.update()
                    return

            # Get the dataset session ID from training state
            session_id = local_infer_info.get("dataset_session_id")
            if not session_id:
                # Try to get from current UI selection only if source is Database
                src = train_source.value or "Database"
                if src == "Database":
                    session_id = train_db_session_dd.value
            if not session_id:
                local_infer_sample_prompts_dd.options = []
                local_infer_sample_prompts_dd.value = None
                try:
                    local_infer_expected_tf.value = ""
                    local_infer_expected_tf.visible = False
                except Exception:
                    pass
                page.update()
                return

            # Fetch full (prompt, expected answer) pairs so we can show ground truth
            from db.core import get_connection, init_db

            init_db()
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT input_text, output_text
                FROM scraped_pairs
                WHERE session_id = ?
                ORDER BY RANDOM()
                LIMIT ?
            """,
                (int(session_id), 5),
            )
            rows = cursor.fetchall()
            pairs = [(row["input_text"], row["output_text"]) for row in rows]

            expected_by_prompt: Dict[str, str] = {}
            if pairs:
                # Truncate long prompts for display
                options = []
                for i, (prompt, expected) in enumerate(pairs):
                    expected_by_prompt[prompt] = expected or ""
                    display_text = prompt[:80] + "..." if len(prompt) > 80 else prompt
                    display_text = display_text.replace("\n", " ")
                    options.append(ft.dropdown.Option(key=prompt, text=f"{i + 1}. {display_text}"))
                local_infer_sample_prompts_dd.options = options
                local_infer_sample_prompts_dd.value = None
            else:
                local_infer_sample_prompts_dd.options = []
                local_infer_sample_prompts_dd.value = None

            # Store mapping for selection -> expected answer
            try:
                li = train_state.get("local_infer")
                if not isinstance(li, dict):
                    li = {}
                    train_state["local_infer"] = li
                li["expected_by_prompt"] = expected_by_prompt
            except Exception:
                pass

            # Clear expected answer until a sample is selected
            try:
                local_infer_expected_tf.value = ""
                local_infer_expected_tf.visible = False
            except Exception:
                pass
            page.update()
        except Exception:
            pass

    def _on_sample_prompt_selected(e=None):
        """When a sample prompt is selected, populate the prompt text field."""
        try:
            selected = local_infer_sample_prompts_dd.value
            if selected:
                local_infer_prompt_tf.value = selected
                expected = ""
                try:
                    expected = train_state.get("local_infer", {}).get("expected_by_prompt", {}).get(selected) or ""
                except Exception:
                    expected = ""
                try:
                    local_infer_expected_tf.value = expected
                    local_infer_expected_tf.visible = bool(expected)
                except Exception:
                    pass
            else:
                try:
                    local_infer_expected_tf.value = ""
                    local_infer_expected_tf.visible = False
                except Exception:
                    pass
            page.update()
        except Exception:
            pass

    def _on_local_infer_prompt_changed(e=None):
        """If the user types a custom prompt, clear the sample selection + expected answer."""
        try:
            current = (local_infer_prompt_tf.value or "").strip()
            selected = (local_infer_sample_prompts_dd.value or "").strip()
            if selected and current != selected:
                local_infer_sample_prompts_dd.value = None
                local_infer_expected_tf.value = ""
                local_infer_expected_tf.visible = False
                page.update()
        except Exception:
            pass

    # Wire up sample prompts handlers
    try:
        local_infer_sample_prompts_dd.on_change = _on_sample_prompt_selected
        local_infer_refresh_samples_btn.on_click = _refresh_sample_prompts
        local_infer_prompt_tf.on_change = _on_local_infer_prompt_changed
    except Exception:
        pass

    def _on_save_logs(e):
        try:
            path = getattr(e, "path", None)
            if not path:
                return
            txt = "\n".join(local_log_buffer)
            if not txt.endswith("\n"):
                txt += "\n"
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            local_train_status.value = f"Saved logs to: {path}"
            page.update()
        except Exception as ex:
            local_train_status.value = f"Failed to save logs: {ex}"
            try:
                page.update()
            except Exception:
                pass

    local_logs_picker = ft.FilePicker(on_result=_on_save_logs)
    try:
        page.overlay.append(local_logs_picker)
    except Exception:
        pass

    def _on_save_chats(e):
        """Save chat history to a file."""
        try:
            path = getattr(e, "path", None)
            if not path:
                return
            if not local_infer_chat_buffer:
                local_infer_status.value = "No chats to export."
                page.update()
                return
            # Format as readable text
            lines = []
            for i, chat in enumerate(local_infer_chat_buffer, 1):
                lines.append(f"=== Chat {i} ===")
                lines.append(f"Prompt:\n{chat.get('prompt', '')}")
                lines.append(f"\nResponse:\n{chat.get('response', '')}")
                lines.append("")
            txt = "\n".join(lines)
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            local_infer_status.value = f"Exported {len(local_infer_chat_buffer)} chats to: {path}"
            page.update()
        except Exception as ex:
            local_infer_status.value = f"Failed to export chats: {ex}"
            try:
                page.update()
            except Exception:
                pass

    local_chats_picker = ft.FilePicker(on_result=_on_save_chats)
    try:
        page.overlay.append(local_chats_picker)
    except Exception:
        pass

    local_infer_export_btn.on_click = lambda e: local_chats_picker.save_file(
        dialog_title="Export chat history",
        file_name=f"inference-chats-{int(time.time())}.txt",
        allowed_extensions=["txt", "md"],
    )

    try:
        local_infer_preset_dd.on_change = on_local_infer_preset_change
    except Exception:
        pass
    local_save_logs_btn = ft.OutlinedButton(
        "Download logs",
        icon=getattr(ICONS, "DOWNLOAD", getattr(ICONS, "SAVE_ALT", ICONS.SAVE)),
        on_click=lambda e: local_logs_picker.save_file(
            dialog_title="Save training logs",
            file_name=f"local-train-{int(time.time())}.log",
            allowed_extensions=["txt", "log"],
        ),
        disabled=True,
    )

    # Keep minimal state for local training process
    train_state.setdefault("local", {})

    def _update_local_gpu_default_from_specs():
        try:
            data = gather_local_specs_helper()
            local_use_gpu_cb.value = bool(data.get("cuda_available"))
        except Exception:
            pass

    _update_local_gpu_default_from_specs()

    async def on_start_local_training(e=None):
        try:
            local_view_metrics_btn.disabled = True
        except Exception:
            pass
        await run_local_training_helper(
            page=page,
            train_state=train_state,
            hf_token_tf=hf_token_tf,
            skill_level=skill_level,
            expert_gpu_dd=expert_gpu_dd,
            local_use_gpu_cb=local_use_gpu_cb,
            local_pass_hf_token_cb=local_pass_hf_token_cb,
            proxy_enable_cb=proxy_enable_cb,
            use_env_cb=use_env_cb,
            proxy_url_tf=proxy_url_tf,
            train_source=train_source,
            train_hf_repo=train_hf_repo,
            train_hf_split=train_hf_split,
            train_json_path=train_json_path,
            train_db_session_dd=train_db_session_dd,
            local_training_run_dd=local_training_run_dd,
            local_train_status=local_train_status,
            local_train_progress=local_train_progress,
            local_train_prog_label=local_train_prog_label,
            local_train_timeline=local_train_timeline,
            local_train_timeline_placeholder=local_train_timeline_placeholder,
            local_log_buffer=local_log_buffer,
            local_save_logs_btn=local_save_logs_btn,
            local_start_btn=local_start_btn,
            local_stop_btn=local_stop_btn,
            build_hp_fn=_build_hp,
            ICONS_module=ICONS,
            on_training_complete=_refresh_training_runs,
        )
        try:
            try:
                out_dir = (train_state.get("local") or {}).get("output_dir") or ""
                local_view_metrics_btn.disabled = not (out_dir and os.path.isdir(out_dir))
            except Exception:
                pass
            info = train_state.get("local_infer") or {}
            adapter_path = (info.get("adapter_path") or "").strip()
            base_model_name = (info.get("base_model") or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit").strip()
            if adapter_path and os.path.isdir(adapter_path):
                local_infer_group_container.visible = True
                local_infer_status.value = (
                    "Training finished  Quick Local Inference is ready. "
                    "Enter a prompt below or select a sample from your dataset."
                )
                local_infer_meta.value = f"Adapter: {adapter_path}  Base model: {base_model_name}"
                # Auto-populate sample prompts from the training dataset
                _refresh_sample_prompts()
            else:
                local_infer_status.value = (
                    "Quick local inference not available yet. Ensure training completed successfully."
                )
                local_infer_meta.value = ""
            await safe_update(page)
        except Exception:
            pass

    async def on_view_local_metrics(e=None):
        out_dir = ""
        try:
            out_dir = (train_state.get("local") or {}).get("output_dir") or ""
        except Exception:
            out_dir = ""
        if not out_dir or (not os.path.isdir(out_dir)):
            try:
                page.snack_bar = ft.SnackBar(ft.Text("No local training output directory found."))
                page.snack_bar.open = True
            except Exception:
                pass
            await safe_update(page)
            return

        train_pts, eval_pts, stats = _load_trainer_metrics(out_dir)
        if not train_pts and not eval_pts:
            content_ctl: ft.Control = ft.Text(
                "No training metrics found. Expected trainer_state.json in the output directory.",
                selectable=True,
            )
        else:
            all_pts = train_pts + eval_pts
            xs = [x for x, _y in all_pts]
            ys = [y for _x, y in all_pts]
            min_x = min(xs) if xs else 0
            max_x = max(xs) if xs else 1
            min_y = min(ys) if ys else 0
            max_y = max(ys) if ys else 1
            if min_x == max_x:
                max_x = min_x + 1
            if min_y == max_y:
                max_y = min_y + 1

            try:
                x_interval = float(max(1.0, (max_x - min_x) / 4))
            except Exception:
                x_interval = 1.0
            try:
                y_interval = float(max(1e-9, (max_y - min_y) / 4))
            except Exception:
                y_interval = 1.0

            series: List[ft.LineChartData] = []
            if train_pts:
                series.append(
                    ft.LineChartData(
                        data_points=[ft.LineChartDataPoint(x=x, y=y) for x, y in train_pts],
                        color=WITH_OPACITY(0.9, getattr(COLORS, "BLUE", ACCENT_COLOR)),
                        stroke_width=2,
                        stroke_cap_round=True,
                        point=ft.ChartCirclePoint(
                            color=WITH_OPACITY(0.9, getattr(COLORS, "BLUE", ACCENT_COLOR)),
                            radius=2,
                        ),
                        curved=False,
                    )
                )
            if eval_pts:
                series.append(
                    ft.LineChartData(
                        data_points=[ft.LineChartDataPoint(x=x, y=y) for x, y in eval_pts],
                        color=WITH_OPACITY(0.9, getattr(COLORS, "ORANGE", getattr(COLORS, "AMBER", ACCENT_COLOR))),
                        stroke_width=2,
                        stroke_cap_round=True,
                        point=ft.ChartCirclePoint(
                            color=WITH_OPACITY(0.9, getattr(COLORS, "ORANGE", getattr(COLORS, "AMBER", ACCENT_COLOR))),
                            radius=2,
                        ),
                        curved=False,
                    )
                )

            summary_rows: List[ft.Control] = []
            try:
                if "final_train_loss" in stats:
                    summary_rows.append(ft.Text(f"Final train loss: {stats['final_train_loss']:.4f}"))
                if "min_train_loss" in stats:
                    summary_rows.append(ft.Text(f"Min train loss: {stats['min_train_loss']:.4f}"))
                if "best_eval_loss" in stats:
                    summary_rows.append(ft.Text(f"Best eval loss: {stats['best_eval_loss']:.4f}"))
                if "final_eval_loss" in stats:
                    summary_rows.append(ft.Text(f"Final eval loss: {stats['final_eval_loss']:.4f}"))
                if "max_step" in stats:
                    summary_rows.append(ft.Text(f"Max step: {int(stats['max_step'])}"))
            except Exception:
                pass

            chart = ft.LineChart(
                data_series=series,
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                left_axis=ft.ChartAxis(show_labels=True, labels_interval=y_interval, labels_size=34),
                bottom_axis=ft.ChartAxis(show_labels=True, labels_interval=x_interval, labels_size=28),
                horizontal_grid_lines=ft.ChartGridLines(interval=y_interval),
                vertical_grid_lines=ft.ChartGridLines(interval=x_interval),
                height=320,
                width=900,
            )

            content_ctl = ft.Column(
                [
                    ft.Text(f"Output dir: {out_dir}", size=12, selectable=True),
                    *summary_rows,
                    ft.Divider(),
                    chart,
                    ft.Row(
                        [
                            ft.Container(
                                ft.Row(
                                    [
                                        ft.Container(
                                            width=10,
                                            height=2,
                                            bgcolor=WITH_OPACITY(0.9, getattr(COLORS, "BLUE", ACCENT_COLOR)),
                                        ),
                                        ft.Text("Train loss", size=12),
                                    ],
                                    spacing=6,
                                ),
                                padding=4,
                            ),
                            ft.Container(
                                ft.Row(
                                    [
                                        ft.Container(
                                            width=10,
                                            height=2,
                                            bgcolor=WITH_OPACITY(
                                                0.9, getattr(COLORS, "ORANGE", getattr(COLORS, "AMBER", ACCENT_COLOR))
                                            ),
                                        ),
                                        ft.Text("Eval loss", size=12),
                                    ],
                                    spacing=6,
                                ),
                                padding=4,
                            ),
                        ],
                        spacing=14,
                        wrap=True,
                    ),
                ],
                tight=True,
                spacing=8,
            )

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(
                        getattr(
                            ICONS,
                            "SHOW_CHART",
                            getattr(ICONS, "QUERY_STATS", getattr(ICONS, "INSIGHTS", getattr(ICONS, "INFO", None))),
                        ),
                        color=ACCENT_COLOR,
                    ),
                    ft.Text("Training metrics"),
                ],
                alignment=ft.MainAxisAlignment.START,
            ),
            content=ft.Container(content=content_ctl, width=940),
            actions=[
                ft.TextButton(
                    "Close",
                    on_click=lambda e: (setattr(dlg, "open", False), page.update()),
                )
            ],
        )
        try:
            if hasattr(page, "open") and callable(getattr(page, "open")):
                page.open(dlg)
            else:
                page.dialog = dlg
                dlg.open = True
        except Exception:
            try:
                page.dialog = dlg
                dlg.open = True
            except Exception:
                pass
        await safe_update(page)

    async def on_stop_local_training(e=None):
        return await stop_local_training_helper(
            page=page,
            train_state=train_state,
            local_train_status=local_train_status,
            local_start_btn=local_start_btn,
            local_stop_btn=local_stop_btn,
            local_train_progress=local_train_progress,
            local_train_prog_label=local_train_prog_label,
        )

    async def on_local_infer_generate(e=None):
        prompt = (local_infer_prompt_tf.value or "").strip()
        if not prompt:
            local_infer_status.value = "Enter a prompt to test the latest local adapter."
            await safe_update(page)
            return
        info = train_state.get("local_infer") or {}
        adapter_path = (info.get("adapter_path") or "").strip()
        base_model_name = (info.get("base_model") or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit").strip()
        if (not adapter_path) or (not os.path.isdir(adapter_path)):
            local_infer_status.value = "Quick local inference is not ready. Run a successful local training first."
            await safe_update(page)
            return
        try:
            max_tokens = int(getattr(local_infer_max_tokens_slider, "value", 256) or 256)
        except Exception:
            max_tokens = 256
        try:
            temperature = float(getattr(local_infer_temp_slider, "value", 0.7) or 0.7)
        except Exception:
            temperature = 0.7
        try:
            rep_penalty = float(getattr(local_infer_rep_penalty_slider, "value", 1.15) or 1.15)
        except Exception:
            rep_penalty = 1.15
        if max_tokens <= 0:
            max_tokens = 1
        if temperature <= 0:
            temperature = 0.1
        if rep_penalty < 1.0:
            rep_penalty = 1.0
        loaded = bool(info.get("model_loaded"))
        local_infer_btn.disabled = True
        try:
            local_infer_busy_ring.visible = True
        except Exception:
            pass
        if not loaded:
            local_infer_status.value = "Loading fine-tuned model and generating response..."
        else:
            local_infer_status.value = "Generating response from fine-tuned model..."
        await safe_update(page)
        try:
            text = await asyncio.to_thread(
                local_infer_generate_text_helper,
                base_model_name,
                adapter_path,
                prompt,
                max_tokens,
                temperature,
                rep_penalty,
            )
            try:
                local_infer_output_placeholder.visible = False
            except Exception:
                pass
            local_infer_output.controls.append(
                ft.Column(
                    [
                        ft.Text(
                            "Prompt",
                            weight=getattr(ft.FontWeight, "BOLD", None),
                        ),
                        ft.Text(prompt),
                        ft.Text(
                            "Response",
                            weight=getattr(ft.FontWeight, "BOLD", None),
                        ),
                        ft.Text(text),
                    ],
                    spacing=4,
                )
            )
            # Store in chat buffer for export
            local_infer_chat_buffer.append({"prompt": prompt, "response": text})
            try:
                train_state.setdefault("local_infer", {})
                train_state["local_infer"]["model_loaded"] = True
            except Exception:
                pass
            local_infer_status.value = "Idle  last inference complete."
        except Exception as ex:
            local_infer_status.value = f"Inference failed: {ex}"
        finally:
            local_infer_btn.disabled = False
            try:
                local_infer_busy_ring.visible = False
            except Exception:
                pass
            await safe_update(page)

    local_start_btn = ft.ElevatedButton(
        "Start Local Training",
        icon=getattr(ICONS, "PLAY_CIRCLE", ICONS.PLAY_ARROW),
        on_click=lambda e: page.run_task(on_start_local_training),
    )
    local_stop_btn = ft.OutlinedButton(
        "Stop",
        icon=getattr(ICONS, "STOP_CIRCLE", ICONS.STOP),
        disabled=True,
        on_click=lambda e: page.run_task(on_stop_local_training),
    )
    local_view_metrics_btn.on_click = lambda e: page.run_task(on_view_local_metrics)
    local_infer_btn.on_click = lambda e: page.run_task(on_local_infer_generate)
    local_infer_clear_btn.on_click = on_local_infer_clear

    # Local specs + local training container (delegated layout)
    local_specs_container = build_local_specs_container(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        REFRESH_ICON=REFRESH_ICON,
        local_os_txt=local_os_txt,
        local_py_txt=local_py_txt,
        local_cpu_txt=local_cpu_txt,
        local_ram_txt=local_ram_txt,
        local_disk_txt=local_disk_txt,
        local_torch_txt=local_torch_txt,
        local_cuda_txt=local_cuda_txt,
        local_gpus_txt=local_gpus_txt,
        local_flags_box=local_flags_box,
        local_capability_txt=local_capability_txt,
        refresh_specs_click_cb=lambda e: page.run_task(on_refresh_local_specs),
        local_training_run_dd=local_training_run_dd,
        local_training_run_refresh_btn=local_training_run_refresh_btn,
        local_new_run_name_tf=local_new_run_name_tf,
        local_create_run_btn=local_create_run_btn,
        local_run_storage_info=local_run_storage_info,
        local_use_gpu_cb=local_use_gpu_cb,
        local_pass_hf_token_cb=local_pass_hf_token_cb,
        local_train_progress=local_train_progress,
        local_train_prog_label=local_train_prog_label,
        local_save_logs_btn=local_save_logs_btn,
        local_train_timeline=local_train_timeline,
        local_train_timeline_placeholder=local_train_timeline_placeholder,
        local_start_btn=local_start_btn,
        local_stop_btn=local_stop_btn,
        local_view_metrics_btn=local_view_metrics_btn,
        local_train_status=local_train_status,
        local_infer_group_container=local_infer_group_container,
        local_save_config_btn=local_save_config_btn,
        mk_help_handler=_mk_help_handler,
    )

    # Pod logs section (delegated)
    pod_logs_section = build_pod_logs_section(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        train_progress=train_progress,
        train_prog_label=train_prog_label,
        train_timeline=train_timeline,
        train_timeline_placeholder=train_timeline_placeholder,
        train_status_row=train_status_row,
        mk_help_handler=_mk_help_handler,
    )

    # Wrap the existing Training content so we can hide it for non-pod targets (delegated)
    pod_content_container = build_pod_content_container(
        config_section=config_section,
        rp_infra_panel=rp_infra_panel,
        rp_ds_divider=rp_ds_divider,
        ds_tp_group_container=ds_tp_group_container,
        pod_logs_section=pod_logs_section,
        teardown_section=teardown_section,
        train_actions=train_actions,
    )

    def _collect_local_ui_state() -> dict:
        data: dict = {}
        try:
            data["use_gpu"] = bool(getattr(local_use_gpu_cb, "value", False))
        except Exception:
            data["use_gpu"] = False
        try:
            data["pass_hf_token"] = bool(getattr(local_pass_hf_token_cb, "value", False))
        except Exception:
            data["pass_hf_token"] = False
        return data

    def _build_config_payload_from_ui() -> dict:
        hp = _build_hp() or {}
        try:
            tgt = train_target_dd.value or "Runpod - Pod"
        except Exception:
            tgt = "Runpod - Pod"
        meta = {
            "skill_level": skill_level.value,
            "beginner_mode": beginner_mode_dd.value if (skill_level.value or "") == "Beginner" else "",
            "train_target": tgt,
        }
        try:
            meta["beginner_simple"] = {
                "duration": simple_duration_dd.value,
                "memory": simple_memory_dd.value,
                "quality": simple_quality_dd.value,
            }
        except Exception:
            pass
        payload: dict = {"hp": hp, "meta": meta}
        # Include Runpod infra UI when on Runpod target
        try:
            if (tgt or "").lower().startswith("runpod - pod"):
                payload["infra_ui"] = {
                    "dc": (rp_dc_tf.value or "US-NC-1"),
                    "vol_name": (rp_vol_name_tf.value or "unsloth-volume"),
                    "vol_size": int(float(rp_vol_size_tf.value or "50")),
                    "resize_if_smaller": bool(getattr(rp_resize_cb, "value", True)),
                    "tpl_name": (rp_tpl_name_tf.value or "unsloth-trainer-template"),
                    "image": (rp_image_tf.value or "docker.io/sbussiso/unsloth-trainer:latest"),
                    "container_disk": int(float(rp_container_disk_tf.value or "30")),
                    "pod_volume_gb": int(float(rp_volume_in_gb_tf.value or "0")),
                    "mount_path": (rp_mount_path_tf.value or "/data"),
                    "category": (rp_category_tf.value or "NVIDIA"),
                    "public": bool(getattr(rp_public_cb, "value", False)),
                }
        except Exception:
            pass
        # Always include local UI so configs are reusable for local runs too
        try:
            payload["local_ui"] = _collect_local_ui_state()
        except Exception:
            pass
        return payload

    async def on_save_current_config():
        try:
            payload = _build_config_payload_from_ui()
        except Exception as ex:  # pragma: no cover - defensive
            try:
                page.snack_bar = ft.SnackBar(ft.Text(f"Failed to build config from UI: {ex}"))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return
        try:
            default_name = f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(payload.get('hp', {}).get('base_model', 'model')).replace('/', '_')}.json"
        except Exception:
            default_name = f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        name_tf = ft.TextField(label="Save as", value=default_name, width=420)

        def _do_save(_=None):
            try:
                name = (name_tf.value or default_name).strip()
                if not name:
                    return
                # Remove .json extension if present (database stores by name only)
                if name.lower().endswith(".json"):
                    name = name[:-5]
                from helpers.training_config import save_config

                save_config(name, payload)
                try:
                    set_last_used_config_name_helper(name)
                except Exception:
                    pass
                try:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Saved config: {name}"))
                    page.snack_bar.open = True
                except Exception:
                    pass
                try:
                    _refresh_config_list()
                except Exception:
                    pass
                try:
                    dlg.open = False
                    page.update()
                except Exception:
                    pass
            except Exception as ex:  # pragma: no cover - defensive
                try:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Failed to save config: {ex}"))
                    page.snack_bar.open = True
                except Exception:
                    pass

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(getattr(ICONS, "SAVE_ALT", ICONS.SAVE), color=ACCENT_COLOR),
                    ft.Text("Save current training setup"),
                ],
                alignment=ft.MainAxisAlignment.START,
            ),
            content=ft.Column(
                [
                    ft.Text(
                        "This will snapshot the current dataset, hyperparameters, target, and local/Runpod settings.",
                    ),
                    name_tf,
                ],
                tight=True,
                spacing=6,
            ),
            actions=[
                ft.TextButton(
                    "Cancel",
                    on_click=lambda e: (setattr(dlg, "open", False), page.update()),
                ),
                ft.ElevatedButton(
                    "Save",
                    icon=getattr(ICONS, "SAVE", ICONS.CHECK),
                    on_click=_do_save,
                ),
            ],
        )
        try:
            if hasattr(page, "open") and callable(getattr(page, "open")):
                page.open(dlg)
            else:
                page.dialog = dlg
                dlg.open = True
        except Exception:
            try:
                page.dialog = dlg
                dlg.open = True
            except Exception:
                pass
        await safe_update(page)

    def _apply_config_to_ui(conf: dict) -> None:
        if not isinstance(conf, dict):
            return
        hp = conf.get("hp") or {}
        try:
            base_model.value = hp.get("base_model", base_model.value)
            epochs_tf.value = str(hp.get("epochs", epochs_tf.value))
            lr_tf.value = str(hp.get("lr", lr_tf.value))
            batch_tf.value = str(hp.get("bsz", batch_tf.value))
            grad_acc_tf.value = str(hp.get("grad_accum", grad_acc_tf.value))
            max_steps_tf.value = str(hp.get("max_steps", max_steps_tf.value))
            use_lora_cb.value = bool(hp.get("use_lora", use_lora_cb.value))
            out_dir_tf.value = hp.get("output_dir", out_dir_tf.value)
            # dataset
            if "hf_dataset_id" in hp:
                train_source.value = "Hugging Face"
                train_hf_repo.value = hp.get("hf_dataset_id", "")
                train_hf_split.value = hp.get("hf_dataset_split", "train")
                train_hf_config.value = hp.get("hf_dataset_config", train_hf_config.value)
            elif "db_session_id" in hp:
                train_source.value = "Database"
                # db_session_id will be loaded via refresh
            # Legacy json_path configs default to Database
            elif "json_path" in hp:
                train_source.value = "Database"
            # toggles
            packing_cb.value = bool(hp.get("packing", packing_cb.value))
            auto_resume_cb.value = bool(hp.get("auto_resume", auto_resume_cb.value))
            resume_from_tf.value = hp.get("resume_from", resume_from_tf.value or "")
        except Exception:
            pass
        # reflect meta.skill_level and meta.beginner_mode in UI without overriding HP
        try:
            meta = conf.get("meta") or {}
            skill = str(meta.get("skill_level") or "").strip().lower()
            mode = str(meta.get("beginner_mode") or "").strip().lower()
            if skill in ("beginner", "expert"):
                try:
                    train_state["suppress_skill_defaults"] = True
                except Exception:
                    pass
                try:
                    skill_level.value = "Beginner" if skill == "beginner" else "Expert"
                    if skill == "beginner" and mode in ("fastest", "cheapest", "simple"):
                        # Map legacy values ("Fastest" / "Cheapest") and new
                        # key-based values ("fastest" / "cheapest") onto the
                        # stable keys used by the dropdown.
                        if mode == "fastest":
                            beginner_mode_dd.value = "fastest"
                        elif mode == "simple":
                            beginner_mode_dd.value = "simple"
                        else:
                            beginner_mode_dd.value = "cheapest"
                    try:
                        sc = meta.get("beginner_simple") or {}
                        if isinstance(sc, dict):
                            if sc.get("duration"):
                                simple_duration_dd.value = str(sc.get("duration"))
                            if sc.get("memory"):
                                simple_memory_dd.value = str(sc.get("memory"))
                            if sc.get("quality"):
                                simple_quality_dd.value = str(sc.get("quality"))
                    except Exception:
                        pass
                    _update_skill_controls()
                except Exception:
                    pass
                finally:
                    try:
                        train_state["suppress_skill_defaults"] = False
                    except Exception:
                        pass
            # training target (Runpod vs local)
            try:
                tgt = str(meta.get("train_target") or "").strip()
                if tgt:
                    train_target_dd.value = tgt
                    try:
                        _update_training_target()
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass
        # infra UI
        iu = conf.get("infra_ui") or {}
        try:
            rp_dc_tf.value = iu.get("dc", rp_dc_tf.value)
            rp_vol_name_tf.value = iu.get("vol_name", rp_vol_name_tf.value)
            rp_vol_size_tf.value = str(iu.get("vol_size", rp_vol_size_tf.value))
            rp_resize_cb.value = bool(iu.get("resize_if_smaller", rp_resize_cb.value))
            rp_tpl_name_tf.value = iu.get("tpl_name", rp_tpl_name_tf.value)
            rp_image_tf.value = iu.get("image", rp_image_tf.value)
            rp_container_disk_tf.value = str(iu.get("container_disk", rp_container_disk_tf.value))
            rp_volume_in_gb_tf.value = str(iu.get("pod_volume_gb", rp_volume_in_gb_tf.value))
            rp_mount_path_tf.value = iu.get("mount_path", rp_mount_path_tf.value)
            rp_category_tf.value = iu.get("category", rp_category_tf.value)
            rp_public_cb.value = bool(iu.get("public", rp_public_cb.value))
        except Exception:
            pass
        # local UI (local training settings)
        try:
            lu = conf.get("local_ui") or {}
            try:
                local_use_gpu_cb.value = bool(lu.get("use_gpu", getattr(local_use_gpu_cb, "value", False)))
            except Exception:
                pass
            try:
                local_pass_hf_token_cb.value = bool(
                    lu.get("pass_hf_token", getattr(local_pass_hf_token_cb, "value", False))
                )
            except Exception:
                pass
        except Exception:
            pass
        # summary
        try:
            m = hp.get("base_model", "")
            ds = hp.get("hf_dataset_id") or hp.get("json_path") or ""
            config_summary_txt.value = f"Model: {m} • Dataset: {ds}"
        except Exception:
            pass
        try:
            _update_train_source()
            page.update()
        except Exception:
            pass

    async def on_load_config():
        name = (config_files_dd.value or "").strip()
        if not name:
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Select a config to load."))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return
        # Remove .json extension if present (database stores by name only)
        config_name = name[:-5] if name.lower().endswith(".json") else name
        from helpers.training_config import read_json_file

        conf = read_json_file(config_name)
        if not isinstance(conf, dict):
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Failed to load config: invalid JSON or unreadable file."))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return
        ok, msg = _validate_config(conf)
        if not ok:
            try:
                page.snack_bar = ft.SnackBar(ft.Text(f"Invalid config: {msg}"))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return
        # Persist and apply
        train_state["loaded_config"] = conf
        try:
            train_state["loaded_config_name"] = name
        except Exception:
            pass
        try:
            set_last_used_config_name_helper(name)
        except Exception:
            pass
        _apply_config_to_ui(conf)
        try:
            page.snack_bar = ft.SnackBar(ft.Text(f"Loaded config: {name}"))
            page.snack_bar.open = True
            await safe_update(page)
        except Exception:
            pass

    async def on_rename_config():
        name = (config_files_dd.value or "").strip()
        if not name:
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Select a config to rename."))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return
        new_tf = ft.TextField(label="New name", value=name, width=420)

        async def _do_rename(_=None):
            try:
                new_name = (new_tf.value or name).strip()
                if not new_name:
                    return
                # Remove .json extension if present (database stores by name only)
                old_name = name[:-5] if name.lower().endswith(".json") else name
                new_name = new_name[:-5] if new_name.lower().endswith(".json") else new_name
                # No-op if unchanged
                if old_name.lower() == new_name.lower():
                    dlg.open = False
                    await safe_update(page)
                    return
                from helpers.training_config import rename_config

                success, msg = rename_config(old_name, new_name)
                if not success:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Rename failed: {msg}"))
                    page.snack_bar.open = True
                    await safe_update(page)
                    return
                _refresh_config_list()
                try:
                    config_files_dd.value = new_name
                except Exception:
                    pass
                page.snack_bar = ft.SnackBar(ft.Text(f"Renamed to: {new_name}"))
                page.snack_bar.open = True
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Rename failed: {ex}"))
                page.snack_bar.open = True
            finally:
                try:
                    dlg.open = False
                    await safe_update(page)
                except Exception:
                    pass

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(
                        getattr(ICONS, "DRIVE_FILE_RENAME_OUTLINE", getattr(ICONS, "EDIT", ICONS.SETTINGS)),
                        color=ACCENT_COLOR,
                    ),
                    ft.Text("Rename configuration"),
                ],
                alignment=ft.MainAxisAlignment.START,
            ),
            content=ft.Column(
                [ft.Text("Choose a new filename (JSON extension optional)."), new_tf],
                tight=True,
                spacing=6,
            ),
            actions=[
                ft.TextButton(
                    "Cancel",
                    on_click=lambda e: (setattr(dlg, "open", False), page.update()),
                ),
                ft.ElevatedButton(
                    "Rename",
                    icon=getattr(ICONS, "CHECK", ICONS.SAVE),
                    on_click=lambda e: page.run_task(_do_rename),
                ),
            ],
        )
        page.dialog = dlg
        dlg.open = True
        await safe_update(page)

    async def on_edit_config():
        name = (config_files_dd.value or "").strip()
        if not name:
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Select a config to edit."))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return
        # Remove .json extension if present (database stores by name only)
        config_name = name[:-5] if name.lower().endswith(".json") else name
        from helpers.training_config import read_json_file

        conf = read_json_file(config_name) or {}
        try:
            raw_text = json.dumps(conf, indent=2, ensure_ascii=False)
        except Exception:
            raw_text = "{}"

        editor_tf = ft.TextField(
            label=f"Editing: {name}",
            value=raw_text,
            multiline=True,
            max_lines=24,
            width=900,
            tooltip="Edit JSON configuration. Use Save to validate and write changes.",
        )
        status_txt = ft.Text("", size=12, color=WITH_OPACITY(0.8, BORDER_BASE))

        async def _validate_only(_=None):
            try:
                data = json.loads(editor_tf.value or "{}")
            except Exception as ex:
                status_txt.value = f"JSON error: {ex}"
                try:
                    status_txt.color = COLORS.RED
                except Exception:
                    pass
                await safe_update(page)
                return
            ok, msg = _validate_config(data)
            if ok:
                status_txt.value = f"Valid  {msg or ''}"
                try:
                    status_txt.color = getattr(COLORS, "GREEN", COLORS.SECONDARY)
                except Exception:
                    pass
            else:
                status_txt.value = f"Invalid: {msg}"
                try:
                    status_txt.color = COLORS.RED
                except Exception:
                    pass
            await safe_update(page)

        async def _save_edits(_=None):
            try:
                data = json.loads(editor_tf.value or "{}")
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"JSON error: {ex}"))
                page.snack_bar.open = True
                await safe_update(page)
                return
            ok, msg = _validate_config(data)
            if not ok:
                page.snack_bar = ft.SnackBar(ft.Text(f"Invalid config: {msg}"))
                page.snack_bar.open = True
                await safe_update(page)
                return
            try:
                from helpers.training_config import save_config

                save_config(config_name, data)
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Save failed: {ex}"))
                page.snack_bar.open = True
                await safe_update(page)
                return
            # Refresh list and update loaded config/UI if applicable
            try:
                _refresh_config_list()
            except Exception:
                pass
            try:
                if (train_state.get("loaded_config_name") or "") == name:
                    train_state["loaded_config"] = data
                    _apply_config_to_ui(data)
            except Exception:
                pass
            page.snack_bar = ft.SnackBar(ft.Text(f"Saved: {name}"))
            page.snack_bar.open = True
            try:
                edit_dlg.open = False
                await safe_update(page)
            except Exception:
                pass

        edit_dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(getattr(ICONS, "EDIT", getattr(ICONS, "MODE_EDIT", ICONS.SETTINGS)), color=ACCENT_COLOR),
                    ft.Text("Edit configuration"),
                ],
                alignment=ft.MainAxisAlignment.START,
            ),
            content=ft.Column(
                [
                    ft.Text("Update the JSON below, then click Save. Use Validate to check without saving."),
                    editor_tf,
                    status_txt,
                ],
                tight=True,
                spacing=8,
            ),
            actions=[
                ft.TextButton(
                    "Cancel",
                    on_click=lambda e: (setattr(edit_dlg, "open", False), page.update()),
                ),
                ft.OutlinedButton(
                    "Validate",
                    icon=getattr(ICONS, "CHECK_CIRCLE", ICONS.CHECK),
                    on_click=lambda e: page.run_task(_validate_only),
                ),
                ft.ElevatedButton(
                    "Save",
                    icon=getattr(ICONS, "SAVE", ICONS.CHECK),
                    on_click=lambda e: page.run_task(_save_edits),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        # Open dialog using the resilient pattern
        opened = False
        try:
            if hasattr(page, "open") and callable(getattr(page, "open")):
                page.open(edit_dlg)
                opened = True
        except Exception:
            opened = False
        if not opened:
            page.dialog = edit_dlg
            edit_dlg.open = True
        await safe_update(page)

    async def on_delete_config():
        name = (config_files_dd.value or "").strip()
        if not name:
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Select a config to delete."))
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return
        confirm_dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(getattr(ICONS, "DELETE", getattr(ICONS, "DELETE_FOREVER", ICONS.CLOSE)), color=COLORS.RED),
                    ft.Text("Delete configuration?"),
                ],
                alignment=ft.MainAxisAlignment.START,
            ),
            content=ft.Text(f"This will permanently delete '{name}'."),
            actions=[
                ft.TextButton(
                    "Cancel",
                    on_click=lambda e: (setattr(confirm_dlg, "open", False), page.update()),
                ),
                ft.ElevatedButton(
                    "Delete",
                    icon=getattr(ICONS, "CHECK", ICONS.DELETE),
                    on_click=lambda e: page.run_task(_do_delete),
                ),
            ],
        )
        # Open confirm dialog using same pattern as dataset previews
        opened = False
        try:
            if hasattr(page, "open") and callable(getattr(page, "open")):
                page.open(confirm_dlg)
                opened = True
        except Exception:
            opened = False
        if not opened:
            page.dialog = confirm_dlg
            confirm_dlg.open = True
        await safe_update(page)

        async def _do_delete(_=None):
            try:
                from helpers.training_config import delete_config

                # Remove .json extension if present
                config_name = name[:-5] if name.lower().endswith(".json") else name
                delete_config(config_name)
                _refresh_config_list()
                try:
                    if (train_state.get("loaded_config_name") or "") == name:
                        train_state["loaded_config_name"] = ""
                        train_state["loaded_config"] = {}
                        config_summary_txt.value = ""
                except Exception:
                    pass
                page.snack_bar = ft.SnackBar(ft.Text(f"Deleted: {config_name}"))
                page.snack_bar.open = True
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Delete failed: {ex}"))
                page.snack_bar.open = True
            finally:
                try:
                    confirm_dlg.open = False
                    await safe_update(page)
                except Exception:
                    pass

    # Hook handlers and initialize config list (mirrored from main.py)
    config_mode_dd.on_change = _update_mode_visibility
    config_files_dd.on_change = _update_config_buttons_enabled
    config_refresh_btn.on_click = _refresh_config_list
    load_config_btn.on_click = lambda e: page.run_task(on_load_config)
    config_save_current_btn.on_click = lambda e: page.run_task(on_save_current_config)
    config_edit_btn.on_click = lambda e: page.run_task(on_edit_config)
    config_rename_btn.on_click = lambda e: page.run_task(on_rename_config)
    config_delete_btn.on_click = lambda e: page.run_task(on_delete_config)

    # Auto-load last used config on startup if available, and set training target from it
    try:
        last_name = get_last_used_config_name_helper()
    except Exception:
        last_name = None
    if last_name:
        conf_for_last: dict = {}
        try:
            # Remove .json extension if present
            config_name = last_name[:-5] if last_name.lower().endswith(".json") else last_name
            conf_for_last = _read_json_file(config_name) or {}
        except Exception:
            conf_for_last = {}
        try:
            meta = conf_for_last.get("meta") or {}
        except Exception:
            meta = {}
        try:
            tgt = str(meta.get("train_target") or "").strip() or "Runpod - Pod"
        except Exception:
            tgt = "Runpod - Pod"
        try:
            train_target_dd.value = tgt
        except Exception:
            pass
        try:
            _refresh_config_list()
        except Exception:
            pass
        try:
            opts = getattr(config_files_dd, "options", []) or []
            keys = {getattr(o, "key", None) or getattr(o, "text", "") for o in opts}
            if last_name in keys:
                config_files_dd.value = last_name
                _schedule_task(page, on_load_config)
        except Exception:
            pass
    else:
        try:
            _refresh_config_list()
        except Exception:
            pass
    # Ensure initial visibility matches the selected mode
    try:
        _update_mode_visibility()
    except Exception:
        pass

    # Per-target HP profiles (Runpod vs local), mirrored from main.py
    train_target_profiles: Dict[str, Dict[str, Any]] = {"runpod": {}, "local": {}}
    train_target_state: Dict[str, str] = {"current": "runpod"}

    def _target_key_from_value(val: str) -> str:
        v = (val or "").lower()
        if v.startswith("runpod - pod"):
            return "runpod"
        return "local"

    def _snapshot_hp_for_target(target_key: str):
        prof = train_target_profiles.get(target_key)
        if prof is None:
            prof = {}
            train_target_profiles[target_key] = prof
        prof["train_source"] = train_source.value
        prof["train_hf_repo"] = train_hf_repo.value
        prof["train_hf_split"] = train_hf_split.value
        prof["train_hf_config"] = train_hf_config.value
        prof["train_json_path"] = train_json_path.value
        prof["base_model"] = base_model.value
        prof["epochs"] = epochs_tf.value
        prof["lr"] = lr_tf.value
        prof["batch"] = batch_tf.value
        prof["grad_acc"] = grad_acc_tf.value
        prof["max_steps"] = max_steps_tf.value
        prof["use_lora"] = bool(getattr(use_lora_cb, "value", False))
        prof["out_dir"] = out_dir_tf.value
        prof["packing"] = bool(getattr(packing_cb, "value", False))
        prof["auto_resume"] = bool(getattr(auto_resume_cb, "value", False))
        prof["resume_from"] = resume_from_tf.value
        prof["warmup_steps"] = warmup_steps_tf.value
        prof["weight_decay"] = weight_decay_tf.value
        prof["lr_sched"] = lr_sched_dd.value
        prof["optim"] = optim_dd.value
        prof["logging_steps"] = logging_steps_tf.value
        prof["logging_first_step"] = bool(getattr(logging_first_step_cb, "value", False))
        prof["disable_tqdm"] = bool(getattr(disable_tqdm_cb, "value", False))
        prof["seed"] = seed_tf.value
        prof["save_strategy"] = save_strategy_dd.value
        prof["save_total_limit"] = save_total_limit_tf.value
        prof["pin_memory"] = bool(getattr(pin_memory_cb, "value", False))
        prof["report_to"] = report_to_dd.value
        prof["fp16"] = bool(getattr(fp16_cb, "value", False))
        prof["bf16"] = bool(getattr(bf16_cb, "value", False))

    def _update_beginner_mode_labels_for_target() -> None:
        """Adjust Beginner preset labels based on Training target.

        Under the hood, values stay as keys ("fastest" / "cheapest"); only
        the visible text and tooltip change between Runpod and local.
        """
        try:
            tgt = (train_target_dd.value or "Runpod - Pod").lower()
        except Exception:
            tgt = "runpod - pod"
        try:
            opts = list(getattr(beginner_mode_dd, "options", []) or [])
        except Exception:
            opts = []
        for o in opts:
            try:
                key = (getattr(o, "key", None) or getattr(o, "text", "")).strip().lower()
            except Exception:
                key = ""
            # Keys stay as stable preset identifiers; only user-facing labels change
            if tgt.startswith("runpod - pod"):
                if key == "fastest":
                    o.text = "Fastest (Runpod)"
                elif key == "cheapest":
                    o.text = "Cheapest (Runpod)"
            else:
                if key == "fastest":
                    o.text = "Quick local test"
                elif key == "cheapest":
                    o.text = "Auto Set (local)"
        try:
            beginner_mode_dd.options = opts
        except Exception:
            pass
        try:
            if tgt.startswith("runpod - pod"):
                beginner_mode_dd.tooltip = (
                    "Beginner presets for Runpod. Fastest favors throughput on higher-tier GPUs; "
                    "Cheapest favors lower-cost GPUs with more conservative params."
                )
            else:
                beginner_mode_dd.tooltip = (
                    "Beginner presets for local training. Quick local test runs a short, small-batch "
                    "experiment; Auto Set (local) uses your detected GPU VRAM to choose batch/grad_acc/"
                    "max_steps that push the system without being overly aggressive."
                )
        except Exception:
            pass
        try:
            page.update()
        except Exception:
            pass

    def _apply_hp_for_target(target_key: str):
        prof = train_target_profiles.get(target_key) or {}
        if (not prof) and (target_key == "local"):
            base_prof = train_target_profiles.get("runpod") or {}
            prof = dict(base_prof)
            out_dir_val = (prof.get("out_dir") or "").strip()
            if (not out_dir_val) or (out_dir_val == "/data/outputs/runpod_run"):
                prof["out_dir"] = "/data/outputs/local_run"
            train_target_profiles["local"] = prof
        v = prof.get("train_source")
        if v is not None:
            train_source.value = v
        v = prof.get("train_hf_repo")
        if v is not None:
            train_hf_repo.value = v
        v = prof.get("train_hf_split")
        if v is not None:
            train_hf_split.value = v
        v = prof.get("train_hf_config")
        if v is not None:
            train_hf_config.value = v
        v = prof.get("train_json_path")
        if v is not None:
            train_json_path.value = v
        v = prof.get("base_model")
        if v is not None:
            base_model.value = v
        v = prof.get("epochs")
        if v is not None:
            epochs_tf.value = v
        v = prof.get("lr")
        if v is not None:
            lr_tf.value = v
        v = prof.get("batch")
        if v is not None:
            batch_tf.value = v
        v = prof.get("grad_acc")
        if v is not None:
            grad_acc_tf.value = v
        v = prof.get("max_steps")
        if v is not None:
            max_steps_tf.value = v
        if "use_lora" in prof:
            use_lora_cb.value = bool(prof.get("use_lora"))
        v = prof.get("out_dir")
        if v is not None:
            out_dir_tf.value = v
        if "packing" in prof:
            packing_cb.value = bool(prof.get("packing"))
        if "auto_resume" in prof:
            auto_resume_cb.value = bool(prof.get("auto_resume"))
        v = prof.get("resume_from")
        if v is not None:
            resume_from_tf.value = v
        v = prof.get("warmup_steps")
        if v is not None:
            warmup_steps_tf.value = v
        v = prof.get("weight_decay")
        if v is not None:
            weight_decay_tf.value = v
        v = prof.get("lr_sched")
        if v is not None:
            lr_sched_dd.value = v
        v = prof.get("optim")
        if v is not None:
            optim_dd.value = v
        v = prof.get("logging_steps")
        if v is not None:
            logging_steps_tf.value = v
        if "logging_first_step" in prof:
            logging_first_step_cb.value = bool(prof.get("logging_first_step"))
        if "disable_tqdm" in prof:
            disable_tqdm_cb.value = bool(prof.get("disable_tqdm"))
        v = prof.get("seed")
        if v is not None:
            seed_tf.value = v
        v = prof.get("save_strategy")
        if v is not None:
            save_strategy_dd.value = v
        v = prof.get("save_total_limit")
        if v is not None:
            save_total_limit_tf.value = v
        if "pin_memory" in prof:
            pin_memory_cb.value = bool(prof.get("pin_memory"))
        v = prof.get("report_to")
        if v is not None:
            report_to_dd.value = v
        if "fp16" in prof:
            fp16_cb.value = bool(prof.get("fp16"))
        if "bf16" in prof:
            bf16_cb.value = bool(prof.get("bf16"))

    def _update_training_target(_=None):
        """Switch between Runpod and local training targets.

        Includes HP snapshot/apply, config refresh, layout visibility
        toggles, and GPU/spec refresh behavior mirrored from `main.py`.
        """
        val = train_target_dd.value or "Runpod - Pod"
        try:
            new_key = _target_key_from_value(val)
            prev_key = train_target_state.get("current") or "runpod"
            if new_key != prev_key:
                try:
                    _snapshot_hp_for_target(prev_key)
                except Exception:
                    pass
                try:
                    _apply_hp_for_target(new_key)
                except Exception:
                    pass
                train_target_state["current"] = new_key
        except Exception:
            pass
        # Refresh config list and Beginner presets when target changes so
        # Runpod/local configs are clearly separated.
        try:
            _refresh_config_list()
        except Exception:
            pass
        try:
            _update_beginner_mode_labels_for_target()
        except Exception:
            pass
        target = (val or "").lower()
        is_pod = target.startswith("runpod - pod")
        try:
            # Always show the wrapper; toggle inner sections as needed
            pod_content_container.visible = True
            # Show Local Specs when local
            local_specs_container.visible = not is_pod
            # Toggle Runpod-only panels and pod log/actions
            try:
                rp_infra_panel.visible = is_pod
            except Exception:
                pass
            try:
                rp_ds_divider.visible = is_pod
            except Exception:
                pass
            try:
                pod_logs_section.visible = is_pod
            except Exception:
                pass
            try:
                train_progress.visible = is_pod
                train_prog_label.visible = is_pod
                train_timeline.visible = is_pod
                train_timeline_placeholder.visible = is_pod
            except Exception:
                pass
            try:
                teardown_section.visible = is_pod
            except Exception:
                pass
            try:
                train_actions.visible = is_pod
            except Exception:
                pass
            # Configuration section is available for managing configs; keep it visible for all targets
            try:
                config_section.visible = True
            except Exception:
                pass
            # Recompose the Dataset + Training Params grouping depending on target
            try:
                if is_pod:
                    ds_tp_group_container.content = ft.Column(
                        [
                            section_title(
                                "Runpod: Dataset & Params",
                                getattr(
                                    ICONS,
                                    "LIST_ALT",
                                    getattr(ICONS, "DESCRIPTION", ICONS.SETTINGS),
                                ),
                                "Choose dataset and configure training parameters for Runpod pods.",
                                on_help_click=_mk_help_handler(
                                    "Choose dataset source and configure training parameters for Runpod pod training.",
                                ),
                            ),
                            ft.Container(
                                content=ft.Column([dataset_section, train_params_section], spacing=0),
                                width=1000,
                                border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                                border_radius=8,
                                padding=10,
                            ),
                        ],
                        spacing=12,
                    )
                else:
                    ds_tp_group_container.content = ft.Column(
                        [
                            section_title(
                                "Local: Dataset & Params",
                                getattr(
                                    ICONS,
                                    "LIST_ALT",
                                    getattr(ICONS, "DESCRIPTION", ICONS.SETTINGS),
                                ),
                                "Choose dataset and configure training parameters for local runs.",
                                on_help_click=_mk_help_handler(
                                    "Choose dataset source and configure training parameters for local runs.",
                                ),
                            ),
                            ft.Container(
                                content=ft.Column([dataset_section, train_params_section], spacing=0),
                                width=1000,
                                border=ft.border.all(1, WITH_OPACITY(0.1, BORDER_BASE)),
                                border_radius=8,
                                padding=10,
                            ),
                        ],
                        spacing=12,
                    )
            except Exception:
                pass
        except Exception:
            pass
        # Ensure normal/config mode visibility still applies for both targets
        try:
            _update_mode_visibility()
        except Exception:
            pass
        if is_pod:
            # Restore Runpod spot visibility depending on skill
            try:
                expert_spot_cb.disabled = False
                expert_spot_cb.visible = (skill_level.value or "Beginner").lower() != "beginner"
            except Exception:
                pass
            # If switching back to Runpod and expert GPU list looks local/unpopulated, refresh Runpod GPUs
            try:
                is_beginner = (skill_level.value or "Beginner").lower() == "beginner"
                if not is_beginner:
                    opts = getattr(expert_gpu_dd, "options", []) or []
                    dd_tip = (getattr(expert_gpu_dd, "tooltip", "") or "").lower()
                    if (len(opts) <= 1) or ("local" in dd_tip):
                        if hasattr(page, "run_task"):
                            page.run_task(refresh_expert_gpus)
            except Exception:
                pass
        else:
            # When switching to local, refresh specs
            try:
                if hasattr(page, "run_task"):
                    page.run_task(on_refresh_local_specs)
            except Exception:
                pass
            # Local target always uses Normal semantics; force dataset/params visible
            try:
                dataset_section.visible = True
                train_params_section.visible = True
                ds_tp_group_container.visible = True
            except Exception:
                pass
            # Hide Runpod-only spot toggle; use local GPUs for expert picker
            try:
                expert_spot_cb.value = False
                expert_spot_cb.disabled = True
                expert_spot_cb.visible = False
            except Exception:
                pass
            # Show/hide local GPU checkbox based on skill (beginner uses checkbox, expert uses dropdown)
            try:
                is_beginner = (skill_level.value or "Beginner").lower() == "beginner"
                local_use_gpu_cb.visible = is_beginner
            except Exception:
                pass
            # If expert mode and dropdown not populated with locals yet, populate
            try:
                if (skill_level.value or "Beginner").lower() != "beginner" and len(
                    (getattr(expert_gpu_dd, "options", []) or [])
                ) <= 1:
                    if hasattr(page, "run_task"):
                        page.run_task(refresh_local_gpus)
            except Exception:
                pass
        try:
            page.update()
        except Exception:
            pass

    # Wire target dropdown to updater and initialize visibility
    train_target_dd.on_change = _update_training_target
    try:
        _update_training_target()
    except Exception:
        pass

    # Build Training tab layout
    offline_banner = build_offline_banner(
        [
            "Hugging Face datasets and Hub push are disabled.",
            "Runpod cloud training is disabled (Local only).",
        ]
    )
    train_tab = build_training_tab(
        section_title=section_title,
        ICONS=ICONS,
        BORDER_BASE=BORDER_BASE,
        WITH_OPACITY=WITH_OPACITY,
        offline_banner=offline_banner,
        train_target_dd=train_target_dd,
        train_target_offline_reason=train_target_offline_reason,
        pod_content_container=pod_content_container,
        local_specs_container=local_specs_container,
    )

    # Hook: respond to Offline Mode changes (disable Runpod cloud actions and
    # non-local dataset sources when offline)
    def apply_offline_mode_to_training(_=None):
        try:
            is_offline = bool(getattr(offline_mode_sw, "value", False))
        except Exception:
            is_offline = False

        try:
            offline_banner.visible = is_offline
        except Exception:
            pass

        try:
            train_target_offline_reason.visible = is_offline
        except Exception:
            pass

        try:
            train_source_offline_reason.visible = is_offline
        except Exception:
            pass

        # When offline, force local target selection and update layout
        try:
            if is_offline:
                train_target_dd.value = "local"
            _update_training_target()
        except Exception:
            pass

        # When offline, keep Hugging Face dataset source visible but disabled
        try:
            opts = list(getattr(train_source, "options", []) or [])
            for opt in opts:
                try:
                    label = str(getattr(opt, "key", getattr(opt, "text", "")) or "").strip()
                except Exception:
                    label = ""
                try:
                    if label == "Hugging Face":
                        opt.disabled = is_offline
                    else:
                        opt.disabled = False
                except Exception:
                    pass
            try:
                train_source.options = opts
            except Exception:
                pass

            # If currently pointing at Hugging Face while offline, reset to Database
            cur_src = (train_source.value or "Database").strip()
            if is_offline and cur_src == "Hugging Face":
                train_source.value = "Database"
                try:
                    _update_train_source()
                except Exception:
                    pass
        except Exception:
            pass

        # When offline, keep Runpod target visible but disabled so only local
        # training can be selected.
        try:
            tgt_opts = list(getattr(train_target_dd, "options", []) or [])
            for opt in tgt_opts:
                try:
                    label = str(getattr(opt, "key", getattr(opt, "text", "")) or "").strip()
                except Exception:
                    label = ""
                try:
                    if label.lower().startswith("runpod - pod"):
                        opt.disabled = is_offline
                    else:
                        opt.disabled = False
                except Exception:
                    pass
            try:
                train_target_dd.options = tgt_opts
            except Exception:
                pass
        except Exception:
            pass

        # Disable Runpod infra + Runpod control buttons when offline
        try:
            ensure_infra_btn.disabled = is_offline or (rp_infra is None)
        except Exception:
            pass
        for ctl in [open_runpod_btn, open_web_terminal_btn, restart_container_btn, copy_ssh_btn]:
            try:
                ctl.disabled = is_offline
            except Exception:
                pass

        # Disable passing HF token into the local training process when offline
        try:
            local_pass_hf_token_cb.disabled = is_offline
            if is_offline:
                local_pass_hf_token_cb.value = False
        except Exception:
            pass

        try:
            page.update()
        except Exception:
            pass

    # Register hook with the shared Offline Mode switch
    try:
        hooks = getattr(offline_mode_sw, "data", None)
        if hooks is None:
            hooks = {}
        hooks["training_tab_offline"] = lambda e=None: apply_offline_mode_to_training(e)
        offline_mode_sw.data = hooks
    except Exception:
        pass

    # Apply current offline state on first load
    apply_offline_mode_to_training()

    return train_tab, train_state
