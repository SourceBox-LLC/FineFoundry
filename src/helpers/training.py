from __future__ import annotations

import asyncio
import getpass
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Callable, List

try:
    import grp  # type: ignore
except Exception:  # pragma: no cover
    grp = None

try:
    import pwd  # type: ignore
except Exception:  # pragma: no cover
    pwd = None

import flet as ft

from helpers.common import safe_update
from helpers.theme import COLORS, ACCENT_COLOR
from helpers.ui import WITH_OPACITY

# Optional datasets dependency for HF sample fetching
try:
    from datasets import load_dataset as hf_load_dataset
except Exception:
    hf_load_dataset = None


async def _fetch_hf_samples(repo_id: str, split: str, num_samples: int = 10):
    """Fetch random sample prompts from a HuggingFace dataset.

    Returns list of (prompt, expected_answer) tuples.
    """
    import asyncio
    import random

    if hf_load_dataset is None:
        return []

    def _load():
        try:
            ds = hf_load_dataset(repo_id, split=split, trust_remote_code=True)
            if len(ds) == 0:
                return []

            # Get random indices
            indices = list(range(len(ds)))
            random.shuffle(indices)
            indices = indices[:num_samples]

            samples = []
            for idx in indices:
                row = ds[idx]
                prompt = ""
                expected = ""

                # Try to extract prompt/response from common column formats
                # ChatML format (messages column)
                if "messages" in row:
                    msgs = row["messages"]
                    if isinstance(msgs, list) and len(msgs) >= 2:
                        for m in msgs:
                            if isinstance(m, dict):
                                role = m.get("role", "")
                                content = m.get("content", "")
                                if role in ("user", "human"):
                                    prompt = content
                                elif role in ("assistant", "gpt", "bot"):
                                    expected = content
                # Alpaca format
                elif "instruction" in row:
                    prompt = row.get("instruction", "") or row.get("input", "")
                    if row.get("input"):
                        prompt = f"{row.get('instruction', '')}\n{row.get('input', '')}".strip()
                    expected = row.get("output", "") or row.get("response", "")
                # Simple Q&A format
                elif "question" in row:
                    prompt = row.get("question", "")
                    expected = row.get("answer", "") or row.get("response", "")
                # Input/output format
                elif "input" in row:
                    prompt = row.get("input", "")
                    expected = row.get("output", "") or row.get("response", "")
                # Text/label format
                elif "text" in row:
                    prompt = row.get("text", "")
                    expected = row.get("label", "") or ""
                # Prompt/completion format
                elif "prompt" in row:
                    prompt = row.get("prompt", "")
                    expected = row.get("completion", "") or row.get("response", "")

                if prompt:
                    samples.append((str(prompt).strip(), str(expected).strip() if expected else ""))

            return samples
        except Exception:
            return []

    return await asyncio.to_thread(_load)


def build_hp_from_controls(
    *,
    train_source: ft.Dropdown,
    train_hf_repo: ft.TextField,
    train_hf_split: ft.Dropdown,
    train_hf_config: ft.TextField,
    train_json_path: ft.TextField,
    train_db_session_dd: ft.Dropdown,
    base_model: ft.TextField,
    out_dir_tf: ft.TextField,
    epochs_tf: ft.TextField,
    lr_tf: ft.TextField,
    batch_tf: ft.TextField,
    grad_acc_tf: ft.TextField,
    max_steps_tf: ft.TextField,
    use_lora_cb: ft.Checkbox,
    lora_r_dd: ft.Dropdown,
    lora_alpha_tf: ft.TextField,
    lora_dropout_tf: ft.TextField,
    use_rslora_cb: ft.Checkbox,
    packing_cb: ft.Checkbox,
    auto_resume_cb: ft.Checkbox,
    push_cb: ft.Checkbox,
    hf_repo_id_tf: ft.TextField,
    resume_from_tf: ft.TextField,
    warmup_steps_tf: ft.TextField,
    weight_decay_tf: ft.TextField,
    lr_sched_dd: ft.Dropdown,
    optim_dd: ft.Dropdown,
    logging_steps_tf: ft.TextField,
    logging_first_step_cb: ft.Checkbox,
    disable_tqdm_cb: ft.Checkbox,
    seed_tf: ft.TextField,
    save_strategy_dd: ft.Dropdown,
    save_total_limit_tf: ft.TextField,
    pin_memory_cb: ft.Checkbox,
    report_to_dd: ft.Dropdown,
    fp16_cb: ft.Checkbox,
    bf16_cb: ft.Checkbox,
) -> dict:
    """Build train.py flags from UI controls. Mirrors previous _build_hp in main.py."""
    src = train_source.value or "Database"
    repo = (train_hf_repo.value or "").strip()
    split = (train_hf_split.value or "train").strip()
    cfg = (train_hf_config.value or "").strip()
    # Note: train_json_path is kept for backward compatibility but not used (database storage)
    model = (base_model.value or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit").strip()
    out_dir = (out_dir_tf.value or "/data/outputs/runpod_run").strip()

    hp: dict = {
        "base_model": model,
        "epochs": (epochs_tf.value or "3").strip(),
        "lr": (lr_tf.value or "2e-4").strip(),
        "bsz": (batch_tf.value or "2").strip(),
        "grad_accum": (grad_acc_tf.value or "4").strip(),
        "max_steps": (max_steps_tf.value or "200").strip(),
        "use_lora": bool(getattr(use_lora_cb, "value", False)),
        "output_dir": out_dir,
    }
    # LoRA configuration
    if bool(getattr(use_lora_cb, "value", False)):
        _lora_r = (lora_r_dd.value or "16").strip()
        hp["lora_r"] = _lora_r
        _lora_alpha = (lora_alpha_tf.value or "").strip()
        if _lora_alpha:
            hp["lora_alpha"] = _lora_alpha
        _lora_dropout = (lora_dropout_tf.value or "0").strip()
        if _lora_dropout and _lora_dropout != "0":
            hp["lora_dropout"] = _lora_dropout
        if bool(getattr(use_rslora_cb, "value", False)):
            hp["use_rslora"] = True
    # Dataset source handling
    if src == "Hugging Face" and repo:
        hp["hf_dataset_id"] = repo
        hp["hf_dataset_split"] = split
        if cfg:
            hp["hf_dataset_config"] = cfg
    elif src == "Database":
        db_session_id = (train_db_session_dd.value or "").strip()
        if db_session_id:
            hp["db_session_id"] = db_session_id
    if bool(getattr(packing_cb, "value", False)):
        hp["packing"] = True
    if bool(getattr(auto_resume_cb, "value", False)):
        hp["auto_resume"] = True

    try:
        if bool(getattr(push_cb, "value", False)):
            hp["push"] = True
            _hf_repo_id = (hf_repo_id_tf.value or "").strip()
            if _hf_repo_id:
                hp["hf_repo_id"] = _hf_repo_id
    except Exception:
        pass

    _resume_from = (resume_from_tf.value or "").strip()
    if _resume_from:
        hp["resume_from"] = _resume_from

    try:
        _ws = (warmup_steps_tf.value or "").strip()
        if _ws:
            hp["warmup_steps"] = _ws
    except Exception:
        pass
    try:
        _wd = (weight_decay_tf.value or "").strip()
        if _wd:
            hp["weight_decay"] = _wd
    except Exception:
        pass
    try:
        _lrs = (lr_sched_dd.value or "").strip()
        if _lrs:
            hp["lr_scheduler"] = _lrs
    except Exception:
        pass
    try:
        _opt = (optim_dd.value or "").strip()
        if _opt:
            hp["optimizer"] = _opt
    except Exception:
        pass
    try:
        _ls = (logging_steps_tf.value or "").strip()
        if _ls:
            hp["logging_steps"] = _ls
    except Exception:
        pass
    try:
        if bool(getattr(logging_first_step_cb, "value", False)):
            hp["logging_first_step"] = True
    except Exception:
        pass
    try:
        if bool(getattr(disable_tqdm_cb, "value", False)):
            hp["disable_tqdm"] = True
    except Exception:
        pass
    try:
        _seed = (seed_tf.value or "").strip()
        if _seed:
            hp["seed"] = _seed
    except Exception:
        pass
    try:
        _ss = (save_strategy_dd.value or "").strip()
        if _ss:
            hp["save_strategy"] = _ss
    except Exception:
        pass
    try:
        _stl = (save_total_limit_tf.value or "").strip()
        if _stl:
            hp["save_total_limit"] = _stl
    except Exception:
        pass
    try:
        if bool(getattr(pin_memory_cb, "value", False)):
            hp["pin_memory"] = True
    except Exception:
        pass
    try:
        _rt = (report_to_dd.value or "").strip()
        if _rt:
            hp["report_to"] = _rt
    except Exception:
        pass
    try:
        if bool(getattr(fp16_cb, "value", False)):
            hp["fp16"] = True
    except Exception:
        pass
    try:
        if bool(getattr(bf16_cb, "value", False)):
            hp["bf16"] = True
    except Exception:
        pass

    if "optimizer" in hp:
        hp["optim"] = hp.pop("optimizer")

    _allowed = {
        "json_path",
        "hf_dataset_id",
        "hf_dataset_split",
        "db_session_id",
        "base_model",
        "epochs",
        "lr",
        "bsz",
        "grad_accum",
        "max_seq_len",
        "max_steps",
        "packing",
        "report_to",
        "optim",
        "lr_scheduler",
        "warmup_steps",
        "weight_decay",
        "eval_every_steps",
        "save_every_steps",
        "save_total_limit",
        "use_lora",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "use_rslora",
        "output_dir",
        "resume_from",
        "auto_resume",
        "push",
        "hf_repo_id",
    }
    hp = {k: v for k, v in hp.items() if k in _allowed}
    return hp


async def _append_local_log_line(
    page: ft.Page,
    timeline: ft.ListView,
    placeholder: ft.Control,
    buffer: List[str],
    save_btn: ft.Control,
    txt: str,
    *,
    color=None,
    ICONS_module=None,
) -> None:
    try:
        try:
            placeholder.visible = False
        except Exception:
            pass
        timeline.controls.append(
            ft.Row(
                [
                    ft.Icon(
                        getattr(ICONS_module, "TERMINAL", getattr(ICONS_module, "CODE", None))
                        if ICONS_module
                        else None,
                        color=color or ACCENT_COLOR,
                        size=14,
                    ),
                    ft.Text(txt),
                ]
            )
        )
        try:
            buffer.append(txt)
            if hasattr(save_btn, "disabled"):
                save_btn.disabled = False
        except Exception:
            pass
        try:
            await safe_update(page)
        except Exception:
            pass
    except Exception:
        pass


async def _stream_local_logs(
    page: ft.Page,
    proc: subprocess.Popen,
    timeline: ft.ListView,
    placeholder: ft.Control,
    buffer: List[str],
    save_btn: ft.Control,
    ICONS_module=None,
):
    try:
        while True:
            stdout = proc.stdout
            if stdout is None:
                break
            line = await asyncio.to_thread(stdout.readline)
            if not line:
                break
            await _append_local_log_line(
                page, timeline, placeholder, buffer, save_btn, line.rstrip(), ICONS_module=ICONS_module
            )
    except Exception:
        pass


async def run_local_training(
    *,
    page: ft.Page,
    train_state: dict,
    hf_token_tf: ft.TextField,
    skill_level: ft.Dropdown,
    expert_gpu_dd: ft.Dropdown,
    local_use_gpu_cb: ft.Checkbox,
    local_pass_hf_token_cb: ft.Checkbox,
    proxy_enable_cb: ft.Checkbox,
    use_env_cb: ft.Checkbox,
    proxy_url_tf: ft.TextField,
    train_source: ft.Dropdown,
    train_hf_repo: ft.TextField,
    train_hf_split: ft.Dropdown,
    train_json_path: ft.TextField,
    train_db_session_dd: ft.Dropdown,
    local_training_run_dd: ft.Dropdown,
    local_train_status: ft.Text,
    local_train_progress: ft.ProgressBar,
    local_train_prog_label: ft.Text,
    local_train_timeline: ft.ListView,
    local_train_timeline_placeholder: ft.Control,
    local_log_buffer: List[str],
    local_save_logs_btn: ft.Control,
    local_start_btn: ft.Control,
    local_stop_btn: ft.Control,
    build_hp_fn: Callable[[], dict],
    ICONS_module=None,
    on_training_complete: Callable[[], None] = None,
) -> None:
    # Prevent concurrent runs
    if train_state.get("local", {}).get("running"):
        return

    # Get managed storage path from training run
    training_run_id = (local_training_run_dd.value or "").strip()
    if not training_run_id:
        local_train_status.value = "Select or create a training run first."
        local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
        await safe_update(page)
        return

    try:
        from db.training_runs import get_run_storage_paths, update_training_run

        paths = get_run_storage_paths(int(training_run_id))
        if not paths:
            local_train_status.value = f"Training run {training_run_id} not found."
            local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
            await safe_update(page)
            return
        host_dir = paths["root"]
    except Exception as e:
        local_train_status.value = f"Error getting training run storage: {e}"
        local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
        await safe_update(page)
        return
    if not os.path.isabs(host_dir) or (not os.path.exists(host_dir)):
        local_train_status.value = "Host data directory must be an existing absolute path."
        local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
        await safe_update(page)
        return

    # Build HP and dataset flags similar to pod flow
    hp = build_hp_fn() or {}
    try:
        if not (hp.get("hf_dataset_id") or hp.get("json_path") or hp.get("db_session_id")):
            src_ui = train_source.value or "Database"
            repo_ui = (train_hf_repo.value or "").strip()
            split_ui = (train_hf_split.value or "train").strip()
            db_session_ui = (train_db_session_dd.value or "").strip()
            if src_ui == "Database" and db_session_ui:
                hp["db_session_id"] = db_session_ui
            elif (src_ui == "Hugging Face") and repo_ui:
                hp["hf_dataset_id"] = repo_ui
                hp["hf_dataset_split"] = split_ui
        if not (hp.get("hf_dataset_id") or hp.get("json_path") or hp.get("db_session_id")):
            local_train_status.value = "Dataset not set. Select a database session or HF dataset."
            local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
            await safe_update(page)
            return
    except Exception:
        pass

    # For database sessions, export to temp JSON file in mounted directory
    try:
        db_session_id = hp.get("db_session_id", "")
        if db_session_id:
            from db.scraped_data import get_pairs_for_session, get_scrape_session

            session = get_scrape_session(int(db_session_id))
            if not session:
                local_train_status.value = f"Database session {db_session_id} not found."
                local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
                await safe_update(page)
                return
            pairs = get_pairs_for_session(int(db_session_id))
            if not pairs:
                local_train_status.value = f"No pairs found in session {db_session_id}."
                local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
                await safe_update(page)
                return
            # Export to ChatML format for training
            chatml_data = []
            for p in pairs:
                chatml_data.append(
                    {
                        "messages": [
                            {"role": "user", "content": p["input"]},
                            {"role": "assistant", "content": p["output"]},
                        ]
                    }
                )
            # Write to temp file in mounted directory
            json_filename = f"db_session_{db_session_id}.json"
            dest_path = os.path.join(host_dir, json_filename)
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(chatml_data, f, ensure_ascii=False, indent=2)
            hp["json_path"] = dest_path
            del hp["db_session_id"]  # Remove db_session_id, trainer uses json_path
            await _append_local_log_line(
                page,
                local_train_timeline,
                local_train_timeline_placeholder,
                local_log_buffer,
                local_save_logs_btn,
                f"Exported {len(pairs)} pairs from DB session {db_session_id} to {json_filename}",
                ICONS_module=ICONS_module,
            )
    except Exception as ex:
        local_train_status.value = f"Failed to export database session: {ex}"
        local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
        await safe_update(page)
        return

    # For local JSON files, copy to managed storage directory
    try:
        host_json_path = hp.get("json_path", "")
        if host_json_path and os.path.isfile(host_json_path):
            # Copy JSON file to the mounted host directory
            json_filename = os.path.basename(host_json_path)
            dest_path = os.path.join(host_dir, json_filename)
            try:
                if os.path.abspath(host_json_path) != os.path.abspath(dest_path):
                    shutil.copy2(host_json_path, dest_path)
            except Exception:
                pass
            hp["json_path"] = dest_path
    except Exception as ex:
        local_train_status.value = f"Failed to copy JSON file: {ex}"
        local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
        await safe_update(page)
        return

    # Ensure output_dir is mapped to managed storage root.
    # Runpod uses /data/...; native local flow uses OUTPUT_ROOT + relative output_dir.
    abs_out_dir = ""
    try:
        out_dir = (hp.get("output_dir") or "").strip()
        if out_dir.startswith("/data"):
            rel = out_dir[len("/data") :].lstrip("/")
            hp["output_dir"] = rel
            abs_out_dir = os.path.join(host_dir, rel) if rel else host_dir
        elif os.path.isabs(out_dir):
            abs_out_dir = out_dir
        else:
            hp["output_dir"] = out_dir.lstrip("/")
            abs_out_dir = os.path.join(host_dir, hp["output_dir"]) if hp["output_dir"] else host_dir
    except Exception:
        pass

    # Preflight output directory writability. If a previous run created a root-owned or
    # otherwise non-writable directory, fall back to a fresh directory under managed storage.
    try:
        if abs_out_dir:
            os.makedirs(abs_out_dir, exist_ok=True)
            _probe = os.path.join(abs_out_dir, ".ff_write_probe")
            with open(_probe, "w", encoding="utf-8") as f:
                f.write("ok")
            try:
                os.remove(_probe)
            except Exception:
                pass
    except Exception as ex:
        _repaired = False
        try:
            _user = ""
            _uid = None
            _gid = None
            try:
                _user = getpass.getuser()
            except Exception:
                _user = ""
            try:
                _uid = os.getuid()
                _gid = os.getgid()
            except Exception:
                _uid = None
                _gid = None

            _stat_path = abs_out_dir or ""
            _st = None
            try:
                if _stat_path:
                    _st = os.stat(_stat_path)
            except Exception:
                _st = None

            if _st is None and _stat_path:
                try:
                    _stat_path = os.path.dirname(_stat_path)
                    if _stat_path:
                        _st = os.stat(_stat_path)
                except Exception:
                    _st = None

            _owner = "?"
            _group = "?"
            _mode = "?"
            _type = "?"
            try:
                if _st is not None:
                    if pwd is not None:
                        _owner = pwd.getpwuid(_st.st_uid).pw_name
                    if grp is not None:
                        _group = grp.getgrgid(_st.st_gid).gr_name
                    _mode = oct(stat.S_IMODE(_st.st_mode))
                    _type = "dir" if stat.S_ISDIR(_st.st_mode) else "file"
            except Exception:
                pass

            _diag = (
                f"Output dir not writable. path={abs_out_dir} stat_path={_stat_path} "
                f"type={_type} owner={_owner}:{_group} mode={_mode} "
                f"current_user={_user} uid={_uid} gid={_gid} error={ex}"
            )
            try:
                print(f"[local-train] {_diag}")
            except Exception:
                pass
            try:
                await _append_local_log_line(
                    page,
                    local_train_timeline,
                    local_train_timeline_placeholder,
                    local_log_buffer,
                    local_save_logs_btn,
                    _diag,
                    color=WITH_OPACITY(0.8, COLORS.ORANGE),
                    ICONS_module=ICONS_module,
                )
            except Exception:
                pass

            try:
                if _st is not None and _uid is not None and _st.st_uid == _uid and _stat_path:
                    _cur_mode = stat.S_IMODE(_st.st_mode)
                    _new_mode = _cur_mode | stat.S_IWUSR
                    if stat.S_ISDIR(_st.st_mode):
                        _new_mode = _new_mode | stat.S_IXUSR
                    if _new_mode != _cur_mode:
                        os.chmod(_stat_path, _new_mode)
                    if abs_out_dir:
                        os.makedirs(abs_out_dir, exist_ok=True)
                        _probe = os.path.join(abs_out_dir, ".ff_write_probe")
                        with open(_probe, "w", encoding="utf-8") as f:
                            f.write("ok")
                        try:
                            os.remove(_probe)
                        except Exception:
                            pass
                        _repaired = True
            except Exception:
                _repaired = False

            if _repaired:
                try:
                    await _append_local_log_line(
                        page,
                        local_train_timeline,
                        local_train_timeline_placeholder,
                        local_log_buffer,
                        local_save_logs_btn,
                        f"Repaired output_dir permissions; continuing with: {abs_out_dir}",
                        color=WITH_OPACITY(
                            0.8,
                            getattr(COLORS, "GREEN_400", getattr(COLORS, "GREEN", ACCENT_COLOR)),
                        ),
                        ICONS_module=ICONS_module,
                    )
                except Exception:
                    pass

            if not _repaired:
                try:
                    if _stat_path:
                        _grp = _group if _group != "?" else "$(id -gn)"
                        _usr = _user if _user else "$(id -un)"
                        _suggest = (
                            f"To fix permanently, run: sudo chown -R {_usr}:{_grp} '{_stat_path}' && "
                            f"sudo chmod -R u+rwX '{_stat_path}'"
                        )
                        await _append_local_log_line(
                            page,
                            local_train_timeline,
                            local_train_timeline_placeholder,
                            local_log_buffer,
                            local_save_logs_btn,
                            _suggest,
                            color=WITH_OPACITY(0.8, COLORS.ORANGE),
                            ICONS_module=ICONS_module,
                        )
                except Exception:
                    pass
        except Exception:
            pass

        if not _repaired:
            try:
                from datetime import datetime

                base_rel = (hp.get("output_dir") or "outputs/local_run").strip()
                if base_rel.startswith("/data"):
                    base_rel = base_rel[len("/data") :].lstrip("/")
                if os.path.isabs(base_rel):
                    base_rel = "outputs/local_run"
                base_rel = base_rel.strip().strip("/") or "outputs/local_run"
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fallback_rel = f"{base_rel}_{ts}"
                hp["output_dir"] = fallback_rel
                abs_out_dir = os.path.join(host_dir, fallback_rel)
                os.makedirs(abs_out_dir, exist_ok=True)
                try:
                    print(f"[local-train] output_dir not writable, falling back: {abs_out_dir} ({ex})")
                except Exception:
                    pass
                try:
                    await _append_local_log_line(
                        page,
                        local_train_timeline,
                        local_train_timeline_placeholder,
                        local_log_buffer,
                        local_save_logs_btn,
                        f"Output dir not writable; using: {abs_out_dir}",
                        color=WITH_OPACITY(0.8, COLORS.ORANGE),
                        ICONS_module=ICONS_module,
                    )
                except Exception:
                    pass
            except Exception:
                pass

    # Show the hyperparameters to confirm what will be passed as --flags
    try:
        await _append_local_log_line(
            page,
            local_train_timeline,
            local_train_timeline_placeholder,
            local_log_buffer,
            local_save_logs_btn,
            "HP: " + json.dumps(hp, ensure_ascii=False),
            ICONS_module=ICONS_module,
        )
    except Exception:
        pass

    # Prepare environment for subprocess
    env = dict(os.environ)
    env["OUTPUT_ROOT"] = host_dir

    try:
        # Get HF token: if field is read-only (masked), use env var; otherwise use field value
        _hf_tok = ""
        tf_value = (hf_token_tf.value or "").strip()
        tf_readonly = getattr(hf_token_tf, "read_only", False)

        if tf_readonly or tf_value.startswith("•"):
            # Token is saved and masked - retrieve from environment (set by _apply_hf_env_from_cfg)
            _hf_tok = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_HUB_TOKEN", "")
        else:
            # Use explicit token from text field, fall back to env
            _hf_tok = tf_value or os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_HUB_TOKEN", "")
        _hf_tok = _hf_tok.strip()

        if bool(getattr(local_pass_hf_token_cb, "value", False)) and _hf_tok:
            env["HF_TOKEN"] = _hf_tok
            env["HUGGINGFACE_HUB_TOKEN"] = _hf_tok
    except Exception:
        pass

    try:
        if bool(getattr(proxy_enable_cb, "value", False)):
            if bool(getattr(use_env_cb, "value", False)):
                pass  # use system env
            else:
                _purl = (proxy_url_tf.value or "").strip()
                if _purl:
                    env["http_proxy"] = _purl
                    env["https_proxy"] = _purl
    except Exception:
        pass

    # GPU selection / disabling
    try:
        is_beginner = (skill_level.value or "Beginner").lower() == "beginner"
        if not bool(getattr(local_use_gpu_cb, "value", False)):
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif not is_beginner:
            sel = (expert_gpu_dd.value or "AUTO").strip()
            if sel and sel.upper() != "AUTO":
                env["CUDA_VISIBLE_DEVICES"] = str(sel)
    except Exception:
        pass

    # Build native training command: python -m trainers.unsloth_trainer --flags
    run_args: List[str] = [sys.executable, "-m", "trainers.unsloth_trainer"]
    try:
        for k, v in (hp or {}).items():
            flag = "--" + str(k)
            if isinstance(v, bool):
                if v:
                    run_args.append(flag)
            elif v is None:
                continue
            else:
                run_args.extend([flag, str(v)])
    except Exception:
        pass

    # Launch process
    try:
        local_train_status.value = "Starting local training…"
        local_train_status.color = None
        local_train_progress.value = 0.1
        local_train_prog_label.value = "Starting…"
        local_train_timeline.controls.clear()
        try:
            local_train_timeline_placeholder.visible = True
        except Exception:
            pass
        try:
            local_log_buffer.clear()
            if hasattr(local_save_logs_btn, "disabled"):
                local_save_logs_btn.disabled = True
        except Exception:
            pass
        # Update training run status to running
        try:
            from db.training_runs import update_training_run
            from datetime import datetime

            update_training_run(
                int(training_run_id),
                status="running",
                started_at=datetime.now().isoformat(),
            )
        except Exception:
            pass
        await safe_update(page)
    except Exception:
        pass

    try:
        popen_cwd = None
        try:
            # Ensure src/ is on PYTHONPATH so `-m trainers.unsloth_trainer` resolves in src-layout checkouts.
            repo_root = Path(__file__).resolve().parents[2]
            src_dir = repo_root / "src"
            if src_dir.is_dir():
                existing_pp = (env.get("PYTHONPATH") or "").strip()
                env["PYTHONPATH"] = str(src_dir) + (os.pathsep + existing_pp if existing_pp else "")
                popen_cwd = str(repo_root)
        except Exception:
            popen_cwd = None

        try:
            print(f"[local-train] cwd={popen_cwd or os.getcwd()}")
            print(f"[local-train] cmd={' '.join(run_args)}")
            _pp = (env.get("PYTHONPATH") or "").strip()
            if _pp:
                print(f"[local-train] PYTHONPATH={_pp}")
        except Exception:
            pass

        proc = subprocess.Popen(
            run_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=popen_cwd,
        )
        train_state.setdefault("local", {})
        train_state["local"] = {"running": True, "proc": proc}
        try:
            if hasattr(local_start_btn, "disabled"):
                local_start_btn.disabled = True
            if hasattr(local_stop_btn, "disabled"):
                local_stop_btn.disabled = False
        except Exception:
            pass
        await safe_update(page)
        await _append_local_log_line(
            page,
            local_train_timeline,
            local_train_timeline_placeholder,
            local_log_buffer,
            local_save_logs_btn,
            "Command: " + " ".join(run_args),
            color=WITH_OPACITY(0.8, COLORS.BLUE),
            ICONS_module=None,
        )
        try:
            await _append_local_log_line(
                page,
                local_train_timeline,
                local_train_timeline_placeholder,
                local_log_buffer,
                local_save_logs_btn,
                "CWD: " + str(popen_cwd or os.getcwd()),
                color=WITH_OPACITY(0.65, COLORS.BLUE),
                ICONS_module=None,
            )
            _pp2 = (env.get("PYTHONPATH") or "").strip()
            if _pp2:
                await _append_local_log_line(
                    page,
                    local_train_timeline,
                    local_train_timeline_placeholder,
                    local_log_buffer,
                    local_save_logs_btn,
                    "PYTHONPATH: " + _pp2,
                    color=WITH_OPACITY(0.65, COLORS.BLUE),
                    ICONS_module=None,
                )
        except Exception:
            pass
        await _stream_local_logs(
            page,
            proc,
            local_train_timeline,
            local_train_timeline_placeholder,
            local_log_buffer,
            local_save_logs_btn,
            ICONS_module=None,
        )
        rc = proc.wait()
        if rc == 0:
            local_train_status.value = "Training finished."
            local_train_status.color = getattr(COLORS, "GREEN_400", getattr(COLORS, "GREEN", None))
            try:
                base_model_name = (hp.get("base_model") or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit").strip()
                adapter_path = ""
                if abs_out_dir:
                    try:
                        train_state.setdefault("local", {})
                        train_state["local"]["output_dir"] = abs_out_dir
                    except Exception:
                        pass
                    adapter_path = os.path.join(abs_out_dir, "adapter")
                    try:
                        train_state.setdefault("local_infer", {})
                        train_state["local_infer"]["adapter_path"] = adapter_path
                        train_state["local_infer"]["base_model"] = base_model_name
                        train_state["local_infer"]["model_loaded"] = False
                        # Store dataset info for sample prompts feature - only for the actual source used
                        train_state["local_infer"]["dataset_session_id"] = None
                        train_state["local_infer"]["hf_dataset_id"] = None
                        src_used = (train_source.value or "Database").strip()
                        if src_used == "Database":
                            db_session_val = (train_db_session_dd.value or "").strip()
                            if db_session_val:
                                train_state["local_infer"]["dataset_session_id"] = db_session_val
                        elif src_used == "Hugging Face":
                            hf_repo_val = (train_hf_repo.value or "").strip()
                            hf_split_val = (train_hf_split.value or "train").strip()
                            if hf_repo_val:
                                train_state["local_infer"]["hf_dataset_id"] = hf_repo_val
                                train_state["local_infer"]["hf_dataset_split"] = hf_split_val
                                # Fetch sample prompts from HF dataset
                                try:
                                    hf_samples = await _fetch_hf_samples(hf_repo_val, hf_split_val, 10)
                                    if hf_samples:
                                        train_state["local_infer"]["hf_samples"] = hf_samples
                                except Exception:
                                    pass
                    except Exception:
                        pass
                # Update training run with completed status and paths - ALWAYS update status
                try:
                    from db.training_runs import update_training_run
                    from datetime import datetime

                    update_training_run(
                        int(training_run_id),
                        status="completed",
                        output_dir=abs_out_dir or None,
                        adapter_path=adapter_path or None,
                        completed_at=datetime.now().isoformat(),
                        logs=local_log_buffer[:500],  # Save last 500 log lines
                    )
                except Exception as e:
                    # Log the error so we can debug if update fails
                    await _append_local_log_line(
                        page,
                        local_train_timeline,
                        local_train_timeline_placeholder,
                        local_log_buffer,
                        local_save_logs_btn,
                        f"Warning: Failed to update training run status: {e}",
                        color=WITH_OPACITY(0.8, COLORS.ORANGE),
                        ICONS_module=None,
                    )
            except Exception:
                pass
        elif rc == 137:
            local_train_status.value = (
                "Training exited with code 137. Likely out-of-memory (OOM) or killed (SIGKILL).\n"
                "This happens when the training process runs out of memory or is manually stopped."
            )
            local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
            try:
                await _append_local_log_line(
                    page,
                    local_train_timeline,
                    local_train_timeline_placeholder,
                    local_log_buffer,
                    local_save_logs_btn,
                    "Exit code 137 detected: OOM or manual kill (SIGKILL). Reduce memory usage or increase limits.",
                    color=WITH_OPACITY(0.9, COLORS.RED),
                    ICONS_module=None,
                )
            except Exception:
                pass
            # Update training run status to failed
            try:
                from db.training_runs import update_training_run
                from datetime import datetime

                update_training_run(
                    int(training_run_id),
                    status="failed",
                    completed_at=datetime.now().isoformat(),
                    logs=local_log_buffer[:500],
                    metadata={"exit_code": 137, "error": "OOM or killed"},
                )
            except Exception:
                pass
        else:
            local_train_status.value = f"Training exited with code {rc}."
            local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
            # Update training run status to failed
            try:
                from db.training_runs import update_training_run
                from datetime import datetime

                update_training_run(
                    int(training_run_id),
                    status="failed",
                    completed_at=datetime.now().isoformat(),
                    logs=local_log_buffer[:500],
                    metadata={"exit_code": rc},
                )
            except Exception:
                pass
    except Exception as ex:
        local_train_status.value = f"Failed to start training: {ex}"
        try:
            local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
        except Exception:
            pass
        # Update training run status to failed
        try:
            from db.training_runs import update_training_run
            from datetime import datetime

            update_training_run(
                int(training_run_id),
                status="failed",
                completed_at=datetime.now().isoformat(),
                metadata={"error": str(ex)},
            )
        except Exception:
            pass
    finally:
        try:
            train_state.setdefault("local", {})
            train_state["local"]["running"] = False
            if hasattr(local_start_btn, "disabled"):
                local_start_btn.disabled = False
            if hasattr(local_stop_btn, "disabled"):
                local_stop_btn.disabled = True
            local_train_progress.value = 0.0
            local_train_prog_label.value = "Idle"
            await safe_update(page)
        except Exception:
            pass
        # Always refresh UI after training ends (success or failure)
        if on_training_complete:
            try:
                on_training_complete()
            except Exception:
                pass


async def stop_local_training(
    *,
    page: ft.Page,
    train_state: dict,
    local_train_status: ft.Text,
    local_start_btn: ft.Control,
    local_stop_btn: ft.Control,
    local_train_progress: ft.ProgressBar,
    local_train_prog_label: ft.Text,
) -> None:
    info = train_state.get("local") or {}
    proc = info.get("proc")
    if not proc:
        return
    # Ensure local process terminates
    try:
        if proc and (proc.poll() is None):
            proc.terminate()
    except Exception:
        pass
    try:
        train_state.setdefault("local", {})
        train_state["local"]["running"] = False
        if hasattr(local_start_btn, "disabled"):
            local_start_btn.disabled = False
        if hasattr(local_stop_btn, "disabled"):
            local_stop_btn.disabled = True
        local_train_status.value = "Stopped."
        local_train_status.color = getattr(COLORS, "ORANGE_400", getattr(COLORS, "ORANGE", None))
        local_train_progress.value = 0.0
        local_train_prog_label.value = "Idle"
        await safe_update(page)
    except Exception:
        pass
