from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from typing import List, Optional, Callable

import flet as ft

from helpers.common import safe_update
from helpers.theme import COLORS, ACCENT_COLOR
from helpers.ui import WITH_OPACITY


def build_hp_from_controls(
    *,
    train_source: ft.Dropdown,
    train_hf_repo: ft.TextField,
    train_hf_split: ft.Dropdown,
    train_hf_config: ft.TextField,
    train_json_path: ft.TextField,
    base_model: ft.TextField,
    out_dir_tf: ft.TextField,
    epochs_tf: ft.TextField,
    lr_tf: ft.TextField,
    batch_tf: ft.TextField,
    grad_acc_tf: ft.TextField,
    max_steps_tf: ft.TextField,
    use_lora_cb: ft.Checkbox,
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
    src = train_source.value or "Hugging Face"
    repo = (train_hf_repo.value or "").strip()
    split = (train_hf_split.value or "train").strip()
    cfg = (train_hf_config.value or "").strip()
    jpath = (train_json_path.value or "").strip()
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
    if (src == "Hugging Face") and repo:
        hp["hf_dataset_id"] = repo
        hp["hf_dataset_split"] = split
    elif jpath:
        hp["json_path"] = jpath
    if (src == "Hugging Face") and cfg:
        hp["hf_dataset_config"] = cfg
    if bool(getattr(packing_cb, "value", False)):
        hp["packing"] = True
    if bool(getattr(auto_resume_cb, "value", False)):
        hp["auto_resume"] = True
    if bool(getattr(push_cb, "value", False)):
        hp["push"] = True
    _hf_repo_id = (hf_repo_id_tf.value or "").strip()
    if _hf_repo_id:
        hp["hf_repo_id"] = _hf_repo_id
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
        "eval_every_steps",
        "save_every_steps",
        "save_total_limit",
        "use_lora",
        "lora_r",
        "lora_alpha",
        "output_dir",
        "resume_from",
        "auto_resume",
        "push",
        "hf_repo_id",
        "hf_private",
    }
    hp = {k: v for k, v in hp.items() if k in _allowed}
    return hp

async def _docker_daemon_ready(page: ft.Page, status_text: ft.Text) -> bool:
    try:
        if not shutil.which("docker"):
            status_text.value = "Docker CLI not found. Install Docker Desktop and ensure 'docker' is on PATH."
            status_text.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
            await safe_update(page)
            return False
        info_res = await asyncio.to_thread(
            lambda: subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=4)
        )
        if info_res.returncode != 0:
            msg = (info_res.stderr or info_res.stdout or "").strip() or "Docker daemon not responding"
            status_text.value = "Docker is not running. Start Docker Desktop, then retry.\n" + msg
            status_text.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
            await safe_update(page)
            return False
        return True
    except Exception as ex:
        status_text.value = f"Docker check failed: {ex}"
        try:
            status_text.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
        except Exception:
            pass
        await safe_update(page)
        return False


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
            ft.Row([
                ft.Icon(getattr(ICONS_module, "TERMINAL", getattr(ICONS_module, "CODE", None)) if ICONS_module else None, color=color or ACCENT_COLOR, size=14),
                ft.Text(txt),
            ])
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


async def _stream_local_logs(page: ft.Page, proc: subprocess.Popen, timeline: ft.ListView, placeholder: ft.Control, buffer: List[str], save_btn: ft.Control, ICONS_module=None):
    try:
        while True:
            line = await asyncio.to_thread(proc.stdout.readline)  # type: ignore[arg-type]
            if not line:
                break
            await _append_local_log_line(page, timeline, placeholder, buffer, save_btn, line.rstrip(), ICONS_module=ICONS_module)
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
    docker_image_tf: ft.TextField,
    train_source: ft.Dropdown,
    train_hf_repo: ft.TextField,
    train_hf_split: ft.Dropdown,
    train_json_path: ft.TextField,
    local_host_dir_tf: ft.TextField,
    local_container_name_tf: ft.TextField,
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
    DEFAULT_DOCKER_IMAGE: str,
    rp_pod_module,
    ICONS_module=None,
) -> None:
    # Prevent concurrent runs
    if train_state.get("local", {}).get("running"):
        return

    # Basic validations
    if not await _docker_daemon_ready(page, local_train_status):
        return

    img = (docker_image_tf.value or "").strip() or DEFAULT_DOCKER_IMAGE
    host_dir = (local_host_dir_tf.value or "").strip()
    if not host_dir:
        local_train_status.value = "Set Host data directory to mount at /data."
        local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
        await safe_update(page)
        return
    if not os.path.isabs(host_dir) or (not os.path.exists(host_dir)):
        local_train_status.value = "Host data directory must be an existing absolute path."
        local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
        await safe_update(page)
        return

    # Ensure image exists (non-blocking pull hint if missing)
    try:
        insp = await asyncio.to_thread(lambda: subprocess.run(["docker", "image", "inspect", img], capture_output=True, text=True))
        if insp.returncode != 0:
            local_train_status.value = f"Image not found locally: {img}. Pull it first in the section above."
            local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
            await safe_update(page)
            return
    except Exception:
        pass

    # Build HP and dataset flags similar to pod flow
    hp = build_hp_fn() or {}
    try:
        if not (hp.get("hf_dataset_id") or hp.get("json_path")):
            src_ui = train_source.value or "Hugging Face"
            repo_ui = (train_hf_repo.value or "").strip()
            split_ui = (train_hf_split.value or "train").strip()
            jpath_ui = (train_json_path.value or "").strip()
            if (src_ui == "Hugging Face") and repo_ui:
                hp["hf_dataset_id"] = repo_ui
                hp["hf_dataset_split"] = split_ui
            elif jpath_ui:
                hp["json_path"] = jpath_ui
        if not (hp.get("hf_dataset_id") or hp.get("json_path")):
            local_train_status.value = "Dataset not set. Provide HF dataset or JSON path."
            local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
            await safe_update(page)
            return
    except Exception:
        pass

    # Show the hyperparameters to confirm what will be passed as --flags
    try:
        await _append_local_log_line(page, local_train_timeline, local_train_timeline_placeholder, local_log_buffer, local_save_logs_btn, "HP: " + json.dumps(hp, ensure_ascii=False), ICONS_module=ICONS_module)
    except Exception:
        pass

    # Build docker run command
    cont_name = (local_container_name_tf.value or "ds-local-train-").strip()
    if cont_name.endswith("-"):
        # Avoid trailing hyphen-only names
        cont_name = f"{cont_name}{str(abs(hash(img)))[:6]}"
    run_args: List[str] = [
        "docker", "run", "--rm",
        "--name", cont_name,
        "-v", f"{host_dir}:/data",
    ]
    try:
        is_beginner = ((skill_level.value or "Beginner").lower() == "beginner")
        if is_beginner:
            if bool(getattr(local_use_gpu_cb, "value", False)):
                run_args += ["--gpus", "all"]
        else:
            sel = (expert_gpu_dd.value or "AUTO")
            if str(sel).upper() == "AUTO":
                run_args += ["--gpus", "all"]
            else:
                run_args += ["--gpus", f"device={sel}"]
    except Exception:
        pass

    try:
        # Prefer an explicit token from the text field; fall back to process env
        _hf_tok = (
            (hf_token_tf.value or "")
            or os.environ.get("HF_TOKEN", "")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN", "")
        ).strip()
        if bool(getattr(local_pass_hf_token_cb, "value", False)) and _hf_tok:
            # Match the environment variable names used by huggingface_hub and the Runpod flow
            run_args += [
                "-e",
                f"HF_TOKEN={_hf_tok}",
                "-e",
                f"HUGGINGFACE_HUB_TOKEN={_hf_tok}",
            ]
    except Exception:
        pass

    try:
        if bool(getattr(proxy_enable_cb, "value", False)):
            if bool(getattr(use_env_cb, "value", False)):
                pass  # use system env
            else:
                _purl = (proxy_url_tf.value or "").strip()
                if _purl:
                    run_args += ["-e", f"http_proxy={_purl}", "-e", f"https_proxy={_purl}"]
    except Exception:
        pass

    run_args += [img]

    # Command inside container mirrors Runpod builder
    try:
        inner_cmd = rp_pod_module.build_cmd(hp)
    except Exception as ex:
        local_train_status.value = f"Failed building training command: {ex}"
        local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
        await safe_update(page)
        return
    run_args += inner_cmd

    # Launch process
    try:
        local_train_status.value = "Starting local Docker container…"
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
        await safe_update(page)
    except Exception:
        pass

    try:
        proc = subprocess.Popen(
            run_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        train_state.setdefault("local", {})
        train_state["local"] = {"running": True, "proc": proc, "container": cont_name}
        try:
            if hasattr(local_start_btn, "disabled"):
                local_start_btn.disabled = True
            if hasattr(local_stop_btn, "disabled"):
                local_stop_btn.disabled = False
        except Exception:
            pass
        await safe_update(page)
        await _append_local_log_line(page, local_train_timeline, local_train_timeline_placeholder, local_log_buffer, local_save_logs_btn, "Command: " + " ".join(run_args), color=WITH_OPACITY(0.8, COLORS.BLUE), ICONS_module=None)
        await _stream_local_logs(page, proc, local_train_timeline, local_train_timeline_placeholder, local_log_buffer, local_save_logs_btn, ICONS_module=None)
        rc = proc.wait()
        if rc == 0:
            local_train_status.value = "Training finished (container exited)."
            local_train_status.color = getattr(COLORS, "GREEN_400", getattr(COLORS, "GREEN", None))
            try:
                out_dir = (hp.get("output_dir") or "").strip()
                base_model_name = (hp.get("base_model") or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit").strip()
                host_root = (local_host_dir_tf.value or "").strip()
                abs_out_dir = ""
                if out_dir:
                    if out_dir.startswith("/data"):
                        rel = out_dir[len("/data"):].lstrip("/")
                        abs_out_dir = os.path.join(host_root, rel)
                    else:
                        abs_out_dir = os.path.join(host_root, out_dir.lstrip("/"))
                if abs_out_dir:
                    adapter_path = os.path.join(abs_out_dir, "adapter")
                    try:
                        train_state.setdefault("local_infer", {})
                        train_state["local_infer"]["adapter_path"] = adapter_path
                        train_state["local_infer"]["base_model"] = base_model_name
                        train_state["local_infer"]["model_loaded"] = False
                    except Exception:
                        pass
            except Exception:
                pass
        elif rc == 137:
            local_train_status.value = (
                "Container exited with code 137. Likely out-of-memory (OOM) or killed (SIGKILL).\n"
                "This happens when the container runs out of memory or is manually stopped."
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
        else:
            local_train_status.value = f"Container exited with code {rc}."
            local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
    except Exception as ex:
        local_train_status.value = f"Failed to start container: {ex}"
        try:
            local_train_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
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
    cont = (info.get("container") or "").strip()
    proc: Optional[subprocess.Popen] = info.get("proc")  # type: ignore
    if not cont and not proc:
        return
    # Try docker stop by name first
    try:
        if cont:
            await asyncio.to_thread(lambda: subprocess.run(["docker", "stop", cont], capture_output=True, text=True, timeout=10))
    except Exception:
        pass
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
