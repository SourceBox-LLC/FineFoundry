from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import Callable, Awaitable, Optional, List

import flet as ft

from helpers.common import safe_update
from helpers.theme import COLORS, ACCENT_COLOR
from helpers.ui import WITH_OPACITY


async def refresh_expert_gpus(
    *,
    page: ft.Page,
    rp_pod_module,
    train_state: dict,
    _runpod_cfg: dict,
    rp_temp_key_tf: ft.TextField,
    rp_dc_tf: ft.TextField,
    expert_gpu_busy: ft.Control,
    expert_gpu_dd: ft.Dropdown,
    expert_spot_cb: ft.Checkbox,
    expert_gpu_avail: dict,
    _update_expert_spot_enabled: Callable[[], None],
) -> None:
    try:
        # Resolve API key similar to Ensure Infra
        saved_key = (((_runpod_cfg.get("api_key") or "") if isinstance(_runpod_cfg, dict) else "").strip())
        temp_key = (rp_temp_key_tf.value or "").strip()
        key = saved_key or temp_key or (os.environ.get("RUNPOD_API_KEY") or "").strip()
        if not key:
            page.snack_bar = ft.SnackBar(ft.Text("Runpod API key missing. Set it in Settings → Runpod API Access."))
            page.open(page.snack_bar)
            return
        infra = train_state.get("infra") or {}
        # Determine datacenter: prefer ensured volume's dc, else current field, else default
        try:
            dc_src = ((infra.get("volume") or {}).get("dc")) or (rp_dc_tf.value or "")
            dc_id = (dc_src.strip() or "US-NC-1")
        except Exception:
            dc_id = "US-NC-1"

        # Show busy while fetching
        try:
            expert_gpu_busy.visible = True
            page.update()
        except Exception:
            pass

        # Fetch GPUs
        def _fetch():
            return rp_pod_module.list_available_gpus(key, dc_id, 1)
        gpus = await asyncio.to_thread(_fetch)
        # Build options with de-duplication by GPU type id, merging flags
        opts = [ft.dropdown.Option(text="AUTO (best secure)", key="AUTO")]
        expert_gpu_avail.clear()
        agg: dict = {}
        for g in (gpus or []):
            gid = str(g.get("id") or "").strip()
            if not gid:
                continue
            d = agg.get(gid) or {
                "displayName": str(g.get("displayName") or gid),
                "memoryInGb": g.get("memoryInGb"),
                "secureAvailable": False,
                "spotAvailable": False,
            }
            # Merge availability flags across duplicates
            d["secureAvailable"] = bool(d.get("secureAvailable")) or bool(g.get("secureAvailable"))
            d["spotAvailable"] = bool(d.get("spotAvailable")) or bool(g.get("spotAvailable"))
            # Prefer max memory if multiple values appear
            try:
                mem_prev = float(d.get("memoryInGb") or 0)
                mem_new = float(g.get("memoryInGb") or 0)
                d["memoryInGb"] = max(mem_prev, mem_new)
            except Exception:
                pass
            agg[gid] = d
        # Emit unique options
        for gid, d in agg.items():
            name = str(d.get("displayName") or gid)
            mem = d.get("memoryInGb")
            sec = bool(d.get("secureAvailable"))
            spot = bool(d.get("spotAvailable"))
            tags = []
            if sec: tags.append("secure")
            if spot: tags.append("spot")
            mem_txt = (f" {int(mem)}GB" if isinstance(mem, (int, float)) and mem else "")
            label = f"{name}{mem_txt} [{'/' .join(tags) if tags else 'limited'}]"
            opts.append(ft.dropdown.Option(text=label, key=gid))
            expert_gpu_avail[gid] = {"secureAvailable": sec, "spotAvailable": spot}
        # Preserve selection if still available
        cur = (expert_gpu_dd.value or "AUTO")
        keys = {getattr(o, 'key', None) or o.text for o in opts}
        expert_gpu_dd.options = opts
        if cur not in keys:
            expert_gpu_dd.value = "AUTO"
        _update_expert_spot_enabled()
        try:
            expert_gpu_dd.tooltip = "Pick a Runpod GPU type or AUTO (best secure). Use Spot for interruptible when available."
        except Exception:
            pass
        try:
            expert_gpu_busy.visible = False
            page.update()
        except Exception:
            pass
    except Exception as ex:
        try:
            expert_gpu_busy.visible = False
            page.snack_bar = ft.SnackBar(ft.Text(f"Failed to refresh GPUs: {ex}"))
            page.open(page.snack_bar)
        except Exception:
            pass

async def refresh_teardown_ui(
    *,
    page: ft.Page,
    rp_pod_module,
    train_state: dict,
    td_template_cb: ft.Checkbox,
    td_volume_cb: ft.Checkbox,
    td_pod_cb: ft.Checkbox,
    teardown_section: ft.Container,
    _runpod_cfg: dict,
    rp_temp_key_tf: ft.TextField,
) -> None:
    try:
        infra = train_state.get("infra") or {}
    except Exception:
        infra = {}
    tpl = infra.get("template") or {}
    vol = infra.get("volume") or {}
    tpl_id = (str(tpl.get("id") or "").strip())
    vol_id = (str(vol.get("id") or "").strip())
    pod_id = str(train_state.get("pod_id") or "").strip()

    # Reconcile pod presence with Runpod (handles external deletions)
    try:
        key = (((_runpod_cfg.get("api_key") or "") if isinstance(_runpod_cfg, dict) else "").strip()) or (rp_temp_key_tf.value or "").strip() or (os.environ.get("RUNPOD_API_KEY") or "").strip()
    except Exception:
        key = (os.environ.get("RUNPOD_API_KEY") or "").strip()
    if pod_id and key:
        try:
            await asyncio.to_thread(rp_pod_module.get_pod, key, pod_id)
        except Exception as ex:
            status = None
            try:
                status = getattr(getattr(ex, "response", None), "status_code", None)
            except Exception:
                status = None
            if status == 404:
                try:
                    train_state["pod_id"] = None
                except Exception:
                    pass
                pod_id = ""

    # Template checkbox
    try:
        if tpl_id:
            name = tpl.get("name") or tpl_id
            img = tpl.get("image") or ""
            td_template_cb.label = f"Template: {name} (id={tpl_id}){f' • {img}' if img else ''}"
            td_template_cb.visible = True
        else:
            td_template_cb.visible = False
            td_template_cb.value = False
    except Exception:
        pass

    # Volume checkbox
    try:
        if vol_id:
            vname = vol.get("name") or vol_id
            vsize = vol.get("size") or "?"
            vdc = vol.get("dc") or vol.get("dataCenterId") or ""
            td_volume_cb.label = f"Volume: {vname} ({vsize}GB){f' • DC {vdc}' if vdc else ''} (id={vol_id})"
            td_volume_cb.visible = True
        else:
            td_volume_cb.visible = False
            td_volume_cb.value = False
    except Exception:
        pass

    # Pod checkbox (optional)
    try:
        if pod_id:
            td_pod_cb.label = f"Pod: {pod_id}"
            td_pod_cb.visible = True
        else:
            td_pod_cb.visible = False
            td_pod_cb.value = False
    except Exception:
        pass

    try:
        teardown_section.visible = bool(tpl_id or vol_id or pod_id)
        await safe_update(page)
    except Exception:
        pass


async def do_teardown(
    *,
    page: ft.Page,
    rp_pod_module,
    rp_infra_module,
    train_state: dict,
    td_template_cb: ft.Checkbox,
    td_volume_cb: ft.Checkbox,
    td_pod_cb: ft.Checkbox,
    td_busy: ft.ProgressRing,
    train_timeline: ft.ListView,
    update_train_placeholders: Callable[[], None],
    _runpod_cfg: dict,
    rp_temp_key_tf: ft.TextField,
    refresh_teardown_ui_fn: Callable[[], Awaitable[None]],
    selected_all: bool = False,
) -> None:
    # Resolve API key with same precedence: Settings > temp > env
    saved_key = (((_runpod_cfg.get("api_key") or "") if isinstance(_runpod_cfg, dict) else "").strip())
    temp_key = (rp_temp_key_tf.value or "").strip()
    key = saved_key or temp_key or (os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not key:
        train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.WARNING, color=COLORS.RED), ft.Text("Runpod API key missing. Set it in Settings → Runpod API Access.")]))
        update_train_placeholders(); await safe_update(page)
        return

    infra = train_state.get("infra") or {}
    tpl_id = str(((infra.get("template") or {}).get("id") or "")).strip()
    vol_id = str(((infra.get("volume") or {}).get("id") or "")).strip()
    pod_id = str(train_state.get("pod_id") or "").strip()

    # Reconcile pod existence before proceeding (avoid trying to delete a non-existent pod)
    if pod_id:
        try:
            await asyncio.to_thread(rp_pod_module.get_pod, key, pod_id)
        except Exception as ex:
            status = getattr(getattr(ex, "response", None), "status_code", None)
            if status == 404:
                try:
                    train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.INFO, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"Pod already absent on Runpod: {pod_id}")]))
                    update_train_placeholders(); await safe_update(page)
                except Exception:
                    pass
                try:
                    train_state["pod_id"] = None
                except Exception:
                    pass
                pod_id = ""

    sel_tpl = bool(td_template_cb.value) or (selected_all and bool(tpl_id))
    sel_vol = bool(td_volume_cb.value) or (selected_all and bool(vol_id))
    sel_pod = bool(td_pod_cb.value) or (selected_all and bool(pod_id))

    if not (sel_tpl or sel_vol or sel_pod):
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Select at least one item to teardown."))
            page.open(page.snack_bar)
            await safe_update(page)
        except Exception:
            pass
        return

    # Busy on
    try:
        td_busy.visible = True
        await safe_update(page)
    except Exception:
        pass

    # Add timeline log to indicate start and selected targets
    try:
        actions = []
        if sel_pod and pod_id:
            actions.append(f"Pod {pod_id}")
        if sel_tpl and tpl_id:
            actions.append(f"Template {tpl_id}")
        if sel_vol and vol_id:
            actions.append(f"Volume {vol_id}")
        if actions:
            train_timeline.controls.append(ft.Row([
                ft.Icon(getattr(ft.Icons, "PLAY_CIRCLE", ft.Icons.PLAY_ARROW), color=ACCENT_COLOR),
                ft.Text("Starting teardown: " + ", ".join(actions))
            ]))
            update_train_placeholders(); await safe_update(page)
    except Exception:
        pass

    # Perform deletions. Order: Pod → Template → Volume
    try:
        if sel_pod and pod_id:
            try:
                try:
                    train_timeline.controls.append(ft.Row([ft.Icon(getattr(ft.Icons, "CLOUD_OFF", getattr(ft.Icons, "CLOUD", ft.Icons.CLOSE)), color=WITH_OPACITY(0.9, COLORS.RED)), ft.Text(f"Deleting Pod: {pod_id}...")]))
                    await safe_update(page)
                except Exception:
                    pass
                await asyncio.to_thread(rp_pod_module.delete_pod, key, pod_id)
                train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.DELETE_FOREVER, color=COLORS.RED), ft.Text(f"Pod deleted: {pod_id}")]))
                try:
                    train_state["pod_id"] = None
                except Exception:
                    pass
            except Exception as ex:
                status = getattr(getattr(ex, "response", None), "status_code", None)
                if status == 404:
                    train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.INFO, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"Pod already deleted: {pod_id}")]))
                    try:
                        train_state["pod_id"] = None
                    except Exception:
                        pass
                else:
                    train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.ERROR, color=COLORS.RED), ft.Text(f"Failed to delete pod: {ex}")]))

        if sel_tpl and tpl_id:
            try:
                try:
                    train_timeline.controls.append(ft.Row([ft.Icon(getattr(ft.Icons, "DESCRIPTION", ft.Icons.ARTICLE), color=WITH_OPACITY(0.9, COLORS.RED)), ft.Text(f"Deleting Template: {tpl_id}...")]))
                    await safe_update(page)
                except Exception:
                    pass
                await asyncio.to_thread(rp_infra_module.delete_template, tpl_id, key)
                train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.DELETE_FOREVER, color=COLORS.RED), ft.Text(f"Template deleted: {tpl_id}")]))
                try:
                    if isinstance(train_state.get("infra"), dict):
                        train_state["infra"]["template"] = {}
                except Exception:
                    pass
            except Exception as ex:
                train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.ERROR, color=COLORS.RED), ft.Text(f"Failed to delete template: {ex}")]))

        if sel_vol and vol_id:
            try:
                try:
                    train_timeline.controls.append(ft.Row([ft.Icon(getattr(ft.Icons, "STORAGE", ft.Icons.SAVE), color=WITH_OPACITY(0.9, COLORS.RED)), ft.Text(f"Deleting Volume: {vol_id}...")]))
                    await safe_update(page)
                except Exception:
                    pass
                await asyncio.to_thread(rp_infra_module.delete_volume, vol_id, key)
                train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.DELETE_FOREVER, color=COLORS.RED), ft.Text(f"Volume deleted: {vol_id}")]))
                try:
                    if isinstance(train_state.get("infra"), dict):
                        train_state["infra"]["volume"] = {}
                except Exception:
                    pass
            except Exception as ex:
                train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.ERROR, color=COLORS.RED), ft.Text(f"Failed to delete volume: {ex}")]))

        # If both infra items are gone, clear infra
        try:
            infra = train_state.get("infra") or {}
            tpl_id2 = str(((infra.get("template") or {}).get("id") or "").strip())
            vol_id2 = str(((infra.get("volume") or {}).get("id") or "").strip())
            if not tpl_id2 and not vol_id2:
                train_state["infra"] = None
        except Exception:
            pass

        # Disable training actions if infra missing
        try:
            has_infra = bool(train_state.get("infra"))
            # The caller may tweak button states; keep minimal here
        except Exception:
            pass
        # Completion log and refresh UI
        try:
            train_timeline.controls.append(ft.Row([ft.Icon(getattr(ft.Icons, "CHECK_CIRCLE", ft.Icons.CHECK), color=COLORS.GREEN), ft.Text("Teardown complete")]))
        except Exception:
            pass
        update_train_placeholders(); await refresh_teardown_ui_fn(); await safe_update(page)
    finally:
        try:
            td_busy.visible = False
            await safe_update(page)
        except Exception:
            pass

async def run_pod_training(
    *,
    page: ft.Page,
    rp_pod_module,
    train_state: dict,
    cancel_train: dict,
    _runpod_cfg: dict,
    rp_temp_key_tf: ft.TextField,
    config_mode_dd: ft.Dropdown,
    config_files_dd: ft.Dropdown,
    skill_level: ft.Dropdown,
    beginner_mode_dd: ft.Dropdown,
    expert_gpu_dd: ft.Dropdown,
    expert_spot_cb: ft.Checkbox,
    base_model: ft.TextField,
    train_source: ft.Dropdown,
    train_hf_repo: ft.TextField,
    train_hf_split: ft.Dropdown,
    train_json_path: ft.TextField,
    train_timeline: ft.ListView,
    train_progress: ft.ProgressBar,
    train_prog_label: ft.Text,
    start_train_btn: ft.Control,
    stop_train_btn: ft.Control,
    refresh_train_btn: ft.Control,
    restart_container_btn: ft.Control,
    open_runpod_btn: ft.Control,
    open_web_terminal_btn: ft.Control,
    copy_ssh_btn: ft.Control,
    auto_terminate_cb: ft.Checkbox,
    update_train_placeholders: Callable[[], None],
    refresh_teardown_ui_fn: Callable[[], Awaitable[None]],
    update_progress_from_logs: Callable[[List[str]], None],
    build_hp_fn: Callable[[], dict],
    on_pod_created: Optional[Callable[[dict], Awaitable[None]]] = None,
) -> None:
    if train_state.get("running"):
        return

    cancel_train["cancelled"] = False
    train_state["running"] = True

    infra = train_state.get("infra") or {}
    tpl_id = ((infra.get("template") or {}).get("id") or "").strip()
    vol_id = ((infra.get("volume") or {}).get("id") or "").strip()
    if not tpl_id or not vol_id:
        train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.WARNING, color=COLORS.RED), ft.Text("Runpod infrastructure not ready. Click Ensure Infrastructure first.")]))
        train_state["running"] = False
        update_train_placeholders(); await safe_update(page)
        return

    saved_key = ((train_state.get("api_key") or (_runpod_cfg.get("api_key") if isinstance(_runpod_cfg, dict) else "") or "").strip())
    temp_key = (rp_temp_key_tf.value or "").strip()
    api_key = saved_key or temp_key or (os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not api_key:
        train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.WARNING, color=COLORS.RED), ft.Text("Runpod API key missing. Set it in Settings or temp field.")]))
        train_state["running"] = False
        update_train_placeholders(); await safe_update(page)
        return

    try:
        train_state["api_key"] = api_key
    except Exception:
        pass
    try:
        start_train_btn.disabled = True
        start_train_btn.visible = False
        stop_train_btn.disabled = False
        refresh_train_btn.disabled = False
    except Exception:
        pass
    train_state["log_seen"] = set()
    try:
        train_progress.value = 0.0
        train_prog_label.value = "Starting..."
        train_state["progress"] = 0.0
    except Exception:
        pass
    await safe_update(page)

    using_loaded_hp = False
    hp = None
    cfg = {}
    try:
        mode_val = (config_mode_dd.value or "Normal").lower()
        if mode_val.startswith("config"):
            cfg = train_state.get("loaded_config") or {}
            cfg_hp = cfg.get("hp") if isinstance(cfg, dict) else None
            if not (isinstance(cfg_hp, dict) and cfg_hp):
                try:
                    name = (config_files_dd.value or "").strip()
                    if name:
                        from helpers.training import build_hp_from_controls  # lazy import safe
                        # If no saved hp in config, rely on current UI controls
                        cfg_hp = build_hp_from_controls(
                            train_source=train_source,
                            train_hf_repo=train_hf_repo,
                            train_hf_split=train_hf_split,
                            train_hf_config=ft.TextField(value=""),
                            train_json_path=train_json_path,
                            base_model=base_model,
                            out_dir_tf=ft.TextField(value="/data/outputs/runpod_run"),
                            epochs_tf=ft.TextField(value="3"),
                            lr_tf=ft.TextField(value="2e-4"),
                            batch_tf=ft.TextField(value="2"),
                            grad_acc_tf=ft.TextField(value="4"),
                            max_steps_tf=ft.TextField(value="200"),
                            use_lora_cb=ft.Checkbox(value=False),
                            packing_cb=ft.Checkbox(value=False),
                            auto_resume_cb=ft.Checkbox(value=False),
                            push_cb=ft.Checkbox(value=False),
                            hf_repo_id_tf=ft.TextField(value=""),
                            resume_from_tf=ft.TextField(value=""),
                            warmup_steps_tf=ft.TextField(value=""),
                            weight_decay_tf=ft.TextField(value=""),
                            lr_sched_dd=ft.Dropdown(value=""),
                            optim_dd=ft.Dropdown(value=""),
                            logging_steps_tf=ft.TextField(value=""),
                            logging_first_step_cb=ft.Checkbox(value=False),
                            disable_tqdm_cb=ft.Checkbox(value=False),
                            seed_tf=ft.TextField(value=""),
                            save_strategy_dd=ft.Dropdown(value=""),
                            save_total_limit_tf=ft.TextField(value=""),
                            pin_memory_cb=ft.Checkbox(value=False),
                            report_to_dd=ft.Dropdown(value=""),
                            fp16_cb=ft.Checkbox(value=False),
                            bf16_cb=ft.Checkbox(value=False),
                        )
                except Exception:
                    pass
            if isinstance(cfg_hp, dict) and cfg_hp:
                hp = dict(cfg_hp)
                using_loaded_hp = True
    except Exception:
        pass
    if not isinstance(hp, dict) or not hp:
        hp = build_hp_fn()

    # Ensure dataset flags present when not in config mode
    mode_now = (config_mode_dd.value or "Normal").lower()
    if not mode_now.startswith("config"):
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
                train_timeline.controls.append(ft.Row([
                    ft.Icon(ft.Icons.WARNING, color=COLORS.RED),
                    ft.Text("Dataset not set. Provide a Hugging Face dataset or JSON path before starting."),
                ]))
                train_state["running"] = False
                update_train_placeholders(); await safe_update(page)
                return
        except Exception:
            pass

    # Beginner mode presets
    meta_cfg = (cfg.get("meta") or {}) if isinstance(cfg, dict) else {}
    level_src = (meta_cfg.get("skill_level") or skill_level.value or "Beginner")
    begin_src = (meta_cfg.get("beginner_mode") or beginner_mode_dd.value or "Fastest")
    level = level_src.lower()
    is_beginner = (level == "beginner")
    beginner_mode = begin_src.lower() if is_beginner else ""

    chosen_gpu_type_id = "AUTO"
    chosen_interruptible = False
    chosen_by = ""

    if not is_beginner:
        try:
            exp_val = (expert_gpu_dd.value or "AUTO").strip()
            if exp_val and exp_val != "AUTO":
                chosen_gpu_type_id = exp_val
                chosen_interruptible = bool(getattr(expert_spot_cb, "value", False))
                chosen_by = "expert"
        except Exception:
            pass
    pod_cfg = (cfg.get("pod") or {}) if using_loaded_hp else {}
    if not chosen_by and isinstance(pod_cfg, dict) and pod_cfg:
        try:
            if pod_cfg.get("gpu_type_id"):
                chosen_gpu_type_id = pod_cfg.get("gpu_type_id")
            if "interruptible" in pod_cfg:
                chosen_interruptible = bool(pod_cfg.get("interruptible"))
            chosen_by = "config"
        except Exception:
            pass
    elif is_beginner:
        if beginner_mode == "fastest":
            chosen_gpu_type_id = "AUTO"
            chosen_interruptible = False
        elif beginner_mode == "cheapest":
            dc_id = (((infra.get("volume") or {}).get("dc") or "").strip()) or "US-NC-1"
            try:
                cheapest_gpu, is_spot = await asyncio.to_thread(rp_pod_module.discover_cheapest_gpu, api_key, dc_id, 1)
                chosen_gpu_type_id = cheapest_gpu
                chosen_interruptible = bool(is_spot)
            except Exception as e:
                train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.WARNING, color=COLORS.RED), ft.Text(f"Cheapest GPU discovery failed, using AUTO: {e}")]))

    model = base_model.value or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    if hp.get("hf_dataset_id"):
        ds_desc = f"HF: {hp.get('hf_dataset_id')} [{hp.get('hf_dataset_split','train')}]"
    elif hp.get("json_path"):
        ds_desc = f"JSON: {hp.get('json_path')}"
    else:
        ds_desc = "Dataset: (unset)"
    train_timeline.controls.append(ft.Row([ft.Icon(getattr(ft.Icons, "SCIENCE", ft.Icons.PLAY_CIRCLE), color=ACCENT_COLOR), ft.Text("Creating Runpod pod and starting training…")]))
    train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.TABLE_VIEW, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"Dataset: {ds_desc}")]))
    train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.SETTINGS, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"Model={model} • Epochs={hp.get('epochs')} • LR={hp.get('lr')} • BSZ={hp.get('bsz')} • GA={hp.get('grad_accum')}")]))
    if is_beginner:
        try:
            if beginner_mode == "fastest":
                bm_text = "Beginner: Fastest — using best GPU (secure) with aggressive params"
            else:
                bm_text = f"Beginner: Cheapest — selecting lowest-cost GPU ({'spot' if chosen_interruptible else 'secure'}) with conservative params"
            train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.SETTINGS, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(bm_text)]))
        except Exception:
            pass
    elif (chosen_by == "expert"):
        try:
            sel_id = chosen_gpu_type_id
            mode_txt = "spot" if chosen_interruptible else "secure"
            train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.SETTINGS, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"Expert: Using GPU {sel_id} ({mode_txt})")]))
        except Exception:
            pass
    update_train_placeholders(); await safe_update(page)

    # Create pod
    try:
        def _mk_pod():
            return rp_pod_module.create_pod(
                api_key=api_key,
                template_id=tpl_id,
                volume_id=vol_id,
                pod_name=f"ds-train-{int(time.time())}",
                hp=hp,
                gpu_type_id=chosen_gpu_type_id,
                gpu_count=1,
                interruptible=chosen_interruptible
            )
        pod = await asyncio.to_thread(_mk_pod)
        pod_id = pod.get("id") or pod.get("podId") or pod.get("pod_id")
        train_state["pod_id"] = pod_id
        train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.CLOUD, color=ACCENT_COLOR), ft.Text(f"Pod created: {pod_id}")]))
        try:
            await refresh_teardown_ui_fn()
        except Exception:
            pass
        try:
            restart_container_btn.disabled = False
            open_runpod_btn.disabled = False
            open_web_terminal_btn.disabled = False
            copy_ssh_btn.disabled = False
        except Exception:
            pass
        # Callback to allow the caller to perform additional actions (e.g., prompt to save config)
        if on_pod_created is not None:
            try:
                await on_pod_created({
                    "hp": hp,
                    "pod": pod,
                    "pod_id": pod_id,
                    "chosen_gpu_type_id": chosen_gpu_type_id,
                    "chosen_interruptible": chosen_interruptible,
                    "skill_level": skill_level.value,
                    "beginner_mode": beginner_mode if is_beginner else "",
                })
            except Exception:
                pass
    except Exception as e:
        train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.ERROR, color=COLORS.RED), ft.Text(f"Pod create failed: {e}")]))
        train_state["running"] = False
        await safe_update(page)
        return

    # Poll status
    try:
        last_state = None
        while True:
            if cancel_train.get("cancelled"):
                try:
                    await asyncio.to_thread(rp_pod_module.delete_pod, api_key, train_state.get("pod_id"))
                    train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.CANCEL, color=COLORS.RED), ft.Text("Cancel requested — pod termination sent")]))
                except Exception as ex:
                    train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.ERROR, color=COLORS.RED), ft.Text(f"Failed to terminate pod: {ex}")]))
                break
            pod = await asyncio.to_thread(rp_pod_module.get_pod, api_key, train_state.get("pod_id"))
            state = (rp_pod_module.state_of(pod) or "").upper()
            if state != last_state:
                train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.TASK_ALT, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"Pod state: {state}")]))
                last_state = state
            if state in rp_pod_module.TERMINAL_STATES:
                try:
                    train_progress.value = 1.0
                    train_prog_label.value = "Progress: 100%"
                    train_state["progress"] = 1.0
                except Exception:
                    pass
                try:
                    await safe_update(page)
                except Exception:
                    pass
                if bool(getattr(auto_terminate_cb, "value", False)):
                    try:
                        await asyncio.to_thread(rp_pod_module.delete_pod, api_key, train_state.get("pod_id"))
                        train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.DELETE_FOREVER, color=COLORS.RED), ft.Text("Auto-terminate enabled — pod deleted after training finished")]))
                    except Exception as ex:
                        train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.ERROR, color=COLORS.RED), ft.Text(f"Failed to auto-delete pod: {ex}")]))
                break
            try:
                lines = await asyncio.to_thread(rp_pod_module.get_pod_logs, api_key, train_state.get("pod_id"), 200)
            except Exception:
                lines = []
            seen = train_state.get("log_seen") or set()
            new_lines = []
            for ln in (lines or []):
                s = str(ln)
                if s not in seen:
                    new_lines.append(s)
                    seen.add(s)
                    if len(seen) > 5000:
                        seen = set(list(seen)[-2000:])
            train_state["log_seen"] = seen
            if new_lines:
                for s in new_lines:
                    _log_icon = getattr(ft.Icons, "ARTICLE", getattr(ft.Icons, "TERMINAL", getattr(ft.Icons, "DESCRIPTION", ft.Icons.TASK_ALT)))
                    train_timeline.controls.append(ft.Row([ft.Icon(_log_icon, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(s)]))
                update_train_placeholders()
                try:
                    if len(train_timeline.controls) > 1200:
                        train_timeline.controls[:] = train_timeline.controls[-900:]
                except Exception:
                    pass
                try:
                    update_progress_from_logs(new_lines)
                except Exception:
                    pass
            await safe_update(page)
            await asyncio.sleep(3.0)

        train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.CHECK_CIRCLE, color=COLORS.GREEN), ft.Text(f"Training finished with state: {last_state}")]))
    except Exception as e:
        train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.ERROR, color=COLORS.RED), ft.Text(f"Polling error: {e}")]))
    finally:
        train_state["running"] = False
        try:
            start_train_btn.visible = True
            start_train_btn.disabled = False
            stop_train_btn.disabled = True
            refresh_train_btn.disabled = False
            restart_container_btn.disabled = True
            open_runpod_btn.disabled = True
            open_web_terminal_btn.disabled = True
            copy_ssh_btn.disabled = True
        except Exception:
            pass
        await safe_update(page)


async def restart_pod_container(
    *,
    page: ft.Page,
    rp_pod_module,
    train_state: dict,
    config_mode_dd: ft.Dropdown,
    config_files_dd: ft.Dropdown,
    train_timeline: ft.ListView,
    update_train_placeholders: Callable[[], None],
    build_hp_fn: Callable[[], dict],
) -> None:
    try:
        pod_id = (train_state.get("pod_id") or "").strip()
        if not pod_id:
            return
        api_key = (train_state.get("api_key") or "").strip() or (os.environ.get("RUNPOD_API_KEY") or "").strip()
        if not api_key:
            return
        hp = None
        try:
            mode_val = (config_mode_dd.value or "Normal").lower()
            if mode_val.startswith("config"):
                cfg = train_state.get("loaded_config") or {}
                cfg_hp = cfg.get("hp") if isinstance(cfg, dict) else None
                if isinstance(cfg_hp, dict) and cfg_hp:
                    hp = dict(cfg_hp)
        except Exception:
            pass
        if not isinstance(hp, dict) or not hp:
            hp = build_hp_fn()
        cmd = rp_pod_module.build_cmd(hp)
        await asyncio.to_thread(rp_pod_module.patch_pod_docker_start_cmd, api_key, pod_id, cmd)
        train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.RESTART_ALT, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text("Container restarting with new hyper-params…")]))
        update_train_placeholders(); await safe_update(page)
    except Exception as e:
        train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.ERROR, color=COLORS.RED), ft.Text(f"Restart failed: {e}")]))
        update_train_placeholders(); await safe_update(page)


def open_runpod(page: ft.Page, train_state: dict, train_timeline: ft.ListView) -> None:
    try:
        pod_id = (train_state.get("pod_id") or "").strip()
        if not pod_id:
            return
        url = f"https://www.runpod.io/console/pods/{pod_id}"
        try:
            page.launch_url(url)
        except Exception:
            train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.OPEN_IN_NEW, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(url)]))
            try:
                page.update()
            except Exception:
                pass
    except Exception:
        pass


def open_web_terminal(page: ft.Page, train_state: dict, train_timeline: ft.ListView) -> None:
    try:
        pod_id = (train_state.get("pod_id") or "").strip()
        if not pod_id:
            return
        url = f"https://console.runpod.io/pods/{pod_id}"
        try:
            page.launch_url(url)
        except Exception:
            train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.OPEN_IN_NEW, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(url)]))
            page.update()
    except Exception:
        pass


async def copy_ssh_command(
    *,
    page: ft.Page,
    rp_pod_module,
    train_state: dict,
    train_timeline: ft.ListView,
) -> None:
    pod_id = (train_state.get("pod_id") or "").strip()
    if not pod_id:
        return
    api_key = (train_state.get("api_key") or os.environ.get("RUNPOD_API_KEY") or "").strip()
    cmd = None
    try:
        info = await asyncio.to_thread(rp_pod_module.get_pod, api_key, pod_id)
        port_map = info.get("portMappings") or {}
        public_ip = info.get("publicIp") or ""
        ssh_port = None
        if isinstance(port_map, dict):
            ssh_port = port_map.get("22")
        if public_ip and ssh_port:
            cmd = f"ssh root@{public_ip} -p {ssh_port} -i ~/.ssh/id_ed25519"
    except Exception:
        cmd = None
    if cmd:
        try:
            page.set_clipboard(cmd)
            page.snack_bar = ft.SnackBar(ft.Text("SSH command copied to clipboard"))
            page.open(page.snack_bar)
            await safe_update(page)
            return
        except Exception:
            train_timeline.controls.append(ft.Row([ft.Icon(getattr(ft.Icons, "CONTENT_COPY", ft.Icons.LINK), color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(cmd)]))
            await safe_update(page)
            return
    url = f"https://console.runpod.io/pods/{pod_id}"
    try:
        page.launch_url(url)
    except Exception:
        train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.OPEN_IN_NEW, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(url)]))
    page.snack_bar = ft.SnackBar(ft.Text("Open the pod → Connect → SSH tab to copy the proxy command."))
    page.open(page.snack_bar)
    await safe_update(page)


async def ensure_infrastructure(
    *,
    page: ft.Page,
    rp_infra_module,
    _hf_cfg: dict,
    _runpod_cfg: dict,
    rp_dc_tf: ft.TextField,
    rp_vol_name_tf: ft.TextField,
    rp_vol_size_tf: ft.TextField,
    rp_resize_cb: ft.Checkbox,
    rp_tpl_name_tf: ft.TextField,
    rp_image_tf: ft.TextField,
    rp_container_disk_tf: ft.TextField,
    rp_volume_in_gb_tf: ft.TextField,
    rp_mount_path_tf: ft.TextField,
    rp_category_tf: ft.TextField,
    rp_public_cb: ft.Checkbox,
    rp_temp_key_tf: ft.TextField,
    rp_infra_busy: ft.ProgressRing,
    train_timeline: ft.ListView,
    refresh_expert_gpus_fn: Callable[[], None],
    refresh_teardown_ui_fn: Callable[[], Awaitable[None]],
    dataset_section: ft.Container,
    train_params_section: ft.Container,
    start_train_btn: ft.Control,
    stop_train_btn: ft.Control,
    refresh_train_btn: ft.Control,
    _update_mode_visibility: Callable[[], None],
) -> None:
    # Validate infra module early
    if (rp_infra_module is None) or (not hasattr(rp_infra_module, "ensure_infrastructure")):
        try:
            train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.WARNING, color=COLORS.RED), ft.Text("Runpod infrastructure module unavailable. Please install or update Runpod helpers.")]))
            await safe_update(page)
        except Exception:
            pass
        return
    # Resolve API key: Settings > temp (this tab) > env
    saved_key = ((_runpod_cfg.get("api_key") or "") if isinstance(_runpod_cfg, dict) else "").strip()
    temp_key = (rp_temp_key_tf.value or "").strip()
    key = saved_key or temp_key or (os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not key:
        try:
            train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.WARNING, color=COLORS.RED), ft.Text("Runpod API key missing. Set it in Settings → Runpod API Access.")]))
            await safe_update(page)
        except Exception:
            pass
        return

    # Read params with defaults
    dc = (rp_dc_tf.value or "US-NC-1").strip()
    vol_name = (rp_vol_name_tf.value or "unsloth-volume").strip()
    vol_size_s = (rp_vol_size_tf.value or "50").strip()
    resize = bool(getattr(rp_resize_cb, "value", True))
    tpl_name = (rp_tpl_name_tf.value or "unsloth-trainer-template").strip()
    image = (rp_image_tf.value or "docker.io/sbussiso/unsloth-trainer:latest").strip()
    container_disk_s = (rp_container_disk_tf.value or "30").strip()
    vol_in_gb_s = (rp_volume_in_gb_tf.value or "0").strip()
    mount_path = (rp_mount_path_tf.value or "/data").strip()
    category = (rp_category_tf.value or "NVIDIA").strip()
    is_public = bool(getattr(rp_public_cb, "value", False))

    try:
        vol_size = int(float(vol_size_s))
    except Exception:
        vol_size = 50
    try:
        container_disk = int(float(container_disk_s))
    except Exception:
        container_disk = 30
    try:
        vol_in_gb = int(float(vol_in_gb_s))
    except Exception:
        vol_in_gb = 0

    train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.CLOUD, color=ACCENT_COLOR), ft.Text("Ensuring Runpod infrastructure (volume + template)…")]))
    train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.SETTINGS, color=WITH_OPACITY(0.9, COLORS.BLUE)), ft.Text(f"DC={dc} • Vol={vol_name} ({vol_size}GB) • Tpl={tpl_name}")]))
    await safe_update(page)

    try:
        rp_infra_busy.visible = True
        await safe_update(page)
    except Exception:
        pass

    # Build env vars for the template (inject HF token if available)
    hf_tok = ""
    try:
        hf_tok = (((_hf_cfg.get("token") or "") if isinstance(_hf_cfg, dict) else "").strip())
    except Exception:
        hf_tok = ""
    if not hf_tok:
        try:
            hf_tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
        except Exception:
            hf_tok = ""
    if not hf_tok:
        try:
            from huggingface_hub import HfFolder  # type: ignore
            hf_tok = getattr(HfFolder, "get_token", lambda: "")() or ""
        except Exception:
            pass

    tpl_env = {"PYTHONUNBUFFERED": "1"}
    if hf_tok:
        tpl_env["HF_TOKEN"] = hf_tok
        tpl_env["HUGGINGFACE_HUB_TOKEN"] = hf_tok
        if is_public:
            try:
                train_timeline.controls.append(ft.Row([
                    ft.Icon(ft.Icons.WARNING, color=COLORS.ORANGE),
                    ft.Text("Template is Public — environment variables (including HF token) may be visible to others."),
                ]))
                await safe_update(page)
            except Exception:
                pass

    def do_call():
        return rp_infra_module.ensure_infrastructure(
            api_key=key,
            datacenter_id=dc,
            volume_name=vol_name,
            volume_size_gb=vol_size,
            resize_if_smaller=resize,
            template_name=tpl_name,
            image_name=image,
            container_disk_gb=container_disk,
            volume_in_gb=vol_in_gb,
            volume_mount_path=mount_path,
            category=category,
            is_public=is_public,
            env_vars=tpl_env,
            ports=[],
        )

    try:
        result = await asyncio.to_thread(do_call)
        vol = result.get("volume", {})
        tpl = result.get("template", {})
        try:
            train_state["infra"] = result
            train_state["api_key"] = key
        except Exception:
            pass
        try:
            refresh_expert_gpus_fn()
        except Exception:
            pass
        train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.DONE_ALL, color=COLORS.GREEN), ft.Text(f"Volume {vol.get('action')} — id={vol.get('id')} size={vol.get('size')}GB")]))
        train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.DONE_ALL, color=COLORS.GREEN), ft.Text(f"Template {tpl.get('action')} — id={tpl.get('id')} image={tpl.get('image')}")]))
        try:
            await refresh_teardown_ui_fn()
        except Exception:
            pass
        try:
            # Enable training controls once infra is ready
            dataset_section.visible = True
            train_params_section.visible = True
            start_train_btn.disabled = False
            stop_train_btn.disabled = False
            refresh_train_btn.disabled = False
            _update_mode_visibility()
        except Exception:
            pass
        await safe_update(page)
    except Exception as e:
        msg = str(e)
        train_timeline.controls.append(ft.Row([ft.Icon(ft.Icons.ERROR, color=COLORS.RED), ft.Text(f"Infra setup failed: {msg}")]))
    finally:
        try:
            rp_infra_busy.visible = False
        except Exception:
            pass
        await safe_update(page)


async def confirm_teardown_selected(
    *,
    page: ft.Page,
    td_template_cb: ft.Checkbox,
    td_volume_cb: ft.Checkbox,
    td_pod_cb: ft.Checkbox,
    td_busy: ft.ProgressRing,
    train_timeline: ft.ListView,
    update_train_placeholders: Callable[[], None],
    _runpod_cfg: dict,
    rp_temp_key_tf: ft.TextField,
    rp_pod_module,
    rp_infra_module,
    train_state: dict,
    refresh_teardown_ui_fn: Callable[[], Awaitable[None]],
) -> None:
    # Timeline + snackbar feedback
    try:
        train_timeline.controls.append(ft.Row([ft.Icon(getattr(ft.Icons, "WARNING_AMBER", ft.Icons.WARNING), color=WITH_OPACITY(0.9, COLORS.ORANGE)), ft.Text("Teardown Selected clicked")]))
        update_train_placeholders(); await safe_update(page)
    except Exception:
        pass
    try:
        page.snack_bar = ft.SnackBar(ft.Text("Preparing teardown confirmation..."))
        page.open(page.snack_bar)
        await safe_update(page)
    except Exception:
        pass

    # Determine items selected
    items = []
    if bool(getattr(td_pod_cb, "value", False)):
        items.append("Pod")
    if bool(getattr(td_template_cb, "value", False)):
        items.append("Template")
    if bool(getattr(td_volume_cb, "value", False)):
        items.append("Volume")

    if not items:
        try:
            page.snack_bar = ft.SnackBar(ft.Text("Select at least one item to teardown."))
            page.open(page.snack_bar)
            await safe_update(page)
        except Exception:
            pass
        return

    msg = "This will delete: " + ", ".join(items) + "."

    confirm_dlg = ft.AlertDialog(
        modal=True,
        title=ft.Row([
            ft.Icon(getattr(ft.Icons, "DELETE_FOREVER", getattr(ft.Icons, "DELETE", ft.Icons.CLOSE)), color=COLORS.RED),
            ft.Text("Confirm Teardown"),
        ], alignment=ft.MainAxisAlignment.START),
        content=ft.Text(msg),
        actions=[],
    )

    def _close():
        try:
            confirm_dlg.open = False
            page.update()
        except Exception:
            pass

    def _on_delete(_):
        _close()
        # Schedule the actual teardown with selected_all=False
        try:
            page.run_task(lambda: do_teardown(
                page=page,
                rp_pod_module=rp_pod_module,
                rp_infra_module=rp_infra_module,
                train_state=train_state,
                td_template_cb=td_template_cb,
                td_volume_cb=td_volume_cb,
                td_pod_cb=td_pod_cb,
                td_busy=td_busy,
                train_timeline=train_timeline,
                update_train_placeholders=update_train_placeholders,
                _runpod_cfg=_runpod_cfg,
                rp_temp_key_tf=rp_temp_key_tf,
                refresh_teardown_ui_fn=refresh_teardown_ui_fn,
                selected_all=False,
            ))
        except Exception:
            pass

    try:
        confirm_dlg.actions = [
            ft.TextButton("Cancel", on_click=lambda e: _close()),
            ft.ElevatedButton("Delete", icon=getattr(ft.Icons, "CHECK", ft.Icons.DELETE), on_click=_on_delete),
        ]
    except Exception:
        pass

    try:
        if hasattr(page, "open") and callable(getattr(page, "open")):
            page.open(confirm_dlg)
        else:
            page.dialog = confirm_dlg
            confirm_dlg.open = True
        await safe_update(page)
    except Exception:
        try:
            page.dialog = confirm_dlg
            confirm_dlg.open = True
            await safe_update(page)
        except Exception:
            pass


async def confirm_teardown_all(
    *,
    page: ft.Page,
    td_template_cb: ft.Checkbox,
    td_volume_cb: ft.Checkbox,
    td_pod_cb: ft.Checkbox,
    td_busy: ft.ProgressRing,
    train_timeline: ft.ListView,
    update_train_placeholders: Callable[[], None],
    _runpod_cfg: dict,
    rp_temp_key_tf: ft.TextField,
    rp_pod_module,
    rp_infra_module,
    train_state: dict,
    refresh_teardown_ui_fn: Callable[[], Awaitable[None]],
) -> None:
    # Timeline + snackbar feedback
    try:
        train_timeline.controls.append(ft.Row([ft.Icon(getattr(ft.Icons, "WARNING_AMBER", ft.Icons.WARNING), color=WITH_OPACITY(0.9, COLORS.ORANGE)), ft.Text("Teardown All clicked")]))
        update_train_placeholders(); await safe_update(page)
    except Exception:
        pass
    try:
        page.snack_bar = ft.SnackBar(ft.Text("Preparing teardown confirmation..."))
        page.snack_bar.open = True
        await safe_update(page)
    except Exception:
        pass

    confirm_dlg = ft.AlertDialog(
        modal=True,
        title=ft.Row([
            ft.Icon(getattr(ft.Icons, "DELETE_FOREVER", getattr(ft.Icons, "DELETE", ft.Icons.CLOSE)), color=COLORS.RED),
            ft.Text("Teardown All infrastructure?"),
        ], alignment=ft.MainAxisAlignment.START),
        content=ft.Text("This will delete the Runpod Template and Network Volume. If a pod exists, it will be deleted first."),
        actions=[],
    )

    def _close():
        try:
            confirm_dlg.open = False
            page.update()
        except Exception:
            pass

    def _on_delete_all(_):
        _close()
        # Schedule teardown with selected_all=True
        try:
            page.run_task(lambda: do_teardown(
                page=page,
                rp_pod_module=rp_pod_module,
                rp_infra_module=rp_infra_module,
                train_state=train_state,
                td_template_cb=td_template_cb,
                td_volume_cb=td_volume_cb,
                td_pod_cb=td_pod_cb,
                td_busy=td_busy,
                train_timeline=train_timeline,
                update_train_placeholders=update_train_placeholders,
                _runpod_cfg=_runpod_cfg,
                rp_temp_key_tf=rp_temp_key_tf,
                refresh_teardown_ui_fn=refresh_teardown_ui_fn,
                selected_all=True,
            ))
        except Exception:
            pass

    try:
        confirm_dlg.actions = [
            ft.TextButton("Cancel", on_click=lambda e: _close()),
            ft.ElevatedButton("Delete All", icon=getattr(ft.Icons, "CHECK", ft.Icons.DELETE), on_click=_on_delete_all),
        ]
    except Exception:
        pass

    try:
        if hasattr(page, "open") and callable(getattr(page, "open")):
            page.open(confirm_dlg)
        else:
            page.dialog = confirm_dlg
            confirm_dlg.open = True
        await safe_update(page)
    except Exception:
        try:
            page.dialog = confirm_dlg
            confirm_dlg.open = True
            await safe_update(page)
        except Exception:
            pass
