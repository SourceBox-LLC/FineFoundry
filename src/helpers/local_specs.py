from __future__ import annotations

import os
import platform
import ctypes
import shutil
import subprocess
import re
from typing import Optional, List

import flet as ft


def _bytes_to_gb(b: int) -> float:
    try:
        return round(float(b) / (1024 ** 3), 1)
    except Exception:
        return 0.0


def _total_ram_bytes() -> Optional[int]:
    # Try POSIX sysconf
    try:
        if (
            hasattr(os, "sysconf") and hasattr(os, "sysconf_names")
            and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names
        ):
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            if isinstance(page_size, int) and isinstance(phys_pages, int):
                return int(page_size) * int(phys_pages)
    except Exception:
        pass
    # Windows via ctypes
    try:
        if platform.system().lower().startswith("win"):
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint),
                    ("dwMemoryLoad", ctypes.c_uint),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "kernel32") and hasattr(ctypes.windll.kernel32, "GlobalMemoryStatusEx"):
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    return int(stat.ullTotalPhys)
    except Exception:
        pass
    # Optional psutil fallback if installed
    try:
        import psutil
        return int(getattr(psutil, "virtual_memory")().total)
    except Exception:
        return None


def _probe_gpus_via_torch():
    gpus = []
    cuda_ok = False
    torch_ok = False
    try:
        import torch
        torch_ok = True
        cuda_ok = bool(torch.cuda.is_available())
        if cuda_ok:
            cnt = int(torch.cuda.device_count())
            for i in range(cnt):
                try:
                    props = torch.cuda.get_device_properties(i)
                    name = getattr(props, "name", f"GPU {i}")
                    vram_b = int(getattr(props, "total_memory", 0))
                    gpus.append({"index": i, "name": str(name), "vram_gb": _bytes_to_gb(vram_b)})
                except Exception:
                    gpus.append({"index": i, "name": f"GPU {i}", "vram_gb": None})
    except Exception:
        pass
    return torch_ok, cuda_ok, gpus


def _probe_gpus_via_nvidia_smi():
    gpus: List[dict] = []
    try:
        if shutil.which("nvidia-smi"):
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                stderr=subprocess.STDOUT,
                timeout=3,
            ).decode("utf-8", errors="ignore").strip().splitlines()
            for idx, line in enumerate(out):
                try:
                    parts = [p.strip() for p in line.split(",")]
                    name = parts[0]
                    mem_part = parts[1] if len(parts) > 1 else ""
                    # e.g., "8192 MiB"
                    vram_gb = None
                    m = re.search(r"(\d+)\s*MiB", mem_part, re.IGNORECASE)
                    if m:
                        vram_gb = round(int(m.group(1)) / 1024.0, 1)
                    gpus.append({"index": idx, "name": name, "vram_gb": vram_gb})
                except Exception:
                    pass
    except Exception:
        pass
    return gpus


def gather_local_specs() -> dict:
    try:
        os_name = platform.platform()
    except Exception:
        os_name = platform.system()
    py_ver = platform.python_version()
    cpu_cores = os.cpu_count() or 0
    ram_b = _total_ram_bytes()
    ram_gb = _bytes_to_gb(ram_b) if ram_b else None
    try:
        du = shutil.disk_usage(os.path.abspath(os.sep))
        disk_free_gb = _bytes_to_gb(du.free)
    except Exception:
        disk_free_gb = None

    torch_ok, cuda_ok, gpus = _probe_gpus_via_torch()
    if not gpus:
        gpus = _probe_gpus_via_nvidia_smi()

    capability = "Unknown"
    try:
        max_vram = max([g.get("vram_gb") or 0 for g in gpus]) if gpus else 0
    except Exception:
        max_vram = 0
    if not torch_ok:
        capability = "PyTorch not installed — GPU training unsupported."
    elif not cuda_ok:
        capability = "CUDA not available — GPU training unavailable (CPU-only)."
    else:
        if max_vram >= 20:
            capability = "OK for LoRA on 7B–13B; 7B full FT may be possible with tweaks."
        elif max_vram >= 12:
            capability = "Good for 7B LoRA; full FT unlikely."
        elif max_vram >= 8:
            capability = "7B LoRA may work with 4-bit and small batch."
        elif max_vram > 0:
            capability = "Likely inference-only; tiny LoRA may work."

    red_flags: List[str] = []
    if not torch_ok:
        red_flags.append("PyTorch not installed — install a CUDA-enabled torch build for GPU acceleration.")
    if torch_ok and not cuda_ok:
        red_flags.append("CUDA not available — install NVIDIA drivers and a CUDA-enabled torch build.")
    if not gpus:
        red_flags.append("No NVIDIA GPUs detected — GPU fine-tuning will not be possible.")
    try:
        if gpus and max_vram is not None:
            if max_vram < 8:
                red_flags.append("GPU VRAM < 8 GB — local fine-tuning of most 7B models may be infeasible.")
            elif max_vram < 12:
                red_flags.append("GPU VRAM < 12 GB — expect heavy quantization/checkpointing and slower runs.")
    except Exception:
        pass
    if ram_gb is not None and ram_gb < 16:
        red_flags.append("System RAM < 16 GB — dataset preprocessing and training may be constrained.")
    if disk_free_gb is not None and disk_free_gb < 20:
        red_flags.append("Disk free < 20 GB — checkpoints/datasets may fail or be truncated.")
    try:
        py_major, py_minor = int(py_ver.split(".")[0]), int(py_ver.split(".")[1])
        if (py_major, py_minor) < (3, 10):
            red_flags.append("Python < 3.10 detected — some ML libraries may have limited support.")
    except Exception:
        pass

    return {
        "os": os_name,
        "python": py_ver,
        "cpu_cores": cpu_cores,
        "ram_gb": ram_gb,
        "disk_free_gb": disk_free_gb,
        "torch_installed": torch_ok,
        "cuda_available": cuda_ok,
        "gpus": gpus,
        "capability": capability,
        "red_flags": red_flags,
    }


async def refresh_local_gpus(
    *,
    page: ft.Page,
    expert_gpu_busy: ft.Control,
    expert_gpu_dd: ft.Dropdown,
    expert_spot_cb: ft.Checkbox,
    expert_gpu_avail: dict,
) -> None:
    try:
        try:
            expert_gpu_busy.visible = True
            page.update()
        except Exception:
            pass
        data = gather_local_specs()
        gpus = list(data.get("gpus") or [])
        opts = [ft.dropdown.Option(text="AUTO (all local GPUs)", key="AUTO")]
        for g in gpus:
            try:
                idx = g.get("index")
                name = str(g.get("name") or f"GPU {idx}")
                vram = g.get("vram_gb")
                mem_txt = (f" {vram}GB" if isinstance(vram, (int, float)) and vram is not None else "")
                if idx is not None:
                    opts.append(ft.dropdown.Option(text=f"GPU {idx}: {name}{mem_txt}", key=str(idx)))
            except Exception:
                pass
        expert_gpu_avail.clear()
        cur = (expert_gpu_dd.value or "AUTO")
        keys = {getattr(o, 'key', None) or o.text for o in opts}
        expert_gpu_dd.options = opts
        expert_gpu_dd.value = "AUTO" if cur not in keys else cur
        try:
            expert_spot_cb.value = False
            expert_spot_cb.disabled = True
            expert_spot_cb.visible = False
            expert_gpu_dd.tooltip = "Pick a local GPU index or AUTO to allow all local GPUs."
        except Exception:
            pass
        try:
            expert_gpu_busy.visible = False
            page.update()
        except Exception:
            pass
    except Exception:
        try:
            expert_gpu_busy.visible = False
        except Exception:
            pass
