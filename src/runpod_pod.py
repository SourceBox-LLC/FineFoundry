#!/usr/bin/env python3
"""
runpod_pod.py
Lightweight helpers to create/run/monitor/delete a Runpod Pod from a Template
and attach a Network Volume. Includes a simple dockerStartCmd builder that
turns a dict of hyperparameters into CLI flags.

Requires: requests
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

REST = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"

# ---------- HTTP helpers ----------

def _headers(api_key: str) -> Dict[str, str]:
    if not api_key:
        raise ValueError("Runpod API key is required")
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _req(method: str, path: str, api_key: str, **kw) -> requests.Response:
    r = requests.request(method, f"{REST}{path}", headers=_headers(api_key), timeout=90, **kw)
    r.raise_for_status()
    return r


def _gql(api_key: str, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = requests.post(f"{GQL}?api_key={api_key}", json={"query": query, "variables": variables or {}}, timeout=90)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])  # pass through graphql errors
    return data["data"]


# ---------- Command builder from HP dict ----------

def _shquote(v) -> str:
    s = str(v)
    if s == "":
        return "''"
    if any(ch in s for ch in " \t\n\r\"'`$\\"):
        s = s.replace("'", "'\"'\"'")
        return f"'{s}'"
    return s


def build_cmd(hp: Dict[str, Any]) -> List[str]:
    """Return dockerStartCmd list suitable for Runpod template/pod.
    Example: ["bash", "-lc", "python -u /workspace/train.py --epochs 1 --lr 2e-4 ..."]
    Uses flag names exactly as keys (underscores preserved) to match train.py.
    Robustness: try several common locations for train.py and emit a helpful
    message if not found, including a hint about volume mount overlays.
    """
    # Build flags string
    flags: List[str] = []
    for k, v in (hp or {}).items():
        flag = "--" + str(k)
        if isinstance(v, bool):
            if v:
                flags.append(flag)
        elif v is None:
            continue
        else:
            flags.append(f"{flag} {_shquote(v)}")
    flags_str = " ".join(flags)

    # Shell script: pick the first existing train.py path
    script = (
        "set -euo pipefail; "
        "P=; "
        "for CAND in /workspace/train.py /app/train.py /opt/train.py /usr/local/src/train.py /root/train.py; do "
        "  if [ -f \"$CAND\" ]; then P=\"$CAND\"; break; fi; "
        "done; "
        "if [ -z \"$P\" ]; then "
        "  echo 'ERROR: train.py not found in expected locations.' >&2; "
        "  echo 'Hint: If you mounted your network volume at /workspace, it may hide train.py baked into the image.' >&2; "
        "  echo 'Try setting the template Mount path to /data (or another path) and re-run.' >&2; "
        "  ls -la /workspace || true; ls -la /data || true; ls -la / || true; exit 1; "
        "fi; "
        "python -u \"$P\" " + flags_str
    )
    return ["bash", "-lc", script]


# ---------- Volume / DC helpers ----------

def get_volume(api_key: str, volume_id: str) -> Dict[str, Any]:
    vols = _req("GET", "/networkvolumes", api_key).json()
    for v in vols:
        if v.get("id") == volume_id:
            return v
    raise RuntimeError(f"Network Volume not found: {volume_id}")


# ---------- GPU discovery ----------

def discover_best_gpu(api_key: str, dc_id: str, prefer_spot: bool, gpu_count: int) -> Tuple[str, bool]:
    """Return (gpu_type_id, chosen_spot_bool). Raises if none available."""
    types = _gql(api_key, "query { gpuTypes { id displayName memoryInGb } }")['gpuTypes']

    q = (
        "\n"  # noqa: W291
        "  query ($id:String!, $dc:String!, $secure:Boolean!, $count:Int!) {\n"
        "    gpuTypes(input:{id:$id}) {\n"
        "      id displayName memoryInGb\n"
        "      lowestPrice(input:{dataCenterId:$dc, secureCloud:$secure, gpuCount:$count}) {\n"
        "        stockStatus\n"
        "        maxUnreservedGpuCount\n"
        "        availableGpuCounts\n"
        "        uninterruptablePrice\n"
        "        minimumBidPrice\n"
        "      }\n"
        "    }\n"
        "  }\n"
    )

    def probe(secure: bool):
        candidates: List[Dict[str, Any]] = []
        for t in types:
            row = _gql(api_key, q, {"id": t["id"], "dc": dc_id, "secure": secure, "count": int(gpu_count)})["gpuTypes"][0]
            lp = row.get("lowestPrice")
            if not lp:
                continue
            maxu = lp.get("maxUnreservedGpuCount") or 0
            stock = (lp.get("stockStatus") or "").lower()
            if maxu >= int(gpu_count) and stock in {"high", "medium", "low"}:
                candidates.append({
                    "id": row["id"],
                    "mem": row.get("memoryInGb") or 0,
                    "spot": (not secure),
                    "avail": maxu,
                })
        candidates.sort(key=lambda c: (c["mem"], c["avail"]), reverse=True)
        return candidates[0] if candidates else None

    order = ([False, True] if not prefer_spot else [True, False])  # False=secure first if prefer_spot=False
    for secure in order:
        pick = probe(secure)
        if pick:
            return (pick["id"], pick["spot"])  # gpu_type_id, is_spot
    raise RuntimeError(f"No GPUs available in DC {dc_id} right now.")


# ---------- Cheapest GPU discovery ----------
def discover_cheapest_gpu(api_key: str, dc_id: str, gpu_count: int) -> Tuple[str, bool]:
    """Return (gpu_type_id, is_spot) for the cheapest available GPU.
    Prefers spot (interruptible) if available and cheaper, otherwise falls back to secure.
    Raises if none available.
    """
    types = _gql(api_key, "query { gpuTypes { id displayName memoryInGb } }")['gpuTypes']

    q = (
        "\n"  # noqa: W291
        "  query ($id:String!, $dc:String!, $secure:Boolean!, $count:Int!) {\n"
        "    gpuTypes(input:{id:$id}) {\n"
        "      id displayName memoryInGb\n"
        "      lowestPrice(input:{dataCenterId:$dc, secureCloud:$secure, gpuCount:$count}) {\n"
        "        stockStatus\n"
        "        maxUnreservedGpuCount\n"
        "        availableGpuCounts\n"
        "        uninterruptablePrice\n"
        "        minimumBidPrice\n"
        "      }\n"
        "    }\n"
        "  }\n"
    )

    candidates: List[Dict[str, Any]] = []

    def probe(secure: bool):
        for t in types:
            row = _gql(api_key, q, {"id": t["id"], "dc": dc_id, "secure": secure, "count": int(gpu_count)})["gpuTypes"][0]
            lp = row.get("lowestPrice")
            if not lp:
                continue
            maxu = lp.get("maxUnreservedGpuCount") or 0
            stock = (lp.get("stockStatus") or "").lower()
            if maxu >= int(gpu_count) and stock in {"high", "medium", "low"}:
                # Price selection depends on secure vs spot
                price = None
                try:
                    price = float(lp.get("minimumBidPrice")) if not secure else float(lp.get("uninterruptablePrice"))
                except Exception:
                    # If price missing, skip this candidate
                    continue
                candidates.append({
                    "id": row["id"],
                    "mem": row.get("memoryInGb") or 0,
                    "spot": (not secure),
                    "price": price,
                })

    # Probe spot first (usually cheapest), then secure
    probe(False)  # secure=False => spot
    probe(True)   # secure=True  => secure

    if not candidates:
        raise RuntimeError(f"No GPUs available in DC {dc_id} right now.")

    # Sort by price asc, then by memory desc as a tie-breaker
    candidates.sort(key=lambda c: (c["price"], -float(c["mem"])))
    pick = candidates[0]
    return (pick["id"], pick["spot"])  # gpu_type_id, is_spot


# ---------- List available GPUs in a DC ----------
def list_available_gpus(api_key: str, dc_id: str, gpu_count: int = 1) -> List[Dict[str, Any]]:
    """Return a list of GPU types available in the given datacenter with availability flags.
    Each item: {id, displayName, memoryInGb, secureAvailable: bool, spotAvailable: bool}
    Filters out types with no capacity for the requested gpu_count.
    """
    types = _gql(api_key, "query { gpuTypes { id displayName memoryInGb } }")['gpuTypes']

    q = (
        "\n"  # noqa: W291
        "  query ($id:String!, $dc:String!, $secure:Boolean!, $count:Int!) {\n"
        "    gpuTypes(input:{id:$id}) {\n"
        "      id displayName memoryInGb\n"
        "      lowestPrice(input:{dataCenterId:$dc, secureCloud:$secure, gpuCount:$count}) {\n"
        "        stockStatus\n"
        "        maxUnreservedGpuCount\n"
        "      }\n"
        "    }\n"
        "  }\n"
    )

    out: List[Dict[str, Any]] = []
    for t in types:
        row_secure = _gql(api_key, q, {"id": t["id"], "dc": dc_id, "secure": True, "count": int(gpu_count)})["gpuTypes"][0]
        row_spot = _gql(api_key, q, {"id": t["id"], "dc": dc_id, "secure": False, "count": int(gpu_count)})["gpuTypes"][0]
        def _avail(row: Dict[str, Any]) -> bool:
            lp = row.get("lowestPrice") or {}
            try:
                maxu = int(lp.get("maxUnreservedGpuCount") or 0)
            except Exception:
                maxu = 0
            stock = (lp.get("stockStatus") or "").lower()
            return maxu >= int(gpu_count) and stock in {"high", "medium", "low"}
        sec_ok = _avail(row_secure)
        spot_ok = _avail(row_spot)
        if not (sec_ok or spot_ok):
            continue
        out.append({
            "id": t["id"],
            "displayName": t.get("displayName") or t["id"],
            "memoryInGb": t.get("memoryInGb") or 0,
            "secureAvailable": bool(sec_ok),
            "spotAvailable": bool(spot_ok),
        })
    # Prefer larger memory first
    out.sort(key=lambda r: float(r.get("memoryInGb") or 0), reverse=True)
    return out

# ---------- Pod operations ----------

def create_pod(
    *,
    api_key: str,
    template_id: str,
    volume_id: str,
    pod_name: str,
    hp: Dict[str, Any],
    gpu_type_id: str = "AUTO",
    gpu_count: int = 1,
    interruptible: bool = False,
) -> Dict[str, Any]:
    vol = get_volume(api_key, volume_id)
    dc = vol["dataCenterId"]

    chosen_gpu = gpu_type_id
    chosen_spot = bool(interruptible)

    if str(gpu_type_id).upper() == "AUTO":
        chosen_gpu, chosen_spot = discover_best_gpu(api_key, dc_id=dc, prefer_spot=bool(interruptible), gpu_count=int(gpu_count))

    body = {
        "name": pod_name,
        "templateId": template_id,
        "gpuTypeIds": [chosen_gpu],
        "gpuCount": int(gpu_count),
        "interruptible": bool(chosen_spot),
        "computeType": "GPU",
        "cloudType": "COMMUNITY" if chosen_spot else "SECURE",
        "networkVolumeId": volume_id,  # mounts at template's volumeMountPath (default /data)
        "dataCenterIds": [dc],
        "dockerStartCmd": build_cmd(hp),
    }
    return _req("POST", "/pods", api_key, data=json.dumps(body)).json()


def get_pod(api_key: str, pod_id: str) -> Dict[str, Any]:
    return _req("GET", f"/pods/{pod_id}", api_key).json()


def delete_pod(api_key: str, pod_id: str) -> Dict[str, Any]:
    r = requests.delete(f"{REST}/pods/{pod_id}", headers=_headers(api_key), timeout=90)
    if r.status_code >= 400:
        try:
            _ = r.json()
        except Exception:
            _ = r.text
        raise requests.HTTPError(response=r, request=r.request)
    return {"status": "deleted", "pod_id": pod_id}


def get_pod_logs(api_key: str, pod_id: str, limit: int = 200) -> List[str]:
    """Best-effort retrieval of recent pod logs.
    Returns a list of log lines or an empty list if unavailable.
    We try multiple REST paths to be compatible with different API versions.
    """
    paths = [
        f"/pods/{pod_id}/logs",
        f"/pods/{pod_id}/events",
    ]
    for path in paths:
        try:
            r = _req("GET", path, api_key, params={"limit": int(limit)})
            # Prefer JSON, but fall back to plain text if needed
            data = None
            try:
                data = r.json()
            except Exception:
                text = r.text or ""
                if text.strip():
                    return [ln for ln in text.splitlines() if ln.strip()]
                data = None

            if isinstance(data, dict):
                # Format 1: {"logs": "...\n..."}
                if "logs" in data:
                    s = data.get("logs") or ""
                    return [ln for ln in str(s).splitlines() if str(ln).strip()]
                # Format 2: {"events": [{"message": "..."}, ...]}
                if isinstance(data.get("events"), list):
                    lines: List[str] = []
                    for ev in data["events"]:
                        if not isinstance(ev, dict):
                            continue
                        msg = ev.get("message") or ev.get("log") or ev.get("status") or ""
                        if str(msg).strip():
                            lines.append(str(msg))
                    if lines:
                        return lines
        except Exception:
            # Try next path
            continue
    return []


TERMINAL_STATES = {"FAILED", "DELETED", "KILLED", "TERMINATED", "EXITED"}


def state_of(pod: Dict[str, Any]) -> str:
    return pod.get("desiredStatus") or pod.get("status") or pod.get("state") or "UNKNOWN"


# ---------- Pod patch (restart container with new args) ----------
def patch_pod_docker_start_cmd(api_key: str, pod_id: str, docker_start_cmd: List[str]) -> Dict[str, Any]:
    """PATCH a running Pod to update dockerStartCmd. Runpod restarts the container with new args.

    Example docker_start_cmd: ["bash", "-lc", "python -u /workspace/train.py --epochs 2 ..."]
    """
    body = {"dockerStartCmd": list(docker_start_cmd or [])}
    return _req("PATCH", f"/pods/{pod_id}", api_key, data=json.dumps(body)).json()
