"""
Runpod infrastructure helper.
Ensures a Network Volume and a Template exist (idempotent) using Runpod REST API v1.

Exposes simple functions for the app to call. Reads API key from param or RUNPOD_API_KEY env.

Based on a standalone script version used during testing.
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional
import requests

API = "https://rest.runpod.io/v1"

class RunpodError(Exception):
    pass


def _headers(api_key: Optional[str]) -> Dict[str, str]:
    key = (api_key or os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not key:
        raise RunpodError("Runpod API key is missing; set RUNPOD_API_KEY in Settings or pass api_key.")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _req(method: str, path: str, api_key: Optional[str] = None, **kw) -> requests.Response:
    r = requests.request(method, f"{API}{path}", headers=_headers(api_key), timeout=90, **kw)
    r.raise_for_status()
    return r

# ---------- Volume ops ----------

def list_volumes(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    return _req("GET", "/networkvolumes", api_key=api_key).json()


def create_volume(name: str, size_gb: int, datacenter_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    body = {"name": name, "size": int(size_gb), "dataCenterId": datacenter_id}
    return _req("POST", "/networkvolumes", api_key=api_key, data=json.dumps(body)).json()


def resize_volume(vol_id: str, new_size_gb: int, keep_name: Optional[str] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"size": int(new_size_gb)}
    if keep_name:
        body["name"] = keep_name
    return _req("PATCH", f"/networkvolumes/{vol_id}", api_key=api_key, data=json.dumps(body)).json()


def ensure_volume(
    *,
    volume_name: str,
    volume_size_gb: int,
    datacenter_id: str,
    resize_if_smaller: bool = True,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    vols = list_volumes(api_key)
    candidates = [v for v in vols if (v.get("name") or "").lower() == (volume_name or "").lower()]
    if candidates:
        same_dc = [v for v in candidates if v.get("dataCenterId") == datacenter_id] or candidates
        vol = same_dc[0]
        try:
            cur_size = int(vol.get("size", 0))
        except Exception:
            cur_size = 0
        if resize_if_smaller and cur_size < int(volume_size_gb):
            vol = resize_volume(vol["id"], int(volume_size_gb), keep_name=vol.get("name"), api_key=api_key)
            return {"action": "resized", "volume": vol}
        return {"action": "exists", "volume": vol}
    if not datacenter_id:
        raise RunpodError("DATACENTER_ID is required to create a new Network Volume.")
    vol = create_volume(volume_name, int(volume_size_gb), datacenter_id, api_key=api_key)
    return {"action": "created", "volume": vol}

# ---------- Template ops ----------

def list_templates(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    return _req("GET", "/templates", api_key=api_key).json()


def create_template(
    *,
    template_name: str,
    image_name: str,
    container_disk_gb: int,
    category: str,
    is_public: bool,
    env_vars: Optional[Dict[str, str]] = None,
    ports: Optional[List[str]] = None,
    readme: str = "",
    docker_entrypoint: Optional[List[str]] = None,
    docker_start_cmd: Optional[List[str]] = None,
    container_registry_auth: Optional[str] = None,
    volume_in_gb: int = 0,
    volume_mount_path: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "name": template_name,
        "imageName": image_name,
        "isServerless": False,
        "containerDiskInGb": int(container_disk_gb),
        "category": category,
        "isPublic": bool(is_public),
        "env": env_vars or {},
        "ports": ports or [],
        "readme": readme or "",
        "dockerEntrypoint": docker_entrypoint or [],
        "dockerStartCmd": docker_start_cmd or [],
    }
    if container_registry_auth:
        body["containerRegistryAuthId"] = container_registry_auth
    if int(volume_in_gb) > 0:
        body["volumeInGb"] = int(volume_in_gb)
        if volume_mount_path:
            body["volumeMountPath"] = volume_mount_path
    return _req("POST", "/templates", api_key=api_key, data=json.dumps(body)).json()


def ensure_template(
    *,
    template_name: str,
    image_name: str,
    container_disk_gb: int,
    category: str,
    is_public: bool,
    env_vars: Optional[Dict[str, str]] = None,
    ports: Optional[List[str]] = None,
    readme: str = "",
    docker_entrypoint: Optional[List[str]] = None,
    docker_start_cmd: Optional[List[str]] = None,
    container_registry_auth: Optional[str] = None,
    volume_in_gb: int = 0,
    volume_mount_path: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    tpls = list_templates(api_key)
    for t in tpls:
        if str(t.get("name")) == template_name:
            return {"action": "exists", "template": t}
    t = create_template(
        template_name=template_name,
        image_name=image_name,
        container_disk_gb=container_disk_gb,
        category=category,
        is_public=is_public,
        env_vars=env_vars,
        ports=ports,
        readme=readme,
        docker_entrypoint=docker_entrypoint,
        docker_start_cmd=docker_start_cmd,
        container_registry_auth=container_registry_auth,
        volume_in_gb=volume_in_gb,
        volume_mount_path=volume_mount_path,
        api_key=api_key,
    )
    return {"action": "created", "template": t}


# ---------- High-level helper ----------

def ensure_infrastructure(
    *,
    api_key: Optional[str] = None,
    datacenter_id: str = "US-NC-1",
    volume_name: str = "unsloth-volume",
    volume_size_gb: int = 50,
    resize_if_smaller: bool = True,
    template_name: str = "unsloth-trainer-template",
    image_name: str = "docker.io/sbussiso/unsloth-trainer:latest",
    container_disk_gb: int = 30,
    volume_in_gb: int = 0,
    volume_mount_path: str = "/workspace",
    category: str = "NVIDIA",
    is_public: bool = False,
    container_registry_auth: Optional[str] = None,
    docker_entrypoint: Optional[List[str]] = None,
    docker_start_cmd: Optional[List[str]] = None,
    env_vars: Optional[Dict[str, str]] = None,
    ports: Optional[List[str]] = None,
    readme: str = "Unsloth fine-tuning trainer template.",
) -> Dict[str, Any]:
    try:
        vres = ensure_volume(
            volume_name=volume_name,
            volume_size_gb=int(volume_size_gb),
            datacenter_id=datacenter_id,
            resize_if_smaller=bool(resize_if_smaller),
            api_key=api_key,
        )
        vol = vres["volume"]
        tres = ensure_template(
            template_name=template_name,
            image_name=image_name,
            container_disk_gb=int(container_disk_gb),
            category=category,
            is_public=bool(is_public),
            env_vars=env_vars or {"PYTHONUNBUFFERED": "1"},
            ports=ports or [],
            readme=readme,
            docker_entrypoint=docker_entrypoint or [],
            docker_start_cmd=docker_start_cmd or [],
            container_registry_auth=container_registry_auth,
            volume_in_gb=int(volume_in_gb),
            volume_mount_path=volume_mount_path,
            api_key=api_key,
        )
        tpl = tres["template"]
        return {
            "volume": {"action": vres["action"], "id": vol.get("id"), "name": vol.get("name"), "dc": vol.get("dataCenterId"), "size": vol.get("size")},
            "template": {"action": tres["action"], "id": tpl.get("id"), "name": tpl.get("name"), "image": tpl.get("imageName")},
        }
    except requests.HTTPError as e:
        try:
            body = e.response.json()
        except Exception:
            body = e.response.text
        raise RunpodError(f"HTTP {e.response.status_code}: {body}") from e
    except Exception as e:
        raise RunpodError(str(e)) from e
