from __future__ import annotations

import json
import shutil
import subprocess

import httpx
import flet as ft

from helpers.common import safe_update


async def on_docker_pull(
    *,
    page: ft.Page,
    ICONS,
    COLORS,
    docker_image_tf: ft.TextField,
    docker_status: ft.Text,
    DEFAULT_DOCKER_IMAGE: str,
) -> None:
    """Pull a Docker image locally with friendly UX messaging.

    Mirrors the previous inline implementation from main.py but extracted here.
    """
    img = (docker_image_tf.value or "").strip() or DEFAULT_DOCKER_IMAGE
    try:
        if not shutil.which("docker"):
            docker_status.value = "Docker CLI not found. Please install Docker Desktop and ensure 'docker' is on PATH."
            try:
                docker_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
            except Exception:
                pass
            await safe_update(page)
            return
        # Quick daemon check before pulling
        try:
            info_res = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=4)
            if info_res.returncode != 0:
                err_txt = (info_res.stderr or info_res.stdout or "").strip()
                raise RuntimeError(err_txt or "Docker daemon not responding")
        except Exception as ex_chk:
            # Pretty error if Docker Desktop/daemon isn't running
            nice_msg = "Docker is not running. Please start Docker Desktop, then retry."
            docker_status.value = nice_msg
            try:
                docker_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
            except Exception:
                pass
            try:
                page.snack_bar = ft.SnackBar(
                    content=ft.Row([
                        ft.Icon(getattr(ICONS, "ERROR_OUTLINE", getattr(ICONS, "ERROR", ICONS.WARNING)), color=COLORS.WHITE),
                        ft.Text(nice_msg, color=COLORS.WHITE),
                    ], spacing=8),
                    bgcolor=getattr(COLORS, "RED_400", getattr(COLORS, "RED", None)),
                )
                page.snack_bar.open = True
                await safe_update(page)
            except Exception:
                pass
            return
        # If daemon OK, first check if image already exists locally
        try:
            insp = subprocess.run(["docker", "image", "inspect", img], capture_output=True, text=True)
            if insp.returncode == 0:
                tag_list = []
                img_id = ""
                created = ""
                try:
                    info = json.loads(insp.stdout or "[]")
                    if isinstance(info, list) and info:
                        tag_list = info[0].get("RepoTags") or []
                        img_id = str(info[0].get("Id", ""))[-12:]
                        created = (info[0].get("Created") or "")[:19].replace("T", " ")
                except Exception:
                    pass
                docker_status.value = f"Image already present locally: {(tag_list and ', '.join(tag_list)) or img}\nID …{img_id}  Created {created}"
                try:
                    docker_status.color = getattr(COLORS, "GREEN_400", getattr(COLORS, "GREEN", None))
                except Exception:
                    pass
                await safe_update(page)
                return
        except Exception:
            pass
        # Proceed to pull
        page.snack_bar = ft.SnackBar(ft.Text(f"Pulling {img}..."))
        page.snack_bar.open = True
        await safe_update(page)
    except Exception:
        pass
    try:
        res = subprocess.run(["docker", "pull", img], capture_output=True, text=True)
        if res.returncode == 0:
            out = (res.stdout or "").strip()
            docker_status.value = f"Pulled successfully: {img}\n" + (out[-800:] if out else "")
            try:
                docker_status.color = getattr(COLORS, "GREEN_400", getattr(COLORS, "GREEN", None))
            except Exception:
                pass
        else:
            out = (res.stdout or "") + "\n" + (res.stderr or "")
            # Friendly hints for common errors
            lower = out.lower()
            hints = []
            repo = img
            tag = "latest"
            try:
                if ":" in img.rsplit("/", 1)[-1]:
                    repo, tag = img.rsplit(":", 1)
                else:
                    repo = img
            except Exception:
                pass
            if "manifest unknown" in lower or "not found" in lower:
                # Try to show local tags for this repo
                try:
                    ref = repo
                    if ref.startswith("docker.io/"):
                        ref = ref[len("docker.io/"):]
                    ls = subprocess.run([
                        "docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}",
                        "--filter", f"reference={ref}:*"
                    ], capture_output=True, text=True)
                    lines = [l.strip() for l in (ls.stdout or "").splitlines() if l.strip()]
                    if lines:
                        hints.append("Local tags found: " + ", ".join(lines[:8]))
                except Exception:
                    pass
                # Optional: probe Docker Hub for available tags
                try:
                    hub_repo = repo
                    if hub_repo.startswith("docker.io/"):
                        hub_repo = hub_repo[len("docker.io/"):]
                    url = f"https://hub.docker.com/v2/repositories/{hub_repo}/tags?page_size=10"
                    r = httpx.get(url, timeout=5.0)
                    if r.status_code == 200:
                        data = r.json()
                        names = [t.get("name") for t in (data.get("results") or []) if t.get("name")]
                        if names:
                            hints.append("Docker Hub tags: " + ", ".join(names[:8]))
                except Exception:
                    pass
            if "authentication required" in lower or "unauthorized" in lower:
                hints.append("If this is a private repo, run 'docker login' first.")
            msg = f"docker pull failed (exit {res.returncode}).\n" + out[-800:]
            if hints:
                msg += "\n\nHints: " + "  •  ".join(hints)
            docker_status.value = msg
            try:
                docker_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
            except Exception:
                pass
    except Exception as ex:
        docker_status.value = f"Error pulling image: {ex}"
        try:
            docker_status.color = getattr(COLORS, "RED_400", getattr(COLORS, "RED", None))
        except Exception:
            pass
    try:
        await safe_update(page)
    except Exception:
        pass
