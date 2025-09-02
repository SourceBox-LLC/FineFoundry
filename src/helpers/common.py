import os
import sys
import platform
import ctypes
import flet as ft

async def safe_update(page: ft.Page):
    """Update the page across Flet versions (async if available, else sync)."""
    if hasattr(page, "update_async"):
        return await page.update_async()
    return page.update()


def set_terminal_title(title: str):
    """Attempt to set the integrated terminal/tab title across platforms."""
    try:
        if platform.system().lower().startswith("win"):
            # Best: Windows API
            try:
                if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "kernel32"):
                    ctypes.windll.kernel32.SetConsoleTitleW(title)
                    return
            except Exception:
                pass
            # Fallback: shell 'title' command
            try:
                os.system(f"title {title}")
                return
            except Exception:
                pass
        # ANSI OSC sequence fallback (works in many terminals)
        try:
            sys.stdout.write(f"\x1b]0;{title}\x07")
            sys.stdout.flush()
        except Exception:
            pass
    except Exception:
        pass

__all__ = ["safe_update", "set_terminal_title"]
