import flet as ft

# Robust color aliasing: prefer ft.Colors, fall back to ft.colors if present
if hasattr(ft, "Colors"):
    COLORS = ft.Colors
else:
    COLORS = getattr(ft, "colors", None)

# Robust icons aliasing: prefer ft.Icons, fall back to ft.icons if present
if hasattr(ft, "Icons"):
    ICONS = ft.Icons
else:
    ICONS = getattr(ft, "icons", None)

# Accent and borders
ACCENT_COLOR = COLORS.AMBER
BORDER_BASE = getattr(COLORS, "ON_SURFACE", getattr(COLORS, "GREY", "#e0e0e0"))

# Common icon fallbacks
REFRESH_ICON = getattr(ICONS, "REFRESH", getattr(ICONS, "AUTORENEW", getattr(ICONS, "RESTART_ALT", None)))
INFO_ICON = getattr(
    ICONS,
    "INFO_OUTLINE",
    getattr(ICONS, "INFO", getattr(ICONS, "HELP_OUTLINE", getattr(ICONS, "HELP", None))),
)
DARK_ICON = getattr(ICONS, "DARK_MODE_OUTLINED", getattr(ICONS, "DARK_MODE", None))

__all__ = [
    "COLORS",
    "ICONS",
    "ACCENT_COLOR",
    "BORDER_BASE",
    "REFRESH_ICON",
    "INFO_ICON",
    "DARK_ICON",
]
