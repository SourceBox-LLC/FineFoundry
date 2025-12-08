from typing import Optional, Callable
import flet as ft

from .theme import COLORS, ICONS, ACCENT_COLOR, BORDER_BASE


def WITH_OPACITY(opacity: float, color):
    """Apply opacity if supported in this Flet build; otherwise return color as-is."""
    # Try ft.colors.with_opacity
    if hasattr(ft, "colors") and hasattr(ft.colors, "with_opacity"):
        try:
            return ft.colors.with_opacity(opacity, color)
        except Exception:
            pass
    # Try Colors.with_opacity
    if hasattr(COLORS, "with_opacity"):
        try:
            return COLORS.with_opacity(opacity, color)
        except Exception:
            pass
    return color


def pill(text: str, color: str, icon: Optional[str] = None) -> ft.Container:
    return ft.Container(
        content=ft.Row(
            [
                ft.Icon(icon, size=14, color=COLORS.WHITE) if icon else ft.Container(),
                ft.Text(text, size=12, weight=ft.FontWeight.W_600, color=COLORS.WHITE),
            ],
            spacing=6,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        bgcolor=color,
        padding=ft.padding.symmetric(6, 6),
        border_radius=999,
    )


def make_board_chip(text: str, selected: bool, base_color):
    """Return a chip-like control compatible with current Flet version.
    Tries FilterChip -> ChoiceChip -> Chip -> styled Container.
    """
    tooltip = f"/{text}/"
    # Preferred: FilterChip
    if hasattr(ft, "FilterChip"):
        return ft.FilterChip(
            text=text,
            selected=selected,
            bgcolor=WITH_OPACITY(0.1, base_color),
            selected_color=base_color,
            tooltip=tooltip,
        )
    # Next: ChoiceChip (API might be similar)
    if hasattr(ft, "ChoiceChip"):
        try:
            return ft.ChoiceChip(text=text, selected=selected, tooltip=tooltip)
        except Exception:
            pass
    # Next: Chip (non-selectable)
    if hasattr(ft, "Chip"):
        try:
            return ft.Chip(label=ft.Text(text), tooltip=tooltip, bgcolor=WITH_OPACITY(0.1, base_color))
        except Exception:
            pass
    # Fallback: simple container
    return ft.Container(
        content=ft.Text(text),
        bgcolor=WITH_OPACITY(0.1, base_color) if selected else None,
        border=ft.border.all(1, WITH_OPACITY(0.2, BORDER_BASE)),
        border_radius=16,
        padding=ft.padding.symmetric(8, 6),
        tooltip=tooltip,
    )


def section_title(
    title: str, icon: str, help_text: Optional[str] = None, on_help_click: Optional[Callable[..., None]] = None
) -> ft.Row:
    controls = [
        ft.Icon(icon, color=ACCENT_COLOR),
        ft.Text(title, size=16, weight=ft.FontWeight.BOLD),
    ]
    if help_text:
        try:
            _info_icon_name = getattr(
                ICONS,
                "INFO_OUTLINE",
                getattr(ICONS, "INFO", getattr(ICONS, "HELP_OUTLINE", getattr(ICONS, "HELP", None))),
            )
            if _info_icon_name is None:
                raise AttributeError("No suitable info icon available")
            try:
                controls.append(
                    ft.IconButton(
                        icon=_info_icon_name,
                        icon_color=WITH_OPACITY(0.8, BORDER_BASE),
                        tooltip=help_text,
                        on_click=on_help_click,
                    )
                )
            except Exception:
                # Fallback to Tooltip wrapper if IconButton not available
                info_ic = ft.Icon(_info_icon_name, size=16, color=WITH_OPACITY(0.8, BORDER_BASE))
                try:
                    controls.append(ft.Tooltip(message=help_text, content=ft.Container(content=info_ic, padding=0)))
                except Exception:
                    controls.append(ft.Container(content=info_ic, tooltip=help_text))
        except Exception:
            # Last resort: simple text with optional tooltip
            try:
                controls.append(ft.Tooltip(message=help_text, content=ft.Text("ⓘ")))
            except Exception:
                controls.append(ft.Text("ⓘ"))
    return ft.Row(controls)


def make_wrap(controls: list, spacing: int = 6, run_spacing: int = 6):
    """Return a wrapping layout compatible with current Flet version.
    Tries Wrap -> Row(wrap=True) -> Row -> Column.
    """
    # Preferred: Wrap
    if hasattr(ft, "Wrap"):
        try:
            return ft.Wrap(controls, spacing=spacing, run_spacing=run_spacing)
        except Exception:
            pass
    # Next: Row with wrap
    try:
        return ft.Row(
            controls, wrap=True, spacing=spacing, run_spacing=run_spacing, alignment=ft.MainAxisAlignment.START
        )
    except TypeError:
        # Older Row without run_spacing or wrap
        try:
            return ft.Row(controls, spacing=spacing, alignment=ft.MainAxisAlignment.START)
        except Exception:
            pass
    # Fallback: Column (no wrapping)
    return ft.Column(controls, spacing=spacing)


def make_selectable_pill(
    label: str, selected: bool = False, base_color: Optional[str] = None, on_change=None
) -> ft.Container:
    """Create a selectable pill using a Container, compatible with older Flet builds."""
    base_color = base_color or ACCENT_COLOR
    pill = ft.Container(
        content=ft.Text(label),
        bgcolor=WITH_OPACITY(0.15, base_color) if selected else None,
        border=ft.border.all(1, WITH_OPACITY(0.2, BORDER_BASE)),
        border_radius=16,
        padding=ft.padding.symmetric(8, 6),
        tooltip=f"/{label}/",
    )
    # Store state in .data
    pill.data = {"label": label, "selected": bool(selected), "base_color": base_color, "on_change": on_change}

    def toggle(_):
        d = pill.data
        d["selected"] = not d.get("selected", False)
        pill.bgcolor = WITH_OPACITY(0.15, base_color) if d["selected"] else None
        pill.update()
        cb = d.get("on_change")
        if callable(cb):
            cb()

    pill.on_click = toggle
    return pill


def make_empty_placeholder(text: str, icon) -> ft.Container:
    """Centered, subtle placeholder shown when a panel has no content."""
    return ft.Container(
        content=ft.Column(
            [
                ft.Icon(icon, color=WITH_OPACITY(0.45, BORDER_BASE), size=18),
                ft.Text(text, size=12, color=WITH_OPACITY(0.7, BORDER_BASE)),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=6,
        ),
        padding=10,
    )


def cell_text(text: str, width: int | None = None, size: int = 13) -> ft.Text:
    """Create text for DataTable cells that wraps properly within its cell.
    Do not force width or VISIBLE overflow to avoid overlap; let the table layout size cells.
    """
    try:
        return ft.Text(
            text or "",
            no_wrap=False,
            max_lines=None,
            size=size,
        )
    except Exception:
        # Fallback if some args are unsupported
        return ft.Text(text or "")


def get_preview_col_width(page: ft.Page) -> int | None:
    """Best-effort column width for preview tables to keep columns aligned.
    Uses window width when available; returns a conservative fallback otherwise.
    """
    try:
        w = getattr(page, "window_width", None) or getattr(page, "width", None)
        if w:
            usable = max(600, int(w - 220))  # account for paddings/controls
            return max(260, int(usable / 2))
    except Exception:
        pass
    return 420


def _estimate_two_col_ratio(samples: list[tuple[str, str]]) -> float:
    """Estimate width ratio for col A (0..1) based on average content length.
    Keeps result within [0.35, 0.65] to avoid extreme skews.
    """
    if not samples:
        return 0.5
    a = sum(min(len(x or ""), 400) for x, _ in samples) / len(samples)
    b = sum(min(len(y or ""), 400) for _, y in samples) / len(samples)
    total = a + b
    if total <= 0:
        return 0.5
    r = a / total
    # Clamp
    return max(0.35, min(0.65, r))


def compute_two_col_widths(
    page: ft.Page,
    samples: list[tuple[str, str]],
    *,
    total_px: int | None = None,
    spacing_px: int = 16,
    min_px_each: int = 180,
) -> tuple[int, int]:
    """Compute two column widths that sum to available width minus spacing.
    If total_px isn't provided, derive from page width conservatively.
    """
    try:
        if total_px is None:
            w = getattr(page, "window_width", None) or getattr(page, "width", None)
            if w:
                total_px = max(600, int(w - 220))
        if total_px is None:
            total_px = 840
    except Exception:
        total_px = 840
    # Space available for both columns combined
    usable = max(2 * min_px_each + 10, total_px)
    ratio = _estimate_two_col_ratio(samples)
    w1 = max(min_px_each, int(usable * ratio) - spacing_px // 2)
    w2 = max(min_px_each, usable - spacing_px - w1)
    return (w1, w2)


def compute_two_col_flex(samples: list[tuple[str, str]]) -> tuple[int, int]:
    """Return left/right flex factors based on content ratio with sane clamps."""
    r = _estimate_two_col_ratio(samples)
    left = max(1, int(round(r * 100)))
    return left, max(1, 100 - left)


def two_col_header(
    left: str = "Input", right: str = "Output", *, left_flex: int = 50, right_flex: int = 50
) -> ft.Container:
    hdr = ft.Row(
        [
            ft.Container(ft.Text(left, weight=ft.FontWeight.BOLD, size=13), expand=left_flex, padding=4),
            ft.Container(ft.Text(right, weight=ft.FontWeight.BOLD, size=13), expand=right_flex, padding=4),
        ],
        spacing=12,
        alignment=ft.MainAxisAlignment.START,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )
    return ft.Container(
        content=hdr,
        padding=ft.padding.only(left=6, right=6, bottom=6),
        border=ft.border.only(bottom=ft.border.BorderSide(1, WITH_OPACITY(0.12, BORDER_BASE))),
    )


def two_col_row(a: str, b: str, left_flex: int, right_flex: int) -> ft.Container:
    """Compact, uniform two-column row; cells are fixed-height and internally scrollable."""
    CELL_H = 88
    COL_SPACING = 12

    def scroll_cell(text: str) -> ft.Container:
        inner = ft.Column(
            [ft.Text(text or "", no_wrap=False, max_lines=None, size=13)], scroll=ft.ScrollMode.AUTO, spacing=0
        )
        return ft.Container(
            content=inner,
            height=CELL_H,
            padding=6,
            border=ft.border.all(1, WITH_OPACITY(0.06, BORDER_BASE)),
            border_radius=6,
        )

    row = ft.Row(
        [
            ft.Container(content=scroll_cell(a), expand=left_flex),
            ft.Container(content=scroll_cell(b), expand=right_flex),
        ],
        spacing=COL_SPACING,
        alignment=ft.MainAxisAlignment.START,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )
    return ft.Container(
        content=row,
        padding=ft.padding.symmetric(6, 6),
        border=ft.border.only(bottom=ft.border.BorderSide(1, WITH_OPACITY(0.06, BORDER_BASE))),
    )


__all__ = [
    "WITH_OPACITY",
    "pill",
    "make_board_chip",
    "section_title",
    "make_wrap",
    "make_selectable_pill",
    "make_empty_placeholder",
    "cell_text",
    "get_preview_col_width",
    "compute_two_col_widths",
    "compute_two_col_flex",
    "two_col_header",
    "two_col_row",
]
