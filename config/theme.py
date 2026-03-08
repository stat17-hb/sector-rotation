"""Central theme tokens and helpers for dark/light UI modes."""

from __future__ import annotations

from typing import Any, Literal

import streamlit as st

ThemeMode = Literal["dark", "light"]
THEME_SESSION_KEY = "theme_mode"


THEME_TOKENS: dict[ThemeMode, dict[str, dict[str, Any]]] = {
    "dark": {
        "ui": {
            "background": "#09090B",
            "foreground": "#FAFAFA",
            "muted": "#A1A1AA",
            "card": "#18181B",
            "card_alt": "#27272A",
            "border": "#3F3F46",
            "border_soft": "#27272A",
            "primary": "#6366F1",
            "primary_soft": "rgba(99, 102, 241, 0.20)",
            "success": "#10B981",
            "danger": "#F43F5E",
            "warning": "#F59E0B",
            "focus_ring": "rgba(99, 102, 241, 0.35)",
            "info": "#818CF8",
            "header_bg": "#09090B",
            "header_border": "#27272A",
            "sidebar_bg": "#18181B",
            "sidebar_gradient_start": "#18181B",
            "sidebar_gradient_end": "#09090B",
            "sidebar_hover": "#27272A",
            "input_bg": "#18181B",
            "input_border": "#3F3F46",
            "inline_code_bg": "#27272A",
            "card_shadow": "none",
            "provisional_badge_text": "#F59E0B",
            "cycle_swatch_border": "rgba(250, 250, 250, 0.24)",
        },
        "chart": {
            "background": "#09090B",
            "grid": "#27272A",
            "axis": "#3F3F46",
            "text": "#FAFAFA",
            "muted": "#A1A1AA",
            "axis_title": "#FAFAFA",
            "legend_text": "#A1A1AA",
            "legend_bg": "rgba(24, 24, 27, 0.84)",
            "paper_bg": "rgba(0, 0, 0, 0)",
            "plot_bg": "rgba(0, 0, 0, 0)",
            "candle_up": "#26A69A",
            "candle_down": "#EF5350",
            "ma20": "#FF9500",
            "ma60": "#0066CC",
            "ma120": "#CC3300",
            "colorway": [
                "#6366F1",
                "#10B981",
                "#F59E0B",
                "#F43F5E",
                "#FF9500",
                "#0066CC",
                "#CC3300",
                "#A1A1AA",
                "#26A69A",
                "#EF5350",
            ],
            "analysis_heatmaps": {
                "classic": [
                    [0.00, "#F43F5E"],
                    [0.50, "#3F3F46"],
                    [1.00, "#10B981"],
                ],
                "contrast": [
                    [0.00, "#FF8FA1"],
                    [0.44, "#B1284F"],
                    [0.50, "#3F3F46"],
                    [0.56, "#7BE1B1"],
                    [1.00, "#10B981"],
                ],
                "blue_orange": [
                    [0.00, "#79A7FF"],
                    [0.44, "#6366F1"],
                    [0.50, "#3F3F46"],
                    [0.56, "#F8C987"],
                    [1.00, "#FF9500"],
                ],
            },
            "selection_row_fill": "rgba(99, 102, 241, 0.12)",
            "selection_col_fill": "rgba(16, 185, 129, 0.10)",
            "selection_outline": "#FAFAFA",
            "muted_lines": ["#71717A", "#A1A1AA", "#52525B"],
        },
        "dataframe": {
            "header_bg": "#27272A",
            "header_text": "#FAFAFA",
            "row_bg_even": "#18181B",
            "row_bg_odd": "#09090B",
            "row_text": "#FAFAFA",
            "grid": "#3F3F46",
            "cell_background": "#18181B",
            "cell_text": "#FAFAFA",
            "cell_border": "#27272A",
        },
        "signal": {
            "actions": {
                "Strong Buy": "#10B981",
                "Watch": "#6366F1",
                "Hold": "#A1A1AA",
                "Avoid": "#F43F5E",
                "N/A": "#71717A",
            },
            "action_badges": {
                "Strong Buy": {"bg": "#052E22", "text": "#D1FAE5", "border": "#10B981"},
                "Watch": {"bg": "#1E1B4B", "text": "#E0E7FF", "border": "#6366F1"},
                "Hold": {"bg": "#27272A", "text": "#FAFAFA", "border": "#A1A1AA"},
                "Avoid": {"bg": "#4C0519", "text": "#FFE4E6", "border": "#F43F5E"},
                "N/A": {"bg": "#18181B", "text": "#D4D4D8", "border": "#71717A"},
            },
            "cycle_swatches": {
                "Recovery": "#818CF8",
                "Expansion": "#10B981",
                "Slowdown": "#F59E0B",
                "Contraction": "#F43F5E",
                "Indeterminate": "#A1A1AA",
            },
            "cycle_phase_styles": {
                "RECOVERY_EARLY": {"fill": "rgba(129, 140, 248, 0.24)", "line": "#818CF8"},
                "RECOVERY_LATE": {"fill": "rgba(99, 102, 241, 0.34)", "line": "#6366F1"},
                "EXPANSION_EARLY": {"fill": "rgba(52, 211, 153, 0.24)", "line": "#34D399"},
                "EXPANSION_LATE": {"fill": "rgba(16, 185, 129, 0.34)", "line": "#10B981"},
                "SLOWDOWN_EARLY": {"fill": "rgba(251, 191, 36, 0.26)", "line": "#FBBF24"},
                "SLOWDOWN_LATE": {"fill": "rgba(245, 158, 11, 0.34)", "line": "#F59E0B"},
                "CONTRACTION_EARLY": {"fill": "rgba(251, 113, 133, 0.24)", "line": "#FB7185"},
                "CONTRACTION_LATE": {"fill": "rgba(244, 63, 94, 0.34)", "line": "#F43F5E"},
                "INDETERMINATE": {"fill": "rgba(161, 161, 170, 0.28)", "line": "#A1A1AA"},
            },
            "cycle_current_line": "#FAFAFA",
        },
        "navigation": {
            "tab_hover_bg": "#27272A",
            "tab_selected_bg": "#4F46E5",
            "tab_selected_text": "#FAFAFA",
            "tab_selected_border": "#6366F1",
        },
    },
    "light": {
        "ui": {
            "background": "#F8FAFC",
            "foreground": "#1E293B",
            "muted": "#64748B",
            "card": "#FFFFFF",
            "card_alt": "#F1F5F9",
            "border": "#E2E8F0",
            "border_soft": "#CBD5E1",
            "primary": "#6366F1",
            "primary_soft": "rgba(99, 102, 241, 0.12)",
            "success": "#10B981",
            "danger": "#F43F5E",
            "warning": "#D97706",
            "focus_ring": "rgba(99, 102, 241, 0.25)",
            "info": "#4F46E5",
            "header_bg": "#FFFFFF",
            "header_border": "#E2E8F0",
            "sidebar_bg": "#F8FAFC",
            "sidebar_gradient_start": "#FFFFFF",
            "sidebar_gradient_end": "#F1F5F9",
            "sidebar_hover": "#EEF2FF",
            "input_bg": "#FFFFFF",
            "input_border": "#CBD5E1",
            "inline_code_bg": "#F1F5F9",
            "card_shadow": "0 1px 2px rgba(15, 23, 42, 0.08)",
            "provisional_badge_text": "#92400E",
            "cycle_swatch_border": "rgba(30, 41, 59, 0.12)",
        },
        "chart": {
            "background": "#FFFFFF",
            "grid": "#E2E8F0",
            "axis": "#CBD5E1",
            "text": "#1E293B",
            "muted": "#64748B",
            "axis_title": "#1E293B",
            "legend_text": "#334155",
            "legend_bg": "rgba(255, 255, 255, 0.96)",
            "paper_bg": "rgba(0, 0, 0, 0)",
            "plot_bg": "rgba(0, 0, 0, 0)",
            "candle_up": "#26A69A",
            "candle_down": "#EF5350",
            "ma20": "#FF9500",
            "ma60": "#0066CC",
            "ma120": "#CC3300",
            "colorway": [
                "#6366F1",
                "#10B981",
                "#D97706",
                "#F43F5E",
                "#FF9500",
                "#0066CC",
                "#CC3300",
                "#64748B",
                "#26A69A",
                "#EF5350",
            ],
            "analysis_heatmaps": {
                "classic": [
                    [0.00, "#F43F5E"],
                    [0.50, "#E2E8F0"],
                    [1.00, "#10B981"],
                ],
                "contrast": [
                    [0.00, "#A31245"],
                    [0.44, "#F43F5E"],
                    [0.50, "#E2E8F0"],
                    [0.56, "#34D399"],
                    [1.00, "#047857"],
                ],
                "blue_orange": [
                    [0.00, "#0066CC"],
                    [0.44, "#818CF8"],
                    [0.50, "#E2E8F0"],
                    [0.56, "#FDBA74"],
                    [1.00, "#D97706"],
                ],
            },
            "selection_row_fill": "rgba(99, 102, 241, 0.10)",
            "selection_col_fill": "rgba(16, 185, 129, 0.08)",
            "selection_outline": "#1E293B",
            "muted_lines": ["#94A3B8", "#CBD5E1", "#64748B"],
        },
        "dataframe": {
            "header_bg": "#F1F5F9",
            "header_text": "#1E293B",
            "row_bg_even": "#FFFFFF",
            "row_bg_odd": "#F8FAFC",
            "row_text": "#1E293B",
            "grid": "#E2E8F0",
            "cell_background": "#FFFFFF",
            "cell_text": "#1E293B",
            "cell_border": "#E2E8F0",
        },
        "signal": {
            "actions": {
                "Strong Buy": "#10B981",
                "Watch": "#6366F1",
                "Hold": "#475569",
                "Avoid": "#F43F5E",
                "N/A": "#334155",
            },
            "action_badges": {
                "Strong Buy": {"bg": "#D1FAE5", "text": "#065F46", "border": "#10B981"},
                "Watch": {"bg": "#E0E7FF", "text": "#3730A3", "border": "#6366F1"},
                "Hold": {"bg": "#E2E8F0", "text": "#334155", "border": "#475569"},
                "Avoid": {"bg": "#FFE4E6", "text": "#9F1239", "border": "#F43F5E"},
                "N/A": {"bg": "#E2E8F0", "text": "#1E293B", "border": "#334155"},
            },
            "cycle_swatches": {
                "Recovery": "#6366F1",
                "Expansion": "#10B981",
                "Slowdown": "#D97706",
                "Contraction": "#F43F5E",
                "Indeterminate": "#64748B",
            },
            "cycle_phase_styles": {
                "RECOVERY_EARLY": {"fill": "rgba(129, 140, 248, 0.16)", "line": "#818CF8"},
                "RECOVERY_LATE": {"fill": "rgba(99, 102, 241, 0.22)", "line": "#4F46E5"},
                "EXPANSION_EARLY": {"fill": "rgba(52, 211, 153, 0.16)", "line": "#34D399"},
                "EXPANSION_LATE": {"fill": "rgba(16, 185, 129, 0.22)", "line": "#059669"},
                "SLOWDOWN_EARLY": {"fill": "rgba(251, 191, 36, 0.18)", "line": "#F59E0B"},
                "SLOWDOWN_LATE": {"fill": "rgba(217, 119, 6, 0.24)", "line": "#B45309"},
                "CONTRACTION_EARLY": {"fill": "rgba(251, 113, 133, 0.16)", "line": "#FB7185"},
                "CONTRACTION_LATE": {"fill": "rgba(244, 63, 94, 0.22)", "line": "#E11D48"},
                "INDETERMINATE": {"fill": "rgba(148, 163, 184, 0.20)", "line": "#64748B"},
            },
            "cycle_current_line": "#1E293B",
        },
        "navigation": {
            "tab_hover_bg": "#EEF2FF",
            "tab_selected_bg": "#4F46E5",
            "tab_selected_text": "#FFFFFF",
            "tab_selected_border": "#4338CA",
        },
    },
}


def normalize_theme_mode(value: Any, fallback: ThemeMode = "dark") -> ThemeMode:
    """Normalize free-form input to a valid theme mode."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("dark", "light"):
            return lowered  # type: ignore[return-value]
    return fallback


def get_theme_mode(default: ThemeMode = "dark") -> ThemeMode:
    """Return the active theme from session state, falling back to default."""
    fallback = normalize_theme_mode(default)
    try:
        if THEME_SESSION_KEY in st.session_state:
            return normalize_theme_mode(st.session_state[THEME_SESSION_KEY], fallback=fallback)
    except Exception:
        pass
    return fallback


def set_theme_mode(mode: Any) -> ThemeMode:
    """Persist theme mode to session state and return the normalized mode."""
    normalized = normalize_theme_mode(mode)
    try:
        st.session_state[THEME_SESSION_KEY] = normalized
    except Exception:
        pass
    return normalized


def get_theme_tokens(theme_mode: str | None = None) -> dict[str, dict[str, Any]]:
    """Return the complete token map for a theme mode."""
    mode = normalize_theme_mode(theme_mode) if theme_mode is not None else get_theme_mode()
    return THEME_TOKENS.get(mode, THEME_TOKENS["dark"])


def get_ui_tokens(theme_mode: str | None = None) -> dict[str, Any]:
    return dict(get_theme_tokens(theme_mode)["ui"])


def get_chart_tokens(theme_mode: str | None = None) -> dict[str, Any]:
    return dict(get_theme_tokens(theme_mode)["chart"])


def get_dataframe_tokens(theme_mode: str | None = None) -> dict[str, Any]:
    return dict(get_theme_tokens(theme_mode)["dataframe"])


def get_signal_tokens(theme_mode: str | None = None) -> dict[str, Any]:
    return dict(get_theme_tokens(theme_mode)["signal"])


def get_navigation_tokens(theme_mode: str | None = None) -> dict[str, Any]:
    return dict(get_theme_tokens(theme_mode)["navigation"])
