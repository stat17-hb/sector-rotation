"""Central theme tokens and helpers for Toss-inspired light/dark UI modes."""

from __future__ import annotations

from typing import Any, Literal

import streamlit as st

ThemeMode = Literal["dark", "light"]
THEME_SESSION_KEY = "theme_mode"

UI_FONT_STACK = (
    "'Pretendard Local', 'Pretendard', 'SUIT', 'Spoqa Han Sans Neo', "
    "'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', 'Segoe UI', sans-serif"
)
DISPLAY_FONT_STACK = (
    "'Pretendard Local', 'Pretendard', 'SUIT', 'Spoqa Han Sans Neo', "
    "'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', 'Segoe UI', sans-serif"
)
MONO_FONT_STACK = (
    "'JetBrains Mono Local', 'JetBrains Mono', 'Fira Code', "
    "'SFMono-Regular', Consolas, 'Liberation Mono', monospace"
)


THEME_TOKENS: dict[ThemeMode, dict[str, dict[str, Any]]] = {
    "dark": {
        "ui": {
            "background": "#101419",
            "foreground": "#F5F7FA",
            "muted": "#B9C3D0",
            "card": "#171C23",
            "card_alt": "#222A34",
            "border": "#303946",
            "border_soft": "#27313D",
            "primary": "#78A7FF",
            "primary_soft": "rgba(120, 167, 255, 0.16)",
            "success": "#55C494",
            "danger": "#FF7685",
            "warning": "#D7A247",
            "focus_ring": "rgba(120, 167, 255, 0.38)",
            "info": "#A4C1FF",
            "header_bg": "#101419",
            "header_border": "#27313D",
            "sidebar_bg": "#151B22",
            "sidebar_gradient_start": "#171D25",
            "sidebar_gradient_end": "#101419",
            "sidebar_hover": "#222B36",
            "input_bg": "#171C23",
            "input_border": "#333D4B",
            "inline_code_bg": "#222A34",
            "card_shadow": "0 10px 28px rgba(0, 0, 0, 0.26)",
            "provisional_badge_text": "#FFD08A",
            "cycle_swatch_border": "rgba(255, 255, 255, 0.16)",
            "accent_blue_hover": "#7EA5FF",
            "accent_link": "#9AB6FF",
            "surface_tint": "rgba(238, 243, 250, 0.06)",
        },
        "layout": {
            "radius_xs": "6px",
            "radius_sm": "8px",
            "radius_md": "9px",
            "radius_lg": "12px",
            "radius_xl": "16px",
            "radius_pill": "18px",
            "radius_full": "9999px",
            "space_1": "0.25rem",
            "space_2": "0.5rem",
            "space_3": "0.75rem",
            "space_4": "1rem",
            "space_5": "1.25rem",
            "space_6": "1.5rem",
            "space_7": "2rem",
            "section_gap": "1.62rem",
            "section_gap_tight": "1.08rem",
            "section_gap_loose": "2.16rem",
            "page_shell_padding": "1.15rem 1.25rem 1.18rem",
            "panel_padding": "0.86rem 0.92rem 0.84rem",
            "tab_padding": "0.46rem 0.72rem",
        },
        "typography": {
            "display_family": DISPLAY_FONT_STACK,
            "ui_family": UI_FONT_STACK,
            "body_family": UI_FONT_STACK,
            "mono_family": MONO_FONT_STACK,
            "display_hero_size": "2.25rem",
            "display_secondary_size": "1.6rem",
            "section_title_size": "1rem",
            "card_title_size": "0.94rem",
            "body_size": "0.94rem",
            "body_small_size": "0.86rem",
            "caption_size": "0.73rem",
            "button_size": "0.88rem",
            "display_line_height": "1.08",
            "heading_line_height": "1.12",
            "body_line_height": "1.56",
            "display_weight": "620",
            "heading_weight": "620",
            "body_weight": "460",
            "caption_weight": "640",
            "button_weight": "620",
        },
        "chart": {
            "background": "#111418",
            "grid": "#27303A",
            "axis": "#303844",
            "text": "#F6F8FB",
            "muted": "#B6C0CF",
            "axis_title": "#F6F8FB",
            "legend_text": "#E8ECF3",
            "legend_bg": "rgba(22, 25, 30, 0.92)",
            "paper_bg": "rgba(0, 0, 0, 0)",
            "plot_bg": "rgba(0, 0, 0, 0)",
            "candle_up": "#49B985",
            "candle_down": "#FF6B7A",
            "ma20": "#D69A3A",
            "ma60": "#8AADFF",
            "ma120": "#E6ECF4",
            "colorway": [
                "#5C8DFF",
                "#49B985",
                "#D69A3A",
                "#FF6B7A",
                "#8AADFF",
                "#E6ECF4",
                "#9AB6FF",
                "#B6C0CF",
                "#85D4B1",
                "#F2A5AF",
            ],
            "analysis_heatmaps": {
                "classic": [
                    [0.00, "#FF6B7A"],
                    [0.50, "#2D3542"],
                    [1.00, "#49B985"],
                ],
                "contrast": [
                    [0.00, "#FF9CA6"],
                    [0.44, "#C84D62"],
                    [0.50, "#27303A"],
                    [0.56, "#7FD9B1"],
                    [1.00, "#49B985"],
                ],
                "blue_orange": [
                    [0.00, "#7DA8FF"],
                    [0.44, "#5C8DFF"],
                    [0.50, "#27303A"],
                    [0.56, "#F7C98A"],
                    [1.00, "#D68A2F"],
                ],
            },
            "selection_row_fill": "rgba(92, 141, 255, 0.16)",
            "selection_col_fill": "rgba(73, 185, 133, 0.12)",
            "selection_outline": "#FFFFFF",
            "muted_lines": ["#596272", "#7A8495", "#333B47"],
        },
        "dataframe": {
            "header_bg": "#222832",
            "header_text": "#FFFFFF",
            "row_bg_even": "#171B21",
            "row_bg_odd": "#111418",
            "row_text": "#F2F5F9",
            "grid": "#303844",
            "cell_background": "#171B21",
            "cell_text": "#F2F5F9",
            "cell_border": "#282B31",
        },
        "signal": {
            "actions": {
                "Strong Buy": "#49B985",
                "Watch": "#9AB6FF",
                "Hold": "#D0D6E2",
                "Avoid": "#FF6B7A",
                "N/A": "#9AA5B5",
            },
            "action_badges": {
                "Strong Buy": {"bg": "#103829", "text": "#DDF8EE", "border": "#49B985"},
                "Watch": {"bg": "#172A55", "text": "#E8F0FF", "border": "#9AB6FF"},
                "Hold": {"bg": "#222832", "text": "#F2F5F9", "border": "#D0D6E2"},
                "Avoid": {"bg": "#4A1720", "text": "#FFECEE", "border": "#FF6B7A"},
                "N/A": {"bg": "#16191E", "text": "#D7DEE9", "border": "#9AA5B5"},
            },
            "cycle_swatches": {
                "Recovery": "#9AB6FF",
                "Expansion": "#49B985",
                "Slowdown": "#D69A3A",
                "Contraction": "#FF6B7A",
                "Indeterminate": "#C2C9D6",
            },
            "cycle_phase_styles": {
                "RECOVERY_EARLY": {"fill": "rgba(154, 182, 255, 0.22)", "line": "#9AB6FF"},
                "RECOVERY_LATE": {"fill": "rgba(92, 141, 255, 0.28)", "line": "#5C8DFF"},
                "EXPANSION_EARLY": {"fill": "rgba(82, 211, 153, 0.20)", "line": "#4FD19B"},
                "EXPANSION_LATE": {"fill": "rgba(73, 185, 133, 0.28)", "line": "#49B985"},
                "SLOWDOWN_EARLY": {"fill": "rgba(247, 201, 138, 0.24)", "line": "#F7C98A"},
                "SLOWDOWN_LATE": {"fill": "rgba(214, 154, 58, 0.30)", "line": "#D69A3A"},
                "CONTRACTION_EARLY": {"fill": "rgba(255, 156, 166, 0.22)", "line": "#FF9CA6"},
                "CONTRACTION_LATE": {"fill": "rgba(255, 107, 122, 0.30)", "line": "#FF6B7A"},
                "INDETERMINATE": {"fill": "rgba(194, 201, 214, 0.24)", "line": "#C2C9D6"},
            },
            "cycle_current_line": "#FFFFFF",
        },
        "navigation": {
            "tab_hover_bg": "#222832",
            "tab_selected_bg": "#EFF3FA",
            "tab_selected_text": "#111418",
            "tab_selected_border": "#8AADFF",
        },
    },
    "light": {
        "ui": {
            "background": "#F7F9FC",
            "foreground": "#1E2734",
            "muted": "#626D7D",
            "card": "#FFFFFF",
            "card_alt": "#F1F4F8",
            "border": "#E1E7EF",
            "border_soft": "#CFD8E4",
            "primary": "#2B64B8",
            "primary_soft": "rgba(43, 100, 184, 0.10)",
            "success": "#15966A",
            "danger": "#D94A55",
            "warning": "#C7821E",
            "focus_ring": "rgba(43, 100, 184, 0.22)",
            "info": "#2B64B8",
            "header_bg": "#FFFFFF",
            "header_border": "#DDE5F0",
            "sidebar_bg": "#F5F7FA",
            "sidebar_gradient_start": "#FFFFFF",
            "sidebar_gradient_end": "#EEF3F8",
            "sidebar_hover": "#EAF0F7",
            "input_bg": "#FFFFFF",
            "input_border": "#D4DEEB",
            "inline_code_bg": "#F0F4F9",
            "card_shadow": "0 1px 2px rgba(23, 32, 51, 0.04)",
            "provisional_badge_text": "#7B4A00",
            "cycle_swatch_border": "rgba(20, 24, 31, 0.1)",
            "accent_blue_hover": "#2B64B8",
            "accent_link": "#2B64B8",
            "surface_tint": "rgba(240, 244, 249, 0.92)",
        },
        "layout": {
            "radius_xs": "4px",
            "radius_sm": "6px",
            "radius_md": "7px",
            "radius_lg": "8px",
            "radius_xl": "10px",
            "radius_pill": "7px",
            "radius_full": "9999px",
            "space_1": "0.25rem",
            "space_2": "0.5rem",
            "space_3": "0.75rem",
            "space_4": "1rem",
            "space_5": "1.25rem",
            "space_6": "1.5rem",
            "space_7": "2rem",
            "section_gap": "1.5rem",
            "section_gap_tight": "1rem",
            "section_gap_loose": "2rem",
            "page_shell_padding": "0.1rem 0 1.08rem",
            "panel_padding": "0.9rem 1rem 0.88rem",
            "tab_padding": "0.46rem 0.78rem",
        },
        "typography": {
            "display_family": DISPLAY_FONT_STACK,
            "ui_family": UI_FONT_STACK,
            "body_family": UI_FONT_STACK,
            "mono_family": MONO_FONT_STACK,
            "display_hero_size": "1.86rem",
            "display_secondary_size": "1.48rem",
            "section_title_size": "1.02rem",
            "card_title_size": "0.96rem",
            "body_size": "0.94rem",
            "body_small_size": "0.86rem",
            "caption_size": "0.74rem",
            "button_size": "0.88rem",
            "display_line_height": "1.08",
            "heading_line_height": "1.12",
            "body_line_height": "1.56",
            "display_weight": "680",
            "heading_weight": "620",
            "body_weight": "460",
            "caption_weight": "640",
            "button_weight": "620",
        },
        "chart": {
            "background": "#F8FAFC",
            "grid": "#D9E1EC",
            "axis": "#C7D2E1",
            "text": "#172033",
            "muted": "#4B5565",
            "axis_title": "#172033",
            "legend_text": "#253247",
            "legend_bg": "rgba(255, 255, 255, 0.96)",
            "paper_bg": "rgba(0, 0, 0, 0)",
            "plot_bg": "rgba(0, 0, 0, 0)",
            "candle_up": "#138A63",
            "candle_down": "#C23B4A",
            "ma20": "#946200",
            "ma60": "#245FB2",
            "ma120": "#172033",
            "colorway": [
                "#245FB2",
                "#138A63",
                "#946200",
                "#C23B4A",
                "#2E6BC9",
                "#27303A",
                "#7299D8",
                "#4B5565",
                "#4AAE84",
                "#D26174",
            ],
            "analysis_heatmaps": {
                "classic": [
                    [0.00, "#C23B4A"],
                    [0.50, "#E6ECF4"],
                    [1.00, "#138A63"],
                ],
                "contrast": [
                    [0.00, "#811A30"],
                    [0.44, "#C23B4A"],
                    [0.50, "#E6ECF4"],
                    [0.56, "#3E9B78"],
                    [1.00, "#138A63"],
                ],
                "blue_orange": [
                    [0.00, "#2E6BC9"],
                    [0.44, "#245FB2"],
                    [0.50, "#E6ECF4"],
                    [0.56, "#E2A75A"],
                    [1.00, "#946200"],
                ],
            },
            "selection_row_fill": "rgba(36, 95, 178, 0.08)",
            "selection_col_fill": "rgba(19, 138, 99, 0.08)",
            "selection_outline": "#172033",
            "muted_lines": ["#9AA5B5", "#CAD2DE", "#687487"],
        },
        "dataframe": {
            "header_bg": "#EEF3F8",
            "header_text": "#172033",
            "row_bg_even": "#FFFFFF",
            "row_bg_odd": "#F4F7FB",
            "row_text": "#172033",
            "grid": "#D9E1EC",
            "cell_background": "#FFFFFF",
            "cell_text": "#172033",
            "cell_border": "#D9E1EC",
        },
        "signal": {
            "actions": {
                "Strong Buy": "#138A63",
                "Watch": "#245FB2",
                "Hold": "#3E4958",
                "Avoid": "#C23B4A",
                "N/A": "#27303A",
            },
            "action_badges": {
                "Strong Buy": {"bg": "#DFF3EC", "text": "#0D5C45", "border": "#138A63"},
                "Watch": {"bg": "#E7EFFC", "text": "#173F7B", "border": "#245FB2"},
                "Hold": {"bg": "#EEF3F8", "text": "#253247", "border": "#3E4958"},
                "Avoid": {"bg": "#FDE8EC", "text": "#8F1E34", "border": "#C23B4A"},
                "N/A": {"bg": "#EEF3F8", "text": "#1E2735", "border": "#27303A"},
            },
            "cycle_swatches": {
                "Recovery": "#245FB2",
                "Expansion": "#138A63",
                "Slowdown": "#946200",
                "Contraction": "#C23B4A",
                "Indeterminate": "#4B5565",
            },
            "cycle_phase_styles": {
                "RECOVERY_EARLY": {"fill": "rgba(107, 146, 234, 0.16)", "line": "#6B92EA"},
                "RECOVERY_LATE": {"fill": "rgba(36, 95, 178, 0.18)", "line": "#245FB2"},
                "EXPANSION_EARLY": {"fill": "rgba(74, 174, 132, 0.14)", "line": "#4AAE84"},
                "EXPANSION_LATE": {"fill": "rgba(19, 138, 99, 0.18)", "line": "#138A63"},
                "SLOWDOWN_EARLY": {"fill": "rgba(226, 167, 90, 0.16)", "line": "#E2A75A"},
                "SLOWDOWN_LATE": {"fill": "rgba(148, 98, 0, 0.20)", "line": "#946200"},
                "CONTRACTION_EARLY": {"fill": "rgba(210, 97, 116, 0.14)", "line": "#D26174"},
                "CONTRACTION_LATE": {"fill": "rgba(194, 59, 74, 0.18)", "line": "#C23B4A"},
                "INDETERMINATE": {"fill": "rgba(91, 97, 110, 0.16)", "line": "#5B616E"},
            },
            "cycle_current_line": "#172033",
        },
        "navigation": {
            "tab_hover_bg": "#E7EEF7",
            "tab_selected_bg": "#2567C8",
            "tab_selected_text": "#FFFFFF",
            "tab_selected_border": "#2567C8",
        },
    },
}


def normalize_theme_mode(value: Any, fallback: ThemeMode = "light") -> ThemeMode:
    """Normalize free-form input to a valid theme mode."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("dark", "light"):
            return lowered  # type: ignore[return-value]
    return fallback


def get_theme_mode(default: ThemeMode = "light") -> ThemeMode:
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
    return THEME_TOKENS.get(mode, THEME_TOKENS["light"])


def get_ui_tokens(theme_mode: str | None = None) -> dict[str, Any]:
    return dict(get_theme_tokens(theme_mode)["ui"])


def get_layout_tokens(theme_mode: str | None = None) -> dict[str, Any]:
    return dict(get_theme_tokens(theme_mode)["layout"])


def get_typography_tokens(theme_mode: str | None = None) -> dict[str, Any]:
    return dict(get_theme_tokens(theme_mode)["typography"])


def get_chart_tokens(theme_mode: str | None = None) -> dict[str, Any]:
    return dict(get_theme_tokens(theme_mode)["chart"])


def get_dataframe_tokens(theme_mode: str | None = None) -> dict[str, Any]:
    return dict(get_theme_tokens(theme_mode)["dataframe"])


def get_signal_tokens(theme_mode: str | None = None) -> dict[str, Any]:
    return dict(get_theme_tokens(theme_mode)["signal"])


def get_navigation_tokens(theme_mode: str | None = None) -> dict[str, Any]:
    return dict(get_theme_tokens(theme_mode)["navigation"])
