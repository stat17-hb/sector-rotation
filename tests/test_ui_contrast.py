"""WCAG contrast and non-color-only status expression tests."""
from __future__ import annotations

from src.ui.components import format_action_label
from src.ui.styles import (
    THEME_TOKENS,
    get_action_badge_styles,
    get_plotly_template,
    get_tab_style_tokens,
    get_table_style_tokens,
)


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    c = color.strip().lstrip("#")
    if len(c) != 6:
        raise ValueError(f"expected #RRGGBB, got: {color}")
    return tuple(int(c[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _to_linear(channel: float) -> float:
    if channel <= 0.03928:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _relative_luminance(color: str) -> float:
    r, g, b = _hex_to_rgb(color)
    return 0.2126 * _to_linear(r) + 0.7152 * _to_linear(g) + 0.0722 * _to_linear(b)


def _contrast_ratio(foreground: str, background: str) -> float:
    l1 = _relative_luminance(foreground)
    l2 = _relative_luminance(background)
    hi = max(l1, l2)
    lo = min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def test_body_and_muted_text_contrast_meet_wcag_aa():
    for mode, tokens in THEME_TOKENS.items():
        assert _contrast_ratio(tokens["text"], tokens["bg"]) >= 4.5, mode
        assert _contrast_ratio(tokens["text_muted"], tokens["bg"]) >= 4.5, mode


def test_tab_selected_state_contrast_meets_wcag_aa():
    for mode in ("dark", "light"):
        tab = get_tab_style_tokens(mode)
        assert _contrast_ratio(tab["tab_selected_text"], tab["tab_selected_bg"]) >= 4.5, mode


def test_action_badge_contrast_meets_wcag_aa():
    for mode in ("dark", "light"):
        badge_styles = get_action_badge_styles(mode)
        for action, styles in badge_styles.items():
            contrast = _contrast_ratio(styles["text"], styles["bg"])
            assert contrast >= 4.5, f"{mode}:{action}={contrast:.2f}"


def test_table_text_contrast_meets_wcag_aa():
    for mode in ("dark", "light"):
        table = get_table_style_tokens(mode)
        assert _contrast_ratio(table["header_text"], table["header_bg"]) >= 4.5, mode
        assert _contrast_ratio(table["row_text"], table["row_bg_even"]) >= 4.5, mode
        assert _contrast_ratio(table["row_text"], table["row_bg_odd"]) >= 4.5, mode


def test_plotly_axis_and_legend_text_contrast_meets_wcag_aa():
    for mode in ("dark", "light"):
        template = get_plotly_template(mode)
        bg = THEME_TOKENS[mode]["bg"]

        x_tick_color = template["xaxis"]["tickfont"]["color"]
        x_title_color = template["xaxis"]["title"]["font"]["color"]
        y_tick_color = template["yaxis"]["tickfont"]["color"]
        y_title_color = template["yaxis"]["title"]["font"]["color"]
        legend_color = template["legend"]["font"]["color"]

        assert _contrast_ratio(x_tick_color, bg) >= 4.5, f"{mode}:x_tick"
        assert _contrast_ratio(x_title_color, bg) >= 4.5, f"{mode}:x_title"
        assert _contrast_ratio(y_tick_color, bg) >= 4.5, f"{mode}:y_tick"
        assert _contrast_ratio(y_title_color, bg) >= 4.5, f"{mode}:y_title"
        assert _contrast_ratio(legend_color, bg) >= 4.5, f"{mode}:legend"


def test_action_labels_are_explicit_text_not_color_only():
    for action in ("Strong Buy", "Watch", "Hold", "Avoid", "N/A"):
        label = format_action_label(action)
        assert action in label
        assert len(label) > len(action)
