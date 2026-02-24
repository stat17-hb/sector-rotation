"""Theme system tests for CSS/template/font behavior."""
from __future__ import annotations

from src.ui.styles import (
    ACTION_BADGE_STYLES,
    ACTION_COLORS_BY_THEME,
    PLOTLY_COLORWAY_BY_THEME,
    THEME_TOKENS,
    build_font_face_css,
    get_plotly_template,
    get_table_style_tokens,
    get_tab_style_tokens,
    inject_css,
    normalize_theme_mode,
)


REQUIRED_TOKEN_KEYS = {
    "bg",
    "surface",
    "border",
    "text",
    "text_muted",
    "primary",
    "success",
    "warning",
    "danger",
    "info",
}


def test_theme_tokens_have_required_keys():
    for mode, tokens in THEME_TOKENS.items():
        assert REQUIRED_TOKEN_KEYS.issubset(tokens.keys()), mode


def test_normalize_theme_mode_defaults_to_dark():
    assert normalize_theme_mode(None) == "dark"
    assert normalize_theme_mode("") == "dark"
    assert normalize_theme_mode("unknown") == "dark"
    assert normalize_theme_mode("LIGHT") == "light"


def test_plotly_template_is_theme_aware_and_no_teal_cyan_palette():
    dark_template = get_plotly_template("dark")
    light_template = get_plotly_template("light")

    assert dark_template["font"]["color"] == THEME_TOKENS["dark"]["text"]
    assert light_template["font"]["color"] == THEME_TOKENS["light"]["text"]
    assert dark_template["colorway"] != light_template["colorway"]

    forbidden = {"14b8a6", "06b6d4", "22d3ee", "0ea5e9"}
    for mode, colorway in PLOTLY_COLORWAY_BY_THEME.items():
        joined = "".join(colorway).lower()
        for token in forbidden:
            assert token not in joined, mode


def test_light_plotly_template_uses_stronger_axis_and_legend_text():
    light_template = get_plotly_template("light")
    light_tokens = THEME_TOKENS["light"]

    assert light_template["xaxis"]["tickfont"]["color"] != light_tokens["text_muted"]
    assert light_template["xaxis"]["title"]["font"]["color"] != light_tokens["text_muted"]
    assert light_template["legend"]["font"]["color"] != light_tokens["text_muted"]
    assert light_template["xaxis"]["automargin"] is True
    assert light_template["yaxis"]["automargin"] is True


def test_build_font_face_css_handles_presence_and_missing_fonts(tmp_path):
    css_empty = build_font_face_css(tmp_path)
    assert css_empty == ""

    (tmp_path / "PretendardVariable.woff2").write_bytes(b"pretendard")
    (tmp_path / "JetBrainsMono[wght].woff2").write_bytes(b"jetbrains")
    css = build_font_face_css(tmp_path)

    assert "Pretendard Local" in css
    assert "JetBrains Mono Local" in css
    assert "/static/fonts/JetBrainsMono%5Bwght%5D.woff2" in css


def test_inject_css_reflects_selected_tab_tokens(monkeypatch):
    rendered: list[str] = []

    def _capture(css: str, unsafe_allow_html: bool) -> None:
        assert unsafe_allow_html is True
        rendered.append(css)

    monkeypatch.setattr("src.ui.styles.st.markdown", _capture)

    inject_css("dark")
    inject_css("light")

    assert len(rendered) == 2
    dark_tab = get_tab_style_tokens("dark")
    light_tab = get_tab_style_tokens("light")

    assert dark_tab["tab_selected_bg"] in rendered[0]
    assert dark_tab["tab_selected_text"] in rendered[0]
    assert light_tab["tab_selected_bg"] in rendered[1]
    assert light_tab["tab_selected_text"] in rendered[1]


def test_inject_css_reflects_table_tokens(monkeypatch):
    rendered: list[str] = []

    def _capture(css: str, unsafe_allow_html: bool) -> None:
        assert unsafe_allow_html is True
        rendered.append(css)

    monkeypatch.setattr("src.ui.styles.st.markdown", _capture)
    inject_css("light")

    assert len(rendered) == 1
    table_tokens = get_table_style_tokens("light")
    assert table_tokens["header_bg"] in rendered[0]
    assert table_tokens["row_bg_even"] in rendered[0]
    assert "glideDataEditor" in rendered[0]
    assert "--gdg-bg-header" in rendered[0]
    assert "[data-testid=\"stHeader\"]" in rendered[0]
    assert "[data-testid=\"stDecoration\"]" in rendered[0]


# ---------------------------------------------------------------------------
# WCAG contrast helpers
# ---------------------------------------------------------------------------

def _srgb_to_linear(c: float) -> float:
    """Convert sRGB [0,1] channel to linear light value."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _relative_luminance(hex_color: str) -> float:
    """Return WCAG relative luminance for a 6-digit hex color (#RRGGBB)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return 0.2126 * _srgb_to_linear(r) + 0.7152 * _srgb_to_linear(g) + 0.0722 * _srgb_to_linear(b)


def _contrast_ratio(fg: str, bg: str) -> float:
    """Return WCAG contrast ratio between two hex colors."""
    l1 = _relative_luminance(fg)
    l2 = _relative_luminance(bg)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


# ---------------------------------------------------------------------------
# WCAG AA contrast tests for light theme
# ---------------------------------------------------------------------------

WCAG_AA_NORMAL = 4.5  # minimum for normal text


def test_light_theme_base_text_tokens_meet_wcag_aa():
    """text / text_muted must be ≥ 4.5:1 on both bg and surface."""
    tokens = THEME_TOKENS["light"]
    for bg_key in ("bg", "surface"):
        bg = tokens[bg_key]
        for fg_key in ("text", "text_muted"):
            ratio = _contrast_ratio(tokens[fg_key], bg)
            assert ratio >= WCAG_AA_NORMAL, (
                f"THEME_TOKENS[light][{fg_key!r}] on {bg_key!r}: "
                f"{ratio:.2f}:1 < {WCAG_AA_NORMAL}:1"
            )


def test_light_theme_action_colors_meet_wcag_aa():
    """Hold and N/A action marker colors must be ≥ 4.5:1 on bg and surface."""
    tokens = THEME_TOKENS["light"]
    action_colors = ACTION_COLORS_BY_THEME["light"]
    badge_styles = ACTION_BADGE_STYLES["light"]

    # Marker / label colors on app background surfaces
    for bg_key in ("bg", "surface"):
        bg = tokens[bg_key]
        for action in ("Hold", "N/A"):
            ratio = _contrast_ratio(action_colors[action], bg)
            assert ratio >= WCAG_AA_NORMAL, (
                f"ACTION_COLORS[light][{action!r}] on {bg_key!r}: "
                f"{ratio:.2f}:1 < {WCAG_AA_NORMAL}:1"
            )

    # Badge text on badge background
    for action in ("Hold", "N/A"):
        badge = badge_styles[action]
        ratio = _contrast_ratio(badge["text"], badge["bg"])
        assert ratio >= WCAG_AA_NORMAL, (
            f"ACTION_BADGE_STYLES[light][{action!r}] text on badge bg: "
            f"{ratio:.2f}:1 < {WCAG_AA_NORMAL}:1"
        )
