"""Theme system tests for CSS/template/font behavior."""
from __future__ import annotations

from pathlib import Path

from config.theme import (
    THEME_SESSION_KEY,
    get_chart_tokens,
    get_layout_tokens,
    get_navigation_tokens,
    get_signal_tokens,
    get_theme_mode,
    get_theme_tokens as get_canonical_theme_tokens,
    get_typography_tokens,
    get_ui_tokens,
    set_theme_mode,
)
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


def test_canonical_theme_tokens_have_expected_sections():
    for mode in ("dark", "light"):
        tokens = get_canonical_theme_tokens(mode)
        assert {"ui", "layout", "typography", "chart", "dataframe", "signal", "navigation"} <= set(tokens.keys())


def test_normalize_theme_mode_defaults_to_light():
    assert normalize_theme_mode(None) == "light"
    assert normalize_theme_mode("") == "light"
    assert normalize_theme_mode("unknown") == "light"
    assert normalize_theme_mode("LIGHT") == "light"


def test_plotly_template_is_theme_aware_and_uses_canonical_chart_colorway():
    dark_template = get_plotly_template("dark")
    light_template = get_plotly_template("light")
    dark_chart = get_chart_tokens("dark")
    light_chart = get_chart_tokens("light")

    assert dark_template["font"]["color"] == THEME_TOKENS["dark"]["text"]
    assert light_template["font"]["color"] == THEME_TOKENS["light"]["text"]
    assert list(dark_template["colorway"]) == list(dark_chart["colorway"])
    assert list(light_template["colorway"]) == list(light_chart["colorway"])
    assert dark_chart["candle_up"] == "#49B985"
    assert dark_chart["ma20"] == "#D69A3A"
    assert dark_chart["ma60"] == "#8AADFF"
    assert dark_chart["ma120"] == "#E6ECF4"


def test_light_plotly_template_uses_stronger_axis_and_legend_text():
    light_template = get_plotly_template("light")
    light_chart = get_chart_tokens("light")

    assert light_template["xaxis"]["tickfont"]["color"] == light_chart["muted"]
    assert light_template["xaxis"]["title"]["font"]["color"] == light_chart["axis_title"]
    assert light_template["legend"]["font"]["color"] == light_chart["legend_text"]
    assert light_template["xaxis"]["automargin"] is True
    assert light_template["yaxis"]["automargin"] is True


def test_theme_state_helpers_use_existing_session_key(monkeypatch):
    class DummyStreamlit:
        session_state: dict[str, str] = {}

    dummy = DummyStreamlit()
    monkeypatch.setattr("config.theme.st", dummy)

    assert get_theme_mode() == "light"
    assert set_theme_mode("light") == "light"
    assert dummy.session_state[THEME_SESSION_KEY] == "light"
    assert get_theme_mode() == "light"


def test_layout_and_typography_tokens_cover_toss_inspired_density_contract():
    light_layout = get_layout_tokens("light")
    light_typography = get_typography_tokens("light")

    assert light_layout["radius_pill"] == "7px"
    assert light_layout["radius_full"] == "9999px"
    assert light_layout["radius_sm"] == "6px"
    assert light_layout["radius_md"] == "7px"
    assert light_layout["radius_lg"] == "8px"
    assert light_typography["display_line_height"] == "1.08"
    assert "Pretendard" in light_typography["ui_family"]
    assert light_typography["display_hero_size"] == "1.86rem"
    assert light_typography["display_secondary_size"] == "1.48rem"
    assert light_typography["body_size"] == "0.94rem"
    assert light_typography["body_small_size"] == "0.86rem"
    assert light_typography["caption_size"] == "0.74rem"
    assert light_typography["button_size"] == "0.88rem"


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
    assert "border-radius: var(--radius-sm)" in rendered[0]
    assert "background: var(--accent-blue-hover)" in rendered[0]


def test_inject_css_reflects_cycle_palette_tokens(monkeypatch):
    rendered: list[str] = []

    monkeypatch.setattr("src.ui.styles.st.markdown", lambda css, unsafe_allow_html: rendered.append(css))
    inject_css("dark")

    css = rendered[0]
    signal_tokens = get_signal_tokens("dark")
    assert signal_tokens["cycle_swatches"]["Recovery"] in css
    assert signal_tokens["cycle_swatches"]["Contraction"] in css


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
    assert "[data-testid=\"stAppDeployButton\"]" in rendered[0]
    assert "[aria-label=\"Deploy\"]" in rendered[0]
    assert "display: block !important" in rendered[0]
    assert "min-height: 3rem !important" in rendered[0]
    assert "[data-testid=\"stSidebarCollapsedControl\"]" in rendered[0]
    assert "[data-testid=\"stSidebarCollapseButton\"]" in rendered[0]


def test_inject_css_includes_new_dashboard_layout_classes(monkeypatch):
    rendered: list[str] = []

    def _capture(css: str, unsafe_allow_html: bool) -> None:
        assert unsafe_allow_html is True
        rendered.append(css)

    monkeypatch.setattr("src.ui.styles.st.markdown", _capture)
    inject_css("dark")

    assert len(rendered) == 1
    css = rendered[0]
    assert ".page-shell" in css
    assert "[data-testid=\"stSidebarNav\"]" in css
    assert ".sidebar-workspace" in css
    assert ".sidebar-ops-panel" in css
    assert ".sidebar-status-chip" in css
    assert ".sidebar-section-label" in css
    assert "[data-testid=\"stIconMaterial\"]" in css
    assert "[data-testid=\"stSidebarNav\"] [data-testid=\"stIconMaterial\"]" in css
    assert "[data-testid=\"collapsedControl\"]" in css
    assert "[data-testid=\"stBaseButton-header\"]" in css
    assert "[data-testid=\"stBaseButton-headerNoPadding\"]" in css
    assert "[aria-expanded=\"false\"] [data-testid=\"stSidebarCollapseButton\"]" in css
    assert "[data-testid=\"stVerticalBlockBorderWrapper\"]" in css
    assert ".status-strip" in css
    assert ".command-bar" in css
    assert ".research-page-frame" in css
    assert ".overview-workbench-header" in css
    assert ".overview-evidence-shell" in css
    assert ".overview-review-candidates" in css
    assert ".overview-taxonomy-surface" in css
    assert ".overview-taxonomy-row" in css
    assert ".sector-momentum-decision-boards" in css
    assert ".theme-lens-panel" in css
    assert ".research-page-frame__summary" in css
    assert "--section-gap:" in css
    assert "--section-gap-loose:" in css
    assert "margin-bottom: var(--section-gap)" in css
    assert "[data-testid=\"stForm\"]" in css
    assert "width: fit-content" in css
    assert "width: max-content" in css
    assert "table-layout: auto" in css
    assert "overflow-x: auto" in css
    assert ".top-bar-summary" in css
    assert ".analysis-toolbar" in css
    assert ".analysis-toolbar__summary" in css
    assert ".phase-chip-row" in css
    assert ".panel-header" in css
    assert ".decision-hero" in css
    assert ".status-card-grid" in css
    assert ".compact-note" in css
    assert ".sector-rank-list__metric" in css
    assert ".page-shell__pill:nth-child(n + 4)" in css
    assert ".overview-market-card__metric-row" in css
    assert ".overview-market-card--status" in css
    assert ".status-strip__message" in css
    assert "--radius-xl" in css
    assert "--ring:" in css
    assert "@media (max-width: 1120px)" in css
    assert "@media (max-width: 840px)" in css


def test_light_theme_matches_refined_finance_palette():
    light_ui = get_ui_tokens("light")

    assert light_ui["background"] == "#F7F9FC"
    assert light_ui["foreground"] == "#1E2734"
    assert light_ui["primary"] == "#2B64B8"
    assert light_ui["card_alt"] == "#F1F4F8"
    assert light_ui["sidebar_bg"] == "#F5F7FA"


def test_streamlit_config_matches_light_theme_tokens():
    config_text = Path(".streamlit/config.toml").read_text(encoding="utf-8")
    light_ui = get_ui_tokens("light")

    assert 'base = "light"' in config_text
    assert f'primaryColor = "{light_ui["primary"]}"' in config_text
    assert f'backgroundColor = "{light_ui["background"]}"' in config_text
    assert f'secondaryBackgroundColor = "{light_ui["card_alt"]}"' in config_text
    assert f'textColor = "{light_ui["foreground"]}"' in config_text


def test_theme_contract_files_no_longer_expose_legacy_or_exchange_specific_theme():
    legacy_literals = ("#6366F1", "#4F46E5", "#818CF8", "#0052FF", "Coinbase")
    for path in (Path("config/theme.py"), Path(".streamlit/config.toml"), Path("DESIGN.md")):
        text = path.read_text(encoding="utf-8")
        for literal in legacy_literals:
            assert literal not in text, f"{literal} still present in {path}"


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
