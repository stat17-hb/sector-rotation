"""Dashboard CSS injection, theme tokens, and Plotly templates."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import streamlit as st

from config.theme import (
    THEME_TOKENS as CANONICAL_THEME_TOKENS,
    get_chart_tokens as get_chart_section_tokens,
    get_dataframe_tokens as get_dataframe_section_tokens,
    get_layout_tokens as get_layout_section_tokens,
    get_navigation_tokens as get_navigation_section_tokens,
    get_signal_tokens as get_signal_section_tokens,
    get_typography_tokens as get_typography_section_tokens,
    get_ui_tokens as get_ui_section_tokens,
    normalize_theme_mode as normalize_theme_mode_base,
)


def _flatten_ui_tokens(theme_mode: str) -> dict[str, str]:
    ui = get_ui_section_tokens(theme_mode)
    return {
        "bg": str(ui["background"]),
        "surface": str(ui["card_alt"]),
        "border": str(ui["border"]),
        "text": str(ui["foreground"]),
        "text_muted": str(ui["muted"]),
        "primary": str(ui["primary"]),
        "success": str(ui["success"]),
        "warning": str(ui["warning"]),
        "danger": str(ui["danger"]),
        "info": str(ui["info"]),
    }


THEME_TOKENS: dict[str, dict[str, str]] = {
    mode: _flatten_ui_tokens(mode) for mode in CANONICAL_THEME_TOKENS
}

ACTION_COLORS_BY_THEME: dict[str, dict[str, str]] = {
    mode: dict(get_signal_section_tokens(mode)["actions"]) for mode in CANONICAL_THEME_TOKENS
}

ACTION_BADGE_STYLES: dict[str, dict[str, dict[str, str]]] = {
    mode: {
        action: dict(style)
        for action, style in get_signal_section_tokens(mode)["action_badges"].items()
    }
    for mode in CANONICAL_THEME_TOKENS
}

TAB_STYLE_TOKENS: dict[str, dict[str, str]] = {
    mode: dict(get_navigation_section_tokens(mode)) for mode in CANONICAL_THEME_TOKENS
}

TABLE_STYLE_TOKENS: dict[str, dict[str, str]] = {
    mode: {
        "header_bg": str(get_dataframe_section_tokens(mode)["header_bg"]),
        "header_text": str(get_dataframe_section_tokens(mode)["header_text"]),
        "row_bg_even": str(get_dataframe_section_tokens(mode)["row_bg_even"]),
        "row_bg_odd": str(get_dataframe_section_tokens(mode)["row_bg_odd"]),
        "row_text": str(get_dataframe_section_tokens(mode)["row_text"]),
        "grid": str(get_dataframe_section_tokens(mode)["grid"]),
    }
    for mode in CANONICAL_THEME_TOKENS
}

PLOTLY_COLORWAY_BY_THEME: dict[str, list[str]] = {
    mode: list(get_chart_section_tokens(mode)["colorway"]) for mode in CANONICAL_THEME_TOKENS
}
# Backward-compatible aliases used by existing tests/imports.
ACTION_COLORS: dict[str, str] = ACTION_COLORS_BY_THEME["dark"]
BLUE = ACTION_COLORS["Watch"]
GREY = ACTION_COLORS["Hold"]
DARK_GREY = ACTION_COLORS["N/A"]


def normalize_theme_mode(theme_mode: str | None) -> str:
    """Normalize theme mode to `dark` or `light` with dark fallback."""
    return str(normalize_theme_mode_base(theme_mode))


def get_theme_tokens(theme_mode: str | None) -> dict[str, str]:
    """Return a copy of theme tokens for the requested mode."""
    return dict(THEME_TOKENS[normalize_theme_mode(theme_mode)])


def get_action_colors(theme_mode: str | None) -> dict[str, str]:
    """Return action marker colors for a theme."""
    return dict(ACTION_COLORS_BY_THEME[normalize_theme_mode(theme_mode)])


def get_action_badge_styles(theme_mode: str | None) -> dict[str, dict[str, str]]:
    """Return action badge styles for a theme."""
    return {
        action: dict(style)
        for action, style in ACTION_BADGE_STYLES[normalize_theme_mode(theme_mode)].items()
    }


def get_tab_style_tokens(theme_mode: str | None) -> dict[str, str]:
    """Return tab style tokens for a theme."""
    return dict(TAB_STYLE_TOKENS[normalize_theme_mode(theme_mode)])


def get_table_style_tokens(theme_mode: str | None) -> dict[str, str]:
    """Return table style tokens for a theme."""
    return dict(TABLE_STYLE_TOKENS[normalize_theme_mode(theme_mode)])


def get_font_asset_paths(font_root: Path | None = None) -> dict[str, Path]:
    """Return expected local font paths."""
    root = font_root or Path("static/fonts")
    return {
        "pretendard": root / "PretendardVariable.woff2",
        "jetbrains_mono": root / "JetBrainsMono[wght].woff2",
    }


def build_font_face_css(font_root: Path | None = None) -> str:
    """Build @font-face blocks only for existing local font files."""
    paths = get_font_asset_paths(font_root)
    chunks: list[str] = []

    if paths["pretendard"].exists():
        pretendard_url = "/static/fonts/" + quote(paths["pretendard"].name)
        chunks.append(
            "@font-face {"
            "font-family: 'Pretendard Local';"
            f"src: url('{pretendard_url}') format('woff2');"
            "font-weight: 100 900;"
            "font-style: normal;"
            "font-display: swap;"
            "}"
        )

    if paths["jetbrains_mono"].exists():
        mono_url = "/static/fonts/" + quote(paths["jetbrains_mono"].name)
        chunks.append(
            "@font-face {"
            "font-family: 'JetBrains Mono Local';"
            f"src: url('{mono_url}') format('woff2');"
            "font-weight: 100 800;"
            "font-style: normal;"
            "font-display: swap;"
            "}"
        )

    return "\n".join(chunks)


def inject_css(theme_mode: str) -> None:
    """Inject global CSS for typography and theme tokens."""
    mode = normalize_theme_mode(theme_mode)
    tokens = get_theme_tokens(mode)
    ui_tokens = get_ui_section_tokens(mode)
    layout_tokens = get_layout_section_tokens(mode)
    signal_tokens = get_signal_section_tokens(mode)
    tab_tokens = get_tab_style_tokens(mode)
    table_tokens = get_table_style_tokens(mode)
    typography_tokens = get_typography_section_tokens(mode)
    badge_styles = get_action_badge_styles(mode)
    font_face_css = build_font_face_css()
    color_scheme = "dark" if mode == "dark" else "light"
    ui_font_css = str(typography_tokens["ui_family"])
    body_font_css = str(typography_tokens["body_family"])
    display_font_css = str(typography_tokens["display_family"])
    mono_font_css = str(typography_tokens["mono_family"])
    button_font_size = str(typography_tokens["button_size"])
    caption_font_size = str(typography_tokens["caption_size"])
    body_font_size = str(typography_tokens["body_size"])
    body_small_font_size = str(typography_tokens["body_small_size"])
    section_title_size = str(typography_tokens["section_title_size"])
    card_title_size = str(typography_tokens["card_title_size"])
    display_hero_size = str(typography_tokens["display_hero_size"])
    display_secondary_size = str(typography_tokens["display_secondary_size"])
    display_line_height = str(typography_tokens["display_line_height"])
    heading_line_height = str(typography_tokens["heading_line_height"])
    body_line_height = str(typography_tokens["body_line_height"])
    body_font_weight = str(typography_tokens["body_weight"])
    heading_font_weight = str(typography_tokens["heading_weight"])
    display_font_weight = str(typography_tokens["display_weight"])
    badge_font_weight = str(typography_tokens["caption_weight"])
    button_font_weight = str(typography_tokens["button_weight"])

    if mode == "dark":
        app_background_css = f"background: {ui_tokens['background']};"
        header_background_css = str(ui_tokens["header_bg"])
        header_border_css = str(ui_tokens["header_border"])
        sidebar_background_css = str(ui_tokens["sidebar_bg"])
        card_background_css = str(ui_tokens["card"])
        card_shadow_css = str(ui_tokens["card_shadow"])
        control_bg_css = str(ui_tokens["input_bg"])
        control_border_css = str(ui_tokens["input_border"])
        control_hover_css = str(ui_tokens["sidebar_hover"])
        inline_code_bg_css = str(ui_tokens["inline_code_bg"])
        font_smoothing_css = "-webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;"
        badge_font_size = caption_font_size
        provisional_badge_text = str(ui_tokens["provisional_badge_text"])
        provisional_badge_border = "color-mix(in srgb, var(--warning) 45%, transparent)"
    else:
        app_background_css = f"background: {ui_tokens['background']};"
        header_background_css = str(ui_tokens["header_bg"])
        header_border_css = str(ui_tokens["header_border"])
        sidebar_background_css = str(ui_tokens["sidebar_bg"])
        card_background_css = str(ui_tokens["card"])
        card_shadow_css = str(ui_tokens["card_shadow"])
        control_bg_css = str(ui_tokens["input_bg"])
        control_border_css = str(ui_tokens["input_border"])
        control_hover_css = str(ui_tokens["sidebar_hover"])
        inline_code_bg_css = str(ui_tokens["inline_code_bg"])
        font_smoothing_css = "-webkit-font-smoothing: subpixel-antialiased; -moz-osx-font-smoothing: auto;"
        badge_font_size = caption_font_size
        provisional_badge_text = str(ui_tokens["provisional_badge_text"])
        provisional_badge_border = "color-mix(in srgb, var(--warning) 60%, transparent)"

    css = f"""
    <style>
    {font_face_css}

    :root {{
        --bg: {tokens['bg']};
        --surface: {tokens['surface']};
        --border: {tokens['border']};
        --text: {tokens['text']};
        --text-muted: {tokens['text_muted']};
        --primary: {tokens['primary']};
        --success: {tokens['success']};
        --warning: {tokens['warning']};
        --danger: {tokens['danger']};
        --info: {tokens['info']};
        --muted: {ui_tokens['card_alt']};
        --muted-foreground: var(--text-muted);
        --accent-surface: color-mix(in srgb, var(--surface) 90%, var(--primary) 10%);
        --ring: {ui_tokens['focus_ring']};
        --accent-link: {ui_tokens['accent_link']};
        --accent-blue-hover: {ui_tokens['accent_blue_hover']};
        --surface-tint: {ui_tokens['surface_tint']};
        --radius-xs: {layout_tokens['radius_xs']};
        --radius-sm: {layout_tokens['radius_sm']};
        --radius-md: {layout_tokens['radius_md']};
        --radius-lg: {layout_tokens['radius_lg']};
        --radius-xl: {layout_tokens['radius_xl']};
        --radius-pill: {layout_tokens['radius_pill']};
        --radius-full: {layout_tokens['radius_full']};
        --space-1: {layout_tokens['space_1']};
        --space-2: {layout_tokens['space_2']};
        --space-3: {layout_tokens['space_3']};
        --space-4: {layout_tokens['space_4']};
        --space-5: {layout_tokens['space_5']};
        --space-6: {layout_tokens['space_6']};
        --space-7: {layout_tokens['space_7']};
        --font-display: {display_font_css};
        --font-ui: {ui_font_css};
        --font-body: {body_font_css};
        --font-mono: {mono_font_css};
        --flow-chip-size: {caption_font_size};

        /* Streamlit theme variable aliases for native widgets */
        --primary-color: {tokens['primary']};
        --background-color: {tokens['bg']};
        --secondary-background-color: {tokens['surface']};
        --text-color: {tokens['text']};
    }}

    html, body, [class*="css"] {{
        font-family: var(--font-body) !important;
        color: var(--text);
        font-weight: {body_font_weight};
        font-size: {body_font_size};
        line-height: {body_line_height};
        letter-spacing: 0;
        font-optical-sizing: auto;
        font-synthesis-weight: none;
        text-rendering: optimizeLegibility;
        word-break: keep-all;
        overflow-wrap: break-word;
        line-break: strict;
        {font_smoothing_css}
        color-scheme: {color_scheme};
    }}

    .stApp {{
        {app_background_css}
        color: var(--text);
    }}

    [data-testid="stHeader"] {{
        display: block !important;
        height: 3rem !important;
        min-height: 3rem !important;
        background: transparent !important;
        pointer-events: auto !important;
    }}

    [data-testid="stHeader"] > div {{
        height: 3rem !important;
        min-height: 3rem !important;
        background: transparent !important;
    }}

    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {{
        display: flex !important;
        visibility: visible !important;
        pointer-events: auto !important;
        position: relative !important;
        top: 0.72rem !important;
        left: 0.72rem !important;
        z-index: 999999 !important;
        width: 2.1rem !important;
        height: 2.1rem !important;
        align-items: center !important;
        justify-content: center !important;
        border: 1px solid color-mix(in srgb, var(--border) 82%, transparent) !important;
        border-radius: var(--radius-sm) !important;
        background: color-mix(in srgb, var(--surface) 92%, transparent) !important;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.10) !important;
    }}

    [data-testid="stSidebarCollapsedControl"] [data-testid="stBaseButton-headerNoPadding"],
    [data-testid="collapsedControl"] [data-testid="stBaseButton-headerNoPadding"],
    [data-testid="stSidebarCollapseButton"] [data-testid="stBaseButton-headerNoPadding"],
    [data-testid="stBaseButton-header"] {{
        display: inline-flex !important;
        visibility: visible !important;
        pointer-events: auto !important;
        width: 2.1rem !important;
        height: 2.1rem !important;
        min-width: 2.1rem !important;
        min-height: 2.1rem !important;
        align-items: center !important;
        justify-content: center !important;
    }}

    [data-testid="stSidebar"][aria-expanded="false"] {{
        transform: none !important;
        width: 4.25rem !important;
        min-width: 4.25rem !important;
        flex-basis: 4.25rem !important;
        overflow: visible !important;
    }}

    [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarContent"] {{
        width: 4.25rem !important;
        padding: 0 !important;
        overflow: visible !important;
    }}

    [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarContent"] > :not([data-testid="stSidebarHeader"]) {{
        display: none !important;
    }}

    [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarHeader"] {{
        width: 4.25rem !important;
        height: 4.75rem !important;
        min-height: 4.75rem !important;
        align-items: flex-start !important;
        justify-content: center !important;
        padding: 1rem 0 0 !important;
    }}

    [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarHeader"] > :not([data-testid="stSidebarCollapseButton"]) {{
        display: none !important;
    }}

    [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarCollapseButton"] {{
        top: 0 !important;
        left: 0 !important;
        transform: none !important;
    }}

    [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {{
        transform: rotate(180deg) !important;
    }}

    [data-testid="stDecoration"] {{
        background: {header_background_css} !important;
    }}

    [data-testid="stToolbar"] {{
        display: none !important;
        background: transparent !important;
    }}

    [data-testid="stDeployButton"],
    [data-testid="stAppDeployButton"],
    [data-testid="stHeader"] [aria-label="Deploy"],
    [data-testid="stStatusWidget"],
    [data-testid="manage-app-button"] {{
        display: none !important;
    }}

    [data-testid="stSidebar"] {{
        background:
            linear-gradient(180deg, color-mix(in srgb, {sidebar_background_css} 92%, var(--surface-tint) 8%) 0%, {sidebar_background_css} 52%),
            {sidebar_background_css};
        border-right: 1px solid color-mix(in srgb, var(--border) 86%, transparent);
        box-shadow: inset -1px 0 0 color-mix(in srgb, var(--surface) 72%, transparent);
    }}

    .block-container {{
        padding-top: 0;
        padding-bottom: 1.3rem;
        padding-left: clamp(1.1rem, 2.2vw, 2rem);
        padding-right: clamp(1.1rem, 2.2vw, 2rem);
        max-width: 1480px;
    }}

    [data-testid="stSidebarContent"] {{
        padding: 0.92rem 0.86rem 1.05rem;
    }}

    .sidebar-workspace {{
        padding: 0.72rem 0.72rem 0.68rem;
        margin: 0 0 0.56rem;
        border: 1px solid color-mix(in srgb, var(--border) 82%, transparent);
        border-radius: var(--radius-md);
        background: color-mix(in srgb, var(--surface) 80%, transparent);
    }}

    .sidebar-workspace__eyebrow {{
        font-family: var(--font-ui);
        font-size: 0.62rem;
        font-weight: 720;
        color: var(--text-muted);
        line-height: 1.2;
        margin-bottom: 0.2rem;
    }}

    .sidebar-workspace__title {{
        font-family: var(--font-display);
        font-size: 1.08rem;
        font-weight: 720;
        line-height: 1.16;
        color: var(--text);
        margin-bottom: 0.24rem;
    }}

    .sidebar-workspace__meta {{
        color: var(--text-muted);
        font-size: 0.74rem;
        line-height: 1.42;
        margin-bottom: 0;
    }}

    .sidebar-ops-panel {{
        padding: 0.68rem 0.7rem 0.72rem;
        margin: 0 0 0.42rem;
        border: 1px solid color-mix(in srgb, var(--primary) 22%, var(--border));
        border-radius: var(--radius-md);
        background: color-mix(in srgb, var(--surface) 88%, var(--primary) 4%);
    }}

    .sidebar-ops-panel__header {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 0.5rem;
        margin-bottom: 0.52rem;
        padding-bottom: 0.42rem;
        border-bottom: 1px solid color-mix(in srgb, var(--border) 72%, transparent);
    }}

    .sidebar-ops-panel__header span {{
        color: var(--text) !important;
        font-size: 0.82rem !important;
        font-weight: 700;
        line-height: 1.2;
    }}

    .sidebar-ops-panel__header strong {{
        color: var(--text-muted);
        font-family: var(--font-mono);
        font-size: 0.66rem;
        font-weight: 680;
        line-height: 1.2;
    }}

    .sidebar-status-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.34rem;
    }}

    .sidebar-status-chip {{
        display: grid;
        gap: 0.06rem;
        min-width: 0;
        padding: 0.38rem 0.44rem;
        border: 1px solid color-mix(in srgb, var(--border) 82%, transparent);
        border-radius: var(--radius-sm);
        background: color-mix(in srgb, var(--bg) 48%, var(--surface) 52%);
    }}

    .sidebar-status-chip span {{
        color: var(--text-muted) !important;
        font-size: 0.68rem !important;
        line-height: 1.15;
    }}

    .sidebar-status-chip strong {{
        color: var(--text);
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 680;
        line-height: 1.2;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}

    .sidebar-status-chip--ready {{
        border-color: color-mix(in srgb, var(--success) 24%, var(--border));
    }}

    .sidebar-status-chip--attention {{
        border-color: color-mix(in srgb, var(--warning) 30%, var(--border));
    }}

    .sidebar-section-label {{
        margin: 0.78rem 0 0.34rem;
        padding: 0.52rem 0.08rem 0;
        border-top: 1px solid color-mix(in srgb, var(--border) 58%, transparent);
        color: var(--text-muted);
        font-family: var(--font-ui);
        font-size: 0.72rem;
        font-weight: 660;
        line-height: 1.2;
    }}

    .sidebar-footer-label {{
        margin-top: 0.78rem;
        padding-top: 0.62rem;
        border-top: 1px solid color-mix(in srgb, var(--border) 70%, transparent);
        color: var(--text-muted);
        font-size: 0.72rem;
        line-height: 1.35;
    }}

    [data-testid="stSidebar"] [data-testid="stHeading"] h1,
    [data-testid="stSidebar"] h1 {{
        font-size: 1.12rem !important;
        font-weight: 680 !important;
        letter-spacing: 0;
        margin-bottom: 0.9rem;
    }}

    [data-testid="stSidebar"] hr {{
        margin: 0.9rem 0;
        border-color: color-mix(in srgb, var(--border) 72%, transparent);
    }}

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
        padding: 0.08rem 0 0.72rem;
        margin-bottom: 0.56rem;
        border-bottom: 1px solid color-mix(in srgb, var(--border) 70%, transparent);
    }}

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] ul {{
        gap: 0.18rem;
    }}

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] span,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] p {{
        font-family: var(--font-ui) !important;
        font-size: 0.86rem !important;
        font-weight: 620 !important;
        color: var(--text-muted) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
        min-height: 2rem;
        border-radius: var(--radius-sm);
        padding: 0.38rem 0.58rem;
        color: var(--text-muted) !important;
        transition: background-color 0.16s ease, color 0.16s ease, border-color 0.16s ease;
    }}

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
        background: {control_hover_css};
        color: var(--text) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"],
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-selected="true"] {{
        background: color-mix(in srgb, var(--primary) 11%, var(--surface) 89%);
        color: var(--primary) !important;
        box-shadow: inset 2px 0 0 var(--primary);
    }}

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] [data-testid="stIconMaterial"],
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] span[class*="material"],
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] i[class*="material"],
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] [class*="material-symbol"] {{
        display: none !important;
    }}

    [data-testid="stSidebar"] [data-testid="stIconMaterial"],
    [data-testid="stSidebar"] span[class*="material"],
    [data-testid="stSidebar"] i[class*="material"],
    [data-testid="stSidebar"] [class*="material-symbol"] {{
        font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
        font-size: 1rem !important;
        line-height: 1 !important;
        width: 1rem;
        min-width: 1rem;
        max-width: 1rem;
        overflow: hidden;
        white-space: nowrap;
        flex: 0 0 auto;
    }}

    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        position: relative;
        padding-right: 2.15rem !important;
    }}

    [data-testid="stSidebar"] [data-baseweb="select"] svg,
    [data-testid="stSidebar"] [data-baseweb="select"] [role="img"] {{
        position: absolute;
        right: 0.72rem;
        top: 50%;
        transform: translateY(-50%);
        pointer-events: none;
    }}

    div[data-testid="stMetricValue"], code, pre {{
        font-family: var(--font-mono) !important;
        font-variant-numeric: tabular-nums;
    }}

    div[data-testid="stMetricValue"] {{
        color: var(--text) !important;
    }}

    div[data-testid="stMetricLabel"], .stCaption {{
        color: var(--text-muted);
    }}

    .stApp label,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stExpander"] summary,
    .stApp [data-testid="stExpander"] summary p {{
        color: var(--text) !important;
    }}

    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: var(--text);
        font-weight: {heading_font_weight};
        line-height: {heading_line_height};
        letter-spacing: 0;
    }}

    .stApp h1,
    .stApp h2 {{
        font-family: var(--font-display) !important;
        font-weight: {display_font_weight};
        line-height: {display_line_height};
        letter-spacing: 0;
    }}

    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {{
        font-family: var(--font-ui) !important;
    }}

    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li {{
        color: var(--text);
    }}

    .stApp [data-testid="stMarkdownContainer"] code,
    .stApp [data-testid="stMarkdownContainer"] pre {{
        color: var(--text);
    }}

    .stApp code {{
        background: {inline_code_bg_css};
        color: var(--text);
        border-radius: var(--radius-xs);
        padding: 1px 6px;
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {{
        color: var(--text) !important;
    }}

    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {{
        color: var(--text-muted) !important;
    }}

    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div,
    [data-testid="stSidebar"] [data-baseweb="textarea"] > div,
    [data-testid="stSidebar"] .stDateInput > div > div,
    [data-testid="stSidebar"] button[kind] {{
        background-color: {control_bg_css};
        border: 1px solid {control_border_css};
        color: var(--text);
        border-radius: var(--radius-sm);
        min-height: 2.36rem;
        box-shadow: 0 1px 0 color-mix(in srgb, var(--surface) 72%, transparent);
    }}

    [data-testid="stSidebar"] button[kind]:hover {{
        background-color: {control_hover_css};
        border-color: {tokens['primary']};
    }}

    .stApp .stButton > button,
    .stApp .stDownloadButton > button,
    .stApp div[data-testid="stFormSubmitButton"] > button {{
        font-family: var(--font-ui) !important;
        font-size: {button_font_size} !important;
        font-weight: {button_font_weight} !important;
        letter-spacing: 0;
        border-radius: var(--radius-sm) !important;
        min-height: 2.52rem;
        padding: 0.58rem 0.96rem;
        border: 1px solid var(--border) !important;
        background: var(--surface) !important;
        color: var(--text) !important;
        box-shadow: none !important;
        transition: background-color 0.18s ease, border-color 0.18s ease, color 0.18s ease, transform 0.18s ease;
    }}

    .stApp .stButton > button:hover,
    .stApp .stDownloadButton > button:hover,
    .stApp div[data-testid="stFormSubmitButton"] > button:hover {{
        background: var(--accent-blue-hover) !important;
        border-color: var(--accent-blue-hover) !important;
        color: #ffffff !important;
        transform: translateY(0);
    }}

    .stApp .stButton > button[kind="primary"],
    .stApp div[data-testid="stFormSubmitButton"] > button[kind="primary"] {{
        background: var(--primary) !important;
        border-color: var(--primary) !important;
        color: #ffffff !important;
    }}

    .stApp [data-testid="stVerticalBlockBorderWrapper"],
    .stApp [data-testid="stContainer"] [data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: var(--border) !important;
        border-radius: var(--radius-md) !important;
        background: color-mix(in srgb, var(--surface) 97%, var(--surface-tint) 3%) !important;
        box-shadow: none !important;
    }}

    .stApp [data-testid="stVerticalBlockBorderWrapper"] > div {{
        border-radius: var(--radius-md) !important;
    }}

    .app-summary-card {{
        border: 1px solid var(--border);
        background: {card_background_css};
        box-shadow: none;
        border-radius: var(--radius-sm);
        padding: 15px 17px;
        margin-bottom: 0.92rem;
        transition: border-color 0.2s ease, background-color 0.2s ease;
    }}

    @keyframes riseIn {{
        from {{ opacity: 0; transform: translate3d(0, 12px, 0); }}
        to {{ opacity: 1; transform: translate3d(0, 0, 0); }}
    }}

    .page-shell {{
        position: relative;
        overflow: hidden;
        border: 0;
        background: transparent;
        box-shadow: none;
        border-radius: 0;
        padding: {layout_tokens['page_shell_padding']};
        margin-bottom: 0.24rem;
        animation: riseIn 420ms cubic-bezier(0.16, 1, 0.3, 1) both;
    }}

    .page-shell__grid {{
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 1rem;
        align-items: center;
    }}

    .page-shell__main {{
        max-width: 58ch;
    }}

    .page-shell__meta {{
        align-self: center;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        gap: 0.45rem;
        padding: 0;
        border-left: 0;
        background: transparent;
        border-radius: 0;
        box-shadow: none;
        overflow: visible;
    }}

    .page-shell__meta-eyebrow {{
        color: var(--text-muted);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        letter-spacing: 0;
        text-transform: none;
        text-align: right;
    }}

    .page-shell__eyebrow {{
        color: var(--text-muted);
        font-family: var(--font-ui);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        letter-spacing: 0;
        text-transform: none;
        margin-bottom: 0.22rem;
    }}

    .page-shell__title {{
        color: var(--text);
        font-family: var(--font-display);
        font-size: {display_hero_size};
        font-weight: {display_font_weight};
        line-height: 1.08;
        letter-spacing: 0;
    }}

    .page-shell__description {{
        color: var(--text-muted);
        font-size: {body_font_size};
        line-height: 1.38;
        max-width: 56ch;
        margin-top: 0.18rem;
        text-wrap: pretty;
    }}

    .page-shell__pills {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.34rem;
        justify-content: flex-end;
    }}

    .page-shell__pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.32rem 0.58rem;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--surface) 84%, var(--surface-tint) 16%);
        font-family: var(--font-ui);
        font-size: {caption_font_size};
        color: var(--text-muted);
        letter-spacing: 0;
        text-transform: none;
    }}

    .page-shell__pill strong {{
        color: var(--text);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
    }}

    .page-shell__pill[data-tone="success"] {{
        border-color: color-mix(in srgb, var(--success) 46%, var(--border));
    }}

    .page-shell__pill[data-tone="warning"] {{
        border-color: color-mix(in srgb, var(--warning) 46%, var(--border));
    }}

    .page-shell__pill[data-tone="danger"] {{
        border-color: color-mix(in srgb, var(--danger) 46%, var(--border));
    }}

    .page-shell__pill[data-tone="info"] {{
        border-color: color-mix(in srgb, var(--primary) 42%, var(--border));
    }}

    .status-strip {{
        display: grid;
        grid-template-columns: auto 1fr auto;
        align-items: center;
        gap: 0.78rem;
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--surface) 96%, var(--surface-tint) 4%);
        box-shadow: none;
        border-radius: var(--radius-md);
        padding: 0.6rem 0.78rem;
        margin-bottom: 0.62rem;
        transition: none;
        animation: riseIn 480ms cubic-bezier(0.16, 1, 0.3, 1) both;
    }}

    .status-strip__badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 64px;
        border-radius: var(--radius-sm);
        padding: 0.28rem 0.52rem;
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        letter-spacing: 0;
        text-transform: none;
    }}

    .status-strip__title {{
        color: var(--text);
        font-family: var(--font-ui);
        font-size: {body_font_size};
        font-weight: {heading_font_weight};
        line-height: {heading_line_height};
    }}

    .status-strip__message {{
        color: var(--text-muted);
        font-size: {caption_font_size};
        margin-top: 0.06rem;
    }}

    .status-strip__meta {{
        color: var(--text-muted);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
    }}

    .status-strip[data-tone="error"] {{
        border-color: color-mix(in srgb, var(--danger) 46%, var(--border));
    }}

    .status-strip[data-tone="error"] .status-strip__badge {{
        background: color-mix(in srgb, var(--danger) 18%, transparent);
        border: 1px solid color-mix(in srgb, var(--danger) 56%, transparent);
        color: var(--danger);
    }}

    .status-strip[data-tone="warning"] {{
        border-color: color-mix(in srgb, var(--warning) 46%, var(--border));
    }}

    .status-strip[data-tone="warning"] .status-strip__badge {{
        background: color-mix(in srgb, var(--warning) 18%, transparent);
        border: 1px solid color-mix(in srgb, var(--warning) 56%, transparent);
        color: var(--warning);
    }}

    .status-strip[data-tone="info"] {{
        border-color: color-mix(in srgb, var(--primary) 40%, var(--border));
    }}

    .status-strip[data-tone="info"] .status-strip__badge {{
        background: color-mix(in srgb, var(--primary) 16%, transparent);
        border: 1px solid color-mix(in srgb, var(--primary) 56%, transparent);
        color: var(--primary);
    }}

    .overview-reference-shell {{
        margin-bottom: 0.48rem;
    }}

    .overview-command-surface {{
        margin-bottom: 0.4rem;
    }}

    .overview-command-surface__header {{
        display: grid;
        grid-template-columns: minmax(150px, 0.48fr) minmax(0, 2fr);
        gap: 0.86rem;
        align-items: start;
        margin-bottom: 0.62rem;
    }}

    .overview-command-surface__copy {{
        color: var(--text-muted);
        font-size: {body_small_font_size};
        line-height: {body_line_height};
        margin-top: 0.18rem;
    }}

    .overview-section-title {{
        color: var(--text);
        font-family: var(--font-ui);
        font-size: {section_title_size};
        font-weight: {heading_font_weight};
        line-height: {heading_line_height};
        margin-bottom: 0.42rem;
        letter-spacing: 0;
    }}

    .overview-market-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.5rem;
    }}

    .overview-market-card {{
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--surface) 88%, var(--surface-tint) 12%);
        border-radius: var(--radius-sm);
        padding: 0.62rem 0.72rem;
        min-height: 3.65rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}

    .overview-market-card--metric {{
        gap: 0.3rem;
    }}

    .overview-market-card__metric-row {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.55rem;
    }}

    .overview-market-card__label {{
        color: var(--text);
        font-size: {caption_font_size};
        font-weight: 620;
        line-height: 1.2;
    }}

    .overview-market-card__value {{
        color: var(--text);
        font-family: var(--font-mono);
        font-size: 1.02rem;
        font-weight: 680;
        font-variant-numeric: tabular-nums;
        line-height: 1.2;
        margin-top: 0;
    }}

    .overview-market-card__change {{
        font-family: var(--font-mono);
        font-size: {caption_font_size};
        font-weight: 620;
        margin-top: 0;
    }}

    .overview-market-card__change[data-tone="positive"] {{
        color: var(--success);
    }}

    .overview-market-card__change[data-tone="negative"] {{
        color: var(--danger);
    }}

    .overview-market-card--status {{
        gap: 0.42rem;
    }}

    .overview-market-card__status-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.8rem;
        color: var(--text-muted);
        font-size: {caption_font_size};
        line-height: 1.3;
    }}

    .overview-market-card__status-row strong {{
        color: var(--text);
        font-size: {body_small_font_size};
        font-weight: 650;
    }}

    .overview-lookup-chips {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.52rem;
    }}

    .overview-lookup-chip {{
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--surface) 88%, var(--surface-tint) 12%);
        color: var(--text-muted);
        border-radius: var(--radius-sm);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        padding: 0.36rem 0.58rem;
    }}

    .overview-lookup-chip[data-selected="true"] {{
        border-color: color-mix(in srgb, var(--primary) 42%, var(--border));
        background: color-mix(in srgb, var(--primary) 9%, #ffffff 91%);
        color: var(--primary);
    }}

    .overview-review-candidates {{
        margin-top: 0.72rem;
        padding-top: 0.72rem;
        border-top: 1px solid color-mix(in srgb, var(--border) 70%, transparent);
    }}

    .overview-review-candidates[data-empty="true"] {{
        padding-bottom: 0.18rem;
    }}

    .overview-review-candidates__header {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.72rem;
        margin-bottom: 0.58rem;
    }}

    .overview-review-candidates__basis {{
        flex-shrink: 0;
        border: 1px solid color-mix(in srgb, var(--primary) 26%, var(--border));
        background: color-mix(in srgb, var(--primary) 8%, transparent);
        color: var(--primary);
        border-radius: var(--radius-sm);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        line-height: 1.15;
        padding: 0.34rem 0.58rem;
        white-space: nowrap;
    }}

    .overview-review-candidates__grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.52rem;
    }}

    .overview-review-card {{
        min-width: 0;
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--surface) 90%, var(--surface-tint) 10%);
        border-radius: var(--radius-sm);
        padding: 0.68rem 0.72rem;
        animation: riseIn 420ms cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: border-color 0.18s ease, background-color 0.18s ease;
    }}

    .overview-review-card:hover {{
        border-color: color-mix(in srgb, var(--primary) 36%, var(--border));
        background: color-mix(in srgb, var(--surface) 84%, var(--surface-tint) 16%);
    }}

    .overview-review-card__topline {{
        display: flex;
        align-items: center;
        gap: 0.42rem;
        min-height: 1.48rem;
    }}

    .overview-review-card__rank {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.34rem;
        height: 1.34rem;
        border-radius: var(--radius-sm);
        background: color-mix(in srgb, var(--primary) 10%, transparent);
        color: var(--primary);
        font-family: var(--font-mono);
        font-size: {caption_font_size};
        font-weight: 650;
    }}

    .overview-review-card__sector {{
        color: var(--text);
        font-size: 0.96rem;
        font-weight: 680;
        line-height: 1.25;
        margin-top: 0.42rem;
        overflow-wrap: break-word;
    }}

    .overview-review-card__reasons {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        min-height: 1.82rem;
        margin-top: 0.52rem;
    }}

    .overview-review-card__metrics {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.54rem;
        margin-top: 0.12rem;
        color: var(--text-muted);
        font-size: {caption_font_size};
        line-height: 1.3;
    }}

    .overview-review-card__metric strong {{
        color: var(--text);
        font-weight: 620;
        margin-right: 0.22rem;
    }}

    .overview-review-card__invalidation {{
        display: flex;
        gap: 0.42rem;
        margin-top: 0.46rem;
        padding-top: 0.44rem;
        border-top: 1px solid color-mix(in srgb, var(--border) 70%, transparent);
        color: var(--text-muted);
        font-size: {caption_font_size};
        line-height: 1.35;
    }}

    .overview-review-card__invalidation span {{
        flex-shrink: 0;
        font-weight: 620;
    }}

    .overview-review-card__invalidation strong {{
        color: var(--text);
        font-weight: 560;
    }}

    .overview-sector-table-wrap {{
        max-height: 462px;
        overflow: auto;
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        background: var(--surface);
    }}

    .overview-sector-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: {caption_font_size};
        color: var(--text);
    }}

    .overview-sector-table th,
    .overview-sector-table td {{
        padding: 0.58rem 0.56rem;
        border-bottom: 1px solid var(--border);
        white-space: nowrap;
        vertical-align: middle;
    }}

    .overview-sector-table th {{
        position: sticky;
        top: 0;
        z-index: 1;
        background: color-mix(in srgb, var(--surface) 88%, var(--surface-tint) 12%);
        color: var(--text-muted);
        font-weight: {badge_font_weight};
        text-align: left;
    }}

    .overview-sector-table td:first-child,
    .overview-sector-table td:nth-child(3),
    .overview-sector-table td:nth-child(4),
    .overview-sector-table td:nth-child(5) {{
        font-family: var(--font-mono);
        font-variant-numeric: tabular-nums;
        text-align: right;
    }}

    .overview-sector-table td:nth-child(2) {{
        font-weight: 650;
        min-width: 8rem;
    }}

    .overview-sector-subtext {{
        display: block;
        margin-top: 0.16rem;
        font-size: 0.72em;
        font-weight: 560;
        color: var(--muted);
        line-height: 1.15;
    }}

    .overview-sector-table td[data-tone="positive"] {{
        color: var(--success);
        font-weight: 650;
    }}

    .overview-sector-table td[data-tone="negative"] {{
        color: var(--danger);
        font-weight: 650;
    }}

    .overview-heatmap-grid {{
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.35rem;
    }}

    .overview-heatmap-tile {{
        min-height: 3.75rem;
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.62rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 0.18rem;
        text-align: center;
    }}

    .overview-heatmap-tile[data-tone="positive"] {{
        background: color-mix(in srgb, var(--success) var(--tile-strength), #ffffff);
        color: #ffffff;
        border-color: color-mix(in srgb, var(--success) 42%, var(--border));
    }}

    .overview-heatmap-tile[data-tone="negative"] {{
        background: color-mix(in srgb, var(--danger) var(--tile-strength), #ffffff);
        color: #ffffff;
        border-color: color-mix(in srgb, var(--danger) 42%, var(--border));
    }}

    .overview-heatmap-tile span {{
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        line-height: 1.24;
    }}

    .overview-heatmap-tile strong {{
        font-family: var(--font-mono);
        font-size: 0.98rem;
        font-weight: 680;
        font-variant-numeric: tabular-nums;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: var(--border) !important;
        background: {card_background_css};
        box-shadow: {card_shadow_css};
        border-radius: var(--radius-lg) !important;
    }}

    .command-bar {{
        margin-bottom: 0.62rem;
    }}

    .command-bar--compact {{
        display: grid;
        grid-template-columns: auto minmax(180px, 1fr) minmax(240px, 1.4fr);
        gap: 0.72rem;
        align-items: baseline;
    }}

    .command-bar__eyebrow {{
        color: var(--text-muted);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        letter-spacing: 0.08em;
        text-transform: lowercase;
        margin-bottom: 0.18rem;
    }}

    .command-bar__title {{
        color: var(--text);
        font-family: var(--font-ui);
        font-size: {section_title_size};
        font-weight: {heading_font_weight};
        line-height: {heading_line_height};
    }}

    .command-bar__note {{
        color: var(--text-muted);
        font-size: {caption_font_size};
        line-height: 1.35;
    }}

    .filter-chip-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        align-items: center;
        min-height: 2.42rem;
    }}

    .filter-chip-row span {{
        display: inline-flex;
        align-items: center;
        gap: 0.28rem;
        min-height: 1.95rem;
        border: 1px solid color-mix(in srgb, var(--primary) 18%, var(--border));
        background: color-mix(in srgb, var(--surface) 92%, var(--surface-tint) 8%);
        border-radius: var(--radius-sm);
        padding: 0.34rem 0.66rem;
        color: var(--text);
        font-size: {caption_font_size};
        font-weight: 680;
        line-height: 1.2;
    }}

    .filter-chip-row b {{
        color: var(--text-muted);
        font-weight: {badge_font_weight};
    }}

    .top-bar-summary {{
        border: 1px solid color-mix(in srgb, var(--primary) 18%, var(--border));
        background: color-mix(in srgb, var(--surface) 90%, var(--surface-tint) 10%);
        border-radius: var(--radius-md);
        color: var(--text);
        font-size: {body_small_font_size};
        line-height: {body_line_height};
        padding: 0.86rem 1rem;
        min-height: 3.35rem;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 0.65rem;
    }}

    .top-bar-summary__item {{
        display: flex;
        flex-direction: column;
        gap: 0.16rem;
    }}

    .top-bar-summary__item span {{
        color: var(--text-muted);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        letter-spacing: 0.05em;
        text-transform: lowercase;
    }}

    .top-bar-summary__item strong {{
        color: var(--text);
        font-size: {body_small_font_size};
        font-weight: {button_font_weight};
    }}

    .analysis-toolbar {{
        border: 1px solid var(--border);
        background: var(--surface);
        box-shadow: none;
        border-radius: var(--radius-lg);
        padding: {layout_tokens['panel_padding']};
        margin-bottom: 0.86rem;
    }}

    .analysis-toolbar__eyebrow {{
        color: var(--text-muted);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        letter-spacing: 0;
        text-transform: none;
        margin-bottom: 0.16rem;
    }}

    .analysis-toolbar__title {{
        color: var(--text);
        font-family: var(--font-ui);
        font-size: {section_title_size};
        font-weight: {heading_font_weight};
        line-height: {heading_line_height};
        letter-spacing: 0;
    }}

    .analysis-toolbar__description {{
        color: var(--text-muted);
        font-size: {body_small_font_size};
        line-height: 1.5;
        letter-spacing: 0;
        max-width: 72ch;
        margin-top: 0.2rem;
    }}

    .analysis-toolbar__summary {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.68rem;
        margin-top: 0.7rem;
    }}

    .analysis-toolbar__summary-item {{
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--surface) 88%, var(--surface-tint) 12%);
        border-radius: var(--radius-sm);
        padding: 0.7rem 0.78rem;
        min-height: 2.95rem;
    }}

    .analysis-toolbar__summary-item span {{
        display: block;
        color: var(--text-muted);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        letter-spacing: 0;
        text-transform: none;
        margin-bottom: 0.22rem;
    }}

    .analysis-toolbar__summary-item strong {{
        color: var(--text);
        font-size: {body_small_font_size};
        font-weight: {button_font_weight};
        line-height: {body_line_height};
    }}

    .phase-chip-row {{
        margin-bottom: 0.55rem;
    }}

    .cycle-palette {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.7rem;
    }}

    .cycle-palette__label {{
        color: var(--text-muted);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        letter-spacing: 0;
        text-transform: none;
    }}

    .cycle-palette__item {{
        display: inline-flex;
        align-items: center;
        gap: 0.42rem;
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        background: color-mix(in srgb, var(--surface) 92%, var(--surface-tint) 8%);
        color: var(--text);
        font-size: {caption_font_size};
        font-weight: {button_font_weight};
        padding: 0.46rem 0.78rem;
    }}

    .cycle-palette__swatch {{
        width: 10px;
        height: 10px;
        border-radius: var(--radius-full);
        display: inline-block;
        border: 1px solid {ui_tokens['cycle_swatch_border']};
    }}

    .cycle-palette__swatch.cycle-recovery {{
        background: {signal_tokens['cycle_swatches']['Recovery']};
    }}

    .cycle-palette__swatch.cycle-expansion {{
        background: {signal_tokens['cycle_swatches']['Expansion']};
    }}

    .cycle-palette__swatch.cycle-slowdown {{
        background: {signal_tokens['cycle_swatches']['Slowdown']};
    }}

    .cycle-palette__swatch.cycle-contraction {{
        background: {signal_tokens['cycle_swatches']['Contraction']};
    }}

    .cycle-palette__swatch.cycle-indeterminate {{
        background: {signal_tokens['cycle_swatches']['Indeterminate']};
    }}

    .sector-rank-list__header {{
        margin-bottom: 0.65rem;
    }}

    .sector-rank-list__eyebrow {{
        color: var(--text-muted);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        letter-spacing: 0.08em;
        text-transform: lowercase;
        margin-bottom: 0.16rem;
    }}

    .sector-rank-list__title {{
        color: var(--text);
        font-family: var(--font-ui);
        font-size: {card_title_size};
        font-weight: {heading_font_weight};
        line-height: {heading_line_height};
    }}

    .sector-rank-list__metric {{
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--surface) 92%, var(--surface-tint) 8%);
        color: var(--text-muted);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        line-height: 1;
        padding: 0.68rem 0.7rem;
        text-align: center;
        margin-top: 0.08rem;
    }}

    .sector-rank-list__metric[data-selected="true"] {{
        border-color: color-mix(in srgb, var(--primary) 42%, var(--border));
        color: var(--text);
    }}

    .sector-rank-list__metric[data-tone="positive"] {{
        color: var(--success);
    }}

    .sector-rank-list__metric[data-tone="negative"] {{
        color: var(--danger);
    }}

    .decision-hero {{
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(260px, 0.85fr);
        gap: 0.95rem;
        border: 1px solid var(--border);
        background: var(--surface);
        box-shadow: none;
        border-radius: var(--radius-lg);
        padding: 1rem 1.08rem 0.98rem;
        margin-bottom: 0.9rem;
        position: relative;
        overflow: hidden;
        animation: riseIn 520ms cubic-bezier(0.16, 1, 0.3, 1) both;
    }}

    .decision-hero::before {{
        content: "";
        position: absolute;
        inset: 0 0 auto;
        height: 2px;
        width: auto;
        background: var(--decision-hero-accent, var(--primary));
    }}

    .decision-hero::after {{
        display: none;
    }}

    .decision-hero__eyebrow {{
        color: var(--text-muted);
        font-size: {caption_font_size};
        letter-spacing: 0;
        text-transform: none;
        margin-bottom: 0.26rem;
    }}

    .decision-hero__copy {{
        position: relative;
        z-index: 1;
        margin-bottom: 0.7rem;
    }}

    .decision-hero__title {{
        color: var(--text);
        font-family: var(--font-display);
        font-size: {display_secondary_size};
        font-weight: {display_font_weight};
        letter-spacing: 0;
        line-height: 1.08;
    }}

    .decision-hero__subtitle {{
        color: var(--text-muted);
        font-size: {body_small_font_size};
        font-weight: 520;
        margin-top: 0.25rem;
        letter-spacing: 0;
        max-width: 56ch;
    }}

    .decision-hero__chips {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.62rem;
    }}

    .decision-hero__chip {{
        border: 1px solid color-mix(in srgb, var(--primary) 32%, var(--border));
        background: color-mix(in srgb, var(--surface) 86%, var(--primary) 14%);
        color: var(--text);
        border-radius: var(--radius-sm);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        line-height: 1;
        padding: 0.34rem 0.52rem;
        white-space: nowrap;
        text-transform: none;
    }}

    .decision-hero__badge {{
        border: 1px solid {provisional_badge_border};
        background: color-mix(in srgb, var(--warning) 18%, transparent);
        color: {provisional_badge_text};
        border-radius: var(--radius-sm);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        line-height: 1;
        padding: 0.34rem 0.52rem;
        white-space: nowrap;
        text-transform: none;
    }}

    .decision-hero__stats,
    .status-card-grid,
    .summary-kpi-grid {{
        display: grid;
        gap: 0.58rem;
    }}

    .decision-hero__stats {{
        position: relative;
        z-index: 1;
        grid-template-columns: 1fr;
        align-self: stretch;
    }}

    .status-card-grid {{
        grid-template-columns: repeat(4, minmax(0, 1fr));
        margin-bottom: 0.72rem;
    }}

    .summary-kpi-grid {{
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        margin-bottom: 0.72rem;
    }}

    .decision-hero__stat,
    .status-card {{
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--surface) 97%, var(--surface-tint) 3%);
        border-radius: var(--radius-lg);
        padding: 0.64rem 0.72rem;
        transition: border-color 0.18s ease, background-color 0.18s ease;
        box-shadow: none;
    }}
    
    .decision-hero__stat:hover,
    .status-card:hover {{
        border-color: color-mix(in srgb, var(--primary) 36%, var(--border));
        background: color-mix(in srgb, var(--surface) 92%, var(--primary) 8%);
    }}

    .decision-hero__stat-label,
    .status-card__eyebrow {{
        display: block;
        color: var(--text-muted);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        letter-spacing: 0;
        text-transform: none;
        margin-bottom: 0.24rem;
    }}

    .decision-hero__stat strong,
    .status-card__value {{
        display: block;
        color: var(--text);
        font-size: 0.96rem;
        font-weight: 650;
        line-height: 1.2;
    }}

    .status-card__detail {{
        color: var(--text-muted);
        font-size: 0.74rem;
        line-height: 1.45;
        margin-top: 0.32rem;
    }}

    .status-card[data-tone="success"] {{
        border-color: color-mix(in srgb, var(--success) 42%, var(--border));
    }}

    .status-card[data-tone="warning"] {{
        border-color: color-mix(in srgb, var(--warning) 46%, var(--border));
    }}

    .status-card[data-tone="danger"] {{
        border-color: color-mix(in srgb, var(--danger) 46%, var(--border));
    }}

    .status-card[data-tone="info"] {{
        border-color: color-mix(in srgb, var(--primary) 42%, var(--border));
    }}

    .compact-note {{
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--surface) 94%, transparent);
        border-radius: var(--radius-md);
        padding: 0.88rem 1rem;
        margin-bottom: 0.9rem;
    }}

    .compact-note b {{
        display: block;
        margin-bottom: 0.2rem;
    }}

    .research-page-frame {{
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(260px, 0.65fr);
        gap: 0.9rem;
        align-items: stretch;
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        background: color-mix(in srgb, var(--surface) 96%, var(--surface-tint) 4%);
        padding: 0.9rem 1rem;
        margin: 0 0 0.82rem;
    }}

    .research-page-frame__copy {{
        min-width: 0;
    }}

    .research-page-frame__eyebrow {{
        color: var(--text-muted);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        margin-bottom: 0.2rem;
    }}

    .research-page-frame__title {{
        color: var(--text);
        font-family: var(--font-ui);
        font-size: {section_title_size};
        font-weight: {heading_font_weight};
        line-height: {heading_line_height};
    }}

    .research-page-frame__description {{
        color: var(--text-muted);
        font-size: {body_small_font_size};
        line-height: {body_line_height};
        margin-top: 0.24rem;
        max-width: 82ch;
    }}

    .research-page-frame__summary {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.48rem;
        align-self: stretch;
    }}

    .research-page-frame__item {{
        min-width: 0;
        border: 1px solid color-mix(in srgb, var(--border) 82%, transparent);
        border-radius: var(--radius-sm);
        background: color-mix(in srgb, var(--surface) 92%, var(--surface-tint) 8%);
        padding: 0.58rem 0.66rem;
    }}

    .research-page-frame__item span {{
        display: block;
        color: var(--text-muted);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        line-height: 1.25;
        margin-bottom: 0.22rem;
    }}

    .research-page-frame__item strong {{
        display: block;
        color: var(--text);
        font-size: {body_small_font_size};
        font-weight: {heading_font_weight};
        line-height: 1.28;
        overflow-wrap: break-word;
    }}

    .panel-header {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.68rem;
    }}

    .panel-header__eyebrow {{
        color: var(--text-muted);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        letter-spacing: 0;
        text-transform: none;
        margin-bottom: 0.18rem;
    }}

    .panel-header__title {{
        color: var(--text);
        font-family: var(--font-ui);
        font-size: {section_title_size};
        font-weight: {heading_font_weight};
        line-height: {heading_line_height};
    }}

    .panel-header__description {{
        color: var(--text-muted);
        font-size: {body_small_font_size};
        line-height: {body_line_height};
        margin-top: 0.18rem;
        max-width: 72ch;
    }}

    .panel-header__badge {{
        display: inline-flex;
        align-items: center;
        border-radius: var(--radius-sm);
        border: 1px solid color-mix(in srgb, var(--primary) 38%, var(--border));
        background: color-mix(in srgb, var(--surface) 86%, var(--primary) 14%);
        color: var(--text);
        font-size: {caption_font_size};
        font-weight: {badge_font_weight};
        padding: 0.42rem 0.72rem;
        white-space: nowrap;
        text-transform: none;
    }}

    .top-picks-container {{
        display: flex;
        flex-direction: column;
        gap: 0.52rem;
        margin-bottom: 0.6rem;
    }}

    .top-pick-card {{
        border: 1px solid var(--border);
        background: var(--surface);
        border-radius: var(--radius-lg);
        padding: 0.76rem 0.84rem;
        transition: border-color 0.18s ease, background-color 0.18s ease;
        animation: riseIn 460ms cubic-bezier(0.16, 1, 0.3, 1) both;
    }}

    .top-pick-card:hover {{
        border-color: color-mix(in srgb, var(--primary) 32%, var(--border));
        background: color-mix(in srgb, var(--surface) 94%, var(--surface-tint) 6%);
    }}

    .top-pick-card__header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.52rem;
        padding-bottom: 0.42rem;
        border-bottom: 1px solid color-mix(in srgb, var(--border) 60%, transparent);
    }}

    .top-pick-card__title {{
        color: var(--text);
        font-size: 0.9rem;
        font-weight: 650;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.45rem;
    }}

    .top-pick-card__rank {{
        color: var(--text-muted);
        font-size: 0.88rem;
        font-weight: 600;
        display: inline-block;
        min-width: 1.2rem;
    }}

    .top-pick-card__held-badge {{
        font-size: {caption_font_size};
        background: color-mix(in srgb, var(--success) 15%, transparent);
        border: 1px solid color-mix(in srgb, var(--success) 35%, transparent);
        color: var(--success);
        padding: 0.15rem 0.45rem;
        border-radius: var(--radius-sm);
        margin-left: 0.2rem;
        font-weight: {badge_font_weight};
    }}

    .top-pick-card__decision {{
        font-size: {body_small_font_size};
        font-weight: {button_font_weight};
        color: var(--primary);
        background: color-mix(in srgb, var(--primary) 12%, transparent);
        border: 1px solid color-mix(in srgb, var(--primary) 25%, transparent);
        padding: 0.3rem 0.7rem;
        border-radius: var(--radius-sm);
        white-space: nowrap;
        margin-left: 0.5rem;
        text-transform: none;
    }}

    .top-pick-card__body {{
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
    }}

    .top-pick-card__row {{
        display: flex;
        align-items: baseline;
        gap: 0.65rem;
        font-size: 0.8rem;
    }}

    .top-pick-card__label {{
        color: var(--text-muted);
        min-width: 75px;
        max-width: 75px;
        font-weight: 620;
        font-size: 0.72rem;
        flex-shrink: 0;
    }}

    .top-pick-card__value {{
        color: var(--text);
        flex: 1;
        line-height: 1.5;
    }}

    .top-pick-card__metrics {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem;
        margin-top: 0.28rem;
        padding-top: 0.48rem;
        border-top: 1px solid color-mix(in srgb, var(--border) 70%, transparent);
        font-size: 0.74rem;
        color: var(--text-muted);
    }}
    
    .top-pick-card__metric strong {{
        color: var(--text);
        font-weight: 650;
        margin-right: 0.25rem;
    }}

    .flow-container {{
        display: flex;
        flex-direction: column;
        gap: 0.52rem;
        margin-bottom: 0.6rem;
    }}

    .flow-card {{
        border: 1px solid var(--border);
        background: var(--surface);
        border-radius: var(--radius-lg);
        padding: 0.76rem 0.84rem;
        transition: border-color 0.18s ease, background-color 0.18s ease;
        animation: riseIn 500ms cubic-bezier(0.16, 1, 0.3, 1) both;
    }}

    .flow-card:hover {{
        border-color: color-mix(in srgb, var(--primary) 32%, var(--border));
        background: color-mix(in srgb, var(--surface) 94%, var(--surface-tint) 6%);
    }}

    .flow-card__header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.52rem;
        padding-bottom: 0.42rem;
        border-bottom: 1px solid color-mix(in srgb, var(--border) 60%, transparent);
    }}

    .flow-card__title {{
        color: var(--text);
        font-size: 0.9rem;
        font-weight: 650;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.45rem;
    }}

    .flow-card__rank {{
        color: var(--text-muted);
        font-size: 0.88rem;
        font-weight: 600;
        display: inline-block;
        min-width: 1.2rem;
    }}

    .flow-card__badge {{
        font-size: {caption_font_size};
        font-weight: {button_font_weight};
        border: 1px solid transparent;
        padding: 0.25rem 0.65rem;
        border-radius: var(--radius-sm);
        white-space: nowrap;
        margin-left: 0.5rem;
        text-transform: none;
    }}

    .flow-card__badge--success {{
        color: var(--success);
        background: color-mix(in srgb, var(--success) 12%, transparent);
        border-color: color-mix(in srgb, var(--success) 25%, transparent);
    }}

    .flow-card__badge--neutral {{
        color: var(--info);
        background: color-mix(in srgb, var(--info) 12%, transparent);
        border-color: color-mix(in srgb, var(--info) 25%, transparent);
    }}

    .flow-card__badge--warning {{
        color: var(--warning);
        background: color-mix(in srgb, var(--warning) 12%, transparent);
        border-color: color-mix(in srgb, var(--warning) 25%, transparent);
    }}

    .flow-card__badge--danger {{
        color: var(--danger);
        background: color-mix(in srgb, var(--danger) 12%, transparent);
        border-color: color-mix(in srgb, var(--danger) 25%, transparent);
    }}

    .flow-card__body {{
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
    }}

    .flow-card__row {{
        display: flex;
        align-items: baseline;
        gap: 0.65rem;
        font-size: {body_small_font_size};
    }}

    .flow-card__label {{
        color: var(--text-muted);
        min-width: 75px;
        max-width: 75px;
        font-weight: 620;
        font-size: {caption_font_size};
        flex-shrink: 0;
    }}

    .flow-card__value {{
        color: var(--text);
        flex: 1;
        line-height: 1.5;
    }}


    [data-testid="stDataFrame"] {{
        border: 1px solid {table_tokens['grid']};
        border-radius: var(--radius-sm);
        overflow: hidden;
        background: {table_tokens['row_bg_even']};
    }}

    [data-testid="stDataFrame"] div[class*="glideDataEditor"] {{
        --gdg-bg-header: {table_tokens['header_bg']};
        --gdg-bg-cell: {table_tokens['row_bg_even']};
        --gdg-bg-cell-medium: {table_tokens['row_bg_odd']};
        --gdg-color: {table_tokens['row_text']};
        --gdg-border-color: {table_tokens['grid']};
        --gdg-horizontal-border-color: {table_tokens['grid']};
        --gdg-vertical-border-color: {table_tokens['grid']};
        --gdg-header-font-style: 700 13.5px {ui_font_css};
    }}

    [data-testid="stDataFrame"] div[class*="glideDataEditor"] > div {{
        background: {table_tokens['row_bg_even']} !important;
    }}

    [data-testid="stDataFrame"] ::-webkit-scrollbar-track {{
        background: {table_tokens['row_bg_odd']};
    }}

    [data-testid="stDataFrame"] ::-webkit-scrollbar-thumb {{
        background: color-mix(in srgb, {table_tokens['grid']} 70%, {tokens['primary']} 30%);
        border-radius: 10px;
    }}

    [data-testid="stDataFrame"] [role="grid"],
    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stDataFrame"] [role="columnheader"] {{
        color: {table_tokens['row_text']} !important;
    }}

    [data-testid="stDataFrame"] [role="columnheader"] {{
        background: {table_tokens['header_bg']} !important;
        color: {table_tokens['header_text']} !important;
    }}

    .provisional-badge {{
        background-color: color-mix(in srgb, var(--warning) 22%, transparent);
        color: {provisional_badge_text};
        border: 1px solid {provisional_badge_border};
        border-radius: var(--radius-sm);
        padding: 3px 9px;
        font-size: 12.5px;
        font-weight: {badge_font_weight};
        letter-spacing: 0;
        margin-left: 8px;
    }}

    .action-strong-buy {{
        background-color: {badge_styles['Strong Buy']['bg']};
        color: {badge_styles['Strong Buy']['text']};
        border: 1px solid {badge_styles['Strong Buy']['border']};
    }}
    .action-watch {{
        background-color: {badge_styles['Watch']['bg']};
        color: {badge_styles['Watch']['text']};
        border: 1px solid {badge_styles['Watch']['border']};
    }}
    .action-hold {{
        background-color: {badge_styles['Hold']['bg']};
        color: {badge_styles['Hold']['text']};
        border: 1px solid {badge_styles['Hold']['border']};
    }}
    .action-avoid {{
        background-color: {badge_styles['Avoid']['bg']};
        color: {badge_styles['Avoid']['text']};
        border: 1px solid {badge_styles['Avoid']['border']};
    }}
    .action-na {{
        background-color: {badge_styles['N/A']['bg']};
        color: {badge_styles['N/A']['text']};
        border: 1px solid {badge_styles['N/A']['border']};
    }}

    .action-strong-buy,
    .action-watch,
    .action-hold,
    .action-avoid,
    .action-na {{
        border-radius: var(--radius-sm);
        padding: 3px 9px;
        font-weight: {badge_font_weight};
        font-size: {badge_font_size};
        letter-spacing: 0;
        display: inline-block;
        text-align: center;
        min-width: 84px;
        text-transform: none;
    }}

    [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        gap: 0.3rem;
        border-bottom: 1px solid var(--border);
    }}

    [data-testid="stTabs"] [data-baseweb="tab"] {{
        font-family: var(--font-ui) !important;
        color: var(--text-muted) !important;
        border-radius: var(--radius-sm) var(--radius-sm) 0 0;
        padding: {layout_tokens['tab_padding']};
        font-size: {button_font_size};
        font-weight: {button_font_weight} !important;
        border: 1px solid transparent;
        background: transparent;
        transition: color 0.16s ease, background-color 0.16s ease, border-color 0.16s ease;
    }}

    [data-testid="stTabs"] [data-baseweb="tab"] p {{
        color: inherit !important;
    }}

    [data-testid="stTabs"] [data-baseweb="tab"]:hover {{
        color: var(--text) !important;
        background-color: {tab_tokens['tab_hover_bg']};
        border-color: var(--border) !important;
    }}

    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {{
        color: {tab_tokens['tab_selected_text']} !important;
        font-weight: {heading_font_weight} !important;
        background-color: {tab_tokens['tab_selected_bg']};
        border-color: {tab_tokens['tab_selected_border']} !important;
    }}

    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] p {{
        color: {tab_tokens['tab_selected_text']} !important;
    }}

    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
        background-color: {tab_tokens['tab_selected_border']} !important;
        height: 0;
    }}

    div[data-testid="stMetricValue"] {{
        font-size: 1.38rem;
        font-weight: {heading_font_weight} !important;
    }}

    .stApp [data-testid="stMetricLabel"] p,
    .stApp [data-testid="stCaptionContainer"] p {{
        color: var(--text-muted) !important;
        font-size: {caption_font_size};
        letter-spacing: 0.004em;
    }}

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span:not([class*="Toggle"]),
    [data-testid="stSidebar"] div:not([class*="Toggle"]) {{
        color: var(--text);
    }}

    .stApp button:focus-visible,
    .stApp input:focus-visible,
    .stApp textarea:focus-visible,
    .stApp [role="button"]:focus-visible {{
        outline: 2px solid var(--ring) !important;
        outline-offset: 2px !important;
        box-shadow: 0 0 0 4px color-mix(in srgb, var(--ring) 18%, transparent) !important;
    }}

    .empty-state-card {{
        border: 2px dashed var(--border);
        background: color-mix(in srgb, var(--surface) 50%, transparent);
        border-radius: var(--radius-lg);
        padding: 3rem 1.5rem;
        text-align: center;
        color: var(--text-muted);
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }}

    .empty-state-card h4 {{
        color: var(--text);
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
        font-weight: 650;
        font-size: 1.15rem;
    }}

    .empty-state-card p {{
        font-size: 0.92rem;
        margin-bottom: 0;
        max-width: 48ch;
    }}

    @media (max-width: 1120px) {{
        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}

        .page-shell__grid {{
            grid-template-columns: minmax(0, 1.2fr) minmax(210px, 0.8fr);
            gap: 1rem;
        }}

        .status-card-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}

        .decision-hero {{
            grid-template-columns: minmax(0, 1.1fr) minmax(220px, 0.9fr);
        }}
    }}

    @media (max-width: 840px) {{
        .stApp h1 {{ font-size: 1.95rem !important; line-height: 1.22 !important; }}
        .stApp h2 {{ font-size: 1.58rem !important; line-height: 1.24 !important; }}
        div[data-testid="stMetricValue"] {{ font-size: 1.42rem !important; }}

        .block-container {{
            padding-top: 1.05rem;
        }}

        .page-shell {{
            padding: 0.46rem 0 0.58rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 0.18rem;
        }}

        .page-shell__grid {{
            grid-template-columns: 1fr;
            gap: 0.34rem;
        }}

        .page-shell__meta {{
            padding: 0;
            border-left: 0;
            border-top: 0;
        }}

        .page-shell__pills {{
            gap: 0.26rem;
            justify-content: flex-start;
        }}

        .page-shell__pill:nth-child(n + 4) {{
            display: none;
        }}

        .page-shell__meta-eyebrow {{
            text-align: left;
            font-size: 0.62rem;
            display: none;
        }}

        .page-shell__title {{
            font-size: 1.36rem;
        }}

        .page-shell__description {{
            display: none;
        }}

        .page-shell__eyebrow {{
            font-size: 0.72rem;
            margin-bottom: 0.12rem;
        }}

        .page-shell__pill {{
            gap: 0.3rem;
            padding: 0.26rem 0.46rem;
            font-size: 0.68rem;
        }}

        .page-shell__pill strong {{
            font-size: 0.7rem;
        }}

        .status-strip {{
            grid-template-columns: auto 1fr auto;
            gap: 0.38rem;
            padding: 0.42rem 0.52rem;
            margin-bottom: 0.42rem;
            border-radius: var(--radius-sm);
        }}

        .status-strip__badge {{
            width: auto;
            min-width: 0;
            padding: 0.16rem 0.34rem;
            font-size: 0.68rem;
        }}

        .status-strip__title {{
            font-size: 0.82rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .status-strip__message {{
            display: none;
        }}

        .status-strip__meta {{
            white-space: nowrap;
            font-size: 0.68rem;
        }}

        [data-testid="stExpander"] details {{
            border-radius: var(--radius-sm) !important;
        }}

        [data-testid="stExpander"] summary {{
            min-height: 2.12rem !important;
            padding: 0.34rem 0.52rem !important;
        }}

        [data-testid="stExpander"] summary p {{
            font-size: 0.9rem !important;
            line-height: 1.18 !important;
        }}

        .decision-hero {{
            grid-template-columns: 1fr;
            padding: 1rem 1rem 0.9rem;
        }}

        .decision-hero__title {{
            font-size: 1.48rem;
        }}

        .decision-hero__stats,
        .status-card-grid,
        .summary-kpi-grid {{
            grid-template-columns: 1fr;
        }}

        .top-bar-summary {{
            grid-template-columns: 1fr;
            font-size: 0.92rem;
            padding: 0.82rem 0.92rem;
        }}

        .overview-command-surface__header,
        .command-bar--compact {{
            grid-template-columns: 1fr;
        }}

        .overview-reference-shell {{
            margin-bottom: 0.28rem;
        }}

        .overview-command-surface {{
            margin-bottom: 0.28rem;
        }}

        .overview-command-surface__header {{
            gap: 0.38rem;
            margin-bottom: 0.36rem;
        }}

        .overview-command-surface__copy {{
            display: none;
        }}

        .overview-heatmap-grid {{
            grid-template-columns: 1fr;
        }}

        .overview-market-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}

        .overview-market-card--status {{
            display: none;
        }}

        .overview-market-card {{
            min-height: 2.9rem;
            padding: 0.48rem 0.54rem;
        }}

        .overview-market-card__metric-row {{
            flex-direction: column;
            align-items: flex-start;
            gap: 0.12rem;
        }}

        .overview-market-card__value {{
            font-size: 0.9rem;
        }}

        .overview-review-candidates {{
            margin-top: 0.56rem;
            padding-top: 0.58rem;
        }}

        .overview-review-candidates__header {{
            flex-direction: column;
            gap: 0.42rem;
            margin-bottom: 0.48rem;
        }}

        .overview-review-candidates__basis {{
            width: fit-content;
        }}

        .overview-review-candidates__grid {{
            grid-template-columns: 1fr;
            gap: 0.42rem;
        }}

        .overview-review-card {{
            padding: 0.58rem 0.62rem;
        }}

        .overview-review-card__invalidation {{
            display: none;
        }}

        .overview-sector-table-wrap {{
            max-height: 520px;
        }}

        .overview-sector-table th,
        .overview-sector-table td {{
            padding: 0.5rem 0.44rem;
        }}

        .research-page-frame {{
            grid-template-columns: 1fr;
            padding: 0.82rem;
        }}

        .research-page-frame__summary {{
            grid-template-columns: 1fr;
        }}

        .analysis-toolbar__summary {{
            grid-template-columns: 1fr;
        }}

        .panel-header {{
            flex-direction: column;
            gap: 0.7rem;
        }}

        [data-testid="stTabs"] [data-baseweb="tab"] {{
            padding: 0.46rem 0.72rem;
            font-size: 0.94rem;
        }}
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def get_plotly_template(theme_mode: str) -> dict:
    """Return custom Plotly layout template for the requested theme mode."""
    mode = normalize_theme_mode(theme_mode)
    tokens = get_theme_tokens(mode)
    chart_tokens = get_chart_section_tokens(mode)
    typography_tokens = get_typography_section_tokens(mode)

    grid_color = str(chart_tokens["grid"])
    axis_color = str(chart_tokens["axis"])
    axis_text_color = str(chart_tokens["muted"])
    axis_title_color = str(chart_tokens["axis_title"])
    legend_text_color = str(chart_tokens["legend_text"])
    legend_bg = str(chart_tokens["legend_bg"])
    paper_bg = str(chart_tokens["paper_bg"])
    plot_bg = str(chart_tokens["plot_bg"])

    return {
        "colorway": PLOTLY_COLORWAY_BY_THEME[mode],
        "paper_bgcolor": paper_bg,
        "plot_bgcolor": plot_bg,
        "font": {
            "color": tokens["text"],
            "family": str(typography_tokens["ui_family"]).replace("'", "").replace('"', ""),
        },
        "xaxis": {
            "gridcolor": grid_color,
            "linecolor": axis_color,
            "zerolinecolor": axis_color,
            "tickfont": {"color": axis_text_color, "size": 12},
            "title": {"font": {"color": axis_title_color, "size": 13}},
            "automargin": True,
        },
        "yaxis": {
            "gridcolor": grid_color,
            "linecolor": axis_color,
            "zerolinecolor": axis_color,
            "tickfont": {"color": axis_text_color, "size": 12},
            "title": {"font": {"color": axis_title_color, "size": 13}},
            "automargin": True,
        },
        "legend": {
            "bgcolor": legend_bg,
            "bordercolor": tokens["border"],
            "borderwidth": 1,
            "font": {"color": legend_text_color},
        },
        "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
    }
