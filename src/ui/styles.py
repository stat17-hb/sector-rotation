"""Dashboard CSS injection, theme tokens, and Plotly templates."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import streamlit as st

from config.theme import (
    THEME_TOKENS as CANONICAL_THEME_TOKENS,
    get_chart_tokens as get_chart_section_tokens,
    get_dataframe_tokens as get_dataframe_section_tokens,
    get_navigation_tokens as get_navigation_section_tokens,
    get_signal_tokens as get_signal_section_tokens,
    get_ui_tokens as get_ui_section_tokens,
    normalize_theme_mode as normalize_theme_mode_base,
)


def _flatten_ui_tokens(theme_mode: str) -> dict[str, str]:
    ui = get_ui_section_tokens(theme_mode)
    return {
        "bg": str(ui["background"]),
        "surface": str(ui["card"]),
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


PRIMARY_FONT_CSS = (
    "'Pretendard Local', 'Pretendard', 'Noto Sans KR', 'Segoe UI', "
    "'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif"
)
PRIMARY_FONT_PLOTLY = (
    "Pretendard Local, Pretendard, Noto Sans KR, Segoe UI, "
    "Apple SD Gothic Neo, Malgun Gothic, sans-serif"
)
MONO_FONT_CSS = (
    "'JetBrains Mono Local', 'JetBrains Mono', 'Fira Code', "
    "'SFMono-Regular', Consolas, 'Liberation Mono', monospace"
)


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
    signal_tokens = get_signal_section_tokens(mode)
    tab_tokens = get_tab_style_tokens(mode)
    table_tokens = get_table_style_tokens(mode)
    badge_styles = get_action_badge_styles(mode)
    font_face_css = build_font_face_css()
    color_scheme = "dark" if mode == "dark" else "light"

    if mode == "dark":
        app_background_css = (
            "background:"
            "radial-gradient(1200px 480px at -8% -10%, color-mix(in srgb, var(--primary) 22%, transparent), transparent 72%),"
            "linear-gradient(180deg, color-mix(in srgb, var(--bg) 92%, #000 8%), var(--bg));"
        )
        header_background_css = str(ui_tokens["header_bg"])
        header_border_css = str(ui_tokens["header_border"])
        sidebar_background_css = (
            "linear-gradient(165deg, "
            f"{ui_tokens['sidebar_gradient_start']}, {ui_tokens['sidebar_gradient_end']})"
        )
        card_background_css = "color-mix(in srgb, var(--surface) 92%, var(--bg) 8%)"
        card_shadow_css = str(ui_tokens["card_shadow"])
        control_bg_css = str(ui_tokens["input_bg"])
        control_border_css = str(ui_tokens["input_border"])
        control_hover_css = str(ui_tokens["sidebar_hover"])
        inline_code_bg_css = str(ui_tokens["inline_code_bg"])
        font_smoothing_css = "-webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;"
        body_font_weight = "400"
        heading_font_weight = "600"
        badge_font_size = "12px"
        badge_font_weight = "600"
        provisional_badge_text = str(ui_tokens["provisional_badge_text"])
        provisional_badge_border = "color-mix(in srgb, var(--warning) 45%, transparent)"
    else:
        app_background_css = (
            "background:"
            "radial-gradient(1250px 520px at -10% -14%, color-mix(in srgb, var(--primary) 12%, white 88%), transparent 70%),"
            f"linear-gradient(180deg, {ui_tokens['sidebar_gradient_start']} 0%, {ui_tokens['background']} 100%);"
        )
        header_background_css = str(ui_tokens["header_bg"])
        header_border_css = str(ui_tokens["header_border"])
        sidebar_background_css = (
            "linear-gradient(180deg, "
            f"{ui_tokens['sidebar_gradient_start']} 0%, {ui_tokens['sidebar_gradient_end']} 100%)"
        )
        card_background_css = str(ui_tokens["card"])
        card_shadow_css = str(ui_tokens["card_shadow"])
        control_bg_css = str(ui_tokens["input_bg"])
        control_border_css = str(ui_tokens["input_border"])
        control_hover_css = str(ui_tokens["sidebar_hover"])
        inline_code_bg_css = str(ui_tokens["inline_code_bg"])
        font_smoothing_css = "-webkit-font-smoothing: subpixel-antialiased; -moz-osx-font-smoothing: auto;"
        body_font_weight = "450"
        heading_font_weight = "650"
        badge_font_size = "12.5px"
        badge_font_weight = "700"
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
        --accent-surface: color-mix(in srgb, var(--surface) 82%, var(--primary) 18%);
        --ring: {ui_tokens['focus_ring']};
        --radius-sm: 10px;
        --radius-md: 14px;
        --radius-lg: 18px;
        --radius-xl: 24px;
        --space-1: 0.25rem;
        --space-2: 0.5rem;
        --space-3: 0.75rem;
        --space-4: 1rem;
        --space-5: 1.25rem;
        --space-6: 1.5rem;

        /* Streamlit theme variable aliases for native widgets */
        --primary-color: {tokens['primary']};
        --background-color: {tokens['bg']};
        --secondary-background-color: {tokens['surface']};
        --text-color: {tokens['text']};
    }}

    html, body, [class*="css"] {{
        font-family: {PRIMARY_FONT_CSS} !important;
        color: var(--text);
        font-weight: {body_font_weight};
        line-height: 1.65;
        letter-spacing: 0.012em;
        font-optical-sizing: auto;
        text-rendering: optimizeLegibility;
        {font_smoothing_css}
        color-scheme: {color_scheme};
    }}

    .stApp {{
        {app_background_css}
        color: var(--text);
    }}

    [data-testid="stHeader"] {{
        background: {header_background_css} !important;
        border-bottom: 1px solid {header_border_css};
    }}

    [data-testid="stDecoration"] {{
        background: {header_background_css} !important;
    }}

    [data-testid="stToolbar"] {{
        background: transparent !important;
    }}

    [data-testid="stSidebar"] {{
        background: {sidebar_background_css};
        border-right: 1px solid var(--border);
    }}

    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
    }}

    div[data-testid="stMetricValue"], code, pre {{
        font-family: {MONO_FONT_CSS} !important;
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
        line-height: 1.3;
        letter-spacing: -0.01em;
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
        border-radius: 6px;
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
    }}

    [data-testid="stSidebar"] button[kind]:hover {{
        background-color: {control_hover_css};
        border-color: {tokens['primary']};
    }}

    .app-summary-card {{
        border: 1px solid var(--border);
        background: {card_background_css};
        box-shadow: {card_shadow_css};
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 0.8rem;
    }}

    .page-shell {{
        position: relative;
        overflow: hidden;
        border: 1px solid color-mix(in srgb, var(--primary) 18%, var(--border));
        background:
            radial-gradient(960px 320px at 110% -20%, color-mix(in srgb, var(--primary) 18%, transparent), transparent 58%),
            linear-gradient(180deg, color-mix(in srgb, var(--surface) 94%, var(--bg) 6%), var(--surface));
        box-shadow: {card_shadow_css};
        border-radius: var(--radius-xl);
        padding: 1.25rem 1.35rem 1.1rem;
        margin-bottom: 1rem;
    }}

    .page-shell__eyebrow {{
        color: var(--text-muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }}

    .page-shell__title {{
        color: var(--text);
        font-size: clamp(1.7rem, 2.8vw, 2.35rem);
        font-weight: 750;
        line-height: 1.08;
        letter-spacing: -0.03em;
    }}

    .page-shell__description {{
        color: var(--text-muted);
        font-size: 0.98rem;
        max-width: 72ch;
        margin-top: 0.45rem;
    }}

    .page-shell__pills {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1rem;
    }}

    .page-shell__pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.45rem 0.72rem;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--surface) 88%, var(--bg) 12%);
        font-size: 0.84rem;
        color: var(--text-muted);
    }}

    .page-shell__pill strong {{
        color: var(--text);
        font-size: 0.86rem;
        font-weight: 700;
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
        gap: 0.9rem;
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--surface) 94%, var(--bg) 6%);
        box-shadow: {card_shadow_css};
        border-radius: var(--radius-lg);
        padding: 0.85rem 1rem;
        margin-bottom: 0.9rem;
    }}

    .status-strip__badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 72px;
        border-radius: 999px;
        padding: 0.36rem 0.6rem;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.06em;
    }}

    .status-strip__title {{
        color: var(--text);
        font-size: 0.95rem;
        font-weight: 700;
        line-height: 1.25;
    }}

    .status-strip__message {{
        color: var(--text-muted);
        font-size: 0.86rem;
        margin-top: 0.12rem;
    }}

    .status-strip__meta {{
        color: var(--text-muted);
        font-size: 0.78rem;
        font-weight: 600;
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

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: var(--border) !important;
        background: {card_background_css};
        box-shadow: {card_shadow_css};
        border-radius: var(--radius-lg) !important;
    }}

    .command-bar {{
        margin-bottom: 0.85rem;
    }}

    .command-bar__eyebrow {{
        color: var(--text-muted);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.18rem;
    }}

    .command-bar__title {{
        color: var(--text);
        font-size: 1.02rem;
        font-weight: 650;
        line-height: 1.35;
    }}

    .top-bar-summary {{
        border: 1px solid color-mix(in srgb, var(--primary) 18%, var(--border));
        background: color-mix(in srgb, var(--surface) 90%, var(--primary) 10%);
        border-radius: var(--radius-md);
        color: var(--text);
        font-size: 0.92rem;
        line-height: 1.5;
        padding: 0.72rem 0.9rem;
        min-height: 3rem;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.65rem;
    }}

    .top-bar-summary__item {{
        display: flex;
        flex-direction: column;
        gap: 0.16rem;
    }}

    .top-bar-summary__item span {{
        color: var(--text-muted);
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}

    .top-bar-summary__item strong {{
        color: var(--text);
        font-size: 0.93rem;
        font-weight: 700;
    }}

    .analysis-toolbar {{
        border: 1px solid color-mix(in srgb, var(--primary) 18%, var(--border));
        background:
            linear-gradient(180deg, color-mix(in srgb, var(--surface) 96%, var(--bg) 4%), var(--surface));
        box-shadow: {card_shadow_css};
        border-radius: var(--radius-lg);
        padding: 1rem 1.05rem 0.95rem;
        margin-bottom: 0.85rem;
    }}

    .analysis-toolbar__eyebrow {{
        color: var(--text-muted);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.16rem;
    }}

    .analysis-toolbar__title {{
        color: var(--text);
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.35;
    }}

    .analysis-toolbar__summary {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.7rem;
        margin-top: 0.85rem;
    }}

    .analysis-toolbar__summary-item {{
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--surface) 90%, var(--primary) 10%);
        border-radius: var(--radius-md);
        padding: 0.7rem 0.82rem;
        min-height: 3.1rem;
    }}

    .analysis-toolbar__summary-item span {{
        display: block;
        color: var(--text-muted);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.22rem;
    }}

    .analysis-toolbar__summary-item strong {{
        color: var(--text);
        font-size: 0.92rem;
        font-weight: 700;
        line-height: 1.35;
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
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}

    .cycle-palette__item {{
        display: inline-flex;
        align-items: center;
        gap: 0.42rem;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: color-mix(in srgb, var(--surface) 92%, var(--bg) 8%);
        color: var(--text);
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.32rem 0.6rem;
    }}

    .cycle-palette__swatch {{
        width: 10px;
        height: 10px;
        border-radius: 999px;
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
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.16rem;
    }}

    .sector-rank-list__title {{
        color: var(--text);
        font-size: 0.94rem;
        font-weight: 650;
        line-height: 1.35;
    }}

    .sector-rank-list__metric {{
        border-radius: 999px;
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--surface) 94%, var(--bg) 6%);
        color: var(--text-muted);
        font-size: 0.78rem;
        font-weight: 700;
        line-height: 1;
        padding: 0.58rem 0.6rem;
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
        border: 1px solid var(--border);
        background:
            linear-gradient(135deg, color-mix(in srgb, var(--surface) 90%, var(--bg) 10%), var(--surface));
        box-shadow: {card_shadow_css};
        border-radius: var(--radius-lg);
        padding: 1.2rem 1.2rem 1rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }}

    .decision-hero::before {{
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 6px;
        background: var(--decision-hero-accent, var(--primary));
    }}

    .decision-hero__eyebrow {{
        color: var(--text-muted);
        font-size: 0.8rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }}

    .decision-hero__copy {{
        margin-bottom: 1rem;
    }}

    .decision-hero__title {{
        color: var(--text);
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }}

    .decision-hero__subtitle {{
        color: var(--text-muted);
        font-size: 0.96rem;
        font-weight: 550;
        margin-top: 0.4rem;
        letter-spacing: 0.01em;
        max-width: 56ch;
    }}

    .decision-hero__chips {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.85rem;
    }}

    .decision-hero__chip {{
        border: 1px solid color-mix(in srgb, var(--primary) 32%, var(--border));
        background: color-mix(in srgb, var(--surface) 86%, var(--primary) 14%);
        color: var(--text);
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        line-height: 1;
        padding: 0.45rem 0.7rem;
        white-space: nowrap;
    }}

    .decision-hero__badge {{
        border: 1px solid {provisional_badge_border};
        background: color-mix(in srgb, var(--warning) 18%, transparent);
        color: {provisional_badge_text};
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        line-height: 1;
        padding: 0.42rem 0.7rem;
        white-space: nowrap;
    }}

    .decision-hero__stats,
    .status-card-grid,
    .summary-kpi-grid {{
        display: grid;
        gap: 0.75rem;
    }}

    .decision-hero__stats {{
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }}

    .status-card-grid {{
        grid-template-columns: repeat(4, minmax(0, 1fr));
        margin-bottom: 1rem;
    }}

    .summary-kpi-grid {{
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        margin-bottom: 0.9rem;
    }}

    .decision-hero__stat,
    .status-card {{
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--surface) 94%, transparent);
        border-radius: var(--radius-md);
        padding: 0.85rem 0.95rem;
    }}

    .decision-hero__stat-label,
    .status-card__eyebrow {{
        display: block;
        color: var(--text-muted);
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }}

    .decision-hero__stat strong,
    .status-card__value {{
        display: block;
        color: var(--text);
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1.2;
    }}

    .status-card__detail {{
        color: var(--text-muted);
        font-size: 0.84rem;
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

    .panel-header {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.95rem;
    }}

    .panel-header__eyebrow {{
        color: var(--text-muted);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.18rem;
    }}

    .panel-header__title {{
        color: var(--text);
        font-size: 1.08rem;
        font-weight: 700;
        line-height: 1.25;
    }}

    .panel-header__description {{
        color: var(--text-muted);
        font-size: 0.89rem;
        line-height: 1.5;
        margin-top: 0.18rem;
        max-width: 72ch;
    }}

    .panel-header__badge {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        border: 1px solid color-mix(in srgb, var(--primary) 38%, var(--border));
        background: color-mix(in srgb, var(--surface) 86%, var(--primary) 14%);
        color: var(--text);
        font-size: 0.76rem;
        font-weight: 700;
        padding: 0.42rem 0.72rem;
        white-space: nowrap;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {table_tokens['grid']};
        border-radius: 12px;
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
        --gdg-header-font-style: 700 12px {PRIMARY_FONT_CSS};
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
        border-radius: 9999px;
        padding: 2px 8px;
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.02em;
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
        border-radius: 9999px;
        padding: 2px 10px;
        font-weight: {badge_font_weight};
        font-size: {badge_font_size};
        letter-spacing: 0.02em;
        display: inline-block;
        text-align: center;
        min-width: 104px;
    }}

    [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        gap: 0.35rem;
    }}

    [data-testid="stTabs"] [data-baseweb="tab"] {{
        font-family: {PRIMARY_FONT_CSS} !important;
        color: var(--text-muted) !important;
        border-radius: 10px 10px 0 0;
        padding: 0.45rem 0.78rem;
        font-weight: 600 !important;
        border: 1px solid transparent;
        transition: color 0.16s ease, background-color 0.16s ease;
    }}

    [data-testid="stTabs"] [data-baseweb="tab"] p {{
        color: inherit !important;
    }}

    [data-testid="stTabs"] [data-baseweb="tab"]:hover {{
        color: var(--text) !important;
        background-color: {tab_tokens['tab_hover_bg']};
    }}

    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {{
        color: {tab_tokens['tab_selected_text']} !important;
        font-weight: 700 !important;
        background-color: {tab_tokens['tab_selected_bg']};
        border-color: {tab_tokens['tab_selected_border']} !important;
    }}

    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] p {{
        color: {tab_tokens['tab_selected_text']} !important;
    }}

    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
        background-color: {tab_tokens['tab_selected_border']} !important;
        height: 2px;
    }}

    div[data-testid="stMetricValue"] {{
        font-size: 1.6rem;
        font-weight: {heading_font_weight} !important;
    }}

    .stApp [data-testid="stMetricLabel"] p,
    .stApp [data-testid="stCaptionContainer"] p {{
        color: var(--text-muted) !important;
        font-size: 13px;
        letter-spacing: 0.01em;
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

    @media (max-width: 840px) {{
        .page-shell {{
            padding: 1rem 1rem 0.95rem;
        }}

        .page-shell__pills {{
            gap: 0.5rem;
        }}

        .status-strip {{
            grid-template-columns: 1fr;
            gap: 0.55rem;
        }}

        .decision-hero {{
            padding: 1rem 1rem 0.9rem;
        }}

        .decision-hero__title {{
            font-size: 1.45rem;
        }}

        .decision-hero__stats,
        .status-card-grid,
        .summary-kpi-grid {{
            grid-template-columns: 1fr;
        }}

        .top-bar-summary {{
            grid-template-columns: 1fr;
            font-size: 0.86rem;
            padding: 0.68rem 0.82rem;
        }}

        .analysis-toolbar__summary {{
            grid-template-columns: 1fr;
        }}

        .panel-header {{
            flex-direction: column;
            gap: 0.7rem;
        }}

        [data-testid="stTabs"] [data-baseweb="tab"] {{
            padding: 0.4rem 0.62rem;
            font-size: 0.92rem;
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
        "font": {"color": tokens["text"], "family": PRIMARY_FONT_PLOTLY},
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
