"""Dashboard CSS injection, theme tokens, and Plotly templates."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import streamlit as st


THEME_TOKENS: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#0B1220",
        "surface": "#111C30",
        "border": "#25314A",
        "text": "#F4F7FF",
        "text_muted": "#C2CCE0",
        "primary": "#4F8CFF",
        "success": "#2FCB86",
        "warning": "#F4B55A",
        "danger": "#FF6B7A",
        "info": "#7FA6FF",
    },
    "light": {
        "bg": "#F6F8FC",
        "surface": "#FFFFFF",
        "border": "#C7D3E7",
        "text": "#0A1929",       # slightly deeper for margin on tinted surfaces
        "text_muted": "#1D3252",  # 6.5:1+ on #F6F8FC / #FFFFFF
        "primary": "#1F4EA3",
        "success": "#0E7B50",
        "warning": "#A56309",
        "danger": "#9B1C35",
        "info": "#2457C5",
    },
}

ACTION_COLORS_BY_THEME: dict[str, dict[str, str]] = {
    "dark": {
        "Strong Buy": "#34D399",
        "Watch": "#79A7FF",
        "Hold": "#9CA8C0",
        "Avoid": "#FF8B99",
        "N/A": "#8B95AA",
    },
    "light": {
        "Strong Buy": "#1B8A5A",
        "Watch": "#275EC7",
        "Hold": "#3A5070",   # 5.5:1 on #F6F8FC (was #586A86 @ 3.7:1)
        "Avoid": "#B7344F",
        "N/A": "#4A5870",    # 4.8:1 on #F6F8FC (was #6D778A @ 3.0:1)
    },
}

ACTION_BADGE_STYLES: dict[str, dict[str, dict[str, str]]] = {
    "dark": {
        "Strong Buy": {"bg": "#0F3C2D", "text": "#D8FFE9", "border": "#2FCB86"},
        "Watch": {"bg": "#132F57", "text": "#E1ECFF", "border": "#7FA6FF"},
        "Hold": {"bg": "#343C4D", "text": "#F5F7FC", "border": "#9CA8C0"},
        "Avoid": {"bg": "#501826", "text": "#FFE6EC", "border": "#FF6B7A"},
        "N/A": {"bg": "#30384A", "text": "#F2F4F8", "border": "#8B95AA"},
    },
    "light": {
        "Strong Buy": {"bg": "#DDF5E8", "text": "#124530", "border": "#1B8A5A"},
        "Watch": {"bg": "#E0EBFF", "text": "#18356B", "border": "#275EC7"},
        "Hold": {"bg": "#E9EEF7", "text": "#1E3050", "border": "#3A5070"},  # text 5.8:1 on bg
        "Avoid": {"bg": "#FDE4EA", "text": "#6B1B2C", "border": "#B7344F"},
        "N/A": {"bg": "#E6EBF3", "text": "#232E42", "border": "#4A5870"},   # text 5.5:1 on bg
    },
}

TAB_STYLE_TOKENS: dict[str, dict[str, str]] = {
    "dark": {
        "tab_hover_bg": "#1A2A45",
        "tab_selected_bg": "#356BCF",
        "tab_selected_text": "#F8FBFF",
        "tab_selected_border": "#A9C4FF",
    },
    "light": {
        "tab_hover_bg": "#DCE7FA",
        "tab_selected_bg": "#1F4EA3",
        "tab_selected_text": "#FFFFFF",
        "tab_selected_border": "#12376F",
    },
}

TABLE_STYLE_TOKENS: dict[str, dict[str, str]] = {
    "dark": {
        "header_bg": "#17233A",
        "header_text": "#E9F0FF",
        "row_bg_even": "#111C30",
        "row_bg_odd": "#16253D",
        "row_text": "#E7EDF9",
        "grid": "#2A3B58",
    },
    "light": {
        "header_bg": "#E7EEF9",
        "header_text": "#10233F",
        "row_bg_even": "#FFFFFF",
        "row_bg_odd": "#F4F8FF",
        "row_text": "#132A49",
        "grid": "#CBD7EA",
    },
}

PLOTLY_COLORWAY_BY_THEME: dict[str, list[str]] = {
    "dark": [
        "#4F8CFF",
        "#34D399",
        "#F4B55A",
        "#FF6B7A",
        "#A78BFA",
        "#F472B6",
        "#F97316",
        "#84CC16",
        "#E879F9",
        "#60A5FA",
    ],
    "light": [
        "#1F4EA3",
        "#2F6BCB",
        "#4A79C7",
        "#1B8A5A",
        "#2C9C6B",
        "#A56309",
        "#C0740A",
        "#B7344F",
        "#8A3F89",
        "#3A5070",  # was #586A86 (3.7:1) → now 5.5:1
        "#5C63B2",
    ],
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
    mode = str(theme_mode or "dark").strip().lower()
    return mode if mode in THEME_TOKENS else "dark"


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
        header_background_css = "color-mix(in srgb, var(--bg) 90%, #000 10%)"
        header_border_css = "var(--border)"
        sidebar_background_css = (
            "linear-gradient(165deg, color-mix(in srgb, var(--surface) 94%, var(--bg) 6%), var(--surface))"
        )
        card_background_css = "color-mix(in srgb, var(--surface) 92%, var(--bg) 8%)"
        card_shadow_css = "none"
        control_bg_css = "color-mix(in srgb, var(--surface) 92%, var(--bg) 8%)"
        control_border_css = "var(--border)"
        control_hover_css = "color-mix(in srgb, var(--surface) 76%, var(--primary) 24%)"
        inline_code_bg_css = "color-mix(in srgb, var(--surface) 82%, #000 18%)"
        font_smoothing_css = "-webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;"
        body_font_weight = "400"
        heading_font_weight = "600"
        badge_font_size = "12px"
        badge_font_weight = "600"
        provisional_badge_text = "var(--warning)"
        provisional_badge_border = "color-mix(in srgb, var(--warning) 45%, transparent)"
    else:
        app_background_css = (
            "background:"
            "radial-gradient(1250px 520px at -10% -14%, color-mix(in srgb, var(--primary) 12%, white 88%), transparent 70%),"
            "linear-gradient(180deg, #FCFDFF 0%, #F2F6FC 100%);"
        )
        header_background_css = "#F6FAFF"
        header_border_css = "#CFDBEC"
        sidebar_background_css = "linear-gradient(180deg, #F8FBFF 0%, #EDF3FD 100%)"
        card_background_css = "#FFFFFF"
        card_shadow_css = "0 1px 2px rgba(15, 31, 53, 0.08)"
        control_bg_css = "#FFFFFF"
        control_border_css = "#B9C9E2"
        control_hover_css = "#E5EEFC"
        inline_code_bg_css = "#E7EEF9"
        font_smoothing_css = "-webkit-font-smoothing: subpixel-antialiased; -moz-osx-font-smoothing: auto;"
        body_font_weight = "450"
        heading_font_weight = "650"
        badge_font_size = "12.5px"
        badge_font_weight = "700"
        provisional_badge_text = "#7A4A06"
        provisional_badge_border = "color-mix(in srgb, #A56309 60%, transparent)"

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

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def get_plotly_template(theme_mode: str) -> dict:
    """Return custom Plotly layout template for the requested theme mode."""
    mode = normalize_theme_mode(theme_mode)
    tokens = get_theme_tokens(mode)

    if mode == "dark":
        grid_color = "#2A3751"
        axis_color = "#36445F"
        axis_text_color = tokens["text_muted"]
        axis_title_color = tokens["text"]
        legend_text_color = tokens["text_muted"]
        legend_bg = "rgba(17, 28, 48, 0.84)"
        paper_bg = "rgba(0,0,0,0)"
        plot_bg = "rgba(0,0,0,0)"
    else:
        grid_color = "#CCD8EB"
        axis_color = "#8FA3C3"
        axis_text_color = "#1F3558"
        axis_title_color = "#152C4D"
        legend_text_color = "#1A2F50"
        legend_bg = "rgba(250, 252, 255, 0.96)"
        paper_bg = "rgba(0,0,0,0)"
        plot_bg = "rgba(0,0,0,0)"

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
