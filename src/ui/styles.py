"""
Dashboard CSS injection and Plotly template.

Enforces Shadcn Zinc Dark aesthetic.
"""
from __future__ import annotations

import streamlit as st


# --- Palette ---
ZINC_950 = "#09090b"  # Background
ZINC_900 = "#18181b"  # Cards
ZINC_800 = "#27272a"  # Borders
ZINC_400 = "#a1a1aa"  # Muted Text
ZINC_50 = "#fafafa"   # Primary Text

# Action colors (Tailwind equivalent)
GREEN = "#10b981"   # emerald-500
RED = "#f43f5e"     # rose-500
AMBER = "#f59e0b"   # amber-500
BLUE = "#3b82f6"    # blue-500
GREY = "#52525b"    # zinc-600

# Plotly colorway
PLOTLY_COLORWAY = [
    "#3b82f6",   # blue-500
    "#10b981",   # emerald-500
    "#f59e0b",   # amber-500
    "#f43f5e",   # rose-500
    "#8b5cf6",   # violet-500
    "#ec4899",   # pink-500
    "#14b8a6",   # teal-500
    "#f97316",   # orange-500
    "#6366f1",   # indigo-500
    "#84cc16",   # lime-500
]

ACTION_COLORS: dict[str, str] = {
    "Strong Buy": GREEN,
    "Watch": AMBER,
    "Hold": BLUE,
    "Avoid": RED,
    "N/A": GREY,
}


def inject_css() -> None:
    """Inject global CSS for typography and Shadcn-inspired styling."""
    css = f"""
    <style>
    @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css");

    html, body, [class*="css"] {{
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, 'Helvetica Neue', 'Segoe UI', 'Apple SD Gothic Neo', 'Noto Sans KR', 'Malgun Gothic', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', sans-serif !important;
    }}

    .stApp {{
        background-color: {ZINC_950};
    }}

    /* Card styling similar to Shadcn */
    .metric-container {{
        background-color: {ZINC_900};
        border-radius: 12px;
        padding: 16px;
        border: 1px solid {ZINC_800};
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
    }}

    .sample-warning {{
        background-color: rgba(244, 63, 94, 0.1);
        border: 1px solid {RED};
        color: #fda4af;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
    }}

    .provisional-badge {{
        background-color: rgba(245, 158, 11, 0.2);
        color: {AMBER};
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 9999px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 500;
        margin-left: 8px;
    }}

    /* Signal table action badges - Pill style */
    .action-strong-buy {{
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        padding: 2px 10px;
        font-weight: 500;
        font-size: 13px;
        display: inline-block;
        text-align: center;
        min-width: 80px;
    }}
    .action-watch {{
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 9999px;
        padding: 2px 10px;
        font-weight: 500;
        font-size: 13px;
        display: inline-block;
        text-align: center;
        min-width: 80px;
    }}
    .action-hold {{
        background-color: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 9999px;
        padding: 2px 10px;
        font-weight: 500;
        font-size: 13px;
        display: inline-block;
        text-align: center;
        min-width: 80px;
    }}
    .action-avoid {{
        background-color: rgba(244, 63, 94, 0.2);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.3);
        border-radius: 9999px;
        padding: 2px 10px;
        font-weight: 500;
        font-size: 13px;
        display: inline-block;
        text-align: center;
        min-width: 80px;
    }}
    .action-na {{
        background-color: rgba(82, 82, 91, 0.2);
        color: #a1a1aa;
        border: 1px solid rgba(82, 82, 91, 0.3);
        border-radius: 9999px;
        padding: 2px 10px;
        font-weight: 500;
        font-size: 13px;
        display: inline-block;
        text-align: center;
        min-width: 80px;
    }}

    /* Streamlit overrides for better UI */
    div[data-testid="stMetricValue"] {{
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.9rem;
        font-weight: 500;
        color: {ZINC_400};
    }}

    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def get_plotly_template() -> dict:
    """Return custom Plotly layout template with Zinc palette.

    Returns:
        dict suitable for go.Figure(layout=get_plotly_template()).
    """
    return {
        "colorway": PLOTLY_COLORWAY,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#fafafa", "family": "Pretendard, -apple-system, sans-serif"},
        "xaxis": {
            "gridcolor": "#27272a",
            "linecolor": "#3f3f46",
            "zerolinecolor": "#3f3f46",
            "tickfont": {"color": "#a1a1aa"},
            "title": {"font": {"color": "#a1a1aa", "size": 13}},
        },
        "yaxis": {
            "gridcolor": "#27272a",
            "linecolor": "#3f3f46",
            "zerolinecolor": "#3f3f46",
            "tickfont": {"color": "#a1a1aa"},
            "title": {"font": {"color": "#a1a1aa", "size": 13}},
        },
        "legend": {
            "bgcolor": "rgba(24, 24, 27, 0.8)",
            "bordercolor": "#27272a",
            "borderwidth": 1,
            "font": {"color": "#a1a1aa"},
        },
        "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
    }
