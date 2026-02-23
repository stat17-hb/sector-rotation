"""
Dashboard CSS injection and Plotly template.

Enforces no-cyan color palette throughout.
Excluded colors: #00FFFF, #17BECF, #00BCD4 and all teal/cyan variants.
"""
from __future__ import annotations

import streamlit as st


# --- Palette ---
NAVY = "#1B2A4A"
CHARCOAL = "#1E1E2E"
SLATE = "#2A2A3E"
SILVER = "#E0E0E0"
GOLD = "#D4AF37"
GREEN = "#2ECC71"
RED = "#E74C3C"
AMBER = "#F39C12"
BLUE = "#3498DB"
PURPLE = "#9B59B6"
GREY = "#7F8C8D"

# Plotly colorway — no cyan/teal
PLOTLY_COLORWAY = [
    "#3498DB",   # blue
    "#E74C3C",   # red
    "#2ECC71",   # green
    "#F39C12",   # amber
    "#9B59B6",   # purple
    "#1B2A4A",   # navy
    "#D4AF37",   # gold
    "#E67E22",   # orange
    "#1ABC9C",   # emerald (green-based, not cyan)
    "#34495E",   # dark slate
]

ACTION_COLORS: dict[str, str] = {
    "Strong Buy": GREEN,
    "Watch": AMBER,
    "Hold": BLUE,
    "Avoid": RED,
    "N/A": GREY,
}


def inject_css() -> None:
    """Inject global CSS for D2Coding font and dashboard styling."""
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'D2Coding', 'Noto Sans KR', 'Malgun Gothic', monospace !important;
    }}

    .stApp {{
        background-color: {CHARCOAL};
    }}

    .metric-container {{
        background-color: {SLATE};
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid {NAVY};
    }}

    .sample-warning {{
        background-color: #5C1A1A;
        border: 1px solid {RED};
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 16px;
    }}

    .provisional-badge {{
        background-color: {AMBER};
        color: #000;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 11px;
        font-weight: 600;
    }}

    /* Signal table action badges */
    .action-strong-buy {{
        background-color: {GREEN};
        color: #000;
        border-radius: 4px;
        padding: 2px 8px;
        font-weight: 600;
    }}
    .action-watch {{
        background-color: {AMBER};
        color: #000;
        border-radius: 4px;
        padding: 2px 8px;
        font-weight: 600;
    }}
    .action-hold {{
        background-color: {BLUE};
        color: #fff;
        border-radius: 4px;
        padding: 2px 8px;
        font-weight: 600;
    }}
    .action-avoid {{
        background-color: {RED};
        color: #fff;
        border-radius: 4px;
        padding: 2px 8px;
        font-weight: 600;
    }}
    .action-na {{
        background-color: {GREY};
        color: #fff;
        border-radius: 4px;
        padding: 2px 8px;
    }}

    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def get_plotly_template() -> dict:
    """Return custom Plotly layout template with no-cyan palette.

    Returns:
        dict suitable for go.Figure(layout=get_plotly_template()).
    """
    return {
        "colorway": PLOTLY_COLORWAY,
        "paper_bgcolor": CHARCOAL,
        "plot_bgcolor": SLATE,
        "font": {"color": SILVER, "family": "D2Coding, Noto Sans KR, monospace"},
        "xaxis": {
            "gridcolor": NAVY,
            "linecolor": GREY,
            "zerolinecolor": GREY,
        },
        "yaxis": {
            "gridcolor": NAVY,
            "linecolor": GREY,
            "zerolinecolor": GREY,
        },
        "legend": {
            "bgcolor": SLATE,
            "bordercolor": NAVY,
            "borderwidth": 1,
        },
    }
