"""
Reusable Streamlit UI components.

Components:
- render_macro_tile: Current regime metrics
- render_rs_scatter: 4-quadrant RS vs trend scatter
- render_returns_heatmap: Sector × period returns heatmap
- render_signal_table: Full signal table with action/regime filter
"""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui.styles import ACTION_COLORS, GREY, get_plotly_template


def render_slider_with_input(
    label: str,
    min_value: int | float,
    max_value: int | float,
    value: int | float,
    step: int | float,
    key: str,
    help: str | None = None,
) -> int | float:
    """Render a slider and a number input synchronized together.
    
    Args:
        label: Label for the slider.
        min_value: Minimum value.
        max_value: Maximum value.
        value: Initial value.
        step: Step size.
        key: Base key for session state.
        help: Help text for the slider.

    Returns:
        The current synchronized value.
    """
    val_key = f"{key}_val"
    slider_key = f"{key}_slider"
    input_key = f"{key}_input"

    if val_key not in st.session_state:
        st.session_state[val_key] = value

    def sync_from_slider():
        st.session_state[val_key] = st.session_state[slider_key]

    def sync_from_input():
        st.session_state[val_key] = st.session_state[input_key]

    st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=st.session_state[val_key],
        step=step,
        help=help,
        key=slider_key,
        on_change=sync_from_slider,
    )
    
    st.number_input(
        f"{label} 직접 입력",
        min_value=min_value,
        max_value=max_value,
        value=st.session_state[val_key],
        step=step,
        key=input_key,
        on_change=sync_from_input,
        label_visibility="collapsed",
    )
    
    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
    
    return st.session_state[val_key]


def render_macro_tile(
    regime: str,
    growth_val: float | None = None,
    inflation_val: float | None = None,
    fx_change: float | None = None,
    is_provisional: bool = False,
) -> None:
    """Render current macro regime summary tile.

    Args:
        regime: Current regime name (Recovery/Expansion/Slowdown/Contraction/Indeterminate).
        growth_val: Latest growth indicator value (e.g. 경기선행지수).
        inflation_val: Latest inflation value (e.g. CPI YoY %).
        fx_change: Recent USD/KRW change % (positive = KRW weakening).
        is_provisional: Show provisional data warning badge.
    """
    regime_colors = {
        "Recovery": "#34d399",      # emerald-400
        "Expansion": "#60a5fa",     # blue-400
        "Slowdown": "#fbbf24",      # amber-400
        "Contraction": "#fb7185",   # rose-400
        "Indeterminate": GREY,
    }
    color = regime_colors.get(regime, GREY)

    regime_labels_kr = {
        "Recovery": "회복기",
        "Expansion": "확장기",
        "Slowdown": "둔화기",
        "Contraction": "수축기",
        "Indeterminate": "미분류",
    }
    kr_label = regime_labels_kr.get(regime, regime)

    provisional_html = '<span class="provisional-badge">잠정치</span>' if is_provisional else ""
    
    st.markdown(
        f'<div style="background-color:#18181b;border:1px solid #27272a;border-left:4px solid {color};border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 1px 3px 0 rgb(0 0 0 / 0.1);">'
        f'<div style="display:flex;align-items:center;">'
        f'<h2 style="margin:0;font-size:1.5rem;font-weight:700;color:{color};">{regime} <span style="font-size:1rem;font-weight:500;color:#a1a1aa;margin-left:8px;">({kr_label})</span></h2>'
        f'{provisional_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if growth_val is not None and not math.isnan(growth_val):
            st.metric("경기선행지수", f"{growth_val:.2f}p")
        else:
            st.metric("경기선행지수", "N/A")
    with col2:
        if inflation_val is not None and not math.isnan(inflation_val):
            st.metric("CPI YoY", f"{inflation_val:.1f}%")
        else:
            st.metric("CPI YoY", "N/A")
    with col3:
        if fx_change is not None and not math.isnan(fx_change):
            delta_color = "inverse" if fx_change > 0 else "normal"
            st.metric("USD/KRW 변동", f"{fx_change:+.1f}%", delta_color=delta_color)
        else:
            st.metric("USD/KRW 변동", "N/A")


def render_rs_scatter(
    signals: list,
    *,
    height: int = 680,
    margin: dict[str, int] | None = None,
) -> go.Figure:
    """Render 4-quadrant Relative Strength vs Trend scatter plot.

    Quadrants:
    - Top-right: RS Strong + Trend OK (momentum_strong=True)
    - Top-left: RS Strong + Trend Weak
    - Bottom-right: RS Weak + Trend OK
    - Bottom-left: RS Weak + Trend Weak

    Args:
        signals: list[SectorSignal].
        height: Chart height in pixels.
        margin: Optional Plotly layout margin override.

    Returns:
        Plotly Figure.
    """
    template = get_plotly_template()

    x_vals, y_vals, texts, colors, hovers = [], [], [], [], []

    for s in signals:
        if s.action == "N/A":
            continue
        if math.isnan(s.rs) or math.isnan(s.rs_ma):
            continue
        x_vals.append(s.rs)
        y_vals.append(s.rs_ma)
        texts.append(s.sector_name.split(" ")[-1])  # short label
        colors.append(ACTION_COLORS.get(s.action, GREY))
        hover = (
            f"<b>{s.sector_name}</b><br>"
            f"Action: {s.action}<br>"
            f"RS: {s.rs:.4f}<br>"
            f"RS MA: {s.rs_ma:.4f}<br>"
            f"RSI(D): {s.rsi_d:.1f}<br>"
            f"Trend: {'OK' if s.trend_ok else 'Weak'}<br>"
            f"Alerts: {', '.join(s.alerts) or 'None'}"
        )
        hovers.append(hover)

    fig = go.Figure()
    axis_range = None
    if x_vals:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                text=texts,
                textposition="top center",
                marker=dict(color=colors, size=12, line=dict(width=1, color="#18181b")),
                hovertext=hovers,
                hoverinfo="text",
            )
        )

        # Diagonal reference line (RS = RS_MA)
        mn_raw = min(min(x_vals), min(y_vals))
        mx_raw = max(max(x_vals), max(y_vals))
        span = max(mx_raw - mn_raw, 1e-6)
        pad = span * 0.06
        mn = mn_raw - pad
        mx = mx_raw + pad
        axis_range = [mn, mx]
        fig.add_shape(
            type="line",
            x0=mn, y0=mn, x1=mx, y1=mx,
            line=dict(color="#52525b", dash="dot", width=1.5),
            layer="below",  # draw behind sector dots
        )
    else:
        fig.add_annotation(
            text="표시 가능한 RS/RS MA 데이터가 없습니다. 벤치마크 누락 또는 데이터 부족을 확인하세요.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": "#a1a1aa"},
        )

    fig.update_layout(
        **template,
        title="Relative Strength vs RS 이동평균",
        xaxis_title="RS",
        yaxis_title="RS MA",
        height=height,
        showlegend=False,
    )
    fig.update_layout(margin=margin or dict(l=72, r=32, t=64, b=64))
    if axis_range:
        fig.update_xaxes(range=axis_range, constrain="domain")
        fig.update_yaxes(
            range=axis_range,
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        )

    return fig


def render_rs_momentum_bar(signals: list) -> go.Figure:
    """Render horizontal bar chart of RS divergence from its moving average.

    RS divergence = (RS - RS_MA) / RS_MA * 100 (%)
    Positive = RS above MA (momentum accelerating), negative = below (decelerating).
    Always computable from snapshot values — no historical series required.

    Args:
        signals: list[SectorSignal].

    Returns:
        Plotly Figure (empty figure if no valid data).
    """
    template = get_plotly_template()

    filtered = [
        s for s in signals
        if s.action != "N/A"
        and not math.isnan(s.rs)
        and not math.isnan(s.rs_ma)
        and s.rs_ma != 0
    ]
    if not filtered:
        return go.Figure()

    def rs_div(s) -> float:
        return (s.rs - s.rs_ma) / s.rs_ma * 100

    filtered_sorted = sorted(filtered, key=rs_div)
    names = [s.sector_name.split(" ")[-1] for s in filtered_sorted]
    values = [rs_div(s) for s in filtered_sorted]
    colors = [ACTION_COLORS.get(s.action, GREY) for s in filtered_sorted]
    hovers = [
        f"<b>{s.sector_name}</b><br>"
        f"RS 이탈도: {rs_div(s):+.2f}%<br>"
        f"RS: {s.rs:.4f} / RS MA: {s.rs_ma:.4f}<br>"
        f"Action: {s.action}"
        for s in filtered_sorted
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        hovertext=hovers,
        hoverinfo="text",
        text=[f"{v:+.2f}%" for v in values],
        textposition="outside",
    ))

    fig.add_vline(x=0, line=dict(color="#52525b", width=1.5))

    fig.update_layout(
        **template,
        title="RS 이탈도 — RS ÷ RS이동평균 − 1 (%)",
        xaxis_title="RS 이탈도 (%)",
        yaxis_title="",
        height=max(300, len(filtered_sorted) * 36 + 80),
        showlegend=False,
    )
    fig.update_xaxes(ticksuffix="%")

    return fig


def render_returns_heatmap(signals: list) -> go.Figure:
    """Render sector × period returns heatmap.

    Args:
        signals: list[SectorSignal].

    Returns:
        Plotly Figure.
    """
    template = get_plotly_template()

    periods = ["1W", "1M", "3M", "6M", "12M"]
    sector_names = []
    z_values = []

    for s in signals:
        if s.action == "N/A" or not s.returns:
            continue
        sector_names.append(s.sector_name.split()[-1])
        row = [s.returns.get(p, float("nan")) for p in periods]
        z_values.append([r * 100 if not math.isnan(r) else None for r in row])

    if not sector_names:
        fig = go.Figure()
        fig.update_layout(**template, title="데이터 없음")
        return fig

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=periods,
            y=sector_names,
            colorscale=[
                [0.0, "#f43f5e"],   # rose-500
                [0.5, "#3f3f46"],   # zinc-700
                [1.0, "#10b981"],   # emerald-500
            ],
            zmid=0,
            texttemplate="%{z:.1f}",
            textfont={"size": 11},
            hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        **template,
        title="구간별 수익률 히트맵 (%)",
        height=max(300, len(sector_names) * 40),
    )

    return fig


def render_signal_table(
    signals: list,
    filter_action: str | None = None,
    filter_regime_only: bool = False,
    current_regime: str | None = None,
) -> None:
    """Render signal table with action and regime filters.

    N/A rows are rendered with grey styling and 데이터 없음 hint.

    Args:
        signals: list[SectorSignal].
        filter_action: If set, show only signals with this action value.
        filter_regime_only: If True, show only sectors matching current_regime.
        current_regime: Current macro regime name (for filter_regime_only).
    """
    if not signals:
        st.info("신호 데이터 없음")
        return

    filtered = signals

    if filter_regime_only and current_regime:
        filtered = [s for s in filtered if s.macro_regime == current_regime]

    if filter_action and filter_action != "전체":
        filtered = [s for s in filtered if s.action == filter_action]

    if not filtered:
        st.info("필터 조건에 맞는 종목 없음")
        return

    rows = []
    for s in filtered:
        alerts_str = ", ".join(s.alerts) if s.alerts else "-"
        ret_1m = f"{s.returns.get('1M', float('nan')) * 100:.1f}%" if s.returns.get("1M") and not math.isnan(s.returns.get("1M", float("nan"))) else "N/A"
        ret_3m = f"{s.returns.get('3M', float('nan')) * 100:.1f}%" if s.returns.get("3M") and not math.isnan(s.returns.get("3M", float("nan"))) else "N/A"
        rsi_str = f"{s.rsi_d:.1f}" if not math.isnan(s.rsi_d) else "N/A"
        vol_str = f"{s.volatility_20d * 100:.1f}%" if not math.isnan(s.volatility_20d) else "N/A"
        mdd_str = f"{s.mdd_3m * 100:.1f}%" if not math.isnan(s.mdd_3m) else "N/A"
        provisional_marker = " *" if s.is_provisional else ""

        rows.append(
            {
                "섹터": s.sector_name + provisional_marker,
                "매크로": s.macro_regime,
                "적합": "✓" if s.macro_fit else "✗",
                "액션": s.action,
                "RSI(D)": rsi_str,
                "1M": ret_1m,
                "3M": ret_3m,
                "변동성": vol_str,
                "MDD(3M)": mdd_str,
                "알림": alerts_str,
                "데이터": "없음" if s.action == "N/A" else "정상",
            }
        )

    df_display = pd.DataFrame(rows)

    # Color rows by action
    def _highlight(row: pd.Series) -> list[str]:
        action = row["액션"]
        if action == "Strong Buy":
            return ["background-color: rgba(16, 185, 129, 0.15)"] * len(row)
        if action == "Watch":
            return ["background-color: rgba(59, 130, 246, 0.15)"] * len(row)
        if action == "Hold":
            return ["background-color: rgba(82, 82, 91, 0.20)"] * len(row)
        if action == "Avoid":
            return ["background-color: rgba(244, 63, 94, 0.15)"] * len(row)
        if action == "N/A":
            return ["color: #a1a1aa"] * len(row)
        return [""] * len(row)

    styled = df_display.style.apply(_highlight, axis=1)
    
    # Calculate exact height to prevent vertical scrolling
    # Header is ~38px, each row is ~35px. Adding small padding.
    table_height = 38 + (len(df_display) * 35) + 5
    
    st.dataframe(styled, use_container_width=True, hide_index=True, height=table_height)

    if any(s.is_provisional for s in filtered):
        st.caption("* 잠정치 포함 (최근 3개월 KOSIS 데이터)")


def _action_badge(action: str) -> str:
    """Return HTML badge for action value."""
    css_class = {
        "Strong Buy": "action-strong-buy",
        "Watch": "action-watch",
        "Hold": "action-hold",
        "Avoid": "action-avoid",
        "N/A": "action-na",
    }.get(action, "action-na")
    return f'<span class="{css_class}">{action}</span>'
