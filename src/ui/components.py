"""
Reusable Streamlit UI components.

Components:
- render_macro_tile: Current regime metrics
- render_action_summary: Action distribution KPI + bar chart
- render_rs_scatter: 4-quadrant RS vs trend scatter
- render_returns_heatmap: Sector × period returns heatmap
- render_signal_table: Full signal table with action/regime filter
"""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui.styles import (
    get_action_badge_styles,
    get_action_colors,
    get_plotly_template,
    get_table_style_tokens,
    get_theme_tokens,
)

ACTION_LABELS: dict[str, str] = {
    "Strong Buy": "▲ Strong Buy",
    "Watch": "● Watch",
    "Hold": "■ Hold",
    "Avoid": "▼ Avoid",
    "N/A": "○ N/A",
}
ACTION_BY_LABEL: dict[str, str] = {label: action for action, label in ACTION_LABELS.items()}
ACTION_CSS_CLASS: dict[str, str] = {
    "Strong Buy": "action-strong-buy",
    "Watch": "action-watch",
    "Hold": "action-hold",
    "Avoid": "action-avoid",
    "N/A": "action-na",
}


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

    def sync_from_slider() -> None:
        st.session_state[val_key] = st.session_state[slider_key]

    def sync_from_input() -> None:
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


def format_action_label(action: str) -> str:
    """Return icon+text label for action status."""
    return ACTION_LABELS.get(action, f"• {action}")


def render_macro_tile(
    regime: str,
    growth_val: float | None = None,
    inflation_val: float | None = None,
    fx_change: float | None = None,
    is_provisional: bool = False,
    theme_mode: str = "dark",
) -> None:
    """Render current macro regime summary tile.

    Args:
        regime: Current regime name (Recovery/Expansion/Slowdown/Contraction/Indeterminate).
        growth_val: Latest growth indicator value (e.g. 경기선행지수).
        inflation_val: Latest inflation value (e.g. CPI YoY %).
        fx_change: Recent USD/KRW change % (positive = KRW weakening).
        is_provisional: Show provisional data warning badge.
        theme_mode: "dark" or "light".
    """
    tokens = get_theme_tokens(theme_mode)

    regime_colors = {
        "Recovery": tokens["success"],
        "Expansion": tokens["primary"],
        "Slowdown": tokens["warning"],
        "Contraction": tokens["danger"],
        "Indeterminate": tokens["text_muted"],
    }
    color = regime_colors.get(regime, tokens["text_muted"])

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
        f'<div style="background-color:{tokens["surface"]};border:1px solid {tokens["border"]};'
        f'border-left:4px solid {color};border-radius:12px;padding:18px;margin-bottom:18px;">'
        f'<div style="display:flex;align-items:center;">'
        f'<h2 style="margin:0;font-size:1.45rem;font-weight:700;color:{color};">{regime} '
        f'<span style="font-size:0.98rem;font-weight:600;color:{tokens["text_muted"]};margin-left:8px;">({kr_label})</span></h2>'
        f"{provisional_html}</div>"
        f"</div>",
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
    theme_mode: str = "dark",
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
        theme_mode: "dark" or "light".

    Returns:
        Plotly Figure.
    """
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)
    action_colors = get_action_colors(theme_mode)

    x_vals, y_vals, texts, colors, hovers = [], [], [], [], []

    for s in signals:
        if s.action == "N/A":
            continue
        if math.isnan(s.rs) or math.isnan(s.rs_ma):
            continue
        x_vals.append(s.rs)
        y_vals.append(s.rs_ma)
        texts.append(s.sector_name.split(" ")[-1])
        colors.append(action_colors.get(s.action, tokens["text_muted"]))
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
                marker=dict(color=colors, size=12, line=dict(width=1, color=tokens["surface"])),
                hovertext=hovers,
                hoverinfo="text",
            )
        )

        mn_raw = min(min(x_vals), min(y_vals))
        mx_raw = max(max(x_vals), max(y_vals))
        span = max(mx_raw - mn_raw, 1e-6)
        pad = span * 0.06
        mn = mn_raw - pad
        mx = mx_raw + pad
        axis_range = [mn, mx]
        fig.add_shape(
            type="line",
            x0=mn,
            y0=mn,
            x1=mx,
            y1=mx,
            line=dict(color=tokens["border"], dash="dot", width=1.5),
            layer="below",
        )
    else:
        fig.add_annotation(
            text="표시 가능한 RS/RS MA 데이터가 없습니다. 벤치마크 누락 또는 데이터 부족을 확인하세요.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": tokens["text"]},
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


def render_rs_momentum_bar(signals: list, theme_mode: str = "dark") -> go.Figure:
    """Render horizontal bar chart of RS divergence from its moving average.

    RS divergence = (RS - RS_MA) / RS_MA * 100 (%).

    Args:
        signals: list[SectorSignal].
        theme_mode: "dark" or "light".

    Returns:
        Plotly Figure (empty figure if no valid data).
    """
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)
    action_colors = get_action_colors(theme_mode)

    filtered = [
        s
        for s in signals
        if s.action != "N/A" and not math.isnan(s.rs) and not math.isnan(s.rs_ma) and s.rs_ma != 0
    ]
    if not filtered:
        return go.Figure()

    def rs_div(s) -> float:
        return (s.rs - s.rs_ma) / s.rs_ma * 100

    filtered_sorted = sorted(filtered, key=rs_div)
    names = [s.sector_name.split(" ")[-1] for s in filtered_sorted]
    values = [rs_div(s) for s in filtered_sorted]
    colors = [action_colors.get(s.action, tokens["text_muted"]) for s in filtered_sorted]
    hovers = [
        f"<b>{s.sector_name}</b><br>"
        f"RS 이탈도: {rs_div(s):+.2f}%<br>"
        f"RS: {s.rs:.4f} / RS MA: {s.rs_ma:.4f}<br>"
        f"Action: {s.action}"
        for s in filtered_sorted
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            hovertext=hovers,
            hoverinfo="text",
            text=[f"{v:+.2f}%" for v in values],
            textposition="outside",
        )
    )

    fig.add_vline(x=0, line=dict(color=tokens["border"], width=1.5))

    fig.update_layout(
        **template,
        title="RS 이탈도 - RS 대비 RS 이동평균 편차 (%)",
        xaxis_title="RS 이탈도 (%)",
        yaxis_title="",
        height=max(300, len(filtered_sorted) * 36 + 80),
        showlegend=False,
    )
    fig.update_xaxes(ticksuffix="%")

    return fig


def render_returns_heatmap(signals: list, theme_mode: str = "dark") -> go.Figure:
    """Render sector × period returns heatmap.

    Args:
        signals: list[SectorSignal].
        theme_mode: "dark" or "light".

    Returns:
        Plotly Figure.
    """
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)

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
                [0.0, tokens["danger"]],
                [0.5, tokens["border"]],
                [1.0, tokens["success"]],
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


def render_action_summary(signals: list, theme_mode: str = "dark") -> None:
    """Render action-distribution KPI row and bar chart."""
    if not signals:
        st.info("신호 데이터 없음")
        return

    action_order = ["Strong Buy", "Watch", "Hold", "Avoid", "N/A"]
    action_counts = {action: 0 for action in action_order}
    for s in signals:
        action_counts[s.action] = action_counts.get(s.action, 0) + 1

    total_count = sum(action_counts.values())
    metric_cols = st.columns(len(action_order) + 1)
    with metric_cols[0]:
        st.metric("Total", total_count)
    for idx, action in enumerate(action_order, start=1):
        with metric_cols[idx]:
            st.metric(action, action_counts[action])

    template = get_plotly_template(theme_mode)
    action_colors = get_action_colors(theme_mode)
    fig = go.Figure(
        data=go.Bar(
            x=action_order,
            y=[action_counts[action] for action in action_order],
            marker_color=[action_colors[action] for action in action_order],
            text=[str(action_counts[action]) for action in action_order],
            textposition="outside",
        )
    )
    fig.update_layout(
        **template,
        title="Action 분포",
        xaxis_title="",
        yaxis_title="섹터 수",
        height=280,
        showlegend=False,
    )
    fig.update_yaxes(dtick=1, rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)


def render_signal_table(
    signals: list,
    filter_action: str | None = None,
    filter_regime_only: bool = False,
    current_regime: str | None = None,
    theme_mode: str = "dark",
) -> None:
    """Render signal table with action and regime filters.

    N/A rows are rendered with neutral style and explicit text hints.

    Args:
        signals: list[SectorSignal].
        filter_action: If set, show only signals with this action value.
        filter_regime_only: If True, show only sectors matching current_regime.
        current_regime: Current macro regime name (for filter_regime_only).
        theme_mode: "dark" or "light".
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

    tokens = get_theme_tokens(theme_mode)
    badge_styles = get_action_badge_styles(theme_mode)
    table_tokens = get_table_style_tokens(theme_mode)

    rows = []
    for s in filtered:
        alerts_str = ", ".join(s.alerts) if s.alerts else "-"
        ret_1m_val = s.returns.get("1M", float("nan"))
        ret_3m_val = s.returns.get("3M", float("nan"))
        ret_1m = f"{ret_1m_val * 100:.1f}%" if not pd.isna(ret_1m_val) else "N/A"
        ret_3m = f"{ret_3m_val * 100:.1f}%" if not pd.isna(ret_3m_val) else "N/A"
        rsi_str = f"{s.rsi_d:.1f}" if not pd.isna(s.rsi_d) else "N/A"
        vol_str = f"{s.volatility_20d * 100:.1f}%" if not pd.isna(s.volatility_20d) else "N/A"
        mdd_str = f"{s.mdd_3m * 100:.1f}%" if not pd.isna(s.mdd_3m) else "N/A"
        provisional_marker = " *" if s.is_provisional else ""

        rows.append(
            {
                "섹터": s.sector_name + provisional_marker,
                "매크로": s.macro_regime,
                "적합": "✔ 적합" if s.macro_fit else "✖ 비적합",
                "액션": format_action_label(s.action),
                "상태": "데이터 없음" if s.action == "N/A" else "데이터 정상",
                "RSI(D)": rsi_str,
                "1M": ret_1m,
                "3M": ret_3m,
                "변동성": vol_str,
                "MDD(3M)": mdd_str,
                "알림": alerts_str,
            }
        )

    df_display = pd.DataFrame(rows)

    def _highlight(row: pd.Series) -> list[str]:
        action = ACTION_BY_LABEL.get(str(row["액션"]), "N/A")
        style = badge_styles.get(action, badge_styles["N/A"])
        emphasis = (
            f"background-color: {style['bg']}; "
            f"color: {style['text']}; "
            f"font-weight: 600;"
        )

        styles = ["" for _ in range(len(row))]
        if "액션" in row.index:
            styles[row.index.get_loc("액션")] = emphasis
        if "상태" in row.index:
            styles[row.index.get_loc("상태")] = emphasis
        if action == "N/A" and "섹터" in row.index:
            styles[row.index.get_loc("섹터")] = f"color: {tokens['text_muted']};"
        return styles

    styled = df_display.style.apply(_highlight, axis=1)

    action_col = df_display.columns[3] if len(df_display.columns) > 3 else None
    status_col = df_display.columns[4] if len(df_display.columns) > 4 else None
    sector_col = df_display.columns[0] if len(df_display.columns) > 0 else None

    def _highlight_with_theme(row: pd.Series) -> list[str]:
        action = (
            ACTION_BY_LABEL.get(str(row[action_col]), "N/A")
            if action_col and action_col in row.index
            else "N/A"
        )
        style = badge_styles.get(action, badge_styles["N/A"])
        row_idx = int(row.name) if isinstance(row.name, int) else 0
        row_bg = (
            table_tokens["row_bg_even"]
            if row_idx % 2 == 0
            else table_tokens["row_bg_odd"]
        )
        base = (
            f"background-color: {row_bg}; "
            f"color: {table_tokens['row_text']}; "
            f"border-bottom: 1px solid {table_tokens['grid']};"
        )
        emphasis = (
            f"background-color: {style['bg']}; "
            f"color: {style['text']}; "
            f"font-weight: 600; "
            f"border-bottom: 1px solid {table_tokens['grid']};"
        )

        styles = [base for _ in range(len(row))]
        if action_col and action_col in row.index:
            styles[row.index.get_loc(action_col)] = emphasis
        if status_col and status_col in row.index:
            styles[row.index.get_loc(status_col)] = emphasis
        if action == "N/A" and sector_col and sector_col in row.index:
            styles[row.index.get_loc(sector_col)] = (
                f"background-color: {row_bg}; "
                f"color: {tokens['text_muted']}; "
                f"border-bottom: 1px solid {table_tokens['grid']};"
            )
        return styles

    styled = (
        df_display.style.apply(_highlight_with_theme, axis=1).set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", table_tokens["header_bg"]),
                        ("color", table_tokens["header_text"]),
                        ("font-weight", "700"),
                        ("border-bottom", f"1px solid {table_tokens['grid']}"),
                    ],
                }
            ],
            overwrite=False,
        )
    )

    table_height = 38 + (len(df_display) * 35) + 5
    st.dataframe(styled, use_container_width=True, hide_index=True, height=table_height)

    if any(s.is_provisional for s in filtered):
        st.caption("* 잠정치 포함 (최근 3개월 KOSIS 데이터)")


def _action_badge(action: str) -> str:
    """Return HTML badge for action value."""
    css_class = ACTION_CSS_CLASS.get(action, "action-na")
    label = format_action_label(action)
    return f'<span class="{css_class}">{label}</span>'
