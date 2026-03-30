"""Figure and chart builders."""
from __future__ import annotations

from src.ui.base import *
def render_rs_scatter(
    signals: Sequence,
    *,
    height: int = 680,
    margin: dict[str, int] | None = None,
    theme_mode: str = "dark",
) -> go.Figure:
    """Render a relative-strength scatter plot."""
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)
    action_colors = get_action_colors(theme_mode)

    x_vals: list[float] = []
    y_vals: list[float] = []
    texts: list[str] = []
    colors: list[str] = []
    hovers: list[str] = []

    for signal in signals:
        rs = _safe_float(getattr(signal, "rs", None))
        rs_ma = _safe_float(getattr(signal, "rs_ma", None))
        if getattr(signal, "action", "N/A") == "N/A":
            continue
        if rs is None or rs_ma is None:
            continue

        x_vals.append(rs)
        y_vals.append(rs_ma)
        texts.append(signal.sector_name.split(" ")[-1])
        colors.append(action_colors.get(signal.action, tokens["text_muted"]))
        hovers.append(
            "<b>{}</b><br>Action: {}<br>RS: {:.4f}<br>RS MA: {:.4f}<br>RSI(D): {:.1f}<br>"
            "Trend: {}<br>Alerts: {}".format(
                html.escape(signal.sector_name),
                html.escape(signal.action),
                rs,
                rs_ma,
                float(signal.rsi_d),
                "Healthy" if signal.trend_ok else "Weakening",
                html.escape(", ".join(signal.alerts) or "None"),
            )
        )

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
                marker=dict(
                    color=colors,
                    size=12,
                    line=dict(width=1, color=tokens["surface"]),
                ),
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
            text="No valid RS / RS MA points are available. Check benchmark coverage first.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": tokens["text"]},
        )

    fig.update_layout(
        **template,
        title="Relative Strength versus RS Moving Average",
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


def render_rs_momentum_bar(signals: Sequence, theme_mode: str = "dark") -> go.Figure:
    """Render a horizontal bar chart of RS divergence."""
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)
    action_colors = get_action_colors(theme_mode)

    filtered = []
    for signal in signals:
        if getattr(signal, "action", "N/A") == "N/A":
            continue
        rs = _safe_float(getattr(signal, "rs", None))
        rs_ma = _safe_float(getattr(signal, "rs_ma", None))
        if rs is None or rs_ma in {None, 0.0}:
            continue
        filtered.append(signal)

    if not filtered:
        return go.Figure()

    def rs_div(signal) -> float:
        assert signal.rs_ma != 0
        return (signal.rs - signal.rs_ma) / signal.rs_ma * 100

    filtered_sorted = sorted(filtered, key=rs_div)
    names = [signal.sector_name.split(" ")[-1] for signal in filtered_sorted]
    values = [rs_div(signal) for signal in filtered_sorted]
    colors = [
        action_colors.get(signal.action, tokens["text_muted"]) for signal in filtered_sorted
    ]
    hovers = [
        "<b>{}</b><br>RS gap: {:+.2f}%<br>RS: {:.4f} / RS MA: {:.4f}<br>Action: {}".format(
            html.escape(signal.sector_name),
            rs_div(signal),
            signal.rs,
            signal.rs_ma,
            html.escape(signal.action),
        )
        for signal in filtered_sorted
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
            text=[f"{value:+.2f}%" for value in values],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line=dict(color=tokens["border"], width=1.5))
    fig.update_layout(
        **template,
        title="RS gap by sector",
        xaxis_title="RS gap (%)",
        yaxis_title="",
        height=max(300, len(filtered_sorted) * 36 + 80),
        showlegend=False,
    )
    fig.update_xaxes(ticksuffix="%")
    return fig


def render_returns_heatmap(signals: Sequence, theme_mode: str = "dark") -> go.Figure:
    """Render a multi-period sector returns heatmap."""
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)

    periods = ["1W", "1M", "3M", "6M", "12M"]
    sector_names: list[str] = []
    z_values: list[list[float | None]] = []

    for signal in signals:
        if getattr(signal, "action", "N/A") == "N/A" or not getattr(signal, "returns", {}):
            continue
        sector_names.append(signal.sector_name.split()[-1])
        row = []
        for period in periods:
            raw = _safe_float(signal.returns.get(period))
            row.append(raw * 100 if raw is not None else None)
        z_values.append(row)

    if not sector_names:
        fig = go.Figure()
        fig.update_layout(**template, title="No return data available")
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
        title="Multi-period return heatmap (%)",
        height=max(300, len(sector_names) * 40),
    )
    return fig


def _resolve_heatmap_density_mode(
    *,
    month_count: int,
    row_count: int,
) -> dict[str, object]:
    """Return deterministic label/text settings for the analysis heatmap."""
    if month_count <= 36:
        label_step = 1
        tickangle = 0
        bottom_margin = 64
        tickfont_size = 11
    elif month_count <= 72:
        label_step = 1
        tickangle = -90
        bottom_margin = 128
        tickfont_size = 10
    else:
        label_step = max(1, math.ceil(month_count / 48))
        tickangle = -90
        bottom_margin = 128
        tickfont_size = 9

    show_cell_text = month_count <= 36 and (month_count * row_count) <= 432
    helper_text = ""
    if not show_cell_text:
        helper_text = "<br><sup>Hover or click a cell to inspect exact monthly return values.</sup>"

    return {
        "label_step": label_step,
        "tickangle": tickangle,
        "bottom_margin": bottom_margin,
        "tickfont_size": tickfont_size,
        "show_cell_text": show_cell_text,
        "helper_text": helper_text,
    }


def build_sector_strength_heatmap(
    heatmap_df: pd.DataFrame,
    *,
    selected_sector: str | None = None,
    selected_month: str | None = None,
    theme_mode: str = "dark",
    palette: str = "classic",
    title: str = "Monthly sector return",
    empty_message: str = "No monthly sector return data is available for the active filters.",
    helper_metric_label: str = "monthly return",
    hover_value_suffix: str = "%",
) -> go.Figure:
    """Build a monthly sector heatmap used by the analysis canvas."""
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)

    if heatmap_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=empty_message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": tokens["text"]},
        )
        fig.update_layout(**template, title=title, height=320)
        return fig

    x_labels = [str(col) for col in heatmap_df.columns]
    y_labels = [str(idx) for idx in heatmap_df.index]
    x_positions = list(range(len(x_labels)))
    y_positions = list(range(len(y_labels)))
    density_mode = _resolve_heatmap_density_mode(
        month_count=len(x_labels),
        row_count=len(y_labels),
    )
    helper_text = str(density_mode["helper_text"] or "")
    if helper_text:
        helper_text = (
            helper_text.replace("exact monthly return values", f"exact {helper_metric_label} values")
            .replace("monthly return values", f"{helper_metric_label} values")
        )
    label_step = int(density_mode["label_step"])
    ticktext = [
        label if idx % label_step == 0 else ""
        for idx, label in enumerate(x_labels)
    ]
    z_values = heatmap_df.fillna(float("nan")).to_numpy()
    customdata = [
        [[x_labels[col_idx], y_labels[row_idx]] for col_idx in range(len(x_labels))]
        for row_idx in range(len(y_labels))
    ]
    colorscale = get_analysis_heatmap_colorscale(
        theme_mode=theme_mode,
        palette=palette,
    )

    _z_df = pd.DataFrame(z_values)
    _z_min = _z_df.min().min(skipna=True)
    _z_max = _z_df.max().max(skipna=True)
    _abs_max = max(
        abs(float(_z_min)) if pd.notna(_z_min) else 1.0,
        abs(float(_z_max)) if pd.notna(_z_max) else 1.0,
        1.0,
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_positions,
            y=y_positions,
            customdata=customdata,
            colorscale=colorscale,
            zmin=-_abs_max,
            zmax=_abs_max,
            xgap=1,
            ygap=1,
            texttemplate="%{z:.1f}" if bool(density_mode["show_cell_text"]) else None,
            textfont={"size": 10},
            colorbar=dict(
                orientation="h",
                y=1.12,
                x=1.0,
                xanchor="right",
                len=0.28,
                thickness=16,
                title=dict(text="%"),
            ),
            hovertemplate=f"%{{customdata[1]}}<br>%{{customdata[0]}}: %{{z:.1f}}{hover_value_suffix}<extra></extra>",
        )
    )

    fig.update_layout(
        **template,
        title=f"{title}{helper_text}",
        height=max(360, len(y_labels) * 34 + 130),
        clickmode="event+select",
        dragmode="select",
    )
    fig.update_layout(margin=dict(l=108, r=28, t=84, b=int(density_mode["bottom_margin"])))
    fig.update_xaxes(
        tickmode="array",
        tickvals=x_positions,
        ticktext=ticktext,
        side="bottom",
        showgrid=False,
        tickangle=int(density_mode["tickangle"]),
        tickfont={"size": int(density_mode["tickfont_size"])},
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_positions,
        ticktext=y_labels,
        autorange="reversed",
        showgrid=False,
    )

    if selected_sector in y_labels:
        row_index = y_labels.index(str(selected_sector))
        fig.add_shape(
            type="rect",
            x0=-0.5,
            x1=len(x_labels) - 0.5,
            y0=row_index - 0.5,
            y1=row_index + 0.5,
            line=dict(width=0),
            fillcolor=get_chart_tokens(theme_mode)["selection_row_fill"],
            layer="below",
        )
    if selected_month in x_labels:
        col_index = x_labels.index(str(selected_month))
        fig.add_shape(
            type="rect",
            x0=col_index - 0.5,
            x1=col_index + 0.5,
            y0=-0.5,
            y1=len(y_labels) - 0.5,
            line=dict(width=0),
            fillcolor=get_chart_tokens(theme_mode)["selection_col_fill"],
            layer="below",
        )
    if selected_sector in y_labels and selected_month in x_labels:
        row_index = y_labels.index(str(selected_sector))
        col_index = x_labels.index(str(selected_month))
        fig.add_shape(
            type="rect",
            x0=col_index - 0.5,
            x1=col_index + 0.5,
            y0=row_index - 0.5,
            y1=row_index + 0.5,
            line=dict(color=get_chart_tokens(theme_mode)["selection_outline"], width=2),
            fillcolor="rgba(0,0,0,0)",
        )

    return fig


def render_cycle_timeline_panel(
    *,
    segments: Sequence[Mapping[str, object]],
    selected_cycle_phase: str,
    theme_mode: str = "dark",
) -> str:
    """Render cycle chips plus a full-width regime timeline card."""
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)
    signal_tokens = get_signal_tokens(theme_mode)
    palette_markup = "".join(
        (
            '<span class="cycle-palette__item">'
            f'<span class="cycle-palette__swatch {css_class}"></span>'
            f"<span>{html.escape(label)}</span>"
            "</span>"
        )
        for label, css_class in CYCLE_REGIME_PALETTE_LABELS
    )

    st.markdown('<div class="phase-chip-row"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="cycle-palette"><span class="cycle-palette__label">Cycle palette</span>{palette_markup}</div>',
        unsafe_allow_html=True,
    )
    selected_phase = st.segmented_control(
        "Cycle phase",
        options=CYCLE_PHASE_ORDER,
        default=selected_cycle_phase if selected_cycle_phase in CYCLE_PHASE_ORDER else "ALL",
        format_func=format_cycle_phase_label,
        selection_mode="single",
        key="cycle_phase_segmented_control",
        label_visibility="collapsed",
        width="stretch",
    )

    fig = go.Figure()
    if not segments:
        fig.add_annotation(
            text="No regime history is available for the selected window.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": tokens["text"]},
        )
        fig.update_layout(**template, title="Cycle timeline (monthly)", height=240)
        fig.update_xaxes(title="", type="date", tickformat="%Y-%m", dtick="M1", tickangle=-45)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        return str(selected_phase or "ALL")

    phase_styles = signal_tokens["cycle_phase_styles"]

    y0 = 0.18
    y1 = 0.82

    for segment in segments:
        phase_key = str(segment.get("phase_key", ""))
        start = pd.Timestamp(segment.get("start"))
        end = pd.Timestamp(segment.get("end"))
        if pd.isna(start) or pd.isna(end):
            continue
        is_selected = selected_phase not in {None, "", "ALL"} and phase_key == selected_phase
        is_current = bool(segment.get("is_current", False))
        style = phase_styles.get(
            phase_key,
            phase_styles["INDETERMINATE"],
        )
        if phase_key == "INDETERMINATE":
            segment_state = "Indeterminate"
        elif is_selected:
            segment_state = "Selected"
        elif is_current:
            segment_state = "Current"
        else:
            segment_state = "Context"
        line_width = 4 if is_selected else 3 if is_current else 1.25
        line_color = (
            tokens["text"]
            if is_selected
            else signal_tokens["cycle_current_line"] if is_current else style["line"]
        )
        trace_opacity = 1.0 if is_selected else 0.92 if is_current else 0.62
        # Enforce minimum display width so short regimes (even single-month) remain visible
        min_width = pd.Timedelta(days=20)
        end_display = end if (end - start) >= min_width else start + min_width

        label_text = str(segment.get("label", phase_key))

        fig.add_trace(
            go.Scatter(
                x=[start, end_display, end_display, start, start],
                y=[y0, y0, y1, y1, y0],
                mode="lines",
                fill="toself",
                fillcolor=style["fill"],
                line=dict(color=line_color, width=line_width),
                opacity=trace_opacity,
                hovertemplate=(
                    f"{html.escape(label_text)}"
                    "<br>%{customdata[0]} -> %{customdata[1]}"
                    "<br>%{customdata[2]}"
                    "<br>Status: %{customdata[3]}<extra></extra>"
                ),
                customdata=[[
                    start.strftime("%Y-%m"),
                    end.strftime("%Y-%m"),
                    str(segment.get("summary", "No sector summary available.")),
                    segment_state,
                ]] * 5,
                name=label_text,
                showlegend=False,
            )
        )
        # Text label centred on the segment (visible without hovering)
        mid_ts = start + (end_display - start) / 2
        fig.add_annotation(
            x=mid_ts,
            y=(y0 + y1) / 2,
            text=label_text,
            showarrow=False,
            font={"size": 10, "color": tokens["text"]},
            opacity=trace_opacity,
            xanchor="center",
            yanchor="middle",
        )

    fig.update_layout(
        **template,
        title="Cycle timeline (monthly)",
        height=270,
    )
    fig.update_layout(margin=dict(l=24, r=24, t=52, b=82))
    fig.update_xaxes(
        title="",
        type="date",
        tickformat="%Y-%m",
        dtick="M1",
        tickangle=-45,
        showgrid=True,
        ticklabelmode="period",
    )
    fig.update_yaxes(title="", visible=False, range=[0, 1], fixedrange=True)

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    return str(selected_phase or "ALL")


def build_sector_detail_figure(
    series_df: pd.DataFrame,
    *,
    selected_sector: str,
    benchmark_label: str | None = None,
    comparison_sectors: Sequence[str] | None = None,
    selected_month: str | None = None,
    theme_mode: str = "dark",
) -> go.Figure:
    """Build the linked sector detail line chart."""
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)

    fig = go.Figure()
    if series_df.empty or selected_sector not in series_df.columns:
        fig.add_annotation(
            text="No sector detail data is available for the current selection.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": tokens["text"]},
        )
        fig.update_layout(**template, title="Selected sector detail", height=360)
        return fig

    compare_order: list[str] = []
    if benchmark_label and benchmark_label in series_df.columns and benchmark_label != selected_sector:
        compare_order.append(benchmark_label)
    for sector in comparison_sectors or []:
        if sector in series_df.columns and sector not in compare_order and sector != selected_sector:
            compare_order.append(sector)

    muted_colors = list(get_chart_tokens(theme_mode)["muted_lines"])
    for index, sector in enumerate(compare_order):
        fig.add_trace(
            go.Scatter(
                x=series_df.index,
                y=series_df[sector],
                mode="lines",
                line=dict(color=muted_colors[index % len(muted_colors)], width=2, dash="solid" if sector != benchmark_label else "dash"),
                name=sector,
                hovertemplate=f"{html.escape(sector)}<br>%{{x|%Y-%m-%d}}: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=series_df.index,
            y=series_df[selected_sector],
            mode="lines",
            line=dict(color=tokens["primary"], width=3.5),
            name=selected_sector,
            hovertemplate=f"{html.escape(selected_sector)}<br>%{{x|%Y-%m-%d}}: %{{y:.2f}}<extra></extra>",
        )
    )

    if selected_month:
        month_mask = series_df.index.to_period("M").astype(str) == str(selected_month)
        if month_mask.any():
            selected_points = series_df.loc[month_mask]
            fig.add_trace(
                go.Scatter(
                    x=selected_points.index,
                    y=selected_points[selected_sector],
                    mode="markers",
                    marker=dict(color=tokens["primary"], size=8, line=dict(color=tokens["surface"], width=1)),
                    name="Pinned month",
                    showlegend=False,
                    hovertemplate=f"{html.escape(selected_sector)}<br>%{{x|%Y-%m-%d}}: %{{y:.2f}}<extra></extra>",
                )
            )

    fig.update_layout(
        **template,
        title="Selected sector detail",
        height=400,
    )
    fig.update_layout(margin=dict(l=48, r=20, t=58, b=48))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_yaxes(title="Indexed performance")
    fig.update_xaxes(title="")
    return fig
