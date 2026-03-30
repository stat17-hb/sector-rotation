"""Panel-oriented UI renderers."""
from __future__ import annotations

from src.ui.base import *
def _render_card_html(
    *,
    eyebrow: str,
    value: str,
    detail: str = "",
    tone: str = "info",
    extra_html: str = "",
) -> str:
    detail_html = (
        f'<div class="status-card__detail">{html.escape(detail)}</div>'
        if detail
        else ""
    )
    return (
        f'<div class="status-card" data-tone="{html.escape(tone)}">'
        f'<div class="status-card__eyebrow">{html.escape(eyebrow)}</div>'
        f'<div class="status-card__value">{html.escape(value)}</div>'
        f"{detail_html}"
        f"{extra_html}"
        "</div>"
    )


def _render_cards_grid(cards: Sequence[str], class_name: str) -> None:
    if not cards:
        return
    markup = "".join(cards)
    st.markdown(f'<div class="{class_name}">{markup}</div>', unsafe_allow_html=True)


def render_page_header(
    *,
    title: str,
    description: str,
    pills: Sequence[Mapping[str, str]] | None = None,
) -> None:
    """Render the app-level page shell header."""
    pill_markup = "".join(
        (
            '<span class="page-shell__pill" '
            f'data-tone="{html.escape(str(item.get("tone", "info")))}">'
            f'<span>{html.escape(str(item.get("label", "")))}</span>'
            f"<strong>{html.escape(str(item.get('value', '')))}</strong>"
            "</span>"
        )
        for item in (pills or [])
        if str(item.get("label", "")).strip() and str(item.get("value", "")).strip()
    )
    pills_html = (
        f'<div class="page-shell__pills">{pill_markup}</div>' if pill_markup else ""
    )
    st.markdown(
        (
            '<section class="page-shell">'
            '<div class="page-shell__eyebrow">Sector rotation cockpit</div>'
            f'<div class="page-shell__title">{html.escape(title)}</div>'
            f'<div class="page-shell__description">{html.escape(description)}</div>'
            f"{pills_html}"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render_status_strip(banner: Mapping[str, object] | None) -> None:
    """Render a persistent top-of-page status strip for system state."""
    if not banner:
        return

    tone = str(banner.get("level", "info")).strip().lower() or "info"
    title = str(banner.get("title", "")).strip()
    message = str(banner.get("message", "")).strip()
    details = [str(item).strip() for item in banner.get("details", []) if str(item).strip()]
    detail_count = len(details)
    detail_html = (
        f'<span class="status-strip__meta">{detail_count} detail'
        f"{'' if detail_count == 1 else 's'}</span>"
        if detail_count
        else ""
    )

    st.markdown(
        (
            '<div class="status-strip" '
            f'data-tone="{html.escape(tone)}">'
            f'<div class="status-strip__badge">{html.escape(tone.upper())}</div>'
            '<div class="status-strip__body">'
            f'<div class="status-strip__title">{html.escape(title)}</div>'
            f'<div class="status-strip__message">{html.escape(message)}</div>'
            "</div>"
            f"{detail_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if details:
        with st.expander("System details", expanded=False):
            for detail in details:
                st.write(f"- {detail}")


def render_panel_header(
    *,
    eyebrow: str,
    title: str,
    description: str,
    badge: str = "",
) -> None:
    """Render a compact header used above chart and table panels."""
    badge_html = (
        f'<span class="panel-header__badge">{html.escape(badge)}</span>' if badge else ""
    )
    st.markdown(
        (
            '<div class="panel-header">'
            '<div class="panel-header__copy">'
            f'<div class="panel-header__eyebrow">{html.escape(eyebrow)}</div>'
            f'<div class="panel-header__title">{html.escape(title)}</div>'
            f'<div class="panel-header__description">{html.escape(description)}</div>'
            "</div>"
            f"{badge_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_analysis_toolbar(
    *,
    min_date: date,
    max_date: date,
    start_date: date,
    end_date: date,
    selected_range_preset: str,
    selected_cycle_phase: str,
    selected_sector: str,
) -> tuple[date, date, str, bool]:
    """Render the top analysis toolbar and return the committed selection."""
    current_preset = normalize_range_preset(selected_range_preset)

    summary_markup = (
        '<div class="analysis-toolbar__summary">'
        '<div class="analysis-toolbar__summary-item"><span>Window</span>'
        f"<strong>{html.escape(str(start_date))} - {html.escape(str(end_date))}</strong></div>"
        '<div class="analysis-toolbar__summary-item"><span>Cycle</span>'
        f"<strong>{html.escape(format_cycle_phase_label(selected_cycle_phase))}</strong></div>"
        '<div class="analysis-toolbar__summary-item"><span>Sector</span>'
        f"<strong>{html.escape(selected_sector or 'Auto')}</strong></div>"
        "</div>"
    )

    st.markdown(
        (
            '<div class="analysis-toolbar">'
            '<div class="analysis-toolbar__eyebrow">Analysis controls</div>'
            '<div class="analysis-toolbar__title">Pin the period first, then scan regime context and sector leadership.</div>'
            f"{summary_markup}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    with st.form("analysis_toolbar_form"):
        start_col, end_col, preset_col, apply_col = st.columns([1.2, 1.2, 1.6, 0.72])
        with start_col:
            start_input = st.date_input(
                "Start",
                value=start_date,
                min_value=min_date,
                max_value=max_date,
            )
        with end_col:
            end_input = st.date_input(
                "End",
                value=end_date,
                min_value=min_date,
                max_value=max_date,
            )
        with preset_col:
            preset_input = st.segmented_control(
                "Quick range",
                options=["1Y", "3Y", "5Y", "ALL", "CUSTOM"],
                default=current_preset,
                format_func=format_range_preset_label,
                selection_mode="single",
                label_visibility="visible",
                width="stretch",
            )
        with apply_col:
            submitted = st.form_submit_button("Apply", width="stretch", type="primary")

    if not submitted:
        return start_date, end_date, current_preset, False

    end_final = min(pd.Timestamp(end_input).date(), max_date)
    selected_preset = normalize_range_preset(str(preset_input or "CUSTOM"))
    if selected_preset != "CUSTOM":
        start_final, end_final = resolve_range_from_preset(
            max_date=end_final,
            min_date=min_date,
            preset=selected_preset,
        )
        return start_final, end_final, selected_preset, True

    start_final = max(pd.Timestamp(start_input).date(), min_date)
    if start_final > end_final:
        start_final = end_final

    inferred = infer_range_preset(
        start_date=start_final,
        end_date=end_final,
        min_date=min_date,
        max_date=max_date,
    )
    return start_final, end_final, inferred, True


def render_top_bar_filters(
    *,
    current_regime: str,
    action_options: Sequence[str],
    filter_action_key: str = "filter_action_global",
    filter_regime_key: str = "filter_regime_only_global",
    position_mode_key: str = "position_mode",
    alerted_only_key: str = "show_alerted_only",
    is_mobile: bool = False,
) -> tuple[str, bool, str, bool]:
    """Render high-frequency filters in the main content area."""
    with st.container(border=True):
        st.markdown(
            (
                '<div class="command-bar">'
                '<div class="command-bar__eyebrow">Quick filters</div>'
                '<div class="command-bar__title">Refine the live signal board before diving back into the analysis canvas.</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        if is_mobile:
            st.selectbox(
                "Action filter",
                options=list(action_options),
                key=filter_action_key,
            )
            st.toggle(
                "Current regime only",
                key=filter_regime_key,
            )
            st.segmented_control(
                "Position scope",
                options=list(POSITION_MODE_OPTIONS),
                default=normalize_position_mode(str(st.session_state.get(position_mode_key, "all"))),
                format_func=format_position_mode_label,
                selection_mode="single",
                key=position_mode_key,
                width="stretch",
            )
            st.toggle(
                "Alerted only",
                key=alerted_only_key,
            )
        else:
            filter_col, toggle_col, mode_col, alerted_col, summary_col = st.columns([1.4, 1.0, 1.4, 1.0, 2.4])
            with filter_col:
                st.selectbox(
                    "Action filter",
                    options=list(action_options),
                    key=filter_action_key,
                )
            with toggle_col:
                st.toggle(
                    "Current regime only",
                    key=filter_regime_key,
                )
            with mode_col:
                st.segmented_control(
                    "Position scope",
                    options=list(POSITION_MODE_OPTIONS),
                    default=normalize_position_mode(str(st.session_state.get(position_mode_key, "all"))),
                    format_func=format_position_mode_label,
                    selection_mode="single",
                    key=position_mode_key,
                    width="stretch",
                )
            with alerted_col:
                st.toggle(
                    "Alerted only",
                    key=alerted_only_key,
                )
            with summary_col:
                current_action = str(st.session_state.get(filter_action_key, action_options[0]))
                regime_only = bool(st.session_state.get(filter_regime_key, False))
                position_mode = normalize_position_mode(str(st.session_state.get(position_mode_key, "all")))
                alerted_only = bool(st.session_state.get(alerted_only_key, False))
                scope_label = "Matching current regime" if regime_only else "Full universe"
                st.markdown(
                    (
                        '<div class="top-bar-summary">'
                        '<div class="top-bar-summary__item"><span>Regime</span>'
                        f"<strong>{html.escape(current_regime)}</strong></div>"
                        '<div class="top-bar-summary__item"><span>Action</span>'
                        f"<strong>{html.escape(current_action)}</strong></div>"
                        '<div class="top-bar-summary__item"><span>Scope</span>'
                        f"<strong>{html.escape(scope_label)}</strong></div>"
                        '<div class="top-bar-summary__item"><span>Positions</span>'
                        f"<strong>{html.escape(format_position_mode_label(position_mode))}</strong></div>"
                        '<div class="top-bar-summary__item"><span>Alerts</span>'
                        f"<strong>{'Only alerted' if alerted_only else 'All signals'}</strong></div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

        if is_mobile:
            current_action = str(st.session_state.get(filter_action_key, action_options[0]))
            regime_only = bool(st.session_state.get(filter_regime_key, False))
            position_mode = normalize_position_mode(str(st.session_state.get(position_mode_key, "all")))
            alerted_only = bool(st.session_state.get(alerted_only_key, False))
            scope_label = "Matching current regime" if regime_only else "Full universe"
            st.markdown(
                (
                    '<div class="top-bar-summary">'
                    '<div class="top-bar-summary__item"><span>Regime</span>'
                    f"<strong>{html.escape(current_regime)}</strong></div>"
                    '<div class="top-bar-summary__item"><span>Action</span>'
                    f"<strong>{html.escape(current_action)}</strong></div>"
                    '<div class="top-bar-summary__item"><span>Scope</span>'
                    f"<strong>{html.escape(scope_label)}</strong></div>"
                    '<div class="top-bar-summary__item"><span>Positions</span>'
                    f"<strong>{html.escape(format_position_mode_label(position_mode))}</strong></div>"
                    '<div class="top-bar-summary__item"><span>Alerts</span>'
                    f"<strong>{'Only alerted' if alerted_only else 'All signals'}</strong></div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    return (
        str(st.session_state.get(filter_action_key, action_options[0])),
        bool(st.session_state.get(filter_regime_key, False)),
        normalize_position_mode(str(st.session_state.get(position_mode_key, "all"))),
        bool(st.session_state.get(alerted_only_key, False)),
    )


def render_decision_hero(
    *,
    regime: str,
    regime_is_confirmed: bool,
    growth_val: float | None = None,
    inflation_val: float | None = None,
    fx_change: float | None = None,
    fx_label: str = "USD/KRW move",
    is_provisional: bool = False,
    theme_mode: str = "dark",
) -> None:
    """Render the hero card summarizing the current macro regime."""
    tokens = get_theme_tokens(theme_mode)
    regime_colors = {
        "Recovery": tokens["success"],
        "Expansion": tokens["primary"],
        "Slowdown": tokens["warning"],
        "Contraction": tokens["danger"],
        "Indeterminate": tokens["text_muted"],
    }
    accent = regime_colors.get(regime, tokens["primary"])
    regime_subtitle = REGIME_SUBLABELS.get(regime, regime)
    regime_state = "Confirmed regime" if regime_is_confirmed else "Provisional regime"

    chips = [
        f'<span class="decision-hero__chip">{html.escape(regime_state)}</span>',
    ]
    if is_provisional:
        chips.append('<span class="decision-hero__badge">Includes provisional macro prints</span>')

    def _hero_metric(label: str, value: float | None, suffix: str) -> str:
        numeric = _safe_float(value)
        display = f"{numeric:.2f}{suffix}" if numeric is not None else "N/A"
        return (
            '<div class="decision-hero__stat">'
            f'<span class="decision-hero__stat-label">{html.escape(label)}</span>'
            f'<strong>{html.escape(display)}</strong>'
            "</div>"
        )

    fx_numeric = _safe_float(fx_change)
    fx_display = f"{fx_numeric:+.1f}%" if fx_numeric is not None else "N/A"
    hero_metrics = "".join(
        [
            _hero_metric("Leading index", growth_val, "p"),
            _hero_metric("CPI YoY", inflation_val, "%"),
            _hero_metric(fx_label, fx_numeric, "%"),
        ]
    )
    if fx_display == "N/A":
        hero_metrics = hero_metrics.replace("nan%", "N/A")

    st.markdown(
        (
            f'<div class="decision-hero" style="--decision-hero-accent: {accent};">'
            '<div class="decision-hero__copy">'
            '<div class="decision-hero__eyebrow">Macro regime</div>'
            f'<div class="decision-hero__title">{html.escape(regime)}</div>'
            f'<div class="decision-hero__subtitle">{html.escape(regime_subtitle)}</div>'
            f'<div class="decision-hero__chips">{"".join(chips)}</div>'
            "</div>"
            f'<div class="decision-hero__stats">{hero_metrics}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_macro_tile(
    regime: str,
    growth_val: float | None = None,
    inflation_val: float | None = None,
    fx_change: float | None = None,
    fx_label: str = "USD/KRW move",
    is_provisional: bool = False,
    theme_mode: str = "dark",
) -> None:
    """Backward-compatible wrapper for the newer decision hero."""
    render_decision_hero(
        regime=regime,
        regime_is_confirmed=regime != "Indeterminate",
        growth_val=growth_val,
        inflation_val=inflation_val,
        fx_change=fx_change,
        fx_label=fx_label,
        is_provisional=is_provisional,
        theme_mode=theme_mode,
    )


def render_status_card_row(
    *,
    current_regime: str,
    regime_is_confirmed: bool,
    price_status: str,
    macro_status: str,
    yield_curve_status: str | None = None,
) -> None:
    """Render consistent status cards under the hero section."""
    regime_detail = "Confirmed" if regime_is_confirmed else "Provisional"
    yield_inverted = str(yield_curve_status or "").strip().lower() == "inverted"

    cards = [
        _render_card_html(
            eyebrow="Current regime",
            value=current_regime,
            detail=regime_detail,
            tone="info" if current_regime == "Indeterminate" else "success",
        ),
        _render_card_html(
            eyebrow="Market data",
            value=price_status,
            detail="Warehouse / live price path",
            tone=_status_tone(price_status),
        ),
        _render_card_html(
            eyebrow="Macro data",
            value=macro_status,
            detail="Monthly macro warehouse",
            tone=_status_tone(macro_status),
        ),
    ]

    if yield_curve_status:
        cards.append(
            _render_card_html(
                eyebrow="Yield curve",
                value="Inverted" if yield_inverted else "Normal",
                detail="3Y government bond versus base rate",
                tone="warning" if yield_inverted else "success",
            )
        )

    _render_cards_grid(cards, "status-card-grid")


def render_investor_decision_boards(
    *,
    signals: Sequence,
    held_sector_options: Sequence[str],
    held_sectors_key: str = "held_sectors",
    limit: int = 6,
) -> list[str]:
    """Render practical-investing decision boards for held positions and new ideas."""
    from src.ui.tables import render_top_picks_table

    raw_held = st.session_state.get(held_sectors_key, [])
    valid_held = [
        str(item)
        for item in (raw_held if isinstance(raw_held, list) else [])
        if str(item) in held_sector_options
    ]
    if valid_held != raw_held:
        st.session_state[held_sectors_key] = valid_held

    with st.container(border=True):
        render_panel_header(
            eyebrow="Decision boards",
            title="Held-position management and new-buy discovery",
            description="Declare your held sectors first so the dashboard can separate adds, trims, and fresh ideas.",
            badge=f"{len(valid_held)} held",
        )
        selected_held = st.multiselect(
            "Held sectors",
            options=list(held_sector_options),
            default=valid_held,
            key=held_sectors_key,
            placeholder="Select sectors you already hold",
        )
        st.caption("Held sectors personalize the recommendation language for position management versus fresh entries.")

        held_col, new_col = st.columns(2, gap="large")
        with held_col:
            render_panel_header(
                eyebrow="Held actions",
                title="Position management",
                description="Surface add candidates, holds, trims, and exit reviews for sectors already on the book.",
                badge=f"{len(selected_held)} tracked",
            )
            render_top_picks_table(
                signals,
                held_sectors=selected_held,
                position_mode="held",
                limit=limit,
                include_held=False,
            )

        with new_col:
            render_panel_header(
                eyebrow="New ideas",
                title="Fresh-buy discovery",
                description="Separate true new-buy candidates from watchlist names and avoid fresh entries that lack confirmation.",
                badge="Opportunity set",
            )
            render_top_picks_table(
                signals,
                held_sectors=selected_held,
                position_mode="new",
                limit=limit,
                include_held=False,
            )

    return [str(item) for item in selected_held if str(item).strip()]

def render_sector_detail_panel(
    *,
    ranking_rows: Sequence[Mapping[str, object]],
    detail_figure: go.Figure,
    selected_sector: str,
    selected_range_preset: str,
    detail_summary: Mapping[str, object] | None = None,
    preset_options: Sequence[str] = ("1Y", "3Y", "5Y", "ALL"),
) -> tuple[str, str]:
    """Render the linked sector ranking list and detail chart."""
    selected_sector_value = str(selected_sector or "")
    normalized_preset = normalize_range_preset(selected_range_preset)
    ranking_col, chart_col = st.columns([1.05, 2.35], gap="large")

    with ranking_col:
        st.markdown(
            (
                '<div class="sector-rank-list__header">'
                '<div class="sector-rank-list__eyebrow">Sector ranking</div>'
                '<div class="sector-rank-list__title">Current return rank in the active window.</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        for rank, row in enumerate(ranking_rows, start=1):
            sector_label = str(row.get("sector", ""))
            return_pct = _safe_float(row.get("return_pct"))
            is_selected = sector_label == selected_sector_value
            button_col, metric_col = st.columns([3.2, 1], gap="small")
            with button_col:
                clicked = st.button(
                    f"{rank}. {sector_label}",
                    key=f"sector_rank_button_{rank}_{sector_label}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                )
                if clicked:
                    selected_sector_value = sector_label
            with metric_col:
                tone = "positive" if (return_pct or 0.0) >= 0 else "negative"
                metric_text = f"{return_pct:+.1f}%" if return_pct is not None else "N/A"
                st.markdown(
                    (
                        '<div class="sector-rank-list__metric" '
                        f'data-tone="{tone}" data-selected="{str(is_selected).lower()}">'
                        f"{html.escape(metric_text)}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

    with chart_col:
        preset_choice = st.segmented_control(
            "Detail range",
            options=list(preset_options),
            default=normalized_preset if normalized_preset in preset_options else None,
            format_func=format_range_preset_label,
            label_visibility="collapsed",
            width="content",
        )
        if detail_summary:
            st.caption(str(detail_summary.get("conclusion", "")).strip())
            summary_cards = [
                _render_card_html(
                    eyebrow="Decision",
                    value=str(detail_summary.get("decision", "N/A")),
                    detail="Held-aware practical action",
                    tone="info",
                ),
                _render_card_html(
                    eyebrow="Regime fit",
                    value=str(detail_summary.get("regime_fit", "N/A")),
                    detail="Macro alignment",
                    tone="success" if str(detail_summary.get("regime_fit", "")).strip().lower() == "fit" else "warning",
                ),
                _render_card_html(
                    eyebrow="RS trend",
                    value=str(detail_summary.get("rs_trend", "N/A")),
                    detail="Relative-strength context",
                    tone="success" if "Above" in str(detail_summary.get("rs_trend", "")) else "warning",
                ),
                _render_card_html(
                    eyebrow="3M return",
                    value=str(detail_summary.get("return_3m", "N/A")),
                    detail="Recent trailing return",
                    tone="info",
                ),
                _render_card_html(
                    eyebrow="20D vol",
                    value=str(detail_summary.get("volatility_20d", "N/A")),
                    detail="Short-term risk",
                    tone="warning",
                ),
                _render_card_html(
                    eyebrow="Warnings",
                    value=str(detail_summary.get("alerts_text", "None")),
                    detail="Active risk flags",
                    tone="warning" if str(detail_summary.get("alerts_text", "None")) != "None" else "success",
                ),
            ]
            _render_cards_grid(summary_cards, "status-card-grid")
        st.plotly_chart(detail_figure, width="stretch", config={"displayModeBar": False})

    return selected_sector_value, normalize_range_preset(str(preset_choice or normalized_preset))


def render_action_summary(signals: Sequence, theme_mode: str = "dark") -> None:
    """Render action counts plus a distribution chart."""
    if not signals:
        st.info("No signal data available.")
        return

    full_order = ["Strong Buy", "Watch", "Hold", "Avoid", "N/A"]
    action_counts = {action: 0 for action in full_order}
    for signal in signals:
        action_counts[signal.action] = action_counts.get(signal.action, 0) + 1

    display_order = (
        full_order
        if action_counts.get("N/A", 0) > 0
        else [action for action in full_order if action != "N/A"]
    )
    cards = [
        _render_card_html(
            eyebrow="Universe",
            value=str(sum(action_counts.values())),
            detail="Filtered sectors in view",
            tone="info",
        )
    ]
    for action in display_order:
        cards.append(
            _render_card_html(
                eyebrow=action,
                value=str(action_counts[action]),
                detail=format_action_label(action),
                tone=_action_tone(action),
            )
        )
    _render_cards_grid(cards, "summary-kpi-grid")

    template = get_plotly_template(theme_mode)
    action_colors = get_action_colors(theme_mode)
    fig = go.Figure(
        data=go.Bar(
            x=display_order,
            y=[action_counts[action] for action in display_order],
            marker_color=[action_colors[action] for action in display_order],
            text=[str(action_counts[action]) for action in display_order],
            textposition="outside",
        )
    )
    fig.update_layout(
        **template,
        title="Action distribution",
        xaxis_title="",
        yaxis_title="Sector count",
        height=280,
        showlegend=False,
    )
    max_count = max(action_counts.values()) if action_counts else 1
    fig.update_yaxes(dtick=1, range=[0, max_count * 1.25])
    st.plotly_chart(fig, width="stretch")
