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
        '<div class="page-shell__meta">'
        '<div class="page-shell__meta-eyebrow">market context</div>'
        f'<div class="page-shell__pills">{pill_markup}</div>'
        "</div>"
        if pill_markup
        else ""
    )
    st.markdown(
        (
            '<section class="page-shell">'
            '<div class="page-shell__grid">'
            '<div class="page-shell__main">'
            '<div class="page-shell__eyebrow">섹터 로테이션 종합상황판</div>'
            f'<div class="page-shell__title">{html.escape(title)}</div>'
            f'<div class="page-shell__description">{html.escape(description)}</div>'
            "</div>"
            f"{pills_html}"
            "</div>"
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
        with st.expander("시스템 상세 정보", expanded=False):
            for detail in details:
                st.write(f"- {detail}")


def render_progress_panel(host, event: Mapping[str, object] | None) -> None:
    """Render a task progress panel into the provided placeholder/container host."""
    if host is None:
        return
    if not event:
        host.empty()
        return

    task = str(event.get("task", "작업 진행")).strip() or "작업 진행"
    phase = str(event.get("phase", "")).strip()
    detail = str(event.get("detail", "")).strip()
    status = str(event.get("status", "running")).strip().lower() or "running"
    try:
        pct = int(round(float(event.get("pct", 0) or 0)))
    except Exception:
        pct = 0
    pct = max(0, min(100, pct))

    host.empty()
    with host.container():
        badge = {
            "running": "진행 중",
            "complete": "완료",
            "error": "실패",
        }.get(status, status.upper())
        st.caption(f"{task} · {badge} · {pct}%")
        st.progress(pct)
        if phase:
            st.markdown(f"**{phase}**")
        if detail:
            if status == "error":
                st.error(detail)
            elif status == "complete":
                st.success(detail)
            else:
                st.caption(detail)


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


def render_research_page_frame(
    *,
    page_key: str,
    eyebrow: str,
    title: str,
    description: str,
    summary_items: Sequence[Mapping[str, object]] | None = None,
) -> None:
    """Render a consistent intent/status frame for non-overview research pages."""
    summary_markup = "".join(
        (
            '<div class="research-page-frame__item">'
            f'<span>{html.escape(str(item.get("label", "")))}</span>'
            f'<strong>{html.escape(str(item.get("value", "")))}</strong>'
            "</div>"
        )
        for item in (summary_items or [])
        if str(item.get("label", "")).strip() and str(item.get("value", "")).strip()
    )
    summary_html = (
        f'<div class="research-page-frame__summary">{summary_markup}</div>'
        if summary_markup
        else ""
    )
    st.markdown(
        (
            '<section class="research-page-frame" '
            f'data-page="{html.escape(str(page_key))}">'
            '<div class="research-page-frame__copy">'
            f'<div class="research-page-frame__eyebrow">{html.escape(eyebrow)}</div>'
            f'<div class="research-page-frame__title">{html.escape(title)}</div>'
            f'<div class="research-page-frame__description">{html.escape(description)}</div>'
            "</div>"
            f"{summary_html}"
            "</section>"
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
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> tuple[date, date, str, bool]:
    """Render the top analysis toolbar and return the committed selection."""
    current_preset = normalize_range_preset(selected_range_preset)

    summary_markup = (
        '<div class="analysis-toolbar__summary">'
        f'<div class="analysis-toolbar__summary-item"><span>{html.escape(get_ui_text("analysis_toolbar_period_label", locale))}</span>'
        f"<strong>{html.escape(str(start_date))} - {html.escape(str(end_date))}</strong></div>"
        f'<div class="analysis-toolbar__summary-item"><span>{html.escape(get_ui_text("analysis_toolbar_cycle_label", locale))}</span>'
        f"<strong>{html.escape(format_cycle_phase_label(selected_cycle_phase, locale=locale))}</strong></div>"
        f'<div class="analysis-toolbar__summary-item"><span>{html.escape(get_ui_text("analysis_toolbar_sector_label", locale))}</span>'
        f"<strong>{html.escape(selected_sector or '자동')}</strong></div>"
        "</div>"
    )

    st.markdown(
        (
            '<div class="analysis-toolbar">'
            f'<div class="analysis-toolbar__eyebrow">{html.escape(get_ui_text("analysis_toolbar_eyebrow", locale))}</div>'
            f'<div class="analysis-toolbar__title">{html.escape(get_ui_text("analysis_toolbar_title", locale))}</div>'
            f'<div class="analysis-toolbar__description">{html.escape(get_ui_text("analysis_toolbar_description", locale))}</div>'
            f"{summary_markup}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    with st.form("analysis_toolbar_form"):
        start_col, end_col, preset_col, apply_col = st.columns([1.2, 1.2, 1.6, 0.72])
        with start_col:
            start_input = st.date_input(
                get_ui_text("analysis_toolbar_start_date", locale),
                value=start_date,
                min_value=min_date,
                max_value=max_date,
            )
        with end_col:
            end_input = st.date_input(
                get_ui_text("analysis_toolbar_end_date", locale),
                value=end_date,
                min_value=min_date,
                max_value=max_date,
            )
        with preset_col:
            preset_input = st.segmented_control(
                get_ui_text("analysis_toolbar_preset_label", locale),
                options=["1Y", "3Y", "5Y", "ALL", "CUSTOM"],
                default=current_preset,
                format_func=lambda value: format_range_preset_label(value, locale=locale),
                selection_mode="single",
                label_visibility="visible",
                width="stretch",
            )
        with apply_col:
            submitted = st.form_submit_button(
                get_ui_text("analysis_toolbar_apply", locale),
                width="stretch",
                type="primary",
            )

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


def render_stock_lookup_control(
    *,
    market_id: str,
    query_value: str,
    status: str = "",
    message: str = "",
    display_model: Mapping[str, Any] | None = None,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> tuple[str, bool]:
    """Render the stock -> sector lookup control and return (query, submitted)."""
    normalized_market = str(market_id or "KR").strip().upper() or "KR"
    market_note_key = "stock_lookup_market_note_us" if normalized_market == "US" else "stock_lookup_market_note_kr"

    st.markdown(
        (
            '<div class="analysis-toolbar">'
            f'<div class="analysis-toolbar__eyebrow">{html.escape(get_ui_text("stock_lookup_eyebrow", locale))}</div>'
            f'<div class="analysis-toolbar__title">{html.escape(get_ui_text("stock_lookup_title", locale))}</div>'
            f'<div class="analysis-toolbar__description">{html.escape(get_ui_text("stock_lookup_description", locale))}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.caption(get_ui_text(market_note_key, locale))

    with st.form("stock_lookup_form"):
        query_col, submit_col = st.columns([3.2, 0.8])
        with query_col:
            query_input = st.text_input(
                get_ui_text("stock_lookup_label", locale),
                value=str(query_value or ""),
            )
        with submit_col:
            submitted = st.form_submit_button(
                get_ui_text("stock_lookup_apply", locale),
                width="stretch",
                type="primary",
            )

    normalized_status = str(status or "").strip().lower()
    if str(message or "").strip():
        if normalized_status == "success":
            st.success(message)
        elif normalized_status in {"ambiguous", "error"}:
            st.warning(message)
        else:
            st.info(message)

    detail = dict(display_model or {})
    matched_sectors = list(detail.get("matched_sectors") or [])
    canonical_sector = dict(detail.get("canonical_sector") or {})
    result = dict(detail.get("result") or {})
    if matched_sectors:
        selected_code = str(canonical_sector.get("sector_code", "")).strip()
        selected_suffix = get_ui_text("stock_lookup_selected_suffix", locale)
        st.caption(get_ui_text("stock_lookup_matches_label", locale))
        for candidate in matched_sectors:
            sector_name = str(candidate.get("sector_name", "")).strip()
            effective = str(candidate.get("snapshot_date") or candidate.get("resolved_from") or "").strip()
            if sector_name:
                selected = str(candidate.get("sector_code", "")).strip() == selected_code
                label = f"{sector_name} ({selected_suffix})" if selected else sector_name
                suffix = f" ({effective})" if effective else ""
                st.markdown(f"- {label}{suffix}")
    if result:
        provenance_bits: list[str] = []
        basis = str(result.get("canonicalization_basis", "")).strip()
        date_mode = str(result.get("match_date_mode", "")).strip()
        effective_date = str(result.get("match_effective_date", "")).strip()
        if basis:
            provenance_bits.append(basis)
        if date_mode:
            provenance_bits.append(date_mode)
        if effective_date:
            provenance_bits.append(effective_date)
        if provenance_bits:
            st.caption(f"{get_ui_text('stock_lookup_provenance_label', locale)}: {' · '.join(provenance_bits)}")

    return str(query_input or ""), submitted


def _format_overview_number(value: object, *, decimals: int = 2) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric:,.{decimals}f}"


def _format_overview_pct(value: object, *, decimals: int = 2) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric:+.{decimals}f}%"


def _series_change_pct(series: pd.Series, periods: int = 1) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= periods:
        return None
    previous = float(clean.iloc[-(periods + 1)])
    latest = float(clean.iloc[-1])
    if previous == 0:
        return None
    return (latest / previous - 1.0) * 100.0


def _period_to_days(period: str) -> int:
    return {"1M": 22, "3M": 66, "6M": 132, "1Y": 252, "YTD": 252}.get(str(period), 66)


def _build_overview_market_cards(
    *,
    prices_wide: pd.DataFrame,
    benchmark_label: str,
    signals: Sequence,
    current_regime: str,
    price_status: str,
    macro_status: str,
) -> list[str]:
    cards: list[str] = []
    preferred = [benchmark_label]
    preferred.extend(
        str(getattr(signal, "sector_name", "")).strip()
        for signal in sorted(signals, key=signal_display_sort_key)[:2]
    )
    labels: list[str] = []
    for label in preferred:
        if label and label in prices_wide.columns and label not in labels:
            labels.append(label)

    for label in labels[:2]:
        series = prices_wide[label]
        latest = pd.to_numeric(series, errors="coerce").dropna()
        value = latest.iloc[-1] if not latest.empty else None
        change = _series_change_pct(series)
        tone = "positive" if (change or 0.0) >= 0 else "negative"
        cards.append(
            '<div class="overview-market-card">'
            f'<div class="overview-market-card__label">{html.escape(label)}</div>'
            f'<div class="overview-market-card__value">{html.escape(_format_overview_number(value))}</div>'
            f'<div class="overview-market-card__change" data-tone="{tone}">{html.escape(_format_overview_pct(change))}</div>'
            "</div>"
        )

    status_items = [
        ("시장 국면", current_regime),
        ("시장 데이터", price_status),
        ("매크로", macro_status),
    ]
    cards.append(
        '<div class="overview-market-card overview-market-card--status">'
        + "".join(
            '<div class="overview-market-card__status-row">'
            f"<span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong>"
            "</div>"
            for label, value in status_items
        )
        + "</div>"
    )
    return cards


def _build_overview_sector_frame(signals: Sequence, *, sort_key: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rank, signal in enumerate(sorted(signals, key=signal_display_sort_key), start=1):
        if str(getattr(signal, "action", "N/A")) == "N/A":
            continue
        rs_gap = _rs_divergence_pct(signal)
        ret_1m = _pct_value(getattr(signal, "returns", {}).get("1M"))
        ret_3m = _pct_value(getattr(signal, "returns", {}).get("3M"))
        mom_score = _safe_float(getattr(signal, "mom_percentile", None))
        if mom_score is None:
            mom_score = rs_gap
        rows.append(
            {
                "순위": rank,
                "섹터": str(getattr(signal, "sector_name", "")),
                "모멘텀 점수": mom_score,
                "상대강도": rs_gap,
                "1M": ret_1m,
                "3M": ret_3m,
                "액션": format_action_label(str(getattr(signal, "action", "N/A"))),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if sort_key == "수익률(3M)":
        frame = frame.sort_values("3M", ascending=False, na_position="last")
    elif sort_key == "상대강도":
        frame = frame.sort_values("상대강도", ascending=False, na_position="last")
    else:
        frame = frame.sort_values("모멘텀 점수", ascending=False, na_position="last")
    frame["순위"] = range(1, len(frame) + 1)
    return frame


def _render_overview_sector_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("표시할 섹터 신호가 없습니다.")
        return
    rows: list[str] = []
    for _, row in frame.head(12).iterrows():
        ret_3m = _safe_float(row.get("3M"))
        ret_tone = "positive" if (ret_3m or 0.0) >= 0 else "negative"
        rows.append(
            "<tr>"
            f"<td>{int(row['순위'])}</td>"
            f"<td>{html.escape(str(row['섹터']))}</td>"
            f"<td>{html.escape(_format_overview_number(row.get('모멘텀 점수'), decimals=2))}</td>"
            f"<td>{html.escape(_format_overview_number(row.get('상대강도'), decimals=2))}</td>"
            f'<td data-tone="{ret_tone}">{html.escape(_format_overview_pct(row.get("3M"), decimals=2))}</td>'
            "</tr>"
        )
    st.markdown(
        '<div class="overview-sector-table-wrap">'
        '<table class="overview-sector-table">'
        "<thead><tr>"
        "<th>순위</th><th>섹터</th><th>모멘텀</th><th>상대강도</th><th>3M</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>",
        unsafe_allow_html=True,
    )


def _build_overview_trend_figure(
    *,
    prices_wide: pd.DataFrame,
    signals: Sequence,
    benchmark_label: str,
    period: str,
    theme_mode: str,
) -> go.Figure:
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)
    fig = go.Figure()
    if prices_wide.empty:
        fig.update_layout(**template, title="섹터 상대강도 추이")
        return fig

    days = _period_to_days(period)
    visible = prices_wide.tail(days).copy()
    candidates = [benchmark_label]
    candidates.extend(str(getattr(signal, "sector_name", "")) for signal in sorted(signals, key=signal_display_sort_key)[:5])
    for label in dict.fromkeys(item for item in candidates if item in visible.columns):
        series = pd.to_numeric(visible[label], errors="coerce").dropna()
        if series.empty:
            continue
        base = float(series.iloc[0])
        if base == 0:
            continue
        normalized = series / base
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized.values,
                mode="lines",
                name=label,
                line={"width": 2.4 if label == benchmark_label else 1.8},
                hovertemplate="%{fullData.name}<br>%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>",
            )
        )
    if not fig.data:
        fig.add_annotation(
            text="표시할 가격 시계열이 없습니다.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": tokens["text"]},
        )
    fig.update_layout(**template)
    fig.update_layout(
        title="섹터 상대강도 추이 (vs 기준값 1.0)",
        height=340,
        margin={"l": 36, "r": 18, "t": 48, "b": 36},
        legend={"orientation": "v", "x": 1.01, "y": 1.0},
    )
    fig.update_yaxes(tickformat=".2f", title="")
    fig.update_xaxes(title="")
    return fig


def _render_overview_heatmap(signals: Sequence) -> None:
    ordered = sorted(
        [signal for signal in signals if str(getattr(signal, "action", "N/A")) != "N/A"],
        key=lambda signal: (_pct_value(getattr(signal, "returns", {}).get("3M")) or -999.0),
        reverse=True,
    )[:10]
    if not ordered:
        st.info("히트맵에 표시할 섹터 수익률 데이터가 없습니다.")
        return
    tiles: list[str] = []
    for signal in ordered:
        ret_3m = _pct_value(getattr(signal, "returns", {}).get("3M"))
        tone = "positive" if (ret_3m or 0.0) >= 0 else "negative"
        magnitude = min(100, max(18, int(abs(ret_3m or 0.0) * 8)))
        tiles.append(
            '<div class="overview-heatmap-tile" '
            f'data-tone="{tone}" style="--tile-strength:{magnitude}%">'
            f'<span>{html.escape(str(getattr(signal, "sector_name", "")))}</span>'
            f'<strong>{html.escape(_format_overview_pct(ret_3m, decimals=2))}</strong>'
            "</div>"
        )
    st.markdown(
        '<div class="overview-heatmap-grid">' + "".join(tiles) + "</div>",
        unsafe_allow_html=True,
    )


def render_toss_overview_dashboard(
    *,
    market_id: str,
    current_regime: str,
    price_status: str,
    macro_status: str,
    prices_wide: pd.DataFrame,
    benchmark_label: str,
    signals: Sequence,
    theme_mode: str,
    sector_map: Mapping[str, object],
    lookup_query_value: str = "",
    lookup_status: str = "",
    lookup_message: str = "",
    lookup_display_model: Mapping[str, Any] | None = None,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> tuple[str, bool]:
    """Render the reference-style overview and return stock lookup submission state."""
    cards = _build_overview_market_cards(
        prices_wide=prices_wide,
        benchmark_label=benchmark_label,
        signals=signals,
        current_regime=current_regime,
        price_status=price_status,
        macro_status=macro_status,
    )
    st.markdown(
        '<section class="overview-reference-shell">'
        '<div class="overview-section-title">시장/국면 한눈에 보기</div>'
        f'<div class="overview-market-grid">{"".join(cards)}</div>'
        "</section>",
        unsafe_allow_html=True,
    )

    lookup_query = str(lookup_query_value or "")
    lookup_submitted = False
    with st.container(border=True):
        st.markdown('<div class="overview-section-title">종목-섹터 조회</div>', unsafe_allow_html=True)
        with st.form("overview_stock_lookup_form"):
            input_col, market_col, button_col = st.columns([3.5, 1.2, 0.82])
            with input_col:
                lookup_query = st.text_input(
                    get_ui_text("stock_lookup_label", locale),
                    value=lookup_query,
                    placeholder="종목명 또는 티커를 입력하세요 (예: 삼성전자, 005930)",
                    label_visibility="collapsed",
                )
            with market_col:
                st.selectbox(
                    "시장",
                    options=[f"{market_id} ({'KOSPI/KOSDAQ' if market_id == 'KR' else 'US'})"],
                    index=0,
                    label_visibility="collapsed",
                    disabled=True,
                )
            with button_col:
                lookup_submitted = st.form_submit_button("상세 조회", width="stretch", type="primary")

        if lookup_status or lookup_message:
            status_text = " · ".join(item for item in [lookup_status, lookup_message] if item)
            st.caption(status_text)
        detail = dict(lookup_display_model or {})
        candidates = list(detail.get("candidates", []) or detail.get("matched_sectors", []) or [])
        if candidates:
            selected_suffix = get_ui_text("stock_lookup_selected_suffix", locale)
            selected_code = str(dict(detail.get("canonical_sector") or {}).get("sector_code", "")).strip()
            chips = []
            for candidate in candidates[:4]:
                sector_name = str(candidate.get("sector_name", "")).strip()
                selected = bool(candidate.get("selected")) or str(candidate.get("sector_code", "")).strip() == selected_code
                if sector_name:
                    chips.append(
                        '<span class="overview-lookup-chip" data-selected="{}">{}</span>'.format(
                            "true" if selected else "false",
                            html.escape(f"{sector_name} ({selected_suffix})" if selected else sector_name),
                        )
                    )
            if chips:
                st.markdown('<div class="overview-lookup-chips">' + "".join(chips) + "</div>", unsafe_allow_html=True)

    filter_col_1, filter_col_2, filter_col_3, filter_col_4 = st.columns([1, 1.2, 1.2, 1.2])
    with filter_col_1:
        period = st.segmented_control(
            "기간",
            options=["1M", "3M", "6M", "1Y"],
            default="3M",
            label_visibility="visible",
        )
    with filter_col_2:
        compare_basis = st.selectbox("비교 기준", options=["벤치마크 대비", "절대 수익률"], index=0)
    with filter_col_3:
        sort_key = st.selectbox("정렬 기준", options=["모멘텀 점수", "수익률(3M)", "상대강도"], index=0)
    with filter_col_4:
        sector_group = st.selectbox("섹터 그룹", options=["WICS 대분류", "전체"], index=0)
    del compare_basis, sector_group, sector_map

    sector_frame = _build_overview_sector_frame(signals, sort_key=str(sort_key))
    left_col, center_col, right_col = st.columns([1.28, 1.52, 1.0], gap="medium")
    with left_col:
        with st.container(border=True):
            st.markdown('<div class="overview-section-title">섹터 모멘텀 & 상대강도</div>', unsafe_allow_html=True)
            _render_overview_sector_table(sector_frame)
    with center_col:
        with st.container(border=True):
            st.markdown('<div class="overview-section-title">섹터 상대강도 추이</div>', unsafe_allow_html=True)
            fig = _build_overview_trend_figure(
                prices_wide=prices_wide,
                signals=signals,
                benchmark_label=benchmark_label,
                period=str(period or "3M"),
                theme_mode=theme_mode,
            )
            st.plotly_chart(fig, width="stretch")
    with right_col:
        with st.container(border=True):
            st.markdown('<div class="overview-section-title">섹터 히트맵 (3M 수익률)</div>', unsafe_allow_html=True)
            _render_overview_heatmap(signals)

    return lookup_query, lookup_submitted


def render_top_bar_filters(
    *,
    current_regime: str,
    action_options: Sequence[str],
    enable_regime_filter: bool = True,
    filter_action_key: str = "filter_action_global",
    filter_regime_key: str = "filter_regime_only_global",
    position_mode_key: str = "position_mode",
    alerted_only_key: str = "show_alerted_only",
    is_mobile: bool = False,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> tuple[str, bool, str, bool]:
    """Render high-frequency filters in the main content area."""
    if not enable_regime_filter:
        st.session_state[filter_regime_key] = False

    with st.container(border=True):
        st.markdown(
            (
                '<div class="command-bar">'
                f'<div class="command-bar__eyebrow">{html.escape(get_ui_text("command_bar_eyebrow", locale))}</div>'
                f'<div class="command-bar__title">{html.escape(get_ui_text("command_bar_title", locale))}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.caption(get_ui_text("command_bar_scope_note", locale))

        if is_mobile:
            st.selectbox(
                get_ui_text("filter_action", locale),
                options=list(action_options),
                format_func=lambda value: get_action_filter_label(value, locale),
                key=filter_action_key,
            )
            if enable_regime_filter:
                st.toggle(
                    get_ui_text("filter_regime_only", locale),
                    key=filter_regime_key,
                )
            st.segmented_control(
                get_ui_text("filter_position_scope", locale),
                options=list(POSITION_MODE_OPTIONS),
                default=normalize_position_mode(str(st.session_state.get(position_mode_key, "all"))),
                format_func=lambda value: format_position_mode_label(value, locale=locale),
                selection_mode="single",
                key=position_mode_key,
                width="stretch",
            )
            st.toggle(
                get_ui_text("filter_alerted_only", locale),
                key=alerted_only_key,
            )
        else:
            column_spec = [1.4, 1.4, 1.0, 2.4] if not enable_regime_filter else [1.4, 1.0, 1.4, 1.0, 2.4]
            columns = st.columns(column_spec)
            if enable_regime_filter:
                filter_col, toggle_col, mode_col, alerted_col, summary_col = columns
            else:
                filter_col, mode_col, alerted_col, summary_col = columns
                toggle_col = None
            with filter_col:
                st.selectbox(
                    get_ui_text("filter_action", locale),
                    options=list(action_options),
                    format_func=lambda value: get_action_filter_label(value, locale),
                    key=filter_action_key,
                )
            if toggle_col is not None:
                with toggle_col:
                    st.toggle(
                        get_ui_text("filter_regime_only", locale),
                        key=filter_regime_key,
                    )
            with mode_col:
                st.segmented_control(
                    get_ui_text("filter_position_scope", locale),
                    options=list(POSITION_MODE_OPTIONS),
                    default=normalize_position_mode(str(st.session_state.get(position_mode_key, "all"))),
                    format_func=lambda value: format_position_mode_label(value, locale=locale),
                    selection_mode="single",
                    key=position_mode_key,
                    width="stretch",
                )
            with alerted_col:
                st.toggle(
                    get_ui_text("filter_alerted_only", locale),
                    key=alerted_only_key,
                )
            with summary_col:
                current_action = str(st.session_state.get(filter_action_key, action_options[0]))
                regime_only = bool(st.session_state.get(filter_regime_key, False)) if enable_regime_filter else False
                position_mode = normalize_position_mode(str(st.session_state.get(position_mode_key, "all")))
                alerted_only = bool(st.session_state.get(alerted_only_key, False))
                scope_label = get_ui_text("scope_matching_regime", locale) if regime_only else get_ui_text("scope_full_universe", locale)
                st.markdown(
                    (
                        '<div class="top-bar-summary">'
                        f'<div class="top-bar-summary__item"><span>{html.escape(get_ui_text("summary_regime_label", locale))}</span>'
                        f"<strong>{html.escape(current_regime)}</strong></div>"
                        f'<div class="top-bar-summary__item"><span>{html.escape(get_ui_text("summary_action_label", locale))}</span>'
                        f"<strong>{html.escape(get_action_filter_label(current_action, locale))}</strong></div>"
                        f'<div class="top-bar-summary__item"><span>{html.escape(get_ui_text("summary_scope_label", locale))}</span>'
                        f"<strong>{html.escape(scope_label)}</strong></div>"
                        f'<div class="top-bar-summary__item"><span>{html.escape(get_ui_text("summary_positions_label", locale))}</span>'
                        f"<strong>{html.escape(format_position_mode_label(position_mode, locale=locale))}</strong></div>"
                        f'<div class="top-bar-summary__item"><span>{html.escape(get_ui_text("summary_alerts_label", locale))}</span>'
                        f"<strong>{html.escape(get_ui_text('summary_alerted_only', locale) if alerted_only else get_ui_text('summary_all_signals', locale))}</strong></div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

        if is_mobile:
            current_action = str(st.session_state.get(filter_action_key, action_options[0]))
            regime_only = bool(st.session_state.get(filter_regime_key, False)) if enable_regime_filter else False
            position_mode = normalize_position_mode(str(st.session_state.get(position_mode_key, "all")))
            alerted_only = bool(st.session_state.get(alerted_only_key, False))
            scope_label = get_ui_text("scope_matching_regime", locale) if regime_only else get_ui_text("scope_full_universe", locale)
            st.markdown(
                (
                    '<div class="top-bar-summary">'
                    f'<div class="top-bar-summary__item"><span>{html.escape(get_ui_text("summary_regime_label", locale))}</span>'
                    f"<strong>{html.escape(current_regime)}</strong></div>"
                    f'<div class="top-bar-summary__item"><span>{html.escape(get_ui_text("summary_action_label", locale))}</span>'
                    f"<strong>{html.escape(get_action_filter_label(current_action, locale))}</strong></div>"
                    f'<div class="top-bar-summary__item"><span>{html.escape(get_ui_text("summary_scope_label", locale))}</span>'
                    f"<strong>{html.escape(scope_label)}</strong></div>"
                    f'<div class="top-bar-summary__item"><span>{html.escape(get_ui_text("summary_positions_label", locale))}</span>'
                    f"<strong>{html.escape(format_position_mode_label(position_mode, locale=locale))}</strong></div>"
                    f'<div class="top-bar-summary__item"><span>{html.escape(get_ui_text("summary_alerts_label", locale))}</span>'
                    f"<strong>{html.escape(get_ui_text('summary_alerted_only', locale) if alerted_only else get_ui_text('summary_all_signals', locale))}</strong></div>"
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
    locale: UiLocale = DEFAULT_UI_LOCALE,
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
    regime_subtitle = get_regime_subtitle(regime, locale)
    regime_state = get_ui_text("hero_regime_confirmed", locale) if regime_is_confirmed else get_ui_text("hero_regime_provisional", locale)

    chips = [
        f'<span class="decision-hero__chip">{html.escape(regime_state)}</span>',
        f'<span class="decision-hero__badge">{html.escape(get_ui_text("hero_method_badge", locale))}</span>',
        f'<span class="decision-hero__badge">{html.escape(get_ui_text("hero_pit_badge", locale))}</span>',
        f'<span class="decision-hero__badge">{html.escape(get_ui_text("hero_contraction_badge", locale))}</span>',
    ]
    if is_provisional:
        chips.append(f'<span class="decision-hero__badge">{html.escape(get_ui_text("hero_provisional_badge", locale))}</span>')

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
    hero_metrics = "".join(
        [
            _hero_metric(get_ui_text("hero_leading_index", locale), growth_val, "p"),
            _hero_metric(get_ui_text("hero_cpi_yoy", locale), inflation_val, "%"),
            _hero_metric(fx_label, fx_numeric, "%"),
        ]
    )

    st.markdown(
        (
            f'<div class="decision-hero" style="--decision-hero-accent: {accent};">'
            '<div class="decision-hero__copy">'
            f'<div class="decision-hero__eyebrow">{html.escape(get_ui_text("hero_eyebrow", locale))}</div>'
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
    locale: UiLocale = DEFAULT_UI_LOCALE,
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
        locale=locale,
    )


def render_status_card_row(
    *,
    current_regime: str,
    regime_is_confirmed: bool,
    price_status: str,
    macro_status: str,
    yield_curve_status: str | None = None,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> None:
    """Render consistent status cards under the hero section."""
    regime_detail = get_ui_text("status_regime_confirmed", locale) if regime_is_confirmed else get_ui_text("status_regime_provisional", locale)
    yield_inverted = str(yield_curve_status or "").strip().lower() == "inverted"

    cards = [
        _render_card_html(
            eyebrow=get_ui_text("status_current_regime", locale),
            value=current_regime,
            detail=regime_detail,
            tone="info" if current_regime == "Indeterminate" else "success",
        ),
        _render_card_html(
            eyebrow=get_ui_text("status_market_data", locale),
            value=price_status,
            detail=get_ui_text("status_market_detail", locale),
            tone=_status_tone(price_status),
        ),
        _render_card_html(
            eyebrow=get_ui_text("status_macro_data", locale),
            value=macro_status,
            detail=get_ui_text("status_macro_detail", locale),
            tone=_status_tone(macro_status),
        ),
    ]

    if yield_curve_status:
        cards.append(
            _render_card_html(
                eyebrow=get_ui_text("status_yield_curve", locale),
                value=get_ui_text("status_yield_inverted", locale) if yield_inverted else get_ui_text("status_yield_normal", locale),
                detail=get_ui_text("status_yield_detail", locale),
                tone="warning" if yield_inverted else "success",
            )
        )

    _render_cards_grid(cards, "status-card-grid")


def render_investor_flow_summary(
    *,
    signals: Sequence,
    investor_flow_status: str,
    investor_flow_fresh: bool,
    investor_flow_profile: str,
    investor_flow_frame: pd.DataFrame | None = None,
    investor_flow_detail: Mapping[str, object] | None = None,
    shared_flow_summary_map: Mapping[str, object] | None = None,
    flow_short_window: int = 20,
    flow_long_window: int = 60,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> None:
    """Render a compact KR investor-flow snapshot under the status cards."""
    has_display_frame = isinstance(investor_flow_frame, pd.DataFrame) and not investor_flow_frame.empty
    if not signals and not has_display_frame:
        return

    with st.container(border=True):
        render_panel_header(
            eyebrow=get_ui_text("flow_status_label", locale),
            title=get_ui_text("flow_summary_title", locale),
            description=get_ui_text("flow_summary_description", locale),
            badge=f"{investor_flow_status} · {get_flow_profile_label(investor_flow_profile, locale)}",
        )
        with st.expander(get_ui_text("flow_sigma_explainer_toggle", locale), expanded=False):
            st.markdown(
                get_ui_text(
                    "flow_sigma_explainer_body",
                    locale,
                    short_window=int(flow_short_window),
                    long_window=int(flow_long_window),
                )
            )
        st.caption(get_ui_text("flow_sidebar_caption", locale))
        reference_only_note = get_flow_reference_only_note(
            investor_flow_status,
            investor_flow_fresh,
            investor_flow_detail,
            locale,
        )
        if reference_only_note and has_display_frame:
            st.warning(reference_only_note)

        signal_rows = build_investor_flow_glance_rows(
            signals,
            locale=locale,
        )
        snapshot_rows = build_investor_flow_snapshot_rows(
            investor_flow_frame,
            shared_flow_summary_map=shared_flow_summary_map,
            locale=locale,
        )
        rows = signal_rows or snapshot_rows
        if not rows:
            if reference_only_note and has_display_frame:
                st.caption(get_ui_text("flow_reference_only_summary_hint", locale))
                return
            st.info(get_ui_text("flow_tab_empty", locale))
            return

        st.caption(get_ui_text("flow_summary_limit_note", locale))
        rows = rows[:4]

        use_signal_rows = bool(signal_rows)

        def _state_tone(raw_state: str | None, *, ratio: float | None = None) -> str:
            if ratio is not None:
                if ratio > 0:
                    return "success"
                if ratio < 0:
                    return "danger"
                return "warning"
            raw_state = str(raw_state or "neutral")
            if raw_state == "supportive":
                return "success"
            if raw_state == "adverse":
                return "danger"
            return "warning"

        def _participant_chip(label: str, value: str, raw_state: str | None = None, *, ratio: float | None = None) -> str:
            tone = _state_tone(raw_state, ratio=ratio)
            return (
                '<span style="display:inline-flex; align-items:center; gap:0.3rem; '
                'padding:0.3rem 0.62rem; border-radius:var(--radius-pill); font-size:var(--flow-chip-size,0.78rem); '
                f'font-weight:700; background:color-mix(in srgb, var(--{tone}) 16%, transparent); '
                f'color:var(--{tone}); border:1px solid color-mix(in srgb, var(--{tone}) 28%, transparent);">'
                f'<span>{html.escape(label)}</span><strong>{html.escape(value)}</strong>'
                '</span>'
            )

        html_chunks = ['<div class="flow-container">']
        for row in rows:
            sector = html.escape(str(row["sector"]))
            flow_label = html.escape(
                str(row["flow_state"]) if use_signal_rows else "참고용 raw snapshot"
            )
            score_label = f'{get_ui_text("flow_col_score", locale)} {float(row["flow_score"]):+.2f}'
            show_score_label = use_signal_rows or str(row.get("flow_state_raw", "unavailable")) != "unavailable"
            score_html = (
                f'<span style="font-size:var(--flow-chip-size,0.78rem); color:var(--text-muted); font-weight:700;">{html.escape(score_label)}</span>'
                if show_score_label
                else ""
            )
            badge_tone = (
                _state_tone(str(row["flow_state_raw"]))
                if use_signal_rows
                else "info"
            )
            badge_class = f"flow-card__badge flow-card__badge--{badge_tone}"

            chips_html = "".join(
                [
                    _participant_chip(
                        get_ui_text("flow_col_foreign", locale),
                        str(row["foreign"]),
                        str(row.get("foreign_raw", "")),
                        ratio=None if use_signal_rows else _safe_float(row.get("foreign_ratio")),
                    ),
                    _participant_chip(
                        get_ui_text("flow_col_institutional", locale),
                        str(row["institutional"]),
                        str(row.get("institutional_raw", "")),
                        ratio=None if use_signal_rows else _safe_float(row.get("institutional_ratio")),
                    ),
                    _participant_chip(
                        get_ui_text("flow_col_retail", locale),
                        str(row["retail"]),
                        str(row.get("retail_raw", "")),
                        ratio=None if use_signal_rows else _safe_float(row.get("retail_ratio")),
                    ),
                ]
            )

            detail_html = (
                f'<div style="margin-top:0.55rem; font-size:0.82rem; color:var(--text-muted);">'
                f'{html.escape(get_ui_text("flow_col_adjustment", locale))}: '
                f'<strong style="color:var(--text);">{html.escape(str(row["action_change"]))}</strong>'
                '</div>'
                if use_signal_rows and not reference_only_note and bool(row["has_action_change"])
                else ""
            )
            cue_parts = []
            for label, key in (
                (get_ui_text("flow_col_foreign", locale), "foreign"),
                (get_ui_text("flow_col_institutional", locale), "institutional"),
                (get_ui_text("flow_col_retail", locale), "retail"),
            ):
                cue = str(row.get(f"{key}_cue", "") or "").strip()
                if cue:
                    cue_parts.append(f"{label} {cue}")
            cue_html = (
                '<div style="margin-top:0.45rem; font-size:0.78rem; color:var(--text-muted);">'
                f'{" · ".join(html.escape(part) for part in cue_parts)}'
                '</div>'
                if cue_parts
                else ""
            )

            card = (
                '<div class="flow-card">'
                '<div class="flow-card__header" style="align-items:flex-start; gap:0.5rem;">'
                '<div class="flow-card__title" style="display:flex; flex-direction:column; gap:0.35rem;">'
                f'<div style="font-weight:800;">{sector}</div>'
                f'<div style="display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap;">'
                f'<span class="{badge_class}">{flow_label}</span>'
                f'{score_html}'
                '</div>'
                '</div>'
                '</div>'
                '<div class="flow-card__body" style="gap:0.65rem;">'
                f'<div style="display:flex; gap:0.45rem; flex-wrap:wrap;">{chips_html}</div>'
                f'{cue_html}'
                f'{detail_html}'
                '</div>'
                '</div>'
            )
            html_chunks.append(card)
        html_chunks.append('</div>')
        st.markdown("".join(html_chunks), unsafe_allow_html=True)

        if not investor_flow_fresh:
            st.caption(get_ui_text("flow_unavailable", locale))


def render_investor_decision_boards(
    *,
    signals: Sequence,
    held_sector_options: Sequence[str],
    held_sectors_key: str = "held_sectors",
    limit: int = 5,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> list[str]:
    """Render practical-investing decision boards for held positions and new ideas."""
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
            eyebrow=get_ui_text("decision_lane_eyebrow", locale),
            title=get_ui_text("decision_lane_title", locale),
            description=get_ui_text("decision_lane_description", locale),
            badge=f"{len(valid_held)}개 보유",
        )
        selected_held = st.multiselect(
            "보유 섹터",
            options=list(held_sector_options),
            default=valid_held,
            key=held_sectors_key,
            placeholder="현재 보유 중인 섹터를 선택하세요",
        )
        st.caption("보유 섹터 설정 시 포지션 관리와 신규 진입에 맞는 추천 문구가 표시됩니다.")
        st.caption(get_ui_text("judgment_disclaimer_caption", locale))

        held_col, new_col = st.columns(2, gap="large")
        with held_col:
            render_panel_header(
                eyebrow="보유 포지션 대응",
                title="포지션 관리",
                description="보유 섹터의 추가 검토, 유지, 비중 축소, 이탈 검토 후보를 제시합니다.",
                badge=f"{len(selected_held)}개 추적 중",
            )
            _render_decision_board_cards(
                signals=signals,
                held_sectors=selected_held,
                position_mode="held",
                limit=limit,
                locale=locale,
            )

        with new_col:
            render_panel_header(
                eyebrow="신규 아이디어",
                title="신규 후보 탐색",
                description="관심 종목 중 신규 진입 후보와 그렇지 않은 종목을 구분합니다.",
                badge="기회 종목군",
            )
            _render_decision_board_cards(
                signals=signals,
                held_sectors=selected_held,
                position_mode="new",
                limit=limit,
                locale=locale,
            )

    return [str(item) for item in selected_held if str(item).strip()]


def _decision_board_empty_message(
    position_mode: str,
    held_sectors: Sequence[str] | None = None,
    *,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> str:
    normalized = normalize_position_mode(position_mode)
    if normalized == "held" and not list(held_sectors or []):
        return get_ui_text("top_picks_empty_held_missing", locale)
    if normalized == "held":
        return get_ui_text("top_picks_empty_held", locale)
    if normalized == "new":
        return get_ui_text("top_picks_empty_new", locale)
    return get_ui_text("top_picks_empty_all", locale)


def _decision_card_parts(text: object, *, limit: int = 3) -> list[str]:
    parts = [str(part).strip() for part in str(text or "").split("|") if str(part).strip()]
    deduped: list[str] = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return deduped[:limit]


def _render_decision_card_chip(text: str, tone: str = "neutral") -> str:
    normalized_tone = "neutral" if tone == "info" else tone
    return (
        f'<span class="flow-card__badge flow-card__badge--{html.escape(normalized_tone)}" '
        'style="margin-left:0; margin-right:0.35rem; margin-bottom:0.35rem;">'
        f"{html.escape(text)}"
        "</span>"
    )


def _render_decision_board_cards(
    *,
    signals: Sequence,
    held_sectors: Sequence[str] | None,
    position_mode: str,
    limit: int = 5,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> None:
    filtered = filter_signals_for_display(
        signals,
        held_sectors=held_sectors,
        position_mode=position_mode,
    )
    if not filtered:
        st.info(_decision_board_empty_message(position_mode, held_sectors, locale=locale))
        return

    filtered = sorted(
        filtered,
        key=lambda signal: signal_display_sort_key(signal, held_sectors),
    )[:limit]
    context_label = get_ui_text(
        "decision_context_held" if normalize_position_mode(position_mode) == "held" else "decision_context_new",
        locale,
    )

    cards: list[str] = []
    for rank, signal in enumerate(filtered, start=1):
        thesis = describe_signal_decision(signal, held_sectors, locale=locale)
        action = str(getattr(signal, "action", "N/A"))
        action_tone = _action_tone(action)
        reason_parts = _decision_card_parts(thesis.get("reason"), limit=3)
        thesis_summary = f"{thesis['judgment_confidence']} · {reason_parts[0]}" if reason_parts else str(thesis["judgment_confidence"])
        chips = [
            _render_decision_card_chip(str(part), tone="neutral")
            for part in reason_parts
        ]
        metrics = [
            (
                get_ui_text("decision_card_confidence", locale),
                str(thesis["judgment_confidence"]),
            ),
            (
                get_ui_text("decision_card_regime_fit", locale),
                str(thesis["regime_fit"]),
            ),
            (
                get_ui_text("period_3m", locale),
                str(thesis["return_3m"]),
            ),
            (
                get_ui_text("decision_card_sector_fit", locale),
                str(thesis["sector_fit_rank"]),
            ),
        ]
        metrics_html = "".join(
            (
                '<span class="top-pick-card__metric">'
                f"<strong>{html.escape(label)}</strong>{html.escape(value)}"
                "</span>"
            )
            for label, value in metrics
            if str(value).strip()
        )
        cards.append(
            (
                '<div class="top-pick-card">'
                '<div class="top-pick-card__header">'
                '<div class="top-pick-card__title">'
                f'<span class="top-pick-card__rank">{rank}.</span>'
                f"{html.escape(str(getattr(signal, 'sector_name', '')))}"
                f'{_render_decision_card_chip(context_label, tone="neutral")}'
                f'{_render_decision_card_chip(str(thesis["decision"]), tone=action_tone)}'
                "</div>"
                "</div>"
                '<div class="top-pick-card__body">'
                '<div class="top-pick-card__row">'
                f'<span class="top-pick-card__label">{html.escape(get_ui_text("decision_card_thesis", locale))}</span>'
                f'<span class="top-pick-card__value">{html.escape(thesis_summary)}</span>'
                "</div>"
                '<div class="top-pick-card__row">'
                f'<span class="top-pick-card__label">{html.escape(get_ui_text("decision_card_why", locale))}</span>'
                f'<span class="top-pick-card__value"><span style="display:flex; flex-wrap:wrap;">{"".join(chips)}</span></span>'
                "</div>"
                '<div class="top-pick-card__row">'
                f'<span class="top-pick-card__label">{html.escape(get_ui_text("decision_card_invalidation", locale))}</span>'
                f'<span class="top-pick-card__value">{html.escape(str(thesis["invalidation"]))}</span>'
                "</div>"
                f'<div class="top-pick-card__metrics">{metrics_html}</div>'
                "</div>"
                "</div>"
            )
        )

    st.markdown("".join(cards), unsafe_allow_html=True)
    if any(getattr(signal, "is_provisional", False) for signal in filtered):
        st.caption(get_ui_text("provisional_caption", locale))

def render_sector_detail_panel(
    *,
    ranking_rows: Sequence[Mapping[str, object]],
    detail_figure: go.Figure,
    selected_sector: str,
    selected_range_preset: str,
    detail_summary: Mapping[str, object] | None = None,
    preset_options: Sequence[str] = ("1Y", "3Y", "5Y", "ALL"),
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> tuple[str, str]:
    """Render the linked sector ranking list and detail chart."""
    selected_sector_value = str(selected_sector or "")
    normalized_preset = normalize_range_preset(selected_range_preset)
    ranking_col, chart_col = st.columns([1.05, 2.35], gap="large")

    with ranking_col:
        st.markdown(
            (
                '<div class="sector-rank-list__header">'
                '<div class="sector-rank-list__eyebrow">섹터 순위</div>'
                '<div class="sector-rank-list__title">선택 기간 기준 현재 수익률 순위입니다.</div>'
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
                    width="stretch",
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
            "상세 기간",
            options=list(preset_options),
            default=normalized_preset if normalized_preset in preset_options else None,
            format_func=lambda value: format_range_preset_label(value, locale=locale),
            label_visibility="collapsed",
            width="content",
        )
        if detail_summary:
            st.caption(str(detail_summary.get("conclusion", "")).strip())
            regime_fit_match = get_ui_text("regime_fit_yes", locale)
            rs_trend_above = get_ui_text("rs_trend_above", locale)
            momentum_state_strong = get_ui_text("momentum_state_strong", locale)
            alerts_none = get_ui_text("alerts_none", locale)
            summary_cards = [
                _render_card_html(
                    eyebrow="투자 판단",
                    value=str(detail_summary.get("decision", "N/A")),
                    detail="보유 여부 반영 실전 대응",
                    tone="info",
                ),
                _render_card_html(
                    eyebrow=get_ui_text("judgment_structure_label", locale),
                    value=str(detail_summary.get("judgment_structure", "N/A")),
                    detail=str(detail_summary.get("judgment_confidence", "N/A")),
                    tone="warning",
                ),
                _render_card_html(
                    eyebrow="국면 적합성",
                    value=str(detail_summary.get("regime_fit", "N/A")),
                    detail="매크로 정합성",
                    tone="success" if str(detail_summary.get("regime_fit", "")).strip() == regime_fit_match else "warning",
                ),
                _render_card_html(
                    eyebrow=get_ui_text("sector_fit_card", locale),
                    value=str(detail_summary.get("sector_fit_rank", "N/A")),
                    detail=str(detail_summary.get("sector_fit_note", "N/A")),
                    tone="info",
                ),
                _render_card_html(
                    eyebrow=str(detail_summary.get("momentum_label", "RS 추세")),
                    value=str(detail_summary.get("rs_trend", "N/A")),
                    detail="핵심 모멘텀 맥락",
                    tone="success" if str(detail_summary.get("rs_trend", "")) in {rs_trend_above, momentum_state_strong} else "warning",
                ),
                _render_card_html(
                    eyebrow="3개월 수익률",
                    value=str(detail_summary.get("return_3m", "N/A")),
                    detail="최근 누적 수익률",
                    tone="info",
                ),
                _render_card_html(
                    eyebrow="20일 변동성",
                    value=str(detail_summary.get("volatility_20d", "N/A")),
                    detail="단기 리스크",
                    tone="warning",
                ),
                _render_card_html(
                    eyebrow="경고",
                    value=str(detail_summary.get("alerts_text", alerts_none)),
                    detail="활성 리스크 신호",
                    tone="warning" if str(detail_summary.get("alerts_text", alerts_none)) != alerts_none else "success",
                ),
            ]
            _render_cards_grid(summary_cards, "status-card-grid")
        st.plotly_chart(detail_figure, width="stretch", config={"displayModeBar": False})

    return selected_sector_value, normalize_range_preset(str(preset_choice or normalized_preset))


def render_action_summary(
    signals: Sequence,
    theme_mode: str = "dark",
    *,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> None:
    """Render action counts plus a distribution chart."""
    if not signals:
        st.info(get_ui_text("signals_empty", locale))
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
            eyebrow=get_ui_text("action_summary_universe", locale),
            value=str(sum(action_counts.values())),
            detail=get_ui_text("action_summary_filtered_count", locale),
            tone="info",
        )
    ]
    for action in display_order:
        cards.append(
            _render_card_html(
                eyebrow=action,
                value=str(action_counts[action]),
                detail=format_action_label(action, locale=locale),
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
        title=get_ui_text("action_distribution_title", locale),
        xaxis_title="",
        yaxis_title=get_ui_text("action_distribution_yaxis", locale),
        height=280,
        showlegend=False,
    )
    max_count = max(action_counts.values()) if action_counts else 1
    fig.update_yaxes(dtick=1, range=[0, max_count * 1.25])
    st.plotly_chart(fig, width="stretch")
