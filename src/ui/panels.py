"""Panel-oriented UI renderers."""
from __future__ import annotations

from contextlib import nullcontext
import math

from src.ui.base import *


STATUS_BADGE_LABELS = {
    "error": "오류",
    "warning": "주의",
    "info": "안내",
    "success": "정상",
}
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
        '<div class="page-shell__meta-eyebrow">시장 컨텍스트</div>'
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
    badge_label = STATUS_BADGE_LABELS.get(tone, "안내")
    title = str(banner.get("title", "")).strip()
    message = str(banner.get("message", "")).strip()
    details = [str(item).strip() for item in banner.get("details", []) if str(item).strip()]
    detail_count = len(details)
    detail_html = (
        f'<span class="status-strip__meta">{detail_count}개 상세</span>'
        if detail_count
        else ""
    )

    st.markdown(
        (
            '<div class="status-strip" '
            f'data-tone="{html.escape(tone)}">'
            f'<div class="status-strip__badge">{html.escape(badge_label)}</div>'
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
        with st.expander("상세 상태", expanded=False):
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


def _overview_optional_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _series_change_pct(series: pd.Series, periods: int = 1) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if isinstance(clean.index, pd.DatetimeIndex):
        clean = clean.sort_index()
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
    reference_index_labels: Sequence[str] | None = None,
    signals: Sequence,
    current_regime: str,
    price_status: str,
    macro_status: str,
    export_growth_val: float | None = None,
    trade_indicators: Mapping[str, float] | None = None,
    has_trade_indicators: bool = False,
) -> list[str]:
    cards: list[str] = []
    preferred = [benchmark_label]
    preferred.extend(str(label).strip() for label in reference_index_labels or [])
    labels: list[str] = []
    for label in preferred:
        if label and label in prices_wide.columns and label not in labels:
            labels.append(label)

    market_card_limit = 1 + len([label for label in reference_index_labels or [] if str(label).strip()])
    for label in labels[:market_card_limit]:
        series = prices_wide[label]
        latest = pd.to_numeric(series, errors="coerce").dropna()
        if isinstance(latest.index, pd.DatetimeIndex):
            latest = latest.sort_index()
        value = latest.iloc[-1] if not latest.empty else None
        change = _series_change_pct(series)
        tone = "positive" if (change or 0.0) >= 0 else "negative"
        cards.append(
            '<div class="overview-market-card overview-market-card--metric">'
            f'<div class="overview-market-card__label">{html.escape(label)}</div>'
            '<div class="overview-market-card__metric-row">'
            f'<div class="overview-market-card__value">{html.escape(_format_overview_number(value))}</div>'
            f'<div class="overview-market-card__change" data-tone="{tone}">1D {html.escape(_format_overview_pct(change))}</div>'
            "</div>"
            "</div>"
        )

    status_items = [
        ("시장 국면", current_regime),
        ("시장 데이터", price_status),
        ("매크로", macro_status),
    ]
    if export_growth_val is not None:
        status_items.append(("수출 전년비", _format_overview_pct(export_growth_val, decimals=1)))
    trade = dict(trade_indicators or {})
    if trade or has_trade_indicators:
        exports_yoy = _safe_float(trade.get("exports_yoy"))
        imports_yoy = _safe_float(trade.get("imports_yoy"))
        pulse_parts: list[str] = []
        if exports_yoy is not None:
            pulse_parts.append(f"수출 {_format_overview_pct(exports_yoy, decimals=1)}")
        if imports_yoy is not None:
            pulse_parts.append(f"수입 {_format_overview_pct(imports_yoy, decimals=1)}")
        status_items.append(("미국 수출입", " / ".join(pulse_parts) if pulse_parts else "데이터 없음"))
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


def _render_sector_trade_lens(
    sector_trade_lens: Sequence[Mapping[str, object]] | None,
    *,
    limit: int = 6,
) -> None:
    source_rows = [dict(row) for row in (sector_trade_lens or [])]
    rows = [
        row
        for row in source_rows
        if _safe_float(row.get("value")) is not None
    ]
    if not rows:
        return

    cards: list[str] = []
    for row in rows[:limit]:
        value = _safe_float(row.get("value"))
        value_text = _format_overview_pct(value, decimals=1) if value is not None else "N/A"
        tone = html.escape(str(row.get("tone") or "neutral"))
        cards.append(
            '<div class="overview-market-card overview-market-card--metric">'
            '<div class="overview-market-card__label">'
            f'{html.escape(str(row.get("sector") or ""))}'
            "</div>"
            '<div class="overview-market-card__metric-row">'
            f'<div class="overview-market-card__value">{html.escape(str(row.get("status") or ""))}</div>'
            f'<div class="overview-market-card__change" data-tone="{tone}">{html.escape(value_text)}</div>'
            "</div>"
            '<div class="overview-sector-subtext">'
            f'{html.escape(str(row.get("exposure_label") or ""))} · '
            f'{html.escape(str(row.get("driver") or ""))} · '
            f'{html.escape(str(row.get("basis") or ""))}'
            "</div>"
            "</div>"
        )

    omitted_count = max(0, len(source_rows) - len(rows))
    omitted_copy = (
        f" · 직접 해석 제한 섹터 {omitted_count}개 제외"
        if omitted_count
        else ""
    )
    st.markdown(
        '<section class="overview-reference-shell">'
        '<div class="overview-section-title">미국 수출입 섹터 렌즈</div>'
        '<div class="overview-command-surface__copy">'
        '총량 proxy입니다. 섹터별 직접 무역 데이터가 아니며 점수 산식에는 반영하지 않습니다.'
        f'{html.escape(omitted_copy)}'
        '</div>'
        f'<div class="overview-market-grid">{"".join(cards)}</div>'
        '</section>',
        unsafe_allow_html=True,
    )


def _overview_taxonomy_lookup(taxonomy_context: Any | None) -> dict[str, object]:
    if taxonomy_context is None:
        return {}
    by_sector_code = getattr(taxonomy_context, "by_sector_code", None)
    if callable(by_sector_code):
        try:
            lookup = by_sector_code()
        except Exception:
            lookup = {}
        if isinstance(lookup, Mapping):
            return {
                str(code).strip(): context
                for code, context in lookup.items()
                if str(code).strip()
            }
    contexts = getattr(taxonomy_context, "sector_contexts", ()) or ()
    return {
        str(getattr(context, "sector_code", "")).strip(): context
        for context in contexts
        if str(getattr(context, "sector_code", "")).strip()
    }


def _overview_text_values(values: object) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    deduped: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _overview_compact_label(labels: Sequence[str], *, limit: int = 2) -> str:
    clean_labels = [label for label in labels if str(label).strip()]
    if not clean_labels:
        return ""
    if len(clean_labels) <= limit:
        return " · ".join(clean_labels)
    return f"{' · '.join(clean_labels[:limit])} 외 {len(clean_labels) - limit}"


def _overview_taxonomy_context_for_signal(signal: object, taxonomy_context: Any | None) -> object | None:
    lookup = _overview_taxonomy_lookup(taxonomy_context)
    sector_code = str(getattr(signal, "index_code", "") or "").strip()
    if sector_code and sector_code in lookup:
        return lookup[sector_code]
    sector_name = str(getattr(signal, "sector_name", "") or "").strip()
    for context in lookup.values():
        if sector_name and sector_name == str(getattr(context, "sector_name", "") or "").strip():
            return context
    return None


def _overview_taxonomy_primary_label(context: object | None, *, fallback: str) -> str:
    if context is None:
        return fallback
    for attr_name in ("theme_labels", "base_labels"):
        label = _overview_compact_label(_overview_text_values(getattr(context, attr_name, ())))
        if label:
            return label
    for attr_name in ("taxonomy_label", "sector_name"):
        label = str(getattr(context, attr_name, "") or "").strip()
        if label:
            return label
    return fallback


def _overview_taxonomy_basis_label(context: object | None) -> str:
    if context is None:
        return ""
    base_label = _overview_compact_label(_overview_text_values(getattr(context, "base_labels", ())))
    cross_label = _overview_compact_label(_overview_text_values(getattr(context, "cross_labels", ())), limit=1)
    parts = []
    if base_label:
        parts.append(f"기본: {base_label}")
    if cross_label:
        parts.append(f"크로스: {cross_label}")
    return " · ".join(parts)


def _overview_taxonomy_display_payload(
    signal: object,
    taxonomy_context: Any | None,
    *,
    fallback_sector_name: str,
) -> dict[str, object]:
    context = _overview_taxonomy_context_for_signal(signal, taxonomy_context)
    if context is None:
        taxonomy_kind = str(getattr(signal, "taxonomy_kind", "") or "").strip().upper()
        taxonomy_label = str(getattr(signal, "taxonomy_label", "") or "").strip()
        if taxonomy_kind or taxonomy_label:
            display_name = taxonomy_label or fallback_sector_name
            subtexts = []
            if taxonomy_kind == "THEME":
                subtexts.append("ETF proxy 테마")
            elif taxonomy_kind:
                subtexts.append(taxonomy_kind)
            runtime_name = fallback_sector_name
            return {
                "display_sector_name": display_name,
                "runtime_sector_name": runtime_name,
                "taxonomy_subtext": " · ".join(subtexts),
                "taxonomy_layer": "theme_lens_proxy",
            }
        return {}
    display_name = _overview_taxonomy_primary_label(context, fallback=fallback_sector_name)
    runtime_name = str(getattr(context, "sector_name", "") or "").strip() or fallback_sector_name
    subtexts = []
    if runtime_name and runtime_name != display_name:
        subtexts.append(f"런타임: {runtime_name}")
    basis = _overview_taxonomy_basis_label(context)
    if basis:
        subtexts.append(basis)
    return {
        "display_sector_name": display_name,
        "runtime_sector_name": runtime_name,
        "taxonomy_subtext": " · ".join(subtexts),
        "taxonomy_layer": "theme_taxonomy",
    }


def _overview_taxonomy_unique_labels(contexts: Sequence[object], attr_name: str) -> list[str]:
    labels: list[str] = []
    for context in contexts:
        for label in _overview_text_values(getattr(context, attr_name, ())):
            if label not in labels:
                labels.append(label)
    return labels


def _overview_taxonomy_context_for_sector_label(
    taxonomy_context: Any | None,
    sector_label: str,
) -> object | None:
    target = str(sector_label or "").strip()
    if not target:
        return None
    for context in _overview_taxonomy_lookup(taxonomy_context).values():
        runtime_name = str(getattr(context, "sector_name", "") or "").strip()
        taxonomy_label = str(getattr(context, "taxonomy_label", "") or "").strip()
        primary_label = _overview_taxonomy_primary_label(context, fallback=runtime_name)
        if target in {runtime_name, taxonomy_label, primary_label}:
            return context
    return None


def _build_overview_taxonomy_map_rows(
    sector_frame: pd.DataFrame,
    taxonomy_context: Any | None,
    *,
    limit: int = 8,
) -> list[dict[str, str]]:
    if taxonomy_context is None or sector_frame.empty:
        return []

    rows: list[dict[str, str]] = []
    for _, row in sector_frame.iterrows():
        visible_sector = _overview_optional_text(row.get("섹터"))
        runtime_sector = _overview_optional_text(row.get("원섹터")) or visible_sector
        context = _overview_taxonomy_context_for_sector_label(taxonomy_context, runtime_sector)
        if context is None:
            context = _overview_taxonomy_context_for_sector_label(taxonomy_context, visible_sector)
        if context is None:
            continue
        display_sector = _overview_taxonomy_primary_label(context, fallback=runtime_sector or visible_sector)
        rows.append(
            {
                "sector": display_sector,
                "runtime": str(getattr(context, "sector_name", "") or runtime_sector),
                "base": _overview_compact_label(_overview_text_values(getattr(context, "base_labels", ())), limit=2) or "미지정",
                "cross": _overview_compact_label(_overview_text_values(getattr(context, "cross_labels", ())), limit=2) or "없음",
                "theme": _overview_compact_label(_overview_text_values(getattr(context, "theme_labels", ())), limit=2) or "없음",
                "action": _overview_optional_text(row.get("액션")) or "N/A",
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _render_overview_taxonomy_surface(
    taxonomy_context: Any | None,
    sector_frame: pd.DataFrame,
    *,
    selected_layer: str,
    market_id: str,
) -> None:
    """Render a first-screen taxonomy summary for the overview page."""
    if taxonomy_context is None:
        return

    contexts = list(getattr(taxonomy_context, "sector_contexts", ()) or ())
    diagnostics = list(getattr(taxonomy_context, "diagnostics", ()) or ())
    coverage_label = "완료" if not diagnostics else f"확인 필요 {len(diagnostics)}건"
    version = str(getattr(taxonomy_context, "taxonomy_version", "") or "")
    base_labels = _overview_taxonomy_unique_labels(contexts, "base_labels")
    cross_labels = _overview_taxonomy_unique_labels(contexts, "cross_labels")
    theme_labels = _overview_taxonomy_unique_labels(contexts, "theme_labels")
    stat_items = [
        ("시장", str(market_id or "KR")),
        ("버전", f"v{version}" if version else "N/A"),
        ("커버리지", coverage_label),
        ("기본산업", f"{len(base_labels)}개"),
        ("크로스테마", f"{len(cross_labels)}개"),
        ("상품테마", f"{len(theme_labels)}개"),
        ("표시 레이어", selected_layer),
    ]
    stats_html = "".join(
        (
            '<div class="overview-taxonomy-stat">'
            f"<span>{html.escape(label)}</span>"
            f"<strong>{html.escape(value)}</strong>"
            "</div>"
        )
        for label, value in stat_items
    )

    map_rows = _build_overview_taxonomy_map_rows(sector_frame, taxonomy_context, limit=8)
    if map_rows:
        row_html = "".join(
            (
                '<div class="overview-taxonomy-row">'
                '<div class="overview-taxonomy-row__sector">'
                f"<strong>{html.escape(row['sector'])}</strong>"
                f"<span>런타임: {html.escape(row['runtime'])}</span>"
                "</div>"
                '<div class="overview-taxonomy-row__cell">'
                "<span>기본</span>"
                f"<strong>{html.escape(row['base'])}</strong>"
                "</div>"
                '<div class="overview-taxonomy-row__cell">'
                "<span>크로스</span>"
                f"<strong>{html.escape(row['cross'])}</strong>"
                "</div>"
                '<div class="overview-taxonomy-row__cell">'
                "<span>상품</span>"
                f"<strong>{html.escape(row['theme'])}</strong>"
                "</div>"
                '<div class="overview-taxonomy-row__action">'
                f"<strong>{html.escape(row['action'])}</strong>"
                "</div>"
                "</div>"
            )
            for row in map_rows
        )
    else:
        row_html = (
            '<div class="overview-taxonomy-empty">'
            "현재 필터 기준으로 표시할 taxonomy 매핑 행이 없습니다."
            "</div>"
        )

    diagnostics_html = (
        '<div class="overview-taxonomy-diagnostics">'
        f"{html.escape(' / '.join(str(item) for item in diagnostics[:3]))}"
        "</div>"
        if diagnostics
        else ""
    )

    st.markdown(
        (
            '<section class="overview-taxonomy-surface" '
            f'data-coverage="{html.escape("complete" if not diagnostics else "attention")}">'
            '<div class="overview-taxonomy-surface__header">'
            "<div>"
            '<div class="overview-taxonomy-surface__eyebrow">KR TAXONOMY WORKBENCH</div>'
            '<div class="overview-taxonomy-surface__title">Theme Taxonomy</div>'
            '<div class="overview-taxonomy-surface__copy">'
            "기본산업, 크로스테마, 상품테마 축으로 기존 섹터 신호를 먼저 정렬합니다."
            "</div>"
            "</div>"
            f'<div class="overview-taxonomy-stats">{stats_html}</div>'
            "</div>"
            f"{diagnostics_html}"
            '<div class="overview-taxonomy-map">'
            '<div class="overview-taxonomy-map__header">'
            "<span>분류축 지도</span>"
            f"<strong>상위 {len(map_rows)}개 신호</strong>"
            "</div>"
            f"{row_html}"
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def _render_overview_market_command_surface(cards: Sequence[str]) -> None:
    st.markdown(
        '<section class="overview-reference-shell overview-command-surface">'
        '<div class="overview-command-surface__header">'
        '<div>'
        '<div class="overview-section-title">시장/조회</div>'
        '<div class="overview-command-surface__copy">국면, 데이터 상태, 종목 조회 조건을 한곳에서 확인합니다.</div>'
        '</div>'
        f'<div class="overview-market-grid">{"".join(cards)}</div>'
        '</div>'
        '</section>',
        unsafe_allow_html=True,
    )


def _build_overview_sector_frame(
    signals: Sequence,
    *,
    sort_key: str,
    sector_export_trends: Mapping[str, float] | None = None,
    has_sector_export_indicators: bool = True,
    taxonomy_context: Any | None = None,
    use_taxonomy_display: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    export_trends = {
        str(sector_name): value
        for sector_name, value in (sector_export_trends or {}).items()
    } if has_sector_export_indicators else {}
    for rank, signal in enumerate(sorted(signals, key=signal_display_sort_key), start=1):
        if str(getattr(signal, "action", "N/A")) == "N/A":
            continue
        sector_name = str(getattr(signal, "sector_name", ""))
        rs_gap = _rs_divergence_pct(signal)
        ret_1m = _pct_value(getattr(signal, "returns", {}).get("1M"))
        ret_3m = _pct_value(getattr(signal, "returns", {}).get("3M"))
        mom_score = _safe_float(getattr(signal, "mom_percentile", None))
        if mom_score is None:
            mom_score = rs_gap
        export_basis = _sector_export_display_label(sector_name) if sector_name in export_trends else ""
        taxonomy_item = (
            _overview_taxonomy_context_for_signal(signal, taxonomy_context)
            if use_taxonomy_display
            else None
        )
        display_sector_name = (
            _overview_taxonomy_primary_label(taxonomy_item, fallback=sector_name)
            if taxonomy_item is not None
            else sector_name
        )
        row: dict[str, object] = {
            "순위": rank,
            "섹터": display_sector_name,
            "모멘텀 점수": mom_score,
            "상대강도": rs_gap,
            "1M": ret_1m,
            "3M": ret_3m,
            "액션": format_action_label(str(getattr(signal, "action", "N/A"))),
        }
        if taxonomy_item is not None:
            row["원섹터"] = sector_name
            taxonomy_basis = _overview_taxonomy_basis_label(taxonomy_item)
            if taxonomy_basis:
                row["분류 근거"] = taxonomy_basis
        if has_sector_export_indicators:
            row["수출 기준"] = export_basis
        rows.append(row)
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


_SECTOR_EXPORT_DISPLAY_LABELS: Mapping[str, str] = {
    "KOSPI200 정보기술": "IT 수출",
    "KRX 반도체": "반도체 수출",
    "KRX 에너지화학": "화학제품 수출",
    "KRX 철강": "철강 수출",
    "KOSPI200 경기소비재": "자동차 수출",
    "KRX 산업재": "기계·장비 수출",
    "KRX 헬스케어": "의약품 수출",
}


def _sector_export_display_label(sector_name: str) -> str:
    return _SECTOR_EXPORT_DISPLAY_LABELS.get(str(sector_name), "")


def _format_export_trace_name(sector_name: str) -> str:
    basis = _sector_export_display_label(sector_name)
    return basis or sector_name


def _render_overview_sector_table(frame: pd.DataFrame, *, has_sector_export_indicators: bool = True) -> None:
    if frame.empty:
        st.info("표시할 섹터 신호가 없습니다.")
        return
    rows: list[str] = []
    for _, row in frame.head(12).iterrows():
        ret_3m = _safe_float(row.get("3M"))
        ret_tone = "positive" if (ret_3m or 0.0) >= 0 else "negative"
        sector_name = str(row["섹터"])
        original_sector_name = _overview_optional_text(row.get("원섹터"))
        taxonomy_basis = _overview_optional_text(row.get("분류 근거"))
        export_basis = _overview_optional_text(row.get("수출 기준"))
        sector_cell = html.escape(sector_name)
        subtexts = []
        if original_sector_name and original_sector_name != sector_name:
            subtexts.append(f"런타임: {original_sector_name}")
        if taxonomy_basis:
            subtexts.append(taxonomy_basis)
        if has_sector_export_indicators and export_basis:
            subtexts.append(f"수출 기준: {export_basis}")
        if subtexts:
            sector_cell += f'<span class="overview-sector-subtext">{html.escape(" · ".join(subtexts))}</span>'
        rows.append(
            "<tr>"
            f"<td>{int(row['순위'])}</td>"
            f"<td>{sector_cell}</td>"
            f"<td>{html.escape(_format_overview_number(row.get('모멘텀 점수'), decimals=2))}</td>"
            f"<td>{html.escape(_format_overview_number(row.get('상대강도'), decimals=2))}</td>"
            f'<td data-tone="{ret_tone}">{html.escape(_format_overview_pct(row.get("3M"), decimals=2))}</td>'
            "</tr>"
        )
    st.markdown(
        '<div class="overview-sector-table-wrap">'
        '<table class="overview-sector-table">'
        "<colgroup>"
        '<col class="overview-sector-table__rank">'
        '<col class="overview-sector-table__sector">'
        '<col class="overview-sector-table__momentum">'
        '<col class="overview-sector-table__strength">'
        '<col class="overview-sector-table__return">'
        "</colgroup>"
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
    chart_tokens = get_chart_tokens(theme_mode)
    colorway = list(chart_tokens["colorway"])
    fig = go.Figure()
    if prices_wide.empty:
        fig.update_layout(**template, title="섹터 상대강도 추이")
        return fig

    days = _period_to_days(period)
    visible = prices_wide.tail(days).copy()
    candidates = [benchmark_label]
    candidates.extend(str(getattr(signal, "sector_name", "")) for signal in sorted(signals, key=signal_display_sort_key)[:5])
    line_end_annotations: list[dict[str, object]] = []
    for trace_index, label in enumerate(dict.fromkeys(item for item in candidates if item in visible.columns)):
        series = pd.to_numeric(visible[label], errors="coerce").dropna()
        if series.empty:
            continue
        base = float(series.iloc[0])
        if base == 0:
            continue
        normalized = series / base
        color = str(colorway[trace_index % len(colorway)])
        line_color = str(chart_tokens["muted_lines"][0]) if label == benchmark_label else color
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized.values,
                mode="lines",
                name=label,
                line={
                    "color": line_color,
                    "width": 1.35 if label == benchmark_label else 2.15,
                    "dash": "dot" if label == benchmark_label else "solid",
                },
                opacity=0.68 if label == benchmark_label else 0.96,
                hovertemplate="%{fullData.name}<br>%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>",
            )
        )
        line_end_annotations.append(
            {
                "x": normalized.index[-1],
                "y": float(normalized.iloc[-1]),
                "text": f"{label} {normalized.iloc[-1]:.2f}",
                "xref": "x",
                "yref": "y",
                "showarrow": False,
                "xanchor": "left",
                "xshift": 8,
                "font": {"size": 11, "color": line_color},
                "bgcolor": chart_tokens["legend_bg"],
                "bordercolor": line_color,
                "borderwidth": 1,
                "borderpad": 2,
            }
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
    _stagger_line_end_annotations(line_end_annotations)
    fig.update_layout(**template)
    fig.update_layout(
        title="섹터 상대강도 추이 (vs 기준값 1.0)",
        height=348,
        margin={"l": 34, "r": 138, "t": 42, "b": 68},
        hovermode="x unified",
        legend={"orientation": "h", "x": 0.0, "y": -0.24, "xanchor": "left"},
        annotations=line_end_annotations,
    )
    fig.add_hline(y=1.0, line_width=1, line_dash="dot", line_color=chart_tokens["axis"])
    fig.update_yaxes(tickformat=".2f", title="", fixedrange=True)
    fig.update_xaxes(
        title="",
        tickformat="%b\n%Y",
        dtick="M1",
        showspikes=True,
        spikemode="across",
        spikecolor=chart_tokens["axis"],
        spikethickness=1,
    )
    return fig


def _stagger_line_end_annotations(annotations: list[dict[str, object]]) -> None:
    numeric_annotations: list[tuple[float, dict[str, object]]] = []
    for annotation in annotations:
        try:
            numeric_annotations.append((float(annotation["y"]), annotation))
        except (KeyError, TypeError, ValueError):
            continue
    if len(numeric_annotations) < 2:
        return

    values = [value for value, _ in numeric_annotations]
    threshold = max((max(values) - min(values)) * 0.045, 0.025)
    offsets = [0, -12, 12, -24, 24, -36, 36]
    cluster: list[dict[str, object]] = []
    previous_value: float | None = None

    for value, annotation in sorted(numeric_annotations, key=lambda item: item[0], reverse=True):
        if previous_value is None or abs(value - previous_value) <= threshold:
            cluster.append(annotation)
        else:
            for index, clustered_annotation in enumerate(cluster):
                clustered_annotation["yshift"] = offsets[index % len(offsets)]
            cluster = [annotation]
        previous_value = value

    for index, clustered_annotation in enumerate(cluster):
        clustered_annotation["yshift"] = offsets[index % len(offsets)]


def _to_month_timestamp(value: object) -> pd.Timestamp | object:
    try:
        if isinstance(value, pd.Period):
            return value.to_timestamp()
        return pd.Period(value, freq="M").to_timestamp()
    except Exception:
        return value


def _build_sector_export_trend_figure(
    *,
    sector_export_history: Mapping[str, pd.Series] | None,
    signals: Sequence,
    theme_mode: str,
    window_months: int = 18,
) -> go.Figure:
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)
    chart_tokens = get_chart_tokens(theme_mode)
    colorway = list(chart_tokens["colorway"])
    fig = go.Figure()
    history = sector_export_history or {}
    if not history:
        fig.update_layout(**template, title="섹터별 수출 YoY 월별 추이")
        fig.add_annotation(
            text="표시할 섹터별 수출 시계열이 없습니다.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": tokens["text"]},
        )
        return fig

    ordered_names = [
        str(getattr(signal, "sector_name", "")).strip()
        for signal in sorted(signals, key=signal_display_sort_key)
        if str(getattr(signal, "sector_name", "")).strip() in history
    ]
    ordered_names.extend(sector_name for sector_name in history if sector_name not in ordered_names)

    line_end_annotations: list[dict[str, object]] = []
    month_count = max(1, int(window_months))
    x_axis_dtick = "M2" if month_count > 12 else "M1"
    for trace_index, sector_name in enumerate(dict.fromkeys(ordered_names).keys()):
        series = pd.to_numeric(history.get(sector_name), errors="coerce").dropna()
        if series.empty:
            continue
        visible = series.tail(month_count)
        x_values = [_to_month_timestamp(idx) for idx in visible.index]
        month_labels = [pd.Timestamp(value).strftime("%Y-%m") if isinstance(value, pd.Timestamp) else str(value) for value in x_values]
        color = str(colorway[trace_index % len(colorway)])
        trace_name = _format_export_trace_name(sector_name)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=visible.values,
                mode="lines+markers",
                name=trace_name,
                line={"color": color, "width": 1.9},
                marker={"color": color, "size": 4.2, "line": {"color": chart_tokens["background"], "width": 0.8}},
                customdata=[[sector_name, month_label] for month_label in month_labels],
                hovertemplate="%{fullData.name}<br>%{customdata[0]}<br>%{customdata[1]}<br>수출 YoY %{y:.1f}%<extra></extra>",
            )
        )
        line_end_annotations.append(
            {
                "x": x_values[-1],
                "y": float(visible.iloc[-1]),
                "text": f"{trace_name} {visible.iloc[-1]:.0f}%",
                "xref": "x",
                "yref": "y",
                "showarrow": False,
                "xanchor": "left",
                "xshift": 8,
                "font": {"size": 11, "color": color},
                "bgcolor": chart_tokens["legend_bg"],
                "bordercolor": color,
                "borderwidth": 1,
                "borderpad": 2,
            }
        )

    if not fig.data:
        fig.add_annotation(
            text="표시할 섹터별 수출 시계열이 없습니다.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": tokens["text"]},
        )

    _stagger_line_end_annotations(line_end_annotations)
    fig.update_layout(**template)
    fig.update_layout(
        title="섹터별 수출 YoY 월별 추이",
        height=360,
        margin={"l": 42, "r": 150, "t": 42, "b": 66},
        hovermode="x unified",
        legend={"orientation": "h", "x": 0.0, "y": -0.22, "xanchor": "left"},
        annotations=line_end_annotations,
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=chart_tokens["axis"])
    fig.update_yaxes(title="", ticksuffix="%", fixedrange=True)
    fig.update_xaxes(
        title="",
        tickformat="%b\n%Y",
        dtick=x_axis_dtick,
        showspikes=True,
        spikemode="across",
        spikecolor=chart_tokens["axis"],
        spikethickness=1,
    )
    return fig


def _render_overview_mobile_decision_strip(frame: pd.DataFrame) -> None:
    """Render a compact mobile-only top-sector strip before secondary controls."""
    candidates = [
        {
            "sector_name": str(row.get("섹터", "")),
            "decision": "검토 후보",
            "reason_parts": [],
            "invalidation": "",
            "action": str(row.get("액션", "")),
            "metrics": [
                ("3M", _format_overview_pct(row.get("3M"))),
            ],
        }
        for _, row in frame.head(3).iterrows()
        if str(row.get("섹터", "")).strip()
    ]
    _render_overview_review_candidates(candidates, compact=True)


_COMPOSITE_REVIEW_CANDIDATE_POLICY = "COMPOSITE_REVIEW_CANDIDATE"
_OVERVIEW_REVIEW_GROUPS = ("buy", "sell")
_OVERVIEW_REVIEW_GROUP_LABELS = {
    "buy": "매수 검토 후보",
    "sell": "매도 검토 후보",
}
_OVERVIEW_REVIEW_GROUP_DESCRIPTIONS = {
    "buy": "상방 proxy와 복합점수가 우세한 신규/증액 검토 대상입니다.",
    "sell": "보유 여부와 무관하게 하방 proxy나 약화 근거가 우세한 축소/매도 검토 대상입니다.",
}
_FLOW_STATES_WITH_SCORE = {"supportive", "neutral", "adverse"}


def _overview_candidate_clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))


def _overview_candidate_is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _overview_candidate_momentum_score(signal: object) -> float:
    mom_percentile = getattr(signal, "mom_percentile", None)
    if _overview_candidate_is_finite(mom_percentile):
        return _overview_candidate_clamp(float(mom_percentile))

    mom_score = getattr(signal, "mom_score", None)
    if _overview_candidate_is_finite(mom_score):
        mom_score_float = float(mom_score)
        if 0.0 <= mom_score_float <= 1.0:
            return _overview_candidate_clamp(mom_score_float * 100.0)

    boolean_scores = [
        100.0 if bool(getattr(signal, "momentum_strong", False)) else 0.0,
        100.0 if bool(getattr(signal, "trend_ok", False)) else 0.0,
        100.0 if bool(getattr(signal, "rs_strong", False)) else 0.0,
    ]
    return sum(boolean_scores) / len(boolean_scores)


def _overview_candidate_macro_score(signal: object) -> tuple[float, bool, list[str]]:
    warnings: list[str] = []
    macro_fit = bool(getattr(signal, "macro_fit", False))
    macro_fit_score = 60.0 if macro_fit else 35.0
    if not macro_fit:
        warnings.append("매크로 약점")

    rank = getattr(signal, "sector_fit_rank", None)
    total = getattr(signal, "sector_fit_total", None)
    if _overview_candidate_is_finite(rank) and _overview_candidate_is_finite(total):
        rank_float = float(rank)
        total_float = float(total)
        if total_float > 1.0:
            sector_fit_score = _overview_candidate_clamp(
                100.0 * (total_float - rank_float) / (total_float - 1.0)
            )
            return (0.5 * macro_fit_score) + (0.5 * sector_fit_score), True, warnings

    warnings.append("실증 적합도 없음")
    return macro_fit_score, False, warnings


def _overview_candidate_flow_score(signal: object) -> tuple[float, bool, list[str]]:
    warnings: list[str] = []
    flow_state = str(getattr(signal, "flow_state", "") or "").strip().lower()
    raw_flow_score = getattr(signal, "flow_score", None)
    if flow_state not in _FLOW_STATES_WITH_SCORE:
        warnings.append("수급 신호 없음")
        return 50.0, False, warnings
    if not _overview_candidate_is_finite(raw_flow_score):
        warnings.append("수급 점수 중립")
        return 50.0, True, warnings

    flow_score = _overview_candidate_clamp(50.0 + _overview_candidate_clamp(float(raw_flow_score), -2.0, 2.0) * 25.0)
    if flow_state == "adverse":
        warnings.append("수급 약점")
    return flow_score, True, warnings


def _overview_candidate_neutral_score(value: object, neutral: float = 50.0) -> float:
    if not _overview_candidate_is_finite(value):
        return neutral
    return _overview_candidate_clamp(float(value))


def _overview_candidate_rs_gap_score(signal: object) -> float:
    rs_gap = _rs_divergence_pct(signal)
    if not _overview_candidate_is_finite(rs_gap):
        return 50.0
    return _overview_candidate_clamp(50.0 + _overview_candidate_clamp(float(rs_gap), -10.0, 10.0) * 5.0)


def _overview_candidate_rs_change_score(signal: object) -> float:
    rs_change = getattr(signal, "rs_change_pct", None)
    if not _overview_candidate_is_finite(rs_change):
        return 50.0
    return _overview_candidate_clamp(50.0 + _overview_candidate_clamp(float(rs_change), -10.0, 10.0) * 5.0)


def _overview_candidate_risk_alert_score(signal: object) -> float:
    score = 35.0
    alerts = {str(item) for item in getattr(signal, "alerts", []) or []}
    if "Overheat" in alerts:
        score += 15.0
    if "FX Shock" in alerts:
        score += 10.0

    volatility = getattr(signal, "volatility_20d", None)
    if _overview_candidate_is_finite(volatility):
        vol = float(volatility)
        if vol >= 0.30:
            score += 15.0
        elif vol >= 0.20:
            score += 10.0

    mdd_3m = getattr(signal, "mdd_3m", None)
    if _overview_candidate_is_finite(mdd_3m):
        mdd = float(mdd_3m)
        if mdd <= -0.20:
            score += 20.0
        elif mdd <= -0.12:
            score += 10.0

    return _overview_candidate_clamp(score)


def _overview_candidate_flow_risk(signal: object) -> float:
    flow_state = str(getattr(signal, "flow_state", "") or "").strip().lower()
    if flow_state == "supportive":
        return 25.0
    if flow_state == "adverse":
        return 75.0
    return 50.0


def _overview_candidate_proxy_bundle(
    signal: object,
    *,
    momentum_score: float,
    macro_score: float,
    flow_score: float,
) -> dict[str, object]:
    momentum = _overview_candidate_neutral_score(momentum_score)
    macro = _overview_candidate_neutral_score(macro_score)
    flow = _overview_candidate_neutral_score(flow_score)
    rs_gap_score = _overview_candidate_rs_gap_score(signal)
    rs_change_score = _overview_candidate_rs_change_score(signal)
    trend_ok = bool(getattr(signal, "trend_ok", False))
    trend_score = 100.0 if trend_ok else 25.0
    trend_risk = 20.0 if trend_ok else 80.0
    flow_risk = _overview_candidate_flow_risk(signal)
    risk_alert_score = _overview_candidate_risk_alert_score(signal)

    upside_proxy = (
        (0.35 * momentum)
        + (0.20 * rs_gap_score)
        + (0.15 * rs_change_score)
        + (0.15 * trend_score)
        + (0.10 * macro)
        + (0.05 * flow)
    )
    downside_proxy = (
        (0.30 * (100.0 - momentum))
        + (0.20 * (100.0 - rs_gap_score))
        + (0.15 * (100.0 - rs_change_score))
        + (0.15 * trend_risk)
        + (0.10 * (100.0 - macro))
        + (0.05 * flow_risk)
        + (0.05 * risk_alert_score)
    )
    edge_proxy = upside_proxy - downside_proxy
    flow_state = str(getattr(signal, "flow_state", "") or "").strip().lower()

    if edge_proxy >= 25.0 and trend_ok and momentum >= 65.0:
        turning_point_state = "Continuation up"
    elif edge_proxy >= 10.0 and rs_change_score >= 65.0 and rs_gap_score >= 45.0:
        turning_point_state = "Bullish turn"
    elif edge_proxy <= -10.0 and (rs_change_score <= 35.0 or rs_gap_score <= 40.0 or flow_state == "adverse"):
        turning_point_state = "Bearish turn"
    elif edge_proxy <= -25.0 and not trend_ok and momentum <= 40.0:
        turning_point_state = "Continuation down"
    else:
        turning_point_state = "Neutral"

    bullish_evidence: list[str] = []
    bearish_evidence: list[str] = []
    if momentum >= 65.0:
        bullish_evidence.append("모멘텀 우위")
    elif momentum <= 40.0:
        bearish_evidence.append("모멘텀 약화")
    if rs_gap_score >= 60.0:
        bullish_evidence.append("RS 상방")
    elif rs_gap_score <= 40.0:
        bearish_evidence.append("RS 하방")
    if rs_change_score >= 65.0:
        bullish_evidence.append("RS 개선")
    elif rs_change_score <= 35.0:
        bearish_evidence.append("RS 둔화")
    if trend_ok:
        bullish_evidence.append("추세 확인")
    else:
        bearish_evidence.append("추세 훼손")
    if macro >= 60.0:
        bullish_evidence.append("국면 근거")
    elif macro <= 40.0:
        bearish_evidence.append("국면 약점")
    if flow_state == "supportive":
        bullish_evidence.append("수급 보강")
    elif flow_state == "adverse":
        bearish_evidence.append("수급 역풍")
    if risk_alert_score >= 60.0:
        bearish_evidence.append("리스크 확대")

    return {
        "upside_proxy": _overview_candidate_clamp(upside_proxy),
        "downside_proxy": _overview_candidate_clamp(downside_proxy),
        "edge_proxy": edge_proxy,
        "turning_point_state": turning_point_state,
        "bullish_evidence": bullish_evidence[:4],
        "bearish_evidence": bearish_evidence[:4],
        "rs_gap_score": rs_gap_score,
        "rs_change_score": rs_change_score,
        "risk_alert_score": risk_alert_score,
    }


def _overview_candidate_type(
    *,
    momentum_score: float,
    macro_score: float,
    flow_score: float,
    warnings: Sequence[str],
) -> str:
    if any("약점" in warning for warning in warnings):
        return "충돌 신호"
    if flow_score >= max(momentum_score, macro_score) and flow_score >= 65.0:
        return "수급 보강"
    if macro_score >= momentum_score:
        return "매크로 주도"
    return "모멘텀 주도"


def _format_overview_candidate_score(value: object) -> str:
    score = _safe_float(value)
    if score is None or not math.isfinite(score):
        return "N/A"
    return f"{score:.1f}"


def _format_overview_candidate_edge(value: object) -> str:
    score = _safe_float(value)
    if score is None or not math.isfinite(score):
        return "N/A"
    return f"{score:+.1f}"


def _build_overview_review_candidate_projection(
    signal: object,
    *,
    taxonomy_context: Any | None = None,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> dict[str, object] | None:
    if str(getattr(signal, "action", "N/A")) == "N/A":
        return None

    momentum_score = _overview_candidate_momentum_score(signal)
    macro_score, macro_available, macro_warnings = _overview_candidate_macro_score(signal)
    flow_score, flow_available, flow_warnings = _overview_candidate_flow_score(signal)
    candidate_score = (0.45 * momentum_score) + (0.35 * macro_score) + (0.20 * flow_score)
    warnings = list(dict.fromkeys([*macro_warnings, *flow_warnings]))
    candidate_type = _overview_candidate_type(
        momentum_score=momentum_score,
        macro_score=macro_score,
        flow_score=flow_score,
        warnings=warnings,
    )
    proxy_bundle = _overview_candidate_proxy_bundle(
        signal,
        momentum_score=momentum_score,
        macro_score=macro_score,
        flow_score=flow_score,
    )

    thesis = describe_signal_decision(signal, [], locale=locale)
    edge_proxy = float(proxy_bundle["edge_proxy"])
    reason_parts = [
        f"변곡 {proxy_bundle['turning_point_state']}",
        f"엣지 {_format_overview_candidate_edge(edge_proxy)}",
    ]
    evidence = list(proxy_bundle["bullish_evidence"] if edge_proxy >= 0 else proxy_bundle["bearish_evidence"])
    reason_parts.extend(str(item) for item in evidence[:2])
    if len(reason_parts) < 3:
        reason_parts.extend(_decision_card_parts(thesis.get("reason"), limit=1))
    if len(reason_parts) < 3:
        reason_parts.extend(warnings[: 3 - len(reason_parts)])
    metrics = [
        ("상방 proxy", _format_overview_candidate_score(proxy_bundle["upside_proxy"])),
        ("하방 proxy", _format_overview_candidate_score(proxy_bundle["downside_proxy"])),
        ("엣지 proxy", _format_overview_candidate_edge(edge_proxy)),
        ("변곡", str(proxy_bundle["turning_point_state"])),
        ("복합점수", _format_overview_candidate_score(candidate_score)),
    ]

    sector_name = str(getattr(signal, "sector_name", "")).strip()
    candidate: dict[str, object] = {
        "sector_name": sector_name,
        "decision": str(thesis.get("decision", "")),
        "reason_parts": reason_parts,
        "invalidation": str(thesis.get("invalidation", "")),
        "action": str(getattr(signal, "action", "N/A")),
        "action_policy": str(getattr(signal, "action_policy", "")),
        "candidate_policy": _COMPOSITE_REVIEW_CANDIDATE_POLICY,
        "candidate_score": candidate_score,
        "momentum_score": momentum_score,
        "macro_score": macro_score,
        "flow_score": flow_score,
        "macro_available": macro_available,
        "flow_available": flow_available,
        "candidate_type": candidate_type,
        "warnings": warnings,
        "upside_proxy": proxy_bundle["upside_proxy"],
        "downside_proxy": proxy_bundle["downside_proxy"],
        "edge_proxy": edge_proxy,
        "turning_point_state": proxy_bundle["turning_point_state"],
        "bullish_evidence": proxy_bundle["bullish_evidence"],
        "bearish_evidence": proxy_bundle["bearish_evidence"],
        "metrics": metrics,
    }
    candidate.update(
        _overview_taxonomy_display_payload(
            signal,
            taxonomy_context,
            fallback_sector_name=sector_name,
        )
    )
    return candidate


def _build_overview_review_candidates(
    signals: Sequence,
    sector_frame: pd.DataFrame,
    *,
    limit: int = 3,
    taxonomy_context: Any | None = None,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> list[dict[str, object]]:
    """Build first-screen review candidates from composite signal projections."""
    if sector_frame.empty:
        return []

    sectors_in_frame = {
        str(row.get("원섹터", row.get("섹터", ""))).strip()
        for _, row in sector_frame.iterrows()
        if str(row.get("원섹터", row.get("섹터", ""))).strip()
    }
    candidates: list[dict[str, object]] = []
    for signal in signals:
        sector_name = str(getattr(signal, "sector_name", "")).strip()
        if sector_name not in sectors_in_frame:
            continue
        candidate = _build_overview_review_candidate_projection(
            signal,
            taxonomy_context=taxonomy_context,
            locale=locale,
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(
        key=lambda candidate: (
            -float(candidate.get("candidate_score", 0.0)),
            -float(candidate.get("momentum_score", 0.0)),
            str(candidate.get("sector_name", "")),
        )
    )
    return candidates[:limit]


def _overview_review_candidate_group(candidate: Mapping[str, object]) -> str | None:
    action = str(candidate.get("action", "") or "")
    edge_proxy = _sector_momentum_safe_float(candidate.get("edge_proxy"), default=0.0)
    turning_point_state = str(candidate.get("turning_point_state", "") or "")
    if action == "Strong Buy":
        return "buy"
    if action == "Avoid":
        return "sell"
    if edge_proxy < 0.0 or turning_point_state == "Bearish turn":
        return "sell"
    if edge_proxy > 0.0 or turning_point_state in {"Bullish turn", "Continuation up"}:
        return "buy"
    return None


def _build_overview_theme_proxy_frame(signals: Sequence, *, limit: int = 8) -> pd.DataFrame:
    """Return a compact first-screen table for theme ETF proxy signals."""
    rows: list[dict[str, object]] = []
    for signal in signals:
        taxonomy_kind = str(getattr(signal, "taxonomy_kind", "") or "").strip().upper()
        if taxonomy_kind != "THEME":
            continue
        returns = dict(getattr(signal, "returns", {}) or {})
        rows.append(
            {
                "테마": str(getattr(signal, "taxonomy_label", "") or getattr(signal, "sector_name", "") or "").strip(),
                "액션": str(getattr(signal, "action", "N/A") or "N/A"),
                "모멘텀 점수": _overview_candidate_momentum_score(signal),
                "6M/12M 상대": _sector_momentum_safe_float(getattr(signal, "mom_raw", pd.NA), default=float("nan")),
                "1M": _sector_momentum_safe_float(returns.get("1M", pd.NA), default=float("nan")) * 100.0,
                "3M": _sector_momentum_safe_float(returns.get("3M", pd.NA), default=float("nan")) * 100.0,
                "추세": "통과" if bool(getattr(signal, "trend_ok", False)) else "미통과",
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        by=["모멘텀 점수", "6M/12M 상대", "테마"],
        ascending=[False, False, True],
        na_position="last",
    )
    return frame.head(max(1, int(limit))).reset_index(drop=True)


def _render_overview_theme_proxy_watchlist(signals: Sequence, *, is_mobile: bool = False) -> None:
    frame = _build_overview_theme_proxy_frame(signals, limit=7 if not is_mobile else 5)
    if frame.empty:
        return
    render_panel_header(
        eyebrow="Theme Proxy",
        title="테마 proxy 모니터",
        description="대표 ETF 가격 기반 테마 신호입니다. 로봇, 전력, 원자력, 우주항공처럼 KRX broad sector가 아닌 테마도 첫 화면에서 확인합니다.",
    )
    st.caption("이 표는 theme_lens ETF proxy를 신호 원장과 같은 기준으로 정렬한 모니터입니다. 개별 종목 추천이나 주문 신호가 아닙니다.")
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        height=min(330, 80 + len(frame) * 36),
        column_config={
            "테마": st.column_config.TextColumn("테마", width="medium"),
            "액션": st.column_config.TextColumn("액션", width="small"),
            "모멘텀 점수": st.column_config.NumberColumn("모멘텀 점수", format="%.1f"),
            "6M/12M 상대": st.column_config.NumberColumn("6M/12M 상대", format="%.3f"),
            "1M": st.column_config.NumberColumn("1M", format="%.1f%%"),
            "3M": st.column_config.NumberColumn("3M", format="%.1f%%"),
            "추세": st.column_config.TextColumn("추세", width="small"),
        },
    )


def _overview_review_group_sort_key(group_key: str, candidate: Mapping[str, object]) -> tuple[object, ...]:
    sector_name = str(candidate.get("sector_name", ""))
    momentum_score = _sector_momentum_safe_float(candidate.get("momentum_score"), default=0.0)
    candidate_score = _sector_momentum_safe_float(candidate.get("candidate_score"), default=0.0)
    edge_proxy = _sector_momentum_safe_float(candidate.get("edge_proxy"), default=0.0)
    downside_proxy = _sector_momentum_safe_float(candidate.get("downside_proxy"), default=0.0)
    if group_key == "sell":
        return (-downside_proxy, edge_proxy, -momentum_score, sector_name)
    return (-candidate_score, -edge_proxy, -momentum_score, sector_name)


def _build_overview_review_candidate_groups(
    signals: Sequence,
    sector_frame: pd.DataFrame,
    *,
    limit_per_group: int = 3,
    taxonomy_context: Any | None = None,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> dict[str, list[dict[str, object]]]:
    """Build directional first-screen review candidates from composite projections."""
    groups: dict[str, list[dict[str, object]]] = {key: [] for key in _OVERVIEW_REVIEW_GROUPS}
    if sector_frame.empty:
        return groups

    sectors_in_frame = {
        str(row.get("원섹터", row.get("섹터", ""))).strip()
        for _, row in sector_frame.iterrows()
        if str(row.get("원섹터", row.get("섹터", ""))).strip()
    }
    for signal in signals:
        sector_name = str(getattr(signal, "sector_name", "")).strip()
        if sector_name not in sectors_in_frame:
            continue
        candidate = _build_overview_review_candidate_projection(
            signal,
            taxonomy_context=taxonomy_context,
            locale=locale,
        )
        if candidate is None:
            continue
        group_key = _overview_review_candidate_group(candidate)
        if group_key is None:
            continue
        candidate = dict(candidate)
        candidate["review_side"] = group_key
        candidate["review_side_label"] = _OVERVIEW_REVIEW_GROUP_LABELS[group_key]
        groups[group_key].append(candidate)

    for group_key, candidates in groups.items():
        candidates.sort(key=lambda candidate: _overview_review_group_sort_key(group_key, candidate))
        groups[group_key] = candidates[:limit_per_group]
    return groups


def _render_review_candidate_card(
    candidate: Mapping[str, object],
    *,
    rank: int | None = None,
    compact: bool = False,
    metric_limit: int = 5,
    risk_flag: bool = False,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> str:
    action = str(candidate.get("action", "N/A"))
    action_tone = _action_tone(action)
    reason_parts = [str(part) for part in candidate.get("reason_parts", []) if str(part).strip()]
    reason_chips = "".join(_render_decision_card_chip(part, tone="neutral") for part in reason_parts[:3])
    metrics = [
        (str(label), str(value))
        for label, value in candidate.get("metrics", [])
        if str(label).strip() and str(value).strip()
    ]
    metrics_html = "".join(
        (
            '<span class="overview-review-card__metric">'
            f"<strong>{html.escape(label)}</strong>{html.escape(value)}"
            "</span>"
        )
        for label, value in metrics[:metric_limit]
    )
    rank_html = f'<span class="overview-review-card__rank">{rank}</span>' if rank is not None else ""
    risk_chip = _render_decision_card_chip("추가 위험", tone="warning") if risk_flag else ""
    display_sector_name = str(candidate.get("display_sector_name") or candidate.get("sector_name", "")).strip()
    taxonomy_subtext = str(candidate.get("taxonomy_subtext") or "").strip()
    taxonomy_html = (
        f'<span class="overview-sector-subtext">{html.escape(taxonomy_subtext)}</span>'
        if taxonomy_subtext
        else ""
    )
    invalidation = str(candidate.get("invalidation", "")).strip()
    invalidation_html = (
        '<div class="overview-review-card__invalidation">'
        f'<span>{html.escape(get_ui_text("decision_card_invalidation", locale))}</span>'
        f"<strong>{html.escape(invalidation)}</strong>"
        "</div>"
        if invalidation and not compact
        else ""
    )
    return (
        '<article class="overview-review-card">'
        '<div class="overview-review-card__topline">'
        f"{rank_html}"
        f'{_render_decision_card_chip(str(candidate.get("decision", "검토 후보")), tone=action_tone)}'
        f"{risk_chip}"
        "</div>"
        f'<div class="overview-review-card__sector">{html.escape(display_sector_name)}{taxonomy_html}</div>'
        f'<div class="overview-review-card__reasons">{reason_chips}</div>'
        f'<div class="overview-review-card__metrics">{metrics_html}</div>'
        f"{invalidation_html}"
        "</article>"
    )


def _render_overview_candidate_group_section(
    group_key: str,
    candidates: Sequence[Mapping[str, object]],
    *,
    compact: bool,
    locale: UiLocale,
) -> str:
    label = _OVERVIEW_REVIEW_GROUP_LABELS.get(group_key, "검토 후보")
    description = _OVERVIEW_REVIEW_GROUP_DESCRIPTIONS.get(group_key, "현재 기준에 맞는 검토 후보입니다.")
    cards = "".join(
        _render_review_candidate_card(candidate, rank=rank, compact=compact, locale=locale)
        for rank, candidate in enumerate(candidates[:3], start=1)
    )
    if not cards:
        cards = (
            '<div class="empty-state-card">'
            "<h4>표시할 후보 없음</h4>"
            "<p>현재 조건에 맞는 섹터가 없습니다.</p>"
            "</div>"
        )
    return (
        '<section class="overview-review-candidates__group" '
        f'data-review-side="{html.escape(group_key)}">'
        '<div class="overview-review-candidates__header">'
        "<div>"
        f'<div class="overview-section-title">{html.escape(label)}</div>'
        f'<div class="overview-command-surface__copy">{html.escape(description)}</div>'
        "</div>"
        f'<span class="overview-review-candidates__basis">{len(candidates[:3])}개 · proxy 근거 기준</span>'
        "</div>"
        f'<div class="overview-review-candidates__grid">{cards}</div>'
        "</section>"
    )


def _render_overview_review_candidates(
    candidates: Sequence[Mapping[str, object]] | Mapping[str, Sequence[Mapping[str, object]]],
    *,
    compact: bool = False,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> None:
    """Render the first-screen sector review candidate strip without mutating state."""
    if isinstance(candidates, Mapping):
        grouped_candidates = {
            key: list(candidates.get(key, []))
            for key in _OVERVIEW_REVIEW_GROUPS
        }
        if not any(grouped_candidates.values()):
            st.markdown(
                (
                    '<section class="overview-review-candidates" data-empty="true">'
                    '<div class="overview-review-candidates__header">'
                    '<div>'
                    '<div class="overview-section-title">검토 후보</div>'
                    '<div class="overview-command-surface__copy">현재 복합 검토 기준에 맞는 섹터 후보가 없습니다.</div>'
                    '</div>'
                    '</div>'
                    '</section>'
                ),
                unsafe_allow_html=True,
            )
            return

        sections = "".join(
            _render_overview_candidate_group_section(
                group_key,
                grouped_candidates[group_key],
                compact=compact,
                locale=locale,
            )
            for group_key in _OVERVIEW_REVIEW_GROUPS
        )
        st.markdown(
            (
                '<section class="overview-review-candidates" data-grouped="true">'
                '<div class="overview-review-candidates__header">'
                '<div>'
                '<div class="overview-section-title">검토 후보</div>'
                '<div class="overview-command-surface__copy">'
                '상방/하방 proxy는 보정 확률이 아니라 근거 점수입니다. '
                '복합점수와 수급은 랭킹 보조이며 canonical action policy는 바꾸지 않습니다.'
                '</div>'
                '</div>'
                f'<span class="overview-review-candidates__basis">{sum(len(items) for items in grouped_candidates.values())}개 · proxy 근거 기준</span>'
                '</div>'
                f'{sections}'
                '</section>'
            ),
            unsafe_allow_html=True,
        )
        return

    if not candidates:
        st.markdown(
            (
                '<section class="overview-review-candidates" data-empty="true">'
                '<div class="overview-review-candidates__header">'
                '<div>'
                '<div class="overview-section-title">검토 후보</div>'
                '<div class="overview-command-surface__copy">현재 복합 검토 기준에 맞는 섹터 후보가 없습니다.</div>'
                '</div>'
                '</div>'
                '</section>'
            ),
            unsafe_allow_html=True,
        )
        return

    card_html: list[str] = []
    for rank, candidate in enumerate(candidates[:3], start=1):
        card_html.append(_render_review_candidate_card(candidate, rank=rank, compact=compact, locale=locale))

    st.markdown(
        (
            '<section class="overview-review-candidates">'
            '<div class="overview-review-candidates__header">'
            '<div>'
            '<div class="overview-section-title">검토 후보</div>'
            '<div class="overview-command-surface__copy">'
            '상방/하방 proxy는 보정 확률이 아니라 근거 점수입니다. '
            '복합점수와 수급은 랭킹 보조이며 canonical action policy는 바꾸지 않습니다.'
            '</div>'
            '</div>'
            f'<span class="overview-review-candidates__basis">{html.escape(str(len(card_html)))}개 · proxy 근거 기준</span>'
            '</div>'
            f'<div class="overview-review-candidates__grid">{"".join(card_html)}</div>'
            '</section>'
        ),
        unsafe_allow_html=True,
    )


def _render_sector_momentum_board_card(candidate: Mapping[str, object]) -> str:
    return _render_review_candidate_card(
        candidate,
        metric_limit=4,
        risk_flag=bool(candidate.get("risk_flag")),
    )


_SECTOR_MOMENTUM_BOARD_LABELS: dict[str, str] = {
    "new_review": "신규/증액 검토",
    "held_monitor": "보유 모니터링",
    "held_reduce": "보유 축소/주의",
    "inflection": "변곡 감시",
}

_SECTOR_MOMENTUM_BOARD_DESCRIPTIONS: dict[str, str] = {
    "new_review": "모멘텀과 추세가 함께 통과한 신규 검토 후보입니다.",
    "held_monitor": "보유 Watch 중 독립 위험 근거가 없는 유지/모니터링 대상입니다.",
    "held_reduce": "보유 섹터 중 추가 위험 근거가 확인된 축소/주의 대상입니다.",
    "inflection": "RS 변화와 proxy 변곡 상태가 새로 강해지거나 약해지는 섹터입니다.",
}

_SECTOR_MOMENTUM_BOARD_ORDER = ("new_review", "held_monitor", "held_reduce", "inflection")


def _sector_momentum_safe_float(value: object, default: float = 0.0) -> float:
    parsed = _safe_float(value)  # type: ignore[arg-type]
    if parsed is None or not math.isfinite(parsed):
        return default
    return float(parsed)


def _sector_momentum_high_risk(signal: object) -> bool:
    mdd_3m = _sector_momentum_safe_float(getattr(signal, "mdd_3m", None), default=0.0)
    volatility_20d = _sector_momentum_safe_float(getattr(signal, "volatility_20d", None), default=0.0)
    return bool(mdd_3m <= -0.15 or volatility_20d >= 0.30)


def _sector_momentum_has_independent_risk(signal: object, candidate: Mapping[str, object]) -> bool:
    alerts = list(getattr(signal, "alerts", []) or [])
    flow_state = str(getattr(signal, "flow_state", "") or "").strip().lower()
    edge_proxy = _sector_momentum_safe_float(candidate.get("edge_proxy"), default=0.0)
    return bool(alerts or flow_state == "adverse" or edge_proxy < 0.0 or _sector_momentum_high_risk(signal))


def _sector_momentum_is_held(signal: object, held_sectors: Sequence[str] | None) -> bool:
    return is_signal_held(signal, held_sectors)


def _sector_momentum_board_for_candidate(
    signal: object,
    candidate: Mapping[str, object],
    held_sectors: Sequence[str] | None,
) -> str | None:
    action = str(getattr(signal, "action", "N/A") or "N/A")
    held = _sector_momentum_is_held(signal, held_sectors)
    independent_risk = _sector_momentum_has_independent_risk(signal, candidate)
    if held and action in {"Avoid", "N/A"}:
        return "held_reduce"
    if held and action == "Watch":
        return "held_reduce" if independent_risk else "held_monitor"
    if (
        action == "Strong Buy"
        and bool(getattr(signal, "momentum_core_pass", False))
        and bool(getattr(signal, "trend_ok", False))
    ):
        return "new_review"

    turning_point_state = str(candidate.get("turning_point_state", "") or "")
    if turning_point_state in {"Bullish turn", "Bearish turn"}:
        return "inflection"
    rs_change_pct = _sector_momentum_safe_float(getattr(signal, "rs_change_pct", None), default=0.0)
    mom_percentile = _sector_momentum_safe_float(getattr(signal, "mom_percentile", None), default=0.0)
    if action in {"Watch", "Hold"} and rs_change_pct >= 2.0 and mom_percentile >= 40.0:
        return "inflection"
    if action in {"Strong Buy", "Hold"} and (rs_change_pct <= -2.0 or not bool(getattr(signal, "trend_ok", True))):
        return "inflection"
    return None


def _build_sector_momentum_na_candidate(
    signal: object,
    held_sectors: Sequence[str] | None,
    *,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> dict[str, object]:
    thesis = describe_signal_decision(signal, held_sectors, locale=locale)
    reason_parts = _decision_card_parts(thesis.get("reason"), limit=2)
    reason_parts.insert(0, "데이터 확인")
    metrics = [
        ("상방 proxy", "N/A"),
        ("하방 proxy", "N/A"),
        ("엣지 proxy", "N/A"),
        ("변곡", "N/A"),
    ]
    return {
        "sector_name": str(getattr(signal, "sector_name", "")).strip(),
        "decision": str(thesis.get("decision", "")),
        "reason_parts": reason_parts,
        "invalidation": str(thesis.get("invalidation", "")),
        "action": "N/A",
        "action_policy": str(getattr(signal, "action_policy", "")),
        "candidate_policy": "SECTOR_MOMENTUM_NA_DATA_CHECK",
        "candidate_score": 0.0,
        "momentum_score": 0.0,
        "macro_score": 0.0,
        "flow_score": 0.0,
        "macro_available": False,
        "flow_available": False,
        "candidate_type": "데이터 확인",
        "warnings": list(getattr(signal, "alerts", []) or []),
        "upside_proxy": float("nan"),
        "downside_proxy": float("nan"),
        "edge_proxy": 0.0,
        "turning_point_state": "N/A",
        "bullish_evidence": [],
        "bearish_evidence": list(getattr(signal, "alerts", []) or []),
        "metrics": metrics,
    }


def _sector_momentum_candidate_sort_key(board_key: str, candidate: Mapping[str, object]) -> tuple[object, ...]:
    edge_proxy = _sector_momentum_safe_float(candidate.get("edge_proxy"), default=0.0)
    momentum_score = _sector_momentum_safe_float(candidate.get("momentum_score"), default=0.0)
    candidate_score = _sector_momentum_safe_float(candidate.get("candidate_score"), default=0.0)
    sector_name = str(candidate.get("sector_name", ""))
    if board_key == "new_review":
        return (-momentum_score, -candidate_score, sector_name)
    if board_key in {"held_monitor", "held_reduce"}:
        return (edge_proxy, -momentum_score, sector_name)
    if board_key == "inflection":
        return (-abs(edge_proxy), -momentum_score, sector_name)
    return (-candidate_score, sector_name)


def _build_sector_momentum_decision_boards(
    signals: Sequence,
    held_sectors: Sequence[str] | None = None,
    *,
    limit_per_board: int = 4,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> dict[str, list[dict[str, object]]]:
    """Group signals into decision boards using the existing overview proxy projection."""
    boards: dict[str, list[dict[str, object]]] = {key: [] for key in _SECTOR_MOMENTUM_BOARD_ORDER}
    for signal in signals:
        candidate = _build_overview_review_candidate_projection(signal, locale=locale)
        if (
            candidate is None
            and str(getattr(signal, "action", "N/A") or "N/A") == "N/A"
            and _sector_momentum_is_held(signal, held_sectors)
        ):
            candidate = _build_sector_momentum_na_candidate(signal, held_sectors, locale=locale)
        if candidate is None:
            continue
        board_key = _sector_momentum_board_for_candidate(signal, candidate, held_sectors)
        if board_key is None:
            continue
        candidate = dict(candidate)
        candidate["board_key"] = board_key
        candidate["held"] = _sector_momentum_is_held(signal, held_sectors)
        candidate["risk_flag"] = _sector_momentum_has_independent_risk(signal, candidate)
        candidate["high_risk"] = _sector_momentum_high_risk(signal)
        boards[board_key].append(candidate)

    for board_key, candidates in boards.items():
        candidates.sort(key=lambda candidate: _sector_momentum_candidate_sort_key(board_key, candidate))
        boards[board_key] = candidates[:limit_per_board]
    return boards


def render_sector_momentum_decision_boards(
    signals: Sequence,
    *,
    held_sectors: Sequence[str] | None = None,
    limit_per_board: int = 4,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> None:
    """Render decision-first boards for the sector momentum tab."""
    boards = _build_sector_momentum_decision_boards(
        signals,
        held_sectors=held_sectors,
        limit_per_board=limit_per_board,
        locale=locale,
    )
    sections: list[str] = []
    for board_key in _SECTOR_MOMENTUM_BOARD_ORDER:
        candidates = boards[board_key]
        if candidates:
            cards = "".join(_render_sector_momentum_board_card(candidate) for candidate in candidates)
        else:
            cards = (
                '<div class="empty-state-card">'
                "<h4>표시할 후보 없음</h4>"
                "<p>현재 조건에 맞는 섹터가 없습니다.</p>"
                "</div>"
            )
        sections.append(
            (
                '<section class="overview-review-candidates sector-momentum-board">'
                '<div class="overview-review-candidates__header">'
                "<div>"
                f'<div class="overview-section-title">{html.escape(_SECTOR_MOMENTUM_BOARD_LABELS[board_key])}</div>'
                f'<div class="overview-command-surface__copy">{html.escape(_SECTOR_MOMENTUM_BOARD_DESCRIPTIONS[board_key])}</div>'
                "</div>"
                f'<span class="overview-review-candidates__basis">{len(candidates)}개 · proxy 근거 기준</span>'
                "</div>"
                f'<div class="overview-review-candidates__grid">{cards}</div>'
                "</section>"
            )
        )

    st.markdown(
        (
            '<section class="sector-momentum-decision-boards">'
            '<div class="overview-review-candidates__header">'
            "<div>"
            '<div class="overview-section-title">의사결정 보드</div>'
            '<div class="overview-command-surface__copy">'
            "상방/하방 proxy는 보정 확률이 아니라 근거 점수입니다. "
            "이 보드는 기존 candidate projection을 재사용하며 canonical action policy는 바꾸지 않습니다."
            "</div>"
            "</div>"
            "</div>"
            f'{"".join(sections)}'
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def _theme_lens_return_label(value: object) -> str:
    numeric = _safe_float(value)
    if numeric is None or not math.isfinite(numeric):
        return "N/A"
    return f"{numeric * 100:+.2f}%"


def _theme_lens_basis_label(row: Mapping[str, object]) -> str:
    basis_items = row.get("classification_basis") or []
    labels: list[str] = []
    if isinstance(basis_items, Sequence) and not isinstance(basis_items, (str, bytes)):
        for item in basis_items:
            if not isinstance(item, Mapping):
                continue
            provider = str(item.get("provider", "")).strip()
            label = str(item.get("label", "")).strip()
            if provider and label:
                labels.append(f"{provider}: {label}")
            elif provider:
                labels.append(provider)
            elif label:
                labels.append(label)
    return " / ".join(labels[:2]) if labels else "테마 분류"


def _theme_lens_etf_label(row: Mapping[str, object]) -> str:
    etfs = row.get("representative_etfs") or []
    labels: list[str] = []
    if isinstance(etfs, Sequence) and not isinstance(etfs, (str, bytes)):
        for item in etfs:
            if not isinstance(item, Mapping):
                continue
            code = str(item.get("code", "")).strip()
            name = str(item.get("name", "")).strip()
            if code and name:
                labels.append(f"{name} ({code})")
            elif code:
                labels.append(code)
    return " / ".join(labels[:3]) if labels else "대표 ETF 없음"


def render_taxonomy_context_panel(
    taxonomy_context: Any | None,
    *,
    page: str,
    expanded: bool = True,
) -> None:
    """Render taxonomy-first traceability for KR dashboard pages."""
    if taxonomy_context is None:
        return
    contexts = list(getattr(taxonomy_context, "sector_contexts", ()) or ())
    diagnostics = list(getattr(taxonomy_context, "diagnostics", ()) or ())
    status = "완료" if not diagnostics else f"확인 필요 {len(diagnostics)}건"
    summary = f"theme_taxonomy v{getattr(taxonomy_context, 'taxonomy_version', '')} · {len(contexts)}개 런타임 섹터 · {status}"
    with st.expander("Theme Taxonomy Context", expanded=expanded):
        st.caption(
            "대시보드의 기존 섹터 신호를 theme_taxonomy 분류축으로 투영합니다. "
            "sector_map.yml은 실행 입력으로 유지하고, 이 패널은 별도 해석 레이어입니다."
        )
        st.markdown(
            (
                '<section class="sector-momentum-decision-boards theme-lens-panel">'
                '<div class="overview-review-candidates__header">'
                "<div>"
                '<div class="overview-section-title">테마/섹터 분류 레이어</div>'
                f'<div class="overview-command-surface__copy">{html.escape(summary)}</div>'
                "</div>"
                f'<span class="overview-review-candidates__basis">{html.escape(str(page))}</span>'
                "</div>"
                "</section>"
            ),
            unsafe_allow_html=True,
        )
        if diagnostics:
            st.warning(" / ".join(str(item) for item in diagnostics[:3]))
        display = pd.DataFrame(
            [
                {
                    "섹터": context.sector_name,
                    "기본 분류": " / ".join(context.base_labels),
                    "크로스 테마": " / ".join(context.cross_labels) or "없음",
                    "상품 테마": " / ".join(context.theme_labels) or "없음",
                    "기존 국면": context.legacy_regime,
                    "점수 역할": context.score_role,
                }
                for context in contexts
            ]
        )
        if not display.empty:
            st.dataframe(
                display,
                hide_index=True,
                width="stretch",
                column_config={
                    "섹터": st.column_config.TextColumn("섹터", width="medium"),
                    "기본 분류": st.column_config.TextColumn("기본 분류", width="large"),
                    "크로스 테마": st.column_config.TextColumn("크로스 테마", width="large"),
                    "상품 테마": st.column_config.TextColumn("상품 테마", width="medium"),
                    "기존 국면": st.column_config.TextColumn("기존 국면", width="small"),
                    "점수 역할": st.column_config.TextColumn("점수 역할", width="small"),
                },
            )


def render_theme_lens_panel(
    rows: Sequence[Mapping[str, object]] | None,
    *,
    status: str = "UNAVAILABLE",
    show_refresh_button: bool = False,
) -> bool:
    """Render the KR theme lens as ETF proxy evidence under the taxonomy layer."""
    theme_rows = list(rows or [])
    status_label = str(status or "UNAVAILABLE").strip().upper() or "UNAVAILABLE"
    summary = f"{len(theme_rows)}개 테마 · {status_label}"
    st.markdown(
        (
            '<section class="sector-momentum-decision-boards theme-lens-panel">'
            '<div class="overview-review-candidates__header">'
            "<div>"
            '<div class="overview-section-title">테마 렌즈</div>'
            '<div class="overview-command-surface__copy">'
            "대표 ETF 가격 기반 proxy로 상품 기반 테마 모멘텀을 확인합니다. "
            "theme_taxonomy 분류 레이어의 보조 증거로 표시하며 기존 섹터 신호 원장(canonical sector action)은 별도 추적합니다."
            "</div>"
            "</div>"
            f'<span class="overview-review-candidates__basis">{html.escape(summary)}</span>'
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )
    clicked = st.button("테마 ETF 갱신", key="theme_lens_refresh") if show_refresh_button else False

    if not theme_rows:
        st.markdown(
            '<div class="empty-state-card">'
            "<h4>테마 proxy 데이터 없음</h4>"
            "<p>테마 설정 또는 대표 ETF 가격 캐시를 확인해 주세요.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return clicked

    display = pd.DataFrame(
        [
            {
                "테마": str(row.get("theme_name", "")),
                "상태": str(row.get("status", "")),
                "대표 ETF": _theme_lens_etf_label(row),
                "기준일": str(row.get("latest_date", "")) or "N/A",
                "1D": _theme_lens_return_label(row.get("return_1d")),
                "1M": _theme_lens_return_label(row.get("return_1m")),
                "3M": _theme_lens_return_label(row.get("return_3m")),
                "분류 근거": _theme_lens_basis_label(row),
                "주의": str(row.get("warning", "")),
            }
            for row in theme_rows
        ]
    )
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "테마": st.column_config.TextColumn("테마", width="medium"),
            "상태": st.column_config.TextColumn("상태", width="small"),
            "대표 ETF": st.column_config.TextColumn("대표 ETF", width="large"),
            "기준일": st.column_config.TextColumn("기준일", width="small"),
            "1D": st.column_config.TextColumn("1D", width="small"),
            "1M": st.column_config.TextColumn("1M", width="small"),
            "3M": st.column_config.TextColumn("3M", width="small"),
            "분류 근거": st.column_config.TextColumn("분류 근거", width="large"),
            "주의": st.column_config.TextColumn("주의", width="medium"),
        },
    )
    return clicked


def _render_overview_workbench_header(
    *,
    market_id: str,
    signal_count: int,
    review_candidate_count: int,
    selected_layer: str,
    period: str,
) -> None:
    st.markdown(
        (
            '<section class="overview-workbench-header">'
            '<div>'
            '<div class="overview-workbench-header__eyebrow">UNIFIED DASHBOARD</div>'
            f'<div class="overview-workbench-header__title">{html.escape(str(market_id).upper())} 통합 워크벤치</div>'
            '<div class="overview-workbench-header__copy">'
            "분류축, 검토 후보, 시장 상태, 섹터 원장, 차트를 한 화면 흐름으로 압축했습니다."
            "</div>"
            "</div>"
            '<div class="overview-workbench-header__meta">'
            f"<span><b>신호</b>{html.escape(str(signal_count))}개</span>"
            f"<span><b>후보</b>{html.escape(str(review_candidate_count))}개</span>"
            f"<span><b>레이어</b>{html.escape(selected_layer)}</span>"
            f"<span><b>기간</b>{html.escape(period)}</span>"
            "</div>"
            "</section>"
        ),
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
    reference_index_labels: Sequence[str] | None = None,
    signals: Sequence,
    theme_mode: str,
    sector_map: Mapping[str, object],
    lookup_query_value: str = "",
    lookup_status: str = "",
    lookup_message: str = "",
    lookup_display_model: Mapping[str, Any] | None = None,
    export_growth_val: float | None = None,
    trade_indicators: Mapping[str, float] | None = None,
    sector_trade_lens: Sequence[Mapping[str, object]] | None = None,
    sector_export_trends: Mapping[str, float] | None = None,
    sector_export_history: Mapping[str, pd.Series] | None = None,
    has_trade_indicators: bool = False,
    has_sector_export_indicators: bool = True,
    taxonomy_context: Any | None = None,
    locale: UiLocale = DEFAULT_UI_LOCALE,
    is_mobile: bool = False,
) -> tuple[str, bool]:
    """Render the unified overview workbench and return stock lookup submission state."""
    cards = _build_overview_market_cards(
        prices_wide=prices_wide,
        benchmark_label=benchmark_label,
        reference_index_labels=reference_index_labels,
        signals=signals,
        current_regime=current_regime,
        price_status=price_status,
        macro_status=macro_status,
        export_growth_val=export_growth_val,
        trade_indicators=trade_indicators,
        has_trade_indicators=has_trade_indicators,
    )

    lookup_query = str(lookup_query_value or "")
    lookup_submitted = False
    sort_key = "모멘텀 점수"
    taxonomy_options = ["Theme Taxonomy", "기존 섹터"]
    taxonomy_layer_key = f"overview_sector_layer_{market_id}"
    default_taxonomy_layer = taxonomy_options[0] if taxonomy_context is not None else taxonomy_options[1]
    session_taxonomy_layer = st.session_state.get(taxonomy_layer_key)
    selected_taxonomy_layer = str(session_taxonomy_layer or default_taxonomy_layer)
    if taxonomy_context is None:
        selected_taxonomy_layer = taxonomy_options[1]
    elif selected_taxonomy_layer not in taxonomy_options:
        selected_taxonomy_layer = default_taxonomy_layer
    if session_taxonomy_layer is not None and str(session_taxonomy_layer) != selected_taxonomy_layer:
        st.session_state[taxonomy_layer_key] = selected_taxonomy_layer
    use_taxonomy_layer = taxonomy_context is not None and selected_taxonomy_layer == taxonomy_options[0]
    sector_frame = _build_overview_sector_frame(
        signals,
        sort_key=sort_key,
        sector_export_trends=sector_export_trends,
        has_sector_export_indicators=has_sector_export_indicators,
        taxonomy_context=taxonomy_context,
        use_taxonomy_display=use_taxonomy_layer,
    )
    overview_candidate_limit = 4 if is_mobile else 6
    review_candidate_groups = _build_overview_review_candidate_groups(
        signals,
        sector_frame,
        limit_per_group=overview_candidate_limit,
        taxonomy_context=taxonomy_context if use_taxonomy_layer else None,
        locale=locale,
    )
    review_candidate_count = sum(len(items[:overview_candidate_limit]) for items in review_candidate_groups.values())
    period = "3M"
    compare_basis = "벤치마크 대비"
    sector_group = selected_taxonomy_layer
    with st.container(border=True):
        _render_overview_workbench_header(
            market_id=market_id,
            signal_count=len([signal for signal in signals if str(getattr(signal, "action", "N/A")) != "N/A"]),
            review_candidate_count=review_candidate_count,
            selected_layer=selected_taxonomy_layer,
            period=period,
        )
        if taxonomy_context is not None:
            _render_overview_taxonomy_surface(
                taxonomy_context,
                sector_frame,
                selected_layer=selected_taxonomy_layer,
                market_id=market_id,
            )
            _render_overview_review_candidates(
                review_candidate_groups,
                compact=is_mobile,
                locale=locale,
            )
        else:
            _render_overview_review_candidates(
                review_candidate_groups,
                compact=is_mobile,
                locale=locale,
            )
        if str(market_id).strip().upper() == "KR":
            _render_overview_theme_proxy_watchlist(signals, is_mobile=is_mobile)
    with st.container(border=True):
        _render_overview_market_command_surface(cards)
        _render_sector_trade_lens(sector_trade_lens, limit=6 if not is_mobile else 3)
        lookup_context = st.expander("종목-섹터 조회", expanded=False) if is_mobile else nullcontext()
        with lookup_context:
            with st.form("overview_stock_lookup_form"):
                if is_mobile:
                    input_col, button_col = st.columns([3.2, 1.0])
                    market_col = None
                else:
                    input_col, market_col, button_col = st.columns([4.2, 1.25, 0.9])
                with input_col:
                    lookup_query = st.text_input(
                        get_ui_text("stock_lookup_label", locale),
                        value=lookup_query,
                        placeholder="종목명 또는 티커를 입력하세요 (예: 삼성전자, 005930)",
                        label_visibility="collapsed",
                    )
                if market_col is not None:
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

        filter_context = st.expander("필터", expanded=False) if is_mobile else nullcontext()
        with filter_context:
            if is_mobile:
                filter_col_1, filter_col_2 = st.columns([1.0, 1.0])
                filter_col_3, filter_col_4 = st.columns([1.0, 1.0])
            else:
                filter_col_1, filter_col_2, filter_col_3, filter_col_4 = st.columns([1.0, 1.2, 1.2, 1.2])
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
                sort_options = ["모멘텀 점수", "수익률(3M)", "상대강도"]
                sort_key = st.selectbox("정렬 기준", options=sort_options, index=0)
            with filter_col_4:
                sector_group = st.selectbox(
                    "분류 레이어",
                    options=taxonomy_options,
                    index=taxonomy_options.index(selected_taxonomy_layer),
                    key=taxonomy_layer_key,
                    disabled=taxonomy_context is None,
                )
                selected_taxonomy_layer = str(sector_group or selected_taxonomy_layer)
                use_taxonomy_layer = taxonomy_context is not None and selected_taxonomy_layer == taxonomy_options[0]
    del compare_basis, sector_group, sector_map

    sector_frame = _build_overview_sector_frame(
        signals,
        sort_key=str(sort_key),
        sector_export_trends=sector_export_trends,
        has_sector_export_indicators=has_sector_export_indicators,
        taxonomy_context=taxonomy_context,
        use_taxonomy_display=use_taxonomy_layer,
    )
    with st.container(border=True):
        st.markdown(
            (
                '<section class="overview-evidence-shell">'
                '<div>'
                '<div class="overview-section-title">섹터 원장과 차트</div>'
                '<div class="overview-command-surface__copy">'
                "정렬된 섹터 원장과 상대강도 추이를 같은 기준으로 봅니다."
                "</div>"
                "</div>"
                f'<span class="overview-review-candidates__basis">{html.escape(str(sort_key))} · {html.escape(str(period or "3M"))}</span>'
                "</section>"
            ),
            unsafe_allow_html=True,
        )

        if has_sector_export_indicators:
            ledger_col, trend_col = st.columns([0.92, 1.58], gap="large")
            with ledger_col:
                st.markdown('<div class="overview-section-title">섹터 모멘텀 & 상대강도</div>', unsafe_allow_html=True)
                _render_overview_sector_table(
                    sector_frame,
                    has_sector_export_indicators=has_sector_export_indicators,
                )
            with trend_col:
                st.markdown('<div class="overview-section-title">섹터 상대강도 추이</div>', unsafe_allow_html=True)
                fig = _build_overview_trend_figure(
                    prices_wide=prices_wide,
                    signals=signals,
                    benchmark_label=benchmark_label,
                    period=str(period or "3M"),
                    theme_mode=theme_mode,
                )
                st.plotly_chart(fig, width="stretch")
            st.markdown(
                '<div class="overview-section-title overview-section-title--sub">섹터별 수출 YoY 월별 추이</div>',
                unsafe_allow_html=True,
            )
            export_fig = _build_sector_export_trend_figure(
                sector_export_history=sector_export_history,
                signals=signals,
                theme_mode=theme_mode,
            )
            st.plotly_chart(export_fig, width="stretch")
        else:
            ledger_col, trend_col = st.columns([0.95, 1.55], gap="large")
            with ledger_col:
                st.markdown('<div class="overview-section-title">섹터 모멘텀 & 상대강도</div>', unsafe_allow_html=True)
                _render_overview_sector_table(
                    sector_frame,
                    has_sector_export_indicators=has_sector_export_indicators,
                )
            with trend_col:
                st.markdown('<div class="overview-section-title">섹터 상대강도 추이</div>', unsafe_allow_html=True)
                fig = _build_overview_trend_figure(
                    prices_wide=prices_wide,
                    signals=signals,
                    benchmark_label=benchmark_label,
                    period=str(period or "3M"),
                    theme_mode=theme_mode,
                )
                st.plotly_chart(fig, width="stretch")

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
    current_position_mode = normalize_position_mode(
        str(st.session_state.get(position_mode_key, "all"))
    )
    if st.session_state.get(position_mode_key) != current_position_mode:
        st.session_state[position_mode_key] = current_position_mode

    position_mode_control_kwargs = {
        "label": get_ui_text("filter_position_scope", locale),
        "options": list(POSITION_MODE_OPTIONS),
        "format_func": lambda value: format_position_mode_label(value, locale=locale),
        "selection_mode": "single",
        "key": position_mode_key,
        "width": "stretch",
    }

    with st.container(border=True):
        st.markdown(
            (
                '<div class="command-bar command-bar--compact">'
                f'<div class="command-bar__eyebrow">{html.escape(get_ui_text("command_bar_eyebrow", locale))}</div>'
                f'<div class="command-bar__title">{html.escape(get_ui_text("command_bar_title", locale))}</div>'
                f'<div class="command-bar__note">{html.escape(get_ui_text("command_bar_scope_note", locale))}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )

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
                **position_mode_control_kwargs,
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
                    **position_mode_control_kwargs,
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
                        '<div class="filter-chip-row">'
                        f'<span><b>{html.escape(get_ui_text("summary_regime_label", locale))}</b>{html.escape(current_regime)}</span>'
                        f'<span><b>{html.escape(get_ui_text("summary_action_label", locale))}</b>{html.escape(get_action_filter_label(current_action, locale))}</span>'
                        f'<span><b>{html.escape(get_ui_text("summary_scope_label", locale))}</b>{html.escape(scope_label)}</span>'
                        f'<span><b>{html.escape(get_ui_text("summary_positions_label", locale))}</b>{html.escape(format_position_mode_label(position_mode, locale=locale))}</span>'
                        f'<span><b>{html.escape(get_ui_text("summary_alerts_label", locale))}</b>{html.escape(get_ui_text("summary_alerted_only", locale) if alerted_only else get_ui_text("summary_all_signals", locale))}</span>'
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
                    '<div class="filter-chip-row">'
                    f'<span><b>{html.escape(get_ui_text("summary_regime_label", locale))}</b>{html.escape(current_regime)}</span>'
                    f'<span><b>{html.escape(get_ui_text("summary_action_label", locale))}</b>{html.escape(get_action_filter_label(current_action, locale))}</span>'
                    f'<span><b>{html.escape(get_ui_text("summary_scope_label", locale))}</b>{html.escape(scope_label)}</span>'
                    f'<span><b>{html.escape(get_ui_text("summary_positions_label", locale))}</b>{html.escape(format_position_mode_label(position_mode, locale=locale))}</span>'
                    f'<span><b>{html.escape(get_ui_text("summary_alerts_label", locale))}</b>{html.escape(get_ui_text("summary_alerted_only", locale) if alerted_only else get_ui_text("summary_all_signals", locale))}</span>'
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
    export_growth_val: float | None = None,
    trade_indicators: Mapping[str, float] | None = None,
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
    trade = dict(trade_indicators or {})
    metric_items = [
        _hero_metric(get_ui_text("hero_leading_index", locale), growth_val, "p"),
        _hero_metric(get_ui_text("hero_cpi_yoy", locale), inflation_val, "%"),
    ]
    if trade:
        metric_items.extend(
            [
                _hero_metric(get_ui_text("hero_trade_exports_yoy", locale), _safe_float(trade.get("exports_yoy")), "%"),
                _hero_metric(get_ui_text("hero_trade_imports_yoy", locale), _safe_float(trade.get("imports_yoy")), "%"),
            ]
        )
    else:
        metric_items.append(_hero_metric(get_ui_text("hero_export_growth", locale), export_growth_val, "%"))
    metric_items.append(_hero_metric(fx_label, fx_numeric, "%"))
    hero_metrics = "".join(metric_items)

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
    export_growth_val: float | None = None,
    trade_indicators: Mapping[str, float] | None = None,
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
        export_growth_val=export_growth_val,
        trade_indicators=trade_indicators,
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
