"""Dashboard UI entrypoints extracted from app.py."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from html import escape
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config.theme import set_theme_mode
from src.dashboard.analysis import _extract_heatmap_selection, top_pick_sort_key
from src.ui.components import (
    ALL_ACTION_KEY,
    DEFAULT_UI_LOCALE,
    FLOW_PROFILE_IDS,
    HEATMAP_PALETTE_OPTIONS,
    build_sector_strength_heatmap,
    describe_signal_decision,
    get_action_filter_label,
    format_cycle_phase_label,
    get_flow_profile_label,
    filter_signals_for_display,
    format_heatmap_palette_label,
    format_position_mode_label,
    get_ui_text,
    normalize_range_preset,
    render_action_summary,
    render_cycle_timeline_panel,
    render_decision_hero,
    render_investor_decision_boards,
    render_investor_flow_summary,
    render_panel_header,
    render_research_page_frame,
    render_returns_heatmap,
    render_rs_momentum_bar,
    render_rs_scatter,
    render_sector_momentum_decision_boards,
    render_theme_lens_panel,
    render_sector_detail_panel,
    render_signal_table,
    signal_display_sort_key,
    render_status_card_row,
    render_top_bar_filters,
    render_top_picks_table,
)


@dataclass(frozen=True)
class DashboardPageOption:
    page_id: str
    label: str
    url_path: str


DEFAULT_DASHBOARD_PAGE_ID = "overview"
KR_FLOW_INVESTOR_COLUMNS = {
    "외국인": "외국인",
    "기관합계": "기관",
    "개인": "개인",
}
KR_FLOW_INVESTOR_COLORS = {
    "외국인": "#2563eb",
    "기관합계": "#059669",
    "개인": "#dc2626",
}


def _format_collection_dataset_label(dataset: object) -> str:
    labels = {
        "market_prices": "시장데이터",
        "macro_data": "매크로데이터",
        "investor_flow": "수급데이터",
    }
    normalized = str(dataset or "").strip()
    return labels.get(normalized, normalized or "—")


def _summarize_collection_errors(row: pd.Series) -> str:
    failed_days = row.get("failed_days", [])
    failed_codes = row.get("failed_codes", {})
    if not isinstance(failed_days, list):
        failed_days = []
    if not isinstance(failed_codes, dict):
        failed_codes = {}

    parts: list[str] = []
    abort_reason = str(row.get("abort_reason", "") or "").strip()
    if bool(row.get("aborted", False)) and abort_reason:
        parts.append(f"중단: {abort_reason}")
    if failed_days:
        preview = ", ".join(str(day) for day in failed_days[:3])
        suffix = f" 외 {len(failed_days) - 3}건" if len(failed_days) > 3 else ""
        parts.append(f"미수집일 {len(failed_days)}건 ({preview}{suffix})")
    if failed_codes:
        key, value = next(iter(failed_codes.items()))
        parts.append(f"오류 {len(failed_codes)}건: {key} {value}")
    return " · ".join(parts) if parts else "없음"


def _format_krw_flow_amount(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "—"
    amount = float(numeric)
    sign = "+" if amount > 0 else "-" if amount < 0 else ""
    abs_amount = abs(amount)
    if abs_amount >= 1_000_000_000_000:
        return f"{sign}{abs_amount / 1_000_000_000_000:.2f}조"
    if abs_amount >= 100_000_000:
        return f"{sign}{abs_amount / 100_000_000:.0f}억"
    return f"{sign}{abs_amount:,.0f}원"


def _build_kr_latest_sector_flow_amounts(investor_flow_frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["Sector", "외국인", "기관", "개인", "합계"]
    if investor_flow_frame.empty:
        return pd.DataFrame(columns=columns)

    latest_date = investor_flow_frame.index.max()
    latest_snapshot = investor_flow_frame.loc[investor_flow_frame.index == latest_date].copy()
    if latest_snapshot.empty:
        return pd.DataFrame(columns=columns)

    latest_snapshot["net_buy_amount"] = pd.to_numeric(latest_snapshot["net_buy_amount"], errors="coerce").fillna(0.0)
    grouped = (
        latest_snapshot.groupby(["sector_name", "investor_type"], dropna=False)["net_buy_amount"]
        .sum()
        .unstack(fill_value=0.0)
    )
    for investor_label in KR_FLOW_INVESTOR_COLUMNS:
        if investor_label not in grouped.columns:
            grouped[investor_label] = 0.0
    grouped = grouped[list(KR_FLOW_INVESTOR_COLUMNS)]
    grouped["합계"] = grouped.sum(axis=1)
    grouped = grouped.reindex(grouped["합계"].abs().sort_values(ascending=False).index)

    display = pd.DataFrame({"Sector": grouped.index.astype(str)})
    for raw_label, display_label in KR_FLOW_INVESTOR_COLUMNS.items():
        display[display_label] = grouped[raw_label].map(_format_krw_flow_amount).to_list()
    display["합계"] = grouped["합계"].map(_format_krw_flow_amount).to_list()
    return display.reset_index(drop=True)


def _get_kr_flow_sector_options(investor_flow_frame: pd.DataFrame) -> list[str]:
    if investor_flow_frame.empty or "sector_name" not in investor_flow_frame.columns:
        return []

    source = investor_flow_frame.copy()
    source["flow_date"] = pd.to_datetime(investor_flow_frame.index)
    source["net_buy_amount"] = pd.to_numeric(source["net_buy_amount"], errors="coerce").fillna(0.0)
    latest_date = source["flow_date"].max()
    latest_totals = (
        source.loc[source["flow_date"].eq(latest_date)]
        .groupby("sector_name", dropna=False)["net_buy_amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
    )
    return [str(sector) for sector in latest_totals.index]


def _build_kr_sector_flow_trend_figure(investor_flow_frame: pd.DataFrame) -> go.Figure:
    if investor_flow_frame.empty:
        fig = go.Figure()
        fig.update_layout(title="기간별 섹터 수급 추이")
        return fig

    trend_source = investor_flow_frame.copy()
    trend_source["flow_date"] = pd.to_datetime(investor_flow_frame.index)
    trend_source["net_buy_amount"] = pd.to_numeric(trend_source["net_buy_amount"], errors="coerce").fillna(0.0)
    sector_options = _get_kr_flow_sector_options(trend_source)
    if not sector_options:
        fig = go.Figure()
        fig.update_layout(title="섹터별 투자자 수급 추이")
        return fig

    trend = (
        trend_source.groupby(["flow_date", "sector_name", "investor_type"], dropna=False)["net_buy_amount"]
        .sum()
        .reset_index()
        .sort_values(["sector_name", "investor_type", "flow_date"])
    )
    if trend.empty:
        fig = go.Figure()
        fig.update_layout(title="섹터별 투자자 수급 추이")
        return fig

    fig = make_subplots(
        rows=len(sector_options),
        cols=1,
        shared_xaxes=True,
        subplot_titles=sector_options,
        vertical_spacing=min(0.04, 0.22 / max(len(sector_options), 1)),
    )
    shown_legend_groups: set[str] = set()
    for row_index, sector in enumerate(sector_options, start=1):
        sector_rows = trend.loc[trend["sector_name"].astype(str).eq(sector)]
        for raw_label, display_label in KR_FLOW_INVESTOR_COLUMNS.items():
            investor_rows = sector_rows.loc[sector_rows["investor_type"].astype(str).eq(raw_label)]
            if investor_rows.empty:
                continue
            show_legend = display_label not in shown_legend_groups
            shown_legend_groups.add(display_label)
            fig.add_trace(
                go.Scatter(
                    x=investor_rows["flow_date"],
                    y=investor_rows["net_buy_amount"],
                    mode="lines+markers",
                    name=display_label,
                    legendgroup=display_label,
                    showlegend=show_legend,
                    line=dict(color=KR_FLOW_INVESTOR_COLORS[raw_label], width=2),
                    marker=dict(size=5, color=KR_FLOW_INVESTOR_COLORS[raw_label]),
                    text=[_format_krw_flow_amount(value) for value in investor_rows["net_buy_amount"]],
                    hovertemplate=f"{sector}<br>%{{x|%Y-%m-%d}}<br>{display_label} 순매수: %{{text}}<extra></extra>",
                ),
                row=row_index,
                col=1,
            )
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#8a8f98", row=row_index, col=1)

    fig.update_layout(
        title="섹터별 투자자 수급 추이",
        height=max(560, len(sector_options) * 128 + 120),
        margin=dict(l=48, r=24, t=76, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="기간", row=len(sector_options), col=1)
    fig.update_yaxes(title_text="순매수", showticklabels=True)
    return fig


def _format_collection_history_sample(history: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if history.empty or "dataset" not in history.columns:
        return pd.DataFrame()

    selected = history[history["dataset"].astype(str).eq(dataset)].copy()
    if selected.empty:
        return pd.DataFrame()

    selected["_created_sort"] = pd.to_datetime(selected["created_at"], errors="coerce", utc=True)
    selected = selected.sort_values("_created_sort", ascending=False, na_position="last").reset_index(drop=True)
    display = pd.DataFrame(
        {
            "수집일시": selected["created_at"],
            "요청범위": selected["requested_start"].fillna("").astype(str)
            + " ~ "
            + selected["requested_end"].fillna("").astype(str),
            "상태": selected["status"].fillna("").astype(str),
            "커버리지": selected["coverage_complete"].map(lambda value: "완료" if bool(value) else "확인 필요"),
            "중단": selected["aborted"].map(lambda value: "예" if bool(value) else "아니오"),
            "오류요약": selected.apply(_summarize_collection_errors, axis=1),
            "완료율(%)": selected["completion_pct"],
            "provider": selected.get("provider", pd.Series([""] * len(selected))).fillna("").astype(str),
            "저장행수": selected["row_count"],
        }
    )
    return display


def _format_collection_date(value: object, *, monthly: bool = False) -> str:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if monthly and len(digits) >= 6:
        return f"{digits[:4]}-{digits[4:6]}"
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    if len(digits) >= 6:
        return f"{digits[:4]}-{digits[4:6]}"
    return text or "—"


def _format_collection_timestamp(value: object) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "—"
    return ts.strftime("%Y-%m-%d %H:%M")


def _collection_bounds_for_dataset(bounds: dict[str, Any], dataset: str, provider: object = "") -> dict[str, Any]:
    provider_token = str(provider or "").strip().upper()
    provider_key = f"{dataset}:{provider_token}" if provider_token else ""
    if provider_key and isinstance(bounds.get(provider_key), dict):
        return dict(bounds.get(provider_key) or {})
    if dataset in bounds and isinstance(bounds.get(dataset), dict):
        return dict(bounds.get(dataset) or {})
    if dataset == "investor_flow" and (
        "min_trade_date" in bounds or "max_trade_date" in bounds
    ):
        return dict(bounds)
    return {}


def _format_collection_data_range(dataset: str, bounds: dict[str, Any], fallback_end: object = "") -> str:
    monthly = dataset == "macro_data"
    if dataset == "macro_data":
        start = bounds.get("min_period_month", "")
        end = bounds.get("max_period_month", fallback_end)
    else:
        start = bounds.get("min_trade_date", "")
        end = bounds.get("max_trade_date", fallback_end)
    formatted_start = _format_collection_date(start, monthly=monthly)
    formatted_end = _format_collection_date(end, monthly=monthly)
    if formatted_start == "—" and formatted_end == "—":
        return "—"
    if formatted_start == "—":
        return f"~ {formatted_end}"
    if formatted_end == "—":
        return f"{formatted_start} ~"
    return f"{formatted_start} ~ {formatted_end}"


def _format_collection_request_range(row: pd.Series | dict[str, Any]) -> str:
    getter = row.get if hasattr(row, "get") else dict(row).get
    start = _format_collection_date(getter("requested_start", ""))
    end = _format_collection_date(getter("requested_end", ""))
    if start == "—" and end == "—":
        return "—"
    return f"{start} ~ {end}"


def _summarize_collection_attention(row: pd.Series | dict[str, Any]) -> str:
    series = row if isinstance(row, pd.Series) else pd.Series(dict(row))
    error_summary = _summarize_collection_errors(series)
    if error_summary != "없음":
        return error_summary
    if bool(series.get("coverage_complete", False)):
        return "없음"
    pct = pd.to_numeric(pd.Series([series.get("completion_pct", pd.NA)]), errors="coerce").iloc[0]
    if pd.notna(pct):
        return f"요청 범위 일부 미충족 ({float(pct):.1f}%)"
    row_count = pd.to_numeric(pd.Series([series.get("row_count", pd.NA)]), errors="coerce").iloc[0]
    if pd.notna(row_count) and int(row_count) > 0:
        return f"부분 수집 데이터 있음 ({int(row_count):,}건)"
    return "요청 범위 일부 미충족"


def _latest_collection_rows_by_dataset(history: pd.DataFrame) -> dict[str, pd.Series]:
    if history.empty or "dataset" not in history.columns:
        return {}
    sortable = history.copy()
    sortable["_created_sort"] = pd.to_datetime(sortable.get("created_at"), errors="coerce", utc=True)
    sortable = sortable.sort_values("_created_sort", ascending=False, na_position="last")
    latest: dict[str, pd.Series] = {}
    for _, row in sortable.iterrows():
        dataset = str(row.get("dataset", "") or "").strip()
        if dataset and dataset not in latest:
            latest[dataset] = row
    return latest


def _format_collection_overview_rows(
    *,
    statuses: dict[str, dict[str, Any]],
    history: pd.DataFrame,
    bounds: dict[str, Any],
    dataset_order: list[str],
) -> pd.DataFrame:
    latest_rows = _latest_collection_rows_by_dataset(history)
    latest_macro_rows: dict[str, pd.Series] = {}
    if not history.empty and {"dataset", "created_at"}.issubset(history.columns):
        macro_history = history[history["dataset"].astype(str).eq("macro_data")].copy()
        macro_history["_created_sort"] = pd.to_datetime(macro_history["created_at"], errors="coerce", utc=True)
        macro_history = macro_history.sort_values("_created_sort", ascending=False, na_position="last")
        for _, row in macro_history.iterrows():
            provider = str(row.get("provider", "") or "").strip().upper()
            if provider and provider not in latest_macro_rows:
                latest_macro_rows[provider] = row
    rows: list[dict[str, object]] = []
    for dataset in dataset_order:
        dataset_sources: list[pd.Series]
        if dataset == "macro_data" and latest_macro_rows:
            dataset_sources = list(latest_macro_rows.values())
        else:
            latest = latest_rows.get(dataset)
            dataset_sources = [latest if latest is not None else pd.Series(statuses.get(dataset) or {})]

        status = dict(statuses.get(dataset) or {})
        for source in dataset_sources:
            provider = str(source.get("provider", status.get("provider", "—")) or "—")
            dataset_bounds = _collection_bounds_for_dataset(bounds, dataset, provider)
            row_count = dataset_bounds.get("row_count", source.get("row_count", pd.NA))
            attention_source = source.copy()
            if "row_count" not in attention_source or pd.isna(attention_source.get("row_count", pd.NA)):
                attention_source["row_count"] = row_count
            rows.append(
                {
                    "데이터": _format_collection_dataset_label(dataset),
                    "상태": str(source.get("status", status.get("status", "—")) or "—"),
                    "마지막 갱신": _format_collection_timestamp(source.get("created_at", status.get("updated_at", ""))),
                    "보유기간": _format_collection_data_range(dataset, dataset_bounds, status.get("watermark_key", status.get("end", ""))),
                    "최근 요청": _format_collection_request_range(source),
                    "실패/주의": _summarize_collection_attention(attention_source),
                    "provider": provider,
                    "저장행수": row_count,
                }
            )
    return pd.DataFrame(rows)


def _format_dataset_status_rows(statuses: dict[str, dict[str, Any]], dataset_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in dataset_order:
        status = dict(statuses.get(dataset) or {})
        predicted = int(status.get("predicted_requests", 0) or 0)
        processed = int(status.get("processed_requests", 0) or 0)
        completion = round(processed / predicted * 100, 1) if predicted > 0 else None
        failed_days = status.get("failed_days") if isinstance(status.get("failed_days"), list) else []
        failed_codes = status.get("failed_codes") if isinstance(status.get("failed_codes"), dict) else {}
        rows.append(
            {
                "데이터": _format_collection_dataset_label(dataset),
                "상태": str(status.get("status", "") or "—"),
                "provider": str(status.get("provider", "") or "—"),
                "워터마크": str(status.get("watermark_key", status.get("end", "")) or "—"),
                "커버리지": "완료" if bool(status.get("coverage_complete", False)) else "확인 필요",
                "최근 완료율(%)": completion,
                "미수집일": len(failed_days),
                "오류항목": len(failed_codes),
                "중단": "예" if bool(status.get("aborted", False)) else "아니오",
            }
        )
    return pd.DataFrame(rows)


def _format_dataset_error_rows(statuses: dict[str, dict[str, Any]], dataset_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in dataset_order:
        label = _format_collection_dataset_label(dataset)
        status = dict(statuses.get(dataset) or {})
        if bool(status.get("aborted", False)):
            rows.append(
                {
                    "데이터": label,
                    "구분": "중단",
                    "항목": "abort_reason",
                    "오류": str(status.get("abort_reason", "") or "수집 중단"),
                }
            )
        failed_days = status.get("failed_days") if isinstance(status.get("failed_days"), list) else []
        for day in failed_days[:10]:
            rows.append({"데이터": label, "구분": "미수집일", "항목": str(day), "오류": "미수집"})
        failed_codes = status.get("failed_codes") if isinstance(status.get("failed_codes"), dict) else {}
        for key, value in list(failed_codes.items())[:10]:
            rows.append({"데이터": label, "구분": "오류항목", "항목": str(key), "오류": str(value)})
    return pd.DataFrame(rows, columns=["데이터", "구분", "항목", "오류"])


def build_dashboard_page_options(market_id: str) -> list[DashboardPageOption]:
    normalized_market = str(market_id).strip().upper() or "KR"
    options = [
        DashboardPageOption("overview", "대시보드", f"{normalized_market.lower()}-overview"),
        DashboardPageOption("signals", "섹터 모멘텀", f"{normalized_market.lower()}-signals"),
        DashboardPageOption("research", "상대강도 분석", f"{normalized_market.lower()}-research"),
        DashboardPageOption("constituents", "구성종목", f"{normalized_market.lower()}-constituents"),
    ]
    if normalized_market in {"KR", "US"}:
        options.append(
            DashboardPageOption(
                "flow",
                "투자자 수급" if normalized_market == "KR" else "ETF 수급 프록시",
                f"{normalized_market.lower()}-flow",
            )
        )
    if normalized_market == "KR":
        options.append(DashboardPageOption("quality", "데이터 수집 이력", "kr-quality"))
    return options


def normalize_dashboard_page_id(page_id: object, market_id: str) -> str:
    available_page_ids = {option.page_id for option in build_dashboard_page_options(market_id)}
    normalized = str(page_id or DEFAULT_DASHBOARD_PAGE_ID).strip()
    if normalized in available_page_ids:
        return normalized
    return DEFAULT_DASHBOARD_PAGE_ID


def resolve_dashboard_page_title(page_id: object, market_id: str) -> str:
    normalized_market = str(market_id).strip().upper() or "KR"
    normalized_page_id = normalize_dashboard_page_id(page_id, normalized_market)
    for option in build_dashboard_page_options(normalized_market):
        if option.page_id == normalized_page_id:
            return f"{normalized_market} {option.label}"
    return f"{normalized_market} 대시보드"


def _format_sidebar_status_chip(label: str, status: object) -> str:
    normalized_status = str(status or "UNKNOWN").strip() or "UNKNOWN"
    status_class = "ready" if normalized_status in {"LIVE", "CACHED", "SAMPLE"} else "attention"
    return (
        f'<span class="sidebar-status-chip sidebar-status-chip--{status_class}">'
        f'<span>{escape(label)}</span><strong>{escape(normalized_status)}</strong>'
        "</span>"
    )


def render_sidebar_controls(
    *,
    market_id: str,
    ui_labels: dict[str, str],
    theme_mode: str,
    analysis_heatmap_palette: str,
    probe_price_status: str,
    probe_macro_status: str,
    btn_states: dict[str, bool],
    asof_default: date,
    probe_investor_flow_status: str = "SAMPLE",
    flow_profile: str = "foreign_lead",
    momentum_method: str = "legacy_rs_ma_v0",
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> tuple[date, str, bool, bool, bool, bool]:
    normalized_market = str(market_id or "KR").strip().upper() or "KR"
    status_chips = [
        _format_sidebar_status_chip("시장", probe_price_status),
        _format_sidebar_status_chip("매크로", probe_macro_status),
    ]
    if normalized_market == "KR":
        status_chips.append(_format_sidebar_status_chip("수급", probe_investor_flow_status))

    st.markdown(
        (
            '<div class="sidebar-workspace">'
            '<div class="sidebar-workspace__eyebrow">OPERATIONS</div>'
            f'<div class="sidebar-workspace__title">{escape(normalized_market)} 섹터 콘솔</div>'
            '<div class="sidebar-workspace__meta">데이터 상태와 실행 기준을 통제합니다.</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="sidebar-ops-panel">'
            '<div class="sidebar-ops-panel__header">'
            '<span>데이터 운용</span><strong>상태 / 갱신</strong>'
            "</div>"
            f'<div class="sidebar-status-grid">{"".join(status_chips)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    refresh_market = st.button(
        "시장데이터 갱신",
        disabled=not btn_states["refresh_market"],
        width="stretch",
    )
    refresh_macro = st.button(
        "매크로데이터 갱신",
        disabled=not btn_states["refresh_macro"],
        width="stretch",
    )
    refresh_flow = False
    if normalized_market == "KR":
        refresh_flow = st.button(
            get_ui_text("flow_refresh_button", ui_locale),
            width="stretch",
        )

    st.markdown('<div class="sidebar-section-label">분석 기준</div>', unsafe_allow_html=True)
    use_light_theme = st.toggle(
        "라이트 테마",
        value=theme_mode == "light",
        help="현재 대시보드 테마를 전환합니다.",
    )
    selected_theme_mode = "light" if use_light_theme else "dark"
    if selected_theme_mode != theme_mode:
        set_theme_mode(selected_theme_mode)
        st.rerun()

    selected_heatmap_palette = st.selectbox(
        "히트맵 색상",
        options=list(HEATMAP_PALETTE_OPTIONS),
        index=list(HEATMAP_PALETTE_OPTIONS).index(analysis_heatmap_palette),
        format_func=lambda value: format_heatmap_palette_label(value, locale=ui_locale),
        help="월간 분석 히트맵의 발산형 색상을 변경합니다.",
    )
    if selected_heatmap_palette != analysis_heatmap_palette:
        st.session_state["analysis_heatmap_palette"] = selected_heatmap_palette
        st.rerun()

    asof_date = st.date_input(
        "기준일",
        value=asof_default,
        max_value=date.today(),
    )
    st.session_state["asof_date_str"] = asof_date.strftime("%Y%m%d")

    with st.popover("모델 파라미터", width="stretch"):
        with st.form("model_params_form"):
            st.caption("슬라이더 숫자를 직접 클릭해 세밀하게 조정할 수 있습니다.")
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                slider_epsilon = st.slider(
                    "Epsilon (방향 민감도)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state["epsilon"]),
                    step=0.05,
                    help="3MA 방향 판정 최소 변화량입니다. 0이면 모든 변화를 반영합니다.",
                )
                slider_price_years = st.slider(
                    "가격 데이터 기간(년)",
                    min_value=1,
                    max_value=5,
                    value=int(st.session_state["price_years"]),
                    step=1,
                )
                slider_rs_ma = None
            with param_col2:
                slider_ma_fast = None
                slider_ma_slow = None
                if str(momentum_method) == "legacy_rs_ma_v0":
                    slider_rs_ma = st.slider(
                        "RS MA 기간",
                        min_value=5,
                        max_value=60,
                        value=int(st.session_state["rs_ma_period"]),
                        step=1,
                    )
                    slider_ma_fast = st.slider(
                        "빠른 MA",
                        min_value=5,
                        max_value=60,
                        value=int(st.session_state["ma_fast"]),
                        step=1,
                    )
                    slider_ma_slow = st.slider(
                        "느린 MA",
                        min_value=20,
                        max_value=120,
                        value=int(st.session_state["ma_slow"]),
                        step=1,
                    )
                else:
                    st.info("KR 하이브리드 모멘텀 활성화 상태입니다. RS/MA 슬라이더는 진단용 레거시 로직에만 사용됩니다.")

            apply_params = st.form_submit_button("적용", width="stretch")

        if apply_params:
            st.session_state["epsilon"] = float(slider_epsilon)
            if slider_rs_ma is not None and slider_ma_fast is not None and slider_ma_slow is not None:
                st.session_state["rs_ma_period"] = int(slider_rs_ma)
                st.session_state["ma_fast"] = int(slider_ma_fast)
                st.session_state["ma_slow"] = int(slider_ma_slow)
            st.session_state["price_years"] = int(slider_price_years)
            st.rerun()

    if normalized_market == "KR":
        st.markdown('<div class="sidebar-section-label">수급 해석</div>', unsafe_allow_html=True)
        selected_flow_profile = st.selectbox(
            get_ui_text("flow_profile_label", ui_locale),
            options=list(FLOW_PROFILE_IDS),
            index=list(FLOW_PROFILE_IDS).index(str(flow_profile or "foreign_lead")),
            format_func=lambda value: get_flow_profile_label(value, ui_locale),
            help=get_ui_text("flow_sidebar_caption", ui_locale),
        )
    else:
        selected_flow_profile = str(flow_profile or "foreign_lead")

    st.markdown(
        f'<div class="sidebar-footer-label">{escape(ui_labels.get("sidebar_title", "섹터 로테이션"))}</div>',
        unsafe_allow_html=True,
    )
    return asof_date, selected_flow_profile, refresh_market, refresh_macro, refresh_flow


def render_analysis_canvas(
    *,
    heatmap_return_display: pd.DataFrame,
    heatmap_strength_display: pd.DataFrame,
    selected_cycle_phase: str,
    theme_mode: str,
    analysis_heatmap_palette: str,
    visible_segments: list[dict[str, Any]],
    current_regime: str,
    analysis_prices_phase: pd.DataFrame,
    analysis_prices: pd.DataFrame,
    sector_columns: list[str],
    benchmark_label: str,
    analysis_max_date: date,
    analysis_min_date: date,
    build_sector_detail_figure,
    resolve_range_from_preset,
    signal_lookup: dict[str, Any] | None = None,
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> None:
    render_panel_header(
        eyebrow=get_ui_text("analysis_canvas_eyebrow", ui_locale),
        title=get_ui_text("analysis_canvas_title", ui_locale),
        description=get_ui_text("analysis_canvas_description", ui_locale),
        badge=format_cycle_phase_label(selected_cycle_phase, locale=ui_locale),
    )
    with st.container(border=True):
        render_panel_header(
            eyebrow="섹터 비교",
            title="월간 섹터 수익률",
            description="절대 월간 섹터 수익률을 먼저 확인하세요. 셀 클릭 시 해당 섹터와 월이 아래 상세 패널에 고정됩니다.",
            badge=format_cycle_phase_label(selected_cycle_phase, locale=ui_locale),
        )
        heatmap_fig = build_sector_strength_heatmap(
            heatmap_return_display,
            selected_sector=str(st.session_state.get("selected_sector", "")),
            selected_month=str(st.session_state.get("selected_month", "")),
            theme_mode=theme_mode,
            palette=analysis_heatmap_palette,
            title="월간 섹터 수익률",
            empty_message="활성 필터에 해당하는 월간 섹터 수익률 데이터가 없습니다.",
            helper_metric_label="월간 수익률",
            hover_value_suffix="%",
        )
        heatmap_event = st.plotly_chart(
            heatmap_fig,
            width="stretch",
            key="analysis_sector_return_heatmap",
            on_select="rerun",
            selection_mode=("points",),
            config={"displayModeBar": False},
        )
        absolute_selection = _extract_heatmap_selection(heatmap_event)

    with st.container(border=True):
        render_panel_header(
            eyebrow="상대강도",
            title=f"월간 섹터 강도 vs {benchmark_label}",
            description=f"각 셀은 {benchmark_label} 대비 월간 초과 수익률입니다. 동일한 섹터/월 연동 선택을 사용합니다.",
            badge=format_cycle_phase_label(selected_cycle_phase, locale=ui_locale),
        )
        strength_fig = build_sector_strength_heatmap(
            heatmap_strength_display,
            selected_sector=str(st.session_state.get("selected_sector", "")),
            selected_month=str(st.session_state.get("selected_month", "")),
            theme_mode=theme_mode,
            palette=analysis_heatmap_palette,
            title=f"월간 섹터 강도 vs {benchmark_label}",
            empty_message=f"활성 필터에 해당하는 {benchmark_label} 대비 월간 섹터 강도 데이터가 없습니다.",
            helper_metric_label="월간 초과 수익률",
            hover_value_suffix=f"%p vs {benchmark_label}",
        )
        strength_event = st.plotly_chart(
            strength_fig,
            width="stretch",
            key="analysis_sector_strength_heatmap",
            on_select="rerun",
            selection_mode=("points",),
            config={"displayModeBar": False},
        )
        relative_selection = _extract_heatmap_selection(strength_event)

    shared_heatmap_selection = relative_selection or absolute_selection
    if shared_heatmap_selection is not None:
        month_value, sector_value = shared_heatmap_selection
        if (
            month_value != str(st.session_state.get("selected_month", ""))
            or sector_value != str(st.session_state.get("selected_sector", ""))
        ):
            st.session_state["selected_month"] = month_value
            st.session_state["selected_sector"] = sector_value
            st.rerun()

    with st.container(border=True):
        render_panel_header(
            eyebrow="사이클 맥락",
            title="사이클 타임라인 맥락",
            description="초기/후기 사이클 필터를 사용해 히트맵과 상세 차트를 특정 매크로 국면으로 압축하세요.",
            badge=current_regime,
        )
        chosen_cycle_phase = render_cycle_timeline_panel(
            segments=visible_segments,
            selected_cycle_phase=selected_cycle_phase,
            theme_mode=theme_mode,
            locale=ui_locale,
        )
        if chosen_cycle_phase != selected_cycle_phase:
            st.session_state["selected_cycle_phase"] = chosen_cycle_phase
            st.rerun()

    detail_prices = analysis_prices_phase if not analysis_prices_phase.empty else analysis_prices
    if not detail_prices.empty:
        detail_columns = [col for col in sector_columns if col in detail_prices.columns]
        if benchmark_label in detail_prices.columns:
            detail_columns.append(benchmark_label)
        detail_prices = detail_prices[detail_columns]
    else:
        detail_prices = pd.DataFrame()
    detail_normalized = pd.DataFrame()
    ranking_rows: list[dict[str, object]] = []
    if not detail_prices.empty:
        detail_prices = detail_prices.dropna(how="all")
        detail_prices = detail_prices.ffill().dropna(axis=1, how="all")
        if len(detail_prices) >= 2:
            detail_normalized = detail_prices.div(detail_prices.iloc[0]).mul(100.0)
            sector_detail_returns = (
                detail_normalized[[col for col in detail_normalized.columns if col != benchmark_label]].iloc[-1] - 100.0
            )
            sector_detail_returns = sector_detail_returns.dropna().sort_values(ascending=False)
            ranking_rows = [
                {"sector": str(name), "return_pct": float(value)}
                for name, value in sector_detail_returns.items()
            ]

    if ranking_rows:
        selected_sector_state = str(st.session_state.get("selected_sector", ""))
        if selected_sector_state not in [row["sector"] for row in ranking_rows]:
            st.session_state["selected_sector"] = str(ranking_rows[0]["sector"])
    else:
        st.session_state["selected_sector"] = ""

    selected_sector = str(st.session_state.get("selected_sector", ""))
    comparison_sectors = [
        str(row["sector"])
        for row in ranking_rows
        if str(row["sector"]) != selected_sector
    ][:2]
    detail_figure = build_sector_detail_figure(
        detail_normalized,
        selected_sector=selected_sector,
        benchmark_label=benchmark_label if benchmark_label in detail_normalized.columns else None,
        comparison_sectors=comparison_sectors,
        selected_month=str(st.session_state.get("selected_month", "")),
        theme_mode=theme_mode,
    )

    with st.container(border=True):
        detail_badge = selected_sector or "선택 없음"
        detail_summary = None
        if signal_lookup and selected_sector in signal_lookup:
            detail_summary = describe_signal_decision(
                signal_lookup[selected_sector],
                st.session_state.get("held_sectors", []),
                locale=ui_locale,
            )
        render_panel_header(
            eyebrow="연동 상세",
            title="선택 섹터 상세 추적",
            description="왼쪽에서 섹터 순위를 확인하고, 선택 섹터를 벤치마크 및 상위 섹터와 비교하세요.",
            badge=detail_badge,
        )
        chosen_sector, chosen_preset = render_sector_detail_panel(
            ranking_rows=ranking_rows,
            detail_figure=detail_figure,
            selected_sector=selected_sector,
            selected_range_preset=normalize_range_preset(st.session_state.get("selected_range_preset", "1Y")),
            detail_summary=detail_summary,
            locale=ui_locale,
        )
        if chosen_sector and chosen_sector != selected_sector:
            st.session_state["selected_sector"] = chosen_sector
            st.rerun()
        if chosen_preset != normalize_range_preset(st.session_state.get("selected_range_preset", "1Y")):
            preset_start, preset_end = resolve_range_from_preset(
                max_date=analysis_max_date,
                min_date=analysis_min_date,
                preset=chosen_preset,
            )
            st.session_state["selected_range_preset"] = chosen_preset
            st.session_state["analysis_start_date"] = preset_start
            st.session_state["analysis_end_date"] = preset_end
            st.rerun()


def render_decision_first_sections(
    *,
    current_regime: str,
    regime_is_confirmed: bool,
    growth_val: float | None,
    inflation_val: float | None,
    export_growth_val: float | None = None,
    trade_indicators: dict[str, float] | None = None,
    fx_change: float | None,
    fx_label: str,
    is_provisional: bool,
    theme_mode: str,
    price_status: str,
    macro_status: str,
    investor_flow_status: str,
    investor_flow_fresh: bool,
    investor_flow_profile: str,
    yield_curve_status: str | None,
    signals: list[Any],
    held_sector_options: list[str],
    analysis_canvas_kwargs: dict[str, Any],
    market_id: str = "KR",
    investor_flow_frame: pd.DataFrame | None = None,
    investor_flow_detail: dict[str, Any] | None = None,
    shared_flow_summary_map: dict[str, Any] | None = None,
    flow_short_window: int = 20,
    flow_long_window: int = 60,
    include_analysis_canvas: bool = True,
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> list[str]:
    """Render the upper decision-first stack and return the selected held sectors."""
    render_decision_hero(
        regime=current_regime,
        regime_is_confirmed=regime_is_confirmed,
        growth_val=growth_val,
        inflation_val=inflation_val,
        export_growth_val=export_growth_val,
        trade_indicators=trade_indicators,
        fx_change=fx_change,
        fx_label=fx_label,
        is_provisional=is_provisional,
        theme_mode=theme_mode,
        locale=ui_locale,
    )
    render_status_card_row(
        current_regime=current_regime,
        regime_is_confirmed=regime_is_confirmed,
        price_status=price_status,
        macro_status=macro_status,
        yield_curve_status=yield_curve_status,
        locale=ui_locale,
    )
    if str(market_id).strip().upper() == "KR" and (
        investor_flow_status != "SAMPLE" or any(
        str(getattr(signal, "flow_state", "unavailable")) != "unavailable" for signal in signals
        )
    ):
        render_investor_flow_summary(
            signals=signals,
            investor_flow_status=investor_flow_status,
            investor_flow_fresh=investor_flow_fresh,
            investor_flow_profile=investor_flow_profile,
            investor_flow_frame=investor_flow_frame,
            investor_flow_detail=investor_flow_detail,
            shared_flow_summary_map=shared_flow_summary_map,
            flow_short_window=flow_short_window,
            flow_long_window=flow_long_window,
            locale=ui_locale,
        )
    held_sectors = render_investor_decision_boards(
        signals=signals,
        held_sector_options=held_sector_options,
        locale=ui_locale,
    )
    if include_analysis_canvas:
        render_analysis_canvas(**analysis_canvas_kwargs)
    return held_sectors


def render_summary_tab(
    *,
    tab,
    theme_mode: str,
    top_pick_signals: list[Any],
    signals_filtered: list[Any],
    held_sectors: list[str],
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> None:
    with tab:
        from src.ui.components import (
            render_action_summary,
            render_top_picks_table,
        )

        st.caption(get_ui_text("summary_tab_role_caption", ui_locale))
        with st.container(border=True):
            render_panel_header(
                eyebrow="감사용 스냅샷",
                title="상위 추천 스냅샷",
                description="현재 필터 기준 상위 추천을 표 형태로 다시 검증하는 audit snapshot입니다.",
                badge=f"{min(5, len(top_pick_signals))}개 표시",
            )
            render_top_picks_table(
                top_pick_signals,
                held_sectors=held_sectors,
                limit=5,
                locale=ui_locale,
            )

        st.divider()
        st.subheader("액션 분포")
        with st.container(border=True):
            render_panel_header(
                eyebrow="폭",
                title="액션 분포",
                description="필터 적용 후 유니버스가 액션 사다리에 걸쳐 어떻게 분포하는지 확인하세요.",
            )
            render_action_summary(signals_filtered, theme_mode=theme_mode, locale=ui_locale)


def render_charts_tab(
    *,
    tab,
    signals_filtered: list[Any],
    theme_mode: str,
    is_mobile_client: bool,
) -> None:
    with tab:
        from src.ui.components import (
            render_returns_heatmap,
            render_rs_momentum_bar,
            render_rs_scatter,
        )

        is_hybrid = any(str(getattr(signal, "momentum_method", "")) == "hybrid_return_rank_v1" for signal in signals_filtered)
        render_panel_header(
            eyebrow="Legacy RS Diagnostic" if is_hybrid else "모멘텀 맵",
            title="Legacy RS 진단 패널" if is_hybrid else "RS 산점도 및 모멘텀 막대",
            description="하이브리드 모멘텀 전환 후에도 RS/RS MA 패널은 진단용으로만 유지됩니다." if is_hybrid else "상대강도와 RS 이격도 패널은 동일한 시각적 구조를 사용해 비교가 쉽습니다.",
        )
        st.markdown(
            """
        <div class="compact-note">
        <b>읽는 법</b>
        RS 산점도는 레거시 상대강도 진단용 패널입니다. 하이브리드 모멘텀 활성화 시 이 패널은 action-driving 근거가 아니라 보조 비교용으로만 사용합니다.
        </div>
        """,
            unsafe_allow_html=True,
        )

        with st.expander("차트 해석 상세", expanded=False):
            st.markdown(
                """
        **이 패널은 레거시 RS 진단용입니다.** 하이브리드 모멘텀 canonical action은 `6M ex-1M`, `12M ex-1M`, percentile rank, `price > 200DMA`를 사용합니다.

        **X축 (RS)**: 섹터 종가 대비 벤치마크(KOSPI) 비율입니다. 값이 높을수록 벤치마크 대비 상대강도가 높습니다.

        **Y축 (RS MA)**: RS의 이동평균(기본 20)입니다. RS의 추세를 부드럽게 보여줍니다.

        **점선 대각선**: `RS = RS MA` 기준선입니다. 기준선 위/아래로 모멘텀 방향을 판단합니다.

        | 위치 | 의미 |
        |------|------|
        | 대각선 위 (RS > RS MA) | RS가 평균을 상회해 모멘텀이 가속되는 강세 신호 |
        | 대각선 아래 (RS < RS MA) | RS가 평균을 하회해 모멘텀이 둔화되는 약세 신호 |
        | 오른쪽 | 벤치마크 대비 상대강도가 강한 섹터 |
        | 왼쪽 | 벤치마크 대비 상대강도가 약한 섹터 |
        """
            )

        if signals_filtered:
            benchmark_missing = any(
                "Benchmark Missing" in getattr(signal, "alerts", []) for signal in signals_filtered
            )
            if benchmark_missing:
                st.markdown(
                    '<div class="empty-state-card">'
                    '<h4>벤치마크 데이터 누락</h4>'
                    '<p>KOSPI(1001) 가격 데이터 누락으로 RS 산점도를 계산할 수 없습니다.<br>시장데이터 갱신 후 다시 시도하세요.</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                scatter_height = 520 if is_mobile_client else 700
                scatter_margin = (
                    dict(l=44, r=18, t=56, b=50)
                    if is_mobile_client
                    else dict(l=72, r=32, t=64, b=64)
                )
                fig_scatter = render_rs_scatter(
                    signals_filtered,
                    height=scatter_height,
                    margin=scatter_margin,
                    theme_mode=theme_mode,
                    diagnostic_only=is_hybrid,
                )
                if is_mobile_client:
                    st.plotly_chart(fig_scatter, width="stretch")
                else:
                    _, scatter_col_c, _ = st.columns([0.7, 3.6, 0.7])
                    with scatter_col_c:
                        st.plotly_chart(fig_scatter, width="stretch")

                st.markdown(
                    """
        <div class="compact-note">
        <b>RS 이격도</b>
        RS 이격도는 현재 RS가 이동평균 대비 얼마나 위나 아래에 있는지를 수치로 보여주는 레거시 진단 지표입니다.
        </div>
        """,
                    unsafe_allow_html=True,
                )
                fig_bar = render_rs_momentum_bar(signals_filtered, theme_mode=theme_mode, diagnostic_only=is_hybrid)
                if fig_bar.data:
                    st.plotly_chart(fig_bar, width="stretch")
                else:
                    st.info("RS/RS MA 데이터가 충분하지 않습니다.")

            st.markdown(
                """
        <div class="compact-note">
        <b>수익률 히트맵</b>
        기간별 수익률을 한눈에 비교해 최근 강도 지속 여부를 확인할 수 있습니다.
        </div>
        """,
                unsafe_allow_html=True,
            )
            render_panel_header(
                eyebrow="횡단면",
                title="수익률 히트맵",
                description="대시보드 전체에서 사용하는 동일한 패널 구조로 복수 기간 섹터 수익률을 비교하세요.",
            )
            fig_heatmap = render_returns_heatmap(signals_filtered, theme_mode=theme_mode)
            st.plotly_chart(fig_heatmap, width="stretch")
        else:
            st.markdown(
                '<div class="empty-state-card">'
                '<h4>표시할 데이터 없음</h4>'
                '<p>글로벌 필터 조건에 맞는 신호가 없습니다.<br>선택된 액션 및 국면 필터를 확인해 주세요.</p>'
                '</div>',
                unsafe_allow_html=True,
            )


def render_all_signals_tab(
    *,
    tab,
    signals: list[Any],
    filter_action_global: str,
    filter_regime_only_global: bool,
    current_regime: str,
    held_sectors: list[str],
    position_mode: str,
    show_alerted_only: bool,
    theme_mode: str,
    settings: dict[str, Any],
    etf_map: dict[str, list] | None = None,
    market_id: str = "KR",
    theme_lens_status: str = "UNAVAILABLE",
    theme_lens_rows: list[dict[str, Any]] | None = None,
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> None:
    with tab:
        kr_momentum_only = any(str(getattr(signal, "action_policy", "") or "") == "KR_MOMENTUM_ONLY" for signal in signals)
        active_alert_count = sum(1 for signal in signals if getattr(signal, "alerts", []))
        render_research_page_frame(
            page_key="signals",
            eyebrow="Signal Review",
            title="섹터 액션 보드",
            description="신규 검토, 보유 모니터링, 축소 주의, 변곡 후보를 같은 신호 계약으로 분리합니다.",
            summary_items=[
                {"label": "유니버스", "value": f"{len(signals)}개 섹터"},
                {"label": "필터", "value": get_action_filter_label(filter_action_global, ui_locale)},
                {"label": "보유 범위", "value": format_position_mode_label(position_mode, locale=ui_locale)},
                {"label": "알림", "value": f"{active_alert_count}개 활성"},
            ],
        )
        render_sector_momentum_decision_boards(
            signals,
            held_sectors=held_sectors,
            limit_per_board=4,
            locale=ui_locale,
        )
        if str(market_id).strip().upper() == "KR":
            snapshot = st.session_state.get("_theme_lens_live_snapshot")
            display_status = theme_lens_status
            display_rows = theme_lens_rows or []
            if isinstance(snapshot, dict) and snapshot.get("rows"):
                display_status = str(snapshot.get("status") or display_status)
                display_rows = list(snapshot.get("rows") or display_rows)
            refresh_clicked = render_theme_lens_panel(
                display_rows,
                status=display_status,
                show_refresh_button=True,
            )
            if refresh_clicked:
                from src.data_sources.theme_lens import refresh_theme_lens_etf_ohlcv

                with st.spinner("테마 ETF proxy 갱신 중..."):
                    summary = refresh_theme_lens_etf_ohlcv(
                        asof_date=str(settings.get("market_end_date_str") or date.today().strftime("%Y%m%d"))
                    )
                status = str(summary.get("status", "")).upper()
                refreshed_count = len(list(summary.get("refreshed_codes", [])))
                fetched_count = len(list(summary.get("fetched_codes", [])))
                failed_count = len(dict(summary.get("failed_codes", {})))
                live_rows = list(summary.get("live_rows", []))
                if live_rows:
                    st.session_state["_theme_lens_live_snapshot"] = {
                        "status": str(summary.get("live_status") or status),
                        "rows": live_rows,
                    }
                if status in {"LIVE", "PARTIAL"}:
                    st.toast(f"테마 ETF 갱신 완료: {fetched_count}개 수신, {refreshed_count}개 저장, {failed_count}개 실패")
                    st.rerun()
                else:
                    st.warning("테마 ETF 갱신이 완료되지 않았습니다. 기존 캐시 또는 unavailable 상태를 유지합니다.")
        render_panel_header(
            eyebrow="전체 원장",
            title="전체 섹터 신호 원장",
            description="위 의사결정 보드와 동일한 신호 계약을 상세 행 단위로 검토합니다.",
        )
        if kr_momentum_only:
            st.caption(
                f"적용 필터: 액션={get_action_filter_label(filter_action_global, ui_locale)}, "
                "국면 필터=KR 비활성"
            )
        else:
            st.caption(
                f"적용 필터: 액션={get_action_filter_label(filter_action_global, ui_locale)}, "
                f"현재 국면만 보기={'ON' if filter_regime_only_global else 'OFF'}"
            )
        with st.expander("적합/비적합 판정 기준", expanded=False):
            momentum_method = str(settings.get("momentum_method", "legacy_rs_ma_v0"))
            if kr_momentum_only:
                momentum_line = (
                    "- 최종 `액션`(Strong Buy/Watch/Hold/Avoid)은 `momentum_core_pass × trend_ok`로 계산합니다. 하이브리드 모드에서는 `momentum_core_pass = momentum_rank_pass`, 레거시 모드에서는 `momentum_core_pass = rs_strong`입니다."
                    if momentum_method == "hybrid_return_rank_v1"
                    else "- 최종 `액션`(Strong Buy/Watch/Hold/Avoid)은 `momentum_core_pass × trend_ok`로 계산합니다. 레거시 모드에서는 `momentum_core_pass = rs_strong`입니다."
                )
                st.markdown(
                    f"""
        - KR은 전체 활성 index universe를 대상으로 momentum-first로 순위화합니다.
        - 현재 거시 국면은 `macro_context_regime`로 참고만 하며 KR 액션을 gating하지 않습니다.
        {momentum_line}
        - `macro_fit`, `macro_regime`는 KR에서 compatibility field로만 남아 있으며 active KR sector classification을 drive하지 않습니다.
        - 실증 적합도 카드는 `lag0 nowcast empirical reference`이며 action-driving 신호가 아닙니다.
        """
                )
            else:
                momentum_line = (
                    "- 최종 `액션`(Strong Buy/Watch/Hold/Avoid)은 국면 적합 여부와 하이브리드 모멘텀(`6M ex-1M`, `12M ex-1M`, percentile rank, `price > 200DMA`)을 결합해 계산합니다."
                    if momentum_method == "hybrid_return_rank_v1"
                    else "- 최종 `액션`(Strong Buy/Watch/Hold/Avoid)은 국면 적합 여부와 모멘텀 조건(RS, 추세)을 결합해 계산합니다."
                )
                st.markdown(
                    f"""
        - `국면 적합`은 현재 시점의 확정 국면에서 해당 섹터가 맵핑되는지(`macro_fit`)로 판정합니다.
        - 현재 시점 국면은 `confirmed_regime` 기준입니다. 아직 확정 전이면 잠정 상태로 해석합니다.
        - 맵핑 기준은 `config/sector_map.yml`의 `regimes -> {{국면}} -> sectors`입니다.
        {momentum_line}
        - 실증 적합도 카드는 `lag0 nowcast empirical reference`이며 `PIT` 또는 action-driving 신호가 아닙니다.
        - 남은 classifier 리스크는 named experiment가 pre-registration gate를 통과하지 못하면 기본적으로 freeze/reporting-only로 닫습니다.
        - `Indeterminate` 국면에서는 맵핑 섹터가 없어 전체가 `비적합`으로 표시될 수 있습니다.
        - `2026-02` 문서는 historical snapshot이며, 현재 canonical reference는 `docs/regime-validity-dashboard-parity-current.md`입니다.
        """
                )
        with st.expander("알림 카테고리 설명", expanded=False):
            rsi_overbought = int(settings.get("rsi_overbought", 70))
            rsi_oversold = int(settings.get("rsi_oversold", 30))
            fx_shock_pct = float(settings.get("fx_shock_pct", 3.0))

            st.markdown(
                f"""
        - **Overheat**: 일간 RSI(`rsi_d`)가 `{rsi_overbought}` 이상이면 추가됩니다.
        - **Oversold**: 일간 RSI(`rsi_d`)가 `{rsi_oversold}` 이하이면 추가됩니다.
        - **FX Shock**: `|USD/KRW 변화율| > {fx_shock_pct:.1f}%`이고 수출 섹터이며 현재 액션이 `Strong Buy`이면 알림을 추가하고 액션을 `Watch`로 조정합니다.
        - **Benchmark Missing**: 벤치마크 가격 데이터가 비어 있으면 모든 섹터에 추가됩니다 (액션 `N/A`).
        - **RS Data Insufficient**: 특정 섹터의 RS/RS MA 계산이 불가능할 때 legacy diagnostic에 추가될 수 있습니다.
        - **Momentum History Insufficient**: 하이브리드 모멘텀에 필요한 lookback 또는 200DMA가 부족할 때 추가됩니다 (액션 `N/A`).
        """
            )
            st.caption(
                "참고: `FX Shock`은 신호 계산 시점의 최신 USD/KRW 변화율 기준으로 판정합니다. "
                "USD/KRW 시계열이 2개 미만이면 해당 시점에서는 `FX Shock`이 계산되지 않습니다."
            )

        render_signal_table(
            signals,
            filter_action=filter_action_global,
            filter_regime_only=filter_regime_only_global,
            current_regime=current_regime,
            held_sectors=held_sectors,
            position_mode=position_mode,
            show_alerted_only=show_alerted_only,
            theme_mode=theme_mode,
            etf_map=etf_map,
            locale=ui_locale,
        )


def render_screening_tab(
    *,
    tab,
    signals: list[Any],
    settings: dict[str, Any],
    benchmark_code: str = "1001",
    etf_map: dict[str, list] | None = None,
) -> None:
    """Render the 종목 스크리닝 tab: constituent stocks of Strong Buy sectors."""
    from src.data_sources.krx_stock_screening import (
        load_representative_etf_context,
        load_screened_stocks,
    )

    with tab:
        hybrid_active = str(settings.get("momentum_method", "legacy_rs_ma_v0")) == "hybrid_return_rank_v1"
        strong_buy_count = sum(1 for sig in signals if getattr(sig, "action", "") == "Strong Buy")
        render_research_page_frame(
            page_key="constituents",
            eyebrow="Constituent Map",
            title="섹터맵 실행 참고",
            description="섹터 신호를 바꾸지 않고 Strong Buy 섹터의 구성종목과 대표 ETF 실행 컨텍스트만 분리해 검토합니다.",
            summary_items=[
                {"label": "Strong Buy 섹터", "value": f"{strong_buy_count}개"},
                {"label": "벤치마크", "value": benchmark_code},
                {"label": "모멘텀", "value": "Hybrid" if hybrid_active else "Legacy RS"},
                {"label": "ETF 매핑", "value": f"{len(etf_map or {})}개"},
            ],
        )
        render_panel_header(
            eyebrow="종목 스크리닝",
            title="Strong Buy 섹터 구성종목",
            description="현재 Strong Buy 섹터의 구성종목을 legacy RS·RSI·SMA 기준으로 필터링한 decision-support 리스트" if hybrid_active else "현재 Strong Buy 섹터의 구성종목을 RS·RSI·SMA 기준으로 필터링한 신규 검토 후보 리스트",
        )
        if hybrid_active:
            st.caption("이 스크리닝 탭은 phase 1에서 legacy execution-support 로직을 유지합니다. KR 하이브리드 모멘텀 cutover evidence에는 포함되지 않습니다.")

        # Derive Strong Buy sectors from current signals
        strong_buy_sectors = [
            {"code": sig.index_code, "name": sig.sector_name}
            for sig in signals
            if getattr(sig, "action", "") == "Strong Buy"
        ]

        if not strong_buy_sectors:
            st.markdown(
                '<div class="empty-state-card">'
                '<h4>투자 유니버스 대기 중</h4>'
                '<p>현재 <b>Strong Buy</b> 섹터가 없습니다.<br>매크로 국면이 확정되거나 필터 조건이 완화되면 종목 스크리닝이 활성화됩니다.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        sector_labels = ", ".join(s["name"] for s in strong_buy_sectors)
        st.caption(f"대상 섹터: **{sector_labels}**")

        col_refresh, col_filter = st.columns([1, 3])
        with col_refresh:
            force_refresh = st.button("데이터 갱신", key="screening_refresh")
        with col_filter:
            show_momentum_only = st.toggle(
                "모멘텀 통과 종목만", value=True,
                help="RS > RS_MA AND SMA20 > SMA60 조건을 모두 충족하는 종목만 표시",
            )

        progress_status = st.empty() if force_refresh else None
        progress_bar = st.progress(0, text="구성종목 갱신 준비 중...") if force_refresh else None

        def _update_screening_progress(event: dict[str, Any]) -> None:
            if progress_status is None or progress_bar is None:
                return
            current = int(event.get("current") or 0)
            total = int(event.get("total") or 0)
            stage = str(event.get("stage") or "")
            if total <= 0:
                label = "갱신 대상 종목을 확인하는 중..."
                progress_value = 0
            elif stage == "done":
                label = f"갱신 완료: 총 {total}종목 처리"
                progress_value = 100
            elif stage == "ticker":
                ticker = str(event.get("ticker") or "")
                sector_name = str(event.get("sector_name") or "")
                subject = f"{ticker} · {sector_name}" if sector_name else ticker
                label = f"갱신 대상 총 {total}종목 중 {current}번째 처리 중: {subject}"
                progress_value = min(100, max(0, round(current / total * 100)))
            else:
                label = f"갱신 대상 총 {total}종목 확인 완료"
                progress_value = 0
            progress_status.caption(label)
            progress_bar.progress(progress_value, text=label)

        with st.spinner("구성종목 로딩 중..."):
            status, rows = load_screened_stocks(
                strong_buy_sectors=strong_buy_sectors,
                benchmark_code=benchmark_code,
                settings=settings,
                force_refresh=force_refresh,
                allow_live_fetch=force_refresh,
                progress_callback=_update_screening_progress if force_refresh else None,
            )

        if progress_status is not None and progress_bar is not None:
            progress_status.empty()
            progress_bar.empty()

        if status == "UNAVAILABLE" or not rows:
            st.markdown(
                '<div class="empty-state-card">'
                '<h4>데이터 조회 불가</h4>'
                '<p>구성종목 데이터를 가져올 수 없습니다.<br>주말·공휴일 또는 서비스 API 점검 중일 수 있습니다.<br>평일 장중/장후에 <b>데이터 갱신</b>을 눌러주세요.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        status_label = {
            "LIVE": "현재 세션",
            "CACHED": "캐시(24h)",
            "STALE_CACHE": "만료 캐시 · 갱신 권장",
        }
        st.caption(f"데이터 상태: **{status_label.get(status, status)}** | 총 {len(rows)}개 종목")

        if show_momentum_only:
            rows = [r for r in rows if r.get("momentum_ok")]

        if not rows:
            st.markdown(
                '<div class="empty-state-card">'
                '<h4>모멘텀 충족 종목 없음</h4>'
                '<p>설정된 모멘텀 조건(RS 상승 + SMA 추세 양호)을 충족하는 종목이 현재 없습니다.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        df = pd.DataFrame([
            {
                "종목코드": r["ticker"],
                "종목명": r["name"],
                "섹터": r["sector_name"],
                "RS": r["rs"],
                "RSI": r["rsi"],
                "RS↑": r["rs_strong"],
                "추세↑": r["trend_ok"],
                "200DMA↑": r.get("above_200dma", False),
                "1M(%)": r["ret_1m"],
                "3M(%)": r["ret_3m"],
                "알림": r["alerts"],
            }
            for r in rows
        ])

        def _color_momentum(val):
            if not isinstance(val, (int, float)) or pd.isna(val):
                return ""
            if val > 0:
                return "color: var(--danger); font-weight: 600;"
            elif val < 0:
                return "color: var(--primary); font-weight: 600;"
            return ""

        try:
            styled_df = df.style.map(_color_momentum, subset=["1M(%)", "3M(%)"])
        except AttributeError:
            styled_df = df

        st.dataframe(
            styled_df,
            width="stretch",
            hide_index=True,
            height=min(700, 76 + len(df) * 35),
            column_config={
                "종목코드": st.column_config.TextColumn("종목코드", width="small"),
                "종목명": st.column_config.TextColumn("종목명", width="medium"),
                "섹터": st.column_config.TextColumn("섹터", width="medium"),
                "RS": st.column_config.NumberColumn("RS", format="%.4f"),
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                "RS↑": st.column_config.CheckboxColumn("RS↑", width="small"),
                "추세↑": st.column_config.CheckboxColumn("추세↑", width="small"),
                "200DMA↑": st.column_config.CheckboxColumn("200DMA↑", width="small"),
                "1M(%)": st.column_config.NumberColumn("1M(%)", format="%.1f%%"),
                "3M(%)": st.column_config.NumberColumn("3M(%)", format="%.1f%%"),
                "알림": st.column_config.TextColumn("알림", width="small"),
            },
        )

        with st.spinner("대표 ETF 실행 컨텍스트 로딩 중..."):
            etf_status, etf_rows = load_representative_etf_context(
                strong_buy_sectors=strong_buy_sectors,
                etf_map=etf_map,
                settings=settings,
                force_refresh=force_refresh,
                allow_live_fetch=force_refresh,
            )

        render_panel_header(
            eyebrow="실행 참고",
            title="대표 ETF 실행 컨텍스트",
            description="섹터 판단 결과는 바꾸지 않고, 대표 ETF의 유동성·규모·freshness만 확인하는 보조 레이어입니다.",
            badge={"LIVE": "현재 세션", "CACHED": "캐시(24h)"}.get(etf_status, etf_status),
        )
        st.caption("이 블록은 실행 참고용입니다. 섹터 액션, 순위, Strong Buy 계산에는 영향을 주지 않습니다.")

        if etf_status == "UNAVAILABLE" or not etf_rows:
            st.info("대표 ETF 실행 컨텍스트를 불러오지 못했습니다.")
        else:
            etf_df = pd.DataFrame(
                [
                    {
                        "섹터": row.get("sector_name", ""),
                        "대표 ETF": (
                            f"{row.get('etf_name', '')} ({row.get('etf_code', '')})"
                            if row.get("etf_code")
                            else str(row.get("etf_name", "—"))
                        ),
                        "스타일": row.get("style_tags", ""),
                        "실행 상태": row.get("execution_state", ""),
                        "최근 유동성 금액": row.get("latest_trade_value"),
                        "20D 평균 유동성 금액": row.get("avg_trade_value_20d"),
                        "순자산": row.get("net_assets"),
                        "NAV": row.get("nav"),
                        "기준일": row.get("reference_date", ""),
                        "Freshness": row.get("freshness_label", ""),
                        "비고": row.get("note", ""),
                    }
                    for row in etf_rows
                ]
            )
            st.dataframe(
                etf_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "섹터": st.column_config.TextColumn("섹터", width="medium"),
                    "대표 ETF": st.column_config.TextColumn("대표 ETF", width="medium"),
                    "스타일": st.column_config.TextColumn("스타일", width="small"),
                    "실행 상태": st.column_config.TextColumn("실행 상태", width="small"),
                    "최근 유동성 금액": st.column_config.NumberColumn("최근 유동성 금액", format="%.0f"),
                    "20D 평균 유동성 금액": st.column_config.NumberColumn("20D 평균 유동성 금액", format="%.0f"),
                    "순자산": st.column_config.NumberColumn("순자산", format="%.0f"),
                    "NAV": st.column_config.NumberColumn("NAV", format="%.2f"),
                    "기준일": st.column_config.TextColumn("기준일", width="small"),
                    "Freshness": st.column_config.TextColumn("Freshness", width="small"),
                    "비고": st.column_config.TextColumn("비고", width="medium"),
                },
            )

        # CSV download
        csv = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="CSV 다운로드",
            data=csv.encode("utf-8-sig"),
            file_name=f"screening_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


def render_investor_flow_tab(
    *,
    tab,
    signals: list[Any],
    investor_flow_frame: pd.DataFrame,
    investor_flow_status: str,
    investor_flow_fresh: bool,
    investor_flow_profile: str,
    investor_flow_detail: dict[str, Any] | None = None,
    shared_flow_summary_map: dict[str, Any] | None = None,
    flow_short_window: int = 20,
    flow_long_window: int = 60,
    market_id: str = "KR",
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> None:
    with tab:
        normalized_market = str(market_id).strip().upper() or "KR"
        flow_frame_rows = len(investor_flow_frame) if isinstance(investor_flow_frame, pd.DataFrame) else 0
        render_research_page_frame(
            page_key="flow",
            eyebrow="Flow Context",
            title="섹터 수급 맥락",
            description="수급 또는 ETF proxy flow를 action-driving 신호가 아닌 보조 맥락으로 분리해 보여줍니다.",
            summary_items=[
                {"label": "시장", "value": normalized_market},
                {"label": "상태", "value": investor_flow_status},
                {"label": "프로파일", "value": get_flow_profile_label(investor_flow_profile, ui_locale) if normalized_market == "KR" else "ETF proxy"},
                {"label": "레코드", "value": f"{flow_frame_rows:,}건"},
            ],
        )
        if normalized_market == "US":
            state_labels = {
                "elevated": "활동 확대",
                "normal": "보통",
                "subdued": "활동 둔화",
                "unavailable": "사용 불가",
            }
            render_panel_header(
                eyebrow="Flow Proxies",
                title="US Flow Proxies",
                description="공개 ETF 데이터 기반 프록시입니다. 한국식 투자자별 수급과 1:1 대응이 아니라, 섹터 ETF 자금/거래 압력 확인 레이어로 보세요.",
                badge=investor_flow_status,
            )
            st.info("이 탭은 `ETF wrapper-flow proxy`를 보여 줍니다. participant-segmented 현물 순매수 데이터는 아닙니다.")
            if investor_flow_frame.empty:
                st.info("US flow proxy 데이터를 불러오지 못했습니다.")
                return

            latest_snapshot = investor_flow_frame.sort_values(
                by=["activity_zscore", "dollar_volume"],
                ascending=[False, False],
            ).copy()
            latest_snapshot["state_label"] = latest_snapshot["activity_state"].astype(str).map(
                lambda value: state_labels.get(str(value), str(value))
            )
            st.dataframe(
                latest_snapshot[
                    [
                        "sector_name",
                        "state_label",
                        "activity_zscore",
                        "dollar_volume",
                        "dollar_volume_short_mean",
                        "dollar_volume_long_mean",
                        "shares_outstanding",
                        "assets_under_management",
                        "nav",
                        "net_cash_amount",
                    ]
                ].rename(
                    columns={
                        "sector_name": "Sector",
                        "state_label": "Activity state",
                        "activity_zscore": "Activity z-score",
                        "dollar_volume": "Latest $Vol",
                        "dollar_volume_short_mean": "5D avg $Vol",
                        "dollar_volume_long_mean": "20D avg $Vol",
                        "shares_outstanding": "Shares Out",
                        "assets_under_management": "AUM",
                        "nav": "NAV",
                        "net_cash_amount": "Net Cash",
                    }
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "Sector": st.column_config.TextColumn("Sector", width="medium"),
                    "Activity state": st.column_config.TextColumn("Activity state", width="small"),
                    "Activity z-score": st.column_config.NumberColumn("Activity z-score", format="%.2f"),
                    "Latest $Vol": st.column_config.NumberColumn("Latest $Vol", format="%.0f"),
                    "5D avg $Vol": st.column_config.NumberColumn("5D avg $Vol", format="%.0f"),
                    "20D avg $Vol": st.column_config.NumberColumn("20D avg $Vol", format="%.0f"),
                    "Shares Out": st.column_config.NumberColumn("Shares Out", format="%.0f"),
                    "AUM": st.column_config.NumberColumn("AUM", format="%.0f"),
                    "NAV": st.column_config.NumberColumn("NAV", format="%.2f"),
                    "Net Cash": st.column_config.NumberColumn("Net Cash", format="%.0f"),
                },
            )
            latest_date = investor_flow_frame.index.max()
            ref_date_str = pd.Timestamp(latest_date).strftime("%Y-%m-%d")
            source_label = {"LIVE": "현재 세션", "CACHED": "캐시", "SAMPLE": "샘플"}.get(
                str(investor_flow_status).upper(), investor_flow_status
            )
            st.caption(f"데이터 기준일: {ref_date_str} · 소스: {source_label}")
            missing_tickers = [str(code) for code in investor_flow_frame.attrs.get("missing_tickers", []) if str(code).strip()]
            if missing_tickers:
                preview = ", ".join(missing_tickers[:4])
                suffix = "" if len(missing_tickers) <= 4 else ", ..."
                st.caption(f"누락 섹터: {preview}{suffix}")
            if not investor_flow_fresh:
                st.caption("최근 영업일까지 완전 커버된 값이 아닐 수 있습니다.")

            ownership_context = dict((investor_flow_detail or {}).get("ownership_context") or {})
            ici_context = dict(ownership_context.get("ici_weekly_flows") or {})
            ici_table = ici_context.get("table")
            if isinstance(ici_table, pd.DataFrame) and not ici_table.empty:
                render_panel_header(
                    eyebrow="ICI",
                    title="Weekly ETF Net Issuance",
                    description="ICI 주간 ETF 순발행 추정치입니다. 섹터별 flow가 아니라 미국 ETF 시장 전체 흐름 컨텍스트입니다.",
                    badge=str(ici_context.get("as_of", "")),
                )
                st.dataframe(
                    ici_table.rename(columns={"category": "Category", "value": "Net issuance (USD mn)"}),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Category": st.column_config.TextColumn("Category", width="medium"),
                        "Net issuance (USD mn)": st.column_config.NumberColumn("Net issuance (USD mn)", format="%.0f"),
                    },
                )

            positions_context = dict(ownership_context.get("sec_13f_positions") or {})
            positions_table = positions_context.get("table")
            if isinstance(positions_table, pd.DataFrame) and not positions_table.empty:
                render_panel_header(
                    eyebrow="SEC 13F",
                    title="Latest 13F Sector ETF Positioning",
                    description="최신 SEC 13F dataset에서 sector ETF 자체에 대한 institutional position을 집계한 값입니다. 분기 지연이 있는 ownership context입니다.",
                    badge=str(positions_context.get("dataset_label", "Latest 13F")),
                )
                st.dataframe(
                    positions_table.rename(
                        columns={
                            "sector_code": "ETF",
                            "sector_name": "Sector",
                            "filing_count": "Filings",
                            "manager_value_total_usd": "Reported value (USD)",
                            "manager_shares_total": "Reported shares",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "ETF": st.column_config.TextColumn("ETF", width="small"),
                        "Sector": st.column_config.TextColumn("Sector", width="medium"),
                        "cusip": st.column_config.TextColumn("CUSIP", width="small"),
                        "Filings": st.column_config.NumberColumn("Filings", format="%d"),
                        "Reported value (USD)": st.column_config.NumberColumn("Reported value (USD)", format="%.0f"),
                        "Reported shares": st.column_config.NumberColumn("Reported shares", format="%.0f"),
                    },
                )

            events_context = dict(ownership_context.get("sec_13dg_events") or {})
            events_table = events_context.get("table")
            if isinstance(events_table, pd.DataFrame) and not events_table.empty:
                render_panel_header(
                    eyebrow="SEC 13D/13G",
                    title="Recent 13D/13G Events",
                    description="최근 13D/13G 이벤트입니다. sector ETF top holdings 기준의 event-driven beneficial ownership context이며, participant flow가 아닙니다.",
                    badge=f"{int(events_context.get('lookback_days', 0))}D",
                )
                st.dataframe(
                    events_table.rename(
                        columns={
                            "sector_code": "ETF",
                            "sector_name": "Sector",
                            "matched_top_holdings": "Matched holdings",
                            "recent_13dg_events": "13D/G events",
                            "sample_events": "Examples",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "ETF": st.column_config.TextColumn("ETF", width="small"),
                        "Sector": st.column_config.TextColumn("Sector", width="medium"),
                        "Matched holdings": st.column_config.NumberColumn("Matched holdings", format="%d"),
                        "13D/G events": st.column_config.NumberColumn("13D/G events", format="%d"),
                        "Examples": st.column_config.TextColumn("Examples", width="large"),
                    },
                )

            form_sho_context = dict(ownership_context.get("form_sho_context") or {})
            if form_sho_context:
                render_panel_header(
                    eyebrow="Form SHO",
                    title="Form SHO Context",
                    description="SEC short-position transparency 레이어입니다. 현재는 공식 정책/availability 컨텍스트만 표시합니다.",
                    badge=str(form_sho_context.get("status", "")),
                )
                st.caption(str(form_sho_context.get("note", "")))

            context_errors = dict(ownership_context.get("errors") or {})
            if context_errors:
                preview = "; ".join(f"{key}: {value}" for key, value in sorted(context_errors.items()))
                st.caption(f"추가 컨텍스트 일부 로드 실패: {preview}")
            return

        if investor_flow_frame.empty:
            st.info(get_ui_text("flow_tab_empty", ui_locale))
            return

        latest_date = investor_flow_frame.index.max()
        latest_amounts = _build_kr_latest_sector_flow_amounts(investor_flow_frame)
        render_panel_header(
            eyebrow="Latest snapshot",
            title="섹터별 수급 금액",
            description="최근 기준일의 외국인, 기관, 개인 순매수 금액입니다. 큰 값은 조/억 단위로 표시합니다.",
            badge=str(investor_flow_status),
        )
        st.dataframe(
            latest_amounts,
            width="stretch",
            hide_index=True,
            column_config={
                "Sector": st.column_config.TextColumn(get_ui_text("col_sector", ui_locale), width="medium"),
                "외국인": st.column_config.TextColumn("외국인", width="small"),
                "기관": st.column_config.TextColumn("기관", width="small"),
                "개인": st.column_config.TextColumn("개인", width="small"),
                "합계": st.column_config.TextColumn("합계", width="small"),
            },
        )

        render_panel_header(
            eyebrow="Trend",
            title="기간별 수급 추이",
            description="전체 섹터를 한 번에 놓고 외국인, 기관, 개인 순매수 금액 변화를 비교합니다.",
        )
        st.plotly_chart(_build_kr_sector_flow_trend_figure(investor_flow_frame), width="stretch")

        try:
            ref_date_str = pd.Timestamp(latest_date).strftime("%Y-%m-%d")
        except Exception:
            ref_date_str = str(latest_date)
        source_label = {"LIVE": "현재 세션", "CACHED": "캐시", "SAMPLE": "샘플"}.get(
            str(investor_flow_status).upper(), investor_flow_status
        )
        st.caption(f"데이터 기준일: {ref_date_str} · 소스: {source_label}")
        if not investor_flow_fresh:
            st.caption(get_ui_text("flow_unavailable", ui_locale))


def _build_etf_map(sector_map: dict | None) -> dict[str, list]:
    """Build {index_code: [{"code":..., "name":...}, ...]} from sector_map config."""
    from src.data_sources.sector_etf_mapping import build_effective_etf_map

    return build_effective_etf_map(sector_map)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_monitoring_data(market_id: str) -> dict:
    """Load dataset collection monitoring data with a 60-second TTL cache."""
    from src.data_sources.warehouse import (
        COLLECTION_HISTORY_DATASETS,
        read_dataset_data_bounds,
        read_dataset_status,
        read_collection_run_history,
    )
    dataset_order = list(COLLECTION_HISTORY_DATASETS)
    statuses = {
        dataset: read_dataset_status(dataset, market=market_id)
        for dataset in dataset_order
    }
    history = read_collection_run_history(
        market=market_id,
        reasons=("manual_refresh",),
        sample_per_dataset=True,
        sample_size=10,
    )
    bounds = {
        dataset: read_dataset_data_bounds(dataset, market=market_id)
        for dataset in dataset_order
    }
    macro_providers = {
        str(provider).strip().upper()
        for provider in (
            history.loc[history["dataset"].astype(str).eq("macro_data"), "provider"].tolist()
            if not history.empty and {"dataset", "provider"}.issubset(history.columns)
            else []
        )
        if str(provider).strip()
    }
    macro_status_provider = str(statuses.get("macro_data", {}).get("provider", "") or "").strip().upper()
    if macro_status_provider:
        macro_providers.add(macro_status_provider)
    for provider in sorted(macro_providers):
        bounds[f"macro_data:{provider}"] = read_dataset_data_bounds(
            "macro_data",
            market=market_id,
            provider=provider,
        )
    return {"statuses": statuses, "history": history, "bounds": bounds, "dataset_order": dataset_order}


def clear_monitoring_data_cache() -> None:
    """Clear cached warehouse monitoring data after manual collection actions."""
    _cached_monitoring_data.clear()


def render_monitoring_tab(
    *,
    tab,
    market_id: str = "KR",
    investor_flow_status: str | None = None,
    investor_flow_fresh: bool | None = None,
    investor_flow_detail: dict[str, Any] | None = None,
    investor_flow_frame: pd.DataFrame | None = None,
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> None:
    """Render the dataset collection history and health page."""
    with tab:
        render_research_page_frame(
            page_key="quality",
            eyebrow="Data Quality",
            title="데이터 수집 이력 관리",
            description="시장, 매크로, 수급 데이터의 수집 상태와 오류 이력을 같은 기준으로 점검합니다.",
            summary_items=[
                {"label": "시장", "value": market_id},
                {"label": "데이터셋", "value": "3개"},
                {"label": "로그", "value": "최신 10건"},
                {"label": "기준", "value": "warehouse"},
            ],
        )
        render_panel_header(
            eyebrow="운영 현황",
            title="데이터셋별 수집 상태",
            description="시장데이터, 매크로데이터, 수급데이터의 최신 실행 상태를 같은 열 구조로 비교합니다.",
            badge=market_id,
        )

        data = _cached_monitoring_data(str(market_id).strip().upper())
        dataset_order: list[str] = list(data.get("dataset_order") or [])
        statuses: dict[str, dict[str, Any]] = {
            str(key): dict(value or {})
            for key, value in dict(data.get("statuses") or {}).items()
        }
        if not statuses and isinstance(data.get("warm"), dict):
            statuses["investor_flow"] = dict(data.get("warm") or {})
        runtime_detail = dict(investor_flow_detail or {})
        runtime_detail_status = str(runtime_detail.pop("status", "") or "").strip()
        runtime_detail_coverage = runtime_detail.pop("coverage_complete", None)
        flow_status = dict(statuses.get("investor_flow") or {})
        if runtime_detail:
            flow_status.update(runtime_detail)
        if str(investor_flow_status or "").strip():
            flow_status["status"] = str(investor_flow_status)
        elif not str(flow_status.get("status", "")).strip() and runtime_detail_status:
            flow_status["status"] = runtime_detail_status
        if investor_flow_fresh is not None:
            flow_status["coverage_complete"] = bool(investor_flow_fresh)
        elif "coverage_complete" not in flow_status and runtime_detail_coverage is not None:
            flow_status["coverage_complete"] = bool(runtime_detail_coverage)
        statuses["investor_flow"] = flow_status
        if not dataset_order:
            dataset_order = list(dict.fromkeys([*statuses.keys(), "market_prices", "macro_data", "investor_flow"]))

        history: pd.DataFrame = data["history"]
        used_runtime_history_fallback = False
        has_flow_history = (
            "dataset" in history.columns
            and history["dataset"].astype(str).eq("investor_flow").any()
        )
        if (history.empty or not has_flow_history) and runtime_detail:
            requested_start = str(runtime_detail.get("anchor_start", "") or "").strip()
            requested_end = str(
                runtime_detail.get("end", runtime_detail.get("watermark_key", ""))
                or ""
            ).strip()
            predicted = int(runtime_detail.get("predicted_requests", 0) or 0)
            processed = int(runtime_detail.get("processed_requests", 0) or 0)
            coverage_complete = bool(runtime_detail.get("coverage_complete", investor_flow_fresh))
            row_count_value = runtime_detail.get("row_count", pd.NA)
            completion_pct = (
                round(processed / predicted * 100, 1)
                if predicted > 0
                else (100.0 if coverage_complete else 0.0)
            )
            history = pd.DataFrame(
                [
                    {
                        "created_at": pd.Timestamp.utcnow(),
                        "dataset": "investor_flow",
                        "reason": str(runtime_detail.get("reason", "runtime_snapshot") or "runtime_snapshot"),
                        "requested_start": requested_start,
                        "requested_end": requested_end,
                        "status": str(investor_flow_status or runtime_detail.get("status", "SAMPLE")),
                        "coverage_complete": coverage_complete,
                        "aborted": bool(runtime_detail.get("aborted", False)),
                        "abort_reason": str(runtime_detail.get("abort_reason", "") or ""),
                        "failed_days": list(runtime_detail.get("failed_days", []) or []),
                        "failed_codes": dict(runtime_detail.get("failed_codes", {}) or {}),
                        "provider": str(runtime_detail.get("provider", "runtime") or "runtime"),
                        "predicted_requests": predicted,
                        "processed_requests": processed,
                        "row_count": row_count_value,
                        "completion_pct": completion_pct,
                        "sample_bucket": "latest",
                    }
                ]
            )
            if not data["history"].empty:
                history = pd.concat([history, data["history"]], ignore_index=True)
            used_runtime_history_fallback = True

        bounds = dict(data.get("bounds") or {})
        status_rows = _format_collection_overview_rows(
            statuses=statuses,
            history=history,
            bounds=bounds,
            dataset_order=dataset_order,
        )
        st.dataframe(
            status_rows,
            hide_index=True,
            width="stretch",
            column_config={
                "데이터": st.column_config.TextColumn("데이터", width="small"),
                "상태": st.column_config.TextColumn("상태", width="small"),
                "마지막 갱신": st.column_config.TextColumn("마지막 갱신", width="medium"),
                "보유기간": st.column_config.TextColumn("보유기간", width="medium"),
                "최근 요청": st.column_config.TextColumn("최근 요청", width="medium"),
                "실패/주의": st.column_config.TextColumn("실패/주의", width="large"),
                "provider": st.column_config.TextColumn("provider", width="small"),
                "저장행수": st.column_config.NumberColumn("저장행수", format="%d"),
            },
        )

        render_panel_header(
            eyebrow="오류 점검",
            title="데이터셋별 최신 오류",
            description="각 데이터셋의 마지막 수집 실행에서 보존된 미수집일, 오류 항목, 중단 사유만 표시합니다.",
        )
        error_rows = _format_dataset_error_rows(statuses, dataset_order)
        if error_rows.empty:
            st.success("최근 수집 실행에서 보존된 오류 없음")
        else:
            st.dataframe(error_rows, hide_index=True, width="stretch")

        render_panel_header(
            eyebrow="이력",
            title="최근 수집 실행 로그",
            description="수집일시 내림차순 기준으로 데이터셋별 최신 10건만 확인합니다.",
        )
        if history.empty:
            st.info("수집 이력이 없습니다.")
        else:
            if used_runtime_history_fallback:
                st.info("warehouse 수급 수집 이력이 비어 있어 현재 세션 runtime snapshot 메타데이터를 함께 표시합니다.")
            if "dataset" not in history.columns:
                history["dataset"] = "investor_flow"
            dataset_order = data.get("dataset_order") or list(dict.fromkeys(history["dataset"].astype(str).tolist()))
            for dataset in dataset_order:
                dataset_history = history[history["dataset"].astype(str).eq(dataset)]
                sample = _format_collection_history_sample(history, dataset)
                label = _format_collection_dataset_label(dataset)
                render_panel_header(
                    eyebrow="로그",
                    title=f"{label}",
                    description="수집일시 내림차순 최신 10건입니다.",
                    badge=f"{len(dataset_history):,}건",
                )
                if sample.empty:
                    st.info(f"{label} 수집 이력이 없습니다.")
                    continue
                st.dataframe(
                    sample.reset_index(drop=True),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "수집일시": st.column_config.DatetimeColumn("수집일시", format="YYYY-MM-DD HH:mm:ss"),
                        "요청범위": st.column_config.TextColumn("요청범위", width="medium"),
                        "상태": st.column_config.TextColumn("상태", width="small"),
                        "커버리지": st.column_config.TextColumn("커버리지", width="small"),
                        "중단": st.column_config.TextColumn("중단", width="small"),
                        "오류요약": st.column_config.TextColumn("오류요약", width="large"),
                        "완료율(%)": st.column_config.NumberColumn("완료율(%)", format="%.1f"),
                        "provider": st.column_config.TextColumn("provider", width="small"),
                        "저장행수": st.column_config.NumberColumn("저장행수", format="%d"),
                    },
                )

        st.caption("데이터 소스: warehouse ingest_runs / ingest_watermarks · 60초 캐시")


def render_dashboard_tabs(
    *,
    current_regime: str,
    theme_mode: str,
    signals: list[Any],
    held_sectors: list[str],
    settings: dict[str, Any],
    is_mobile_client: bool,
    market_id: str = "KR",
    investor_flow_status: str = "SAMPLE",
    investor_flow_fresh: bool = False,
    investor_flow_profile: str = "foreign_lead",
    investor_flow_frame: pd.DataFrame | None = None,
    investor_flow_detail: dict[str, Any] | None = None,
    shared_flow_summary_map: dict[str, Any] | None = None,
    sector_map: dict[str, Any] | None = None,
    ui_locale: str = DEFAULT_UI_LOCALE,
    selected_page_id: str = DEFAULT_DASHBOARD_PAGE_ID,
    theme_lens_status: str = "UNAVAILABLE",
    theme_lens_rows: list[dict[str, Any]] | None = None,
) -> None:
    etf_map = _build_etf_map(sector_map)
    normalized_market = str(market_id).strip().upper() or "KR"
    selected_page_id = normalize_dashboard_page_id(selected_page_id, normalized_market)
    if selected_page_id == "research":
        return

    page_container = st.container()
    if selected_page_id == "overview":
        filter_action_global, filter_regime_only_global, position_mode, show_alerted_only = render_top_bar_filters(
            current_regime=current_regime,
            action_options=[ALL_ACTION_KEY, "Strong Buy", "Watch", "Hold", "Avoid", "N/A"],
            enable_regime_filter=normalized_market != "KR",
            is_mobile=is_mobile_client,
            locale=ui_locale,
        )
        signals_filtered = filter_signals_for_display(
            list(signals),
            filter_action=filter_action_global,
            filter_regime_only=filter_regime_only_global,
            current_regime=current_regime,
            held_sectors=held_sectors,
            position_mode=position_mode,
            show_alerted_only=show_alerted_only,
        )
        if normalized_market == "KR" and any(
            str(getattr(signal, "momentum_method", "")) == "hybrid_return_rank_v1"
            for signal in signals_filtered
        ):
            top_pick_signals = sorted(signals_filtered, key=top_pick_sort_key)
        else:
            top_pick_signals = sorted(
                signals_filtered,
                key=lambda signal: signal_display_sort_key(signal, held_sectors),
            )
        render_summary_tab(
            tab=page_container,
            theme_mode=theme_mode,
            top_pick_signals=top_pick_signals,
            signals_filtered=signals_filtered,
            held_sectors=held_sectors,
            ui_locale=ui_locale,
        )
    elif selected_page_id == "signals":
        filter_action_global, filter_regime_only_global, position_mode, show_alerted_only = render_top_bar_filters(
            current_regime=current_regime,
            action_options=[ALL_ACTION_KEY, "Strong Buy", "Watch", "Hold", "Avoid", "N/A"],
            enable_regime_filter=normalized_market != "KR",
            is_mobile=is_mobile_client,
            locale=ui_locale,
        )
        render_all_signals_tab(
            tab=page_container,
            signals=signals,
            filter_action_global=filter_action_global,
            filter_regime_only_global=filter_regime_only_global,
            current_regime=current_regime,
            held_sectors=held_sectors,
            position_mode=position_mode,
            show_alerted_only=show_alerted_only,
            theme_mode=theme_mode,
            settings=settings,
            etf_map=etf_map,
            market_id=normalized_market,
            theme_lens_status=theme_lens_status,
            theme_lens_rows=theme_lens_rows,
            ui_locale=ui_locale,
        )
    elif selected_page_id == "constituents":
        render_screening_tab(
            tab=page_container,
            signals=signals,
            settings=settings,
            benchmark_code=str(settings.get("benchmark_code", "1001")),
            etf_map=etf_map,
        )
    elif selected_page_id == "flow":
        render_investor_flow_tab(
            tab=page_container,
            signals=signals,
            investor_flow_frame=investor_flow_frame if investor_flow_frame is not None else pd.DataFrame(),
            investor_flow_status=investor_flow_status,
            investor_flow_fresh=investor_flow_fresh,
            investor_flow_profile=investor_flow_profile,
            investor_flow_detail=investor_flow_detail,
            shared_flow_summary_map=shared_flow_summary_map,
            flow_short_window=int(settings.get("investor_flow_short_window", 20)),
            flow_long_window=int(settings.get("investor_flow_long_window", 60)),
            market_id=normalized_market,
            ui_locale=ui_locale,
        )
    elif selected_page_id == "quality":
        render_monitoring_tab(
            tab=page_container,
            market_id=str(market_id),
            investor_flow_status=investor_flow_status,
            investor_flow_fresh=investor_flow_fresh,
            investor_flow_detail=investor_flow_detail,
            investor_flow_frame=investor_flow_frame if investor_flow_frame is not None else pd.DataFrame(),
            ui_locale=ui_locale,
        )
