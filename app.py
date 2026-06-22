"""
Korea Sector Rotation Dashboard
Streamlit SPA entrypoint.

This module intentionally keeps compatibility exports for tests while delegating
dashboard orchestration into `src.dashboard`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from config.markets import get_market_profile, load_market_configs
from config.theme import THEME_SESSION_KEY, get_theme_mode
from src.dashboard import data as dashboard_data_module
from src.dashboard.analysis import (
    build_cycle_segments,
    build_heatmap_display,
    build_monthly_return_views,
    build_monthly_sector_returns,
    build_prices_wide,
    build_sector_name_map,
    extract_heatmap_selection,
    filter_monthly_frame_for_analysis,
    filter_prices_for_phase,
    rs_divergence_pct,
    top_pick_sort_key,
)
from src.dashboard.data import (
    cached_analysis_sector_prices,
    cached_api_preflight,
    get_krx_provider_configured,
    get_krx_provider_effective,
    get_investor_flow_artifact_key,
    get_macro_artifact_key,
    get_macro_cache_token,
    get_price_artifact_key,
    get_price_cache_token,
    is_mobile_client,
    load_analysis_sector_prices_from_cache as dashboard_load_analysis_sector_prices_from_cache,
    load_api_key,
    maybe_schedule_startup_krx_warm,
    probe_investor_flow_status,
    probe_macro_status,
    probe_market_status,
    render_dashboard_status_banner,
    resolve_market_end_date,
    show_notice_toast,
    configure_dashboard_env,
    load_dashboard_runtime_data,
)
from src.dashboard.runtime import (
    invalidate_dashboard_caches,
    run_investor_flow_refresh,
    run_macro_refresh,
    run_market_refresh,
)
from src.dashboard.state import (
    apply_stock_lookup_result,
    apply_analysis_toolbar_selection,
    build_stock_lookup_display_model,
    ensure_analysis_bounds,
    ensure_session_defaults,
    ensure_visible_month_selection,
    normalize_session_state,
    parse_asof_default,
)
from src.dashboard.theme_taxonomy_adapter import build_taxonomy_dashboard_model
from src.dashboard.tabs import (
    build_dashboard_page_options,
    normalize_dashboard_page_id,
    render_analysis_canvas,
    render_dashboard_tabs,
    render_sidebar_controls,
    resolve_dashboard_page_title,
)
from src.dashboard.types import AnalysisWindow, DashboardContext, DashboardDataBundle
from src.macro.series_utils import (
    build_enabled_sector_export_aliases,
    build_sector_trade_proxy_lens,
    extract_kr_export_growth_yoy,
    extract_macro_series,
    extract_trade_indicators,
    is_macro_alias_enabled,
)
from src.data_sources.warehouse import (
    read_market_prices,
    read_active_index_dimension,
)
from src.data_sources.theme_lens import get_theme_lens_artifact_key
from src.data_sources.stock_sector_lookup import resolve_stock_to_sector
from src.ui.copy import ALL_ACTION_KEY, DEFAULT_UI_LOCALE, normalize_locale
from src.ui.components import (
    HEATMAP_PALETTE_OPTIONS,
    infer_range_preset,
    normalize_range_preset,
    render_analysis_toolbar,
    render_page_header,
    render_research_page_frame,
    render_stock_lookup_control,
    render_toss_overview_dashboard,
    resolve_range_from_preset,
    build_sector_detail_figure,
    render_progress_panel,
)
from src.ui.data_status import (
    get_button_states,
    resolve_dashboard_status_banner,
    resolve_macro_status_display,
    resolve_price_cache_banner_case,
)
from src.ui.styles import inject_css

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _make_progress_callback(*hosts):
    def _callback(event: dict[str, object]) -> None:
        for host in hosts:
            render_progress_panel(host, event)

    return _callback


def _resolve_shared_flow_summary_map(runtime_payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(runtime_payload, dict):
        return {}
    return dict(runtime_payload.get("shared_flow_summary_map") or {})


def _build_lookup_error_result(*, market_id: str, query: str, exc: Exception) -> dict[str, object]:
    return {
        "status": "error",
        "market": market_id,
        "query": query,
        "normalized_query": "",
        "matched_symbol": "",
        "matched_name": "",
        "sector_code": "",
        "sector_name": "",
        "resolution_kind": "",
        "source": "",
        "confidence": "",
        "explanation": f"Stock-sector lookup failed: {exc}",
        "canonicalization_applied": False,
        "canonicalization_basis": "not_applicable",
        "match_effective_date": "",
        "match_date_mode": "not_applicable",
        "matched_sector_candidates": [],
    }


def _resolve_stock_lookup_result(
    *,
    market_id: str,
    query: str,
    sector_map: dict,
    asof_date: object,
) -> dict[str, object]:
    try:
        lookup_sector_universe_rows = (
            read_active_index_dimension(market=market_id).to_dict("records")
            if market_id == "KR"
            else None
        )
        return resolve_stock_to_sector(
            query,
            market_id,
            sector_map,
            asof_date=asof_date,
            sector_universe_rows=lookup_sector_universe_rows,
        )
    except Exception as exc:
        return _build_lookup_error_result(
            market_id=market_id,
            query=query,
            exc=exc,
        )


# Backward-compatible analysis exports kept for tests.
_build_cycle_segments = build_cycle_segments
_build_heatmap_display = build_heatmap_display
_build_monthly_return_views = build_monthly_return_views
_build_monthly_sector_returns = build_monthly_sector_returns
_build_prices_wide = build_prices_wide
_build_sector_name_map = build_sector_name_map
_extract_heatmap_selection = extract_heatmap_selection
_filter_monthly_frame_for_analysis = filter_monthly_frame_for_analysis
_filter_prices_for_phase = filter_prices_for_phase
_rs_divergence_pct = rs_divergence_pct
_top_pick_sort_key = top_pick_sort_key

st.set_page_config(
    page_title="Sector Rotation Dashboard",
    layout="wide",
    initial_sidebar_state="auto",
)


def _select_navigation_page(market_id: str, page_id: str) -> None:
    st.session_state["_nav_market_id"] = market_id
    st.session_state["_nav_page_id"] = page_id


def _detect_navigation_from_url() -> tuple[str, str] | None:
    """Infer the requested Streamlit page from the active browser URL."""
    url = str(getattr(st.context, "url", "") or "")
    path = urlparse(url).path.strip("/")
    if not path:
        return None
    for market_id in ("KR", "US"):
        for option in build_dashboard_page_options(market_id):
            if path == option.url_path:
                return market_id, option.page_id
        if path == market_id.lower():
            return market_id, "overview"
    return None


def _build_navigation_pages() -> dict[str, list]:
    requested = _detect_navigation_from_url()
    if requested is not None:
        st.session_state["_nav_market_id"], st.session_state["_nav_page_id"] = requested

    active_market = str(st.session_state.get("_nav_market_id", "KR")).strip().upper() or "KR"
    if active_market not in {"KR", "US"}:
        active_market = "KR"

    pages: dict[str, list] = {
        "시장": [
            st.Page(
                lambda market_id=market_id: _select_navigation_page(market_id, "overview"),
                title=market_id,
                url_path=market_id.lower(),
                default=market_id == active_market,
            )
            for market_id in ("KR", "US")
        ],
        f"{active_market} 페이지": [],
    }
    for option in build_dashboard_page_options(active_market):
        pages[f"{active_market} 페이지"].append(
            st.Page(
                lambda market_id=active_market, page_id=option.page_id: _select_navigation_page(market_id, page_id),
                title=option.label,
                url_path=option.url_path,
            )
        )
    return pages


st.navigation(_build_navigation_pages(), position="sidebar", expanded=True).run()


def _config_cache_token(market_id: str) -> tuple[tuple[str, float], ...]:
    profile = get_market_profile(market_id)
    paths = [
        profile.settings_base_path,
        profile.sector_map_path,
        profile.macro_series_path,
    ]
    if profile.settings_override_path is not None:
        paths.append(profile.settings_override_path)
    return tuple(
        (str(path), path.stat().st_mtime if path.exists() else 0.0)
        for path in paths
    )


@st.cache_data(ttl=3600)
def _load_config(market_id: str, config_cache_token: tuple[tuple[str, float], ...]) -> tuple[dict, dict, dict, object]:
    """Load YAML config files. Cached for 1 hour."""
    del config_cache_token
    return load_market_configs(market_id)


selected_market_id = str(st.session_state.get("_nav_market_id", "KR")).strip().upper() or "KR"
selected_dashboard_page = normalize_dashboard_page_id(
    st.session_state.get("_nav_page_id", "overview"),
    selected_market_id,
)
st.session_state["_nav_page_id"] = selected_dashboard_page
if str(st.session_state.get("market_id", selected_market_id)).strip().upper() != selected_market_id:
    st.session_state["market_id"] = selected_market_id
    invalidate_dashboard_caches("all")
else:
    st.session_state["market_id"] = selected_market_id
settings, sector_map, macro_series_cfg, market_profile = _load_config(
    selected_market_id,
    _config_cache_token(selected_market_id),
)
CACHE_TTL = int(settings.get("cache_ttl", 21600))
CURATED_SECTOR_PRICES_PATH = Path(
    "data/curated/sector_prices_us.parquet" if selected_market_id == "US" else "data/curated/sector_prices.parquet"
)

configure_dashboard_env(
    settings_obj=settings,
    sector_map_obj=sector_map,
    macro_series_cfg_obj=macro_series_cfg,
    market_id_obj=selected_market_id,
    market_profile_obj=market_profile,
    cache_ttl=CACHE_TTL,
    curated_sector_prices_path=CURATED_SECTOR_PRICES_PATH,
)


def _load_analysis_sector_prices_from_cache(
    end_date_str: str,
    benchmark_code: str,
    market_id: str = "KR",
) -> pd.DataFrame:
    """Compatibility wrapper so app-level monkeypatches still affect cache loading tests."""
    original_reader = dashboard_data_module.read_market_prices
    dashboard_data_module.read_market_prices = read_market_prices
    try:
        return dashboard_load_analysis_sector_prices_from_cache(market_id, end_date_str, benchmark_code)
    finally:
        dashboard_data_module.read_market_prices = original_reader

# Session state bootstrap / theme
ensure_session_defaults(
    st.session_state,
    settings=settings,
    theme_key=THEME_SESSION_KEY,
    default_theme_mode=get_theme_mode(),
    all_action_option=ALL_ACTION_KEY,
)
theme_mode = get_theme_mode()
ui_locale = normalize_locale(settings.get("ui_locale", DEFAULT_UI_LOCALE))
analysis_heatmap_palette = normalize_session_state(
    st.session_state,
    theme_key=THEME_SESSION_KEY,
    theme_mode=theme_mode,
    heatmap_palette_options=HEATMAP_PALETTE_OPTIONS,
    all_action_option=ALL_ACTION_KEY,
    normalize_range_preset=normalize_range_preset,
)
inject_css(theme_mode)
manual_progress_host = st.empty()

# Sidebar / runtime status
macro_cache_token = get_macro_cache_token()
price_cache_token = get_price_cache_token()
krx_provider_configured = get_krx_provider_configured()
krx_provider_effective = get_krx_provider_effective()
krx_openapi_key_present = bool(load_api_key("KRX_OPENAPI_KEY")) if selected_market_id == "KR" else False

probe_price_status = probe_market_status()
probe_macro_status = probe_macro_status()
probe_flow_status = probe_investor_flow_status()
probe_data_status = {"price": probe_price_status, "macro": probe_macro_status}
btn_states = get_button_states(probe_data_status)
asof_default = parse_asof_default(st.session_state)
selected_flow_profile_state = str(st.session_state.get("flow_profile", "foreign_lead"))

with st.sidebar:
    asof_date, selected_flow_profile, refresh_market, refresh_macro, refresh_flow = render_sidebar_controls(
        market_id=selected_market_id,
        ui_labels=getattr(market_profile, "ui_labels", {}),
        theme_mode=theme_mode,
        analysis_heatmap_palette=analysis_heatmap_palette,
        probe_price_status=probe_price_status,
        probe_macro_status=probe_macro_status,
        probe_investor_flow_status=probe_flow_status,
        flow_profile=selected_flow_profile_state,
        momentum_method=str(settings.get("momentum_method", "legacy_rs_ma_v0")),
        btn_states=btn_states,
        asof_default=asof_default,
        ui_locale=ui_locale,
    )
    sidebar_progress_host = st.empty()

st.session_state["flow_profile"] = selected_flow_profile

rs_ma_period = int(st.session_state.get("rs_ma_period", settings.get("rs_ma_period", 20)))
ma_fast = int(st.session_state.get("ma_fast", settings.get("ma_fast", 20)))
ma_slow = int(st.session_state.get("ma_slow", settings.get("ma_slow", 60)))
momentum_method = str(settings.get("momentum_method", "legacy_rs_ma_v0"))
momentum_skip_recent_days = int(settings.get("momentum_skip_recent_days", 21))
momentum_lookback_6m_days = int(settings.get("momentum_lookback_6m_days", 126))
momentum_lookback_12m_days = int(settings.get("momentum_lookback_12m_days", 252))
momentum_rank_threshold_pct = float(settings.get("momentum_rank_threshold_pct", 0.60))
momentum_abs_filter = str(settings.get("momentum_abs_filter", "price_gt_200dma"))
price_years = int(st.session_state.get("price_years", settings.get("price_years", 3)))
benchmark_code = str(settings.get("benchmark_code", "1001"))
market_end_date = resolve_market_end_date(benchmark_code)
market_end_date_str = market_end_date.strftime("%Y%m%d")

context = DashboardContext(
    market_id=selected_market_id,
    market_profile=market_profile,
    settings=settings,
    sector_map=sector_map,
    macro_series_cfg=macro_series_cfg,
    benchmark_code=benchmark_code,
    market_end_date=market_end_date,
    market_end_date_str=market_end_date_str,
    macro_cache_token=macro_cache_token,
    price_cache_token=price_cache_token,
    price_artifact_key=get_price_artifact_key(),
    macro_artifact_key=get_macro_artifact_key(),
    provider_configured=krx_provider_configured,
    provider_effective=krx_provider_effective,
    openapi_key_present=krx_openapi_key_present,
    theme_mode=theme_mode,
    analysis_heatmap_palette=analysis_heatmap_palette,
    ui_locale=ui_locale,
    investor_flow_artifact_key=get_investor_flow_artifact_key(),
    investor_flow_status=probe_flow_status,
    investor_flow_profile=str(st.session_state.get("flow_profile", "foreign_lead")),
)

maybe_schedule_startup_krx_warm(benchmark_code, price_years, market_end_date)

# Button handlers
market_refresh_notice: tuple[str, str] | None = None
macro_refresh_notice: tuple[str, str] | None = None
investor_flow_refresh_notice: tuple[str, str] | None = None

if refresh_market:
    market_refresh_notice = run_market_refresh(
        context,
        price_years,
        progress_callback=_make_progress_callback(manual_progress_host, sidebar_progress_host),
    )

if refresh_macro:
    macro_refresh_notice = run_macro_refresh(
        context,
        macro_series_cfg,
        progress_callback=_make_progress_callback(manual_progress_host, sidebar_progress_host),
    )

if refresh_flow:
    investor_flow_refresh_notice = run_investor_flow_refresh(
        context,
        progress_callback=_make_progress_callback(manual_progress_host, sidebar_progress_host),
    )

# Core data load
page_progress_host = st.empty()
data_load_ok = False
runtime_payload: dict[str, object] = {}
try:
    prices_key = context.price_artifact_key
    macro_key = context.macro_artifact_key
    theme_lens_key = get_theme_lens_artifact_key()
    params = {
        "epsilon": float(st.session_state["epsilon"]),
        "rs_ma_period": rs_ma_period,
        "ma_fast": ma_fast,
        "ma_slow": ma_slow,
        "momentum_method": momentum_method,
        "momentum_skip_recent_days": momentum_skip_recent_days,
        "momentum_lookback_6m_days": momentum_lookback_6m_days,
        "momentum_lookback_12m_days": momentum_lookback_12m_days,
        "momentum_rank_threshold_pct": momentum_rank_threshold_pct,
        "momentum_abs_filter": momentum_abs_filter,
        "price_years": price_years,
    }
    params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    runtime_payload = load_dashboard_runtime_data(
        context.market_id,
        context.market_end_date_str,
        prices_key,
        macro_key,
        params_hash,
        context.macro_cache_token,
        context.price_cache_token,
        context.price_artifact_key,
        context.investor_flow_artifact_key,
        theme_lens_key,
        epsilon=float(st.session_state["epsilon"]),
        rs_ma_period=rs_ma_period,
        ma_fast=ma_fast,
        ma_slow=ma_slow,
        momentum_method=momentum_method,
        momentum_skip_recent_days=momentum_skip_recent_days,
        momentum_lookback_6m_days=momentum_lookback_6m_days,
        momentum_lookback_12m_days=momentum_lookback_12m_days,
        momentum_rank_threshold_pct=momentum_rank_threshold_pct,
        momentum_abs_filter=momentum_abs_filter,
        price_years=price_years,
        flow_profile=str(st.session_state.get("flow_profile", "foreign_lead")),
        progress_callback=_make_progress_callback(page_progress_host),
    )
    signals = list(runtime_payload["signals"])
    macro_result = runtime_payload["macro_result"]
    macro_df = runtime_payload["macro_df"]
    price_status = str(runtime_payload["price_status"])
    macro_status = str(runtime_payload["macro_status"])
    market_data_reference_date = str(runtime_payload.get("market_data_reference_date", "")).strip()
    market_blocking_error = str(runtime_payload["market_blocking_error"])
    investor_flow_status = str(runtime_payload["investor_flow_status"])
    investor_flow_fresh = bool(runtime_payload["investor_flow_fresh"])
    investor_flow_detail = dict(runtime_payload["investor_flow_detail"])
    investor_flow_frame = runtime_payload["investor_flow_frame"]
    theme_lens_status = str(runtime_payload.get("theme_lens_status", "UNAVAILABLE"))
    theme_lens_rows = list(runtime_payload.get("theme_lens_rows", []))
    data_status = {"price": price_status, "macro": macro_status}
    data_load_ok = True
except Exception as exc:
    logger.error("Data load failed: %s", exc)
    signals = []
    macro_result = pd.DataFrame(
        {
            "growth_dir": ["Flat"],
            "inflation_dir": ["Flat"],
            "regime": ["Indeterminate"],
            "confirmed_regime": ["Indeterminate"],
        },
        index=pd.DatetimeIndex([date.today()]),
    )
    macro_df = pd.DataFrame()
    price_status = "SAMPLE"
    macro_status = "SAMPLE"
    market_data_reference_date = ""
    market_blocking_error = ""
    investor_flow_status = "SAMPLE"
    investor_flow_fresh = False
    investor_flow_detail = {}
    investor_flow_frame = pd.DataFrame()
    theme_lens_status = "UNAVAILABLE"
    theme_lens_rows = []
    data_status = {"price": price_status, "macro": macro_status}
finally:
    if data_load_ok:
        page_progress_host.empty()

price_warm_status: dict[str, object] = {}
price_cache_case = None
macro_status_detail = {}
if price_status == "CACHED":
    if context.market_id == "US":
        from src.data_sources.yfinance_sectors import read_warm_status
    else:
        from src.data_sources.krx_indices import read_warm_status

    price_warm_status = read_warm_status()
    if context.market_id == "KR":
        price_cache_case = resolve_price_cache_banner_case(
            price_status=price_status,
            provider_mode=context.provider_effective,
            openapi_key_present=context.openapi_key_present,
            market_end_date_str=context.market_end_date_str,
            warm_status=price_warm_status,
        )

try:
    preflight_status = cached_api_preflight(timeout_sec=3, market_id_arg=context.market_id)
except Exception as exc:
    preflight_status = {
        "PRECHECK": {
            "status": "HTTP_ERROR",
            "detail": str(exc),
            "url": "",
            "checked_at": "",
        }
    }

try:
    from src.data_sources.warehouse import read_dataset_status

    macro_status_detail = read_dataset_status("macro_data", market=context.market_id)
except Exception as exc:
    logger.warning("Macro dataset status lookup failed: %s", exc)
    macro_status_detail = {}

openapi_missing_key_warning_shown = (
    context.market_id == "KR" and context.provider_configured == "OPENAPI" and not context.openapi_key_present
)
show_notice_toast(market_refresh_notice)
show_notice_toast(macro_refresh_notice)
show_notice_toast(investor_flow_refresh_notice)

dashboard_status_banner = resolve_dashboard_status_banner(
    data_status=data_status,
    market_blocking_error=market_blocking_error,
    price_cache_case=price_cache_case,
    openapi_key_warning=openapi_missing_key_warning_shown,
    preflight_status=preflight_status,
    price_warm_status=price_warm_status,
    macro_status_detail=macro_status_detail,
)

current_regime = "Indeterminate"
regime_is_confirmed = False
yield_curve_status: str | None = None
if not macro_result.empty:
    if "confirmed_regime" in macro_result.columns:
        current_regime = str(macro_result["confirmed_regime"].iloc[-1])
        raw_regime = str(macro_result["regime"].iloc[-1]) if "regime" in macro_result.columns else current_regime
        regime_is_confirmed = current_regime == raw_regime and current_regime != "Indeterminate"
    elif "regime" in macro_result.columns:
        current_regime = str(macro_result["regime"].iloc[-1])
    if "yield_curve" in macro_result.columns:
        yield_curve_status = str(macro_result["yield_curve"].iloc[-1])

is_provisional = any(getattr(signal, "is_provisional", False) for signal in signals)

growth_val: float | None = None
inflation_val: float | None = None
export_growth_val: float | None = None
trade_indicators: dict[str, float] = {}
sector_export_trends: dict[str, float] = {}
sector_export_history: dict[str, pd.Series] = {}
sector_export_aliases = build_enabled_sector_export_aliases(sector_map, macro_series_cfg)
has_sector_export_indicators = bool(sector_export_aliases)
has_trade_indicators = any(
    is_macro_alias_enabled(macro_series_cfg, alias)
    for alias in ("trade_exports_yoy", "trade_imports_yoy")
)
sector_trade_lens: list[dict[str, object]] = []
fx_change: float | None = None
if not macro_df.empty:
    growth_series = extract_macro_series(macro_df, macro_series_cfg, "leading_index")
    if not growth_series.empty:
        growth_val = float(growth_series.iloc[-1])
    inflation_series = extract_macro_series(macro_df, macro_series_cfg, "cpi_yoy")
    if not inflation_series.empty:
        inflation_val = float(inflation_series.iloc[-1])
    if context.market_id == "KR":
        export_growth_val = extract_kr_export_growth_yoy(macro_df, macro_series_cfg)
    trade_indicators = extract_trade_indicators(macro_df, macro_series_cfg)
    for sector_name, export_alias in sector_export_aliases.items():
        sector_export_series = extract_macro_series(macro_df, macro_series_cfg, export_alias)
        sector_export_yoy = sector_export_series.pct_change(12).dropna() * 100 if len(sector_export_series) >= 13 else pd.Series(dtype=float)
        if not sector_export_yoy.empty:
            sector_export_trends[sector_name] = float(sector_export_yoy.iloc[-1])
            sector_export_history[sector_name] = sector_export_yoy
    fx_series = extract_macro_series(macro_df, macro_series_cfg, str(settings.get("fx_series_alias", "usdkrw")))
    if len(fx_series) >= 2:
        fx_change = float((fx_series.iloc[-1] / fx_series.iloc[-2] - 1) * 100)
if context.market_id == "US" and (trade_indicators or has_trade_indicators):
    sector_trade_lens = build_sector_trade_proxy_lens(sector_map, trade_indicators)

dashboard_query_date_label = context.market_end_date.strftime("%Y-%m-%d")
dashboard_data_date_label = market_data_reference_date or dashboard_query_date_label
macro_status_display = resolve_macro_status_display(
    macro_status=macro_status,
    macro_status_detail=macro_status_detail,
)

render_page_header(
    title=resolve_dashboard_page_title(selected_dashboard_page, selected_market_id),
    description={
        "overview": "국면, 수급, taxonomy, 상대강도 원장을 한 화면에서 확인하는 통합 작업 화면입니다.",
        "research": "기간, 국면, 섹터를 바꿔가며 신호의 근거를 검증합니다.",
        "quality": "데이터 수집 상태, 캐시, provider, 오류 이력을 점검합니다.",
    }.get(selected_dashboard_page, "섹터 로테이션 대시보드"),
    pills=[
        {"label": "국면", "value": current_regime, "tone": "success" if regime_is_confirmed else "warning"},
        {"label": "시장", "value": price_status, "tone": "danger" if price_status == "SAMPLE" else "warning" if price_status == "CACHED" else "success"},
        {"label": "매크로", **macro_status_display},
        {"label": "조회 기준일", "value": dashboard_data_date_label, "tone": "info"},
        {"label": "목표일", "value": dashboard_query_date_label, "tone": "warning" if dashboard_data_date_label != dashboard_query_date_label else "info"},
        {"label": "제공자", "value": context.provider_effective, "tone": "info"},
    ] + (
        [
            {
                "label": "수급",
                "value": investor_flow_status if investor_flow_fresh else f"{investor_flow_status}*",
                "tone": "success" if investor_flow_fresh else "warning" if investor_flow_status != "SAMPLE" else "info",
            }
        ]
        if context.market_id == "KR"
        else []
    ),
)
render_dashboard_status_banner(dashboard_status_banner)

try:
    sector_prices_canvas = pd.DataFrame() if price_status == "BLOCKED" else cached_analysis_sector_prices(
        context.market_id,
        context.market_end_date_str,
        benchmark_code,
        price_years,
        context.price_artifact_key,
    )
except Exception as exc:
    logger.warning("Analysis canvas price load fallback: %s", exc)
    sector_prices_canvas = pd.DataFrame()

sector_name_map = build_sector_name_map(
    signals=list(signals),
    sector_prices=sector_prices_canvas,
    benchmark_code=benchmark_code,
    benchmark_label=str(settings.get("benchmark_label", getattr(market_profile, "benchmark_label", benchmark_code))),
)
prices_wide = build_prices_wide(
    sector_prices=sector_prices_canvas,
    sector_name_map=sector_name_map,
)
benchmark_label = sector_name_map.get(
    benchmark_code,
    str(settings.get("benchmark_label", getattr(market_profile, "benchmark_label", benchmark_code))),
)
reference_index_label_overrides = {
    str(code): str(label)
    for code, label in dict(settings.get("reference_index_labels", {}) or {}).items()
}
reference_index_labels = [
    sector_name_map.get(str(code), reference_index_label_overrides.get(str(code), str(code)))
    for code in settings.get("reference_index_codes", []) or []
    if str(code).strip()
]
reference_index_label_set = {label for label in reference_index_labels if str(label).strip()}
sector_columns = [
    column
    for column in prices_wide.columns
    if column != benchmark_label and column not in reference_index_label_set
]
monthly_close_full, monthly_returns_full, benchmark_monthly_return, monthly_excess_returns_full = build_monthly_return_views(
    prices_wide=prices_wide,
    sector_columns=sector_columns,
    benchmark_label=benchmark_label,
)
cycle_segments_all, phase_by_month = build_cycle_segments(
    macro_result=macro_result,
    monthly_close=monthly_close_full,
)

if not prices_wide.empty:
    analysis_min_date = prices_wide.index.min().date()
    analysis_max_date = prices_wide.index.max().date()
else:
    analysis_min_date = context.market_end_date - timedelta(days=365)
    analysis_max_date = context.market_end_date

ensure_analysis_bounds(
    st.session_state,
    analysis_min_date=analysis_min_date,
    analysis_max_date=analysis_max_date,
    normalize_range_preset=normalize_range_preset,
    resolve_range_from_preset=resolve_range_from_preset,
)

analysis_start_date = pd.Timestamp(st.session_state["analysis_start_date"]).date()
analysis_end_date = pd.Timestamp(st.session_state["analysis_end_date"]).date()
current_range_preset = infer_range_preset(
    start_date=analysis_start_date,
    end_date=analysis_end_date,
    min_date=analysis_min_date,
    max_date=analysis_max_date,
)
st.session_state["selected_range_preset"] = current_range_preset

toolbar_selected_sector = str(st.session_state.get("selected_sector", "")).strip() or "Auto"
toolbar_selected_phase = str(st.session_state.get("selected_cycle_phase", "ALL")).strip() or "ALL"
toolbar_selected_preset = normalize_range_preset(st.session_state.get("selected_range_preset", "1Y"))
if selected_dashboard_page == "research":
    render_research_page_frame(
        page_key="research",
        eyebrow="Research Canvas",
        title="상대강도 분석 워크스페이스",
        description="기간, 국면, 섹터 선택을 하나의 분석 상태로 고정하고 히트맵과 상세 추적 패널을 같은 기준으로 검증합니다.",
        summary_items=[
            {"label": "분석 기간", "value": f"{analysis_start_date} - {analysis_end_date}"},
            {"label": "프리셋", "value": toolbar_selected_preset},
            {"label": "국면", "value": toolbar_selected_phase},
            {"label": "선택 섹터", "value": toolbar_selected_sector},
        ],
    )
    resolved_start, resolved_end, resolved_preset, toolbar_submitted = render_analysis_toolbar(
        min_date=analysis_min_date,
        max_date=analysis_max_date,
        start_date=analysis_start_date,
        end_date=analysis_end_date,
        selected_range_preset=toolbar_selected_preset if current_range_preset == "CUSTOM" else current_range_preset,
        selected_cycle_phase=toolbar_selected_phase,
        selected_sector=toolbar_selected_sector,
        locale=context.ui_locale,
    )
    if toolbar_submitted and apply_analysis_toolbar_selection(
        st.session_state,
        resolved_start=resolved_start,
        resolved_end=resolved_end,
        resolved_preset=resolved_preset,
    ):
        st.rerun()

    lookup_query, lookup_submitted = render_stock_lookup_control(
        market_id=context.market_id,
        query_value=str(st.session_state.get("stock_lookup_query", "")),
        status=str(st.session_state.get("stock_lookup_status", "")),
        message=str(st.session_state.get("stock_lookup_message", "")),
        display_model=build_stock_lookup_display_model(st.session_state.get("stock_lookup_result"), sector_map),
        locale=context.ui_locale,
    )
    st.session_state["stock_lookup_query"] = lookup_query
    if lookup_submitted:
        lookup_result = _resolve_stock_lookup_result(
            market_id=context.market_id,
            query=lookup_query,
            sector_map=sector_map,
            asof_date=st.session_state.get("asof_date_str"),
        )
        apply_stock_lookup_result(
            st.session_state,
            result=lookup_result,
        )
        st.rerun()

analysis_prices = prices_wide.loc[
    (prices_wide.index >= pd.Timestamp(st.session_state["analysis_start_date"]))
    & (prices_wide.index <= pd.Timestamp(st.session_state["analysis_end_date"]))
]
phase_by_month_visible = phase_by_month.loc[
    (phase_by_month.index >= pd.Timestamp(st.session_state["analysis_start_date"]).to_period("M").to_timestamp("M"))
    & (phase_by_month.index <= pd.Timestamp(st.session_state["analysis_end_date"]).to_period("M").to_timestamp("M"))
] if not phase_by_month.empty else pd.Series(dtype=object)

selected_cycle_phase = str(st.session_state.get("selected_cycle_phase", "ALL") or "ALL")
analysis_prices_phase = filter_prices_for_phase(
    prices_wide=analysis_prices,
    phase_by_month=phase_by_month_visible,
    selected_cycle_phase=selected_cycle_phase,
)

heatmap_return_source = filter_monthly_frame_for_analysis(
    monthly_frame=monthly_returns_full,
    start_date=analysis_start_date,
    end_date=analysis_end_date,
    selected_cycle_phase=selected_cycle_phase,
    phase_by_month=phase_by_month_visible,
)
heatmap_strength_source = filter_monthly_frame_for_analysis(
    monthly_frame=monthly_excess_returns_full,
    start_date=analysis_start_date,
    end_date=analysis_end_date,
    selected_cycle_phase=selected_cycle_phase,
    phase_by_month=phase_by_month_visible,
)
heatmap_return_display = build_heatmap_display(heatmap_return_source)
heatmap_strength_display = build_heatmap_display(heatmap_strength_source)
visible_months = list(heatmap_return_display.columns) if not heatmap_return_display.empty else []
ensure_visible_month_selection(st.session_state, visible_months=visible_months)

visible_segments = [
    segment
    for segment in cycle_segments_all
    if pd.Timestamp(segment["end"]) >= pd.Timestamp(st.session_state["analysis_start_date"]).to_period("M").to_timestamp("M")
    and pd.Timestamp(segment["start"]) <= pd.Timestamp(st.session_state["analysis_end_date"]).to_period("M").to_timestamp("M")
]

analysis_window = AnalysisWindow(
    start_date=analysis_start_date,
    end_date=analysis_end_date,
    selected_range_preset=current_range_preset,
    selected_cycle_phase=selected_cycle_phase,
    selected_sector=str(st.session_state.get("selected_sector", "")),
    selected_month=str(st.session_state.get("selected_month", "")),
)

mobile_client = is_mobile_client()
signal_lookup = {str(signal.sector_name): signal for signal in signals}
held_sector_options = sorted({str(signal.sector_name) for signal in signals if str(signal.sector_name).strip()})
shared_flow_summary_map = _resolve_shared_flow_summary_map(runtime_payload)
flow_short_window = int(settings.get("investor_flow_short_window", 20))
flow_long_window = int(settings.get("investor_flow_long_window", 60))
taxonomy_context = build_taxonomy_dashboard_model(
    sector_map=sector_map,
    market=context.market_id,
)
analysis_canvas_kwargs = {
    "heatmap_return_display": heatmap_return_display,
    "heatmap_strength_display": heatmap_strength_display,
    "selected_cycle_phase": analysis_window.selected_cycle_phase,
    "theme_mode": context.theme_mode,
    "analysis_heatmap_palette": context.analysis_heatmap_palette,
    "visible_segments": visible_segments,
    "current_regime": current_regime,
    "analysis_prices_phase": analysis_prices_phase,
    "analysis_prices": analysis_prices,
    "sector_columns": sector_columns,
    "benchmark_label": benchmark_label,
    "analysis_max_date": analysis_max_date,
    "analysis_min_date": analysis_min_date,
    "build_sector_detail_figure": build_sector_detail_figure,
    "resolve_range_from_preset": resolve_range_from_preset,
    "signal_lookup": signal_lookup,
    "ui_locale": context.ui_locale,
}
if selected_dashboard_page == "overview":
    held_sectors = [
        str(item)
        for item in st.session_state.get("held_sectors", [])
        if str(item).strip()
    ]
    lookup_query, lookup_submitted = render_toss_overview_dashboard(
        market_id=context.market_id,
        current_regime=current_regime,
        price_status=price_status,
        macro_status=macro_status,
        prices_wide=prices_wide,
        benchmark_label=benchmark_label,
        reference_index_labels=reference_index_labels,
        signals=list(signals),
        theme_mode=context.theme_mode,
        sector_map=sector_map,
        lookup_query_value=str(st.session_state.get("stock_lookup_query", "")),
        lookup_status=str(st.session_state.get("stock_lookup_status", "")),
        lookup_message=str(st.session_state.get("stock_lookup_message", "")),
        lookup_display_model=build_stock_lookup_display_model(
            st.session_state.get("stock_lookup_result"),
            sector_map,
            taxonomy_context,
        ),
        export_growth_val=export_growth_val,
        trade_indicators=trade_indicators,
        sector_export_trends=sector_export_trends,
        sector_export_history=sector_export_history,
        sector_trade_lens=sector_trade_lens,
        has_trade_indicators=has_trade_indicators,
        has_sector_export_indicators=has_sector_export_indicators,
        taxonomy_context=taxonomy_context,
        locale=context.ui_locale,
        is_mobile=mobile_client,
    )
    st.session_state["stock_lookup_query"] = lookup_query
    if lookup_submitted:
        lookup_result = _resolve_stock_lookup_result(
            market_id=context.market_id,
            query=lookup_query,
            sector_map=sector_map,
            asof_date=st.session_state.get("asof_date_str"),
        )
        apply_stock_lookup_result(
            st.session_state,
            result=lookup_result,
        )
        st.rerun()
else:
    held_sectors = [
        str(item)
        for item in st.session_state.get("held_sectors", [])
        if str(item).strip()
    ]

bundle = DashboardDataBundle(
    signals=list(signals),
    macro_result=macro_result,
    macro_df=macro_df,
    price_status=price_status,
    macro_status=macro_status,
    market_blocking_error=market_blocking_error,
    dashboard_status_banner=dashboard_status_banner,
    current_regime=current_regime,
    regime_is_confirmed=regime_is_confirmed,
    yield_curve_status=yield_curve_status,
    growth_val=growth_val,
    inflation_val=inflation_val,
    export_growth_val=export_growth_val,
    trade_indicators=dict(trade_indicators),
    sector_trade_lens=list(sector_trade_lens),
    sector_export_trends=dict(sector_export_trends),
    sector_export_history=dict(sector_export_history),
    has_sector_export_indicators=has_sector_export_indicators,
    fx_change=fx_change,
    price_warm_status=dict(price_warm_status),
    price_cache_case=price_cache_case,
    preflight_status=preflight_status,
    sector_prices_canvas=sector_prices_canvas,
    sector_name_map=sector_name_map,
    prices_wide=prices_wide,
    benchmark_label=benchmark_label,
    sector_columns=sector_columns,
    monthly_close_full=monthly_close_full,
    monthly_returns_full=monthly_returns_full,
    benchmark_monthly_return=benchmark_monthly_return,
    monthly_excess_returns_full=monthly_excess_returns_full,
    cycle_segments_all=cycle_segments_all,
    phase_by_month=phase_by_month,
    analysis_min_date=analysis_min_date,
    analysis_max_date=analysis_max_date,
    market_refresh_notice=market_refresh_notice,
    macro_refresh_notice=macro_refresh_notice,
    investor_flow_status=investor_flow_status,
    investor_flow_fresh=investor_flow_fresh,
    investor_flow_profile=str(st.session_state.get("flow_profile", "foreign_lead")),
    investor_flow_frame=investor_flow_frame,
    investor_flow_detail=dict(investor_flow_detail),
    shared_flow_summary_map=shared_flow_summary_map,
    investor_flow_refresh_notice=investor_flow_refresh_notice,
    theme_lens_status=theme_lens_status,
    theme_lens_rows=theme_lens_rows,
)

if selected_dashboard_page == "research":
    render_analysis_canvas(**analysis_canvas_kwargs)
elif selected_dashboard_page != "overview":
    render_dashboard_tabs(
        current_regime=bundle.current_regime,
        theme_mode=context.theme_mode,
        signals=bundle.signals,
        held_sectors=held_sectors,
        settings=settings,
        is_mobile_client=mobile_client,
        market_id=context.market_id,
        investor_flow_status=bundle.investor_flow_status,
        investor_flow_fresh=bundle.investor_flow_fresh,
        investor_flow_profile=bundle.investor_flow_profile,
        investor_flow_frame=bundle.investor_flow_frame,
        investor_flow_detail=bundle.investor_flow_detail,
        shared_flow_summary_map=bundle.shared_flow_summary_map,
        theme_lens_status=bundle.theme_lens_status,
        theme_lens_rows=bundle.theme_lens_rows,
        taxonomy_context=taxonomy_context,
        sector_map=sector_map,
        ui_locale=context.ui_locale,
        selected_page_id=selected_dashboard_page,
    )

st.divider()
st.caption(
    f"조회 기준일 {dashboard_data_date_label} | 목표일 {dashboard_query_date_label} | 선택 기준일 {asof_date} | "
    f"{', '.join(getattr(market_profile, 'source_badges', (context.provider_effective,)))} | "
    f"Regime: {bundle.current_regime}"
)
