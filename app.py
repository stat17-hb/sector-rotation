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

import pandas as pd
import streamlit as st

from config.markets import load_market_configs
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
    cached_investor_flow,
    cached_macro,
    cached_signals,
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
)
from src.dashboard.runtime import (
    invalidate_dashboard_caches,
    run_feature_recompute,
    run_investor_flow_refresh,
    run_macro_refresh,
    run_market_refresh,
)
from src.dashboard.state import (
    apply_market_selection,
    apply_analysis_toolbar_selection,
    ensure_analysis_bounds,
    ensure_session_defaults,
    ensure_visible_month_selection,
    normalize_session_state,
    parse_asof_default,
)
from src.dashboard.tabs import (
    render_decision_first_sections,
    render_dashboard_tabs,
    render_sidebar_controls,
)
from src.dashboard.types import AnalysisWindow, DashboardContext, DashboardDataBundle
from src.macro.series_utils import extract_macro_series
from src.data_sources.warehouse import (
    read_market_prices,
)
from src.ui.copy import ALL_ACTION_KEY, DEFAULT_UI_LOCALE, normalize_locale
from src.ui.components import (
    HEATMAP_PALETTE_OPTIONS,
    filter_signals_for_display,
    infer_range_preset,
    normalize_range_preset,
    render_analysis_toolbar,
    render_page_header,
    resolve_range_from_preset,
    signal_display_sort_key,
    build_sector_detail_figure,
)
from src.ui.data_status import (
    get_button_states,
    resolve_dashboard_status_banner,
    resolve_price_cache_banner_case,
)
from src.ui.styles import inject_css

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

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
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600)
def _load_config(market_id: str) -> tuple[dict, dict, dict, object]:
    """Load YAML config files. Cached for 1 hour."""
    return load_market_configs(market_id)


selected_market_id = str(st.session_state.get("market_id", "KR")).strip().upper() or "KR"
settings, sector_map, macro_series_cfg, market_profile = _load_config(selected_market_id)
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
    selected_market_sidebar, asof_date, selected_flow_profile, refresh_market, refresh_macro, refresh_flow, recompute = render_sidebar_controls(
        market_id=selected_market_id,
        ui_labels=getattr(market_profile, "ui_labels", {}),
        theme_mode=theme_mode,
        analysis_heatmap_palette=analysis_heatmap_palette,
        probe_price_status=probe_price_status,
        probe_macro_status=probe_macro_status,
        probe_investor_flow_status=probe_flow_status,
        flow_profile=selected_flow_profile_state,
        btn_states=btn_states,
        asof_default=asof_default,
        ui_locale=ui_locale,
    )

st.session_state["flow_profile"] = selected_flow_profile

if apply_market_selection(st.session_state, market_id=selected_market_sidebar):
    invalidate_dashboard_caches("all")
    st.rerun()

rs_ma_period = int(st.session_state.get("rs_ma_period", settings.get("rs_ma_period", 20)))
ma_fast = int(st.session_state.get("ma_fast", settings.get("ma_fast", 20)))
ma_slow = int(st.session_state.get("ma_slow", settings.get("ma_slow", 60)))
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
    with st.spinner("Refreshing market data..."):
        market_refresh_notice = run_market_refresh(context, price_years)

if refresh_macro:
    with st.spinner("Refreshing macro data..."):
        macro_refresh_notice = run_macro_refresh(context, macro_series_cfg)

if refresh_flow:
    with st.spinner("Refreshing investor-flow data..."):
        investor_flow_refresh_notice = run_investor_flow_refresh(context)

if recompute:
    run_feature_recompute()
    st.rerun()

# Core data load
with st.spinner("Loading dashboard data..."):
    try:
        prices_key = context.price_artifact_key
        macro_key = context.macro_artifact_key
        params = {
            "epsilon": float(st.session_state["epsilon"]),
            "rs_ma_period": rs_ma_period,
            "ma_fast": ma_fast,
            "ma_slow": ma_slow,
            "price_years": price_years,
        }
        params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
        signals, macro_result, price_status, macro_status, market_blocking_error = cached_signals(
            context.market_id,
            context.market_end_date_str,
            prices_key,
            macro_key,
            params_hash,
            context.macro_cache_token,
            context.price_cache_token,
            context.price_artifact_key,
            context.investor_flow_artifact_key,
            float(st.session_state["epsilon"]),
            rs_ma_period,
            ma_fast,
            ma_slow,
            price_years,
            str(st.session_state.get("flow_profile", "foreign_lead")),
        )
        data_status = {"price": price_status, "macro": macro_status}
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
        price_status = "SAMPLE"
        macro_status = "SAMPLE"
        market_blocking_error = ""
        data_status = {"price": price_status, "macro": macro_status}

price_warm_status: dict[str, object] = {}
price_cache_case = None
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

_, macro_df = cached_macro(context.market_id, context.macro_cache_token, context.market_end_date_str)
growth_val: float | None = None
inflation_val: float | None = None
fx_change: float | None = None
if not macro_df.empty:
    growth_series = extract_macro_series(macro_df, macro_series_cfg, "leading_index")
    if not growth_series.empty:
        growth_val = float(growth_series.iloc[-1])
    inflation_series = extract_macro_series(macro_df, macro_series_cfg, "cpi_yoy")
    if not inflation_series.empty:
        inflation_val = float(inflation_series.iloc[-1])
    fx_series = extract_macro_series(macro_df, macro_series_cfg, str(settings.get("fx_series_alias", "usdkrw")))
    if len(fx_series) >= 2:
        fx_change = float((fx_series.iloc[-1] / fx_series.iloc[-2] - 1) * 100)

if context.market_id == "KR":
    investor_flow_status, investor_flow_fresh, investor_flow_detail, investor_flow_frame = cached_investor_flow(
        context.market_id,
        context.market_end_date_str,
        context.investor_flow_artifact_key,
    )
else:
    investor_flow_status, investor_flow_fresh, investor_flow_detail, investor_flow_frame = cached_investor_flow(
        context.market_id,
        context.market_end_date_str,
        context.investor_flow_artifact_key,
    )

render_page_header(
    title=str(getattr(market_profile, "page_header", "Sector Rotation")),
    description="Move from range selection to cycle context, sector comparison, and linked detail tracking without leaving the main canvas.",
    pills=[
        {"label": "Regime", "value": current_regime, "tone": "success" if regime_is_confirmed else "warning"},
        {"label": "Market", "value": price_status, "tone": "danger" if price_status == "SAMPLE" else "warning" if price_status == "CACHED" else "success"},
        {"label": "Macro", "value": macro_status, "tone": "danger" if macro_status == "SAMPLE" else "warning" if macro_status == "CACHED" else "success"},
        {"label": "Provider", "value": context.provider_effective, "tone": "info"},
    ] + (
        [
            {
                "label": "Flow",
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
sector_columns = [column for column in prices_wide.columns if column != benchmark_label]
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
held_sectors, filter_action_global, filter_regime_only_global, position_mode, show_alerted_only = render_decision_first_sections(
    current_regime=current_regime,
    regime_is_confirmed=regime_is_confirmed,
    growth_val=growth_val,
    inflation_val=inflation_val,
    fx_change=fx_change,
    fx_label=str(getattr(market_profile, "ui_labels", {}).get("fx_metric_label", "FX move")),
    is_provisional=is_provisional,
    theme_mode=context.theme_mode,
    price_status=price_status,
    macro_status=macro_status,
    investor_flow_status=investor_flow_status,
    investor_flow_fresh=investor_flow_fresh,
    investor_flow_profile=str(st.session_state.get("flow_profile", "foreign_lead")),
    investor_flow_frame=investor_flow_frame,
    investor_flow_detail=dict(investor_flow_detail),
    yield_curve_status=yield_curve_status,
    signals=list(signals),
    held_sector_options=held_sector_options,
    action_options=[ALL_ACTION_KEY, "Strong Buy", "Watch", "Hold", "Avoid", "N/A"],
    is_mobile_client=mobile_client,
    analysis_canvas_kwargs={
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
    },
    market_id=context.market_id,
    ui_locale=context.ui_locale,
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
top_pick_signals = sorted(signals_filtered, key=lambda signal: signal_display_sort_key(signal, held_sectors))

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
    investor_flow_refresh_notice=investor_flow_refresh_notice,
)

render_dashboard_tabs(
    current_regime=bundle.current_regime,
    regime_is_confirmed=bundle.regime_is_confirmed,
    growth_val=bundle.growth_val,
    inflation_val=bundle.inflation_val,
    fx_change=bundle.fx_change,
    fx_label=str(getattr(market_profile, "ui_labels", {}).get("fx_metric_label", "FX move")),
    is_provisional=is_provisional,
    theme_mode=context.theme_mode,
    price_status=bundle.price_status,
    macro_status=bundle.macro_status,
    yield_curve_status=bundle.yield_curve_status,
    top_pick_signals=top_pick_signals,
    signals_filtered=signals_filtered,
    signals=bundle.signals,
    filter_action_global=filter_action_global,
    filter_regime_only_global=filter_regime_only_global,
    held_sectors=held_sectors,
    position_mode=position_mode,
    show_alerted_only=show_alerted_only,
    settings=settings,
    is_mobile_client=mobile_client,
    market_id=context.market_id,
    investor_flow_status=bundle.investor_flow_status,
    investor_flow_fresh=bundle.investor_flow_fresh,
    investor_flow_profile=bundle.investor_flow_profile,
    investor_flow_frame=bundle.investor_flow_frame,
    investor_flow_detail=bundle.investor_flow_detail,
    sector_map=sector_map,
    ui_locale=context.ui_locale,
)

st.divider()
st.caption(
    f"As of {asof_date} | "
    f"{', '.join(getattr(market_profile, 'source_badges', (context.provider_effective,)))} | "
    f"Regime: {bundle.current_regime}"
)
