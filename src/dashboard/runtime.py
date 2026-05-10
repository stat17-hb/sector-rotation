"""Runtime orchestration helpers for dashboard refresh flows."""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Callable, Literal

import pandas as pd

from src.dashboard.data import (
    build_investor_flow_refresh_notice,
    build_macro_refresh_notice,
    build_market_refresh_notice,
    cached_analysis_sector_prices,
    cached_api_preflight,
    cached_investor_flow,
    cached_macro,
    cached_sector_prices,
    cached_signals,
    clear_investor_flow_transient_override,
    clear_market_price_transient_override,
    get_all_sector_codes,
    get_market_index_universe_codes,
    get_market_range_strings,
    set_investor_flow_transient_override,
    set_market_price_transient_override,
)
from src.dashboard.types import DashboardContext
from src.data_sources.warehouse import close_cached_read_only_connection
from src.data_sources.warehouse import get_market_latest_dates


logger = logging.getLogger(__name__)

CacheScope = Literal["all", "market", "macro", "flow", "signals"]
ProgressCallback = Callable[[dict[str, Any]], None]


def _emit_progress(
    progress_callback: ProgressCallback | None,
    *,
    task: str,
    phase: str,
    pct: float | int,
    detail: str = "",
    status: str = "running",
) -> None:
    if progress_callback is None:
        return
    try:
        pct_value = int(round(float(pct)))
    except Exception:
        pct_value = 0
    progress_callback(
        {
            "task": str(task).strip(),
            "phase": str(phase).strip(),
            "pct": max(0, min(100, pct_value)),
            "detail": str(detail).strip(),
            "status": str(status).strip().lower() or "running",
        }
    )


def _clear_monitoring_data_cache() -> None:
    try:
        from src.dashboard.tabs import clear_monitoring_data_cache
    except Exception:
        logger.debug("Monitoring data cache clearer is unavailable", exc_info=True)
        return
    clear_monitoring_data_cache()


def invalidate_dashboard_caches(scope: CacheScope) -> None:
    """Clear the dashboard cache family associated with the requested scope."""
    if scope == "all":
        cached_api_preflight.clear()
        cached_sector_prices.clear()
        cached_analysis_sector_prices.clear()
        cached_investor_flow.clear()
        cached_macro.clear()
        cached_signals.clear()
        _clear_monitoring_data_cache()
        return
    if scope == "market":
        cached_api_preflight.clear()
        cached_sector_prices.clear()
        cached_analysis_sector_prices.clear()
        cached_investor_flow.clear()
        cached_signals.clear()
        _clear_monitoring_data_cache()
        return
    if scope == "macro":
        cached_macro.clear()
        cached_signals.clear()
        _clear_monitoring_data_cache()
        return
    if scope == "flow":
        cached_investor_flow.clear()
        cached_signals.clear()
        _clear_monitoring_data_cache()
        return
    if scope == "signals":
        cached_signals.clear()
        return
    raise ValueError(f"Unsupported cache scope: {scope}")


def _resolve_market_refresh_runner(market_id: str) -> Callable[[list[str], str, str], tuple[tuple[str, object], dict[str, object]]]:
    if str(market_id).strip().upper() == "US":
        from src.data_sources.yfinance_sectors import run_manual_price_refresh
    else:
        from src.data_sources.krx_indices import run_manual_price_refresh
    return run_manual_price_refresh


def _resolve_incremental_market_range(
    *,
    index_codes: list[str],
    fallback_start: str,
    end: str,
    market: str,
) -> tuple[str, str, dict[str, Any]]:
    """Return the market refresh window anchored by warehouse max dates when complete."""
    normalized_codes = [str(code).strip() for code in index_codes if str(code).strip()]
    fallback_start_digits = "".join(ch for ch in str(fallback_start or "") if ch.isdigit())[:8]
    end_digits = "".join(ch for ch in str(end or "") if ch.isdigit())[:8]
    if not normalized_codes or len(fallback_start_digits) != 8 or len(end_digits) != 8:
        return fallback_start, end, {"mode": "configured_window", "reason": "invalid_window"}

    latest_dates = get_market_latest_dates(normalized_codes, market=market)
    missing_codes = [code for code in normalized_codes if not str(latest_dates.get(code, "")).strip()]
    if missing_codes:
        return fallback_start_digits, end_digits, {
            "mode": "configured_window",
            "reason": "missing_code_history",
            "missing_codes": missing_codes,
        }

    parsed_latest = [
        pd.Timestamp(str(latest_dates[code])).normalize()
        for code in normalized_codes
        if str(latest_dates.get(code, "")).strip()
    ]
    if not parsed_latest:
        return fallback_start_digits, end_digits, {"mode": "configured_window", "reason": "empty_latest_dates"}

    start_ts = min(parsed_latest) + timedelta(days=1)
    fallback_ts = pd.Timestamp(fallback_start_digits).normalize()
    start_ts = max(start_ts, fallback_ts)
    return start_ts.strftime("%Y%m%d"), end_digits, {
        "mode": "incremental",
        "reason": "day_after_oldest_code_latest",
        "latest_dates": latest_dates,
    }


def run_market_refresh(
    context: DashboardContext,
    price_years: int,
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, str] | None:
    """Execute a manual market refresh and convert the result into a UI notice."""
    configured_start_str, refresh_end_str = get_market_range_strings(context.market_end_date_str, price_years)
    runner = _resolve_market_refresh_runner(context.market_id)
    try:
        normalized_market = str(context.market_id or "").strip().upper()
        sector_codes = (
            get_market_index_universe_codes(context.benchmark_code, context.market_id)
            if normalized_market == "KR"
            else get_all_sector_codes(context.benchmark_code)
        )
        refresh_start_str, refresh_end_str, window_meta = _resolve_incremental_market_range(
            index_codes=sector_codes,
            fallback_start=configured_start_str,
            end=refresh_end_str,
            market=context.market_id,
        )
        _emit_progress(
            progress_callback,
            task="시장데이터 갱신",
            phase="준비 중",
            pct=5,
            detail=f"{refresh_start_str} ~ {refresh_end_str}",
        )
        if pd.Timestamp(refresh_start_str) > pd.Timestamp(refresh_end_str):
            summary = {
                "status": "CACHED",
                "coverage_complete": True,
                "delta_codes": [],
                "failed_days": [],
                "failed_codes": {},
                "provider": "YFINANCE" if normalized_market == "US" else "OPENAPI",
                "reason": "manual_refresh",
                "start": refresh_start_str,
                "end": refresh_end_str,
                "window": window_meta,
            }
            clear_market_price_transient_override()
            invalidate_dashboard_caches("market")
            notice = build_market_refresh_notice(summary)
            _emit_progress(
                progress_callback,
                task="시장데이터 갱신",
                phase="이미 최신 구간",
                pct=100,
                detail=notice[1] if notice else "",
                status="complete",
            )
            return notice
        close_cached_read_only_connection()
        cached_api_preflight.clear()
        _emit_progress(
            progress_callback,
            task="시장데이터 갱신",
            phase="데이터 요청 중",
            pct=35,
            detail=f"{len(sector_codes)} indices",
        )
        (status, frame), refresh_summary = runner(
            sector_codes,
            refresh_start_str,
            refresh_end_str,
        )
        refresh_summary.setdefault("window", window_meta)
        _emit_progress(
            progress_callback,
            task="시장데이터 갱신",
            phase="캐시 정리 중",
            pct=85,
            detail=f"상태 {refresh_summary.get('status', '')}",
        )
        if bool(refresh_summary.get("warehouse_write_skipped")) and isinstance(frame, pd.DataFrame) and not frame.empty:
            set_market_price_transient_override(
                market_id_arg=context.market_id,
                requested_end=context.market_end_date_str,
                status=status,
                summary=refresh_summary,
                frame=frame,
            )
        else:
            clear_market_price_transient_override()
        invalidate_dashboard_caches("market")
        notice = build_market_refresh_notice(refresh_summary)
        _emit_progress(
            progress_callback,
            task="시장데이터 갱신",
            phase="완료",
            pct=100,
            detail=notice[1] if notice else "",
            status="complete",
        )
        return notice
    except Exception as exc:
        clear_market_price_transient_override()
        _emit_progress(
            progress_callback,
            task="시장데이터 갱신",
            phase="실패",
            pct=100,
            detail=str(exc),
            status="error",
        )
        logger.exception("Manual market refresh failed")
        return ("error", f"Market data refresh failed: {exc}")


def _sync_macro_warehouse(*, start_ym: str, end_ym: str, macro_series_cfg: dict, market: str):
    from src.data_sources.macro_sync import sync_macro_warehouse

    return sync_macro_warehouse(
        start_ym=start_ym,
        end_ym=end_ym,
        macro_series_cfg=macro_series_cfg,
        reason="manual_refresh",
        force=False,
        market=market,
    )


def run_macro_refresh(
    context: DashboardContext,
    macro_series_cfg: dict,
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, str] | None:
    """Execute a manual macro refresh and convert the result into a UI notice."""
    macro_end_period = pd.Period(context.market_end_date_str[:6], freq="M")
    macro_end_ym = macro_end_period.strftime("%Y%m")
    macro_start_ym = (macro_end_period - 119).strftime("%Y%m")
    provider_count = sum(bool((macro_series_cfg or {}).get(name)) for name in ("ecos", "kosis", "fred"))
    try:
        _emit_progress(
            progress_callback,
            task="매크로데이터 갱신",
            phase="준비 중",
            pct=5,
            detail=f"{macro_start_ym} ~ {macro_end_ym}",
        )
        close_cached_read_only_connection()
        _emit_progress(
            progress_callback,
            task="매크로데이터 갱신",
            phase="공급자 동기화 중",
            pct=40,
            detail=f"{provider_count or 1} providers",
        )
        _, _, macro_summary = _sync_macro_warehouse(
            start_ym=macro_start_ym,
            end_ym=macro_end_ym,
            macro_series_cfg=macro_series_cfg,
            market=context.market_id,
        )
        _emit_progress(
            progress_callback,
            task="매크로데이터 갱신",
            phase="캐시 정리 중",
            pct=85,
            detail=f"상태 {macro_summary.get('status', '')}",
        )
        invalidate_dashboard_caches("macro")
        notice = build_macro_refresh_notice(macro_summary)
        _emit_progress(
            progress_callback,
            task="매크로데이터 갱신",
            phase="완료",
            pct=100,
            detail=notice[1] if notice else "",
            status="complete",
        )
        return notice
    except Exception as exc:
        _emit_progress(
            progress_callback,
            task="매크로데이터 갱신",
            phase="실패",
            pct=100,
            detail=str(exc),
            status="error",
        )
        logger.exception("Manual macro refresh failed")
        return ("error", f"Macro data refresh failed: {exc}")


def run_investor_flow_refresh(
    context: DashboardContext,
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, str] | None:
    """Execute a manual KR investor-flow refresh and convert the result into a UI notice."""
    if str(context.market_id).strip().upper() != "KR":
        return None

    from src.data_sources.krx_investor_flow import run_manual_investor_flow_refresh

    inner_progress_emitted = False

    def _inner_progress_callback(event: dict[str, Any]) -> None:
        nonlocal inner_progress_emitted
        inner_progress_emitted = True
        if progress_callback is not None:
            progress_callback(event)

    try:
        _emit_progress(
            progress_callback,
            task="투자자수급 갱신",
            phase="준비 중",
            pct=5,
            detail=context.market_end_date_str,
        )
        close_cached_read_only_connection()
        (status, frame), refresh_summary = run_manual_investor_flow_refresh(
            sector_map=context.sector_map,
            end_date_str=context.market_end_date_str,
            market=context.market_id,
            progress_callback=_inner_progress_callback,
        )
        if bool(refresh_summary.get("warehouse_write_skipped")) and isinstance(frame, pd.DataFrame) and not frame.empty:
            set_investor_flow_transient_override(
                market_id_arg=context.market_id,
                requested_end=context.market_end_date_str,
                status=status,
                summary=refresh_summary,
                frame=frame,
            )
        else:
            clear_investor_flow_transient_override()
        invalidate_dashboard_caches("flow")
        notice = build_investor_flow_refresh_notice(refresh_summary)
        if not inner_progress_emitted:
            _emit_progress(
                progress_callback,
                task="투자자수급 갱신",
                phase="완료",
                pct=100,
                detail=notice[1] if notice else "",
                status="complete",
            )
        return notice
    except Exception as exc:
        clear_investor_flow_transient_override()
        _emit_progress(
            progress_callback,
            task="투자자수급 갱신",
            phase="실패",
            pct=100,
            detail=str(exc),
            status="error",
        )
        logger.exception("Manual investor-flow refresh failed")
        return ("error", f"Investor-flow refresh failed: {exc}")

__all__ = [
    "invalidate_dashboard_caches",
    "run_investor_flow_refresh",
    "run_macro_refresh",
    "run_market_refresh",
]
