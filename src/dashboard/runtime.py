"""Runtime orchestration helpers for dashboard refresh flows."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable, Literal

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
    get_all_sector_codes,
    get_investor_flow_range_strings,
    get_market_range_strings,
)
from src.dashboard.types import DashboardContext
from src.data_sources.warehouse import close_cached_read_only_connection


logger = logging.getLogger(__name__)

CacheScope = Literal["all", "market", "macro", "flow", "signals"]


def invalidate_dashboard_caches(scope: CacheScope) -> None:
    """Clear the dashboard cache family associated with the requested scope."""
    if scope == "all":
        cached_api_preflight.clear()
        cached_sector_prices.clear()
        cached_analysis_sector_prices.clear()
        cached_investor_flow.clear()
        cached_macro.clear()
        cached_signals.clear()
        return
    if scope == "market":
        cached_api_preflight.clear()
        cached_sector_prices.clear()
        cached_analysis_sector_prices.clear()
        cached_signals.clear()
        return
    if scope == "macro":
        cached_macro.clear()
        cached_signals.clear()
        return
    if scope == "flow":
        cached_investor_flow.clear()
        cached_signals.clear()
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


def run_market_refresh(context: DashboardContext, price_years: int) -> tuple[str, str] | None:
    """Execute a manual market refresh and convert the result into a UI notice."""
    refresh_start_str, refresh_end_str = get_market_range_strings(context.market_end_date_str, price_years)
    runner = _resolve_market_refresh_runner(context.market_id)
    try:
        close_cached_read_only_connection()
        cached_api_preflight.clear()
        (_, _), refresh_summary = runner(
            get_all_sector_codes(context.benchmark_code),
            refresh_start_str,
            refresh_end_str,
        )
        invalidate_dashboard_caches("market")
        return build_market_refresh_notice(refresh_summary)
    except Exception as exc:
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


def run_macro_refresh(context: DashboardContext, macro_series_cfg: dict) -> tuple[str, str] | None:
    """Execute a manual macro refresh and convert the result into a UI notice."""
    macro_end_period = pd.Period(context.market_end_date_str[:6], freq="M")
    macro_end_ym = macro_end_period.strftime("%Y%m")
    macro_start_ym = (macro_end_period - 119).strftime("%Y%m")
    try:
        close_cached_read_only_connection()
        _, _, macro_summary = _sync_macro_warehouse(
            start_ym=macro_start_ym,
            end_ym=macro_end_ym,
            macro_series_cfg=macro_series_cfg,
            market=context.market_id,
        )
        invalidate_dashboard_caches("macro")
        return build_macro_refresh_notice(macro_summary)
    except Exception as exc:
        logger.exception("Manual macro refresh failed")
        return ("error", f"Macro data refresh failed: {exc}")


def run_investor_flow_refresh(context: DashboardContext) -> tuple[str, str] | None:
    """Execute a manual KR investor-flow refresh and convert the result into a UI notice."""
    if str(context.market_id).strip().upper() != "KR":
        return None

    from src.data_sources.krx_investor_flow import run_manual_investor_flow_refresh

    _, end_str = get_investor_flow_range_strings(context.market_end_date_str)
    try:
        close_cached_read_only_connection()
        (_, _), refresh_summary = run_manual_investor_flow_refresh(
            sector_map=context.sector_map,
            end_date_str=end_str,
            market=context.market_id,
        )
        invalidate_dashboard_caches("flow")
        return build_investor_flow_refresh_notice(refresh_summary)
    except Exception as exc:
        logger.exception("Manual investor-flow refresh failed")
        return ("error", f"Investor-flow refresh failed: {exc}")


def run_feature_recompute() -> None:
    """Clear derived feature artifacts and invalidate signal caches."""
    close_cached_read_only_connection()
    shutil.rmtree("data/features", ignore_errors=True)
    Path("data/features").mkdir(exist_ok=True)
    invalidate_dashboard_caches("signals")


__all__ = [
    "invalidate_dashboard_caches",
    "run_feature_recompute",
    "run_investor_flow_refresh",
    "run_macro_refresh",
    "run_market_refresh",
]
