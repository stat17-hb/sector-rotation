"""Internal dashboard orchestration types."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DashboardContext:
    """Immutable dashboard runtime context."""

    market_id: str
    market_profile: Any
    settings: dict[str, Any]
    sector_map: dict[str, Any]
    macro_series_cfg: dict[str, Any]
    benchmark_code: str
    market_end_date: date
    market_end_date_str: str
    macro_cache_token: str
    price_cache_token: str
    price_artifact_key: tuple[Any, ...]
    macro_artifact_key: tuple[Any, ...]
    provider_configured: str
    provider_effective: str
    openapi_key_present: bool
    theme_mode: str
    analysis_heatmap_palette: str
    ui_locale: str
    investor_flow_artifact_key: tuple[Any, ...] = ()
    investor_flow_status: str = "SAMPLE"
    investor_flow_profile: str = "foreign_lead"


@dataclass
class DashboardDataBundle:
    """Prepared dashboard data used across the main page."""

    signals: list[Any] = field(default_factory=list)
    macro_result: pd.DataFrame = field(default_factory=pd.DataFrame)
    macro_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    price_status: str = "SAMPLE"
    macro_status: str = "SAMPLE"
    market_blocking_error: str = ""
    dashboard_status_banner: dict[str, Any] | None = None
    current_regime: str = "Indeterminate"
    regime_is_confirmed: bool = False
    yield_curve_status: str | None = None
    growth_val: float | None = None
    inflation_val: float | None = None
    fx_change: float | None = None
    price_warm_status: dict[str, Any] = field(default_factory=dict)
    price_cache_case: str | None = None
    preflight_status: dict[str, Any] = field(default_factory=dict)
    sector_prices_canvas: pd.DataFrame = field(default_factory=pd.DataFrame)
    sector_name_map: dict[str, str] = field(default_factory=dict)
    prices_wide: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_label: str = ""
    sector_columns: list[str] = field(default_factory=list)
    monthly_close_full: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_returns_full: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_monthly_return: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    monthly_excess_returns_full: pd.DataFrame = field(default_factory=pd.DataFrame)
    cycle_segments_all: list[dict[str, Any]] = field(default_factory=list)
    phase_by_month: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    analysis_min_date: date | None = None
    analysis_max_date: date | None = None
    market_refresh_notice: tuple[str, str] | None = None
    macro_refresh_notice: tuple[str, str] | None = None
    investor_flow_status: str = "SAMPLE"
    investor_flow_fresh: bool = False
    investor_flow_profile: str = "foreign_lead"
    investor_flow_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    investor_flow_detail: dict[str, Any] = field(default_factory=dict)
    investor_flow_refresh_notice: tuple[str, str] | None = None


@dataclass(frozen=True)
class AnalysisWindow:
    """Selected analysis range state."""

    start_date: date
    end_date: date
    selected_range_preset: str
    selected_cycle_phase: str
    selected_sector: str
    selected_month: str
