"""
Signal matrix: macro_fit × momentum_state → action.

R7 — ACTION_VALUES = {"Strong Buy", "Watch", "Hold", "Avoid", "N/A"}
"N/A" is only assigned when a sector's price data could not be loaded.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# R7 — 5-value action domain
ACTION_VALUES = frozenset({"Strong Buy", "Watch", "Hold", "Avoid", "N/A"})
LEGACY_MOMENTUM_METHOD = "legacy_rs_ma_v0"
HYBRID_MOMENTUM_METHOD = "hybrid_return_rank_v1"


@dataclass
class SectorSignal:
    """Complete signal record for one sector.

    action domain: "Strong Buy" | "Watch" | "Hold" | "Avoid" | "N/A"
    "N/A" only when price data could not be loaded for this sector.
    """
    index_code: str
    sector_name: str
    macro_regime: str
    macro_fit: bool
    rs: float
    rs_ma: float
    rs_strong: bool
    trend_ok: bool
    momentum_strong: bool
    rsi_d: float            # daily RSI
    rsi_w: float            # weekly RSI
    action: str             # ACTION_VALUES member
    base_action: str = ""
    flow_adjusted_action: str = ""
    flow_adjustment: str = "none"
    flow_profile: str = "foreign_lead"
    flow_state: str = "unavailable"
    flow_score: float = 0.0
    flow_reason: str = ""
    foreign_flow_state: str = "unavailable"
    institutional_flow_state: str = "unavailable"
    retail_flow_state: str = "unavailable"
    foreign_flow_ratio: float = float("nan")
    institutional_flow_ratio: float = float("nan")
    retail_flow_ratio: float = float("nan")
    foreign_flow_z: float = float("nan")
    institutional_flow_z: float = float("nan")
    retail_flow_z: float = float("nan")
    alerts: list[str] = field(default_factory=list)   # e.g. ["Overheat", "Oversold", "FX Shock", "Benchmark Missing", "RS Data Insufficient"]
    returns: dict[str, float] = field(default_factory=dict)  # {1W, 1M, 3M, 6M, 12M}
    volatility_20d: float = float("nan")
    mdd_3m: float = float("nan")
    asof_date: str = ""
    is_provisional: bool = False
    rs_change_pct: float = float("nan")  # 20-day RS change %
    momentum_method: str = LEGACY_MOMENTUM_METHOD
    legacy_trend_ok: bool = False
    legacy_momentum_strong: bool = False
    momentum_core_pass: bool = False
    momentum_rank_pass: bool = False
    mom_rel_6m_ex1m: float = float("nan")
    mom_rel_12m_ex1m: float = float("nan")
    mom_score: float = float("nan")
    mom_raw: float = float("nan")
    mom_rank: int | None = None
    mom_percentile: float = float("nan")
    sector_fit_rank: int | None = None
    sector_fit_total: int | None = None
    sector_fit_avg_excess_pct: float = float("nan")
    sector_fit_note: str = ""
    sector_fit_cross_regimes: tuple[str, ...] = ()
    macro_context_regime: str = ""
    action_policy: str = "LEGACY_REGIME_MOMENTUM"
    taxonomy_kind: str = ""
    taxonomy_label: str = ""

    def __post_init__(self) -> None:
        if self.action not in ACTION_VALUES:
            raise ValueError(f"Invalid action: {self.action!r}. Must be one of {ACTION_VALUES}")
        if self.base_action and self.base_action not in ACTION_VALUES:
            raise ValueError(f"Invalid base_action: {self.base_action!r}. Must be one of {ACTION_VALUES}")
        if self.flow_adjusted_action and self.flow_adjusted_action not in ACTION_VALUES:
            raise ValueError(
                f"Invalid flow_adjusted_action: {self.flow_adjusted_action!r}. Must be one of {ACTION_VALUES}"
            )
        if not self.base_action:
            self.base_action = self.action
        if not self.flow_adjusted_action:
            self.flow_adjusted_action = self.action


def compute_action(macro_fit: bool, momentum_strong: bool) -> str:
    """Map macro_fit × momentum_strong to action label.

    Matrix:
        (True,  True)  → "Strong Buy"
        (True,  False) → "Watch"
        (False, True)  → "Hold"
        (False, False) → "Avoid"

    Args:
        macro_fit: True if sector aligns with current macro regime.
        momentum_strong: True if the active momentum method passes.

    Returns:
        Action string (never "N/A" — that is reserved for data load failures).
    """
    if macro_fit and momentum_strong:
        return "Strong Buy"
    if macro_fit and not momentum_strong:
        return "Watch"
    if not macro_fit and momentum_strong:
        return "Hold"
    return "Avoid"


def compute_kr_action(momentum_core_pass: bool, trend_ok: bool) -> str:
    """Map KR momentum_core_pass × trend_ok to the 4-value action label."""
    if momentum_core_pass and trend_ok:
        return "Strong Buy"
    if momentum_core_pass and not trend_ok:
        return "Watch"
    if not momentum_core_pass and trend_ok:
        return "Hold"
    return "Avoid"


def _normalize_momentum_method(settings: dict) -> str:
    method = str(settings.get("momentum_method", LEGACY_MOMENTUM_METHOD) or "").strip()
    if method == HYBRID_MOMENTUM_METHOD:
        return HYBRID_MOMENTUM_METHOD
    return LEGACY_MOMENTUM_METHOD


def _build_hybrid_momentum_map(
    *,
    sector_close_map: dict[str, pd.Series],
    benchmark_series: pd.Series,
    skip_recent_days: int,
    lookback_6m_days: int,
    lookback_12m_days: int,
    rank_threshold_pct: float,
) -> dict[str, dict[str, float | int | bool]]:
    from src.indicators.momentum import (
        compute_descending_rank,
        compute_percentile_rank,
        compute_price_above_sma,
        compute_relative_return_excluding_recent,
    )

    raw_6m: dict[str, float] = {}
    raw_12m: dict[str, float] = {}
    trend_flags: dict[str, bool] = {}

    for code, close in sector_close_map.items():
        aligned_sector, aligned_bench = close.align(benchmark_series, join="inner")
        if aligned_sector.empty or aligned_bench.empty:
            continue
        required_history = lookback_12m_days + skip_recent_days
        if len(aligned_sector) < required_history or len(aligned_bench) < required_history:
            continue
        mom_6m = compute_relative_return_excluding_recent(
            aligned_sector,
            aligned_bench,
            lookback_days=lookback_6m_days,
            skip_recent_days=skip_recent_days,
        )
        mom_12m = compute_relative_return_excluding_recent(
            aligned_sector,
            aligned_bench,
            lookback_days=lookback_12m_days,
            skip_recent_days=skip_recent_days,
        )
        if pd.isna(mom_6m) or pd.isna(mom_12m):
            continue
        raw_6m[code] = float(mom_6m)
        raw_12m[code] = float(mom_12m)
        trend_flags[code] = bool(compute_price_above_sma(aligned_sector, window=200))

    if not raw_6m or not raw_12m:
        return {}

    pct_6m = compute_percentile_rank(raw_6m, ascending=True)
    pct_12m = compute_percentile_rank(raw_12m, ascending=True)
    combined_score = (pct_6m * 0.5) + (pct_12m * 0.5)
    combined_raw = (pd.Series(raw_6m, dtype="float64") * 0.5) + (pd.Series(raw_12m, dtype="float64") * 0.5)
    descending_rank = compute_descending_rank(combined_score)

    result: dict[str, dict[str, float | int | bool]] = {}
    for code, score in combined_score.items():
        if pd.isna(score):
            continue
        raw_value = float(combined_raw.get(code, float("nan")))
        rank_pass = bool(float(score) >= float(rank_threshold_pct) and not pd.isna(raw_value) and raw_value > 0)
        trend_ok = bool(trend_flags.get(code, False))
        rank_value = descending_rank.get(code)
        result[str(code)] = {
            "mom_rel_6m_ex1m": float(raw_6m[code]),
            "mom_rel_12m_ex1m": float(raw_12m[code]),
            "mom_score": float(score),
            "mom_raw": raw_value,
            "mom_rank": None if pd.isna(rank_value) else int(rank_value),
            "mom_percentile": float(score) * 100.0,
            "trend_ok": trend_ok,
            "momentum_rank_pass": rank_pass,
            "momentum_strong": bool(rank_pass and trend_ok),
        }
    return result


def build_signal_table(
    sector_prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    macro_result: pd.DataFrame,
    sector_map: dict,
    settings: dict,
    market_id: str = "KR",
    sector_universe_rows: list[dict[str, Any]] | None = None,
    fx_change_pct: float | None = None,
    sector_investor_flow: pd.DataFrame | None = None,
    flow_profile: str = "foreign_lead",
    flow_enabled: bool = False,
) -> list[SectorSignal]:
    """Build complete signal table for all sectors.

    Partial-failure tolerance: if a sector's price data is missing/fails,
    its action is set to "N/A" and remaining sectors are processed normally.

    Args:
        sector_prices: DataFrame with columns [index_code, index_name, close]
                       and DatetimeIndex. From load_sector_prices().
        benchmark_prices: Series of benchmark close prices (DatetimeIndex).
        macro_result: DataFrame from compute_regime_history() with columns
                      [growth_dir, inflation_dir, regime].
        sector_map: Parsed config/sector_map.yml.
        settings: Dict with keys: rs_ma_period, ma_fast, ma_slow, rsi_period,
                  rsi_overbought, rsi_oversold, fx_shock_pct.
        fx_change_pct: Recent USD/KRW change (%). If None/NaN, FX shock is skipped.

    Returns:
        list[SectorSignal] — one entry per sector in sector_map or sector_universe_rows.
    """
    from src.indicators.momentum import (
        compute_mdd,
        compute_period_returns,
        compute_rs,
        compute_rs_ma,
        compute_volatility,
        is_rs_strong,
        is_trend_positive,
    )
    from src.indicators.rsi import compute_rsi, compute_weekly_rsi
    from src.macro.regime import get_regime_sectors
    from src.signals.flow import apply_flow_overlay
    from src.signals.sector_fit import build_sector_fit_lookup
    from src.signals.scoring import apply_fx_shock_filter, apply_rsi_alerts

    # Validate inputs
    from src.contracts.validators import validate_only
    validate_only(sector_prices, "sector_prices")

    # Current regime: use confirmed_regime if available, else raw regime
    current_regime = "Indeterminate"
    if not macro_result.empty:
        if "confirmed_regime" in macro_result.columns:
            current_regime = str(macro_result["confirmed_regime"].iloc[-1])
        elif "regime" in macro_result.columns:
            current_regime = str(macro_result["regime"].iloc[-1])

    normalized_market = str(market_id or "KR").strip().upper() or "KR"

    use_kr_official_universe = normalized_market == "KR" and bool(sector_universe_rows)

    # Non-KR markets and KR fallback paths still use sector_map membership directly.
    if use_kr_official_universe:
        regime_codes: set[str] = set()
        all_sectors: list[dict[str, Any]] = []
    else:
        regime_sector_entries = get_regime_sectors(current_regime, sector_map)
        regime_codes = {s["code"] for s in regime_sector_entries}

        all_regimes = sector_map.get("regimes", {})
        all_sectors = []
        for regime_name, regime_data in all_regimes.items():
            for s in regime_data.get("sectors", []):
                code = str(s["code"])
                all_sectors.append(
                    {
                        "code": code,
                        "name": str(s["name"]),
                        "regime": regime_name,
                        "export_sector": bool(s.get("export_sector", False)),
                        "taxonomy_kind": "",
                        "taxonomy_label": "",
                    }
                )

    if use_kr_official_universe:
        unique_sectors = [
            {
                "code": str(row.get("index_code", "")).strip(),
                "name": str(row.get("index_name", "")).strip() or str(row.get("index_code", "")).strip(),
                "regime": "Unassigned",
                "export_sector": bool(row.get("export_sector", False)),
                "taxonomy_kind": str(row.get("taxonomy_kind", "") or "").strip(),
                "taxonomy_label": str(row.get("taxonomy_label", "") or "").strip(),
            }
            for row in sector_universe_rows
            if str(row.get("index_code", "")).strip()
        ]
    else:
        seen: set[str] = set()
        unique_sectors = []
        for s in all_sectors:
            if s["code"] not in seen:
                seen.add(s["code"])
                unique_sectors.append(s)

    rs_ma_period = int(settings.get("rs_ma_period", 20))
    ma_fast = int(settings.get("ma_fast", 20))
    ma_slow = int(settings.get("ma_slow", 60))
    rsi_period = int(settings.get("rsi_period", 14))
    fx_shock_pct = float(settings.get("fx_shock_pct", 3.0))
    momentum_method = _normalize_momentum_method(settings)
    momentum_skip_recent_days = int(settings.get("momentum_skip_recent_days", 21))
    momentum_lookback_6m_days = int(settings.get("momentum_lookback_6m_days", 126))
    momentum_lookback_12m_days = int(settings.get("momentum_lookback_12m_days", 252))
    momentum_rank_threshold_pct = float(settings.get("momentum_rank_threshold_pct", 0.60))

    # Runtime FX change comes from app wiring (USD/KRW %).
    fx_change_value = float("nan")
    if fx_change_pct is not None:
        try:
            fx_change_value = float(fx_change_pct)
        except (TypeError, ValueError):
            fx_change_value = float("nan")
    asof_str = ""
    if not sector_prices.empty:
        asof_str = sector_prices.index[-1].strftime("%Y-%m-%d")

    export_sector_codes = [s["code"] for s in unique_sectors if s["export_sector"]]

    signals: list[SectorSignal] = []
    sector_close_map: dict[str, pd.Series] = {}

    def _build_na_signal(
        *,
        code: str,
        name: str,
        sector_regime: str,
        macro_fit: bool,
        taxonomy_kind: str = "",
        taxonomy_label: str = "",
        alerts: list[str] | None = None,
    ) -> SectorSignal:
        action_policy = "KR_MOMENTUM_ONLY" if normalized_market == "KR" else "LEGACY_REGIME_MOMENTUM"
        return SectorSignal(
            index_code=code,
            sector_name=name,
            macro_regime=sector_regime,
            macro_fit=macro_fit,
            rs=float("nan"),
            rs_ma=float("nan"),
            rs_strong=False,
            trend_ok=False,
            momentum_strong=False,
            rsi_d=float("nan"),
            rsi_w=float("nan"),
            action="N/A",
            alerts=list(alerts or []),
            returns={},
            asof_date=asof_str,
            momentum_method=momentum_method,
            macro_context_regime=current_regime,
            action_policy=action_policy,
            taxonomy_kind=taxonomy_kind,
            taxonomy_label=taxonomy_label,
        )

    benchmark_series = benchmark_prices.sort_index().dropna()
    if benchmark_series.empty:
        logger.warning(
            "Benchmark prices missing/empty; assigning N/A with 'Benchmark Missing' to all %d sectors",
            len(unique_sectors),
        )
        for sector_info in unique_sectors:
            code = sector_info["code"]
            name = sector_info["name"]
            sector_regime = sector_info["regime"]
            macro_fit = code in regime_codes
            signals.append(
                _build_na_signal(
                    code=code,
                    name=name,
                    sector_regime=sector_regime,
                    macro_fit=macro_fit,
                    taxonomy_kind=str(sector_info.get("taxonomy_kind", "") or ""),
                    taxonomy_label=str(sector_info.get("taxonomy_label", "") or ""),
                    alerts=["Benchmark Missing"],
                )
            )
        return signals

    for sector_info in unique_sectors:
        code = sector_info["code"]
        mask = sector_prices["index_code"].astype(str) == code
        sector_df = sector_prices[mask].copy()
        if sector_df.empty:
            continue
        sector_close_map[code] = sector_df["close"].sort_index()

    hybrid_momentum_map: dict[str, dict[str, float | int | bool]] = {}
    if momentum_method == HYBRID_MOMENTUM_METHOD:
        hybrid_momentum_map = _build_hybrid_momentum_map(
            sector_close_map=sector_close_map,
            benchmark_series=benchmark_series,
            skip_recent_days=momentum_skip_recent_days,
            lookback_6m_days=momentum_lookback_6m_days,
            lookback_12m_days=momentum_lookback_12m_days,
            rank_threshold_pct=momentum_rank_threshold_pct,
        )

    for sector_info in unique_sectors:
        code = sector_info["code"]
        name = sector_info["name"]
        sector_regime = sector_info["regime"]
        macro_fit = code in regime_codes
        taxonomy_kind = str(sector_info.get("taxonomy_kind", "") or "")
        taxonomy_label = str(sector_info.get("taxonomy_label", "") or "")
        action_policy = "KR_MOMENTUM_ONLY" if normalized_market == "KR" else "LEGACY_REGIME_MOMENTUM"

        try:
            # Extract this sector's close prices
            mask = sector_prices["index_code"].astype(str) == code
            sector_df = sector_prices[mask].copy()
            if sector_df.empty:
                raise ValueError(f"No price data for sector {code}")

            close = sector_df["close"].sort_index()

            # Legacy RS diagnostics stay available under both methods.
            rs_series = compute_rs(close, benchmark_series)
            rs_ma_series = compute_rs_ma(rs_series, period=rs_ma_period)
            rs_val = float(rs_series.iloc[-1]) if not rs_series.empty else float("nan")
            rs_ma_val = float(rs_ma_series.iloc[-1]) if not rs_ma_series.empty else float("nan")
            if momentum_method == LEGACY_MOMENTUM_METHOD and (
                rs_series.empty or rs_ma_series.empty or pd.isna(rs_val) or pd.isna(rs_ma_val)
            ):
                signals.append(
                    _build_na_signal(
                        code=code,
                        name=name,
                        sector_regime=sector_regime,
                        macro_fit=macro_fit,
                        taxonomy_kind=taxonomy_kind,
                        taxonomy_label=taxonomy_label,
                        alerts=["RS Data Insufficient"],
                    )
                )
                continue
            rs_strong = bool(is_rs_strong(rs_val, rs_ma_val)) if not (pd.isna(rs_val) or pd.isna(rs_ma_val)) else False

            # RS 20-day momentum (acceleration)
            rs_change_pct = float("nan")
            if len(rs_series) >= 21 and not pd.isna(rs_val):
                rs_20d_ago = float(rs_series.iloc[-21])
                if not pd.isna(rs_20d_ago) and rs_20d_ago != 0:
                    rs_change_pct = (rs_val - rs_20d_ago) / abs(rs_20d_ago) * 100

            legacy_trend_ok = is_trend_positive(close, fast=ma_fast, slow=ma_slow)
            legacy_momentum_strong = rs_strong and legacy_trend_ok

            # RSI
            rsi_d_series = compute_rsi(close, period=rsi_period)
            rsi_d = float(rsi_d_series.iloc[-1]) if not rsi_d_series.empty else float("nan")
            rsi_w_series = compute_weekly_rsi(close, period=rsi_period)
            rsi_w = float(rsi_w_series.iloc[-1]) if not rsi_w_series.empty else float("nan")

            # Returns, vol, MDD
            period_returns = compute_period_returns(close)
            vol_20d = compute_volatility(close, window=20)
            mdd_3m = compute_mdd(close, window=63)

            alerts: list[str] = []
            mom_rel_6m_ex1m = float("nan")
            mom_rel_12m_ex1m = float("nan")
            mom_score = float("nan")
            mom_raw = float("nan")
            mom_rank: int | None = None
            mom_percentile = float("nan")
            trend_ok = legacy_trend_ok
            momentum_strong = legacy_momentum_strong
            momentum_core_pass = rs_strong
            momentum_rank_pass = False

            if momentum_method == HYBRID_MOMENTUM_METHOD:
                hybrid_meta = hybrid_momentum_map.get(code)
                if hybrid_meta is None:
                    signals.append(
                        _build_na_signal(
                            code=code,
                            name=name,
                            sector_regime=sector_regime,
                            macro_fit=macro_fit,
                            taxonomy_kind=taxonomy_kind,
                            taxonomy_label=taxonomy_label,
                            alerts=["Momentum History Insufficient"],
                        )
                    )
                    continue
                mom_rel_6m_ex1m = float(hybrid_meta["mom_rel_6m_ex1m"])
                mom_rel_12m_ex1m = float(hybrid_meta["mom_rel_12m_ex1m"])
                mom_score = float(hybrid_meta["mom_score"])
                mom_raw = float(hybrid_meta["mom_raw"])
                mom_rank = int(hybrid_meta["mom_rank"]) if hybrid_meta["mom_rank"] is not None else None
                mom_percentile = float(hybrid_meta["mom_percentile"])
                trend_ok = bool(hybrid_meta["trend_ok"])
                momentum_rank_pass = bool(hybrid_meta.get("momentum_rank_pass", False))
                momentum_core_pass = momentum_rank_pass
                momentum_strong = bool(hybrid_meta["momentum_strong"])

            # Action
            if normalized_market == "KR":
                action = compute_kr_action(momentum_core_pass, trend_ok)
                macro_fit = code in regime_codes
            else:
                action = compute_action(macro_fit, momentum_strong)

            sig = SectorSignal(
                index_code=code,
                sector_name=name,
                macro_regime=sector_regime,
                macro_fit=macro_fit,
                rs=rs_val,
                rs_ma=rs_ma_val,
                rs_strong=rs_strong,
                trend_ok=trend_ok,
                momentum_strong=momentum_strong,
                momentum_core_pass=momentum_core_pass,
                momentum_rank_pass=momentum_rank_pass,
                rsi_d=rsi_d,
                rsi_w=rsi_w,
                action=action,
                alerts=alerts,
                returns=period_returns,
                volatility_20d=vol_20d,
                mdd_3m=mdd_3m,
                asof_date=asof_str,
                is_provisional=False,
                rs_change_pct=rs_change_pct,
                momentum_method=momentum_method,
                legacy_trend_ok=legacy_trend_ok,
                legacy_momentum_strong=legacy_momentum_strong,
                mom_rel_6m_ex1m=mom_rel_6m_ex1m,
                mom_rel_12m_ex1m=mom_rel_12m_ex1m,
                mom_score=mom_score,
                mom_raw=mom_raw,
                mom_rank=mom_rank,
                mom_percentile=mom_percentile,
                macro_context_regime=current_regime,
                action_policy=action_policy,
                taxonomy_kind=taxonomy_kind,
                taxonomy_label=taxonomy_label,
            )

            # Apply RSI alerts
            sig = apply_rsi_alerts(
                sig,
                overbought=int(settings.get("rsi_overbought", 70)),
                oversold=int(settings.get("rsi_oversold", 30)),
            )

            # Apply FX shock filter
            sig = apply_fx_shock_filter(
                sig,
                fx_change_pct=fx_change_value,
                export_sectors=export_sector_codes,
                threshold_pct=fx_shock_pct,
            )

            signals.append(sig)

        except Exception as exc:
            logger.warning(
                "Failed to compute signal for sector %s (%s): %s — assigning N/A",
                code,
                name,
                exc,
            )
            signals.append(
                _build_na_signal(
                    code=code,
                    name=name,
                    sector_regime=sector_regime,
                    macro_fit=macro_fit,
                    taxonomy_kind=taxonomy_kind,
                    taxonomy_label=taxonomy_label,
                )
            )

    flow_frame = sector_investor_flow if sector_investor_flow is not None else pd.DataFrame()
    flow_short_window = int(settings.get("investor_flow_short_window", 20))
    flow_long_window = int(settings.get("investor_flow_long_window", 60))
    signals, _flow_summary_map = apply_flow_overlay(
        signals,
        flow_frame=flow_frame,
        flow_profile=flow_profile,
        enabled=flow_enabled,
        short_window=flow_short_window,
        long_window=flow_long_window,
    )
    sector_fit_lookup = build_sector_fit_lookup(regime=current_regime, lag_months=0)
    if not sector_fit_lookup:
        return signals

    enriched: list[SectorSignal] = []
    for signal in signals:
        fit_meta = sector_fit_lookup.get(str(signal.index_code))
        if not fit_meta:
            enriched.append(signal)
            continue
        enriched.append(
            replace(
                signal,
                sector_fit_rank=int(fit_meta["sector_fit_rank"]),
                sector_fit_total=int(fit_meta["sector_fit_total"]),
                sector_fit_avg_excess_pct=float(fit_meta["sector_fit_avg_excess_pct"]),
                sector_fit_note=str(fit_meta.get("sector_fit_note", "") or ""),
                sector_fit_cross_regimes=tuple(fit_meta.get("sector_fit_cross_regimes", ()) or ()),
            )
        )
    return enriched
