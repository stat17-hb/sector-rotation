"""
Signal matrix: macro_fit × momentum_state → action.

R7 — ACTION_VALUES = {"Strong Buy", "Watch", "Hold", "Avoid", "N/A"}
"N/A" is only assigned when a sector's price data could not be loaded.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# R7 — 5-value action domain
ACTION_VALUES = frozenset({"Strong Buy", "Watch", "Hold", "Avoid", "N/A"})


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
    trend_ok: bool          # SMA20 > SMA60
    momentum_strong: bool   # rs_strong AND trend_ok
    rsi_d: float            # daily RSI
    rsi_w: float            # weekly RSI
    action: str             # ACTION_VALUES member
    alerts: list[str] = field(default_factory=list)   # e.g. ["Overheat", "Oversold", "FX Shock", "Benchmark Missing", "RS Data Insufficient"]
    returns: dict[str, float] = field(default_factory=dict)  # {1W, 1M, 3M, 6M, 12M}
    volatility_20d: float = float("nan")
    mdd_3m: float = float("nan")
    asof_date: str = ""
    is_provisional: bool = False
    rs_change_pct: float = float("nan")  # 20-day RS change %

    def __post_init__(self) -> None:
        if self.action not in ACTION_VALUES:
            raise ValueError(f"Invalid action: {self.action!r}. Must be one of {ACTION_VALUES}")


def compute_action(macro_fit: bool, momentum_strong: bool) -> str:
    """Map macro_fit × momentum_strong to action label.

    Matrix:
        (True,  True)  → "Strong Buy"
        (True,  False) → "Watch"
        (False, True)  → "Hold"
        (False, False) → "Avoid"

    Args:
        macro_fit: True if sector aligns with current macro regime.
        momentum_strong: True if RS > RS_MA AND SMA20 > SMA60.

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


def build_signal_table(
    sector_prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    macro_result: pd.DataFrame,
    sector_map: dict,
    settings: dict,
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

    Returns:
        list[SectorSignal] — one entry per sector in sector_map.
    """
    from src.indicators.momentum import (
        compute_mdd,
        compute_period_returns,
        compute_rs,
        compute_rs_ma,
        compute_sma,
        compute_volatility,
        is_rs_strong,
        is_trend_positive,
    )
    from src.indicators.rsi import compute_rsi, compute_weekly_rsi
    from src.macro.regime import get_regime_sectors
    from src.signals.scoring import apply_fx_shock_filter, apply_rsi_alerts

    # Validate inputs
    from src.contracts.validators import validate_only
    validate_only(sector_prices, "sector_prices")

    # Current regime (last row of macro_result)
    current_regime = "Indeterminate"
    if not macro_result.empty and "regime" in macro_result.columns:
        current_regime = str(macro_result["regime"].iloc[-1])

    # Regime sectors (the ones that macro_fit=True)
    regime_sector_entries = get_regime_sectors(current_regime, sector_map)
    regime_codes = {s["code"] for s in regime_sector_entries}

    # All sectors across all regimes
    all_regimes = sector_map.get("regimes", {})
    all_sectors: list[dict] = []
    for regime_name, regime_data in all_regimes.items():
        for s in regime_data.get("sectors", []):
            all_sectors.append(
                {
                    "code": str(s["code"]),
                    "name": str(s["name"]),
                    "regime": regime_name,
                    "export_sector": bool(s.get("export_sector", False)),
                }
            )

    # Remove duplicates (same code can appear in multiple regimes)
    seen: set[str] = set()
    unique_sectors: list[dict] = []
    for s in all_sectors:
        if s["code"] not in seen:
            seen.add(s["code"])
            unique_sectors.append(s)

    rs_ma_period = int(settings.get("rs_ma_period", 20))
    ma_fast = int(settings.get("ma_fast", 20))
    ma_slow = int(settings.get("ma_slow", 60))
    rsi_period = int(settings.get("rsi_period", 14))
    fx_shock_pct = float(settings.get("fx_shock_pct", 3.0))

    # FX change — look for USD/KRW in macro_result or use 0
    fx_change_pct = 0.0
    asof_str = ""
    if not sector_prices.empty:
        asof_str = sector_prices.index[-1].strftime("%Y-%m-%d")

    export_sector_codes = [s["code"] for s in unique_sectors if s["export_sector"]]

    signals: list[SectorSignal] = []

    def _build_na_signal(
        *,
        code: str,
        name: str,
        sector_regime: str,
        macro_fit: bool,
        alerts: list[str] | None = None,
    ) -> SectorSignal:
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
                    alerts=["Benchmark Missing"],
                )
            )
        return signals

    rs_insufficient_count = 0

    for sector_info in unique_sectors:
        code = sector_info["code"]
        name = sector_info["name"]
        sector_regime = sector_info["regime"]
        macro_fit = code in regime_codes

        try:
            # Extract this sector's close prices
            mask = sector_prices["index_code"].astype(str) == code
            sector_df = sector_prices[mask].copy()
            if sector_df.empty:
                raise ValueError(f"No price data for sector {code}")

            close = sector_df["close"].sort_index()

            # RS
            rs_series = compute_rs(close, benchmark_series)
            rs_ma_series = compute_rs_ma(rs_series, period=rs_ma_period)
            rs_val = float(rs_series.iloc[-1]) if not rs_series.empty else float("nan")
            rs_ma_val = float(rs_ma_series.iloc[-1]) if not rs_ma_series.empty else float("nan")
            if rs_series.empty or rs_ma_series.empty or pd.isna(rs_val) or pd.isna(rs_ma_val):
                rs_insufficient_count += 1
                logger.warning(
                    "RS data insufficient for sector %s (%s); assigning N/A",
                    code,
                    name,
                )
                signals.append(
                    _build_na_signal(
                        code=code,
                        name=name,
                        sector_regime=sector_regime,
                        macro_fit=macro_fit,
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

            # Trend
            trend_ok = is_trend_positive(close, fast=ma_fast, slow=ma_slow)

            # Momentum
            momentum_strong = rs_strong and trend_ok

            # RSI
            rsi_d_series = compute_rsi(close, period=rsi_period)
            rsi_d = float(rsi_d_series.iloc[-1]) if not rsi_d_series.empty else float("nan")
            rsi_w_series = compute_weekly_rsi(close, period=rsi_period)
            rsi_w = float(rsi_w_series.iloc[-1]) if not rsi_w_series.empty else float("nan")

            # Returns, vol, MDD
            period_returns = compute_period_returns(close)
            vol_20d = compute_volatility(close, window=20)
            mdd_3m = compute_mdd(close, window=63)

            # Action
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
                rsi_d=rsi_d,
                rsi_w=rsi_w,
                action=action,
                alerts=[],
                returns=period_returns,
                volatility_20d=vol_20d,
                mdd_3m=mdd_3m,
                asof_date=asof_str,
                is_provisional=False,
                rs_change_pct=rs_change_pct,
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
                fx_change_pct=fx_change_pct,
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
                )
            )

    if rs_insufficient_count:
        logger.warning(
            "RS data insufficient for %d/%d sectors",
            rs_insufficient_count,
            len(unique_sectors),
        )

    return signals
