"""
RSI alerts and FX shock downgrade scoring.
"""
from __future__ import annotations

from src.signals.matrix import SectorSignal


def apply_rsi_alerts(
    signal: SectorSignal,
    overbought: int = 70,
    oversold: int = 30,
) -> SectorSignal:
    """Add RSI alert tags to signal.

    Adds "Overheat" if daily RSI >= overbought.
    Adds "Oversold" if daily RSI <= oversold.
    Does not modify action.

    Args:
        signal: SectorSignal to evaluate.
        overbought: RSI threshold for overbought (default 70).
        oversold: RSI threshold for oversold (default 30).

    Returns:
        Updated SectorSignal with alerts list modified (copy via replacement).
    """
    import math
    alerts = list(signal.alerts)
    if not math.isnan(signal.rsi_d):
        if signal.rsi_d >= overbought:
            if "Overheat" not in alerts:
                alerts.append("Overheat")
        elif signal.rsi_d <= oversold:
            if "Oversold" not in alerts:
                alerts.append("Oversold")

    return SectorSignal(
        index_code=signal.index_code,
        sector_name=signal.sector_name,
        macro_regime=signal.macro_regime,
        macro_fit=signal.macro_fit,
        rs=signal.rs,
        rs_ma=signal.rs_ma,
        rs_strong=signal.rs_strong,
        trend_ok=signal.trend_ok,
        momentum_strong=signal.momentum_strong,
        rsi_d=signal.rsi_d,
        rsi_w=signal.rsi_w,
        action=signal.action,
        alerts=alerts,
        returns=signal.returns,
        volatility_20d=signal.volatility_20d,
        mdd_3m=signal.mdd_3m,
        asof_date=signal.asof_date,
        is_provisional=signal.is_provisional,
    )


def apply_fx_shock_filter(
    signal: SectorSignal,
    fx_change_pct: float,
    export_sectors: list[str],
    threshold_pct: float = 3.0,
) -> SectorSignal:
    """Downgrade export-sector Strong Buy to Watch on FX shock.

    Rule: if abs(fx_change_pct) > threshold_pct AND sector is in export_sectors
          AND current action == "Strong Buy" → downgrade to "Watch" and add "FX Shock" alert.

    Note: FX shock check uses absolute value — both USD/KRW spike and crash are shocks.

    Args:
        signal: SectorSignal to evaluate.
        fx_change_pct: Recent FX change percentage (e.g. 4.0 for +4%).
        export_sectors: List of index_code strings that are export-sensitive.
        threshold_pct: Minimum |fx_change_pct| to trigger downgrade (default 3.0).

    Returns:
        Updated SectorSignal. May have action changed to "Watch" and "FX Shock" added to alerts.
    """
    import math

    if math.isnan(fx_change_pct):
        return signal

    is_export = signal.index_code in export_sectors
    is_shock = abs(fx_change_pct) > threshold_pct
    is_strong_buy = signal.action == "Strong Buy"

    if is_export and is_shock and is_strong_buy:
        alerts = list(signal.alerts)
        if "FX Shock" not in alerts:
            alerts.append("FX Shock")
        return SectorSignal(
            index_code=signal.index_code,
            sector_name=signal.sector_name,
            macro_regime=signal.macro_regime,
            macro_fit=signal.macro_fit,
            rs=signal.rs,
            rs_ma=signal.rs_ma,
            rs_strong=signal.rs_strong,
            trend_ok=signal.trend_ok,
            momentum_strong=signal.momentum_strong,
            rsi_d=signal.rsi_d,
            rsi_w=signal.rsi_w,
            action="Watch",
            alerts=alerts,
            returns=signal.returns,
            volatility_20d=signal.volatility_20d,
            mdd_3m=signal.mdd_3m,
            asof_date=signal.asof_date,
            is_provisional=signal.is_provisional,
        )

    return signal
