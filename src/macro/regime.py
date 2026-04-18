"""
4-phase macro regime classification.

Phases: Recovery | Expansion | Slowdown | Contraction | Indeterminate
"""
from __future__ import annotations

import pandas as pd


# Regime classification matrix:
#   growth_dir × inflation_dir → regime
_REGIME_MATRIX: dict[tuple[str, str], str] = {
    ("Up", "Down"): "Recovery",
    ("Up", "Up"): "Expansion",
    ("Down", "Up"): "Slowdown",
    ("Down", "Down"): "Contraction",
}


def classify_regime(growth_dir: str, inflation_dir: str) -> str:
    """Classify macro regime from growth and inflation directions.

    Args:
        growth_dir: "Up", "Down", or "Flat".
        inflation_dir: "Up", "Down", or "Flat".

    Returns:
        One of: "Recovery", "Expansion", "Slowdown", "Contraction", "Indeterminate".
    """
    return _REGIME_MATRIX.get((growth_dir, inflation_dir), "Indeterminate")


def get_regime_sectors(regime: str, sector_map: dict) -> list[dict]:
    """Return sectors associated with a given regime.

    Args:
        regime: One of the four regime names or "Indeterminate".
        sector_map: Parsed config/sector_map.yml content.

    Returns:
        list of dicts with keys: code (str), name (str), export_sector (bool).
        Returns empty list for "Indeterminate" or unknown regimes.
    """
    regimes = sector_map.get("regimes", {})
    regime_data = regimes.get(regime, {})
    sectors = regime_data.get("sectors", [])
    return [
        {
            "code": str(s["code"]),
            "name": str(s["name"]),
            "export_sector": bool(s.get("export_sector", False)),
        }
        for s in sectors
    ]


def compute_regime_history(
    growth_series: pd.Series,
    inflation_series: pd.Series,
    epsilon: float = 0.0,
    use_adaptive_epsilon: bool = False,
    epsilon_factor: float = 0.5,
    confirmation_periods: int = 1,
    carry_single_flat_regime: bool = False,
    yield_curve_spread: pd.Series | None = None,
    yield_curve_threshold: float = 0.0,
) -> pd.DataFrame:
    """Compute regime classification over time.

    Args:
        growth_series: Monthly leading indicator values (e.g. 경기선행지수순환변동치).
        inflation_series: Monthly inflation values (e.g. CPI YoY or MoM).
        epsilon: Minimum absolute change to count as directional (default 0).
            When 0 and use_adaptive_epsilon=True, computed per-series from std.
        use_adaptive_epsilon: If True and epsilon==0, auto-compute per-series epsilon
            as epsilon_factor × std(diff).
        epsilon_factor: Multiplier for adaptive epsilon (default 0.5).
        confirmation_periods: Regime must persist this many consecutive months
            before being confirmed (default 1 = no filter).
        carry_single_flat_regime: When True, preserve the last non-Indeterminate
            raw regime if exactly one leg turns Flat. Dual-Flat and startup
            periods stay Indeterminate.
        yield_curve_spread: Optional monthly spread series (e.g. KTB3Y - base_rate).
            Adds a "yield_curve" column ("Normal" / "Inverted") to the result.
        yield_curve_threshold: Spread below this value is classified "Inverted"
            (default 0.0).

    Returns:
        DataFrame indexed same as inputs with columns:
        [growth_dir, inflation_dir, regime, confirmed_regime]
        and optionally [yield_curve].
    """
    from src.transforms.resample import (
        apply_confirmation_filter,
        compute_3ma_direction,
        compute_adaptive_epsilon,
    )

    if use_adaptive_epsilon and epsilon == 0.0:
        g_eps = compute_adaptive_epsilon(growth_series, factor=epsilon_factor)
        i_eps = compute_adaptive_epsilon(inflation_series, factor=epsilon_factor)
    else:
        g_eps = epsilon
        i_eps = epsilon

    growth_dir = compute_3ma_direction(growth_series, epsilon=g_eps)
    inflation_dir = compute_3ma_direction(inflation_series, epsilon=i_eps)

    regimes_raw: list[str] = []
    last_non_indeterminate: str | None = None
    for growth_label, inflation_label in zip(growth_dir, inflation_dir):
        regime = classify_regime(growth_label, inflation_label)
        if carry_single_flat_regime and regime == "Indeterminate":
            has_single_flat_leg = (growth_label == "Flat") ^ (inflation_label == "Flat")
            if has_single_flat_leg and last_non_indeterminate is not None:
                regime = last_non_indeterminate
        if regime != "Indeterminate":
            last_non_indeterminate = regime
        regimes_raw.append(regime)

    regimes = pd.Series(regimes_raw, index=growth_series.index, name="regime")

    confirmed = (
        apply_confirmation_filter(regimes, n=confirmation_periods)
        if confirmation_periods > 1
        else regimes.copy()
    )
    confirmed.name = "confirmed_regime"

    result = pd.DataFrame(
        {
            "growth_dir": growth_dir,
            "inflation_dir": inflation_dir,
            "regime": regimes,
            "confirmed_regime": confirmed,
        }
    )

    if yield_curve_spread is not None:
        spread = yield_curve_spread.reindex(growth_series.index).ffill()
        result["yield_curve"] = spread.apply(
            lambda x: "Inverted" if pd.notna(x) and x < yield_curve_threshold else "Normal"
        )

    return result
