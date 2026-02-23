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
) -> pd.DataFrame:
    """Compute regime classification over time.

    Args:
        growth_series: Monthly leading indicator values (e.g. 경기선행지수순환변동치).
        inflation_series: Monthly inflation values (e.g. CPI YoY).
        epsilon: Minimum absolute change to count as directional (default 0).

    Returns:
        DataFrame indexed same as inputs with columns:
        [growth_dir, inflation_dir, regime].
    """
    from src.transforms.resample import compute_3ma_direction

    growth_dir = compute_3ma_direction(growth_series, epsilon=epsilon)
    inflation_dir = compute_3ma_direction(inflation_series, epsilon=epsilon)

    regimes = pd.Series(
        [classify_regime(g, i) for g, i in zip(growth_dir, inflation_dir)],
        index=growth_series.index,
        name="regime",
    )

    return pd.DataFrame(
        {
            "growth_dir": growth_dir,
            "inflation_dir": inflation_dir,
            "regime": regimes,
        }
    )
