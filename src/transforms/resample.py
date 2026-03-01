"""
Time-series resampling utilities.
"""
from __future__ import annotations

import pandas as pd


def to_monthly_last(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Resample a daily-indexed DataFrame to monthly (last trading day).

    Args:
        df_daily: DataFrame with DatetimeIndex at daily frequency.

    Returns:
        DataFrame resampled to month-end, keeping only last observation per month.
    """
    if not isinstance(df_daily.index, pd.DatetimeIndex):
        raise TypeError("df_daily must have a DatetimeIndex")
    return df_daily.resample("ME").last().dropna(how="all")


def compute_adaptive_epsilon(series: pd.Series, factor: float = 0.5) -> float:
    """Compute epsilon as factor × std of 1-period differences.

    Args:
        series: Numeric time series.
        factor: Multiplier applied to std (default 0.5).

    Returns:
        Adaptive epsilon value. Returns 0.0 if std is NaN or zero.
    """
    std = series.diff().std()
    return float(std * factor) if pd.notna(std) and std > 0 else 0.0


def compute_3ma_direction(
    series: pd.Series, epsilon: float = 0.0
) -> pd.Series:
    """Compute 3-period moving average direction.

    Args:
        series: Numeric time series (monthly preferred).
        epsilon: Minimum absolute change to count as Up/Down (default 0).
                 Values within [-epsilon, +epsilon] → "Flat".

    Returns:
        pd.Series of str: "Up", "Down", or "Flat" aligned with input index.
        Leading NaN values (first 3 periods) are filled with "Flat".
    """
    ma3 = series.rolling(window=3).mean()
    delta = ma3.diff()

    direction = pd.Series("Flat", index=series.index, dtype=object)
    direction[delta > epsilon] = "Up"
    direction[delta < -epsilon] = "Down"
    return direction


def apply_confirmation_filter(regime_series: pd.Series, n: int = 2) -> pd.Series:
    """Confirm a regime only after n consecutive identical periods.

    Args:
        regime_series: Series of regime strings (e.g. "Recovery", "Expansion").
        n: Number of consecutive identical values required for confirmation.

    Returns:
        pd.Series with confirmed regime values. Returns "Indeterminate" until
        any regime has persisted for n periods.
    """
    confirmed = pd.Series("Indeterminate", index=regime_series.index, dtype=object)
    run_val: str | None = None
    run_count = 0
    last_confirmed: str | None = None

    for idx, val in regime_series.items():
        if val == run_val:
            run_count += 1
        else:
            run_val = val
            run_count = 1
        if run_count >= n:
            last_confirmed = run_val
        confirmed[idx] = last_confirmed if last_confirmed is not None else "Indeterminate"

    return confirmed
