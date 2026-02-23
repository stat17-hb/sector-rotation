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
