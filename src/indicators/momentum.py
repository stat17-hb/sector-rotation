"""
Momentum indicators: Relative Strength, SMA trend, period returns, volatility, MDD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rs(sector_close: pd.Series, benchmark_close: pd.Series) -> pd.Series:
    """Compute Relative Strength of sector vs benchmark.

    RS = sector_close / benchmark_close (ratio, not percentage).
    Both series must be aligned to the same DatetimeIndex.

    Args:
        sector_close: Sector close prices.
        benchmark_close: Benchmark close prices.

    Returns:
        pd.Series of RS ratio values.
    """
    aligned_sector, aligned_bench = sector_close.align(benchmark_close, join="inner")
    return aligned_sector / aligned_bench


def compute_rs_ma(rs_series: pd.Series, period: int = 20) -> pd.Series:
    """Compute simple moving average of RS series.

    Args:
        rs_series: RS ratio series.
        period: SMA window (default 20).

    Returns:
        pd.Series of RS SMA values.
    """
    return rs_series.rolling(window=period).mean()


def is_rs_strong(rs: float | pd.Series, rs_ma: float | pd.Series) -> bool | pd.Series:
    """Return True if RS is above its moving average (relative momentum positive).

    Args:
        rs: Current RS value or series.
        rs_ma: RS moving average value or series.

    Returns:
        bool or pd.Series of bool.
    """
    result = rs > rs_ma
    if isinstance(result, pd.Series):
        return result
    return bool(result)


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Compute simple moving average.

    Args:
        series: Input price series.
        window: Rolling window period.

    Returns:
        pd.Series of SMA values.
    """
    return series.rolling(window=window).mean()


def is_trend_positive(
    close: pd.Series, fast: int = 20, slow: int = 60
) -> bool:
    """Return True if fast SMA > slow SMA (uptrend) for the latest observation.

    Args:
        close: Close price series.
        fast: Fast SMA window (default 20).
        slow: Slow SMA window (default 60).

    Returns:
        bool — True if SMA(fast) > SMA(slow) on the last observation.
    """
    sma_fast = compute_sma(close, fast)
    sma_slow = compute_sma(close, slow)
    if sma_fast.empty or sma_slow.empty:
        return False
    last_fast = sma_fast.iloc[-1]
    last_slow = sma_slow.iloc[-1]
    if pd.isna(last_fast) or pd.isna(last_slow):
        return False
    return bool(last_fast > last_slow)


def compute_period_returns(close: pd.Series) -> dict[str, float]:
    """Compute standard period returns from the last available price.

    Args:
        close: Daily close prices with DatetimeIndex.

    Returns:
        dict with keys "1W", "1M", "3M", "6M", "12M" and float values (decimal).
        Returns NaN for periods with insufficient history.
    """
    if close.empty:
        return {"1W": float("nan"), "1M": float("nan"), "3M": float("nan"),
                "6M": float("nan"), "12M": float("nan")}

    last_price = close.iloc[-1]
    last_date = close.index[-1]

    periods = {
        "1W": 5,     # ~5 trading days
        "1M": 21,    # ~21 trading days
        "3M": 63,
        "6M": 126,
        "12M": 252,
    }

    result = {}
    for label, n_days in periods.items():
        if len(close) > n_days:
            past_price = close.iloc[-n_days - 1]
            result[label] = float((last_price - past_price) / past_price)
        else:
            result[label] = float("nan")

    return result


def compute_volatility(close: pd.Series, window: int = 20) -> float:
    """Compute annualized volatility from daily returns.

    Args:
        close: Daily close prices.
        window: Rolling window for standard deviation.

    Returns:
        Annualized volatility as a float (e.g. 0.20 for 20%).
    """
    if len(close) < 2:
        return float("nan")
    daily_returns = close.pct_change().dropna()
    if len(daily_returns) < window:
        return float("nan")
    rolling_std = daily_returns.rolling(window=window).std().iloc[-1]
    return float(rolling_std * (252 ** 0.5))


def compute_mdd(close: pd.Series, window: int = 63) -> float:
    """Compute Maximum Drawdown over the trailing window.

    Args:
        close: Daily close prices.
        window: Look-back window in trading days (default 63 ≈ 3 months).

    Returns:
        MDD as a negative float (e.g. -0.15 for -15%). Returns NaN if insufficient data.
    """
    if len(close) < 2:
        return float("nan")
    trailing = close.iloc[-window:] if len(close) >= window else close
    rolling_max = trailing.cummax()
    drawdown = (trailing - rolling_max) / rolling_max
    return float(drawdown.min())
