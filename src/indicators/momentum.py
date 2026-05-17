"""Momentum indicators: relative strength, return windows, gates, and diagnostics."""
from __future__ import annotations

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


def compute_price_above_sma(close: pd.Series, window: int = 200) -> bool:
    """Return True when the latest close is above the latest SMA(window)."""
    if close.empty:
        return False
    sma_series = compute_sma(close, window)
    if sma_series.empty:
        return False
    latest_close = close.iloc[-1]
    latest_sma = sma_series.iloc[-1]
    if pd.isna(latest_close) or pd.isna(latest_sma):
        return False
    return bool(float(latest_close) > float(latest_sma))


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


def compute_return_excluding_recent(
    close: pd.Series,
    *,
    lookback_days: int,
    skip_recent_days: int = 0,
) -> float:
    """Return trailing simple return with an optional recent skip window.

    The calculation uses:
    - with ``skip_recent_days == 0``: latest price vs ``lookback_days`` observations back
    - with ``skip_recent_days > 0``: the historical ex-recent window used by legacy tests
    """
    if close.empty:
        return float("nan")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if skip_recent_days < 0:
        raise ValueError("skip_recent_days must be non-negative")

    required = lookback_days + skip_recent_days
    if skip_recent_days == 0:
        required += 1
    if len(close) < required:
        return float("nan")

    end_idx = -skip_recent_days if skip_recent_days > 0 else -1
    start_idx = -(lookback_days + skip_recent_days)
    if skip_recent_days == 0:
        start_idx -= 1
    start_price = float(close.iloc[start_idx])
    end_price = float(close.iloc[end_idx])
    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
        return float("nan")
    return float(end_price / start_price - 1.0)


def compute_relative_return_excluding_recent(
    sector_close: pd.Series,
    benchmark_close: pd.Series,
    *,
    lookback_days: int,
    skip_recent_days: int = 0,
) -> float:
    """Return benchmark-relative trailing simple return with an optional skip window."""
    aligned_sector, aligned_bench = sector_close.align(benchmark_close, join="inner")
    if aligned_sector.empty or aligned_bench.empty:
        return float("nan")
    sector_return = compute_return_excluding_recent(
        aligned_sector,
        lookback_days=lookback_days,
        skip_recent_days=skip_recent_days,
    )
    bench_return = compute_return_excluding_recent(
        aligned_bench,
        lookback_days=lookback_days,
        skip_recent_days=skip_recent_days,
    )
    if pd.isna(sector_return) or pd.isna(bench_return):
        return float("nan")
    return float(sector_return - bench_return)


def compute_percentile_rank(
    values: pd.Series | dict[str, float],
    *,
    ascending: bool = True,
) -> pd.Series:
    """Return percentile ranks for valid numeric values."""
    series = values if isinstance(values, pd.Series) else pd.Series(values, dtype="float64")
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(index=numeric.index, dtype="float64")
    ranked = valid.rank(pct=True, ascending=ascending, method="average")
    return ranked.reindex(numeric.index)


def compute_descending_rank(values: pd.Series | dict[str, float]) -> pd.Series:
    """Return integer descending ranks where 1 is strongest."""
    series = values if isinstance(values, pd.Series) else pd.Series(values, dtype="float64")
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(index=numeric.index, dtype="float64")
    ranked = valid.rank(ascending=False, method="min").astype("int64")
    return ranked.reindex(numeric.index)


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
