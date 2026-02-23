"""
RSI indicator with TA-Lib → ta library → manual Wilder smoothing fallback chain.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _rsi_talib(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using TA-Lib (C extension, fastest)."""
    import talib  # type: ignore[import]
    return pd.Series(talib.RSI(close.values, timeperiod=period), index=close.index)


def _rsi_ta(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using the 'ta' library."""
    from ta.momentum import RSIIndicator  # type: ignore[import]
    indicator = RSIIndicator(close=close, window=period)
    return indicator.rsi()


def _rsi_manual(close: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing method (pure pandas)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Initial average gain/loss via simple mean for the first period
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Wilder's smoothing for subsequent periods
    for i in range(period, len(close)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.clip(0, 100)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI with fallback chain: TA-Lib → ta → manual Wilder smoothing.

    Args:
        close: Daily close prices with DatetimeIndex.
        period: RSI look-back period (default 14).

    Returns:
        pd.Series of RSI values (0–100), same index as close.
    """
    for attempt, (name, func) in enumerate(
        [("talib", _rsi_talib), ("ta", _rsi_ta), ("manual", _rsi_manual)]
    ):
        try:
            result = func(close, period)
            if attempt > 0:
                logger.info("RSI computed via fallback: %s", name)
            return result
        except ImportError:
            logger.debug("RSI %s not available, trying next fallback", name)
        except Exception as exc:
            logger.warning("RSI %s failed (%s), trying next fallback", name, exc)

    # Should never reach here since manual is always available
    return _rsi_manual(close, period)


def compute_weekly_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute weekly RSI by resampling daily closes to weekly (last observation).

    Args:
        close: Daily close prices with DatetimeIndex.
        period: RSI look-back period (default 14).

    Returns:
        pd.Series of weekly RSI values.
    """
    weekly = close.resample("W").last().dropna()
    return compute_rsi(weekly, period=period)
