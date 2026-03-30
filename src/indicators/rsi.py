"""
RSI indicator with TA-Lib → ta library → manual Wilder smoothing fallback chain.
"""
from __future__ import annotations

import logging

import numpy as np
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
    """Compute RSI using Wilder's smoothing method (pure numpy to avoid pandas CoW issues)."""
    if len(close) <= period:
        return pd.Series(np.nan, index=close.index)

    delta = close.diff().to_numpy(dtype=float)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)

    # Seed: simple mean of first `period` changes (indices 1..period inclusive)
    avg_gain[period] = gain[1 : period + 1].mean()
    avg_loss[period] = loss[1 : period + 1].mean()

    # Wilder's smoothing for subsequent periods
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss == 0, np.nan, avg_gain / avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(np.clip(rsi, 0, 100), index=close.index)


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
