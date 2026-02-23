"""Tests for momentum indicators. (4 tests)"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.momentum import (
    compute_rs,
    compute_rs_ma,
    compute_sma,
    compute_volatility,
    is_rs_strong,
    is_trend_positive,
)
from src.indicators.rsi import compute_rsi, compute_weekly_rsi


def _make_price_series(n: int = 100, start: float = 1000.0, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.01, n)
    prices = start * np.cumprod(1 + returns)
    return pd.Series(prices, index=pd.date_range("2023-01-01", periods=n, freq="B"))


class TestMomentum:
    def test_rs_above_ma_is_strong(self):
        """is_rs_strong returns True when RS > RS_MA."""
        sector = _make_price_series(100, start=1200.0, seed=1)
        bench = _make_price_series(100, start=1000.0, seed=2)
        rs = compute_rs(sector, bench)
        rs_ma = compute_rs_ma(rs, period=20)
        # Use last value
        rs_last = rs.dropna().iloc[-1]
        rs_ma_last = rs_ma.dropna().iloc[-1]
        # Test both directions explicitly
        assert is_rs_strong(rs_last + 0.1, rs_last) is True
        assert is_rs_strong(rs_last - 0.1, rs_last) is False

    def test_trend_positive_sma20_gt_sma60(self):
        """is_trend_positive returns True when SMA20 > SMA60."""
        # Uptrending series: SMA20 should be above SMA60
        n = 120
        prices = pd.Series(
            [1000.0 + i * 2 for i in range(n)],
            index=pd.date_range("2023-01-01", periods=n, freq="B"),
        )
        assert is_trend_positive(prices, fast=20, slow=60) is True

        # Downtrending series: SMA20 should be below SMA60
        prices_down = pd.Series(
            [2000.0 - i * 2 for i in range(n)],
            index=pd.date_range("2023-01-01", periods=n, freq="B"),
        )
        assert is_trend_positive(prices_down, fast=20, slow=60) is False

    def test_rsi_bounds_0_100(self):
        """RSI values must be within [0, 100]."""
        close = _make_price_series(200, seed=7)
        rsi = compute_rsi(close, period=14)
        rsi_valid = rsi.dropna()
        assert (rsi_valid >= 0).all(), "RSI values must be >= 0"
        assert (rsi_valid <= 100).all(), "RSI values must be <= 100"

    def test_weekly_rsi_resampling(self):
        """Weekly RSI has fewer observations than daily RSI."""
        close = _make_price_series(200, seed=9)
        daily_rsi = compute_rsi(close, period=14)
        weekly_rsi = compute_weekly_rsi(close, period=14)
        assert len(weekly_rsi.dropna()) < len(daily_rsi.dropna()), (
            "Weekly RSI should have fewer non-null observations than daily RSI"
        )
