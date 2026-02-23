"""Tests for macro regime classification. (3 tests)"""
from __future__ import annotations

import pandas as pd
import pytest

from src.macro.regime import classify_regime, compute_regime_history, get_regime_sectors
from src.transforms.resample import compute_3ma_direction


class TestClassifyRegime:
    def test_classify_regime_all_four_phases(self):
        """All four (growth, inflation) combinations produce correct regime."""
        assert classify_regime("Up", "Down") == "Recovery"
        assert classify_regime("Up", "Up") == "Expansion"
        assert classify_regime("Down", "Up") == "Slowdown"
        assert classify_regime("Down", "Down") == "Contraction"

    def test_classify_regime_flat_returns_indeterminate(self):
        """Any Flat direction returns Indeterminate."""
        assert classify_regime("Flat", "Up") == "Indeterminate"
        assert classify_regime("Up", "Flat") == "Indeterminate"
        assert classify_regime("Flat", "Flat") == "Indeterminate"

    def test_compute_3ma_direction(self):
        """compute_3ma_direction returns Up/Down/Flat based on 3MA delta."""
        # Strictly increasing series — direction should be Up
        series = pd.Series(
            [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            index=pd.date_range("2024-01", periods=7, freq="ME"),
        )
        direction = compute_3ma_direction(series, epsilon=0.0)
        # First 3 values are Flat (insufficient window), then Up
        assert "Up" in direction.values
        # Decreasing series — direction should be Down
        series_down = pd.Series(
            [106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0],
            index=pd.date_range("2024-01", periods=7, freq="ME"),
        )
        direction_down = compute_3ma_direction(series_down, epsilon=0.0)
        assert "Down" in direction_down.values
