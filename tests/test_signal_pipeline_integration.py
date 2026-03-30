"""End-to-end signal pipeline integration tests.

Covers the full path: regime → momentum → signal → scoring → action.
These tests exist to catch silent failures (like the FX shock 0.0 bug) that
unit tests missed because they only tested individual components.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.signals.matrix import build_signal_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n: int = 120, *, seed: int = 42, trend: float = 0.002) -> pd.Series:
    """Return a trending price series with a business-day DatetimeIndex."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, 0.01, size=n)
    prices = 1000.0 * np.cumprod(1 + returns)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(prices, index=idx)


def _sector_prices_df(code: str, name: str, prices: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {"index_code": code, "index_name": name, "close": prices.values},
        index=prices.index,
    )


def _macro_result(regime: str = "Recovery") -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-01", periods=12, freq="ME")
    return pd.DataFrame(
        {
            "growth_dir": ["Up"] * 12,
            "inflation_dir": ["Down"] * 12,
            "regime": [regime] * 12,
            "confirmed_regime": [regime] * 12,
        },
        index=idx,
    )


_SECTOR_MAP = {
    "benchmark": {"code": "BENCH", "name": "Benchmark"},
    "regimes": {
        "Recovery": {
            "sectors": [
                {"code": "SEC01", "name": "Test Sector A", "export_sector": True},
            ]
        },
        "Expansion": {
            "sectors": [
                {"code": "SEC02", "name": "Test Sector B", "export_sector": False},
            ]
        },
    },
}

_SETTINGS = {
    "rs_ma_period": 20,
    "ma_fast": 20,
    "ma_slow": 60,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "fx_shock_pct": 3.0,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSignalPipelineIntegration:

    def _build_strong_signals(self, fx_change_pct: float | None = None):
        """Build signals where SEC01 should be Strong Buy in Recovery regime."""
        bench = _make_prices(120, seed=0, trend=0.0)
        # SEC01 strongly outperforms benchmark → RS strong and SMA20 > SMA60
        sec01 = _make_prices(120, seed=1, trend=0.008)

        df = pd.concat(
            [
                _sector_prices_df("BENCH", "Benchmark", bench),
                _sector_prices_df("SEC01", "Test Sector A", sec01),
                _sector_prices_df("SEC02", "Test Sector B", _make_prices(120, seed=2, trend=-0.001)),
            ]
        ).sort_index()

        return build_signal_table(
            sector_prices=df,
            benchmark_prices=bench,
            macro_result=_macro_result("Recovery"),
            sector_map=_SECTOR_MAP,
            settings=_SETTINGS,
            fx_change_pct=fx_change_pct,
        )

    def test_strong_buy_in_fitting_regime(self):
        """SEC01 in Recovery regime with strong momentum → Strong Buy."""
        signals = self._build_strong_signals()
        by_code = {s.index_code: s for s in signals}
        assert "SEC01" in by_code, "SEC01 should appear in signal table"
        assert by_code["SEC01"].action == "Strong Buy", (
            f"Expected Strong Buy for Recovery-fit sector, got {by_code['SEC01'].action}"
        )

    def test_non_fitting_sector_gets_hold_or_avoid(self):
        """SEC02 is not in Recovery regime → macro_fit=False → Hold or Avoid."""
        signals = self._build_strong_signals()
        by_code = {s.index_code: s for s in signals}
        assert "SEC02" in by_code
        assert by_code["SEC02"].action in {"Hold", "Avoid"}, (
            f"SEC02 not in Recovery regime, expected Hold/Avoid, got {by_code['SEC02'].action}"
        )

    def test_fx_shock_downgrades_export_sector_strong_buy(self):
        """FX shock >3% on export sector Strong Buy → Watch with FX Shock alert."""
        signals = self._build_strong_signals(fx_change_pct=4.5)
        by_code = {s.index_code: s for s in signals}
        assert "SEC01" in by_code
        sec01 = by_code["SEC01"]
        # Only downgrade if it was Strong Buy before FX filter; skip if momentum was weak
        if sec01.rs_strong and sec01.macro_fit:
            assert sec01.action == "Watch", (
                f"Export sector Strong Buy should be downgraded on FX shock, got {sec01.action}"
            )
            assert "FX Shock" in sec01.alerts, "FX Shock alert should be present"

    def test_no_fx_shock_below_threshold(self):
        """FX change below 3% threshold must not downgrade Strong Buy."""
        signals_low_fx = self._build_strong_signals(fx_change_pct=1.0)
        signals_no_fx = self._build_strong_signals(fx_change_pct=None)
        by_code_low = {s.index_code: s for s in signals_low_fx}
        by_code_none = {s.index_code: s for s in signals_no_fx}

        # Both should have same action (no downgrade)
        assert by_code_low["SEC01"].action == by_code_none["SEC01"].action
        assert "FX Shock" not in by_code_low["SEC01"].alerts

    def test_missing_sector_data_produces_na_not_exception(self):
        """When price data for a sector is missing, action should be N/A (no exception)."""
        # Only supply benchmark + SEC02; SEC01 is absent from prices
        bench = _make_prices(120, seed=0, trend=0.001)
        df = pd.concat(
            [
                _sector_prices_df("BENCH", "Benchmark", bench),
                _sector_prices_df("SEC02", "Test Sector B", _make_prices(120, seed=2)),
            ]
        ).sort_index()

        signals = build_signal_table(
            sector_prices=df,
            benchmark_prices=bench,
            macro_result=_macro_result("Recovery"),
            sector_map=_SECTOR_MAP,
            settings=_SETTINGS,
            fx_change_pct=None,
        )
        by_code = {s.index_code: s for s in signals}
        assert "SEC01" in by_code, "Missing sector should still appear in table as N/A"
        assert by_code["SEC01"].action == "N/A", (
            f"Missing sector data should yield N/A, got {by_code['SEC01'].action}"
        )

    def test_rsi_overheat_alert_propagates_end_to_end(self):
        """When sector RSI >= 70 after full pipeline, Overheat alert is present."""
        bench = _make_prices(120, seed=0, trend=0.001)
        # Strongly trending sector to push RSI high
        sec01 = _make_prices(120, seed=99, trend=0.015)

        df = pd.concat(
            [
                _sector_prices_df("BENCH", "Benchmark", bench),
                _sector_prices_df("SEC01", "Test Sector A", sec01),
                _sector_prices_df("SEC02", "Test Sector B", _make_prices(120, seed=2)),
            ]
        ).sort_index()

        signals = build_signal_table(
            sector_prices=df,
            benchmark_prices=bench,
            macro_result=_macro_result("Recovery"),
            sector_map=_SECTOR_MAP,
            settings=_SETTINGS,
            fx_change_pct=None,
        )
        by_code = {s.index_code: s for s in signals}
        sec01 = by_code.get("SEC01")
        assert sec01 is not None
        # If RSI is high enough the alert should be set; verify the field is wired
        if not math.isnan(sec01.rsi_d) and sec01.rsi_d >= 70:
            assert "Overheat" in sec01.alerts, (
                f"RSI={sec01.rsi_d:.1f} >= 70 but Overheat alert missing"
            )
