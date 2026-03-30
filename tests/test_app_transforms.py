"""Unit tests for pure data-transform functions in app.py.

These functions contain core analysis logic (cycle segments, price pivoting,
monthly return views) but had zero test coverage.  Importing app.py is safe
here because all Streamlit side-effects are skipped when there is no
ScriptRunContext.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

# app.py has Streamlit session-state setup at module level; that's harmless in
# test mode (Streamlit prints warnings but does not raise).
import app as app_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_long_prices(codes: list[str], n_days: int = 60) -> pd.DataFrame:
    """Return a long-format price DataFrame similar to load_sector_prices output."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for code in codes:
        prices = 1000.0 * np.cumprod(1 + rng.normal(0.001, 0.01, size=n_days))
        for ts, p in zip(idx, prices):
            rows.append({"index_code": code, "index_name": f"Sector {code}", "close": p})
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex([idx[i % n_days] for i in range(len(rows))])
    # Reconstruct proper DatetimeIndex: one row per (date, code)
    df2 = pd.DataFrame(rows)
    df2["trade_date"] = list(idx) * len(codes)
    df2 = df2.set_index("trade_date")
    return df2


def _make_macro_result(regimes: list[str], months_per_regime: int = 1) -> pd.DataFrame:
    """Return a monthly macro DataFrame with the given regime sequence.

    `regimes` is repeated as a flat list; each entry lasts `months_per_regime` months.
    """
    regime_seq = []
    for r in regimes:
        regime_seq.extend([r] * months_per_regime)
    n = len(regime_seq)
    idx = pd.bdate_range("2022-01-01", periods=n, freq="ME")
    return pd.DataFrame(
        {
            "growth_dir": ["Up"] * n,
            "inflation_dir": ["Down"] * n,
            "regime": regime_seq,
            "confirmed_regime": regime_seq,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# _build_sector_name_map
# ---------------------------------------------------------------------------

class TestBuildSectorNameMap:
    def test_includes_benchmark(self):
        result = app_module._build_sector_name_map(
            signals=[],
            sector_prices=pd.DataFrame(),
            benchmark_code="1001",
        )
        assert "1001" in result
        assert result["1001"] == "KOSPI"

    def test_picks_up_signal_names(self):
        from src.signals.matrix import SectorSignal
        sig = SectorSignal(
            index_code="5044",
            sector_name="KRX 반도체",
            macro_regime="Recovery",
            macro_fit=True,
            rs=1.0, rs_ma=1.0, rs_strong=True, trend_ok=True, momentum_strong=True,
            rsi_d=55.0, rsi_w=50.0, action="Strong Buy", alerts=[],
            returns={}, volatility_20d=0.1, mdd_3m=-0.05,
            asof_date="2024-01-31", is_provisional=False,
        )
        result = app_module._build_sector_name_map(
            signals=[sig],
            sector_prices=pd.DataFrame(),
            benchmark_code="1001",
        )
        assert result["5044"] == "KRX 반도체"


# ---------------------------------------------------------------------------
# _build_prices_wide
# ---------------------------------------------------------------------------

class TestBuildPricesWide:
    def test_empty_input_returns_empty(self):
        result = app_module._build_prices_wide(
            sector_prices=pd.DataFrame(), sector_name_map={}
        )
        assert result.empty

    def test_pivot_creates_one_column_per_sector(self):
        prices = _make_long_prices(["A", "B"])
        name_map = {"A": "Sector A", "B": "Sector B"}
        result = app_module._build_prices_wide(
            sector_prices=prices, sector_name_map=name_map
        )
        assert "Sector A" in result.columns
        assert "Sector B" in result.columns
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_pivot_has_no_duplicate_dates(self):
        prices = _make_long_prices(["A"])
        name_map = {"A": "Sector A"}
        result = app_module._build_prices_wide(
            sector_prices=prices, sector_name_map=name_map
        )
        assert not result.index.duplicated().any()

    def test_ffill_fills_gaps(self):
        """Forward-fill should eliminate NaN gaps after pivot."""
        prices = _make_long_prices(["A", "B"])
        name_map = {"A": "A", "B": "B"}
        result = app_module._build_prices_wide(
            sector_prices=prices, sector_name_map=name_map
        )
        # After ffill, NaN count should equal rows at the very start (before first value)
        assert result.iloc[1:].isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# _build_monthly_sector_returns
# ---------------------------------------------------------------------------

class TestBuildMonthlySectorReturns:
    def test_returns_monthly_index(self):
        prices = _make_long_prices(["A"])
        name_map = {"A": "Sector A"}
        wide = app_module._build_prices_wide(sector_prices=prices, sector_name_map=name_map)
        closes, returns = app_module._build_monthly_sector_returns(
            prices_wide=wide, sector_columns=["Sector A"]
        )
        assert closes.index.freqstr in {"ME", "M"}

    def test_returns_are_pct(self):
        """Monthly returns should be in percent (not fraction)."""
        prices = _make_long_prices(["A"], n_days=90)
        name_map = {"A": "Sector A"}
        wide = app_module._build_prices_wide(sector_prices=prices, sector_name_map=name_map)
        _, returns = app_module._build_monthly_sector_returns(
            prices_wide=wide, sector_columns=["Sector A"]
        )
        non_nan = returns["Sector A"].dropna()
        # Percentage returns should be in a reasonable range (-100% to +100%)
        assert (non_nan.abs() < 100).all()


# ---------------------------------------------------------------------------
# _build_cycle_segments
# ---------------------------------------------------------------------------

class TestBuildCycleSegments:
    def test_empty_macro_returns_empty(self):
        segs, phase_by_month = app_module._build_cycle_segments(
            macro_result=pd.DataFrame(),
            monthly_close=pd.DataFrame(),
        )
        assert segs == []
        assert phase_by_month.empty

    def test_single_regime_produces_two_segments(self):
        """12 months of Recovery → RECOVERY_EARLY + RECOVERY_LATE."""
        macro = _make_macro_result(["Recovery"], months_per_regime=12)
        segs, _ = app_module._build_cycle_segments(
            macro_result=macro, monthly_close=pd.DataFrame()
        )
        phase_keys = [s["phase_key"] for s in segs]
        assert "RECOVERY_EARLY" in phase_keys
        assert "RECOVERY_LATE" in phase_keys

    def test_four_regime_cycle_produces_eight_segments(self):
        """One full cycle (3 months each) → 8 segments (2 per regime)."""
        macro = _make_macro_result(
            ["Recovery", "Expansion", "Slowdown", "Contraction"], months_per_regime=3
        )
        segs, _ = app_module._build_cycle_segments(
            macro_result=macro, monthly_close=pd.DataFrame()
        )
        assert len(segs) == 8

    def test_last_segment_is_current(self):
        """Final regime's last phase segment should have is_current=True.

        When each regime appears exactly once, only one segment matches
        the last phase_key, so current_count == 1.
        """
        macro = _make_macro_result(["Recovery", "Expansion"], months_per_regime=4)
        segs, _ = app_module._build_cycle_segments(
            macro_result=macro, monthly_close=pd.DataFrame()
        )
        assert segs[-1]["is_current"] is True
        current_count = sum(1 for s in segs if s.get("is_current"))
        assert current_count == 1

    def test_segment_start_le_end(self):
        """Every segment must have start <= end (no inverted ranges)."""
        macro = _make_macro_result(
            ["Recovery", "Expansion", "Slowdown"], months_per_regime=3
        )
        segs, _ = app_module._build_cycle_segments(
            macro_result=macro, monthly_close=pd.DataFrame()
        )
        for seg in segs:
            assert pd.Timestamp(seg["start"]) <= pd.Timestamp(seg["end"]), (
                f"Inverted segment: {seg}"
            )

    def test_phase_by_month_covers_all_input_months(self):
        """phase_by_month should have an entry for every input month."""
        macro = _make_macro_result(["Recovery"], months_per_regime=6)
        _, phase_by_month = app_module._build_cycle_segments(
            macro_result=macro, monthly_close=pd.DataFrame()
        )
        assert phase_by_month.notna().sum() == 6

    def test_single_month_regime_does_not_crash(self):
        """A single-month regime run (edge case) should not raise."""
        macro = _make_macro_result(["Recovery", "Expansion", "Recovery"])
        segs, _ = app_module._build_cycle_segments(
            macro_result=macro, monthly_close=pd.DataFrame()
        )
        assert len(segs) >= 2


# ---------------------------------------------------------------------------
# _filter_monthly_frame_for_analysis
# ---------------------------------------------------------------------------

class TestFilterMonthlyFrameForAnalysis:
    def _make_monthly_frame(self) -> pd.DataFrame:
        idx = pd.bdate_range("2022-01-01", periods=12, freq="ME")
        return pd.DataFrame({"A": range(12)}, index=idx)

    def test_all_phase_returns_full_frame_in_range(self):
        frame = self._make_monthly_frame()
        result = app_module._filter_monthly_frame_for_analysis(
            monthly_frame=frame,
            start_date=date(2022, 1, 1),
            end_date=date(2022, 12, 31),
            selected_cycle_phase="ALL",
            phase_by_month=pd.Series(dtype=object),
        )
        assert len(result) == len(frame)

    def test_date_window_trims_rows(self):
        frame = self._make_monthly_frame()
        result = app_module._filter_monthly_frame_for_analysis(
            monthly_frame=frame,
            start_date=date(2022, 4, 1),
            end_date=date(2022, 9, 30),
            selected_cycle_phase="ALL",
            phase_by_month=pd.Series(dtype=object),
        )
        assert len(result) < len(frame)

    def test_empty_frame_returns_empty(self):
        result = app_module._filter_monthly_frame_for_analysis(
            monthly_frame=pd.DataFrame(),
            start_date=date(2022, 1, 1),
            end_date=date(2022, 12, 31),
            selected_cycle_phase="ALL",
            phase_by_month=pd.Series(dtype=object),
        )
        assert result.empty
