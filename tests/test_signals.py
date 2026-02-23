"""Tests for signal matrix and scoring. (3 tests)"""
from __future__ import annotations

import pytest

from src.signals.matrix import SectorSignal, compute_action
from src.signals.scoring import apply_fx_shock_filter, apply_rsi_alerts


def _make_signal(**kwargs) -> SectorSignal:
    defaults = dict(
        index_code="5044",
        sector_name="KRX 반도체",
        macro_regime="Recovery",
        macro_fit=True,
        rs=1.05,
        rs_ma=1.00,
        rs_strong=True,
        trend_ok=True,
        momentum_strong=True,
        rsi_d=55.0,
        rsi_w=50.0,
        action="Strong Buy",
        alerts=[],
        returns={"1M": 0.03, "3M": 0.07},
        volatility_20d=0.15,
        mdd_3m=-0.08,
        asof_date="2024-01-31",
        is_provisional=False,
    )
    defaults.update(kwargs)
    return SectorSignal(**defaults)


class TestSignals:
    def test_action_matrix_all_four_combinations(self):
        """compute_action covers all four macro_fit × momentum_strong combinations."""
        assert compute_action(True, True) == "Strong Buy"
        assert compute_action(True, False) == "Watch"
        assert compute_action(False, True) == "Hold"
        assert compute_action(False, False) == "Avoid"

    def test_overheat_alert_added_when_rsi_above_70(self):
        """apply_rsi_alerts adds 'Overheat' when RSI >= 70."""
        sig = _make_signal(rsi_d=75.0)
        updated = apply_rsi_alerts(sig, overbought=70, oversold=30)
        assert "Overheat" in updated.alerts

        # Below threshold — no alert
        sig_normal = _make_signal(rsi_d=55.0)
        updated_normal = apply_rsi_alerts(sig_normal, overbought=70, oversold=30)
        assert "Overheat" not in updated_normal.alerts
        assert "Oversold" not in updated_normal.alerts

    def test_fx_shock_downgrades_strong_buy(self):
        """apply_fx_shock_filter downgrades export sector Strong Buy → Watch on FX shock."""
        sig = _make_signal(index_code="5044", action="Strong Buy")
        updated = apply_fx_shock_filter(
            sig,
            fx_change_pct=4.0,  # > 3.0% threshold
            export_sectors=["5044"],
            threshold_pct=3.0,
        )
        assert updated.action == "Watch"
        assert "FX Shock" in updated.alerts

        # Non-export sector — no downgrade
        sig_non_export = _make_signal(index_code="1168", action="Strong Buy")
        no_change = apply_fx_shock_filter(
            sig_non_export,
            fx_change_pct=4.0,
            export_sectors=["5044"],
            threshold_pct=3.0,
        )
        assert no_change.action == "Strong Buy"
