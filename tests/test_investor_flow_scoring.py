from __future__ import annotations

import pandas as pd

from src.signals.flow import apply_flow_overlay, summarize_sector_investor_flow
from src.signals.matrix import SectorSignal


def _signal(action: str = "Watch") -> SectorSignal:
    return SectorSignal(
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
        action=action,
        alerts=[],
        returns={"1M": 0.03, "3M": 0.07},
        volatility_20d=0.15,
        mdd_3m=-0.08,
        asof_date="2026-04-10",
        is_provisional=False,
    )


def _flow_frame(
    *,
    foreign: float,
    institutional: float,
    retail: float,
) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=65)
    rows: list[dict[str, object]] = []
    for idx, trade_date in enumerate(dates):
        for investor_type, baseline, last_boost in (
            ("외국인", foreign, foreign * 0.5),
            ("기관합계", institutional, institutional * 0.5),
            ("개인", retail, retail * 0.5),
        ):
            ratio = baseline
            if idx >= 45:
                ratio += last_boost
            rows.append(
                {
                    "trade_date": trade_date,
                    "sector_code": "5044",
                    "sector_name": "KRX 반도체",
                    "investor_type": investor_type,
                    "buy_amount": 1000,
                    "sell_amount": 900,
                    "net_buy_amount": int(ratio * 10000),
                    "net_flow_ratio": ratio,
                }
            )
    frame = pd.DataFrame(rows).set_index("trade_date")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def test_supportive_flow_upgrades_watch_once():
    signals, summary_map = apply_flow_overlay(
        [_signal("Watch")],
        flow_frame=_flow_frame(foreign=0.15, institutional=0.10, retail=-0.08),
        flow_profile="foreign_lead",
        enabled=True,
    )

    updated = signals[0]
    assert updated.base_action == "Watch"
    assert updated.action == "Strong Buy"
    assert updated.flow_adjustment == "upgrade"
    assert updated.flow_state == "supportive"
    assert "5044" in summary_map


def test_adverse_flow_downgrades_strong_buy_once_and_keeps_na():
    adverse_frame = _flow_frame(foreign=-0.14, institutional=-0.12, retail=0.09)
    signals, _ = apply_flow_overlay(
        [_signal("Strong Buy"), _signal("N/A")],
        flow_frame=adverse_frame,
        flow_profile="foreign_lead",
        enabled=True,
    )

    assert signals[0].base_action == "Strong Buy"
    assert signals[0].action == "Watch"
    assert signals[0].flow_adjustment == "downgrade"
    assert signals[1].action == "N/A"
    assert signals[1].flow_adjustment == "experimental unavailable"


def test_profile_switch_changes_summary_without_mutating_input():
    frame = _flow_frame(foreign=0.14, institutional=-0.08, retail=-0.10)
    foreign_led = summarize_sector_investor_flow(frame, flow_profile="foreign_lead")
    institutional = summarize_sector_investor_flow(frame, flow_profile="institutional_confirmation")

    assert foreign_led["5044"].flow_profile == "foreign_lead"
    assert institutional["5044"].flow_profile == "institutional_confirmation"
    assert foreign_led["5044"].flow_score != institutional["5044"].flow_score
    assert set(frame["investor_type"].unique()) == {"외국인", "기관합계", "개인"}


def test_disabled_flow_overlay_keeps_base_action_even_when_frame_is_supportive():
    signals, summary_map = apply_flow_overlay(
        [_signal("Watch")],
        flow_frame=_flow_frame(foreign=0.15, institutional=0.10, retail=-0.08),
        flow_profile="foreign_lead",
        enabled=False,
    )

    updated = signals[0]
    assert updated.action == "Watch"
    assert updated.base_action == "Watch"
    assert updated.flow_adjustment == "experimental unavailable"
    assert summary_map == {}
