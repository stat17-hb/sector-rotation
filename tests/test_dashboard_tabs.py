from __future__ import annotations

import pandas as pd

from src.dashboard import tabs


def test_render_decision_first_sections_orders_main_canvas(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(
        tabs,
        "render_decision_hero",
        lambda **kwargs: calls.append("hero"),
    )
    monkeypatch.setattr(
        tabs,
        "render_status_card_row",
        lambda **kwargs: calls.append("status"),
    )
    monkeypatch.setattr(
        tabs,
        "render_investor_flow_summary",
        lambda **kwargs: calls.append("flow"),
    )
    monkeypatch.setattr(
        tabs,
        "render_investor_decision_boards",
        lambda **kwargs: calls.append("boards") or ["Sector A"],
    )
    monkeypatch.setattr(
        tabs,
        "render_top_bar_filters",
        lambda **kwargs: calls.append("filters") or ("Watch", True, "held", True),
    )
    monkeypatch.setattr(
        tabs,
        "render_analysis_canvas",
        lambda **kwargs: calls.append("analysis"),
    )

    result = tabs.render_decision_first_sections(
        current_regime="Recovery",
        regime_is_confirmed=True,
        growth_val=100.0,
        inflation_val=2.0,
        fx_change=1.0,
        fx_label="FX move",
        is_provisional=False,
        theme_mode="light",
        price_status="LIVE",
        macro_status="LIVE",
        investor_flow_status="CACHED",
        investor_flow_fresh=True,
        investor_flow_profile="foreign_lead",
        yield_curve_status="Normal",
        signals=[],
        held_sector_options=["Sector A", "Sector B"],
        action_options=["ALL", "Watch"],
        is_mobile_client=False,
        analysis_canvas_kwargs={"heatmap_return_display": None},
    )

    assert calls == ["hero", "status", "flow", "boards", "filters", "analysis"]
    assert result == (["Sector A"], "Watch", True, "held", True)


def test_render_dashboard_tabs_hides_flow_tab_for_us(monkeypatch):
    tab_labels_seen: list[list[str]] = []
    flow_calls: list[str] = []

    monkeypatch.setattr(
        tabs.st,
        "tabs",
        lambda labels: tab_labels_seen.append(list(labels)) or [object() for _ in labels],
    )
    monkeypatch.setattr(tabs, "render_summary_tab", lambda **kwargs: None)
    monkeypatch.setattr(tabs, "render_charts_tab", lambda **kwargs: None)
    monkeypatch.setattr(tabs, "render_all_signals_tab", lambda **kwargs: None)
    monkeypatch.setattr(tabs, "render_screening_tab", lambda **kwargs: None)
    monkeypatch.setattr(tabs, "render_investor_flow_tab", lambda **kwargs: flow_calls.append("flow"))

    common_kwargs = dict(
        current_regime="Recovery",
        regime_is_confirmed=True,
        growth_val=100.0,
        inflation_val=2.0,
        fx_change=1.0,
        fx_label="FX move",
        is_provisional=False,
        theme_mode="dark",
        price_status="LIVE",
        macro_status="LIVE",
        yield_curve_status=None,
        top_pick_signals=[],
        signals_filtered=[],
        signals=[],
        filter_action_global="__ALL__",
        filter_regime_only_global=False,
        held_sectors=[],
        position_mode="all",
        show_alerted_only=False,
        settings={},
        is_mobile_client=False,
        investor_flow_status="CACHED",
        investor_flow_fresh=True,
        investor_flow_profile="foreign_lead",
        investor_flow_frame=pd.DataFrame(),
        sector_map={},
        ui_locale="ko",
    )

    tabs.render_dashboard_tabs(market_id="KR", **common_kwargs)
    assert "투자자 수급" in tab_labels_seen[0]
    assert flow_calls == ["flow"]

    tab_labels_seen.clear()
    flow_calls.clear()
    tabs.render_dashboard_tabs(market_id="US", **common_kwargs)
    assert "투자자 수급" not in tab_labels_seen[0]
    assert flow_calls == []
