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
        market_id="KR",
    )

    assert calls == ["hero", "status", "flow", "boards", "filters", "analysis"]
    assert result == (["Sector A"], "Watch", True, "held", True)


def test_render_decision_first_sections_skips_kr_flow_summary_for_us(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(tabs, "render_decision_hero", lambda **kwargs: calls.append("hero"))
    monkeypatch.setattr(tabs, "render_status_card_row", lambda **kwargs: calls.append("status"))
    monkeypatch.setattr(tabs, "render_investor_flow_summary", lambda **kwargs: calls.append("flow"))
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
    monkeypatch.setattr(tabs, "render_analysis_canvas", lambda **kwargs: calls.append("analysis"))

    tabs.render_decision_first_sections(
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
        investor_flow_status="LIVE",
        investor_flow_fresh=True,
        investor_flow_profile="foreign_lead",
        yield_curve_status="Normal",
        signals=[],
        held_sector_options=["Sector A", "Sector B"],
        action_options=["ALL", "Watch"],
        is_mobile_client=False,
        analysis_canvas_kwargs={"heatmap_return_display": None},
        market_id="US",
    )

    assert calls == ["hero", "status", "boards", "filters", "analysis"]


def test_render_dashboard_tabs_uses_market_specific_flow_tab(monkeypatch):
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
    monkeypatch.setattr(tabs, "render_monitoring_tab", lambda **kwargs: None)

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
    assert "US Flow Proxies" in tab_labels_seen[0]
    assert flow_calls == ["flow"]


def test_render_investor_flow_tab_renders_us_context_sections(monkeypatch):
    calls: list[str] = []

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: calls.append(str(kwargs.get("title", ""))))
    monkeypatch.setattr(tabs.st, "info", lambda *args, **kwargs: calls.append("info"))
    monkeypatch.setattr(tabs.st, "dataframe", lambda *args, **kwargs: calls.append("dataframe"))
    monkeypatch.setattr(tabs.st, "caption", lambda *args, **kwargs: calls.append("caption"))

    frame = pd.DataFrame(
        {
            "sector_name": ["Financials"],
            "activity_state": ["elevated"],
            "activity_zscore": [1.2],
            "dollar_volume": [100.0],
            "dollar_volume_short_mean": [90.0],
            "dollar_volume_long_mean": [70.0],
            "shares_outstanding": [985_650_000.0],
            "assets_under_management": [50_588_090_000.0],
            "nav": [51.32],
            "net_cash_amount": [92_163_529.64],
        },
        index=pd.to_datetime(["2026-04-07"]),
    )
    frame.attrs["missing_tickers"] = ["XLK"]

    tabs.render_investor_flow_tab(
        tab=_DummyTab(),
        signals=[],
        investor_flow_frame=frame,
        investor_flow_status="LIVE",
        investor_flow_fresh=False,
        investor_flow_profile="foreign_lead",
        investor_flow_detail={
            "ownership_context": {
                "ici_weekly_flows": {"as_of": "3/4/2026", "table": pd.DataFrame({"category": ["Equity"], "value": [12154.0]})},
                "sec_13f_positions": {"dataset_label": "Latest 13F", "table": pd.DataFrame({"sector_code": ["XLF"], "sector_name": ["Financials"], "cusip": ["81369Y605"], "filing_count": [2], "manager_value_total_usd": [3000000.0], "manager_shares_total": [50000.0]})},
                "sec_13dg_events": {"lookback_days": 180, "table": pd.DataFrame({"sector_code": ["XLF"], "sector_name": ["Financials"], "matched_top_holdings": [1], "recent_13dg_events": [2], "sample_events": ["SC 13G 2026-03-01"]})},
                "form_sho_context": {"status": "policy_only", "note": "policy"},
                "errors": {},
            }
        },
        market_id="US",
        ui_locale="ko",
    )

    assert "US Flow Proxies" in calls
    assert "Weekly ETF Net Issuance" in calls
    assert "Latest 13F Sector ETF Positioning" in calls
    assert "Recent 13D/13G Events" in calls
    assert "Form SHO Context" in calls
