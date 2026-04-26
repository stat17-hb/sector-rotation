from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd

from src.dashboard import tabs
import src.data_sources.krx_stock_screening as screening_mod
from src.signals.flow import summarize_sector_investor_flow
import src.ui.components as ui_components


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _fail_st_tabs(_labels):
    raise AssertionError("st.tabs should not be used")


def test_render_decision_first_sections_orders_main_canvas(monkeypatch):
    calls: list[str] = []
    shared_maps: list[dict[str, object] | None] = []
    shared_flow_summary_map = {"5044": {"dummy": True}}
    sigma_windows: list[tuple[int, int]] = []
    monkeypatch.setattr(
        tabs.st,
        "session_state",
        {
            "filter_action_global": "__ALL__",
            "filter_regime_only_global": False,
            "show_alerted_only": False,
            "position_mode": "all",
            "held_sectors": [],
        },
    )

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
        lambda **kwargs: calls.append("flow") or shared_maps.append(kwargs.get("shared_flow_summary_map")) or sigma_windows.append((kwargs.get("flow_short_window"), kwargs.get("flow_long_window"))),
    )
    monkeypatch.setattr(
        tabs,
        "render_investor_decision_boards",
        lambda **kwargs: calls.append("boards") or ["Sector A"],
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
        analysis_canvas_kwargs={"heatmap_return_display": None},
        market_id="KR",
        shared_flow_summary_map=shared_flow_summary_map,
        flow_short_window=15,
        flow_long_window=45,
    )

    assert calls == ["hero", "status", "flow", "boards", "analysis"]
    assert result == ["Sector A"]
    assert shared_maps == [shared_flow_summary_map]
    assert sigma_windows == [(15, 45)]


def test_render_decision_first_sections_skips_kr_flow_summary_for_us(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        tabs.st,
        "session_state",
        {
            "filter_action_global": "__ALL__",
            "filter_regime_only_global": False,
            "show_alerted_only": False,
            "position_mode": "all",
            "held_sectors": [],
        },
    )

    monkeypatch.setattr(tabs, "render_decision_hero", lambda **kwargs: calls.append("hero"))
    monkeypatch.setattr(tabs, "render_status_card_row", lambda **kwargs: calls.append("status"))
    monkeypatch.setattr(tabs, "render_investor_flow_summary", lambda **kwargs: calls.append("flow"))
    monkeypatch.setattr(
        tabs,
        "render_investor_decision_boards",
        lambda **kwargs: calls.append("boards") or ["Sector A"],
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
        analysis_canvas_kwargs={"heatmap_return_display": None},
        market_id="US",
    )

    assert calls == ["hero", "status", "boards", "analysis"]


def test_render_decision_first_sections_filter_results_do_not_modify_upper_boards_or_canvas(monkeypatch):
    boards_inputs: list[dict[str, object]] = []
    canvas_inputs: list[dict[str, object]] = []
    monkeypatch.setattr(tabs.st, "session_state", {})

    monkeypatch.setattr(tabs, "render_decision_hero", lambda **kwargs: None)
    monkeypatch.setattr(tabs, "render_status_card_row", lambda **kwargs: None)
    monkeypatch.setattr(tabs, "render_investor_flow_summary", lambda **kwargs: None)
    monkeypatch.setattr(
        tabs,
        "render_investor_decision_boards",
        lambda **kwargs: boards_inputs.append(kwargs) or ["Sector A"],
    )
    monkeypatch.setattr(
        tabs,
        "render_top_bar_filters",
        lambda **kwargs: ("Watch", False if kwargs.get("enable_regime_filter") is False else True, "held", True),
    )
    monkeypatch.setattr(
        tabs,
        "render_analysis_canvas",
        lambda **kwargs: canvas_inputs.append(kwargs),
    )

    analysis_kwargs = {"heatmap_return_display": "heatmap", "selected_cycle_phase": "Recovery"}
    signals = [SimpleNamespace(sector_name="Sector A"), SimpleNamespace(sector_name="Sector B")]

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
        investor_flow_status="SAMPLE",
        investor_flow_fresh=True,
        investor_flow_profile="foreign_lead",
        yield_curve_status="Normal",
        signals=signals,
        held_sector_options=["Sector A", "Sector B"],
        analysis_canvas_kwargs=analysis_kwargs,
        market_id="US",
    )

    assert result == ["Sector A"]
    assert boards_inputs == [{
        "signals": signals,
        "held_sector_options": ["Sector A", "Sector B"],
        "locale": "ko",
    }]
    assert canvas_inputs == [analysis_kwargs]


def test_build_dashboard_page_options_returns_kr_pages_in_expected_order():
    options = tabs.build_dashboard_page_options("KR")

    assert [option.page_id for option in options] == [
        "overview",
        "signals",
        "research",
        "constituents",
        "flow",
        "quality",
    ]
    assert [option.label for option in options] == [
        "대시보드",
        "섹터 모멘텀",
        "상대강도 분석",
        "구성종목",
        "투자자 수급",
        "데이터 수집 이력",
    ]
    assert [option.url_path for option in options] == [
        "kr-overview",
        "kr-signals",
        "kr-research",
        "kr-constituents",
        "kr-flow",
        "kr-quality",
    ]


def test_build_dashboard_page_options_returns_us_pages_without_monitoring():
    options = tabs.build_dashboard_page_options("US")

    assert [option.page_id for option in options] == [
        "overview",
        "signals",
        "research",
        "constituents",
        "flow",
    ]
    assert [option.label for option in options] == [
        "대시보드",
        "섹터 모멘텀",
        "상대강도 분석",
        "구성종목",
        "ETF 수급 프록시",
    ]
    assert [option.url_path for option in options] == [
        "us-overview",
        "us-signals",
        "us-research",
        "us-constituents",
        "us-flow",
    ]


def test_normalize_dashboard_page_id_falls_back_to_summary_when_page_is_unavailable():
    assert tabs.normalize_dashboard_page_id("quality", "US") == "overview"
    assert tabs.normalize_dashboard_page_id("missing", "KR") == "overview"
    assert tabs.normalize_dashboard_page_id(None, "KR") == "overview"
    assert tabs.normalize_dashboard_page_id("flow", "US") == "flow"


def test_render_sidebar_controls_returns_runtime_controls_without_page_radio(monkeypatch):
    monkeypatch.setattr(
        tabs.st,
        "session_state",
        {
            "epsilon": 0.1,
            "price_years": 3,
            "rs_ma_period": 20,
            "ma_fast": 20,
            "ma_slow": 60,
        },
    )

    monkeypatch.setattr(tabs.st, "radio", _fail_st_tabs)
    monkeypatch.setattr(tabs.st, "title", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "subheader", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "toggle", lambda *_args, value=False, **_kwargs: value)
    monkeypatch.setattr(tabs.st, "selectbox", lambda _label, *, options, index=0, **_kwargs: list(options)[index])
    monkeypatch.setattr(tabs.st, "date_input", lambda *_args, value, **_kwargs: value)
    monkeypatch.setattr(tabs.st, "divider", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "popover", lambda *_args, **_kwargs: _DummyContext())
    monkeypatch.setattr(tabs.st, "form", lambda *_args, **_kwargs: _DummyContext())
    monkeypatch.setattr(tabs.st, "columns", lambda *_args, **_kwargs: [_DummyContext(), _DummyContext()])
    monkeypatch.setattr(tabs.st, "slider", lambda _label, *, value, **_kwargs: value)
    monkeypatch.setattr(tabs.st, "form_submit_button", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(tabs.st, "button", lambda *_args, **_kwargs: False)

    result = tabs.render_sidebar_controls(
        market_id="KR",
        ui_labels={"market_selector": "시장", "sidebar_title": "Sector Rotation"},
        theme_mode="dark",
        analysis_heatmap_palette="classic",
        probe_price_status="LIVE",
        probe_macro_status="LIVE",
        probe_investor_flow_status="LIVE",
        flow_profile="foreign_lead",
        momentum_method="legacy_rs_ma_v0",
        btn_states={"refresh_market": True, "refresh_macro": True, "recompute": True},
        asof_default=date(2026, 4, 26),
        ui_locale="ko",
    )

    assert result == (date(2026, 4, 26), "foreign_lead", False, False, False, False)


def test_render_sidebar_controls_hides_kr_flow_profile_for_us(monkeypatch):
    subheaders: list[str] = []
    session_state = {
        "epsilon": 0.1,
        "price_years": 3,
        "rs_ma_period": 20,
        "ma_fast": 20,
        "ma_slow": 60,
    }
    monkeypatch.setattr(tabs.st, "session_state", session_state)

    monkeypatch.setattr(tabs.st, "radio", _fail_st_tabs)
    monkeypatch.setattr(tabs.st, "title", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "subheader", lambda text, **_kwargs: subheaders.append(str(text)))
    monkeypatch.setattr(tabs.st, "toggle", lambda *_args, value=False, **_kwargs: value)
    monkeypatch.setattr(tabs.st, "selectbox", lambda _label, *, options, index=0, **_kwargs: list(options)[index])
    monkeypatch.setattr(tabs.st, "date_input", lambda *_args, value, **_kwargs: value)
    monkeypatch.setattr(tabs.st, "divider", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "popover", lambda *_args, **_kwargs: _DummyContext())
    monkeypatch.setattr(tabs.st, "form", lambda *_args, **_kwargs: _DummyContext())
    monkeypatch.setattr(tabs.st, "columns", lambda *_args, **_kwargs: [_DummyContext(), _DummyContext()])
    monkeypatch.setattr(tabs.st, "slider", lambda _label, *, value, **_kwargs: value)
    monkeypatch.setattr(tabs.st, "form_submit_button", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(tabs.st, "button", lambda *_args, **_kwargs: False)

    result = tabs.render_sidebar_controls(
        market_id="US",
        ui_labels={"market_selector": "시장", "sidebar_title": "Sector Rotation"},
        theme_mode="dark",
        analysis_heatmap_palette="classic",
        probe_price_status="LIVE",
        probe_macro_status="LIVE",
        probe_investor_flow_status="LIVE",
        flow_profile="foreign_lead",
        momentum_method="legacy_rs_ma_v0",
        btn_states={"refresh_market": True, "refresh_macro": True, "recompute": True},
        asof_default=date(2026, 4, 26),
        ui_locale="ko",
    )

    assert result == (date(2026, 4, 26), "foreign_lead", False, False, False, False)
    assert tabs.get_ui_text("flow_profile_label", "ko") not in subheaders


def test_render_dashboard_tabs_routes_flow_page_for_kr_and_us(monkeypatch):
    calls: list[dict[str, object]] = []
    blocked_renderers: list[str] = []

    monkeypatch.setattr(tabs.st, "tabs", _fail_st_tabs)
    monkeypatch.setattr(tabs.st, "container", lambda: _DummyContext())
    monkeypatch.setattr(tabs, "render_summary_tab", lambda **_kwargs: blocked_renderers.append("summary"))
    monkeypatch.setattr(tabs, "render_charts_tab", lambda **_kwargs: blocked_renderers.append("charts"))
    monkeypatch.setattr(tabs, "render_all_signals_tab", lambda **_kwargs: blocked_renderers.append("all_signals"))
    monkeypatch.setattr(tabs, "render_screening_tab", lambda **_kwargs: blocked_renderers.append("screening"))
    monkeypatch.setattr(tabs, "render_monitoring_tab", lambda **_kwargs: blocked_renderers.append("monitoring"))
    monkeypatch.setattr(tabs, "render_investor_flow_tab", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        tabs,
        "render_top_bar_filters",
        lambda **kwargs: ("__ALL__", False, "all", False),
    )

    common_kwargs = dict(
        current_regime="Recovery",
        theme_mode="dark",
        signals=[],
        held_sectors=[],
        settings={},
        is_mobile_client=False,
        investor_flow_status="CACHED",
        investor_flow_fresh=True,
        investor_flow_profile="foreign_lead",
        investor_flow_frame=pd.DataFrame(),
        shared_flow_summary_map={"5044": {"dummy": True}},
        sector_map={},
        ui_locale="ko",
        selected_page_id="flow",
    )

    tabs.render_dashboard_tabs(market_id="KR", **common_kwargs)
    tabs.render_dashboard_tabs(market_id="US", **common_kwargs)

    assert blocked_renderers == []
    assert [call["market_id"] for call in calls] == ["KR", "US"]
    assert [call["shared_flow_summary_map"] for call in calls] == [
        {"5044": {"dummy": True}},
        {"5044": {"dummy": True}},
    ]
    assert all(call["investor_flow_status"] == "CACHED" for call in calls)
    assert all(call["investor_flow_fresh"] is True for call in calls)


def test_render_dashboard_tabs_routes_summary_page_only(monkeypatch):
    summary_calls: list[dict[str, object]] = []
    blocked_renderers: list[str] = []

    monkeypatch.setattr(tabs.st, "tabs", _fail_st_tabs)
    monkeypatch.setattr(tabs.st, "container", lambda: _DummyContext())
    monkeypatch.setattr(tabs, "render_summary_tab", lambda **kwargs: summary_calls.append(kwargs))
    monkeypatch.setattr(tabs, "render_charts_tab", lambda **_kwargs: blocked_renderers.append("charts"))
    monkeypatch.setattr(tabs, "render_all_signals_tab", lambda **_kwargs: blocked_renderers.append("all_signals"))
    monkeypatch.setattr(tabs, "render_screening_tab", lambda **_kwargs: blocked_renderers.append("screening"))
    monkeypatch.setattr(tabs, "render_investor_flow_tab", lambda **_kwargs: blocked_renderers.append("flow"))
    monkeypatch.setattr(tabs, "render_monitoring_tab", lambda **_kwargs: blocked_renderers.append("monitoring"))
    monkeypatch.setattr(tabs, "render_top_bar_filters", lambda **_kwargs: ("__ALL__", False, "all", False))

    tabs.render_dashboard_tabs(
        current_regime="Recovery",
        theme_mode="dark",
        signals=[],
        held_sectors=[],
        settings={},
        is_mobile_client=False,
        market_id="KR",
        investor_flow_status="CACHED",
        investor_flow_fresh=True,
        investor_flow_profile="foreign_lead",
        investor_flow_frame=pd.DataFrame(),
        sector_map={},
        ui_locale="ko",
        selected_page_id="overview",
    )

    assert len(summary_calls) == 1
    assert summary_calls[0]["theme_mode"] == "dark"
    assert summary_calls[0]["ui_locale"] == "ko"
    assert summary_calls[0]["held_sectors"] == []
    assert blocked_renderers == []


def test_render_dashboard_tabs_routes_filter_state_to_selected_pages(monkeypatch):
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(tabs.st, "tabs", _fail_st_tabs)
    monkeypatch.setattr(tabs.st, "container", lambda: _DummyContext())
    monkeypatch.setattr(tabs, "render_summary_tab", lambda **kwargs: calls.append(("summary", kwargs)))
    monkeypatch.setattr(tabs, "render_charts_tab", lambda **kwargs: calls.append(("charts", kwargs)))
    monkeypatch.setattr(tabs, "render_all_signals_tab", lambda **kwargs: calls.append(("all_signals", kwargs)))
    monkeypatch.setattr(tabs, "render_screening_tab", lambda **kwargs: calls.append(("screening", kwargs)))
    monkeypatch.setattr(tabs, "render_investor_flow_tab", lambda **kwargs: calls.append(("flow", kwargs)))
    monkeypatch.setattr(tabs, "render_monitoring_tab", lambda **kwargs: calls.append(("monitoring", kwargs)))
    monkeypatch.setattr(
        tabs,
        "render_top_bar_filters",
        lambda **kwargs: ("Watch", False if kwargs.get("enable_regime_filter") is False else True, "held", True),
    )

    held_signal = SimpleNamespace(
        sector_name="Held A",
        action="Watch",
        macro_regime="Recovery",
        alerts=["Alert"],
        momentum_method="legacy_rs_ma_v0",
    )
    new_signal = SimpleNamespace(
        sector_name="New B",
        action="Hold",
        macro_regime="Slowdown",
        alerts=[],
        momentum_method="legacy_rs_ma_v0",
    )
    signals = [held_signal, new_signal]
    common_kwargs = dict(
        current_regime="Recovery",
        theme_mode="dark",
        signals=signals,
        held_sectors=["Held A"],
        settings={},
        is_mobile_client=False,
        market_id="KR",
        investor_flow_status="CACHED",
        investor_flow_fresh=True,
        investor_flow_profile="foreign_lead",
        investor_flow_frame=pd.DataFrame(),
        shared_flow_summary_map={"5044": {"dummy": True}},
        sector_map={},
        ui_locale="ko",
    )

    for page_id in ["overview", "signals", "constituents", "flow"]:
        tabs.render_dashboard_tabs(selected_page_id=page_id, **common_kwargs)

    assert [name for name, _kwargs in calls] == ["summary", "all_signals", "screening", "flow"]
    assert calls[0][1]["top_pick_signals"] == [held_signal]
    assert calls[0][1]["signals_filtered"] == [held_signal]
    assert calls[0][1]["held_sectors"] == ["Held A"]
    assert calls[1][1]["signals"] == signals
    assert calls[1][1]["filter_action_global"] == "Watch"
    assert calls[1][1]["filter_regime_only_global"] is False
    assert calls[1][1]["current_regime"] == "Recovery"
    assert calls[1][1]["held_sectors"] == ["Held A"]
    assert calls[1][1]["position_mode"] == "held"
    assert calls[1][1]["show_alerted_only"] is True
    assert calls[2][1]["signals"] == signals
    assert calls[2][1]["benchmark_code"] == "1001"
    assert calls[2][1]["settings"] == {}
    assert calls[3][1]["signals"] == signals
    assert calls[3][1]["shared_flow_summary_map"] == {"5044": {"dummy": True}}
    assert calls[3][1]["investor_flow_profile"] == "foreign_lead"


def test_render_dashboard_tabs_does_not_route_research_canvas_or_duplicate_summary(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(tabs.st, "tabs", _fail_st_tabs)
    monkeypatch.setattr(tabs.st, "container", lambda: _DummyContext())
    monkeypatch.setattr(tabs, "render_summary_tab", lambda **_kwargs: calls.append("summary"))
    monkeypatch.setattr(tabs, "render_charts_tab", lambda **_kwargs: calls.append("charts"))
    monkeypatch.setattr(tabs, "render_all_signals_tab", lambda **_kwargs: calls.append("all_signals"))
    monkeypatch.setattr(tabs, "render_screening_tab", lambda **_kwargs: calls.append("screening"))
    monkeypatch.setattr(tabs, "render_investor_flow_tab", lambda **_kwargs: calls.append("flow"))
    monkeypatch.setattr(tabs, "render_monitoring_tab", lambda **_kwargs: calls.append("monitoring"))
    monkeypatch.setattr(tabs, "render_top_bar_filters", lambda **_kwargs: ("__ALL__", False, "all", False))

    tabs.render_dashboard_tabs(
        current_regime="Recovery",
        theme_mode="dark",
        signals=[],
        held_sectors=[],
        settings={},
        is_mobile_client=False,
        market_id="KR",
        investor_flow_status="CACHED",
        investor_flow_fresh=True,
        investor_flow_profile="foreign_lead",
        investor_flow_frame=pd.DataFrame(),
        sector_map={},
        ui_locale="ko",
        selected_page_id="research",
    )

    assert calls == []


def test_render_summary_tab_keeps_summary_surfaces_without_hero_status_duplication(monkeypatch):
    calls: list[str] = []

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyBlock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(tabs.st, "container", lambda **kwargs: _DummyBlock())
    monkeypatch.setattr(tabs.st, "divider", lambda: calls.append("divider"))
    monkeypatch.setattr(tabs.st, "subheader", lambda text: calls.append(f"subheader:{text}"))
    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: calls.append(f"panel:{kwargs.get('title', '')}"))
    monkeypatch.setattr(ui_components, "render_decision_hero", lambda **kwargs: calls.append("hero"))
    monkeypatch.setattr(ui_components, "render_status_card_row", lambda **kwargs: calls.append("status"))
    monkeypatch.setattr(ui_components, "render_top_picks_table", lambda *args, **kwargs: calls.append("top_picks"))
    monkeypatch.setattr(ui_components, "render_action_summary", lambda *args, **kwargs: calls.append("action_summary"))

    tabs.render_summary_tab(
        tab=_DummyTab(),
        theme_mode="dark",
        top_pick_signals=[],
        signals_filtered=[],
        held_sectors=[],
        ui_locale="ko",
    )

    assert "hero" not in calls
    assert "status" not in calls
    assert "top_picks" in calls
    assert "action_summary" in calls
    assert "panel:상위 추천 스냅샷" in calls
    assert "panel:액션 분포" in calls


def test_render_summary_tab_uses_top_pick_and_filtered_inputs_separately(monkeypatch):
    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyBlock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    top_pick_calls: list[tuple[object, dict]] = []
    action_summary_calls: list[tuple[object, dict]] = []

    monkeypatch.setattr(tabs.st, "container", lambda **kwargs: _DummyBlock())
    monkeypatch.setattr(tabs.st, "divider", lambda: None)
    monkeypatch.setattr(tabs.st, "subheader", lambda text: None)
    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(
        ui_components,
        "render_top_picks_table",
        lambda signals, **kwargs: top_pick_calls.append((signals, kwargs)),
    )
    monkeypatch.setattr(
        ui_components,
        "render_action_summary",
        lambda signals, **kwargs: action_summary_calls.append((signals, kwargs)),
    )
    top_pick_signals = [SimpleNamespace(sector_name="Top 1"), SimpleNamespace(sector_name="Top 2")]
    filtered_signals = [SimpleNamespace(sector_name="Filtered 1")]

    tabs.render_summary_tab(
        tab=_DummyTab(),
        theme_mode="dark",
        top_pick_signals=top_pick_signals,
        signals_filtered=filtered_signals,
        held_sectors=["Held A"],
        ui_locale="ko",
    )

    assert top_pick_calls == [(
        top_pick_signals,
        {"held_sectors": ["Held A"], "limit": 5, "locale": "ko"},
    )]
    assert action_summary_calls == [(
        filtered_signals,
        {"theme_mode": "dark", "locale": "ko"},
    )]


def test_render_analysis_canvas_adds_research_context_header(monkeypatch):
    header_calls: list[dict[str, object]] = []

    class _DummyBlock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: header_calls.append(kwargs))
    monkeypatch.setattr(tabs.st, "container", lambda **kwargs: _DummyBlock())
    monkeypatch.setattr(tabs.st, "plotly_chart", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs, "render_cycle_timeline_panel", lambda **kwargs: kwargs["selected_cycle_phase"])
    monkeypatch.setattr(tabs, "render_sector_detail_panel", lambda **kwargs: ("", "1Y"))
    monkeypatch.setattr(tabs.st, "columns", lambda spec, **kwargs: [_DummyBlock() for _ in range(len(spec))])
    monkeypatch.setattr(tabs.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(tabs.st, "markdown", lambda *args, **kwargs: None)

    tabs.render_analysis_canvas(
        heatmap_return_display=pd.DataFrame(),
        heatmap_strength_display=pd.DataFrame(),
        selected_cycle_phase="ALL",
        theme_mode="dark",
        analysis_heatmap_palette="classic",
        visible_segments=[],
        current_regime="Recovery",
        analysis_prices_phase=pd.DataFrame(),
        analysis_prices=pd.DataFrame(),
        sector_columns=[],
        benchmark_label="KOSPI",
        analysis_max_date=pd.Timestamp("2026-04-19").date(),
        analysis_min_date=pd.Timestamp("2025-04-19").date(),
        build_sector_detail_figure=lambda *args, **kwargs: object(),
        resolve_range_from_preset=lambda **kwargs: (
            pd.Timestamp("2025-04-19").date(),
            pd.Timestamp("2026-04-19").date(),
        ),
        signal_lookup={},
        ui_locale="ko",
    )

    assert header_calls
    assert header_calls[0]["title"] == "신호 판단 검증용 섹터 비교"
    assert "연구 표면" in str(header_calls[0]["description"])


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


def test_render_investor_flow_tab_hides_action_change_table_for_reference_only_state(monkeypatch):
    warnings: list[str] = []
    infos: list[str] = []
    dataframes: list[pd.DataFrame] = []

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(tabs.st, "expander", lambda *_args, **_kwargs: _DummyTab())
    monkeypatch.setattr(tabs.st, "warning", lambda text: warnings.append(text))
    monkeypatch.setattr(tabs.st, "info", lambda text: infos.append(text))
    monkeypatch.setattr(tabs.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "dataframe", lambda df, **kwargs: dataframes.append(df.copy()))

    frame = pd.DataFrame(
        {
            "sector_code": ["5044"],
            "sector_name": ["KRX 반도체"],
            "investor_type": ["외국인"],
            "buy_amount": [100],
            "sell_amount": [50],
            "net_buy_amount": [50],
            "net_flow_ratio": [0.2],
        },
        index=pd.to_datetime(["2026-04-07"]),
    )

    signal = SimpleNamespace(
        index_code="5044",
        sector_name="KRX 반도체",
        action="Strong Buy",
    )
    signal.base_action = "Watch"
    signal.flow_profile = "foreign_lead"
    signal.flow_state = "supportive"
    signal.flow_score = 1.2
    signal.foreign_flow_state = "supportive"
    signal.institutional_flow_state = "supportive"
    signal.retail_flow_state = "adverse"

    tabs.render_investor_flow_tab(
        tab=_DummyTab(),
        signals=[signal],
        investor_flow_frame=frame,
        investor_flow_status="CACHED",
        investor_flow_fresh=False,
        investor_flow_profile="foreign_lead",
        investor_flow_detail={"bootstrap_partial_preview": True},
        market_id="KR",
        ui_locale="ko",
    )

    assert any("partial preview" in text or "cached snapshot" in text for text in warnings)
    assert any("reference-only" in text for text in warnings)
    assert any("최종 신호에는 반영되지 않았습니다" in text for text in warnings)
    assert any("의견 변화 표를 숨기고 참여 주체 비교표와 raw snapshot만 표시합니다" in text for text in infos)
    assert len(dataframes) == 2
    assert list(dataframes[0].columns) == [
        "Sector",
        "Flow state",
        "Flow sigma",
        "Foreign",
        "Institutional",
        "Retail",
        "Foreign σ",
        "Institutional σ",
        "Retail σ",
    ]
    assert list(dataframes[1].columns) == ["Sector", "Investor", "Latest ratio", "net_buy_amount", "Flow state"]


def test_render_investor_flow_tab_shows_glance_matrix_before_raw_snapshot(monkeypatch):
    dataframes: list[pd.DataFrame] = []
    warnings: list[str] = []

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(tabs.st, "warning", lambda text, *_args, **_kwargs: warnings.append(str(text)))
    monkeypatch.setattr(tabs.st, "info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "dataframe", lambda df, **kwargs: dataframes.append(df.copy()))

    frame = pd.DataFrame(
        {
            "sector_code": ["5044", "5044"],
            "sector_name": ["KRX 반도체", "KRX 반도체"],
            "investor_type": ["외국인", "기관합계"],
            "buy_amount": [100, 110],
            "sell_amount": [50, 60],
            "net_buy_amount": [50, 50],
            "net_flow_ratio": [0.2, 0.15],
        },
        index=pd.to_datetime(["2026-04-07", "2026-04-07"]),
    )
    signal = SimpleNamespace(
        index_code="5044",
        sector_name="KRX 반도체",
        action="Strong Buy",
    )
    signal.base_action = "Watch"
    signal.flow_profile = "foreign_lead"
    signal.flow_state = "supportive"
    signal.flow_score = 1.2
    signal.foreign_flow_state = "supportive"
    signal.institutional_flow_state = "neutral"
    signal.retail_flow_state = "adverse"

    tabs.render_investor_flow_tab(
        tab=_DummyTab(),
        signals=[signal],
        investor_flow_frame=frame,
        investor_flow_status="LIVE",
        investor_flow_fresh=True,
        investor_flow_profile="foreign_lead",
        investor_flow_detail={},
        market_id="KR",
        ui_locale="ko",
    )

    assert len(dataframes) == 2
    assert list(dataframes[0].columns) == [
        "Sector",
        "Flow state",
        "Flow sigma",
        "Foreign",
        "Institutional",
        "Retail",
        "Foreign σ",
        "Institutional σ",
        "Retail σ",
        "Action change",
    ]
    assert list(dataframes[1].columns) == ["Sector", "Investor", "Latest ratio", "net_buy_amount", "Flow state"]
    assert not any("reference-only" in text or "참고용" in text for text in warnings)


def test_render_investor_flow_tab_uses_raw_snapshot_fallback_when_signal_flow_is_missing(monkeypatch):
    dataframes: list[pd.DataFrame] = []
    infos: list[str] = []

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(tabs.st, "expander", lambda *_args, **_kwargs: _DummyTab())
    monkeypatch.setattr(tabs.st, "warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "info", lambda text, **kwargs: infos.append(str(text)))
    monkeypatch.setattr(tabs.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "dataframe", lambda df, **kwargs: dataframes.append(df.copy()))

    frame = pd.DataFrame(
        {
            "sector_code": ["5044", "5044", "5044"],
            "sector_name": ["KRX 반도체", "KRX 반도체", "KRX 반도체"],
            "investor_type": ["외국인", "기관합계", "개인"],
            "buy_amount": [100, 110, 90],
            "sell_amount": [50, 60, 120],
            "net_buy_amount": [50, 50, -30],
            "net_flow_ratio": [0.2, 0.15, -0.05],
        },
        index=pd.to_datetime(["2026-04-07", "2026-04-07", "2026-04-07"]),
    )

    signal = SimpleNamespace(
        index_code="5044",
        sector_name="KRX 반도체",
        action="Watch",
        base_action="Watch",
        flow_profile="foreign_lead",
        flow_state="unavailable",
        flow_score=0.0,
        foreign_flow_state="unavailable",
        institutional_flow_state="unavailable",
        retail_flow_state="unavailable",
    )

    tabs.render_investor_flow_tab(
        tab=_DummyTab(),
        signals=[signal],
        investor_flow_frame=frame,
        investor_flow_status="CACHED",
        investor_flow_fresh=False,
        investor_flow_profile="foreign_lead",
        investor_flow_detail={"bootstrap_partial_preview": True},
        market_id="KR",
        ui_locale="ko",
    )

    assert any("참여 주체 비교표와 raw snapshot" in text for text in infos)
    assert len(dataframes) == 2
    assert list(dataframes[0].columns) == [
        "Sector",
        "Flow state",
        "Flow sigma",
        "Foreign",
        "Institutional",
        "Retail",
        "Foreign σ",
        "Institutional σ",
        "Retail σ",
    ]
    assert dataframes[0].iloc[0]["Foreign"] == "+20.00%"


def test_render_investor_flow_tab_adds_participant_matched_reference_cues_to_raw_table(monkeypatch):
    dataframes: list[pd.DataFrame] = []
    markdown_calls: list[str] = []

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(tabs.st, "expander", lambda *_args, **_kwargs: _DummyTab())
    monkeypatch.setattr(tabs.st, "warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "markdown", lambda text, **kwargs: markdown_calls.append(text))
    monkeypatch.setattr(tabs.st, "dataframe", lambda df, **kwargs: dataframes.append(df.copy()))

    frame = pd.DataFrame(
        {
            "sector_code": ["5044"] * 15 + ["9999"],
            "sector_name": ["KRX 반도체"] * 15 + ["KRX 미확인"],
            "investor_type": ["외국인"] * 5 + ["기관합계"] * 5 + ["개인"] * 5 + ["외국인"],
            "buy_amount": [101, 102, 103, 104, 105, 90, 91, 92, 93, 94, 80, 79, 78, 77, 76, 50],
            "sell_amount": [100, 99, 98, 97, 96, 90, 89, 88, 87, 86, 81, 82, 83, 84, 85, 49],
            "net_buy_amount": [1, 3, 5, 7, 9, 0, 2, 4, 6, 8, -1, -3, -5, -7, -9, 1],
            "net_flow_ratio": [0.01, 0.02, 0.03, 0.05, 0.06, 0.00, 0.01, 0.02, 0.04, 0.05, -0.01, -0.02, -0.03, -0.04, -0.05, 0.01],
        },
        index=pd.to_datetime(
            ["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04", "2026-04-05"] * 3 + ["2026-04-05"]
        ),
    )
    summary_map = summarize_sector_investor_flow(
        frame.loc[frame["sector_code"] == "5044"],
        flow_profile="foreign_lead",
        short_window=3,
        long_window=5,
    )

    tabs.render_investor_flow_tab(
        tab=_DummyTab(),
        signals=[],
        investor_flow_frame=frame,
        investor_flow_status="LIVE",
        investor_flow_fresh=True,
        investor_flow_profile="foreign_lead",
        investor_flow_detail={},
        shared_flow_summary_map=summary_map,
        market_id="KR",
        ui_locale="ko",
    )

    assert len(dataframes) == 2
    raw_df = dataframes[1]
    assert list(raw_df.columns) == ["Sector", "Investor", "Latest ratio", "net_buy_amount", "Flow state"]
    matched_rows = raw_df[raw_df["Sector"] == "KRX 반도체"].reset_index(drop=True)
    assert "수급 우호" in matched_rows.loc[0, "Flow state"]
    assert "수급 우호" in matched_rows.loc[1, "Flow state"]
    assert "수급 역풍" in matched_rows.loc[2, "Flow state"]
    unavailable_rows = raw_df[raw_df["Sector"] == "KRX 미확인"].reset_index(drop=True)
    assert unavailable_rows.loc[0, "Flow state"] == "실험 데이터 없음"
    assert any("장기 표준편차" in text for text in markdown_calls)


def test_render_monitoring_tab_splits_sector_and_ticker_failures(monkeypatch):
    calls: list[tuple[str, str]] = []

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyCol:
        def metric(self, *args, **kwargs):
            return None

    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(
        tabs,
        "_cached_monitoring_data",
        lambda market_id: {
            "warm": {
                "status": "LIVE",
                "watermark_key": "20260410",
                "coverage_complete": False,
                "predicted_requests": 984,
                "processed_requests": 984,
                "failed_days": [],
                "failed_codes": {"sector:5044": "lookup failed", "005930": "buy frame empty"},
                "failed_sector_codes": {"sector:5044": "lookup failed"},
                "failed_ticker_codes": {"005930": "buy frame empty"},
                "failed_ticker_family_counts": {"exception_backed_empty": 1},
                "aborted": False,
                "abort_reason": "",
            },
            "bounds": {"min_trade_date": "2025-01-02", "max_trade_date": "2026-04-10"},
            "history": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(tabs.st, "subheader", lambda text: calls.append(("subheader", str(text))))
    monkeypatch.setattr(tabs.st, "warning", lambda text, **kwargs: calls.append(("warning", str(text))))
    monkeypatch.setattr(tabs.st, "info", lambda text, **kwargs: calls.append(("info", str(text))))
    monkeypatch.setattr(tabs.st, "success", lambda text, **kwargs: calls.append(("success", str(text))))
    monkeypatch.setattr(tabs.st, "error", lambda text, **kwargs: calls.append(("error", str(text))))
    monkeypatch.setattr(tabs.st, "dataframe", lambda *args, **kwargs: calls.append(("dataframe", "shown")))
    monkeypatch.setattr(tabs.st, "columns", lambda n: [_DummyCol() for _ in range(n)])

    tabs.render_monitoring_tab(tab=_DummyTab(), market_id="KR", ui_locale="ko")

    warning_texts = [text for kind, text in calls if kind == "warning"]
    info_texts = [text for kind, text in calls if kind == "info"]
    assert "오류 섹터 1건" in warning_texts
    assert "오류 종목 1건" in warning_texts
    assert any("마지막 incomplete manual_refresh 저장 결과" in text for text in info_texts)
    assert any("실패 유형 요약" in text for text in info_texts)


def test_render_monitoring_tab_separates_other_collection_errors(monkeypatch):
    calls: list[tuple[str, str]] = []

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyCol:
        def metric(self, *args, **kwargs):
            return None

    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(
        tabs,
        "_cached_monitoring_data",
        lambda market_id: {
            "warm": {
                "status": "CACHED",
                "watermark_key": "20260410",
                "coverage_complete": False,
                "predicted_requests": 0,
                "processed_requests": 0,
                "failed_days": [],
                "failed_codes": {"warehouse": "lock held", "refresh": "transport timeout"},
                "aborted": False,
                "abort_reason": "",
            },
            "bounds": {"min_trade_date": "", "max_trade_date": ""},
            "history": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(tabs.st, "subheader", lambda text: calls.append(("subheader", str(text))))
    monkeypatch.setattr(tabs.st, "warning", lambda text, **kwargs: calls.append(("warning", str(text))))
    monkeypatch.setattr(tabs.st, "info", lambda text, **kwargs: calls.append(("info", str(text))))
    monkeypatch.setattr(tabs.st, "success", lambda text, **kwargs: calls.append(("success", str(text))))
    monkeypatch.setattr(tabs.st, "error", lambda text, **kwargs: calls.append(("error", str(text))))
    monkeypatch.setattr(tabs.st, "dataframe", lambda *args, **kwargs: calls.append(("dataframe", "shown")))
    monkeypatch.setattr(tabs.st, "columns", lambda n: [_DummyCol() for _ in range(n)])

    tabs.render_monitoring_tab(tab=_DummyTab(), market_id="KR", ui_locale="ko")

    warning_texts = [text for kind, text in calls if kind == "warning"]
    assert "기타 수집 오류 2건" in warning_texts


def test_render_monitoring_tab_keeps_warm_status_when_runtime_status_is_omitted(monkeypatch):
    metrics: list[tuple[str, str]] = []

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyCol:
        def metric(self, label, value, *args, **kwargs):
            metrics.append((str(label), str(value)))
            return None

    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(
        tabs,
        "_cached_monitoring_data",
        lambda market_id: {
            "warm": {
                "status": "LIVE",
                "watermark_key": "20260410",
                "coverage_complete": True,
                "predicted_requests": 10,
                "processed_requests": 10,
                "failed_days": [],
                "failed_codes": {},
                "aborted": False,
                "abort_reason": "",
            },
            "bounds": {"min_trade_date": "2025-01-02", "max_trade_date": "2026-04-10"},
            "history": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(tabs.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "error", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "columns", lambda n: [_DummyCol() for _ in range(n)])

    tabs.render_monitoring_tab(
        tab=_DummyTab(),
        market_id="KR",
        investor_flow_detail={"status": "SAMPLE", "coverage_complete": False},
        ui_locale="ko",
    )

    assert ("상태", "LIVE") in metrics


def test_render_monitoring_tab_uses_runtime_flow_snapshot_when_warehouse_history_is_empty(monkeypatch):
    calls: list[tuple[str, str]] = []

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyCol:
        def metric(self, *args, **kwargs):
            return None

    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(
        tabs,
        "_cached_monitoring_data",
        lambda market_id: {
            "warm": {
                "status": "SAMPLE",
            },
            "bounds": {"min_trade_date": "", "max_trade_date": ""},
            "history": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(tabs.st, "subheader", lambda text: calls.append(("subheader", str(text))))
    monkeypatch.setattr(tabs.st, "warning", lambda text, **kwargs: calls.append(("warning", str(text))))
    monkeypatch.setattr(tabs.st, "info", lambda text, **kwargs: calls.append(("info", str(text))))
    monkeypatch.setattr(tabs.st, "success", lambda text, **kwargs: calls.append(("success", str(text))))
    monkeypatch.setattr(tabs.st, "error", lambda text, **kwargs: calls.append(("error", str(text))))
    monkeypatch.setattr(tabs.st, "dataframe", lambda *args, **kwargs: calls.append(("dataframe", "shown")))
    monkeypatch.setattr(tabs.st, "columns", lambda n: [_DummyCol() for _ in range(n)])

    runtime_frame = pd.DataFrame(
        {
            "sector_code": ["5044"],
            "sector_name": ["KRX 반도체"],
            "investor_type": ["외국인"],
            "buy_amount": [100],
            "sell_amount": [50],
            "net_buy_amount": [50],
            "net_flow_ratio": [0.2],
        },
        index=pd.to_datetime(["2026-04-10"]),
    )

    tabs.render_monitoring_tab(
        tab=_DummyTab(),
        market_id="KR",
        investor_flow_status="CACHED",
        investor_flow_fresh=False,
        investor_flow_detail={
            "watermark_key": "20260410",
            "coverage_complete": False,
            "reason": "manual_refresh",
            "anchor_start": "20251211",
            "predicted_requests": 876,
            "processed_requests": 876,
        },
        investor_flow_frame=runtime_frame,
        ui_locale="ko",
    )

    info_texts = [text for kind, text in calls if kind == "info"]
    assert any("2026-04-10" in text for text in info_texts)
    assert any("runtime snapshot" in text for text in info_texts)
    assert ("dataframe", "shown") in calls


def test_render_screening_tab_renders_representative_etf_context(monkeypatch):
    calls: list[str] = []
    dataframe_payloads: list[object] = []

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyContainer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: calls.append(str(kwargs.get("title", ""))))
    monkeypatch.setattr(screening_mod, "load_screened_stocks", lambda **kwargs: ("LIVE", [{"ticker": "005930", "name": "삼성전자", "sector_name": "KRX 반도체", "rs": 1.2, "rsi": 58.0, "rs_strong": True, "trend_ok": True, "momentum_ok": True, "ret_1m": 5.0, "ret_3m": 11.0, "alerts": ""}]))
    monkeypatch.setattr(
        screening_mod,
        "load_representative_etf_context",
        lambda **kwargs: (
            "LIVE",
            [
                {
                    "sector_name": "KRX 반도체",
                    "etf_code": "396500",
                    "etf_name": "TIGER 반도체TOP10",
                    "style_tags": "TOP10",
                    "execution_state": "정상",
                    "latest_trade_value": 1_200_000_000.0,
                    "avg_trade_value_20d": 850_000_000.0,
                    "net_assets": 220_000_000_000.0,
                    "nav": 10200.0,
                    "reference_date": "20260428",
                    "freshness_label": "20260428",
                    "note": "",
                }
            ],
        ),
    )
    monkeypatch.setattr(tabs.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(tabs.st, "toggle", lambda *args, **kwargs: True)
    monkeypatch.setattr(tabs.st, "columns", lambda spec: [_DummyContainer() for _ in range(len(spec))])
    monkeypatch.setattr(tabs.st, "spinner", lambda *args, **kwargs: _DummyContainer())
    monkeypatch.setattr(tabs.st, "dataframe", lambda data, **kwargs: dataframe_payloads.append(data))
    monkeypatch.setattr(tabs.st, "download_button", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "info", lambda *args, **kwargs: calls.append("info"))
    monkeypatch.setattr(tabs.st, "markdown", lambda *args, **kwargs: calls.append("markdown"))

    tabs.render_screening_tab(
        tab=_DummyTab(),
        signals=[type("Signal", (), {"index_code": "5044", "sector_name": "KRX 반도체", "action": "Strong Buy"})()],
        settings={},
        benchmark_code="1001",
        etf_map={"5044": [{"code": "396500", "name": "TIGER 반도체TOP10"}]},
    )

    assert "Strong Buy 섹터 구성종목" in calls
    assert "대표 ETF 실행 컨텍스트" in calls
    assert len(dataframe_payloads) == 2
    assert isinstance(dataframe_payloads[-1], pd.DataFrame)
    assert "대표 ETF" in dataframe_payloads[-1].columns
