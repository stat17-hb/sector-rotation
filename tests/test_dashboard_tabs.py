from __future__ import annotations

import importlib
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
        lambda **kwargs: calls.append("hero") or shared_maps.append({"export_growth_val": kwargs.get("export_growth_val")}),
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
        export_growth_val=7.5,
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
    assert shared_maps == [{"export_growth_val": 7.5}, shared_flow_summary_map]
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
        "research",
        "quality",
    ]
    assert [option.label for option in options] == [
        "대시보드",
        "상대강도 분석",
        "데이터 수집 이력",
    ]
    assert [option.url_path for option in options] == [
        "kr-overview",
        "kr-research",
        "kr-quality",
    ]


def test_build_dashboard_page_options_returns_us_pages_without_monitoring():
    options = tabs.build_dashboard_page_options("US")

    assert [option.page_id for option in options] == [
        "overview",
        "research",
    ]
    assert [option.label for option in options] == [
        "대시보드",
        "상대강도 분석",
    ]
    assert [option.url_path for option in options] == [
        "us-overview",
        "us-research",
    ]


def test_normalize_dashboard_page_id_falls_back_to_summary_when_page_is_unavailable():
    assert tabs.normalize_dashboard_page_id("quality", "US") == "overview"
    assert tabs.normalize_dashboard_page_id("missing", "KR") == "overview"
    assert tabs.normalize_dashboard_page_id(None, "KR") == "overview"
    assert tabs.normalize_dashboard_page_id("flow", "US") == "overview"
    assert tabs.normalize_dashboard_page_id("signals", "KR") == "overview"
    assert tabs.normalize_dashboard_page_id("constituents", "KR") == "overview"


def test_resolve_dashboard_page_title_uses_current_market_and_page_label():
    assert tabs.resolve_dashboard_page_title("research", "KR") == "KR 상대강도 분석"
    assert tabs.resolve_dashboard_page_title("flow", "US") == "US 대시보드"
    assert tabs.resolve_dashboard_page_title("quality", "US") == "US 대시보드"


def test_format_sidebar_status_chip_escapes_values_and_marks_attention():
    chip = tabs._format_sidebar_status_chip("시장<script>", "STALE<")

    assert "sidebar-status-chip--attention" in chip
    assert "시장&lt;script&gt;" in chip
    assert "STALE&lt;" in chip


def test_render_sidebar_controls_returns_runtime_controls_without_page_radio(monkeypatch):
    markdown_calls: list[str] = []
    button_labels: list[str] = []
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
    monkeypatch.setattr(tabs.st, "markdown", lambda text, **_kwargs: markdown_calls.append(str(text)))
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
    monkeypatch.setattr(tabs.st, "button", lambda label, **_kwargs: (button_labels.append(str(label)), False)[1])

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
        btn_states={"refresh_market": True, "refresh_macro": True},
        asof_default=date(2026, 4, 26),
        ui_locale="ko",
    )

    assert result == (date(2026, 4, 26), "foreign_lead", False, False, False)
    assert any("KR 섹터 콘솔" in call for call in markdown_calls)
    assert any("데이터 운용" in call for call in markdown_calls)
    assert any("sidebar-status-chip--ready" in call for call in markdown_calls)
    assert any("분석 기준" in call for call in markdown_calls)
    assert any("수급 해석" in call for call in markdown_calls)
    assert button_labels == ["시장데이터 갱신", "매크로데이터 갱신", tabs.get_ui_text("flow_refresh_button", "ko")]


def test_render_sidebar_controls_hides_kr_flow_profile_for_us(monkeypatch):
    markdown_calls: list[str] = []
    session_state = {
        "epsilon": 0.1,
        "price_years": 3,
        "rs_ma_period": 20,
        "ma_fast": 20,
        "ma_slow": 60,
    }
    monkeypatch.setattr(tabs.st, "session_state", session_state)

    monkeypatch.setattr(tabs.st, "radio", _fail_st_tabs)
    monkeypatch.setattr(tabs.st, "markdown", lambda text, **_kwargs: markdown_calls.append(str(text)))
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
        market_id="US",
        ui_labels={"market_selector": "시장", "sidebar_title": "Sector Rotation"},
        theme_mode="dark",
        analysis_heatmap_palette="classic",
        probe_price_status="LIVE",
        probe_macro_status="LIVE",
        probe_investor_flow_status="LIVE",
        flow_profile="foreign_lead",
        momentum_method="legacy_rs_ma_v0",
        btn_states={"refresh_market": True, "refresh_macro": True},
        asof_default=date(2026, 4, 26),
        ui_locale="ko",
    )

    assert result == (date(2026, 4, 26), "foreign_lead", False, False, False)
    assert not any("수급 해석" in call for call in markdown_calls)
    assert not any("투자자 수급" in call for call in markdown_calls)


def test_render_dashboard_tabs_routes_hidden_flow_page_to_overview_for_kr_and_us(monkeypatch):
    summary_calls: list[dict[str, object]] = []
    blocked_renderers: list[str] = []

    monkeypatch.setattr(tabs.st, "tabs", _fail_st_tabs)
    monkeypatch.setattr(tabs.st, "container", lambda: _DummyContext())
    monkeypatch.setattr(tabs, "render_summary_tab", lambda **kwargs: summary_calls.append(kwargs))
    monkeypatch.setattr(tabs, "render_charts_tab", lambda **_kwargs: blocked_renderers.append("charts"))
    monkeypatch.setattr(tabs, "render_all_signals_tab", lambda **_kwargs: blocked_renderers.append("all_signals"))
    monkeypatch.setattr(tabs, "render_screening_tab", lambda **_kwargs: blocked_renderers.append("screening"))
    monkeypatch.setattr(tabs, "render_monitoring_tab", lambda **_kwargs: blocked_renderers.append("monitoring"))
    monkeypatch.setattr(tabs, "render_investor_flow_tab", lambda **_kwargs: blocked_renderers.append("flow"))
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
    assert len(summary_calls) == 2
    assert [call["ui_locale"] for call in summary_calls] == ["ko", "ko"]
    assert all(call["top_pick_signals"] == [] for call in summary_calls)
    assert all(call["signals_filtered"] == [] for call in summary_calls)


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

    assert [name for name, _kwargs in calls] == ["summary", "summary", "summary", "summary"]
    for _name, kwargs in calls:
        assert kwargs["top_pick_signals"] == [held_signal]
        assert kwargs["signals_filtered"] == [held_signal]
        assert kwargs["held_sectors"] == ["Held A"]
        assert kwargs["theme_mode"] == "dark"
        assert kwargs["ui_locale"] == "ko"


def test_render_dashboard_tabs_quality_page_does_not_render_research_filters(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(tabs.st, "tabs", _fail_st_tabs)
    monkeypatch.setattr(tabs.st, "container", lambda: _DummyContext())
    monkeypatch.setattr(tabs, "render_summary_tab", lambda **_kwargs: calls.append("summary"))
    monkeypatch.setattr(tabs, "render_charts_tab", lambda **_kwargs: calls.append("charts"))
    monkeypatch.setattr(tabs, "render_all_signals_tab", lambda **_kwargs: calls.append("all_signals"))
    monkeypatch.setattr(tabs, "render_screening_tab", lambda **_kwargs: calls.append("screening"))
    monkeypatch.setattr(tabs, "render_investor_flow_tab", lambda **_kwargs: calls.append("flow"))
    monkeypatch.setattr(tabs, "render_monitoring_tab", lambda **_kwargs: calls.append("monitoring"))
    monkeypatch.setattr(
        tabs,
        "render_top_bar_filters",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("quality page must not render research filters")),
    )

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
        selected_page_id="quality",
    )

    assert calls == ["monitoring"]


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


def test_render_all_signals_tab_renders_decision_boards_before_ledger(monkeypatch):
    calls: list[str] = []
    frames: list[dict[str, object]] = []

    monkeypatch.setattr(tabs, "render_research_page_frame", lambda **kwargs: calls.append("frame") or frames.append(kwargs))
    monkeypatch.setattr(tabs, "render_sector_momentum_decision_boards", lambda *_args, **_kwargs: calls.append("decision_boards"))
    monkeypatch.setattr(tabs, "render_theme_lens_panel", lambda *_args, **_kwargs: calls.append("theme_lens") or False)
    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: calls.append(f"panel:{kwargs.get('title')}"))
    monkeypatch.setattr(tabs, "render_signal_table", lambda *_args, **_kwargs: calls.append("signal_table"))
    monkeypatch.setattr(tabs.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "expander", lambda *_args, **_kwargs: _DummyContext())
    monkeypatch.setattr(tabs.st, "markdown", lambda *_args, **_kwargs: None)

    tabs.render_all_signals_tab(
        tab=_DummyContext(),
        signals=[],
        filter_action_global="__ALL__",
        filter_regime_only_global=False,
        current_regime="Recovery",
        held_sectors=[],
        position_mode="all",
        show_alerted_only=False,
        theme_mode="dark",
        settings={},
        etf_map={},
        ui_locale="ko",
    )

    assert calls[:5] == ["frame", "decision_boards", "theme_lens", "panel:전체 섹터 신호 원장", "signal_table"]
    assert frames[0]["title"] == "섹터 액션 보드"
    assert "신규 검토" in frames[0]["description"]


def test_render_all_signals_tab_hides_theme_lens_for_us(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(tabs, "render_research_page_frame", lambda **_kwargs: calls.append("frame"))
    monkeypatch.setattr(tabs, "render_sector_momentum_decision_boards", lambda *_args, **_kwargs: calls.append("decision_boards"))
    monkeypatch.setattr(tabs, "render_theme_lens_panel", lambda *_args, **_kwargs: calls.append("theme_lens") or False)
    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: calls.append(f"panel:{kwargs.get('title')}"))
    monkeypatch.setattr(tabs, "render_signal_table", lambda *_args, **_kwargs: calls.append("signal_table"))
    monkeypatch.setattr(tabs.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "expander", lambda *_args, **_kwargs: _DummyContext())
    monkeypatch.setattr(tabs.st, "markdown", lambda *_args, **_kwargs: None)

    tabs.render_all_signals_tab(
        tab=_DummyContext(),
        signals=[],
        filter_action_global="__ALL__",
        filter_regime_only_global=False,
        current_regime="Recovery",
        held_sectors=[],
        position_mode="all",
        show_alerted_only=False,
        theme_mode="dark",
        settings={},
        etf_map={},
        market_id="US",
        ui_locale="ko",
    )

    assert "theme_lens" not in calls


def test_render_all_signals_tab_uses_live_theme_snapshot(monkeypatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        tabs.st,
        "session_state",
        {
            "_theme_lens_live_snapshot": {
                "status": "LIVE",
                "rows": [{"theme_name": "조선", "status": "LIVE"}],
            }
        },
    )
    monkeypatch.setattr(tabs, "render_research_page_frame", lambda **_kwargs: None)
    monkeypatch.setattr(tabs, "render_sector_momentum_decision_boards", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs, "render_theme_lens_panel", lambda rows, **kwargs: calls.append({"rows": rows, **kwargs}) or False)
    monkeypatch.setattr(tabs, "render_panel_header", lambda **_kwargs: None)
    monkeypatch.setattr(tabs, "render_signal_table", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "expander", lambda *_args, **_kwargs: _DummyContext())
    monkeypatch.setattr(tabs.st, "markdown", lambda *_args, **_kwargs: None)

    tabs.render_all_signals_tab(
        tab=_DummyContext(),
        signals=[],
        filter_action_global="__ALL__",
        filter_regime_only_global=False,
        current_regime="Recovery",
        held_sectors=[],
        position_mode="all",
        show_alerted_only=False,
        theme_mode="dark",
        settings={},
        etf_map={},
        theme_lens_status="UNAVAILABLE",
        theme_lens_rows=[],
        ui_locale="ko",
    )

    assert calls == [
        {
            "rows": [{"theme_name": "조선", "status": "LIVE"}],
            "status": "LIVE",
            "show_refresh_button": True,
        }
    ]


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


def test_build_kr_latest_sector_flow_amounts_formats_large_values():
    frame = pd.DataFrame(
        {
            "sector_name": ["KRX 반도체", "KRX 반도체", "KRX 바이오", "KRX 바이오"],
            "investor_type": ["외국인", "기관합계", "외국인", "개인"],
            "net_buy_amount": [1_250_000_000_000, -230_000_000_000, 19_000_000_000, -11_000_000_000],
        },
        index=pd.to_datetime(["2026-04-07"] * 4),
    )

    display = tabs._build_kr_latest_sector_flow_amounts(frame)

    assert display.iloc[0].to_dict() == {
        "Sector": "KRX 반도체",
        "외국인": "+1.25조",
        "기관": "-2300억",
        "개인": "0원",
        "합계": "+1.02조",
    }
    assert display.iloc[1]["외국인"] == "+190억"
    assert display.iloc[1]["개인"] == "-110억"


def test_build_kr_sector_flow_trend_figure_accepts_trade_date_index_name():
    frame = pd.DataFrame(
        {
            "sector_name": ["KRX 반도체", "KRX 반도체", "KRX 반도체", "KRX 바이오"],
            "investor_type": ["외국인", "기관합계", "개인", "외국인"],
            "net_buy_amount": [100, -20, -30, 50],
        },
        index=pd.to_datetime(["2026-04-06", "2026-04-07", "2026-04-06", "2026-04-07"]),
    )
    frame.index.name = "trade_date"
    frame["trade_date"] = frame.index

    fig = tabs._build_kr_sector_flow_trend_figure(frame)

    assert len(fig.data) == 4
    assert {trace.name for trace in fig.data} == {"외국인", "기관", "개인"}
    assert sum(1 for trace in fig.data if trace.showlegend) == 3
    assert "섹터별 투자자 수급 추이" == fig.layout.title.text


def test_get_kr_flow_sector_options_sorts_by_latest_abs_total():
    frame = pd.DataFrame(
        {
            "sector_name": ["KRX 반도체", "KRX 바이오", "KRX 반도체", "KRX 바이오"],
            "investor_type": ["외국인", "외국인", "기관합계", "개인"],
            "net_buy_amount": [10, 50, -300, -10],
        },
        index=pd.to_datetime(["2026-04-06", "2026-04-06", "2026-04-07", "2026-04-07"]),
    )

    assert tabs._get_kr_flow_sector_options(frame) == ["KRX 반도체", "KRX 바이오"]


def test_render_investor_flow_tab_does_not_render_sector_selectbox(monkeypatch):
    charts: list[object] = []

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(tabs.st, "info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "dataframe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tabs.st, "plotly_chart", lambda fig, **kwargs: charts.append(fig))
    monkeypatch.setattr(
        tabs.st,
        "selectbox",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("flow tab must show all sectors without selectbox")),
    )

    frame = pd.DataFrame(
        {
            "sector_name": ["KRX 반도체", "KRX 반도체", "KRX 바이오", "KRX 바이오"],
            "investor_type": ["외국인", "기관합계", "외국인", "개인"],
            "net_buy_amount": [10, -5, 7, -3],
        },
        index=pd.to_datetime(["2026-04-06", "2026-04-07", "2026-04-06", "2026-04-07"]),
    )

    tabs.render_investor_flow_tab(
        tab=_DummyTab(),
        signals=[],
        investor_flow_frame=frame,
        investor_flow_status="LIVE",
        investor_flow_fresh=True,
        investor_flow_profile="foreign_lead",
        investor_flow_detail={},
        market_id="KR",
        ui_locale="ko",
    )

    assert len(charts) == 1


def test_render_investor_flow_tab_hides_action_change_table_for_reference_only_state(monkeypatch):
    warnings: list[str] = []
    infos: list[str] = []
    dataframes: list[pd.DataFrame] = []
    charts: list[object] = []

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
    monkeypatch.setattr(tabs.st, "plotly_chart", lambda fig, **kwargs: charts.append(fig))

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

    assert warnings == []
    assert infos == []
    assert len(dataframes) == 1
    assert list(dataframes[0].columns) == [
        "Sector",
        "외국인",
        "기관",
        "개인",
        "합계",
    ]
    assert dataframes[0].iloc[0].to_dict() == {
        "Sector": "KRX 반도체",
        "외국인": "+50원",
        "기관": "0원",
        "개인": "0원",
        "합계": "+50원",
    }
    assert len(charts) == 1


def test_render_investor_flow_tab_shows_glance_matrix_before_raw_snapshot(monkeypatch):
    dataframes: list[pd.DataFrame] = []
    warnings: list[str] = []
    charts: list[object] = []

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
    monkeypatch.setattr(tabs.st, "plotly_chart", lambda fig, **kwargs: charts.append(fig))

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

    assert len(dataframes) == 1
    assert list(dataframes[0].columns) == [
        "Sector",
        "외국인",
        "기관",
        "개인",
        "합계",
    ]
    assert dataframes[0].iloc[0].to_dict() == {
        "Sector": "KRX 반도체",
        "외국인": "+50원",
        "기관": "+50원",
        "개인": "0원",
        "합계": "+100원",
    }
    assert len(charts) == 1
    assert not any("reference-only" in text or "참고용" in text for text in warnings)


def test_render_investor_flow_tab_uses_raw_snapshot_fallback_when_signal_flow_is_missing(monkeypatch):
    dataframes: list[pd.DataFrame] = []
    infos: list[str] = []
    charts: list[object] = []

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
    monkeypatch.setattr(tabs.st, "plotly_chart", lambda fig, **kwargs: charts.append(fig))

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

    assert infos == []
    assert len(dataframes) == 1
    assert list(dataframes[0].columns) == [
        "Sector",
        "외국인",
        "기관",
        "개인",
        "합계",
    ]
    assert dataframes[0].iloc[0]["외국인"] == "+50원"
    assert dataframes[0].iloc[0]["기관"] == "+50원"
    assert dataframes[0].iloc[0]["개인"] == "-30원"
    assert dataframes[0].iloc[0]["합계"] == "+70원"
    assert len(charts) == 1


def test_render_investor_flow_tab_adds_participant_matched_reference_cues_to_raw_table(monkeypatch):
    dataframes: list[pd.DataFrame] = []
    markdown_calls: list[str] = []
    charts: list[object] = []

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
    monkeypatch.setattr(tabs.st, "plotly_chart", lambda fig, **kwargs: charts.append(fig))

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

    assert len(dataframes) == 1
    latest_df = dataframes[0]
    assert list(latest_df.columns) == ["Sector", "외국인", "기관", "개인", "합계"]
    matched_rows = latest_df[latest_df["Sector"] == "KRX 반도체"].reset_index(drop=True)
    assert matched_rows.loc[0, "외국인"] == "+9원"
    assert matched_rows.loc[0, "기관"] == "+8원"
    assert matched_rows.loc[0, "개인"] == "-9원"
    assert matched_rows.loc[0, "합계"] == "+8원"
    assert not any("장기 표준편차" in text for text in markdown_calls)
    assert len(charts) == 1


def test_render_monitoring_tab_splits_sector_and_ticker_failures(monkeypatch):
    calls: list[tuple[str, str]] = []
    dataframes: list[pd.DataFrame] = []

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
    monkeypatch.setattr(tabs.st, "dataframe", lambda frame, **kwargs: dataframes.append(frame))
    monkeypatch.setattr(tabs.st, "columns", lambda n: [_DummyCol() for _ in range(n)])

    tabs.render_monitoring_tab(tab=_DummyTab(), market_id="KR", ui_locale="ko")

    error_rows = dataframes[1]
    assert error_rows["데이터"].tolist() == ["수급데이터", "수급데이터"]
    assert error_rows["항목"].tolist() == ["sector:5044", "005930"]


def test_render_monitoring_tab_separates_other_collection_errors(monkeypatch):
    calls: list[tuple[str, str]] = []
    dataframes: list[pd.DataFrame] = []

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
    monkeypatch.setattr(tabs.st, "dataframe", lambda frame, **kwargs: dataframes.append(frame))
    monkeypatch.setattr(tabs.st, "columns", lambda n: [_DummyCol() for _ in range(n)])

    tabs.render_monitoring_tab(tab=_DummyTab(), market_id="KR", ui_locale="ko")

    error_rows = dataframes[1]
    assert error_rows["항목"].tolist() == ["warehouse", "refresh"]


def test_render_monitoring_tab_keeps_warm_status_when_runtime_status_is_omitted(monkeypatch):
    dataframes: list[pd.DataFrame] = []

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

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
            "history": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(tabs.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "error", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "dataframe", lambda frame, **kwargs: dataframes.append(frame))
    monkeypatch.setattr(tabs.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "markdown", lambda *args, **kwargs: None)

    tabs.render_monitoring_tab(
        tab=_DummyTab(),
        market_id="KR",
        investor_flow_detail={"status": "SAMPLE", "coverage_complete": False},
        ui_locale="ko",
    )

    flow_status = dataframes[0][dataframes[0]["데이터"].eq("수급데이터")].iloc[0]
    assert flow_status["상태"] == "LIVE"


def test_render_monitoring_tab_uses_runtime_flow_snapshot_when_warehouse_history_is_empty(monkeypatch):
    calls: list[tuple[str, str]] = []
    dataframes: list[pd.DataFrame] = []

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
    monkeypatch.setattr(tabs.st, "dataframe", lambda frame, **kwargs: dataframes.append(frame))
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

    flow_status = dataframes[0][dataframes[0]["데이터"].eq("수급데이터")].iloc[0]
    assert flow_status["상태"] == "CACHED"
    assert dataframes[-1]["요청범위"].tolist() == ["20251211 ~ 20260410"]


def test_format_collection_overview_splits_macro_providers():
    history = pd.DataFrame(
        [
            {
                "created_at": pd.Timestamp("2026-05-12T13:31:49Z"),
                "dataset": "macro_data",
                "provider": "ECOS",
                "requested_start": "201606",
                "requested_end": "202605",
                "status": "LIVE",
                "coverage_complete": False,
                "failed_days": [],
                "failed_codes": {},
                "aborted": False,
                "completion_pct": 99.9,
                "row_count": 1302,
            },
            {
                "created_at": pd.Timestamp("2026-05-12T13:31:51Z"),
                "dataset": "macro_data",
                "provider": "KOSIS",
                "requested_start": "201606",
                "requested_end": "202605",
                "status": "LIVE",
                "coverage_complete": False,
                "failed_days": [],
                "failed_codes": {},
                "aborted": False,
                "completion_pct": 81.0,
                "row_count": 475,
            },
        ]
    )

    rows = tabs._format_collection_overview_rows(
        statuses={"macro_data": {"status": "LIVE", "provider": "KOSIS"}},
        history=history,
        bounds={
            "macro_data:ECOS": {"min_period_month": "20160531", "max_period_month": "20260531", "row_count": 1302},
            "macro_data:KOSIS": {"min_period_month": "20160331", "max_period_month": "20260430", "row_count": 475},
        },
        dataset_order=["macro_data"],
    )

    assert rows["provider"].tolist() == ["KOSIS", "ECOS"]
    assert rows["보유기간"].tolist() == ["2016-03 ~ 2026-04", "2016-05 ~ 2026-05"]
    assert rows["실패/주의"].tolist() == ["요청 범위 일부 미충족 (81.0%)", "요청 범위 일부 미충족 (99.9%)"]


def test_format_collection_overview_shows_partial_rows_when_completion_unknown():
    history = pd.DataFrame(
        [
            {
                "created_at": pd.Timestamp("2026-05-12T13:31:49Z"),
                "dataset": "investor_flow",
                "provider": "PYKRX_UNOFFICIAL",
                "requested_start": "20260401",
                "requested_end": "20260410",
                "status": "LIVE",
                "coverage_complete": False,
                "failed_days": [],
                "failed_codes": {},
                "aborted": False,
                "completion_pct": float("nan"),
                "row_count": 20,
            },
        ]
    )

    rows = tabs._format_collection_overview_rows(
        statuses={"investor_flow": {"status": "LIVE", "provider": "PYKRX_UNOFFICIAL"}},
        history=history,
        bounds={"investor_flow": {"min_trade_date": "20260401", "max_trade_date": "20260410", "row_count": 20}},
        dataset_order=["investor_flow"],
    )

    assert rows["실패/주의"].tolist() == ["부분 수집 데이터 있음 (20건)"]


def test_format_collection_overview_formats_theme_taxonomy_metadata():
    history = pd.DataFrame(
        [
            {
                "created_at": pd.Timestamp("2026-05-17T10:00:00Z"),
                "dataset": "theme_taxonomy",
                "provider": "THEME_TAXONOMY",
                "requested_start": "20260517",
                "requested_end": "20260517",
                "status": "LIVE",
                "coverage_complete": True,
                "failed_days": [],
                "failed_codes": {},
                "aborted": False,
                "completion_pct": 100.0,
                "row_count": 11,
            },
        ]
    )

    rows = tabs._format_collection_overview_rows(
        statuses={"theme_taxonomy": {"status": "LIVE", "provider": "THEME_TAXONOMY"}},
        history=history,
        bounds={
            "theme_taxonomy": {
                "row_count": 11,
                "taxonomy_version": 2,
                "last_verified_at": "2026-05-17",
                "verification_status": "verified",
            }
        },
        dataset_order=["theme_taxonomy"],
    )

    assert rows["데이터"].tolist() == ["테마분류"]
    assert rows["보유기간"].tolist() == ["v2 · 검증 2026-05-17 · verified"]
    assert rows["최근 요청"].tolist() == ["2026-05-17 ~ 2026-05-17"]
    assert rows["실패/주의"].tolist() == ["없음"]


def test_render_monitoring_tab_shows_dataset_sample_history(monkeypatch):
    dataframe_payloads: list[pd.DataFrame] = []
    header_titles: list[str] = []

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    rows = []
    providers = {
        "market_prices": "OPENAPI",
        "macro_data": "ECOS",
        "investor_flow": "KRX_UNOFFICIAL",
        "theme_taxonomy": "THEME_TAXONOMY",
    }
    for dataset in ("market_prices", "macro_data", "investor_flow", "theme_taxonomy"):
        for idx in range(12):
            if idx < 2:
                continue
            rows.append(
                {
                    "created_at": pd.Timestamp("2026-05-05T00:00:00Z") + pd.Timedelta(hours=idx),
                    "dataset": dataset,
                    "reason": "manual_refresh",
                    "provider": providers[dataset],
                    "requested_start": "20260501",
                    "requested_end": "20260505",
                    "status": "LIVE" if idx != 1 else "CACHED",
                    "coverage_complete": idx != 1,
                    "aborted": idx == 1,
                    "abort_reason": "transport timeout" if idx == 1 else "",
                    "failed_days": ["20260502"] if idx == 1 else [],
                    "failed_codes": {"1001": "empty close"} if idx == 1 else {},
                    "predicted_requests": 4,
                    "processed_requests": 3,
                    "row_count": 10 + idx,
                    "completion_pct": 75.0,
                    "sample_bucket": "latest",
                }
            )
    history = pd.DataFrame(rows)

    monkeypatch.setattr(
        tabs,
        "render_panel_header",
        lambda **kwargs: header_titles.append(str(kwargs.get("title", ""))),
    )
    monkeypatch.setattr(
        tabs,
        "_cached_monitoring_data",
        lambda market_id: {
            "statuses": {
                "market_prices": {
                    "status": "LIVE",
                    "provider": "OPENAPI",
                    "watermark_key": "20260505",
                    "coverage_complete": True,
                    "predicted_requests": 1,
                    "processed_requests": 1,
                    "failed_days": [],
                    "failed_codes": {},
                    "aborted": False,
                },
                "macro_data": {"status": "LIVE", "provider": "ECOS", "coverage_complete": True},
                "investor_flow": {"status": "LIVE", "provider": "KRX_UNOFFICIAL", "coverage_complete": True},
                "theme_taxonomy": {"status": "LIVE", "provider": "THEME_TAXONOMY", "coverage_complete": True},
            },
            "history": history,
            "dataset_order": ["market_prices", "macro_data", "investor_flow", "theme_taxonomy"],
        },
    )
    monkeypatch.setattr(tabs.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "error", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "dataframe", lambda frame, **kwargs: dataframe_payloads.append(frame))

    tabs.render_monitoring_tab(tab=_DummyTab(), market_id="KR", ui_locale="ko")

    assert "최근 수집 실행 로그" in header_titles
    assert {"시장데이터", "매크로데이터", "수급데이터", "테마분류"}.issubset(set(header_titles))
    assert len(dataframe_payloads) == 5
    expected_columns = [
        "데이터",
        "상태",
        "마지막 갱신",
        "보유기간",
        "최근 요청",
        "실패/주의",
        "provider",
        "저장행수",
    ]
    overview_table = dataframe_payloads[0]
    assert overview_table.columns.tolist() == expected_columns
    assert overview_table["데이터"].tolist() == ["시장데이터", "매크로데이터", "수급데이터", "테마분류"]
    assert overview_table["최근 요청"].tolist()[0] == "2026-05-01 ~ 2026-05-05"
    assert overview_table["실패/주의"].tolist() == ["없음", "없음", "없음", "없음"]
    market_sample = dataframe_payloads[1]
    assert market_sample.columns.tolist() == [
        "수집일시",
        "요청범위",
        "상태",
        "커버리지",
        "중단",
        "오류요약",
        "완료율(%)",
        "provider",
        "저장행수",
    ]
    assert len(market_sample) == 10
    assert pd.to_datetime(market_sample["수집일시"]).tolist() == sorted(
        pd.to_datetime(market_sample["수집일시"]).tolist(),
        reverse=True,
    )


def test_cached_monitoring_data_reads_manual_refresh_history(monkeypatch):
    calls: list[dict[str, object]] = []
    warehouse_mod = importlib.import_module("src.data_sources.warehouse")

    monkeypatch.setattr(warehouse_mod, "read_dataset_status", lambda dataset, market: {})
    monkeypatch.setattr(warehouse_mod, "read_dataset_data_bounds", lambda dataset, market: {})

    def _fake_history(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(warehouse_mod, "read_collection_run_history", _fake_history)
    tabs._cached_monitoring_data.clear()

    tabs._cached_monitoring_data("KR")

    assert calls == [
        {
            "market": "KR",
            "reasons": ("manual_refresh", "sync_warehouse", "bootstrap_warehouse"),
            "sample_per_dataset": True,
            "sample_size": 10,
        }
    ]


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


def test_render_screening_tab_initial_render_uses_cache_only_loaders(monkeypatch):
    load_kwargs: list[dict[str, object]] = []
    etf_kwargs: list[dict[str, object]] = []

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

    def _load_screened_stocks(**kwargs):
        load_kwargs.append(kwargs)
        return (
            "CACHED",
            [
                {
                    "ticker": "005930",
                    "name": "삼성전자",
                    "sector_name": "KRX 반도체",
                    "rs": 1.2,
                    "rsi": 58.0,
                    "rs_strong": True,
                    "trend_ok": True,
                    "momentum_ok": True,
                    "ret_1m": 5.0,
                    "ret_3m": 11.0,
                    "alerts": "",
                }
            ],
        )

    def _load_representative_etf_context(**kwargs):
        etf_kwargs.append(kwargs)
        return (
            "CACHED",
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
        )

    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(screening_mod, "load_screened_stocks", _load_screened_stocks)
    monkeypatch.setattr(screening_mod, "load_representative_etf_context", _load_representative_etf_context)
    monkeypatch.setattr(tabs.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(tabs.st, "toggle", lambda *args, **kwargs: True)
    monkeypatch.setattr(tabs.st, "columns", lambda spec: [_DummyContainer() for _ in range(len(spec))])
    monkeypatch.setattr(tabs.st, "spinner", lambda *args, **kwargs: _DummyContainer())
    monkeypatch.setattr(tabs.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "download_button", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "markdown", lambda *args, **kwargs: None)

    tabs.render_screening_tab(
        tab=_DummyTab(),
        signals=[SimpleNamespace(index_code="5044", sector_name="KRX 반도체", action="Strong Buy")],
        settings={},
        benchmark_code="1001",
        etf_map={"5044": [{"code": "396500", "name": "TIGER 반도체TOP10"}]},
    )

    assert load_kwargs[-1]["force_refresh"] is False
    assert load_kwargs[-1]["allow_live_fetch"] is False
    assert etf_kwargs[-1]["force_refresh"] is False
    assert etf_kwargs[-1]["allow_live_fetch"] is False


def test_render_screening_tab_refresh_allows_live_loaders(monkeypatch):
    load_kwargs: list[dict[str, object]] = []
    etf_kwargs: list[dict[str, object]] = []
    progress_labels: list[str] = []
    progress_values: list[int] = []

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

    class _DummyPlaceholder:
        def caption(self, text):
            progress_labels.append(str(text))

        def progress(self, value, text=None):
            progress_values.append(int(value))
            if text:
                progress_labels.append(str(text))
            return self

        def empty(self):
            progress_labels.append("empty")

    stock_row = {
        "ticker": "005930",
        "name": "삼성전자",
        "sector_name": "KRX 반도체",
        "rs": 1.2,
        "rsi": 58.0,
        "rs_strong": True,
        "trend_ok": True,
        "momentum_ok": True,
        "ret_1m": 5.0,
        "ret_3m": 11.0,
        "alerts": "",
    }
    etf_row = {
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

    def _load_screened_stocks(**kwargs):
        load_kwargs.append(kwargs)
        progress_callback = kwargs.get("progress_callback")
        assert callable(progress_callback)
        progress_callback({"stage": "start", "current": 0, "total": 2})
        progress_callback(
            {
                "stage": "ticker",
                "current": 1,
                "total": 2,
                "ticker": "005930",
                "sector_name": "KRX 반도체",
            }
        )
        progress_callback({"stage": "done", "current": 2, "total": 2})
        return ("LIVE", [stock_row])

    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(screening_mod, "load_screened_stocks", _load_screened_stocks)
    monkeypatch.setattr(
        screening_mod,
        "load_representative_etf_context",
        lambda **kwargs: (etf_kwargs.append(kwargs) or ("LIVE", [etf_row])),
    )
    monkeypatch.setattr(tabs.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "button", lambda *args, **kwargs: True)
    monkeypatch.setattr(tabs.st, "toggle", lambda *args, **kwargs: True)
    monkeypatch.setattr(tabs.st, "columns", lambda spec: [_DummyContainer() for _ in range(len(spec))])
    monkeypatch.setattr(tabs.st, "spinner", lambda *args, **kwargs: _DummyContainer())
    monkeypatch.setattr(tabs.st, "empty", lambda: _DummyPlaceholder())
    monkeypatch.setattr(tabs.st, "progress", lambda value, text=None: _DummyPlaceholder().progress(value, text=text))
    monkeypatch.setattr(tabs.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "download_button", lambda *args, **kwargs: None)
    monkeypatch.setattr(tabs.st, "markdown", lambda *args, **kwargs: None)

    tabs.render_screening_tab(
        tab=_DummyTab(),
        signals=[SimpleNamespace(index_code="5044", sector_name="KRX 반도체", action="Strong Buy")],
        settings={},
        benchmark_code="1001",
        etf_map={"5044": [{"code": "396500", "name": "TIGER 반도체TOP10"}]},
    )

    assert load_kwargs[-1]["force_refresh"] is True
    assert load_kwargs[-1]["allow_live_fetch"] is True
    assert load_kwargs[-1]["progress_callback"] is not None
    assert etf_kwargs[-1]["force_refresh"] is True
    assert etf_kwargs[-1]["allow_live_fetch"] is True
    assert any("갱신 대상 총 2종목 중 1번째 처리 중: 005930 · KRX 반도체" in label for label in progress_labels)
    assert 50 in progress_values
    assert 100 in progress_values
