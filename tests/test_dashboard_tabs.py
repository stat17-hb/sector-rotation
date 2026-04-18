from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.dashboard import tabs
import src.data_sources.krx_stock_screening as screening_mod
import src.ui.components as ui_components


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
        ui_locale="ko",
    )

    assert "hero" not in calls
    assert "status" not in calls
    assert "top_picks" in calls
    assert "action_summary" in calls
    assert "panel:상위 추천" in calls
    assert "panel:액션 분포" in calls


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

    assert any("최종 투자판단에는 반영되지 않았습니다" in text for text in warnings)
    assert any("의견 변화 표를 숨기고 참여 주체 비교표와 raw snapshot만 표시합니다" in text for text in infos)
    assert len(dataframes) == 2
    assert list(dataframes[0].columns) == [
        "Sector",
        "Flow state",
        "Flow score",
        "Foreign",
        "Institutional",
        "Retail",
    ]
    assert list(dataframes[1].columns) == ["Sector", "Investor", "Latest ratio", "net_buy_amount"]


def test_render_investor_flow_tab_shows_glance_matrix_before_raw_snapshot(monkeypatch):
    dataframes: list[pd.DataFrame] = []

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(tabs.st, "warning", lambda *_args, **_kwargs: None)
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
        "Flow score",
        "Foreign",
        "Institutional",
        "Retail",
        "Action change",
    ]
    assert list(dataframes[1].columns) == ["Sector", "Investor", "Latest ratio", "net_buy_amount"]


def test_render_investor_flow_tab_uses_raw_snapshot_fallback_when_signal_flow_is_missing(monkeypatch):
    dataframes: list[pd.DataFrame] = []
    infos: list[str] = []

    class _DummyTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(tabs, "render_panel_header", lambda **kwargs: None)
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
        "Flow score",
        "Foreign",
        "Institutional",
        "Retail",
    ]
    assert dataframes[0].iloc[0]["Foreign"] == "+20.00%"


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
    assert "오류 섹터 1건" in warning_texts
    assert "오류 종목 1건" in warning_texts


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
