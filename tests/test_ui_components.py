"""UI component rendering tests for dashboard visuals and layout helpers."""
from __future__ import annotations

import pandas as pd
import pytest
import src.dashboard.data as dashboard_data_module
import src.ui.panels as panels_module

from src.dashboard.analysis import (
    build_cycle_segments as _build_cycle_segments,
    build_heatmap_display as _build_heatmap_display,
    build_monthly_return_views as _build_monthly_return_views,
    build_monthly_sector_returns as _build_monthly_sector_returns,
    build_prices_wide as _build_prices_wide,
    extract_heatmap_selection as _extract_heatmap_selection,
    filter_monthly_frame_for_analysis as _filter_monthly_frame_for_analysis,
)
from src.signals.flow import summarize_sector_investor_flow
from src.signals.matrix import SectorSignal
from src.ui.copy import ALL_ACTION_KEY
from src.ui.components import (
    HEATMAP_PALETTE_OPTIONS,
    build_investor_flow_glance_rows,
    build_investor_flow_snapshot_rows,
    build_sector_detail_figure,
    build_sector_strength_heatmap,
    describe_signal_decision,
    format_heatmap_palette_label,
    get_analysis_heatmap_colorscale,
    infer_range_preset,
    normalize_heatmap_palette,
    normalize_range_preset,
    render_action_summary,
    render_analysis_toolbar,
    render_cycle_timeline_panel,
    render_decision_hero,
    render_investor_decision_boards,
    render_page_header,
    render_panel_header,
    render_research_page_frame,
    render_rs_momentum_bar,
    render_rs_scatter,
    render_signal_table,
    render_sector_detail_panel,
    render_stock_lookup_control,
    render_status_card_row,
    render_status_strip,
    render_progress_panel,
    render_top_bar_filters,
    render_top_picks_table,
    resolve_range_from_preset,
    signal_display_sort_key,
)
from src.ui.styles import ACTION_COLORS, BLUE, DARK_GREY, GREY, get_action_colors


def _signal(
    code: str,
    action: str,
    rs: float,
    rs_ma: float,
    *,
    macro_regime: str = "Recovery",
    macro_fit: bool = True,
    macro_context_regime: str = "Recovery",
    action_policy: str = "",
    taxonomy_kind: str = "",
    taxonomy_label: str = "",
    alerts: list[str] | None = None,
    returns: dict[str, float] | None = None,
    is_provisional: bool = False,
) -> SectorSignal:
    return SectorSignal(
        index_code=code,
        sector_name=f"Sector {code}",
        macro_regime=macro_regime,
        macro_fit=macro_fit,
        rs=rs,
        rs_ma=rs_ma,
        rs_strong=False,
        trend_ok=rs >= rs_ma,
        momentum_strong=rs >= rs_ma,
        momentum_core_pass=rs >= rs_ma,
        rsi_d=50.0,
        rsi_w=50.0,
        action=action,
        alerts=list(alerts or []),
        returns=returns or {"1M": 0.05, "3M": 0.10},
        volatility_20d=0.12,
        mdd_3m=-0.08,
        asof_date="2024-01-31",
        is_provisional=is_provisional,
        macro_context_regime=macro_context_regime,
        action_policy=action_policy,
        taxonomy_kind=taxonomy_kind,
        taxonomy_label=taxonomy_label,
    )


class _DummyBlock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_render_page_header_renders_shell_markup(monkeypatch):
    markdown_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: markdown_calls.append(text))

    render_page_header(
        title="Korea Sector Rotation",
        description="Macro regime and sector signal cockpit.",
        pills=[
            {"label": "Regime", "value": "Recovery", "tone": "success"},
            {"label": "Market", "value": "LIVE", "tone": "success"},
        ],
    )

    assert markdown_calls
    assert "page-shell" in markdown_calls[0]
    assert "page-shell__grid" in markdown_calls[0]
    assert "page-shell__meta" in markdown_calls[0]
    assert "Korea Sector Rotation" in markdown_calls[0]
    assert "page-shell__pill" in markdown_calls[0]


def test_render_page_header_avoids_forbidden_brokerage_or_live_claims(monkeypatch):
    markdown_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: markdown_calls.append(text))

    render_page_header(
        title="섹터 로테이션",
        description="규칙 기반 의사결정 지원",
        pills=[
            {"label": "시장", "value": "KR", "tone": "info"},
            {"label": "가격", "value": "캐시", "tone": "success"},
        ],
    )

    markup = markdown_calls[0]
    assert "시장 컨텍스트" in markup
    for phrase in (
        "live overview",
        "Watchlist",
        "order",
        "account",
        "brokerage",
        "risk-stock",
        "거래량 순위",
        "거래대금 순위",
    ):
        assert phrase not in markup


def test_render_status_strip_renders_markup_and_details(monkeypatch):
    markdown_calls: list[str] = []
    write_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: markdown_calls.append(text))
    monkeypatch.setattr("src.ui.components.st.expander", lambda *_args, **_kwargs: _DummyBlock())
    monkeypatch.setattr("src.ui.components.st.write", lambda text: write_calls.append(text))

    render_status_strip(
        {
            "level": "warning",
            "title": "Cache fallback",
            "message": "Using the local warehouse snapshot.",
            "details": ["KRX: HTTP_ERROR", "ECOS: OK"],
        }
    )

    assert markdown_calls
    assert "status-strip" in markdown_calls[0]
    assert "주의" in markdown_calls[0]
    assert "Cache fallback" in markdown_calls[0]
    assert "2개 상세" in markdown_calls[0]
    assert any("KRX: HTTP_ERROR" in call for call in write_calls)


def test_render_progress_panel_renders_progress_and_terminal_detail(monkeypatch):
    calls: list[tuple[str, object]] = []

    class _DummyHost:
        def empty(self):
            calls.append(("empty", None))

        def container(self):
            return _DummyBlock()

    monkeypatch.setattr("src.ui.components.st.caption", lambda text: calls.append(("caption", text)))
    monkeypatch.setattr("src.ui.components.st.progress", lambda value: calls.append(("progress", value)))
    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: calls.append(("markdown", text)))
    monkeypatch.setattr("src.ui.components.st.success", lambda text: calls.append(("success", text)))

    render_progress_panel(
        _DummyHost(),
        {
            "task": "투자자수급 갱신",
            "phase": "재시도 수집 중",
            "pct": 78,
            "detail": "234/300 requests",
            "status": "complete",
        },
    )

    assert ("caption", "투자자수급 갱신 · 완료 · 78%") in calls
    assert ("progress", 78) in calls
    assert ("markdown", "**재시도 수집 중**") in calls
    assert ("success", "234/300 requests") in calls


def test_render_panel_header_renders_panel_markup(monkeypatch):
    markdown_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: markdown_calls.append(text))

    render_panel_header(
        eyebrow="Momentum map",
        title="RS scatter",
        description="Compare RS and RS MA.",
        badge="Live",
    )

    assert markdown_calls
    assert "panel-header" in markdown_calls[0]
    assert "RS scatter" in markdown_calls[0]
    assert "panel-header__badge" in markdown_calls[0]


def test_render_research_page_frame_renders_consistent_non_dashboard_shell(monkeypatch):
    markdown_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: markdown_calls.append(text))

    render_research_page_frame(
        page_key="signals",
        eyebrow="Signal Review",
        title="섹터 모멘텀 필터 보드",
        description="필터 조건과 보유 범위를 같은 기준으로 검토합니다.",
        summary_items=[
            {"label": "유니버스", "value": "12개 섹터"},
            {"label": "필터", "value": "전체"},
        ],
    )

    assert markdown_calls
    markup = markdown_calls[0]
    assert "research-page-frame" in markup
    assert 'data-page="signals"' in markup
    assert "research-page-frame__summary" in markup
    assert "섹터 모멘텀 필터 보드" in markup
    assert "12개 섹터" in markup


def test_render_rs_scatter_filters_nan_points():
    signals = [
        _signal("A", "Watch", 1.10, 1.00),
        _signal("B", "Strong Buy", float("nan"), 1.00),
        _signal("C", "N/A", 1.20, 1.10),
    ]

    fig = render_rs_scatter(signals, theme_mode="dark")
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [1.10]
    assert list(fig.data[0].y) == [1.00]
    assert list(fig.data[0].marker.color) == [ACTION_COLORS["Watch"]]


def test_render_rs_scatter_empty_annotation_when_no_valid_points():
    signals = [
        _signal("A", "N/A", 1.10, 1.00),
        _signal("B", "Watch", float("nan"), 1.00),
    ]

    fig = render_rs_scatter(signals, theme_mode="dark")
    assert len(fig.data) == 0
    annotations = list(fig.layout.annotations or [])
    assert annotations
    assert "RS / RS MA" in annotations[0].text


def test_render_rs_scatter_allows_custom_height_and_margin():
    signals = [
        _signal("A", "Watch", 1.08, 1.02),
        _signal("B", "Hold", 0.96, 0.99),
    ]

    fig = render_rs_scatter(
        signals,
        height=540,
        margin={"l": 20, "r": 18, "t": 22, "b": 24},
    )
    assert fig.layout.height == 540
    assert fig.layout.margin.l == 20
    assert fig.layout.margin.r == 18
    assert fig.layout.margin.t == 22
    assert fig.layout.margin.b == 24


def test_render_rs_scatter_diagnostic_only_title_and_legacy_trend_hover():
    signal = _signal("A", "Watch", 1.08, 1.02)
    signal.momentum_method = "hybrid_return_rank_v1"
    signal.trend_ok = False
    signal.legacy_trend_ok = True

    fig = render_rs_scatter([signal], diagnostic_only=True)

    assert fig.layout.title.text.startswith("Legacy RS Diagnostic")
    assert "추세: 양호" in str(fig.data[0].hovertext[0])


def test_render_rs_momentum_bar_returns_empty_when_no_valid_data():
    signals = [
        _signal("A", "N/A", 1.10, 1.00),
        _signal("B", "Watch", float("nan"), 1.00),
        _signal("C", "Hold", 1.00, float("nan")),
        _signal("D", "Hold", 1.00, 0.00),
    ]

    fig = render_rs_momentum_bar(signals, theme_mode="dark")
    assert len(fig.data) == 0


def test_render_rs_momentum_bar_valid_data_has_percent_suffix():
    signals = [
        _signal("A", "Watch", 1.10, 1.00),
        _signal("B", "Hold", 0.95, 1.00),
    ]

    fig = render_rs_momentum_bar(signals, theme_mode="dark")
    assert len(fig.data) == 1
    assert fig.layout.xaxis.ticksuffix == "%"


def test_render_rs_momentum_bar_diagnostic_title():
    signals = [
        _signal("A", "Watch", 1.10, 1.00),
        _signal("B", "Hold", 0.95, 1.00),
    ]

    fig = render_rs_momentum_bar(signals, theme_mode="dark", diagnostic_only=True)

    assert fig.layout.title.text.startswith("Legacy RS Diagnostic")


def test_action_colors_watch_hold_mapping():
    assert ACTION_COLORS["Watch"] == BLUE
    assert ACTION_COLORS["Hold"] == GREY
    assert ACTION_COLORS["N/A"] == DARK_GREY


def test_render_action_summary_renders_cards_and_bar(monkeypatch):
    signals = [
        _signal("A", "Strong Buy", 1.10, 1.00),
        _signal("B", "Watch", 1.05, 1.00),
        _signal("C", "Watch", 1.01, 1.00),
        _signal("D", "Hold", 0.98, 1.00),
        _signal("E", "Avoid", 0.94, 1.00),
        _signal("F", "N/A", 1.00, 1.00),
    ]

    markdown_calls: list[str] = []
    chart_calls: list[tuple[object, dict]] = []

    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: markdown_calls.append(text))
    monkeypatch.setattr(
        "src.ui.components.st.plotly_chart",
        lambda fig, **kwargs: chart_calls.append((fig, kwargs)),
    )

    render_action_summary(signals, theme_mode="dark")

    assert any("summary-kpi-grid" in call for call in markdown_calls)
    assert len(chart_calls) == 1
    fig, kwargs = chart_calls[0]
    assert kwargs.get("width") == "stretch"
    assert list(fig.data[0].x) == ["Strong Buy", "Watch", "Hold", "Avoid", "N/A"]
    assert list(fig.data[0].y) == [1, 2, 1, 1, 1]


def test_render_action_summary_handles_empty_signal_list(monkeypatch):
    info_calls: list[str] = []
    chart_calls: list[object] = []

    monkeypatch.setattr("src.ui.components.st.info", lambda text: info_calls.append(text))
    monkeypatch.setattr(
        "src.ui.components.st.plotly_chart",
        lambda fig, **kwargs: chart_calls.append(fig),
    )

    render_action_summary([], theme_mode="dark")

    assert info_calls == ["신호 데이터가 없습니다."]
    assert not chart_calls


def test_render_rs_scatter_uses_light_theme_action_palette():
    signals = [_signal("A", "Watch", 1.10, 1.00)]

    fig = render_rs_scatter(signals, theme_mode="light")
    assert len(fig.data) == 1
    assert list(fig.data[0].marker.color) == [get_action_colors("light")["Watch"]]


def test_render_top_bar_filters_returns_selected_state(monkeypatch):
    session_state: dict[str, object] = {}
    markdown_calls: list[str] = []
    segmented_kwargs: list[dict[str, object]] = []

    monkeypatch.setattr("src.ui.components.st.session_state", session_state)
    monkeypatch.setattr("src.ui.components.st.container", lambda **_: _DummyBlock())
    monkeypatch.setattr(
        "src.ui.components.st.columns",
        lambda spec: [_DummyBlock() for _ in range(len(spec))],
    )
    monkeypatch.setattr(
        "src.ui.components.st.selectbox",
        lambda _label, options, key, **kwargs: session_state.__setitem__(key, options[1]),
    )
    monkeypatch.setattr(
        "src.ui.components.st.toggle",
        lambda _label, key: session_state.__setitem__(key, True),
    )
    monkeypatch.setattr(
        "src.ui.components.st.segmented_control",
        lambda **kwargs: (
            segmented_kwargs.append(dict(kwargs)),
            session_state.__setitem__(str(kwargs["key"]), list(kwargs["options"])[1]),
        )[-1],
    )
    monkeypatch.setattr(
        "src.ui.components.st.markdown",
        lambda text, **_: markdown_calls.append(text),
    )
    monkeypatch.setattr("src.ui.components.st.caption", lambda text: None)

    action, regime_only, position_mode, alerted_only = render_top_bar_filters(
        current_regime="Recovery",
        action_options=[ALL_ACTION_KEY, "Strong Buy", "Watch"],
        is_mobile=False,
    )

    assert action == "Strong Buy"
    assert regime_only is True
    assert position_mode == "held"
    assert alerted_only is True
    assert segmented_kwargs
    assert "default" not in segmented_kwargs[0]
    assert any("command-bar" in call for call in markdown_calls)
    assert any("filter-chip-row" in call for call in markdown_calls)
    assert any("Recovery" in call for call in markdown_calls)
    assert any("하단 상세 뷰 필터" in call or "Downstream detail filters" in call for call in markdown_calls)
    assert any("아래 요약·차트·테이블·탭의 연구 뷰만 정제합니다." in call for call in markdown_calls)


def test_render_decision_hero_renders_regime_and_provisional_badge(monkeypatch):
    markdown_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: markdown_calls.append(text))

    render_decision_hero(
        regime="Recovery",
        regime_is_confirmed=False,
        growth_val=101.25,
        inflation_val=2.1,
        export_growth_val=7.5,
        fx_change=1.4,
        is_provisional=True,
        theme_mode="light",
    )

    assert markdown_calls
    assert "decision-hero" in markdown_calls[0]
    assert "Recovery" in markdown_calls[0]
    assert "잠정 매크로 데이터 포함" in markdown_calls[0]
    assert "규칙 기반 판단" in markdown_calls[0]
    assert "confirmed_regime 기준" in markdown_calls[0]
    assert "역사 문서 분리" in markdown_calls[0]
    assert "수출 전년비" in markdown_calls[0]
    assert "7.50%" in markdown_calls[0]

    markdown_calls.clear()
    render_decision_hero(
        regime="Recovery",
        regime_is_confirmed=False,
        growth_val=101.25,
        inflation_val=2.1,
        export_growth_val=7.5,
        fx_change=1.4,
        is_provisional=True,
        theme_mode="light",
        locale="en",
    )

    assert "Includes provisional macro prints" in markdown_calls[0]
    assert "Rules-based heuristic" in markdown_calls[0]
    assert "confirmed_regime primary" in markdown_calls[0]
    assert "Historical docs separated" in markdown_calls[0]
    assert "Exports YoY" in markdown_calls[0]


def test_render_decision_hero_renders_us_trade_copy_without_kr_export_label(monkeypatch):
    markdown_calls: list[str] = []
    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: markdown_calls.append(text))

    render_decision_hero(
        regime="Expansion",
        regime_is_confirmed=True,
        growth_val=2.4,
        inflation_val=3.1,
        export_growth_val=7.5,
        trade_indicators={"exports_yoy": 4.2, "imports_yoy": -1.5},
        fx_change=0.8,
        theme_mode="light",
        locale="en",
    )

    assert markdown_calls
    assert "US Exports YoY" in markdown_calls[0]
    assert "US Imports YoY" in markdown_calls[0]
    assert "4.20%" in markdown_calls[0]
    assert "-1.50%" in markdown_calls[0]
    assert "수출 전년비" not in markdown_calls[0]


def test_overview_market_cards_include_kr_export_growth_status():
    cards = panels_module._build_overview_market_cards(
        prices_wide=pd.DataFrame({"KOSPI": [100.0, 101.0]}),
        benchmark_label="KOSPI",
        reference_index_labels=[],
        signals=[],
        current_regime="Recovery",
        price_status="LIVE",
        macro_status="CACHED",
        export_growth_val=7.5,
    )

    rendered = "".join(cards)
    assert "수출 전년비" in rendered
    assert "+7.5%" in rendered


def test_overview_market_cards_show_latest_daily_return_basis():
    prices = pd.DataFrame(
        {"KOSPI": [101.0, 100.0, 98.0]},
        index=pd.to_datetime(["2026-05-11", "2026-05-10", "2026-05-12"]),
    )

    cards = panels_module._build_overview_market_cards(
        prices_wide=prices,
        benchmark_label="KOSPI",
        reference_index_labels=[],
        signals=[],
        current_regime="Recovery",
        price_status="LIVE",
        macro_status="CACHED",
    )

    rendered = "".join(cards)
    assert "1D -2.97%" in rendered
    assert "98.00" in rendered


def test_overview_market_cards_do_not_promote_sector_signals_without_reference_indexes():
    cards = panels_module._build_overview_market_cards(
        prices_wide=pd.DataFrame(
            {
                "KOSPI": [100.0, 101.0],
                "KRX 증권": [200.0, 210.0],
            }
        ),
        benchmark_label="KOSPI",
        reference_index_labels=[],
        signals=[_signal("KRX 증권", "Strong Buy", 1.2, 1.0)],
        current_regime="Recovery",
        price_status="CACHED",
        macro_status="CACHED",
    )

    rendered = "".join(cards)
    assert "KOSPI" in rendered
    assert "KRX 증권" not in rendered


def test_overview_market_cards_include_us_trade_status_without_export_growth():
    cards = panels_module._build_overview_market_cards(
        prices_wide=pd.DataFrame({"S&P 500": [100.0, 101.0]}),
        benchmark_label="S&P 500",
        reference_index_labels=[],
        signals=[],
        current_regime="Expansion",
        price_status="LIVE",
        macro_status="CACHED",
        trade_indicators={"exports_yoy": 4.2, "imports_yoy": -1.5},
    )

    rendered = "".join(cards)
    assert "미국 수출입" in rendered
    assert "수출 +4.2%" in rendered
    assert "수입 -1.5%" in rendered
    assert "수출 전년비" not in rendered


def test_overview_market_cards_show_trade_placeholder_when_configured_without_data():
    cards = panels_module._build_overview_market_cards(
        prices_wide=pd.DataFrame({"S&P 500": [100.0, 101.0]}),
        benchmark_label="S&P 500",
        reference_index_labels=[],
        signals=[],
        current_regime="Expansion",
        price_status="LIVE",
        macro_status="CACHED",
        trade_indicators={},
        has_trade_indicators=True,
    )

    rendered = "".join(cards)
    assert "미국 수출입" in rendered
    assert "데이터 없음" in rendered
    assert "수출 전년비" not in rendered


def test_render_sector_trade_lens_labels_aggregate_proxy(monkeypatch):
    markdown_calls: list[str] = []
    monkeypatch.setattr("src.ui.panels.st.markdown", lambda text, **_: markdown_calls.append(text))

    panels_module._render_sector_trade_lens(
        [
            {
                "sector": "Technology",
                "exposure_label": "수출 민감",
                "basis": "글로벌 IT/장비 수요",
                "driver": "수출 YoY",
                "value": 4.2,
                "status": "교역 순풍",
                "tone": "positive",
            }
        ]
    )

    assert markdown_calls
    rendered = markdown_calls[0]
    assert "미국 수출입 섹터 렌즈" in rendered
    assert "총량 proxy" in rendered
    assert "섹터별 직접 무역 데이터가 아니며" in rendered
    assert "Technology" in rendered
    assert "교역 순풍" in rendered
    assert "+4.2%" in rendered


def test_render_sector_trade_lens_omits_direct_limitation_na_cards(monkeypatch):
    markdown_calls: list[str] = []
    monkeypatch.setattr("src.ui.panels.st.markdown", lambda text, **_: markdown_calls.append(text))

    panels_module._render_sector_trade_lens(
        [
            {
                "sector": "Technology",
                "exposure_label": "수출 민감",
                "basis": "글로벌 IT/장비 수요",
                "driver": "수출 YoY",
                "value": 4.2,
                "status": "교역 순풍",
                "tone": "positive",
            },
            {
                "sector": "Financials",
                "exposure_label": "낮음",
                "basis": "교역 직접성 낮음",
                "driver": "직접성 낮음",
                "value": None,
                "status": "직접 해석 제한",
                "tone": "neutral",
            },
        ]
    )

    assert markdown_calls
    rendered = markdown_calls[0]
    assert "Technology" in rendered
    assert "Financials" not in rendered
    assert "N/A" not in rendered
    assert "직접 해석 제한 섹터 1개 제외" in rendered


def test_render_sector_trade_lens_skips_panel_when_only_na_rows(monkeypatch):
    markdown_calls: list[str] = []
    monkeypatch.setattr("src.ui.panels.st.markdown", lambda text, **_: markdown_calls.append(text))

    panels_module._render_sector_trade_lens(
        [
            {
                "sector": "Financials",
                "value": None,
                "status": "직접 해석 제한",
            }
        ]
    )

    assert markdown_calls == []


def test_overview_sector_frame_includes_sector_export_basis_when_capability_enabled():
    signals = [
        type(
            "Signal",
            (),
            {
                "sector_name": "KRX 반도체",
                "action": "Buy",
                "returns": {"1M": 0.02, "3M": 0.08},
                "mom_percentile": 84.0,
            },
        )(),
        type(
            "Signal",
            (),
            {
                "sector_name": "KOSPI200 경기소비재",
                "action": "Watch",
                "returns": {"1M": 0.01, "3M": 0.02},
                "mom_percentile": 61.0,
            },
        )(),
    ]

    frame = panels_module._build_overview_sector_frame(
        signals,
        sort_key="모멘텀 점수",
        sector_export_trends={
            "KRX 반도체": 18.4,
            "KOSPI200 경기소비재": 1.2,
        },
    )

    assert list(frame["섹터"]) == ["KRX 반도체", "KOSPI200 경기소비재"]
    assert list(frame["수출 기준"]) == ["반도체 수출", "자동차 수출"]
    assert "수출 YoY" not in frame.columns


def test_render_overview_sector_table_omits_export_yoy_but_keeps_export_basis(monkeypatch):
    markdown_calls: list[str] = []
    monkeypatch.setattr("src.ui.panels.st.markdown", lambda text, **_: markdown_calls.append(text))

    panels_module._render_overview_sector_table(
        pd.DataFrame(
            [
                {
                    "순위": 1,
                    "섹터": "KOSPI200 경기소비재",
                    "수출 기준": "자동차 수출",
                    "모멘텀 점수": 84.0,
                    "상대강도": 4.2,
                    "3M": 8.0,
                }
            ]
        )
    )

    assert markdown_calls
    assert "수출 YoY" not in markdown_calls[0]
    assert "수출 기준: 자동차 수출" in markdown_calls[0]
    assert "+8.00%" in markdown_calls[0]


def test_overview_sector_frame_omits_export_columns_when_capability_disabled():
    signals = [
        type(
            "Signal",
            (),
            {
                "sector_name": "Industrials",
                "action": "Watch",
                "rs": 1.03,
                "rs_ma": 1.00,
                "returns": {"1M": 0.01, "3M": 0.03},
                "mom_percentile": 80.0,
            },
        )()
    ]

    frame = panels_module._build_overview_sector_frame(
        signals,
        sort_key="수출 YoY",
        sector_export_trends={"Industrials": 8.4},
        has_sector_export_indicators=False,
    )

    assert "수출 YoY" not in frame.columns
    assert "수출 기준" not in frame.columns


def test_render_overview_sector_table_omits_export_headers_when_capability_disabled(monkeypatch):
    markdown_calls: list[str] = []
    monkeypatch.setattr("src.ui.panels.st.markdown", lambda text, **_: markdown_calls.append(text))

    panels_module._render_overview_sector_table(
        pd.DataFrame(
            [
                {
                    "순위": 1,
                    "섹터": "Industrials",
                    "모멘텀 점수": 84.0,
                    "상대강도": 4.2,
                    "3M": 8.0,
                }
            ]
        ),
        has_sector_export_indicators=False,
    )

    assert markdown_calls
    assert "수출 YoY" not in markdown_calls[0]
    assert "수출 기준" not in markdown_calls[0]


def test_build_sector_export_trend_figure_renders_monthly_series():
    index = pd.period_range("2025-01", periods=14, freq="M")
    signals = [
        type(
            "Signal",
            (),
            {
                "sector_name": "KRX 반도체",
                "action": "Buy",
                "returns": {"3M": 0.08},
                "mom_percentile": 84.0,
            },
        )()
    ]

    fig = panels_module._build_sector_export_trend_figure(
        sector_export_history={
            "KRX 반도체": pd.Series(range(14), index=index, dtype="float64"),
            "KOSPI200 경기소비재": pd.Series(range(100, 114), index=index, dtype="float64"),
        },
        signals=signals,
        theme_mode="dark",
        window_months=12,
    )

    assert len(fig.data) == 2
    assert fig.data[0].name == "반도체 수출"
    assert fig.data[1].name == "자동차 수출"
    assert list(fig.data[0].x)[0] == pd.Timestamp("2025-03-01")
    assert list(fig.data[0].x)[-1] == pd.Timestamp("2026-02-01")
    assert fig.layout.xaxis.tickformat == "%b\n%Y"
    assert fig.layout.xaxis.dtick == "M1"
    assert fig.layout.hovermode == "x unified"
    assert "반도체 수출" in str(fig.layout.annotations[0].text)
    assert "섹터별 수출 YoY 월별 추이" in str(fig.layout.title.text)


def test_build_overview_trend_figure_labels_line_ends_and_month_ticks():
    dates = pd.date_range("2026-01-02", periods=45, freq="B")
    prices = pd.DataFrame(
        {
            "KOSPI": range(100, 145),
            "KRX 반도체": range(100, 190, 2),
            "KRX 정보기술": range(100, 235, 3),
        },
        index=dates,
    )
    signals = [
        type(
            "Signal",
            (),
            {
                "sector_name": "KRX 반도체",
                "action": "Strong Buy",
                "returns": {"3M": 0.08},
                "mom_percentile": 84.0,
            },
        )(),
        type(
            "Signal",
            (),
            {
                "sector_name": "KRX 정보기술",
                "action": "Watch",
                "returns": {"3M": 0.05},
                "mom_percentile": 74.0,
            },
        )(),
    ]

    fig = panels_module._build_overview_trend_figure(
        prices_wide=prices,
        signals=signals,
        benchmark_label="KOSPI",
        period="3M",
        theme_mode="light",
    )

    assert len(fig.data) == 3
    assert fig.layout.xaxis.tickformat == "%b\n%Y"
    assert fig.layout.xaxis.dtick == "M1"
    assert fig.layout.hovermode == "x unified"
    annotation_text = " ".join(str(annotation.text) for annotation in fig.layout.annotations)
    assert "KOSPI" in annotation_text
    assert "KRX 반도체" in annotation_text
    assert "KRX 정보기술" in annotation_text


def test_render_status_card_row_renders_card_markup(monkeypatch):
    markdown_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: markdown_calls.append(text))

    render_status_card_row(
        current_regime="Recovery",
        regime_is_confirmed=True,
        price_status="LIVE",
        macro_status="CACHED",
        yield_curve_status="Inverted",
    )

    assert markdown_calls
    assert "status-card-grid" in markdown_calls[0]
    assert "수익률 곡선" in markdown_calls[0]
    assert "역전" in markdown_calls[0]

    markdown_calls.clear()
    render_status_card_row(
        current_regime="Recovery",
        regime_is_confirmed=True,
        price_status="LIVE",
        macro_status="CACHED",
        yield_curve_status="Inverted",
        locale="en",
    )

    assert "Yield curve" in markdown_calls[0]
    assert "Inverted" in markdown_calls[0]


def test_describe_signal_decision_changes_by_held_state():
    signal = _signal("A", "Strong Buy", 1.10, 1.00, alerts=["Overheat"])

    held_view = describe_signal_decision(signal, ["Sector A"])
    new_view = describe_signal_decision(signal, [])
    english_view = describe_signal_decision(signal, ["Sector A"], locale="en")

    assert held_view["held"] is True
    assert held_view["decision"] == "추가 검토 후보"
    assert "국면 적합" in held_view["reason"]
    assert held_view["alerts_text"] == "Overheat"
    assert new_view["held"] is False
    assert new_view["decision"] == "신규 검토 후보"
    assert english_view["decision"] == "Add-review candidate"
    assert "Regime fit" in english_view["reason"]
    assert held_view["judgment_structure"] == "기본 모형"
    assert held_view["judgment_confidence"] == "제한적 판단 규칙"


def test_describe_signal_decision_marks_experimental_flow_overlay():
    signal = _signal("A", "Strong Buy", 1.10, 1.00)
    signal.base_action = "Watch"
    signal.flow_adjustment = "upgrade"
    signal.flow_state = "supportive"
    signal.flow_profile = "foreign_lead"

    view = describe_signal_decision(signal, ["Sector A"])

    assert "실험적 수급 보정" in str(view["judgment_structure"])
    assert view["judgment_confidence"] == "실험 보정 포함"


def test_describe_signal_decision_hybrid_uses_raw_percentile_value():
    signal = _signal("A", "Strong Buy", 1.10, 1.00)
    signal.momentum_method = "hybrid_return_rank_v1"
    signal.mom_percentile = 82.0
    signal.mom_raw = 0.12
    signal.mom_rank = 1
    signal.trend_ok = True
    signal.momentum_strong = True

    view = describe_signal_decision(signal, ["Sector A"])

    assert "모멘텀 백분위 82p" in str(view["reason"])
    assert "8200" not in str(view["reason"])


def test_signal_display_sort_key_uses_hybrid_rank_then_raw():
    held_sectors: list[str] = []
    signal_a = _signal("A", "Strong Buy", 1.10, 1.00)
    signal_b = _signal("B", "Strong Buy", 1.10, 1.00)
    signal_c = _signal("C", "Strong Buy", 1.10, 1.00)
    for signal in (signal_a, signal_b, signal_c):
        signal.momentum_method = "hybrid_return_rank_v1"
    signal_a.mom_rank = 2
    signal_a.mom_raw = 0.12
    signal_b.mom_rank = 1
    signal_b.mom_raw = 0.08
    signal_c.mom_rank = 2
    signal_c.mom_raw = 0.25

    ordered = sorted([signal_a, signal_b, signal_c], key=lambda item: signal_display_sort_key(item, held_sectors))

    assert [signal.sector_name for signal in ordered] == ["Sector B", "Sector C", "Sector A"]


def test_render_investor_flow_summary_marks_reference_only_preview(monkeypatch):
    warning_calls: list[str] = []
    caption_calls: list[str] = []
    info_calls: list[str] = []
    markdown_calls: list[str] = []

    monkeypatch.setattr(panels_module.st, "container", lambda **_: _DummyBlock())
    monkeypatch.setattr(panels_module.st, "expander", lambda *_args, **_kwargs: _DummyBlock())
    monkeypatch.setattr(panels_module, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(panels_module.st, "warning", lambda text: warning_calls.append(text))
    monkeypatch.setattr(panels_module.st, "caption", lambda text: caption_calls.append(text))
    monkeypatch.setattr(panels_module.st, "info", lambda text: info_calls.append(text))
    monkeypatch.setattr(panels_module.st, "markdown", lambda text, **kwargs: markdown_calls.append(text))

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

    panels_module.render_investor_flow_summary(
        signals=[_signal("A", "Watch", 1.10, 1.00)],
        investor_flow_status="CACHED",
        investor_flow_fresh=False,
        investor_flow_profile="foreign_lead",
        investor_flow_frame=frame,
        investor_flow_detail={"bootstrap_partial_preview": True},
    )

    assert any("최종 신호에는 반영되지 않았습니다" in text for text in warning_calls)
    assert not any("표시할 투자자 수급 데이터가 없습니다" in text for text in info_calls)
    assert not any("투자자 수급 탭" in text for text in caption_calls)
    assert any("단기 평균" in text for text in markdown_calls)


def test_build_investor_flow_glance_rows_sorts_by_abs_score_and_sector_tie_break():
    alpha = _signal("A", "Watch", 1.10, 1.00)
    alpha.flow_state = "supportive"
    alpha.flow_score = -1.4
    alpha.base_action = "Watch"
    alpha.action = "Hold"
    alpha.foreign_flow_state = "adverse"
    alpha.institutional_flow_state = "neutral"
    alpha.retail_flow_state = "supportive"
    alpha.sector_name = "Alpha"

    beta = _signal("B", "Strong Buy", 1.10, 1.00)
    beta.flow_state = "supportive"
    beta.flow_score = 1.4
    beta.base_action = "Watch"
    beta.action = "Strong Buy"
    beta.foreign_flow_state = "supportive"
    beta.institutional_flow_state = "supportive"
    beta.retail_flow_state = "adverse"
    beta.sector_name = "Beta"

    gamma = _signal("C", "Watch", 1.10, 1.00)
    gamma.flow_state = "neutral"
    gamma.flow_score = 0.4
    gamma.base_action = "Watch"
    gamma.action = "Watch"
    gamma.foreign_flow_state = "neutral"
    gamma.institutional_flow_state = "neutral"
    gamma.retail_flow_state = "neutral"
    gamma.sector_name = "Gamma"

    rows = build_investor_flow_glance_rows(
        [gamma, beta, alpha],
        locale="ko",
    )

    assert [row["sector"] for row in rows] == ["Alpha", "Beta", "Gamma"]
    assert rows[0]["foreign"] == "수급 역풍"
    assert rows[0]["institutional"] == "수급 중립"
    assert rows[0]["retail"] == "수급 우호"


def test_render_investor_flow_summary_limits_to_top_four_and_shows_participants(monkeypatch):
    markdown_calls: list[str] = []

    monkeypatch.setattr(panels_module.st, "container", lambda **_: _DummyBlock())
    monkeypatch.setattr(panels_module.st, "expander", lambda *_args, **_kwargs: _DummyBlock())
    monkeypatch.setattr(panels_module, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(panels_module.st, "warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panels_module.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panels_module.st, "info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panels_module.st, "markdown", lambda text, **kwargs: markdown_calls.append(text))

    signals: list[SectorSignal] = []
    for idx, score in enumerate([1.8, -1.6, 1.2, -0.9, 0.3], start=1):
        signal = _signal(str(idx), "Watch", 1.10, 1.00)
        signal.flow_state = "supportive" if score > 0 else "adverse"
        signal.flow_score = score
        signal.base_action = "Watch"
        signal.action = "Strong Buy" if score > 0.8 else "Hold" if score < -0.8 else "Watch"
        signal.foreign_flow_state = "supportive"
        signal.institutional_flow_state = "neutral"
        signal.retail_flow_state = "adverse"
        signal.sector_name = f"Sector {idx}"
        signals.append(signal)

    panels_module.render_investor_flow_summary(
        signals=signals,
        investor_flow_status="LIVE",
        investor_flow_fresh=True,
        investor_flow_profile="foreign_lead",
        investor_flow_frame=pd.DataFrame(),
        investor_flow_detail={},
    )

    flow_markup = next(text for text in markdown_calls if "flow-container" in text)
    assert "Sector 1" in flow_markup
    assert "Sector 2" in flow_markup
    assert "Sector 3" in flow_markup
    assert "Sector 4" in flow_markup
    assert "Sector 5" not in flow_markup
    assert "외국인" in flow_markup
    assert "기관" in flow_markup
    assert "개인" in flow_markup


def test_render_investor_flow_summary_signal_rows_tone_from_sigma_state_not_ratio(monkeypatch):
    markdown_calls: list[str] = []

    monkeypatch.setattr(panels_module.st, "container", lambda **_: _DummyBlock())
    monkeypatch.setattr(panels_module.st, "expander", lambda *_args, **_kwargs: _DummyBlock())
    monkeypatch.setattr(panels_module, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(panels_module.st, "warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panels_module.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panels_module.st, "info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panels_module.st, "markdown", lambda text, **kwargs: markdown_calls.append(text))

    signal = _signal("A", "Watch", 1.10, 1.00)
    signal.flow_state = "neutral"
    signal.flow_score = 0.2
    signal.foreign_flow_state = "adverse"
    signal.foreign_flow_ratio = 0.15
    signal.institutional_flow_state = "neutral"
    signal.institutional_flow_ratio = 0.02
    signal.retail_flow_state = "supportive"
    signal.retail_flow_ratio = -0.03

    panels_module.render_investor_flow_summary(
        signals=[signal],
        investor_flow_status="LIVE",
        investor_flow_fresh=True,
        investor_flow_profile="foreign_lead",
        investor_flow_frame=pd.DataFrame(),
        investor_flow_detail={},
    )

    flow_markup = next(text for text in markdown_calls if "flow-container" in text)
    assert "수급 역풍" in flow_markup
    assert "var(--danger)" in flow_markup


def test_build_investor_flow_snapshot_rows_pivots_latest_snapshot(monkeypatch):
    frame = pd.DataFrame(
        {
            "sector_code": ["5044", "5044", "5044", "1234", "1234", "1234"],
            "sector_name": ["KRX 반도체", "KRX 반도체", "KRX 반도체", "KRX 은행", "KRX 은행", "KRX 은행"],
            "investor_type": ["외국인", "기관합계", "개인", "외국인", "기관합계", "개인"],
            "net_buy_amount": [50, 20, -30, -10, 5, 15],
            "net_flow_ratio": [0.2, 0.1, -0.05, -0.04, 0.01, 0.03],
        },
        index=pd.to_datetime(["2026-04-07"] * 6),
    )

    rows = build_investor_flow_snapshot_rows(frame, locale="ko")

    assert rows[0]["sector"] == "KRX 반도체"
    assert rows[0]["foreign"] == "+20.00%"
    assert rows[0]["institutional"] == "+10.00%"
    assert rows[0]["retail"] == "-5.00%"


def test_build_investor_flow_snapshot_rows_keeps_raw_fields_and_adds_cues():
    frame = pd.DataFrame(
        {
            "sector_code": ["5044"] * 9 + ["1234"] * 9,
            "sector_name": ["KRX 반도체"] * 9 + ["KRX 은행"] * 9,
            "investor_type": (["외국인"] * 3 + ["기관합계"] * 3 + ["개인"] * 3) * 2,
            "net_buy_amount": [10, 20, 30, 5, 10, 15, -8, -10, -12, 1, 2, 3, 1, 1, 2, -1, -2, -2],
            "net_flow_ratio": [0.01, 0.02, 0.03, 0.00, 0.01, 0.02, -0.01, -0.02, -0.03, 0.01, 0.01, 0.01, 0.0, 0.01, 0.01, -0.01, -0.01, -0.01],
        },
        index=pd.to_datetime(
            ["2026-04-01", "2026-04-02", "2026-04-03"] * 6
        ),
    )
    summary_map = summarize_sector_investor_flow(
        frame,
        flow_profile="foreign_lead",
        short_window=2,
        long_window=3,
    )

    rows = build_investor_flow_snapshot_rows(
        frame,
        shared_flow_summary_map=summary_map,
        locale="ko",
    )

    assert rows[0]["sector"] == "KRX 반도체"
    assert rows[0]["sector_code"] == "5044"
    assert rows[0]["flow_score"] == pytest.approx(0.6123724356957945)
    assert rows[0]["foreign"] == "+3.00%"
    assert rows[0]["institutional"] == "+2.00%"
    assert rows[0]["retail"] == "-3.00%"
    assert "σ" in rows[0]["foreign_cue"]
    assert rows[0]["foreign_cue_raw"] == "supportive"
    assert rows[1]["sector"] == "KRX 은행"


def test_render_investor_flow_summary_uses_raw_snapshot_fallback_when_signal_flow_is_missing(monkeypatch):
    markdown_calls: list[str] = []

    monkeypatch.setattr(panels_module.st, "container", lambda **_: _DummyBlock())
    monkeypatch.setattr(panels_module.st, "expander", lambda *_args, **_kwargs: _DummyBlock())
    monkeypatch.setattr(panels_module, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(panels_module.st, "warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panels_module.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panels_module.st, "info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panels_module.st, "markdown", lambda text, **kwargs: markdown_calls.append(text))

    frame = pd.DataFrame(
        {
            "sector_code": ["5044", "5044", "5044"],
            "sector_name": ["KRX 반도체", "KRX 반도체", "KRX 반도체"],
            "investor_type": ["외국인", "기관합계", "개인"],
            "net_buy_amount": [50, 20, -30],
            "net_flow_ratio": [0.2, 0.1, -0.05],
        },
        index=pd.to_datetime(["2026-04-07"] * 3),
    )

    panels_module.render_investor_flow_summary(
        signals=[_signal("A", "Watch", 1.10, 1.00)],
        investor_flow_status="CACHED",
        investor_flow_fresh=False,
        investor_flow_profile="foreign_lead",
        investor_flow_frame=frame,
        investor_flow_detail={"bootstrap_partial_preview": True},
    )

    flow_markup = next(text for text in markdown_calls if "flow-container" in text)
    assert "KRX 반도체" in flow_markup
    assert "참고용 raw snapshot" in flow_markup
    assert "+20.00%" in flow_markup
    assert "+10.00%" in flow_markup
    assert "-5.00%" in flow_markup
    assert "σ" not in flow_markup


def test_render_investor_flow_summary_uses_raw_snapshot_fallback_when_signals_are_empty(monkeypatch):
    markdown_calls: list[str] = []

    monkeypatch.setattr(panels_module.st, "container", lambda **_: _DummyBlock())
    monkeypatch.setattr(panels_module.st, "expander", lambda *_args, **_kwargs: _DummyBlock())
    monkeypatch.setattr(panels_module, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(panels_module.st, "warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panels_module.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panels_module.st, "info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panels_module.st, "markdown", lambda text, **kwargs: markdown_calls.append(text))

    frame = pd.DataFrame(
        {
            "sector_code": ["5044"] * 9,
            "sector_name": ["KRX 반도체"] * 9,
            "investor_type": ["외국인"] * 3 + ["기관합계"] * 3 + ["개인"] * 3,
            "net_buy_amount": [10, 20, 30, 5, 10, 15, -8, -10, -12],
            "net_flow_ratio": [0.01, 0.02, 0.03, 0.00, 0.01, 0.02, -0.01, -0.02, -0.03],
        },
        index=pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"] * 3),
    )
    summary_map = summarize_sector_investor_flow(
        frame,
        flow_profile="foreign_lead",
        short_window=2,
        long_window=3,
    )

    panels_module.render_investor_flow_summary(
        signals=[],
        investor_flow_status="CACHED",
        investor_flow_fresh=False,
        investor_flow_profile="foreign_lead",
        investor_flow_frame=frame,
        investor_flow_detail={"bootstrap_partial_preview": True},
        shared_flow_summary_map=summary_map,
    )

    flow_markup = next(text for text in markdown_calls if "flow-container" in text)
    assert "KRX 반도체" in flow_markup
    assert "참고용 raw snapshot" in flow_markup
    assert "+3.00%" in flow_markup
    assert "σ" in flow_markup


def test_render_top_picks_table_uses_native_dataframe_and_limit(monkeypatch):
    dataframe_calls: list[tuple[pd.DataFrame, dict]] = []
    caption_calls: list[str] = []

    monkeypatch.setattr(
        "src.ui.components.st.dataframe",
        lambda df, **kwargs: dataframe_calls.append((df.copy(), kwargs)),
    )
    monkeypatch.setattr("src.ui.components.st.caption", lambda text: caption_calls.append(text))
    monkeypatch.setattr("src.ui.components.st.info", lambda *_args, **_kwargs: None)

    signals = [
        _signal(str(idx), "Strong Buy" if idx < 3 else "Watch", 1.2 - idx * 0.01, 1.0, alerts=["Overheat"])
        for idx in range(6)
    ]

    render_top_picks_table(signals, held_sectors=["Sector 0"], limit=5)

    assert len(dataframe_calls) == 1
    df, kwargs = dataframe_calls[0]
    assert kwargs.get("width") == "stretch"
    assert list(df.columns) == [
        "Rank",
        "Sector",
        "Decision",
        "Reason",
        "Risk",
        "Invalidation",
        "3M",
        "Alerts",
        "Held",
    ]
    assert len(df) == 5
    assert "column_config" in kwargs
    assert any("상위 5개" in text for text in caption_calls)

    caption_calls.clear()
    dataframe_calls.clear()
    render_top_picks_table(signals, held_sectors=["Sector 0"], limit=5, locale="en")

    assert any("Showing top 5 of 6" in text for text in caption_calls)


def test_render_top_picks_table_orders_rows_with_signal_display_sort_key(monkeypatch):
    dataframe_calls: list[tuple[pd.DataFrame, dict]] = []

    monkeypatch.setattr(
        "src.ui.components.st.dataframe",
        lambda df, **kwargs: dataframe_calls.append((df.copy(), kwargs)),
    )
    monkeypatch.setattr("src.ui.components.st.caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ui.components.st.info", lambda *_args, **_kwargs: None)

    held_watch = _signal("A", "Watch", 1.08, 1.00)
    held_watch.sector_name = "Alpha"
    held_buy = _signal("B", "Strong Buy", 1.02, 1.00)
    held_buy.sector_name = "Beta"
    new_buy = _signal("C", "Strong Buy", 1.06, 1.00)
    new_buy.sector_name = "Gamma"
    new_hold = _signal("D", "Hold", 1.11, 1.00)
    new_hold.sector_name = "Delta"

    signals = [new_hold, held_watch, new_buy, held_buy]
    held_sectors = ["Alpha", "Beta"]

    render_top_picks_table(signals, held_sectors=held_sectors, limit=5)

    assert len(dataframe_calls) == 1
    df, _ = dataframe_calls[0]
    expected_order = [
        signal.sector_name
        for signal in sorted(
            signals,
            key=lambda signal: signal_display_sort_key(signal, held_sectors),
        )
    ]
    assert list(df["Sector"]) == expected_order


def test_render_top_picks_table_preserves_held_and_new_empty_states(monkeypatch):
    info_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.info", lambda text: info_calls.append(text))
    monkeypatch.setattr("src.ui.components.st.dataframe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ui.components.st.caption", lambda *_args, **_kwargs: None)

    new_only_signal = _signal("A", "Strong Buy", 1.10, 1.00)
    new_only_signal.sector_name = "Alpha"

    render_top_picks_table([new_only_signal], held_sectors=[], position_mode="held")
    render_top_picks_table([new_only_signal], held_sectors=["Alpha"], position_mode="new")

    assert info_calls == [
        "보유 섹터 검토 후보를 보려면 보유 섹터를 먼저 추가하세요.",
        "현재 결정 규칙에 부합하는 신규 검토 후보가 없습니다.",
    ]


def test_render_investor_decision_boards_routes_through_dedicated_board_card_path(monkeypatch):
    session_state: dict[str, object] = {"held_sectors": ["Sector A"]}
    board_calls: list[dict[str, object]] = []
    signal = _signal("A", "Strong Buy", 1.10, 1.00)

    monkeypatch.setattr(panels_module.st, "session_state", session_state)
    monkeypatch.setattr(panels_module.st, "container", lambda **_: _DummyBlock())
    monkeypatch.setattr(panels_module.st, "columns", lambda count, **_: [_DummyBlock() for _ in range(count)])
    monkeypatch.setattr(
        panels_module.st,
        "multiselect",
        lambda _label, options, default, **kwargs: list(default),
    )
    monkeypatch.setattr(panels_module.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panels_module, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(
        panels_module,
        "_render_decision_board_cards",
        lambda **kwargs: board_calls.append(kwargs),
    )

    selected = render_investor_decision_boards(
        signals=[signal],
        held_sector_options=["Sector A", "Sector B"],
        limit=5,
    )

    assert selected == ["Sector A"]
    assert board_calls == [
        {
            "signals": [signal],
            "held_sectors": ["Sector A"],
            "position_mode": "held",
            "limit": 5,
            "locale": "ko",
        },
        {
            "signals": [signal],
            "held_sectors": ["Sector A"],
            "position_mode": "new",
            "limit": 5,
            "locale": "ko",
        },
    ]


def test_render_investor_decision_boards_defaults_to_five_items_per_board(monkeypatch):
    session_state: dict[str, object] = {"held_sectors": ["Sector A"]}
    limits: list[int] = []

    monkeypatch.setattr(panels_module.st, "session_state", session_state)
    monkeypatch.setattr(panels_module.st, "container", lambda **_: _DummyBlock())
    monkeypatch.setattr(panels_module.st, "columns", lambda count, **_: [_DummyBlock() for _ in range(count)])
    monkeypatch.setattr(
        panels_module.st,
        "multiselect",
        lambda _label, options, default, **kwargs: list(default),
    )
    monkeypatch.setattr(panels_module.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panels_module, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(
        panels_module,
        "_render_decision_board_cards",
        lambda **kwargs: limits.append(int(kwargs["limit"])),
    )

    render_investor_decision_boards(
        signals=[_signal("A", "Strong Buy", 1.10, 1.00)],
        held_sector_options=["Sector A", "Sector B"],
    )

    assert limits == [5, 5]


def test_render_decision_board_cards_renders_action_why_and_invalidation(monkeypatch):
    markdown_calls: list[str] = []
    info_calls: list[str] = []
    signal = _signal("A", "Strong Buy", 1.12, 1.00, returns={"1M": 0.04, "3M": 0.13})
    signal.sector_name = "Alpha"

    monkeypatch.setattr(panels_module.st, "markdown", lambda text, **_: markdown_calls.append(text))
    monkeypatch.setattr(panels_module.st, "info", lambda text: info_calls.append(text))
    monkeypatch.setattr(panels_module.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ui.components.st.dataframe", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dataframe should not be used")))

    panels_module._render_decision_board_cards(
        signals=[signal],
        held_sectors=["Alpha"],
        position_mode="held",
        limit=5,
        locale="ko",
    )

    assert not info_calls
    assert len(markdown_calls) == 1
    markup = markdown_calls[0]
    assert "Alpha" in markup
    assert "보유 검토" in markup
    assert "추가 검토 후보" in markup
    assert "핵심 판단" in markup
    assert "왜" in markup
    assert "무효화 조건" in markup


def test_render_decision_board_cards_maps_watch_to_supported_badge_tone(monkeypatch):
    markdown_calls: list[str] = []
    signal = _signal("A", "Watch", 1.01, 1.00)
    signal.sector_name = "Alpha"

    monkeypatch.setattr(panels_module.st, "markdown", lambda text, **_: markdown_calls.append(text))
    monkeypatch.setattr(panels_module.st, "info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panels_module.st, "caption", lambda *_args, **_kwargs: None)

    panels_module._render_decision_board_cards(
        signals=[signal],
        held_sectors=[],
        position_mode="new",
        limit=5,
        locale="ko",
    )

    assert len(markdown_calls) == 1
    markup = markdown_calls[0]
    assert "flow-card__badge--neutral" in markup
    assert "flow-card__badge--info" not in markup


def test_render_signal_table_uses_native_dataframe_and_applies_filters(monkeypatch):
    dataframe_calls: list[tuple[pd.DataFrame, dict]] = []
    info_calls: list[str] = []

    monkeypatch.setattr(
        "src.ui.components.st.dataframe",
        lambda df, **kwargs: dataframe_calls.append((df.copy(), kwargs)),
    )
    monkeypatch.setattr("src.ui.components.st.info", lambda text: info_calls.append(text))
    monkeypatch.setattr("src.ui.components.st.caption", lambda *_args, **_kwargs: None)

    signals = [
        _signal("A", "Watch", 1.10, 1.00, macro_regime="Recovery", macro_fit=True),
        _signal("B", "Hold", 0.95, 1.00, macro_regime="Slowdown", macro_fit=False),
        _signal("C", "N/A", 1.00, 1.00, macro_regime="Recovery", macro_fit=True, alerts=["Benchmark Missing"]),
    ]

    render_signal_table(
        signals,
        filter_action="Watch",
        filter_regime_only=True,
        current_regime="Recovery",
        held_sectors=["Sector A"],
        position_mode="held",
        show_alerted_only=False,
    )

    assert not info_calls
    assert len(dataframe_calls) == 1
    df, kwargs = dataframe_calls[0]
    assert kwargs.get("width") == "stretch"
    assert len(df) == 1
    assert df.iloc[0]["Action"] == "[~] 관망 (Watch)"
    assert bool(df.iloc[0]["Held"]) is True
    assert df.iloc[0]["Decision"] == "유지 / 모니터링"
    assert bool(df.iloc[0]["In Regime"]) is True
    assert "column_config" in kwargs
    assert set(kwargs["column_config"]) == {
        "Sector",
        "Held",
        "Decision",
        "In Regime",
        "Action",
        "Taxonomy",
        "ETF",
        "Reason",
        "Invalidation",
        "RSI",
        "1M",
        "3M",
        "Volatility",
        "MDD (3M)",
        "Alerts",
    }

    dataframe_calls.clear()
    render_signal_table(
        signals,
        filter_action="Watch",
        filter_regime_only=True,
        current_regime="Recovery",
        held_sectors=["Sector A"],
        position_mode="held",
        show_alerted_only=False,
        locale="en",
    )

    df, _ = dataframe_calls[0]
    assert df.iloc[0]["Action"] == "[~] Watch"
    assert df.iloc[0]["Decision"] == "Hold / monitor"


def test_render_signal_table_uses_macro_context_for_kr_rows(monkeypatch):
    dataframe_calls: list[tuple[pd.DataFrame, dict]] = []

    monkeypatch.setattr(
        "src.ui.components.st.dataframe",
        lambda df, **kwargs: dataframe_calls.append((df.copy(), kwargs)),
    )
    monkeypatch.setattr("src.ui.components.st.info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ui.components.st.caption", lambda *_args, **_kwargs: None)

    render_signal_table(
        [
            _signal(
                "A",
                "Strong Buy",
                1.10,
                1.00,
                action_policy="KR_MOMENTUM_ONLY",
                macro_fit=False,
                macro_regime="Unassigned",
                macro_context_regime="Recovery",
                taxonomy_kind="THEME",
                taxonomy_label="반도체",
            )
        ],
        filter_action="Strong Buy",
        filter_regime_only=True,
        current_regime="Recovery",
    )

    df, kwargs = dataframe_calls[0]
    assert "Macro Context" in df.columns
    assert "In Regime" not in df.columns
    assert df.iloc[0]["Macro Context"] == "참고: Recovery"
    assert df.iloc[0]["Taxonomy"] == "THEME · 반도체"
    assert "Macro Context" in kwargs["column_config"]


def test_render_top_bar_filters_hides_regime_toggle_when_disabled(monkeypatch):
    session_state = {
        "filter_action_global": ALL_ACTION_KEY,
        "filter_regime_only_global": True,
        "position_mode": "all",
        "show_alerted_only": False,
    }
    toggle_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.session_state", session_state)
    monkeypatch.setattr("src.ui.components.st.container", lambda **kwargs: _DummyBlock())
    monkeypatch.setattr("src.ui.components.st.markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.ui.components.st.caption", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.ui.components.st.columns", lambda spec, **kwargs: [_DummyBlock() for _ in range(len(spec))])
    monkeypatch.setattr("src.ui.components.st.selectbox", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.ui.components.st.segmented_control", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.ui.components.st.toggle", lambda label, **kwargs: toggle_calls.append(label))

    _, regime_only, _, _ = render_top_bar_filters(
        current_regime="Recovery",
        action_options=[ALL_ACTION_KEY, "Strong Buy", "Watch", "Hold", "Avoid", "N/A"],
        enable_regime_filter=False,
    )

    assert regime_only is False
    assert toggle_calls == ["알림 있는 항목만"]


def test_render_signal_table_handles_empty_after_filters(monkeypatch):
    info_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.info", lambda text: info_calls.append(text))

    render_signal_table(
        [_signal("A", "Watch", 1.10, 1.00, macro_regime="Recovery")],
        filter_action="Avoid",
        filter_regime_only=False,
        current_regime="Recovery",
    )

    assert info_calls == ["활성 필터 조건에 맞는 섹터가 없습니다."]

    info_calls.clear()
    render_signal_table(
        [_signal("A", "Watch", 1.10, 1.00, macro_regime="Recovery")],
        filter_action="Avoid",
        filter_regime_only=False,
        current_regime="Recovery",
        locale="en",
    )

    assert info_calls == ["No sectors match the active filters."]


def test_render_signal_table_accepts_legacy_english_all_filter(monkeypatch):
    info_calls: list[str] = []
    dataframe_calls: list[tuple[pd.DataFrame, dict]] = []

    monkeypatch.setattr("src.ui.components.st.info", lambda text: info_calls.append(text))
    monkeypatch.setattr(
        "src.ui.components.st.dataframe",
        lambda df, **kwargs: dataframe_calls.append((df.copy(), kwargs)),
    )

    render_signal_table(
        [_signal("A", "Watch", 1.10, 1.00, macro_regime="Recovery")],
        filter_action="All",
        filter_regime_only=False,
        current_regime="Recovery",
    )

    assert not info_calls
    assert len(dataframe_calls) == 1
    df, _ = dataframe_calls[0]
    assert len(df) == 1
    assert df.iloc[0]["Action"] == "[~] 관망 (Watch)"

    dataframe_calls.clear()
    render_signal_table(
        [_signal("A", "Watch", 1.10, 1.00, macro_regime="Recovery")],
        filter_action="All",
        filter_regime_only=False,
        current_regime="Recovery",
        locale="en",
    )

    df, _ = dataframe_calls[0]
    assert df.iloc[0]["Action"] == "[~] Watch"


def test_render_analysis_toolbar_returns_form_selection(monkeypatch):
    markdown_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: markdown_calls.append(text))
    monkeypatch.setattr("src.ui.components.st.form", lambda *_args, **_kwargs: _DummyBlock())
    monkeypatch.setattr("src.ui.components.st.columns", lambda spec, **_: [_DummyBlock() for _ in range(len(spec))])
    monkeypatch.setattr(
        "src.ui.components.st.date_input",
        lambda label, value, **_: value,
    )
    monkeypatch.setattr(
        "src.ui.components.st.segmented_control",
        lambda *_args, **_kwargs: "3Y",
    )
    monkeypatch.setattr(
        "src.ui.components.st.form_submit_button",
        lambda *_args, **_kwargs: True,
    )

    start, end, preset, submitted = render_analysis_toolbar(
        min_date=pd.Timestamp("2024-01-31").date(),
        max_date=pd.Timestamp("2025-01-31").date(),
        start_date=pd.Timestamp("2024-07-31").date(),
        end_date=pd.Timestamp("2025-01-31").date(),
        selected_range_preset="1Y",
        selected_cycle_phase="ALL",
        selected_sector="Sector A",
    )

    expected_start, expected_end = resolve_range_from_preset(
        max_date=pd.Timestamp("2025-01-31").date(),
        min_date=pd.Timestamp("2024-01-31").date(),
        preset="3Y",
    )
    assert submitted is True
    assert preset == "3Y"
    assert start == expected_start
    assert end == expected_end
    assert any("analysis-toolbar" in call for call in markdown_calls)


def test_render_analysis_toolbar_surfaces_page_level_context_summary(monkeypatch):
    markdown_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: markdown_calls.append(text))
    monkeypatch.setattr("src.ui.components.st.form", lambda *_args, **_kwargs: _DummyBlock())
    monkeypatch.setattr("src.ui.components.st.columns", lambda spec, **_: [_DummyBlock() for _ in range(len(spec))])
    monkeypatch.setattr(
        "src.ui.components.st.date_input",
        lambda label, value, **_: value,
    )
    monkeypatch.setattr(
        "src.ui.components.st.segmented_control",
        lambda *_args, **_kwargs: "1Y",
    )
    monkeypatch.setattr(
        "src.ui.components.st.form_submit_button",
        lambda *_args, **_kwargs: False,
    )

    start, end, preset, submitted = render_analysis_toolbar(
        min_date=pd.Timestamp("2024-01-31").date(),
        max_date=pd.Timestamp("2025-01-31").date(),
        start_date=pd.Timestamp("2024-07-31").date(),
        end_date=pd.Timestamp("2025-01-31").date(),
        selected_range_preset="1Y",
        selected_cycle_phase="ALL",
        selected_sector="Sector A",
    )

    assert submitted is False
    assert preset == "1Y"
    assert start == pd.Timestamp("2024-07-31").date()
    assert end == pd.Timestamp("2025-01-31").date()
    assert any('analysis-toolbar__summary-item"><span>기간</span>' in call for call in markdown_calls)
    assert any('analysis-toolbar__summary-item"><span>사이클</span>' in call for call in markdown_calls)
    assert any('analysis-toolbar__summary-item"><span>섹터</span>' in call for call in markdown_calls)
    assert any("리서치 범위" in call or "Research scope" in call for call in markdown_calls)
    assert any("리서치 캔버스 범위 조정" in call or "Adjust research canvas scope" in call for call in markdown_calls)
    assert any("기본 판단 규칙은 바꾸지 않습니다" in call or "does not change the base judgment rules" in call for call in markdown_calls)


def test_render_stock_lookup_control_returns_query_and_submit(monkeypatch):
    markdown_calls: list[str] = []
    caption_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: markdown_calls.append(text))
    monkeypatch.setattr("src.ui.components.st.caption", lambda text, **_: caption_calls.append(text))
    monkeypatch.setattr("src.ui.components.st.form", lambda *_args, **_kwargs: _DummyBlock())
    monkeypatch.setattr("src.ui.components.st.columns", lambda spec, **_: [_DummyBlock() for _ in range(len(spec))])
    monkeypatch.setattr("src.ui.components.st.text_input", lambda *_args, **_kwargs: "005930")
    monkeypatch.setattr("src.ui.components.st.form_submit_button", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("src.ui.components.st.success", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ui.components.st.warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ui.components.st.info", lambda *_args, **_kwargs: None)

    query, submitted = render_stock_lookup_control(
        market_id="KR",
        query_value="",
        status="",
        message="",
    )

    assert submitted is True
    assert query == "005930"
    assert any("종목" in call or "Stock" in call for call in markdown_calls)
    assert any("구성종목" in call or "constituent membership" in call for call in caption_calls)


def test_render_stock_lookup_control_surfaces_market_specific_feedback(monkeypatch):
    caption_calls: list[str] = []
    success_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.markdown", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ui.components.st.caption", lambda text, **_: caption_calls.append(text))
    monkeypatch.setattr("src.ui.components.st.form", lambda *_args, **_kwargs: _DummyBlock())
    monkeypatch.setattr("src.ui.components.st.columns", lambda spec, **_: [_DummyBlock() for _ in range(len(spec))])
    monkeypatch.setattr("src.ui.components.st.text_input", lambda *_args, **_kwargs: "MSFT")
    monkeypatch.setattr("src.ui.components.st.form_submit_button", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("src.ui.components.st.success", lambda text, **_: success_calls.append(text))
    monkeypatch.setattr("src.ui.components.st.warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ui.components.st.info", lambda *_args, **_kwargs: None)

    query, submitted = render_stock_lookup_control(
        market_id="US",
        query_value="MSFT",
        status="success",
        message="resolved",
    )

    assert submitted is False
    assert query == "MSFT"
    assert success_calls == ["resolved"]
    assert any("issuer classification" in call or "발행사 업종" in call for call in caption_calls)


def test_render_stock_lookup_control_renders_all_matched_sectors(monkeypatch):
    markdown_calls: list[str] = []
    caption_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: markdown_calls.append(text))
    monkeypatch.setattr("src.ui.components.st.caption", lambda text, **_: caption_calls.append(text))
    monkeypatch.setattr("src.ui.components.st.form", lambda *_args, **_kwargs: _DummyBlock())
    monkeypatch.setattr("src.ui.components.st.columns", lambda spec, **_: [_DummyBlock() for _ in range(len(spec))])
    monkeypatch.setattr("src.ui.components.st.text_input", lambda *_args, **_kwargs: "005930")
    monkeypatch.setattr("src.ui.components.st.form_submit_button", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("src.ui.components.st.success", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ui.components.st.warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ui.components.st.info", lambda *_args, **_kwargs: None)

    query, submitted = render_stock_lookup_control(
        market_id="KR",
        query_value="005930",
        status="success",
        message="resolved",
        display_model={
            "canonical_sector": {"sector_code": "5044", "sector_name": "KRX 반도체"},
            "matched_sectors": [
                {"sector_code": "5044", "sector_name": "KRX 반도체", "snapshot_date": "20260417"},
                {"sector_code": "1155", "sector_name": "KOSPI200 정보기술", "snapshot_date": "20260417"},
                {"sector_code": "5042", "sector_name": "KRX 산업재", "snapshot_date": "20260417"},
            ],
            "result": {
                "canonicalization_basis": "lookup_priority_same_date",
                "match_date_mode": "same_date",
                "match_effective_date": "20260417",
            },
        },
    )

    assert submitted is False
    assert query == "005930"
    assert any("매칭 섹터" in call or "Matched sectors" in call for call in caption_calls)
    assert any("KRX 반도체 (현재 적용)" in call or "KRX 반도체 (Selected)" in call for call in markdown_calls)
    assert any("KOSPI200 정보기술" in call for call in markdown_calls)
    assert any("KRX 산업재" in call for call in markdown_calls)


def test_render_stock_lookup_control_does_not_require_hierarchy_metadata(monkeypatch):
    markdown_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: markdown_calls.append(text))
    monkeypatch.setattr("src.ui.components.st.caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ui.components.st.form", lambda *_args, **_kwargs: _DummyBlock())
    monkeypatch.setattr("src.ui.components.st.columns", lambda spec, **_: [_DummyBlock() for _ in range(len(spec))])
    monkeypatch.setattr("src.ui.components.st.text_input", lambda *_args, **_kwargs: "005930")
    monkeypatch.setattr("src.ui.components.st.form_submit_button", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("src.ui.components.st.success", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ui.components.st.warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.ui.components.st.info", lambda *_args, **_kwargs: None)

    render_stock_lookup_control(
        market_id="KR",
        query_value="005930",
        status="success",
        message="resolved",
        display_model={
            "canonical_sector": {"sector_code": "5044", "sector_name": "KRX 반도체"},
            "matched_sectors": [
                {"sector_code": "5044", "sector_name": "KRX 반도체", "snapshot_date": "20260417"},
            ],
            "result": {"canonicalization_basis": "single_match", "match_date_mode": "not_applicable", "match_effective_date": ""},
        },
    )

    assert any("KRX 반도체" in call for call in markdown_calls)


def test_range_preset_helpers_use_year_windows():
    start, end = resolve_range_from_preset(
        max_date=pd.Timestamp("2026-03-06").date(),
        min_date=pd.Timestamp("2016-03-08").date(),
        preset="5Y",
    )

    assert start == pd.Timestamp("2021-03-07").date()
    assert end == pd.Timestamp("2026-03-06").date()
    assert infer_range_preset(
        start_date=pd.Timestamp("2023-03-07").date(),
        end_date=pd.Timestamp("2026-03-06").date(),
        min_date=pd.Timestamp("2016-03-08").date(),
        max_date=pd.Timestamp("2026-03-06").date(),
    ) == "3Y"
    assert infer_range_preset(
        start_date=pd.Timestamp("2016-03-08").date(),
        end_date=pd.Timestamp("2026-03-06").date(),
        min_date=pd.Timestamp("2016-03-08").date(),
        max_date=pd.Timestamp("2026-03-06").date(),
    ) == "ALL"


def test_normalize_range_preset_maps_legacy_values():
    assert normalize_range_preset("12M") == "1Y"
    assert normalize_range_preset("3M") == "CUSTOM"
    assert normalize_range_preset("6M") == "CUSTOM"
    assert normalize_range_preset("18M") == "CUSTOM"
    assert normalize_range_preset("5Y") == "5Y"


def test_heatmap_palette_helpers_normalize_and_label_presets():
    assert HEATMAP_PALETTE_OPTIONS == ("classic", "contrast", "blue_orange")
    assert normalize_heatmap_palette("contrast") == "contrast"
    assert normalize_heatmap_palette("BLUE_ORANGE") == "blue_orange"
    assert normalize_heatmap_palette("unknown") == "classic"
    assert format_heatmap_palette_label("classic") == "기본 빨강/초록"
    assert format_heatmap_palette_label("contrast") == "고대비 빨강/초록"
    assert format_heatmap_palette_label("classic", locale="en") == "Classic red/green"
    assert format_heatmap_palette_label("contrast", locale="en") == "High-contrast red/green"


def test_get_analysis_heatmap_colorscale_returns_distinct_presets():
    classic = get_analysis_heatmap_colorscale(theme_mode="light", palette="classic")
    contrast = get_analysis_heatmap_colorscale(theme_mode="light", palette="contrast")
    blue_orange = get_analysis_heatmap_colorscale(theme_mode="light", palette="blue_orange")

    assert classic != contrast
    assert classic != blue_orange
    assert contrast != blue_orange
    assert classic[0][0] == 0.0 and classic[-1][0] == 1.0
    assert contrast[2][0] == 0.50
    assert blue_orange[0][1] == "#2E6BC9"


def test_build_sector_strength_heatmap_marks_selected_row_column():
    heatmap_df = pd.DataFrame(
        [[1.2, -0.4], [3.1, 2.0]],
        index=["Sector A", "Sector B"],
        columns=["2025-01", "2025-02"],
    )

    fig = build_sector_strength_heatmap(
        heatmap_df,
        selected_sector="Sector B",
        selected_month="2025-02",
        theme_mode="light",
    )

    assert len(fig.data) == 1
    assert fig.layout.title.text == "Monthly sector return"
    assert len(fig.layout.shapes) == 3
    assert fig.layout.xaxis.tickangle == 0
    assert fig.data[0].texttemplate == "%{z:.1f}"
    assert list(fig.data[0].colorscale) == [
        tuple(item) for item in get_analysis_heatmap_colorscale(theme_mode="light", palette="classic")
    ]


def test_build_sector_strength_heatmap_supports_custom_title_and_hover_suffix():
    heatmap_df = pd.DataFrame(
        [[1.2, -0.4]],
        index=["Sector A"],
        columns=["2025-01", "2025-02"],
    )

    fig = build_sector_strength_heatmap(
        heatmap_df,
        theme_mode="light",
        title="Monthly sector strength vs KOSPI",
        empty_message="No monthly sector strength vs KOSPI data is available for the active filters.",
        helper_metric_label="monthly excess return",
        hover_value_suffix="%p vs KOSPI",
    )

    assert fig.layout.title.text == "Monthly sector strength vs KOSPI"
    assert "%p vs KOSPI" in fig.data[0].hovertemplate


def test_build_sector_strength_heatmap_applies_selected_palette():
    heatmap_df = pd.DataFrame(
        [[1.2, -0.4]],
        index=["Sector A"],
        columns=["2025-01", "2025-02"],
    )

    fig = build_sector_strength_heatmap(
        heatmap_df,
        theme_mode="light",
        palette="contrast",
    )

    assert list(fig.data[0].colorscale) == [
        tuple(item)
        for item in get_analysis_heatmap_colorscale(
            theme_mode="light",
            palette="contrast",
        )
    ]


def test_build_sector_strength_heatmap_rotates_5y_x_labels_vertical():
    heatmap_df = pd.DataFrame(
        [list(range(60)), list(range(60, 120))],
        index=["Sector A", "Sector B"],
        columns=[f"2021-{month:02d}" for month in range(1, 13)]
        + [f"2022-{month:02d}" for month in range(1, 13)]
        + [f"2023-{month:02d}" for month in range(1, 13)]
        + [f"2024-{month:02d}" for month in range(1, 13)]
        + [f"2025-{month:02d}" for month in range(1, 13)],
    )

    fig = build_sector_strength_heatmap(heatmap_df, theme_mode="dark")

    assert fig.layout.xaxis.tickangle == -90
    assert list(fig.layout.xaxis.ticktext) == list(heatmap_df.columns)
    assert fig.layout.margin.b == 128
    assert "Hover or click a cell" in fig.layout.title.text
    assert fig.data[0].texttemplate is None


def test_build_sector_strength_heatmap_thins_all_range_x_labels_and_hides_cell_text():
    month_labels = [
        f"{year}-{month:02d}"
        for year in range(2016, 2026)
        for month in range(1, 13)
    ]
    heatmap_df = pd.DataFrame(
        [list(range(len(month_labels))), list(range(len(month_labels), len(month_labels) * 2))],
        index=["Sector A", "Sector B"],
        columns=month_labels,
    )

    fig = build_sector_strength_heatmap(heatmap_df, theme_mode="light")
    ticktext = list(fig.layout.xaxis.ticktext)
    visible_labels = [label for label in ticktext if label]

    assert len(month_labels) == 120
    assert fig.layout.xaxis.tickangle == -90
    assert fig.layout.margin.b == 128
    assert fig.data[0].texttemplate is None
    assert "Hover or click a cell" in fig.layout.title.text
    assert len(visible_labels) == 40
    assert visible_labels[0] == "2016-01"
    assert visible_labels[1] == "2016-04"
    assert ticktext[1] == ""


def test_build_overview_review_candidate_projection_scores_numeric_examples():
    signal = _signal("A", "Strong Buy", 1.10, 1.00, action_policy="KR_MOMENTUM_ONLY")
    signal.mom_percentile = 80.0
    signal.rs_change_pct = 4.0
    signal.macro_fit = True
    signal.sector_fit_rank = 2
    signal.sector_fit_total = 5
    signal.flow_state = "supportive"
    signal.flow_score = 1.2

    candidate = panels_module._build_overview_review_candidate_projection(signal)

    assert candidate is not None
    assert candidate["momentum_score"] == pytest.approx(80.0)
    assert candidate["macro_score"] == pytest.approx(67.5)
    assert candidate["flow_score"] == pytest.approx(80.0)
    assert candidate["candidate_score"] == pytest.approx(75.625)
    assert candidate["upside_proxy"] == pytest.approx(84.25)
    assert candidate["downside_proxy"] == pytest.approx(19.75)
    assert candidate["edge_proxy"] == pytest.approx(64.5)
    assert candidate["turning_point_state"] == "Continuation up"
    assert candidate["bullish_evidence"][:3] == ["모멘텀 우위", "RS 상방", "RS 개선"]
    assert candidate["candidate_policy"] == "COMPOSITE_REVIEW_CANDIDATE"
    assert candidate["action_policy"] == "KR_MOMENTUM_ONLY"
    assert signal.action_policy == "KR_MOMENTUM_ONLY"

    neutral_signal = _signal("B", "Watch", 1.00, 1.00, macro_fit=False)
    neutral_signal.mom_percentile = float("nan")
    neutral_signal.mom_score = 0.4
    neutral_signal.flow_state = "unavailable"
    neutral_signal.flow_score = float("nan")

    neutral_candidate = panels_module._build_overview_review_candidate_projection(neutral_signal)

    assert neutral_candidate is not None
    assert neutral_candidate["momentum_score"] == pytest.approx(40.0)
    assert neutral_candidate["macro_score"] == pytest.approx(35.0)
    assert neutral_candidate["flow_score"] == pytest.approx(50.0)
    assert neutral_candidate["flow_available"] is False
    assert neutral_candidate["candidate_score"] == pytest.approx(40.25)


def test_build_overview_review_candidate_projection_marks_bearish_turn_exactly():
    signal = _signal(
        "A",
        "Avoid",
        0.94,
        1.00,
        macro_fit=False,
        alerts=["Overheat"],
    )
    signal.mom_percentile = 20.0
    signal.rs_change_pct = -5.0
    signal.trend_ok = False
    signal.flow_state = "adverse"
    signal.flow_score = -1.0
    signal.volatility_20d = 0.32
    signal.mdd_3m = -0.22

    candidate = panels_module._build_overview_review_candidate_projection(signal)

    assert candidate is not None
    assert candidate["upside_proxy"] == pytest.approx(23.25)
    assert candidate["downside_proxy"] == pytest.approx(77.75)
    assert candidate["edge_proxy"] == pytest.approx(-54.5)
    assert candidate["turning_point_state"] == "Bearish turn"
    assert candidate["bearish_evidence"][:4] == ["모멘텀 약화", "RS 하방", "RS 둔화", "추세 훼손"]
    assert signal.action == "Avoid"


def test_build_overview_review_candidate_projection_uses_boolean_momentum_and_flow_guards():
    signal = _signal("A", "Strong Buy", 1.10, 1.00)
    signal.mom_percentile = float("inf")
    signal.mom_score = float("-inf")
    signal.flow_state = "supportive"
    signal.flow_score = float("inf")

    candidate = panels_module._build_overview_review_candidate_projection(signal)

    assert candidate is not None
    assert candidate["momentum_score"] == pytest.approx(66.6667)
    assert candidate["flow_score"] == pytest.approx(50.0)
    assert candidate["upside_proxy"] == pytest.approx(74.33, abs=0.01)
    assert candidate["downside_proxy"] == pytest.approx(27.50, abs=0.01)
    assert candidate["edge_proxy"] == pytest.approx(46.83, abs=0.01)
    assert "수급 점수 중립" in candidate["warnings"]


def test_build_sector_momentum_decision_boards_groups_watch_without_trend_as_monitor():
    strong = _signal("A", "Strong Buy", 1.10, 1.00, action_policy="KR_MOMENTUM_ONLY")
    strong.mom_percentile = 90.0
    watch = _signal("B", "Watch", 0.99, 1.00, action_policy="KR_MOMENTUM_ONLY")
    watch.momentum_core_pass = True
    watch.trend_ok = False
    watch.mom_percentile = 75.0
    risky_watch = _signal("C", "Watch", 0.98, 1.00, action_policy="KR_MOMENTUM_ONLY", alerts=["Overheat"])
    risky_watch.momentum_core_pass = True
    risky_watch.trend_ok = False
    risky_watch.mom_percentile = 65.0
    risky_watch.flow_state = "neutral"
    bearish = _signal("D", "Hold", 0.94, 1.00, action_policy="KR_MOMENTUM_ONLY")
    bearish.trend_ok = False
    bearish.rs_change_pct = -3.0
    bearish.mom_percentile = 45.0

    boards = panels_module._build_sector_momentum_decision_boards(
        [strong, watch, risky_watch, bearish],
        held_sectors=["Sector B", "Sector C"],
    )

    assert [candidate["sector_name"] for candidate in boards["new_review"]] == ["Sector A"]
    assert [candidate["sector_name"] for candidate in boards["held_monitor"]] == ["Sector B"]
    assert [candidate["sector_name"] for candidate in boards["held_reduce"]] == ["Sector C"]
    assert [candidate["sector_name"] for candidate in boards["inflection"]] == ["Sector D"]
    assert boards["held_monitor"][0]["risk_flag"] is False
    assert boards["held_reduce"][0]["risk_flag"] is True


def test_build_sector_momentum_decision_boards_includes_held_na_as_reduce():
    missing = _signal(
        "A",
        "N/A",
        1.00,
        1.00,
        action_policy="KR_MOMENTUM_ONLY",
        alerts=["Benchmark Missing"],
    )

    boards = panels_module._build_sector_momentum_decision_boards(
        [missing],
        held_sectors=["Sector A"],
    )

    assert [candidate["sector_name"] for candidate in boards["held_reduce"]] == ["Sector A"]
    assert boards["held_reduce"][0]["candidate_policy"] == "SECTOR_MOMENTUM_NA_DATA_CHECK"
    assert boards["held_reduce"][0]["metrics"][0] == ("상방 proxy", "N/A")


def test_render_sector_momentum_decision_boards_uses_proxy_guardrail_copy(monkeypatch):
    markdown_calls: list[str] = []
    signal = _signal("A", "Strong Buy", 1.10, 1.00, action_policy="KR_MOMENTUM_ONLY")
    signal.mom_percentile = 88.0

    monkeypatch.setattr(panels_module.st, "markdown", lambda text, **_: markdown_calls.append(text))

    panels_module.render_sector_momentum_decision_boards([signal], held_sectors=[])

    assert markdown_calls
    markup = markdown_calls[0]
    assert "의사결정 보드" in markup
    assert "신규/증액 검토" in markup
    assert "보유 모니터링" in markup
    assert "보유 축소/주의" in markup
    assert "변곡 감시" in markup
    assert "보정 확률이 아니라 근거 점수" in markup
    assert "canonical action policy는 바꾸지 않습니다" in markup
    assert "상승 확률" not in markup
    assert "하락 확률" not in markup


def test_render_theme_lens_panel_uses_proxy_guardrail_copy(monkeypatch):
    markdown_calls: list[str] = []
    dataframe_payloads: list[pd.DataFrame] = []

    monkeypatch.setattr(panels_module.st, "markdown", lambda text, **_: markdown_calls.append(text))
    monkeypatch.setattr(panels_module.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        panels_module.st,
        "dataframe",
        lambda data, **kwargs: dataframe_payloads.append(data.copy()),
    )

    clicked = panels_module.render_theme_lens_panel(
        [
            {
                "theme_name": "조선",
                "status": "CACHED",
                "representative_etfs": [{"code": "0141S0", "name": "SOL 조선기자재"}],
                "latest_date": "2026-05-15",
                "return_1d": 0.01,
                "return_1m": 0.05,
                "return_3m": 0.12,
                "classification_basis": [{"provider": "FnGuide", "label": "조선기자재"}],
                "warning": "",
            }
        ],
        status="CACHED",
        show_refresh_button=True,
    )

    assert clicked is False
    assert markdown_calls
    assert "테마 렌즈" in markdown_calls[0]
    assert "대표 ETF 가격 기반 proxy" in markdown_calls[0]
    assert "canonical sector action" in markdown_calls[0]
    assert dataframe_payloads[0]["테마"].tolist() == ["조선"]
    assert dataframe_payloads[0]["대표 ETF"].tolist() == ["SOL 조선기자재 (0141S0)"]
    assert dataframe_payloads[0]["1M"].tolist() == ["+5.00%"]


def test_build_overview_review_candidates_uses_composite_order_and_preserves_policy():
    signals = [
        _signal("A", "Strong Buy", 1.10, 1.00, macro_fit=False, action_policy="KR_MOMENTUM_ONLY"),
        _signal("B", "Watch", 1.08, 1.00, macro_fit=True, action_policy="KR_MOMENTUM_ONLY"),
        _signal("C", "Hold", 1.06, 1.00, macro_fit=True, action_policy="KR_MOMENTUM_ONLY"),
        _signal("D", "Avoid", 1.04, 1.00),
        _signal("E", "N/A", 1.20, 1.00),
    ]
    for signal, score in zip(signals, [95.0, 80.0, 70.0, 50.0, 99.0], strict=True):
        signal.mom_percentile = score
        signal.flow_state = "unavailable"
    signals[1].sector_fit_rank = 2
    signals[1].sector_fit_total = 5
    signals[1].flow_state = "supportive"
    signals[1].flow_score = 1.2
    signals[2].sector_fit_rank = 1
    signals[2].sector_fit_total = 5

    frame = panels_module._build_overview_sector_frame(signals, sort_key="모멘텀 점수")
    candidates = panels_module._build_overview_review_candidates(signals, frame, limit=3)

    assert list(frame["섹터"].head(3)) == ["Sector A", "Sector B", "Sector C"]
    assert [candidate["sector_name"] for candidate in candidates] == ["Sector B", "Sector C", "Sector A"]
    assert all(candidate["action"] != "N/A" for candidate in candidates)
    assert len(candidates) == 3
    assert all(candidate["reason_parts"] for candidate in candidates)
    assert all(candidate["candidate_policy"] == "COMPOSITE_REVIEW_CANDIDATE" for candidate in candidates)
    assert all(candidate["action_policy"] == "KR_MOMENTUM_ONLY" for candidate in candidates)
    assert all(signal.action_policy == "KR_MOMENTUM_ONLY" for signal in signals[:3])
    assert "매크로 약점" in candidates[2]["warnings"]


def test_build_overview_review_candidate_groups_keeps_buy_and_sell_visible():
    buy = _signal("A", "Strong Buy", 1.10, 1.00, macro_fit=True, action_policy="KR_MOMENTUM_ONLY")
    buy.mom_percentile = 88.0
    buy.flow_state = "supportive"
    buy.flow_score = 1.0
    sell = _signal("B", "Avoid", 0.94, 1.00, macro_fit=False, action_policy="KR_MOMENTUM_ONLY")
    sell.mom_percentile = 20.0
    sell.rs_change_pct = -5.0
    sell.trend_ok = False
    sell.flow_state = "adverse"
    sell.flow_score = -1.0
    sell.volatility_20d = 0.32
    sell.mdd_3m = -0.22
    neutral = _signal("C", "Watch", 1.02, 1.00, macro_fit=True, action_policy="KR_MOMENTUM_ONLY")
    neutral.mom_percentile = 55.0
    neutral.flow_state = "unavailable"

    frame = panels_module._build_overview_sector_frame([buy, sell, neutral], sort_key="모멘텀 점수")
    groups = panels_module._build_overview_review_candidate_groups(
        [buy, sell, neutral],
        frame,
        limit_per_group=2,
    )

    assert [candidate["sector_name"] for candidate in groups["buy"]] == ["Sector A", "Sector C"]
    assert [candidate["sector_name"] for candidate in groups["sell"]] == ["Sector B"]
    assert all(candidate["review_side"] == "buy" for candidate in groups["buy"])
    assert groups["buy"][0]["review_side_label"] == "매수 검토 후보"
    assert groups["sell"][0]["edge_proxy"] < 0
    assert groups["sell"][0]["review_side"] == "sell"
    assert groups["sell"][0]["review_side_label"] == "매도 검토 후보"
    assert groups["sell"][0]["candidate_policy"] == "COMPOSITE_REVIEW_CANDIDATE"
    assert groups["sell"][0]["action_policy"] == "KR_MOMENTUM_ONLY"


def test_overview_review_candidate_group_does_not_default_neutral_to_buy():
    assert panels_module._overview_review_candidate_group(
        {"action": "Hold", "edge_proxy": 0.0, "turning_point_state": "Flat"}
    ) is None
    assert panels_module._overview_review_candidate_group(
        {"action": "Hold", "edge_proxy": -0.1, "turning_point_state": "Flat"}
    ) == "sell"
    assert panels_module._overview_review_candidate_group(
        {"action": "Watch", "edge_proxy": 0.1, "turning_point_state": "Flat"}
    ) == "buy"


def test_render_overview_review_candidates_renders_reasons_and_guardrail_copy(monkeypatch):
    markdown_calls: list[str] = []
    candidates = [
        {
            "sector_name": "KRX 반도체",
            "decision": "신규 검토 후보",
            "reason_parts": ["국면 적합", "3M 수익률 +12.00%"],
            "invalidation": "다음 검토 시까지 RS가 추세 하회 지속 시 무효화.",
            "action": "Strong Buy",
            "candidate_policy": "COMPOSITE_REVIEW_CANDIDATE",
            "candidate_score": 75.625,
            "metrics": [
                ("상방 proxy", "84.2"),
                ("하방 proxy", "19.8"),
                ("엣지 proxy", "+64.5"),
                ("변곡", "Continuation up"),
                ("복합점수", "75.6"),
            ],
        }
    ]

    monkeypatch.setattr(panels_module.st, "markdown", lambda text, **_: markdown_calls.append(text))

    panels_module._render_overview_review_candidates(candidates)

    assert markdown_calls
    markup = markdown_calls[0]
    assert "overview-review-candidates" in markup
    assert "검토 후보" in markup
    assert "proxy 근거 기준" in markup
    assert "기본 모멘텀 기준" not in markup
    assert "KRX 반도체" in markup
    assert "국면 적합" in markup
    metric_positions = [
        markup.rindex("상방 proxy"),
        markup.rindex("하방 proxy"),
        markup.rindex("엣지 proxy"),
        markup.rindex("변곡"),
        markup.rindex("복합점수"),
    ]
    assert metric_positions == sorted(metric_positions)
    assert "복합점수" in markup
    assert "75.6" in markup
    assert "Continuation up" in markup
    assert "보정 확률이 아니라 근거 점수" in markup
    assert "canonical action policy는 바꾸지 않습니다" in markup
    assert "상승 확률" not in markup
    assert "하락 확률" not in markup
    assert "다음 검토 시까지" in markup
    for forbidden in ("brokerage", "order", "매수 ETF", "실시간 거래", "보장"):
        assert forbidden not in markup


def test_render_overview_review_candidates_renders_buy_and_sell_groups(monkeypatch):
    markdown_calls: list[str] = []
    groups = {
        "buy": [
            {
                "sector_name": "KRX 반도체",
                "decision": "신규 검토 후보",
                "reason_parts": ["변곡 Continuation up", "엣지 +64.5"],
                "invalidation": "",
                "action": "Strong Buy",
                "candidate_policy": "COMPOSITE_REVIEW_CANDIDATE",
                "candidate_score": 75.625,
                "edge_proxy": 64.5,
                "metrics": [("상방 proxy", "84.2"), ("하방 proxy", "19.8")],
            }
        ],
        "sell": [
            {
                "sector_name": "KRX 건설",
                "decision": "이탈 검토",
                "reason_parts": ["변곡 Bearish turn", "엣지 -54.5"],
                "invalidation": "",
                "action": "Avoid",
                "candidate_policy": "COMPOSITE_REVIEW_CANDIDATE",
                "candidate_score": 32.0,
                "edge_proxy": -54.5,
                "metrics": [("상방 proxy", "23.2"), ("하방 proxy", "77.8")],
            }
        ],
    }

    monkeypatch.setattr(panels_module.st, "markdown", lambda text, **_: markdown_calls.append(text))

    panels_module._render_overview_review_candidates(groups)

    assert markdown_calls
    markup = markdown_calls[0]
    assert 'data-grouped="true"' in markup
    assert "매수 검토 후보" in markup
    assert "매도 검토 후보" in markup
    assert "KRX 반도체" in markup
    assert "KRX 건설" in markup
    assert "보정 확률이 아니라 근거 점수" in markup
    assert "canonical action policy는 바꾸지 않습니다" in markup
    assert markup.index("매수 검토 후보") < markup.index("매도 검토 후보")


def test_render_overview_review_candidates_empty_state(monkeypatch):
    markdown_calls: list[str] = []

    monkeypatch.setattr(panels_module.st, "markdown", lambda text, **_: markdown_calls.append(text))

    panels_module._render_overview_review_candidates([])

    assert markdown_calls
    assert 'data-empty="true"' in markdown_calls[0]
    assert "현재 복합 검토 기준에 맞는 섹터 후보가 없습니다." in markdown_calls[0]
    assert "기본 모멘텀 기준" not in markdown_calls[0]


def test_render_overview_mobile_decision_strip_uses_review_candidate_markup(monkeypatch):
    markdown_calls: list[str] = []
    frame = pd.DataFrame(
        [
            {"섹터": "KRX 반도체", "3M": 51.8, "액션": "신규 검토 후보"},
            {"섹터": "KRX 증권", "3M": 61.28, "액션": "관찰 후보"},
            {"섹터": "KRX 건설", "3M": 80.83, "액션": "유지"},
            {"섹터": "KRX 보험", "3M": 28.30, "액션": "회피"},
        ]
    )

    monkeypatch.setattr(panels_module.st, "markdown", lambda text, **_: markdown_calls.append(text))

    panels_module._render_overview_mobile_decision_strip(frame)

    assert markdown_calls
    markup = markdown_calls[0]
    assert "overview-review-candidates" in markup
    assert "검토 후보" in markup
    assert "KRX 반도체" in markup
    assert "KRX 증권" in markup
    assert "KRX 건설" in markup
    assert "KRX 보험" not in markup
    assert "overview-mobile-decision-strip" not in markup


def test_render_cycle_timeline_panel_returns_selected_phase(monkeypatch):
    markdown_calls: list[str] = []
    chart_calls: list[tuple[object, dict]] = []

    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: markdown_calls.append(text))
    monkeypatch.setattr(
        "src.ui.components.st.segmented_control",
        lambda *_args, **_kwargs: "SLOWDOWN_LATE",
    )
    monkeypatch.setattr(
        "src.ui.components.st.plotly_chart",
        lambda fig, **kwargs: chart_calls.append((fig, kwargs)),
    )

    phase = render_cycle_timeline_panel(
        segments=[
            {
                "phase_key": "RECOVERY_EARLY",
                "label": "Recovery / Early",
                "start": pd.Timestamp("2024-01-31"),
                "end": pd.Timestamp("2024-06-30"),
                "summary": "Top sector: Tech (+8.0%)",
                "is_current": False,
            },
            {
                "phase_key": "SLOWDOWN_LATE",
                "label": "Slowdown / Late",
                "start": pd.Timestamp("2024-07-31"),
                "end": pd.Timestamp("2024-12-31"),
                "summary": "Top sector: Staples (+4.1%)",
                "is_current": True,
            },
        ],
        selected_cycle_phase="ALL",
        theme_mode="dark",
    )

    assert phase == "SLOWDOWN_LATE"
    assert any("phase-chip-row" in call for call in markdown_calls)
    assert any("cycle-palette" in call for call in markdown_calls)
    assert len(chart_calls) == 1
    assert chart_calls[0][1]["width"] == "stretch"
    fig = chart_calls[0][0]
    assert fig.layout.title.text == "사이클 타임라인 (월별)"
    assert fig.layout.xaxis.tickformat == "%Y-%m"
    assert fig.layout.xaxis.dtick == "M1"
    assert len(fig.data) == 2
    assert all(getattr(trace, "fill", "") == "toself" for trace in fig.data)

    chart_calls.clear()
    phase = render_cycle_timeline_panel(
        segments=[
            {
                "phase_key": "RECOVERY_EARLY",
                "label": "Recovery / Early",
                "start": pd.Timestamp("2024-01-31"),
                "end": pd.Timestamp("2024-06-30"),
                "summary": "Top sector: Tech (+8.0%)",
                "is_current": False,
            }
        ],
        selected_cycle_phase="ALL",
        theme_mode="dark",
        locale="en",
    )

    assert phase == "SLOWDOWN_LATE"
    fig = chart_calls[0][0]
    assert fig.layout.title.text == "Cycle timeline (monthly)"
    assert len(fig.data) == 1
    assert "Status:" in fig.data[0].hovertemplate


def test_build_cycle_segments_normalizes_month_bounds_and_skips_nan():
    macro_result = pd.DataFrame(
        {
            "confirmed_regime": [
                "Recovery",
                "Recovery",
                None,
                "Indeterminate",
                "Expansion",
            ]
        },
        index=pd.to_datetime(
            ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31", "2025-06-30"]
        ),
    )
    monthly_close = pd.DataFrame(
        {
            "Sector A": [100.0, 102.0, 103.0, 104.0, 106.0],
            "Sector B": [100.0, 101.0, 99.0, 101.0, 103.0],
        },
        index=pd.to_datetime(
            ["2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31", "2025-06-30"]
        ),
    )

    segments, phase_by_month = _build_cycle_segments(
        macro_result=macro_result,
        monthly_close=monthly_close,
    )

    assert [segment["phase_key"] for segment in segments] == [
        "RECOVERY_EARLY",
        "RECOVERY_LATE",
        "INDETERMINATE",
        "EXPANSION_EARLY",
    ]
    assert pd.Timestamp(segments[0]["start"]) == pd.Timestamp("2025-02-01")
    assert pd.Timestamp(segments[0]["end"]) == pd.Timestamp("2025-02-28 23:59:59.999999999")
    assert pd.Timestamp(segments[1]["start"]) == pd.Timestamp("2025-03-01")
    assert pd.Timestamp(segments[1]["end"]) == pd.Timestamp("2025-03-31 23:59:59.999999999")
    assert pd.Timestamp(segments[2]["start"]) == pd.Timestamp("2025-05-01")
    assert pd.Timestamp(segments[2]["end"]) == pd.Timestamp("2025-05-31 23:59:59.999999999")
    assert phase_by_month.index.min() == pd.Timestamp("2025-02-01")
    assert pd.isna(phase_by_month.loc[pd.Timestamp("2025-04-01")])


def test_analysis_cache_loader_exposes_full_history_for_5y_and_all(monkeypatch):
    cached = pd.DataFrame(
        {
            "index_code": ["1001", "5044", "1001", "5044"],
            "index_name": ["KOSPI", "KRX Semiconductor", "KOSPI", "KRX Semiconductor"],
            "close": [100.0, 120.0, 300.0, 360.0],
        },
        index=pd.to_datetime(["2016-03-08", "2016-03-08", "2026-03-06", "2026-03-06"]),
    )
    cached.index.name = "trade_date"

    monkeypatch.setattr(dashboard_data_module, "read_market_prices", lambda *_args, **_kwargs: cached)

    history = dashboard_data_module.load_analysis_sector_prices_from_cache(
        "KR",
        end_date_str="20260306",
        benchmark_code="1001",
    )
    prices_wide = _build_prices_wide(
        sector_prices=history,
        sector_name_map={"1001": "KOSPI", "5044": "KRX Semiconductor"},
    )

    analysis_min_date = prices_wide.index.min().date()
    analysis_max_date = prices_wide.index.max().date()
    five_year_start, five_year_end = resolve_range_from_preset(
        max_date=analysis_max_date,
        min_date=analysis_min_date,
        preset="5Y",
    )
    all_start, all_end = resolve_range_from_preset(
        max_date=analysis_max_date,
        min_date=analysis_min_date,
        preset="ALL",
    )

    assert analysis_min_date == pd.Timestamp("2016-03-08").date()
    assert analysis_max_date == pd.Timestamp("2026-03-06").date()
    assert five_year_start == pd.Timestamp("2021-03-07").date()
    assert five_year_end == pd.Timestamp("2026-03-06").date()
    assert all_start == pd.Timestamp("2016-03-08").date()
    assert all_end == pd.Timestamp("2026-03-06").date()


def test_build_monthly_sector_returns_keeps_first_visible_month_non_null():
    prices_wide = pd.DataFrame(
        {
            "Sector A": [100.0, 110.0, 121.0],
            "KOSPI": [100.0, 102.0, 103.0],
        },
        index=pd.to_datetime(["2023-02-28", "2023-03-07", "2023-03-31"]),
    )

    monthly_close_full, monthly_returns_full = _build_monthly_sector_returns(
        prices_wide=prices_wide,
        sector_columns=["Sector A"],
    )
    visible = monthly_returns_full.loc[monthly_returns_full.index >= pd.Timestamp("2023-03-31")]

    assert list(monthly_close_full.index) == [pd.Timestamp("2023-02-28"), pd.Timestamp("2023-03-31")]
    assert not visible.empty
    assert not visible.iloc[0].isna().all()
    assert visible.iloc[0]["Sector A"] == pytest.approx(21.0)


def test_build_monthly_return_views_computes_excess_return_vs_kospi():
    prices_wide = pd.DataFrame(
        {
            "Sector A": [100.0, 110.0, 121.0],
            "Sector B": [100.0, 90.0, 99.0],
            "KOSPI": [100.0, 105.0, 110.25],
        },
        index=pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"]),
    )

    monthly_close_full, monthly_returns_full, benchmark_monthly_return, monthly_excess_returns_full = _build_monthly_return_views(
        prices_wide=prices_wide,
        sector_columns=["Sector A", "Sector B"],
        benchmark_label="KOSPI",
    )

    assert list(monthly_close_full.columns) == ["Sector A", "Sector B"]
    assert benchmark_monthly_return.loc[pd.Timestamp("2025-02-28")] == pytest.approx(5.0)
    assert monthly_returns_full.loc[pd.Timestamp("2025-02-28"), "Sector A"] == pytest.approx(10.0)
    assert monthly_excess_returns_full.loc[pd.Timestamp("2025-02-28"), "Sector A"] == pytest.approx(5.0)
    assert monthly_excess_returns_full.loc[pd.Timestamp("2025-03-31"), "Sector B"] == pytest.approx(5.0)


def test_filter_monthly_frame_for_analysis_excludes_trailing_partial_month():
    monthly_frame = pd.DataFrame(
        {"Sector A": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"]),
    )

    filtered = _filter_monthly_frame_for_analysis(
        monthly_frame=monthly_frame,
        start_date=pd.Timestamp("2025-01-01").date(),
        end_date=pd.Timestamp("2025-03-06").date(),
        selected_cycle_phase="ALL",
        phase_by_month=pd.Series(dtype=object),
    )

    assert list(filtered.index) == [
        pd.Timestamp("2025-01-31"),
        pd.Timestamp("2025-02-28"),
    ]


def test_build_heatmap_display_preserves_shared_visible_month_set():
    monthly_frame = pd.DataFrame(
        {"Sector A": [1.0, 2.0], "Sector B": [3.0, 4.0]},
        index=pd.to_datetime(["2025-01-31", "2025-02-28"]),
    )

    display = _build_heatmap_display(monthly_frame)

    assert list(display.columns) == ["2025-01", "2025-02"]
    assert list(display.index) == ["Sector A", "Sector B"]


def test_extract_heatmap_selection_returns_shared_month_and_sector_pair():
    event = {
        "selection": {
            "points": [
                {"customdata": ["2025-02", "Sector A"]},
            ]
        }
    }

    assert _extract_heatmap_selection(event) == ("2025-02", "Sector A")


def test_render_sector_detail_panel_returns_clicked_sector_and_preset(monkeypatch):
    chart_calls: list[tuple[object, dict]] = []

    monkeypatch.setattr("src.ui.components.st.columns", lambda spec, **_: [_DummyBlock() for _ in range(len(spec))])
    monkeypatch.setattr(
        "src.ui.components.st.markdown",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "src.ui.components.st.button",
        lambda label, **_: "Sector B" in label,
    )
    monkeypatch.setattr(
        "src.ui.components.st.segmented_control",
        lambda *_args, **_kwargs: "5Y",
    )
    monkeypatch.setattr(
        "src.ui.components.st.plotly_chart",
        lambda fig, **kwargs: chart_calls.append((fig, kwargs)),
    )

    detail_df = pd.DataFrame(
        {
            "Sector A": [100.0, 104.0, 108.0],
            "Sector B": [100.0, 103.0, 111.0],
            "KOSPI": [100.0, 101.0, 102.0],
        },
        index=pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"]),
    )
    detail_fig = build_sector_detail_figure(
        detail_df,
        selected_sector="Sector A",
        benchmark_label="KOSPI",
        comparison_sectors=["Sector B"],
        selected_month="2025-03",
        theme_mode="light",
    )

    selected_sector, selected_preset = render_sector_detail_panel(
        ranking_rows=[
            {"sector": "Sector A", "return_pct": 8.0},
            {"sector": "Sector B", "return_pct": 11.0},
        ],
        detail_figure=detail_fig,
        selected_sector="Sector A",
        selected_range_preset="1Y",
    )

    assert selected_sector == "Sector B"
    assert selected_preset == "5Y"
    assert len(chart_calls) == 1
