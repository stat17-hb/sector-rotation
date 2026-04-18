"""UI component rendering tests for dashboard visuals and layout helpers."""
from __future__ import annotations

import pandas as pd
import pytest
import app as app_module
import src.ui.panels as panels_module

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
    render_page_header,
    render_panel_header,
    render_rs_momentum_bar,
    render_rs_scatter,
    render_signal_table,
    render_sector_detail_panel,
    render_status_card_row,
    render_status_strip,
    render_top_bar_filters,
    render_top_picks_table,
    resolve_range_from_preset,
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
        rsi_d=50.0,
        rsi_w=50.0,
        action=action,
        alerts=list(alerts or []),
        returns=returns or {"1M": 0.05, "3M": 0.10},
        volatility_20d=0.12,
        mdd_3m=-0.08,
        asof_date="2024-01-31",
        is_provisional=is_provisional,
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
    assert "Korea Sector Rotation" in markdown_calls[0]
    assert "page-shell__pill" in markdown_calls[0]


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
    assert "Cache fallback" in markdown_calls[0]
    assert any("KRX: HTTP_ERROR" in call for call in write_calls)


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
        lambda _label, options, default, format_func, selection_mode, key, width: session_state.__setitem__(key, options[1]),
    )
    monkeypatch.setattr(
        "src.ui.components.st.markdown",
        lambda text, **_: markdown_calls.append(text),
    )

    action, regime_only, position_mode, alerted_only = render_top_bar_filters(
        current_regime="Recovery",
        action_options=[ALL_ACTION_KEY, "Strong Buy", "Watch"],
        is_mobile=False,
    )

    assert action == "Strong Buy"
    assert regime_only is True
    assert position_mode == "held"
    assert alerted_only is True
    assert any("command-bar" in call for call in markdown_calls)
    assert any("top-bar-summary" in call for call in markdown_calls)
    assert any("Recovery" in call for call in markdown_calls)


def test_render_decision_hero_renders_regime_and_provisional_badge(monkeypatch):
    markdown_calls: list[str] = []

    monkeypatch.setattr("src.ui.components.st.markdown", lambda text, **_: markdown_calls.append(text))

    render_decision_hero(
        regime="Recovery",
        regime_is_confirmed=False,
        growth_val=101.25,
        inflation_val=2.1,
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

    markdown_calls.clear()
    render_decision_hero(
        regime="Recovery",
        regime_is_confirmed=False,
        growth_val=101.25,
        inflation_val=2.1,
        fx_change=1.4,
        is_provisional=True,
        theme_mode="light",
        locale="en",
    )

    assert "Includes provisional macro prints" in markdown_calls[0]
    assert "Rules-based heuristic" in markdown_calls[0]
    assert "confirmed_regime primary" in markdown_calls[0]
    assert "Historical docs separated" in markdown_calls[0]


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
    assert held_view["decision"] == "추가 매수 후보"
    assert "국면 적합" in held_view["reason"]
    assert held_view["alerts_text"] == "Overheat"
    assert new_view["held"] is False
    assert new_view["decision"] == "신규 매수 후보"
    assert english_view["decision"] == "Add candidate"
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


def test_render_investor_flow_summary_marks_reference_only_preview(monkeypatch):
    warning_calls: list[str] = []
    caption_calls: list[str] = []
    info_calls: list[str] = []

    monkeypatch.setattr(panels_module.st, "container", lambda **_: _DummyBlock())
    monkeypatch.setattr(panels_module, "render_panel_header", lambda **kwargs: None)
    monkeypatch.setattr(panels_module.st, "warning", lambda text: warning_calls.append(text))
    monkeypatch.setattr(panels_module.st, "caption", lambda text: caption_calls.append(text))
    monkeypatch.setattr(panels_module.st, "info", lambda text: info_calls.append(text))
    monkeypatch.setattr(panels_module.st, "markdown", lambda *args, **kwargs: None)

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

    assert any("최종 투자판단에는 반영되지 않았습니다" in text for text in warning_calls)
    assert not any("표시할 투자자 수급 데이터가 없습니다" in text for text in info_calls)
    assert not any("투자자 수급 탭" in text for text in caption_calls)


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


def test_render_investor_flow_summary_uses_raw_snapshot_fallback_when_signal_flow_is_missing(monkeypatch):
    markdown_calls: list[str] = []

    monkeypatch.setattr(panels_module.st, "container", lambda **_: _DummyBlock())
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
    assert blue_orange[0][1] == "#0066CC"


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

    segments, phase_by_month = app_module._build_cycle_segments(
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

    monkeypatch.setattr(app_module, "read_market_prices", lambda *_args, **_kwargs: cached)

    history = app_module._load_analysis_sector_prices_from_cache(
        end_date_str="20260306",
        benchmark_code="1001",
    )
    prices_wide = app_module._build_prices_wide(
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

    monthly_close_full, monthly_returns_full = app_module._build_monthly_sector_returns(
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

    monthly_close_full, monthly_returns_full, benchmark_monthly_return, monthly_excess_returns_full = app_module._build_monthly_return_views(
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

    filtered = app_module._filter_monthly_frame_for_analysis(
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

    display = app_module._build_heatmap_display(monthly_frame)

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

    assert app_module._extract_heatmap_selection(event) == ("2025-02", "Sector A")


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
