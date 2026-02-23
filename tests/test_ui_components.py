"""UI component rendering tests for momentum visuals."""
from __future__ import annotations

from src.signals.matrix import SectorSignal
from src.ui.components import (
    render_action_summary,
    render_rs_momentum_bar,
    render_rs_scatter,
)
from src.ui.styles import ACTION_COLORS, BLUE, DARK_GREY, GREY


def _signal(code: str, action: str, rs: float, rs_ma: float) -> SectorSignal:
    return SectorSignal(
        index_code=code,
        sector_name=f"Sector {code}",
        macro_regime="Recovery",
        macro_fit=True,
        rs=rs,
        rs_ma=rs_ma,
        rs_strong=False,
        trend_ok=False,
        momentum_strong=False,
        rsi_d=50.0,
        rsi_w=50.0,
        action=action,
        alerts=[],
        returns={},
        asof_date="2024-01-31",
    )


def test_render_rs_scatter_filters_nan_points():
    signals = [
        _signal("A", "Watch", 1.10, 1.00),
        _signal("B", "Strong Buy", float("nan"), 1.00),
        _signal("C", "N/A", 1.20, 1.10),
    ]

    fig = render_rs_scatter(signals)
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [1.10]
    assert list(fig.data[0].y) == [1.00]
    assert list(fig.data[0].marker.color) == [ACTION_COLORS["Watch"]]


def test_render_rs_scatter_empty_annotation_when_no_valid_points():
    signals = [
        _signal("A", "N/A", 1.10, 1.00),
        _signal("B", "Watch", float("nan"), 1.00),
    ]

    fig = render_rs_scatter(signals)
    assert len(fig.data) == 0
    annotations = list(fig.layout.annotations or [])
    assert annotations
    assert "RS/RS MA" in annotations[0].text


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

    fig = render_rs_momentum_bar(signals)
    assert len(fig.data) == 0


def test_render_rs_momentum_bar_valid_data_has_percent_suffix():
    signals = [
        _signal("A", "Watch", 1.10, 1.00),
        _signal("B", "Hold", 0.95, 1.00),
    ]

    fig = render_rs_momentum_bar(signals)
    assert len(fig.data) == 1
    assert fig.layout.xaxis.ticksuffix == "%"


def test_action_colors_watch_hold_mapping():
    assert ACTION_COLORS["Watch"] == BLUE
    assert ACTION_COLORS["Hold"] == GREY
    assert ACTION_COLORS["N/A"] == DARK_GREY


class _DummyColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_render_action_summary_renders_metrics_and_bar(monkeypatch):
    signals = [
        _signal("A", "Strong Buy", 1.10, 1.00),
        _signal("B", "Watch", 1.05, 1.00),
        _signal("C", "Watch", 1.01, 1.00),
        _signal("D", "Hold", 0.98, 1.00),
        _signal("E", "Avoid", 0.94, 1.00),
        _signal("F", "N/A", 1.00, 1.00),
    ]

    metric_calls: list[tuple[str, int]] = []
    chart_calls: list[tuple[object, bool]] = []

    monkeypatch.setattr(
        "src.ui.components.st.columns",
        lambda n: [_DummyColumn() for _ in range(n)],
    )
    monkeypatch.setattr(
        "src.ui.components.st.metric",
        lambda label, value, *_, **__: metric_calls.append((label, value)),
    )
    monkeypatch.setattr(
        "src.ui.components.st.plotly_chart",
        lambda fig, use_container_width=False: chart_calls.append((fig, use_container_width)),
    )

    render_action_summary(signals)

    assert ("Total", 6) in metric_calls
    assert ("Strong Buy", 1) in metric_calls
    assert ("Watch", 2) in metric_calls
    assert ("Hold", 1) in metric_calls
    assert ("Avoid", 1) in metric_calls
    assert ("N/A", 1) in metric_calls
    assert len(chart_calls) == 1
    fig, use_container_width = chart_calls[0]
    assert use_container_width is True
    assert list(fig.data[0].x) == ["Strong Buy", "Watch", "Hold", "Avoid", "N/A"]
    assert list(fig.data[0].y) == [1, 2, 1, 1, 1]
    assert list(fig.data[0].marker.color) == [
        ACTION_COLORS["Strong Buy"],
        ACTION_COLORS["Watch"],
        ACTION_COLORS["Hold"],
        ACTION_COLORS["Avoid"],
        ACTION_COLORS["N/A"],
    ]


def test_render_action_summary_handles_empty_signal_list(monkeypatch):
    info_calls: list[str] = []
    chart_calls: list[object] = []

    monkeypatch.setattr("src.ui.components.st.info", lambda text: info_calls.append(text))
    monkeypatch.setattr(
        "src.ui.components.st.plotly_chart",
        lambda fig, use_container_width=False: chart_calls.append(fig),
    )

    render_action_summary([])

    assert info_calls
    assert info_calls[0] == "신호 데이터 없음"
    assert not chart_calls
