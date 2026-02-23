"""UI component rendering tests for momentum visuals."""
from __future__ import annotations

from src.signals.matrix import SectorSignal
from src.ui.components import render_rs_momentum_bar, render_rs_scatter
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
