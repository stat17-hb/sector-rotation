from __future__ import annotations

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
        yield_curve_status="Normal",
        signals=[],
        held_sector_options=["Sector A", "Sector B"],
        action_options=["ALL", "Watch"],
        is_mobile_client=False,
        analysis_canvas_kwargs={"heatmap_return_display": None},
    )

    assert calls == ["hero", "status", "boards", "filters", "analysis"]
    assert result == (["Sector A"], "Watch", True, "held", True)
