from __future__ import annotations

from datetime import date

from src.dashboard import state


def _normalize_range_preset(value: str | None) -> str:
    return str(value or "1Y").upper()


def _resolve_range_from_preset(*, max_date: date, min_date: date, preset: str) -> tuple[date, date]:
    if preset == "ALL":
        return min_date, max_date
    return date(2024, 1, 1), date(2024, 12, 31)


def test_ensure_session_defaults_populates_missing_keys():
    session = {}
    state.ensure_session_defaults(
        session,
        settings={"epsilon": 0.2, "rs_ma_period": 21, "ma_fast": 10, "ma_slow": 50, "price_years": 4},
        theme_key="theme_mode",
        default_theme_mode="dark",
        all_action_option="전체",
    )
    assert session["theme_mode"] == "dark"
    assert session["epsilon"] == 0.2
    assert session["price_years"] == 4
    assert session["selected_cycle_phase"] == "ALL"


def test_normalize_session_state_repairs_legacy_values():
    session = {
        "theme_mode": "light",
        "analysis_heatmap_palette": "unknown",
        "filter_action_global": "All",
        "selected_range_preset": "all",
    }
    palette = state.normalize_session_state(
        session,
        theme_key="theme_mode",
        theme_mode="dark",
        heatmap_palette_options=("classic", "contrast"),
        all_action_option="전체",
        normalize_range_preset=_normalize_range_preset,
    )
    assert palette == "classic"
    assert session["theme_mode"] == "dark"
    assert session["filter_action_global"] == "전체"
    assert session["selected_range_preset"] == "ALL"


def test_apply_analysis_toolbar_selection_updates_state_when_values_change():
    session = {"analysis_start_date": None, "analysis_end_date": None, "selected_range_preset": "1Y"}
    changed = state.apply_analysis_toolbar_selection(
        session,
        resolved_start=date(2024, 1, 1),
        resolved_end=date(2024, 12, 31),
        resolved_preset="CUSTOM",
    )
    assert changed is True
    assert session["analysis_start_date"] == date(2024, 1, 1)
    assert session["selected_range_preset"] == "CUSTOM"


def test_apply_detail_selection_updates_preset_range():
    session = {"selected_sector": "A", "selected_range_preset": "1Y"}
    changed = state.apply_detail_selection(
        session,
        chosen_sector="A",
        chosen_preset="ALL",
        analysis_max_date=date(2024, 12, 31),
        analysis_min_date=date(2020, 1, 1),
        normalize_range_preset=_normalize_range_preset,
        resolve_range_from_preset=_resolve_range_from_preset,
    )
    assert changed is True
    assert session["selected_range_preset"] == "ALL"
    assert session["analysis_start_date"] == date(2020, 1, 1)
    assert session["analysis_end_date"] == date(2024, 12, 31)
