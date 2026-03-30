"""Session-state helpers for dashboard orchestration."""
from __future__ import annotations

from datetime import date
from typing import Any, Iterable, Mapping, MutableMapping


SESSION_DEFAULTS: dict[str, Any] = {
    "market_id": "KR",
    "asof_date_str": "",
    "epsilon": 0.0,
    "rs_ma_period": 20,
    "ma_fast": 20,
    "ma_slow": 60,
    "price_years": 3,
    "filter_action_global": "",
    "filter_regime_only_global": False,
    "position_mode": "all",
    "show_alerted_only": False,
    "held_sectors": [],
    "selected_sector": "",
    "selected_month": "",
    "selected_cycle_phase": "ALL",
    "selected_range_preset": "1Y",
    "analysis_start_date": None,
    "analysis_end_date": None,
    "analysis_heatmap_palette": "classic",
}


def ensure_session_defaults(
    session_state: MutableMapping[str, Any],
    *,
    settings: Mapping[str, Any],
    theme_key: str,
    default_theme_mode: str,
    all_action_option: str,
) -> None:
    """Populate missing dashboard session defaults in place."""
    if "market_id" not in session_state:
        session_state["market_id"] = "KR"
    if theme_key not in session_state:
        session_state[theme_key] = default_theme_mode

    if "asof_date_str" not in session_state:
        session_state["asof_date_str"] = date.today().strftime("%Y%m%d")
    if "epsilon" not in session_state:
        session_state["epsilon"] = float(settings.get("epsilon", 0))
    if "rs_ma_period" not in session_state:
        session_state["rs_ma_period"] = int(settings.get("rs_ma_period", 20))
    if "ma_fast" not in session_state:
        session_state["ma_fast"] = int(settings.get("ma_fast", 20))
    if "ma_slow" not in session_state:
        session_state["ma_slow"] = int(settings.get("ma_slow", 60))
    if "price_years" not in session_state:
        session_state["price_years"] = int(settings.get("price_years", 3))
    if "filter_action_global" not in session_state:
        session_state["filter_action_global"] = all_action_option
    if "filter_regime_only_global" not in session_state:
        session_state["filter_regime_only_global"] = False
    if "position_mode" not in session_state:
        session_state["position_mode"] = "all"
    if "show_alerted_only" not in session_state:
        session_state["show_alerted_only"] = False
    if "held_sectors" not in session_state or not isinstance(session_state.get("held_sectors"), list):
        session_state["held_sectors"] = []
    if "selected_sector" not in session_state:
        session_state["selected_sector"] = ""
    if "selected_month" not in session_state:
        session_state["selected_month"] = ""
    if "selected_cycle_phase" not in session_state:
        session_state["selected_cycle_phase"] = "ALL"
    if "selected_range_preset" not in session_state:
        session_state["selected_range_preset"] = "1Y"
    if "analysis_start_date" not in session_state:
        session_state["analysis_start_date"] = None
    if "analysis_end_date" not in session_state:
        session_state["analysis_end_date"] = None
    if "analysis_heatmap_palette" not in session_state:
        session_state["analysis_heatmap_palette"] = "classic"


def apply_market_selection(
    session_state: MutableMapping[str, Any],
    *,
    market_id: str,
) -> bool:
    """Persist a market change and clear state tied to the previous universe."""
    normalized = str(market_id or "KR").strip().upper() or "KR"
    if str(session_state.get("market_id", "KR")).strip().upper() == normalized:
        return False

    session_state["market_id"] = normalized
    session_state["held_sectors"] = []
    session_state["selected_sector"] = ""
    session_state["selected_month"] = ""
    session_state["selected_cycle_phase"] = "ALL"
    session_state["selected_range_preset"] = "1Y"
    session_state["analysis_start_date"] = None
    session_state["analysis_end_date"] = None
    return True


def normalize_session_state(
    session_state: MutableMapping[str, Any],
    *,
    theme_key: str,
    theme_mode: str,
    heatmap_palette_options: Iterable[str],
    all_action_option: str,
    normalize_range_preset,
) -> str:
    """Normalize state values and return the active analysis heatmap palette."""
    if session_state.get(theme_key) != theme_mode:
        session_state[theme_key] = theme_mode

    analysis_heatmap_palette = str(session_state.get("analysis_heatmap_palette", "classic")).strip().lower()
    valid_palettes = tuple(str(value) for value in heatmap_palette_options)
    if analysis_heatmap_palette not in valid_palettes:
        analysis_heatmap_palette = "classic"
        session_state["analysis_heatmap_palette"] = analysis_heatmap_palette

    if session_state.get("filter_action_global") == "All":
        session_state["filter_action_global"] = all_action_option
    elif "filter_action_global" not in session_state:
        session_state["filter_action_global"] = all_action_option

    position_mode = str(session_state.get("position_mode", "all")).strip().lower()
    if position_mode not in {"all", "held", "new"}:
        position_mode = "all"
    session_state["position_mode"] = position_mode

    held_sectors = session_state.get("held_sectors", [])
    if not isinstance(held_sectors, list):
        held_sectors = []
    session_state["held_sectors"] = [
        str(item).strip()
        for item in held_sectors
        if str(item).strip()
    ]

    session_state["show_alerted_only"] = bool(session_state.get("show_alerted_only", False))

    normalized_range_preset = normalize_range_preset(session_state.get("selected_range_preset"))
    if session_state.get("selected_range_preset") != normalized_range_preset:
        session_state["selected_range_preset"] = normalized_range_preset

    return analysis_heatmap_palette


def parse_asof_default(session_state: Mapping[str, Any]) -> date:
    """Return the sidebar default `asof_date` from state, with today fallback."""
    try:
        return date(
            int(str(session_state["asof_date_str"])[:4]),
            int(str(session_state["asof_date_str"])[4:6]),
            int(str(session_state["asof_date_str"])[6:8]),
        )
    except Exception:
        return date.today()


def ensure_analysis_bounds(
    session_state: MutableMapping[str, Any],
    *,
    analysis_min_date: date,
    analysis_max_date: date,
    normalize_range_preset,
    resolve_range_from_preset,
) -> None:
    """Ensure analysis start/end dates exist in session state."""
    if not session_state.get("analysis_end_date"):
        session_state["analysis_end_date"] = analysis_max_date
    if not session_state.get("analysis_start_date"):
        preset_start, preset_end = resolve_range_from_preset(
            max_date=analysis_max_date,
            min_date=analysis_min_date,
            preset=normalize_range_preset(session_state.get("selected_range_preset", "1Y")),
        )
        session_state["analysis_start_date"] = preset_start
        session_state["analysis_end_date"] = preset_end


def apply_analysis_toolbar_selection(
    session_state: MutableMapping[str, Any],
    *,
    resolved_start: date,
    resolved_end: date,
    resolved_preset: str,
) -> bool:
    """Persist toolbar-driven range selection; return True when state changed."""
    changed = (
        session_state.get("analysis_start_date") != resolved_start
        or session_state.get("analysis_end_date") != resolved_end
        or session_state.get("selected_range_preset") != resolved_preset
    )
    if changed:
        session_state["analysis_start_date"] = resolved_start
        session_state["analysis_end_date"] = resolved_end
        session_state["selected_range_preset"] = resolved_preset
    return changed


def ensure_visible_month_selection(
    session_state: MutableMapping[str, Any],
    *,
    visible_months: list[str],
) -> None:
    """Ensure selected month stays aligned with the visible heatmap columns."""
    if visible_months:
        if session_state.get("selected_month") not in visible_months:
            session_state["selected_month"] = visible_months[-1]
    else:
        session_state["selected_month"] = ""


def apply_heatmap_selection(
    session_state: MutableMapping[str, Any],
    *,
    selection: tuple[str, str] | None,
) -> bool:
    """Apply a shared (month, sector) heatmap selection to state."""
    if selection is None:
        return False
    month_value, sector_value = selection
    changed = (
        month_value != str(session_state.get("selected_month", ""))
        or sector_value != str(session_state.get("selected_sector", ""))
    )
    if changed:
        session_state["selected_month"] = month_value
        session_state["selected_sector"] = sector_value
    return changed


def apply_cycle_phase_selection(
    session_state: MutableMapping[str, Any],
    *,
    chosen_cycle_phase: str,
) -> bool:
    """Apply a cycle-phase change from the timeline widget."""
    changed = str(session_state.get("selected_cycle_phase", "ALL") or "ALL") != chosen_cycle_phase
    if changed:
        session_state["selected_cycle_phase"] = chosen_cycle_phase
    return changed


def normalize_ranked_sector_selection(
    session_state: MutableMapping[str, Any],
    *,
    ranking_rows: list[dict[str, object]],
) -> None:
    """Keep the selected sector aligned with the ranking list."""
    if ranking_rows:
        selected_sector_state = str(session_state.get("selected_sector", ""))
        if selected_sector_state not in [str(row["sector"]) for row in ranking_rows]:
            session_state["selected_sector"] = str(ranking_rows[0]["sector"])
    else:
        session_state["selected_sector"] = ""


def apply_detail_selection(
    session_state: MutableMapping[str, Any],
    *,
    chosen_sector: str,
    chosen_preset: str,
    analysis_max_date: date,
    analysis_min_date: date,
    normalize_range_preset,
    resolve_range_from_preset,
) -> bool:
    """Apply detail-panel selection changes; return True when rerun is required."""
    if chosen_sector and chosen_sector != str(session_state.get("selected_sector", "")):
        session_state["selected_sector"] = chosen_sector
        return True

    current_preset = normalize_range_preset(session_state.get("selected_range_preset", "1Y"))
    if chosen_preset != current_preset:
        preset_start, preset_end = resolve_range_from_preset(
            max_date=analysis_max_date,
            min_date=analysis_min_date,
            preset=chosen_preset,
        )
        session_state["selected_range_preset"] = chosen_preset
        session_state["analysis_start_date"] = preset_start
        session_state["analysis_end_date"] = preset_end
        return True

    return False
