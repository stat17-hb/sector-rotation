"""Session-state helpers for dashboard orchestration."""
from __future__ import annotations

from datetime import date
from typing import Any, Iterable, Mapping, MutableMapping

from src.signals.flow import normalize_flow_profile
from src.ui.copy import ALL_ACTION_KEY, normalize_action_filter


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
    "flow_profile": "foreign_lead",
    "stock_lookup_query": "",
    "stock_lookup_status": "",
    "stock_lookup_message": "",
    "stock_lookup_result": {},
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
        session_state["filter_action_global"] = normalize_action_filter(all_action_option)
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
    if "flow_profile" not in session_state:
        session_state["flow_profile"] = "foreign_lead"
    if "stock_lookup_query" not in session_state:
        session_state["stock_lookup_query"] = ""
    if "stock_lookup_status" not in session_state:
        session_state["stock_lookup_status"] = ""
    if "stock_lookup_message" not in session_state:
        session_state["stock_lookup_message"] = ""
    if "stock_lookup_result" not in session_state or not isinstance(session_state.get("stock_lookup_result"), dict):
        session_state["stock_lookup_result"] = {}


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
    session_state["stock_lookup_query"] = ""
    session_state["stock_lookup_status"] = ""
    session_state["stock_lookup_message"] = ""
    session_state["stock_lookup_result"] = {}
    return True


def normalize_stock_lookup_result(result: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(result or {})
    normalized = {
        "status": str(payload.get("status", "")).strip().lower(),
        "market": str(payload.get("market", "")).strip().upper(),
        "query": str(payload.get("query", "")).strip(),
        "normalized_query": str(payload.get("normalized_query", "")).strip(),
        "matched_symbol": str(payload.get("matched_symbol", "")).strip(),
        "matched_name": str(payload.get("matched_name", "")).strip(),
        "sector_code": str(payload.get("sector_code", "")).strip(),
        "sector_name": str(payload.get("sector_name", "")).strip(),
        "resolution_kind": str(payload.get("resolution_kind", "")).strip(),
        "source": str(payload.get("source", "")).strip(),
        "confidence": str(payload.get("confidence", "")).strip(),
        "explanation": str(payload.get("explanation", "")).strip(),
        "canonicalization_applied": bool(payload.get("canonicalization_applied", False)),
        "canonicalization_basis": str(payload.get("canonicalization_basis", "not_applicable") or "not_applicable").strip(),
        "match_effective_date": str(payload.get("match_effective_date", "")).strip(),
        "match_date_mode": str(payload.get("match_date_mode", "not_applicable") or "not_applicable").strip(),
        "matched_sector_candidates": [],
    }
    candidates = payload.get("matched_sector_candidates", [])
    if isinstance(candidates, list):
        normalized["matched_sector_candidates"] = [
            {
                "sector_code": str(dict(item or {}).get("sector_code", "")).strip(),
                "sector_name": str(dict(item or {}).get("sector_name", "")).strip(),
                "lookup_priority": dict(item or {}).get("lookup_priority"),
                "source": str(dict(item or {}).get("source", "")).strip(),
                "resolved_from": str(dict(item or {}).get("resolved_from", "")).strip(),
                "snapshot_date": str(dict(item or {}).get("snapshot_date", "")).strip(),
            }
            for item in candidates
            if isinstance(item, Mapping)
        ]
    return normalized


def _sector_name_lookup(sector_map: Mapping[str, Any] | None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for regime_payload in dict(sector_map or {}).get("regimes", {}).values():
        for item in list(dict(regime_payload or {}).get("sectors", [])):
            code = str(dict(item or {}).get("code", "")).strip()
            name = str(dict(item or {}).get("name", "")).strip()
            if code and name and code not in lookup:
                lookup[code] = name
    return lookup


def _resolve_display_sector_name(
    sector_code: str,
    sector_name: str,
    *,
    name_lookup: Mapping[str, str],
) -> str:
    code = str(sector_code or "").strip()
    name = str(sector_name or "").strip()
    if code and (not name or name == code):
        return str(name_lookup.get(code, code)).strip()
    return name


def _taxonomy_traceability_fields(taxonomy_lookup: Mapping[str, Any], sector_code: str) -> dict[str, Any]:
    context = taxonomy_lookup.get(str(sector_code or "").strip())
    if context is None:
        return {}
    return {
        "taxonomy_label": context.taxonomy_label,
        "taxonomy_base_labels": list(context.base_labels),
        "taxonomy_cross_labels": list(context.cross_labels),
        "taxonomy_theme_labels": list(context.theme_labels),
    }


def apply_stock_lookup_result(
    session_state: MutableMapping[str, Any],
    *,
    result: Mapping[str, Any],
) -> bool:
    """Persist a stock-lookup result; return True when selected sector changed."""
    normalized = normalize_stock_lookup_result(result)
    status = str(normalized.get("status", "")).strip().lower()
    query = str(normalized.get("query", "")).strip()
    explanation = str(normalized.get("explanation", "")).strip()
    sector_name = str(normalized.get("sector_name", "")).strip()

    session_state["stock_lookup_query"] = query
    session_state["stock_lookup_status"] = status
    session_state["stock_lookup_message"] = explanation
    session_state["stock_lookup_result"] = normalized

    if status != "success" or not sector_name:
        return False

    changed = sector_name != str(session_state.get("selected_sector", "")).strip()
    session_state["selected_sector"] = sector_name
    return changed


def build_stock_lookup_display_model(
    stock_lookup_result: Mapping[str, Any] | None,
    sector_map: Mapping[str, Any] | None,
    taxonomy_context: Any | None = None,
) -> dict[str, Any]:
    normalized = normalize_stock_lookup_result(stock_lookup_result)
    name_lookup = _sector_name_lookup(sector_map)
    taxonomy_lookup = taxonomy_context.by_sector_code() if hasattr(taxonomy_context, "by_sector_code") else {}
    candidates = [
        {
            **dict(item),
            "sector_name": _resolve_display_sector_name(
                str(dict(item).get("sector_code", "")).strip(),
                str(dict(item).get("sector_name", "")).strip(),
                name_lookup=name_lookup,
            ),
            **_taxonomy_traceability_fields(taxonomy_lookup, str(dict(item).get("sector_code", "")).strip()),
        }
        for item in list(normalized.get("matched_sector_candidates", []))
    ]
    canonical_code = str(normalized.get("sector_code", "")).strip()
    canonical_name = _resolve_display_sector_name(
        canonical_code,
        str(normalized.get("sector_name", "")).strip(),
        name_lookup=name_lookup,
    )
    matched_sectors = sorted(
        [
            item for item in candidates
            if str(dict(item).get("sector_code", "")).strip()
        ],
        key=lambda item: (
            0 if str(dict(item).get("sector_code", "")).strip() == canonical_code else 1,
            int(dict(item).get("lookup_priority")) if dict(item).get("lookup_priority") is not None else 9999,
            str(dict(item).get("sector_name", "")).strip(),
        ),
    )
    return {
        "result": normalized,
        "canonical_sector": {
            "sector_code": canonical_code,
            "sector_name": canonical_name,
            **_taxonomy_traceability_fields(taxonomy_lookup, canonical_code),
        },
        "matched_sectors": matched_sectors,
    }


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
        session_state["filter_action_global"] = ALL_ACTION_KEY
    elif "filter_action_global" not in session_state:
        session_state["filter_action_global"] = normalize_action_filter(all_action_option)
    else:
        session_state["filter_action_global"] = normalize_action_filter(session_state.get("filter_action_global"))

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
    session_state["flow_profile"] = normalize_flow_profile(session_state.get("flow_profile"))

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
