from __future__ import annotations

from datetime import date

from src.dashboard import state
from src.ui.copy import ALL_ACTION_KEY


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
        all_action_option=ALL_ACTION_KEY,
    )
    assert session["theme_mode"] == "dark"
    assert session["epsilon"] == 0.2
    assert session["price_years"] == 4
    assert session["position_mode"] == "all"
    assert session["show_alerted_only"] is False
    assert session["held_sectors"] == []
    assert session["selected_cycle_phase"] == "ALL"
    assert session["flow_profile"] == "foreign_lead"
    assert session["stock_lookup_query"] == ""
    assert session["stock_lookup_status"] == ""
    assert session["stock_lookup_message"] == ""
    assert session["stock_lookup_result"] == {}
    assert "momentum_method" not in session


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
        all_action_option=ALL_ACTION_KEY,
        normalize_range_preset=_normalize_range_preset,
    )
    assert palette == "classic"
    assert session["theme_mode"] == "dark"
    assert session["filter_action_global"] == ALL_ACTION_KEY
    assert session["selected_range_preset"] == "ALL"
    assert session["position_mode"] == "all"
    assert session["flow_profile"] == "foreign_lead"


def test_normalize_session_state_repairs_localized_all_filter_to_key():
    session = {"filter_action_global": "전체"}

    state.normalize_session_state(
        session,
        theme_key="theme_mode",
        theme_mode="dark",
        heatmap_palette_options=("classic",),
        all_action_option=ALL_ACTION_KEY,
        normalize_range_preset=_normalize_range_preset,
    )

    assert session["filter_action_global"] == ALL_ACTION_KEY


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


def test_apply_market_selection_clears_held_sector_context():
    session = {
        "market_id": "KR",
        "held_sectors": ["KRX Semiconductor"],
        "selected_sector": "KRX Semiconductor",
        "selected_month": "2025-01",
        "stock_lookup_query": "005930",
        "stock_lookup_status": "success",
        "stock_lookup_message": "done",
        "stock_lookup_result": {"status": "success"},
    }

    changed = state.apply_market_selection(session, market_id="US")

    assert changed is True
    assert session["market_id"] == "US"
    assert session["held_sectors"] == []
    assert session["selected_sector"] == ""
    assert session["stock_lookup_query"] == ""
    assert session["stock_lookup_status"] == ""
    assert session["stock_lookup_message"] == ""
    assert session["stock_lookup_result"] == {}


def test_apply_stock_lookup_result_updates_selected_sector_on_success():
    session = {"selected_sector": "Old Sector"}

    changed = state.apply_stock_lookup_result(
        session,
        result={
            "status": "success",
            "query": "005930",
            "sector_name": "KRX 반도체",
            "explanation": "resolved",
        },
    )

    assert changed is True
    assert session["selected_sector"] == "KRX 반도체"
    assert session["stock_lookup_query"] == "005930"
    assert session["stock_lookup_status"] == "success"
    assert session["stock_lookup_message"] == "resolved"
    assert session["stock_lookup_result"]["status"] == "success"


def test_apply_stock_lookup_result_preserves_selected_sector_on_non_success():
    session = {"selected_sector": "Keep Sector"}

    changed = state.apply_stock_lookup_result(
        session,
        result={
            "status": "ambiguous",
            "query": "Apple",
            "sector_name": "",
            "explanation": "ambiguous",
        },
    )

    assert changed is False
    assert session["selected_sector"] == "Keep Sector"
    assert session["stock_lookup_query"] == "Apple"
    assert session["stock_lookup_status"] == "ambiguous"
    assert session["stock_lookup_message"] == "ambiguous"
    assert session["stock_lookup_result"]["canonicalization_basis"] == "not_applicable"


def test_build_stock_lookup_display_model_returns_all_matched_sectors_with_canonical_first():
    model = state.build_stock_lookup_display_model(
        {
            "status": "success",
            "sector_code": "5044",
            "sector_name": "KRX 반도체",
            "matched_sector_candidates": [
                {"sector_code": "1155", "sector_name": "KOSPI200 정보기술", "lookup_priority": 20, "source": "raw", "resolved_from": "20260417", "snapshot_date": "20260417"},
                {"sector_code": "5044", "sector_name": "KRX 반도체", "lookup_priority": 10, "source": "raw", "resolved_from": "20260417", "snapshot_date": "20260417"},
                {"sector_code": "5042", "sector_name": "KRX 산업재", "lookup_priority": 30, "source": "raw", "resolved_from": "20260417", "snapshot_date": "20260417"},
            ],
        },
        {
            "regimes": {
                "Recovery": {"sectors": [{"code": "1155", "name": "KOSPI200 정보기술"}]},
                "Expansion": {
                    "sectors": [
                        {"code": "5044", "name": "KRX 반도체", "lookup_display_parent_code": "1155"},
                        {"code": "5042", "name": "KRX 산업재"},
                    ]
                },
            }
        },
    )

    assert [item["sector_code"] for item in model["matched_sectors"]] == ["5044", "1155", "5042"]


def test_build_stock_lookup_display_model_keeps_only_present_matches():
    model = state.build_stock_lookup_display_model(
        {
            "status": "success",
            "sector_code": "5044",
            "sector_name": "KRX 반도체",
            "matched_sector_candidates": [
                {"sector_code": "5044", "sector_name": "KRX 반도체", "lookup_priority": 10, "source": "raw", "resolved_from": "20260417", "snapshot_date": "20260417"},
            ],
        },
        {
            "regimes": {
                "Expansion": {
                    "sectors": [
                        {"code": "5044", "name": "KRX 반도체", "lookup_display_parent_code": "1155"},
                        {"code": "1155", "name": "KOSPI200 정보기술"},
                    ]
                }
            }
        },
    )

    assert [item["sector_code"] for item in model["matched_sectors"]] == ["5044"]


def test_build_stock_lookup_display_model_enriches_code_only_names_from_sector_map():
    model = state.build_stock_lookup_display_model(
        {
            "status": "success",
            "sector_code": "5044",
            "sector_name": "5044",
            "matched_sector_candidates": [
                {"sector_code": "1155", "sector_name": "1155", "lookup_priority": 20, "source": "raw", "resolved_from": "20260417", "snapshot_date": "20260417"},
                {"sector_code": "5044", "sector_name": "5044", "lookup_priority": 10, "source": "raw", "resolved_from": "20260417", "snapshot_date": "20260417"},
            ],
        },
        {
            "regimes": {
                "Recovery": {"sectors": [{"code": "1155", "name": "KOSPI200 정보기술"}]},
                "Expansion": {"sectors": [{"code": "5044", "name": "KRX 반도체"}]},
            }
        },
    )

    assert model["canonical_sector"]["sector_name"] == "KRX 반도체"
    assert [item["sector_name"] for item in model["matched_sectors"]] == [
        "KRX 반도체",
        "KOSPI200 정보기술",
    ]


def test_build_stock_lookup_display_model_adds_taxonomy_traceability():
    from config.markets import load_market_configs
    from src.dashboard.theme_taxonomy_adapter import build_taxonomy_dashboard_model

    _, sector_map, _, _ = load_market_configs("KR")
    taxonomy_context = build_taxonomy_dashboard_model(sector_map=sector_map, market="KR")

    model = state.build_stock_lookup_display_model(
        {
            "status": "success",
            "sector_code": "5044",
            "sector_name": "5044",
            "matched_sector_candidates": [
                {"sector_code": "5044", "sector_name": "5044", "lookup_priority": 10},
            ],
        },
        sector_map,
        taxonomy_context,
    )

    assert model["canonical_sector"]["taxonomy_label"] == "KRX 반도체"
    assert model["canonical_sector"]["taxonomy_base_labels"] == ["반도체"]
    assert model["matched_sectors"][0]["taxonomy_theme_labels"] == ["반도체"]
