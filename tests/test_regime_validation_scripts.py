from __future__ import annotations

import pandas as pd
import io
from contextlib import redirect_stdout

from scripts import evaluate_regime_validity
from scripts import validate_sector_mapping
from src.macro.series_utils import build_regime_history_from_macro


def _macro_cfg() -> dict:
    return {
        "kosis": {
            "leading_index": {"enabled": True, "org_id": "101", "tbl_id": "L", "item_id": "A"},
            "cpi_yoy": {"enabled": True, "org_id": "101", "tbl_id": "C", "item_id": "Y"},
            "cpi_mom": {"enabled": True, "org_id": "101", "tbl_id": "C", "item_id": "M"},
            "cpi_index_legacy": {"enabled": True, "org_id": "101", "tbl_id": "CL", "item_id": "I"},
        },
        "ecos": {},
    }


def _settings() -> dict:
    return {
        "epsilon": 0.0,
        "use_adaptive_epsilon": False,
        "epsilon_factor": 0.5,
        "confirmation_periods": 2,
        "carry_single_flat_regime": True,
        "yield_curve_spread_threshold": 0.0,
    }


def _macro_frame() -> pd.DataFrame:
    idx = pd.period_range("2024-01", periods=8, freq="M")
    rows: list[dict[str, object]] = []
    for period, leading, yoy, legacy in zip(
        idx,
        [100, 101, 102, 103, 104, 103, 102, 101],
        [3.0, 2.5, 2.0, 1.5, 1.0, 1.2, 1.4, 1.6],
        [100, 101, 102, 103, 104, 105, 106, 107],
        strict=True,
    ):
        rows.append({"period_month": period, "series_alias": "leading_index", "series_id": "101/L/A", "value": leading, "is_provisional": False})
        rows.append({"period_month": period, "series_alias": "cpi_yoy", "series_id": "101/C/Y", "value": yoy, "is_provisional": False})
        rows.append({"period_month": period, "series_alias": "cpi_index_legacy", "series_id": "101/CL/I", "value": legacy, "is_provisional": False})
    rows.append({"period_month": pd.Period("2024-09", freq="M"), "series_alias": "leading_index", "series_id": "101/L/A", "value": 100.0, "is_provisional": True})
    rows.append({"period_month": pd.Period("2024-09", freq="M"), "series_alias": "cpi_yoy", "series_id": "101/C/Y", "value": 1.8, "is_provisional": True})
    frame = pd.DataFrame(rows).set_index("period_month").sort_index()
    return frame


def test_extract_regime_history_dashboard_parity_matches_shared_builder():
    macro_df = _macro_frame()
    macro_cfg = _macro_cfg()
    settings = _settings()

    expected = build_regime_history_from_macro(
        macro_df=macro_df,
        macro_series_cfg=macro_cfg,
        settings=settings,
        market_id="KR",
        include_provisional=False,
        window_months=None,
    )
    result = validate_sector_mapping.extract_regime_history(
        macro_df,
        macro_cfg,
        settings=settings,
        path="dashboard-parity",
        include_provisional=False,
    )

    pd.testing.assert_frame_equal(result, expected)


def test_extract_regime_history_dashboard_parity_excludes_provisional_rows():
    macro_df = _macro_frame()
    macro_cfg = _macro_cfg()
    settings = _settings()

    without_provisional = validate_sector_mapping.extract_regime_history(
        macro_df,
        macro_cfg,
        settings=settings,
        path="dashboard-parity",
        include_provisional=False,
    )
    with_provisional = validate_sector_mapping.extract_regime_history(
        macro_df,
        macro_cfg,
        settings=settings,
        path="dashboard-parity",
        include_provisional=True,
    )

    assert len(without_provisional) < len(with_provisional)
    assert with_provisional.index.max() == pd.Period("2024-09", freq="M")
    assert without_provisional.index.max() == pd.Period("2024-08", freq="M")


def test_stage1_gate_is_deterministic_for_known_inputs():
    lag_scenarios = [
        {"lag_months": 0, "fit_rate_pct": 62.5},
        {"lag_months": 1, "fit_rate_pct": 25.0},
        {"lag_months": 2, "fit_rate_pct": 55.0},
    ]
    divergence = {"count": 42, "points": 118, "pct": 35.6}

    gate = evaluate_regime_validity._stage1_gate(
        lag_scenarios=lag_scenarios,
        divergence=divergence,
    )

    assert gate["open"] is True
    assert gate["lag0_fit_pct"] == 62.5
    assert gate["lag1_fit_pct"] == 25.0
    assert gate["triggers"] == [
        "confirmed-vs-raw divergence 35.6% > 15%",
        "PIT fit 25.0% < 35% with nowcast gap 37.5%p",
    ]


def test_preregistered_bridge_metrics_reports_required_fields():
    macro_df = _macro_frame()
    macro_cfg = _macro_cfg()
    settings = _settings()
    regime_hist = build_regime_history_from_macro(
        macro_df=macro_df,
        macro_series_cfg=macro_cfg,
        settings=settings,
        market_id="KR",
        include_provisional=False,
        window_months=None,
    )
    excess = pd.DataFrame(
        {
            "5044": [0.01, 0.02, -0.01, 0.03],
            "1155": [0.02, 0.01, -0.02, 0.04],
        },
        index=pd.period_range("2024-05", periods=4, freq="M").to_timestamp("M"),
    )
    assignment = {"5044": "Recovery", "1155": "Expansion"}

    result = evaluate_regime_validity._pre_registered_bridge_metrics(
        macro_df=macro_df,
        macro_series_cfg=macro_cfg,
        settings=settings,
        include_provisional=False,
        current_hist=regime_hist,
        excess=excess,
        assignment=assignment,
    )

    assert result["name"] == "flat_to_prior_nonflat_bridge"
    assert result["lookback_months"] == 3
    assert "unexplained_churn_formula" in result
    assert "affected_months" in result
    assert "changed_confirmed_months_vs_current" in result
    assert "candidate_lag1_fit_pct" in result
    assert "latest_same_as_current" in result


def test_sector_fit_risk_summary_exposes_hits_and_overlap_metric():
    rankings = [
        {"regime": "Recovery", "rank": 1, "code": "5044", "is_top_half": True, "assigned_regime": "Recovery"},
        {"regime": "Expansion", "rank": 1, "code": "5044", "is_top_half": True, "assigned_regime": "Recovery"},
        {"regime": "Expansion", "rank": 2, "code": "1155", "is_top_half": True, "assigned_regime": "Expansion"},
        {"regime": "Slowdown", "rank": 8, "code": "5049", "is_top_half": False, "assigned_regime": "Slowdown"},
    ]
    sector_map = {
        "benchmark": {"code": "1001", "name": "KOSPI"},
        "regimes": {
            "Recovery": {"sectors": [{"code": "5044", "name": "KRX 반도체"}]},
            "Expansion": {"sectors": [{"code": "1155", "name": "KOSPI200 정보기술"}]},
            "Slowdown": {"sectors": [{"code": "5049", "name": "KRX 철강"}]},
            "Contraction": {"sectors": []},
        },
    }
    regime_summary = {"counts": {"Recovery": 10, "Expansion": 11, "Slowdown": 12, "Contraction": 13}}

    result = validate_sector_mapping.build_sector_fit_risk_summary(
        rankings=rankings,
        sector_map=sector_map,
        regime_summary=regime_summary,
    )

    assert result["top_half_hits"] == 2
    assert result["top_half_total"] == 3
    assert result["sample_counts"]["Recovery"] == 10
    assert result["shared_top_half_memberships"] == 2
    assert result["overlap_rate"] == 2 / 3
    assert result["candidate_proposed"] is False
    assert result["map_gate_open"] is False
    assert result["map_gate_decision"] == "map remains unchanged"


def test_sector_fit_risk_summary_can_open_gate_for_future_followup():
    rankings = [
        {"regime": "Recovery", "rank": 1, "code": "5044", "is_top_half": True, "assigned_regime": "Recovery"},
        {"regime": "Expansion", "rank": 1, "code": "5044", "is_top_half": True, "assigned_regime": "Recovery"},
        {"regime": "Expansion", "rank": 2, "code": "1155", "is_top_half": True, "assigned_regime": "Expansion"},
        {"regime": "Slowdown", "rank": 8, "code": "5049", "is_top_half": False, "assigned_regime": "Slowdown"},
    ]
    sector_map = {
        "benchmark": {"code": "1001", "name": "KOSPI"},
        "regimes": {
            "Recovery": {"sectors": [{"code": "5044", "name": "KRX 반도체"}]},
            "Expansion": {"sectors": [{"code": "1155", "name": "KOSPI200 정보기술"}]},
            "Slowdown": {"sectors": [{"code": "5049", "name": "KRX 철강"}]},
            "Contraction": {"sectors": []},
        },
    }
    regime_summary = {"counts": {"Recovery": 10, "Expansion": 11, "Slowdown": 12, "Contraction": 13}}

    result = validate_sector_mapping.build_sector_fit_risk_summary(
        rankings=rankings,
        sector_map=sector_map,
        regime_summary=regime_summary,
        candidate_top_half_hits=5,
        candidate_overlap_rate=0.2,
        candidate_generic_overlap=False,
    )

    assert result["candidate_proposed"] is True
    assert result["map_gate_open"] is True
    assert result["map_gate_decision"] == "map follow-up justified later"


def test_sector_fit_risk_summary_accepts_current_mapping_when_fit_is_high():
    rankings = []
    for regime, code in [
        ("Recovery", "1155"),
        ("Recovery", "1168"),
        ("Expansion", "5042"),
        ("Expansion", "5044"),
        ("Expansion", "5045"),
        ("Slowdown", "1157"),
        ("Slowdown", "1165"),
        ("Slowdown", "1170"),
        ("Contraction", "5046"),
        ("Contraction", "5048"),
        ("Contraction", "5049"),
    ]:
        rankings.append(
            {
                "regime": regime,
                "rank": 1,
                "code": code,
                "is_top_half": True,
                "assigned_regime": regime,
            }
        )
    sector_map = {
        "benchmark": {"code": "1001", "name": "KOSPI"},
        "regimes": {
            "Recovery": {"sectors": [{"code": "1155", "name": "KOSPI200 정보기술"}, {"code": "1168", "name": "KOSPI200 금융"}]},
            "Expansion": {"sectors": [{"code": "5042", "name": "KRX 산업재"}, {"code": "5044", "name": "KRX 반도체"}, {"code": "5045", "name": "KRX 헬스케어"}]},
            "Slowdown": {"sectors": [{"code": "1157", "name": "KOSPI200 생활소비재"}, {"code": "1165", "name": "KOSPI200 경기소비재"}, {"code": "1170", "name": "KOSPI200 유틸리티"}]},
            "Contraction": {"sectors": [{"code": "5046", "name": "KRX 미디어통신"}, {"code": "5048", "name": "KRX 에너지화학"}, {"code": "5049", "name": "KRX 철강"}]},
        },
    }
    regime_summary = {"counts": {"Recovery": 10, "Expansion": 11, "Slowdown": 12, "Contraction": 13}}

    result = validate_sector_mapping.build_sector_fit_risk_summary(
        rankings=rankings,
        sector_map=sector_map,
        regime_summary=regime_summary,
        current_matches_candidate=True,
        candidate_top_half_hits=11,
        candidate_overlap_rate=0.1,
        candidate_generic_overlap=False,
    )

    assert result["top_half_hits"] == 11
    assert result["map_gate_open"] is False
    assert result["map_gate_decision"] == "current mapping accepted"


def test_build_lag1_candidate_map_is_deterministic():
    ranking_rows = [
        {"regime": "Recovery", "code": "1155", "avg_excess_pct": 2.0, "rank": 1, "is_top_half": True},
        {"regime": "Recovery", "code": "1168", "avg_excess_pct": 1.5, "rank": 2, "is_top_half": True},
        {"regime": "Recovery", "code": "5044", "avg_excess_pct": 1.4, "rank": 3, "is_top_half": True},
        {"regime": "Recovery", "code": "5042", "avg_excess_pct": 0.5, "rank": 4, "is_top_half": True},
        {"regime": "Recovery", "code": "5045", "avg_excess_pct": 0.1, "rank": 5, "is_top_half": True},
        {"regime": "Expansion", "code": "5042", "avg_excess_pct": 2.4, "rank": 1, "is_top_half": True},
        {"regime": "Expansion", "code": "5044", "avg_excess_pct": 2.3, "rank": 2, "is_top_half": True},
        {"regime": "Expansion", "code": "5045", "avg_excess_pct": 2.2, "rank": 3, "is_top_half": True},
        {"regime": "Expansion", "code": "1155", "avg_excess_pct": 1.0, "rank": 4, "is_top_half": True},
        {"regime": "Expansion", "code": "5046", "avg_excess_pct": 0.8, "rank": 5, "is_top_half": True},
        {"regime": "Slowdown", "code": "1157", "avg_excess_pct": 2.5, "rank": 1, "is_top_half": True},
        {"regime": "Slowdown", "code": "1165", "avg_excess_pct": 2.0, "rank": 2, "is_top_half": True},
        {"regime": "Slowdown", "code": "1170", "avg_excess_pct": 1.8, "rank": 3, "is_top_half": True},
        {"regime": "Slowdown", "code": "5042", "avg_excess_pct": 1.1, "rank": 4, "is_top_half": True},
        {"regime": "Slowdown", "code": "5044", "avg_excess_pct": 1.0, "rank": 5, "is_top_half": True},
        {"regime": "Contraction", "code": "5046", "avg_excess_pct": 2.3, "rank": 1, "is_top_half": True},
        {"regime": "Contraction", "code": "5048", "avg_excess_pct": 2.1, "rank": 2, "is_top_half": True},
        {"regime": "Contraction", "code": "5049", "avg_excess_pct": 1.9, "rank": 3, "is_top_half": True},
        {"regime": "Contraction", "code": "5045", "avg_excess_pct": 1.0, "rank": 4, "is_top_half": True},
        {"regime": "Contraction", "code": "5042", "avg_excess_pct": 0.9, "rank": 5, "is_top_half": True},
    ]
    capacity_by_regime = {"Recovery": 2, "Expansion": 3, "Slowdown": 3, "Contraction": 3}

    result = validate_sector_mapping.build_lag1_candidate_map(
        ranking_rows=ranking_rows,
        capacity_by_regime=capacity_by_regime,
    )

    assert result == {
        "Recovery": ["1155", "1168"],
        "Expansion": ["1170", "5042", "5044"],
        "Slowdown": ["1157", "1165", "5045"],
        "Contraction": ["5046", "5048", "5049"],
    }


def test_print_regime_rankings_uses_ascii_markers():
    combined = pd.DataFrame(
        {
            "5044": [0.02],
            "1155": [0.01],
            "regime": ["Recovery"],
        },
        index=pd.period_range("2024-05", periods=1, freq="M"),
    )
    sector_names = {"5044": "KRX 반도체", "1155": "KOSPI200 정보기술"}
    current_assignment = {"5044": "Recovery", "1155": "Expansion"}

    capture = io.StringIO()
    with redirect_stdout(capture):
        validate_sector_mapping.print_regime_rankings(
            combined,
            sector_names,
            current_assignment,
        )

    output = capture.getvalue()
    assert "OK Recovery" in output
    assert "-- Expansion" in output
    assert "✓" not in output
    assert "✗" not in output
