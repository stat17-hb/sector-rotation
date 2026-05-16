from __future__ import annotations

import pandas as pd
import pytest

from src.macro.series_utils import (
    build_enabled_sector_export_aliases,
    build_sector_trade_proxy_lens,
    extract_kr_export_growth_yoy,
    extract_trade_indicators,
    is_macro_alias_enabled,
)


def test_sector_export_capability_is_true_when_export_series_alias_exists_and_enabled():
    sector_map = {
        "regimes": {
            "Expansion": {
                "sectors": [
                    {"name": "KRX 반도체", "export_series_alias": "export_semiconductor"},
                ]
            }
        }
    }
    macro_series_cfg = {"ecos": {"export_semiconductor": {"enabled": True, "stat_code": "901Y118"}}}

    assert is_macro_alias_enabled(macro_series_cfg, "export_semiconductor") is True
    assert build_enabled_sector_export_aliases(sector_map, macro_series_cfg) == {
        "KRX 반도체": "export_semiconductor"
    }


def test_sector_export_capability_ignores_export_sector_flag_without_alias():
    sector_map = {
        "regimes": {
            "Expansion": {
                "sectors": [
                    {"name": "Industrials", "export_sector": True},
                ]
            }
        }
    }
    macro_series_cfg = {"fred": {"trade_exports_yoy": {"enabled": True, "series_id": "BOPTEXP"}}}

    assert build_enabled_sector_export_aliases(sector_map, macro_series_cfg) == {}


def test_sector_export_capability_is_false_when_export_series_alias_is_disabled():
    sector_map = {
        "regimes": {
            "Expansion": {
                "sectors": [
                    {"name": "KRX 반도체", "export_series_alias": "export_semiconductor"},
                ]
            }
        }
    }
    macro_series_cfg = {"ecos": {"export_semiconductor": {"enabled": False, "stat_code": "901Y118"}}}

    assert is_macro_alias_enabled(macro_series_cfg, "export_semiconductor") is False
    assert build_enabled_sector_export_aliases(sector_map, macro_series_cfg) == {}


def test_us_trade_indicators_use_pretransformed_alias_values_without_level_yoy():
    macro_df = pd.DataFrame(
        {
            "series_alias": ["trade_exports_yoy", "trade_imports_yoy", "trade_balance"],
            "series_id": ["BOPTEXP", "BOPTIMP", "BOPGSTB"],
            "value": [4.2, -1.5, -67_100.0],
        },
        index=pd.PeriodIndex(["2026-03", "2026-03", "2026-03"], freq="M"),
    )
    macro_series_cfg = {
        "fred": {
            "trade_exports_yoy": {"enabled": True, "series_id": "BOPTEXP"},
            "trade_imports_yoy": {"enabled": True, "series_id": "BOPTIMP"},
            "trade_balance": {"enabled": True, "series_id": "BOPGSTB"},
        }
    }

    assert extract_trade_indicators(macro_df, macro_series_cfg) == {
        "exports_yoy": 4.2,
        "imports_yoy": -1.5,
        "balance": -67_100.0,
    }
    assert extract_kr_export_growth_yoy(macro_df, macro_series_cfg) is None


def test_kr_export_growth_still_uses_export_amount_level_yoy():
    periods = pd.period_range("2025-03", periods=13, freq="M")
    macro_df = pd.DataFrame(
        {
            "series_alias": ["export_amount"] * 13,
            "series_id": ["901Y118/T002"] * 13,
            "value": [100.0] * 12 + [112.0],
        },
        index=periods,
    )
    macro_series_cfg = {"ecos": {"export_amount": {"enabled": True, "stat_code": "901Y118", "item_code": "T002"}}}

    assert extract_kr_export_growth_yoy(macro_df, macro_series_cfg) == pytest.approx(12.0)
    assert extract_trade_indicators(macro_df, macro_series_cfg) == {}


def test_sector_trade_proxy_lens_maps_aggregate_trade_by_exposure():
    sector_map = {
        "regimes": {
            "Recovery": {
                "sectors": [
                    {
                        "name": "Technology",
                        "trade_exposure": "export_sensitive",
                        "trade_proxy_label": "글로벌 IT/장비 수요",
                    },
                    {
                        "name": "Consumer Discretionary",
                        "trade_exposure": "import_sensitive",
                        "trade_proxy_label": "수입 소비재/내수 수요",
                    },
                    {
                        "name": "Utilities",
                        "trade_exposure": "low_linkage",
                    },
                ]
            }
        }
    }

    rows = build_sector_trade_proxy_lens(
        sector_map,
        {"exports_yoy": 4.2, "imports_yoy": -5.1},
    )

    by_sector = {str(row["sector"]): row for row in rows}
    assert by_sector["Technology"]["status"] == "교역 순풍"
    assert by_sector["Technology"]["driver"] == "수출 YoY"
    assert by_sector["Consumer Discretionary"]["status"] == "교역 역풍"
    assert by_sector["Consumer Discretionary"]["driver"] == "수입 YoY"
    assert by_sector["Utilities"]["status"] == "직접 해석 제한"
    assert by_sector["Utilities"]["value"] is None


def test_sector_trade_proxy_lens_handles_missing_trade_data():
    sector_map = {
        "regimes": {
            "Expansion": {
                "sectors": [
                    {"name": "Industrials", "trade_exposure": "export_sensitive"},
                ]
            }
        }
    }

    rows = build_sector_trade_proxy_lens(sector_map, {})

    assert rows[0]["sector"] == "Industrials"
    assert rows[0]["status"] == "데이터 없음"
    assert rows[0]["value"] is None
