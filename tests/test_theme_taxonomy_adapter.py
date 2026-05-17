from __future__ import annotations

from config.markets import load_market_configs
from src.dashboard.theme_taxonomy_adapter import build_taxonomy_dashboard_model


def test_taxonomy_adapter_covers_current_kr_sector_map():
    _, sector_map, _, _ = load_market_configs("KR")

    model = build_taxonomy_dashboard_model(sector_map=sector_map, market="KR")

    assert model is not None
    assert model.coverage_complete
    assert model.diagnostics == ()
    assert model.sector_count == 11
    lookup = model.by_sector_code()
    assert lookup["5044"].taxonomy_label == "KRX 반도체"
    assert lookup["5044"].base_labels == ("반도체",)
    assert lookup["5049"].base_labels == ("철강·비철소재",)
    assert "증권·자본시장" in lookup["1168"].base_labels


def test_taxonomy_adapter_surfaces_unmapped_sector_map_rows():
    sector_map = {
        "regimes": {
            "Expansion": {
                "sectors": [
                    {"code": "9999", "name": "임시 섹터"},
                ]
            }
        }
    }

    model = build_taxonomy_dashboard_model(sector_map=sector_map, market="KR")

    assert model is not None
    assert not model.coverage_complete
    assert model.sector_contexts == ()
    assert model.diagnostics == ("unmapped sector_map.yml sector: 9999 임시 섹터",)


def test_taxonomy_adapter_is_disabled_for_us_market():
    assert build_taxonomy_dashboard_model(sector_map={"regimes": {}}, market="US") is None
