from pathlib import Path

from src.data_sources.sector_etf_mapping import (
    build_config_etf_map,
    build_effective_etf_map,
    load_generated_etf_map,
    merge_etf_maps,
)


def test_build_config_etf_map_reads_sector_map_entries():
    sector_map = {
        "regimes": {
            "Expansion": {
                "sectors": [
                    {
                        "code": "5044",
                        "name": "KRX 반도체",
                        "etfs": [{"code": "396500", "name": "TIGER 반도체TOP10"}],
                    },
                    {"code": "5052", "name": "KRX 건설", "etfs": []},
                ]
            }
        }
    }

    assert build_config_etf_map(sector_map) == {
        "5044": [{"code": "396500", "name": "TIGER 반도체TOP10"}]
    }


def test_load_generated_etf_map_reads_overlay_schema(tmp_path: Path):
    overlay_path = tmp_path / "sector_etf_map.generated.yml"
    overlay_path.write_text(
        """
mappings:
  "5052":
    sector_name: "KRX 건설"
    etfs:
      - code: "117700"
        name: "KODEX 건설"
""".strip(),
        encoding="utf-8",
    )

    assert load_generated_etf_map(overlay_path) == {
        "5052": [{"code": "117700", "name": "KODEX 건설"}]
    }


def test_merge_etf_maps_keeps_config_as_authority():
    generated_map = {
        "5044": [{"code": "091160", "name": "KODEX 반도체"}],
        "5052": [{"code": "117700", "name": "KODEX 건설"}],
    }
    config_map = {
        "5044": [{"code": "396500", "name": "TIGER 반도체TOP10"}],
    }

    assert merge_etf_maps(generated_map, config_map) == {
        "5044": [{"code": "396500", "name": "TIGER 반도체TOP10"}],
        "5052": [{"code": "117700", "name": "KODEX 건설"}],
    }


def test_build_effective_etf_map_fills_sector_map_gaps(tmp_path: Path):
    overlay_path = tmp_path / "sector_etf_map.generated.yml"
    overlay_path.write_text(
        """
mappings:
  "5054":
    etfs:
      - code: "102970"
        name: "KODEX 증권"
""".strip(),
        encoding="utf-8",
    )
    sector_map = {
        "regimes": {
            "Expansion": {
                "sectors": [
                    {"code": "5044", "etfs": [{"code": "396500", "name": "TIGER 반도체TOP10"}]},
                    {"code": "5054", "etfs": []},
                ]
            }
        }
    }

    assert build_effective_etf_map(sector_map, generated_path=overlay_path) == {
        "5044": [{"code": "396500", "name": "TIGER 반도체TOP10"}],
        "5054": [{"code": "102970", "name": "KODEX 증권"}],
    }
