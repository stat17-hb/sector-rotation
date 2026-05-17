from __future__ import annotations

import importlib
from pathlib import Path

from src.data_sources.theme_taxonomy_sync import (
    get_theme_taxonomy_index_codes,
    sync_theme_taxonomy_warehouse,
)


def test_theme_taxonomy_index_codes_preserve_runtime_mapping_order():
    codes = get_theme_taxonomy_index_codes(market="KR")

    assert codes[:3] == ["1155", "1168", "5042"]
    assert "5044" in codes
    assert len(codes) == len(set(codes))


def test_sync_theme_taxonomy_records_first_class_warehouse_status():
    warehouse = importlib.import_module("src.data_sources.warehouse")
    status, frame, summary = sync_theme_taxonomy_warehouse(
        reason="manual_refresh",
        market="KR",
    )

    assert status == "LIVE"
    assert not frame.empty
    assert summary["runtime_index_count"] == len(frame)
    assert summary["taxonomy_version"] == 2
    assert "5044" in frame["index_code"].astype(str).tolist()

    dataset_status = warehouse.read_dataset_status("theme_taxonomy", market="KR")
    assert dataset_status["status"] == "LIVE"
    assert dataset_status["provider"] == "THEME_TAXONOMY"
    assert dataset_status["coverage_complete"] is True
    assert dataset_status["row_count"] == len(frame)
    assert dataset_status["details"]["last_verified_at"] == "2026-05-17"

    bounds = warehouse.read_dataset_data_bounds("theme_taxonomy", market="KR")
    assert bounds == {
        "min_trade_date": "20260517",
        "max_trade_date": "20260517",
        "row_count": len(frame),
        "taxonomy_version": 2,
        "last_verified_at": "2026-05-17",
        "verification_status": "verified",
    }
    assert warehouse.probe_dataset_mode("theme_taxonomy", market="KR") == "CACHED"

    history = warehouse.read_collection_run_history(
        market="KR",
        reasons=("manual_refresh",),
        sample_per_dataset=True,
        sample_size=10,
    )
    assert history["dataset"].tolist() == ["theme_taxonomy"]
    assert history["row_count"].tolist() == [len(frame)]


def test_sync_theme_taxonomy_marks_missing_runtime_mapping_as_incomplete(tmp_path):
    warehouse = importlib.import_module("src.data_sources.warehouse")
    config_path = tmp_path / "theme_taxonomy.yml"
    config_text = Path("config/theme_taxonomy.yml").read_text(encoding="utf-8")
    config_path.write_text(
        config_text.replace(
            '      - market: "KR"\n        index_code: "1155"',
            '      - market: "US"\n        index_code: "1155"',
            1,
        ),
        encoding="utf-8",
    )

    status, frame, summary = sync_theme_taxonomy_warehouse(
        reason="manual_refresh",
        market="KR",
        config_path=config_path,
    )

    assert status == "LIVE"
    assert "1155" not in frame["index_code"].astype(str).tolist()
    assert summary["coverage_complete"] is False
    assert summary["expected_mapping_count"] == 11
    assert summary["missing_mapping_ids"] == ["kr_sector_1155_information_technology"]

    dataset_status = warehouse.read_dataset_status("theme_taxonomy", market="KR")
    assert dataset_status["coverage_complete"] is False
    assert dataset_status["predicted_requests"] == 11
    assert dataset_status["processed_requests"] == 10
    assert dataset_status["failed_codes"] == {
        "mapping:kr_sector_1155_information_technology": "no runtime index reference for KR"
    }
