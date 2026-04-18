from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import src.data_sources.macro_sync as macro_sync
import src.data_sources.warehouse as warehouse


def _macro_frame(provider_series_id: str, period: str, value: float) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    return pd.DataFrame(
        {
            "series_id": [provider_series_id],
            "value": [value],
            "source": ["KOSIS"],
            "fetched_at": [now],
            "is_provisional": [False],
        },
        index=pd.PeriodIndex([period], freq="M"),
    )


def test_read_macro_data_and_export_follow_current_dimension_mapping(tmp_path):
    warehouse.upsert_macro_dimension(
        [
            {
                "series_alias": "cpi_yoy",
                "provider": "KOSIS",
                "provider_series_id": "OLD_SERIES",
                "enabled": True,
                "label": "",
                "unit": "%",
            }
        ],
        market="KR",
    )
    warehouse.upsert_macro_series_frame(
        series_alias="cpi_yoy",
        provider="KOSIS",
        provider_series_id="OLD_SERIES",
        frame=_macro_frame("OLD_SERIES", "2024-01", 1.0),
        market="KR",
    )

    warehouse.upsert_macro_dimension(
        [
            {
                "series_alias": "cpi_yoy",
                "provider": "KOSIS",
                "provider_series_id": "NEW_SERIES",
                "enabled": True,
                "label": "",
                "unit": "%",
            }
        ],
        market="KR",
    )
    warehouse.upsert_macro_series_frame(
        series_alias="cpi_yoy",
        provider="KOSIS",
        provider_series_id="NEW_SERIES",
        frame=_macro_frame("NEW_SERIES", "2024-02", 2.0),
        market="KR",
    )

    loaded = warehouse.read_macro_data(
        series_aliases=["cpi_yoy"],
        start_ym="202401",
        end_ym="202402",
        market="KR",
    )

    assert list(loaded["series_id"].astype(str)) == ["NEW_SERIES"]
    export_path = warehouse.export_macro_parquet(tmp_path / "macro.parquet", market="KR")
    exported = pd.read_parquet(export_path)
    assert sorted(exported["series_id"].astype(str).unique()) == ["NEW_SERIES"]


def test_sync_provider_macro_refetches_full_window_when_provider_series_changes():
    expected_provider_series_id = "101/DT_1J22042/T03/0"
    warehouse.upsert_macro_dimension(
        [
            {
                "series_alias": "cpi_yoy",
                "provider": "KOSIS",
                "provider_series_id": "OLD_SERIES",
                "enabled": True,
                "label": "",
                "unit": "%",
            }
        ],
        market="KR",
    )
    warehouse.upsert_macro_series_frame(
        series_alias="cpi_yoy",
        provider="KOSIS",
        provider_series_id="OLD_SERIES",
        frame=_macro_frame("OLD_SERIES", "2024-03", 1.0),
        market="KR",
    )

    captured: dict[str, str] = {}

    def _fake_fetch(alias: str, cfg: dict[str, str], provider_start: str, provider_end: str) -> pd.DataFrame:
        captured["alias"] = alias
        captured["start"] = provider_start
        captured["end"] = provider_end
        return _macro_frame("NEW_SERIES", "2024-01", 2.0)

    status, frame, summary = macro_sync.sync_provider_macro(
        provider="KOSIS",
        start_ym="202401",
        end_ym="202403",
        series_config={
            "cpi_yoy": {
                "org_id": "101",
                "tbl_id": "DT_1J22042",
                "item_id": "T03",
                "obj_params": {"objL1": "0"},
                "enabled": True,
                "label": "CPI YoY",
                "unit": "%",
            }
        },
        fetch_fn=_fake_fetch,
        reason="test_provider_series_change",
        force=False,
        market="KR",
    )

    assert captured == {"alias": "cpi_yoy", "start": "202401", "end": "202403"}
    assert status == "LIVE"
    assert summary["delta_aliases"] == ["cpi_yoy"]
    assert list(frame["series_id"].astype(str)) == [expected_provider_series_id]


def test_sync_provider_macro_bypasses_cache_fast_path_when_provider_series_changes_with_full_stale_coverage():
    expected_provider_series_id = "101/DT_1J22042/T03/0"
    warehouse.upsert_macro_dimension(
        [
            {
                "series_alias": "cpi_yoy",
                "provider": "KOSIS",
                "provider_series_id": "OLD_SERIES",
                "enabled": True,
                "label": "",
                "unit": "%",
            }
        ],
        market="KR",
    )
    warehouse.upsert_macro_series_frame(
        series_alias="cpi_yoy",
        provider="KOSIS",
        provider_series_id="OLD_SERIES",
        frame=pd.concat(
            [
                _macro_frame("OLD_SERIES", "2024-01", 1.0),
                _macro_frame("OLD_SERIES", "2024-02", 1.1),
                _macro_frame("OLD_SERIES", "2024-03", 1.2),
            ]
        ),
        market="KR",
    )

    captured: dict[str, str] = {}

    def _fake_fetch(alias: str, cfg: dict[str, str], provider_start: str, provider_end: str) -> pd.DataFrame:
        captured["alias"] = alias
        captured["start"] = provider_start
        captured["end"] = provider_end
        return _macro_frame("NEW_SERIES", "2024-01", 2.0)

    status, frame, summary = macro_sync.sync_provider_macro(
        provider="KOSIS",
        start_ym="202401",
        end_ym="202403",
        series_config={
            "cpi_yoy": {
                "org_id": "101",
                "tbl_id": "DT_1J22042",
                "item_id": "T03",
                "obj_params": {"objL1": "0"},
                "enabled": True,
                "label": "CPI YoY",
                "unit": "%",
            }
        },
        fetch_fn=_fake_fetch,
        reason="test_provider_series_change_complete_cache",
        force=False,
        market="KR",
    )

    assert captured == {"alias": "cpi_yoy", "start": "202401", "end": "202403"}
    assert status == "LIVE"
    assert summary["delta_aliases"] == ["cpi_yoy"]
    assert list(frame["series_id"].astype(str)) == [expected_provider_series_id]


def test_sync_provider_macro_refetches_full_window_when_current_alias_has_gaps():
    expected_provider_series_id = "101/DT_1J22042/T03/0"
    warehouse.upsert_macro_dimension(
        [
            {
                "series_alias": "cpi_yoy",
                "provider": "KOSIS",
                "provider_series_id": "CURRENT_SERIES",
                "enabled": True,
                "label": "",
                "unit": "%",
            }
        ],
        market="KR",
    )
    warehouse.upsert_macro_series_frame(
        series_alias="cpi_yoy",
        provider="KOSIS",
        provider_series_id="CURRENT_SERIES",
        frame=_macro_frame("CURRENT_SERIES", "2024-12", 1.0),
        market="KR",
    )

    captured: dict[str, str] = {}

    def _fake_fetch(alias: str, cfg: dict[str, str], provider_start: str, provider_end: str) -> pd.DataFrame:
        captured["alias"] = alias
        captured["start"] = provider_start
        captured["end"] = provider_end
        return _macro_frame("CURRENT_SERIES", "2024-01", 2.0)

    status, frame, summary = macro_sync.sync_provider_macro(
        provider="KOSIS",
        start_ym="202401",
        end_ym="202412",
        series_config={
            "cpi_yoy": {
                "org_id": "101",
                "tbl_id": "DT_1J22042",
                "item_id": "T03",
                "obj_params": {"objL1": "0"},
                "enabled": True,
                "label": "CPI YoY",
                "unit": "%",
            }
        },
        fetch_fn=_fake_fetch,
        reason="test_alias_gap_backfill",
        force=False,
        market="KR",
    )

    assert captured == {"alias": "cpi_yoy", "start": "202401", "end": "202412"}
    assert status == "LIVE"
    assert summary["delta_aliases"] == ["cpi_yoy"]
    assert expected_provider_series_id in set(frame["series_id"].astype(str))
