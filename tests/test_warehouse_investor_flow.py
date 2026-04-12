from __future__ import annotations

import pandas as pd

import src.data_sources.warehouse as warehouse


def test_investor_flow_rows_are_scoped_and_readable():
    raw_frame = pd.DataFrame(
        {
            "ticker": ["005930", "005930"],
            "ticker_name": ["삼성전자", "삼성전자"],
            "investor_type": ["외국인", "개인"],
            "buy_amount": [1200, 900],
            "sell_amount": [800, 1100],
            "net_buy_amount": [400, -200],
        },
        index=pd.DatetimeIndex(["2026-04-08", "2026-04-08"]),
    )
    sector_frame = pd.DataFrame(
        {
            "sector_code": ["5044", "5044"],
            "sector_name": ["KRX 반도체", "KRX 반도체"],
            "investor_type": ["외국인", "개인"],
            "buy_amount": [1200, 900],
            "sell_amount": [800, 1100],
            "net_buy_amount": [400, -200],
            "net_flow_ratio": [0.2, -0.1],
        },
        index=pd.DatetimeIndex(["2026-04-08", "2026-04-08"]),
    )

    warehouse.upsert_investor_flow_raw(raw_frame, provider="PYKRX_UNOFFICIAL", market="KR")
    warehouse.upsert_investor_flow_sector(sector_frame, provider="PYKRX_UNOFFICIAL", market="KR")
    warehouse.record_ingest_run(
        dataset="investor_flow",
        reason="test_seed",
        provider="PYKRX_UNOFFICIAL",
        requested_start="20260401",
        requested_end="20260408",
        status="LIVE",
        coverage_complete=True,
        failed_days=[],
        failed_codes={},
        delta_keys=["5044"],
        row_count=2,
        summary={"status": "LIVE"},
        market="KR",
    )
    warehouse.update_ingest_watermark(
        dataset="investor_flow",
        watermark_key="20260408",
        status="LIVE",
        coverage_complete=True,
        provider="PYKRX_UNOFFICIAL",
        details={"rows": 2},
        market="KR",
    )

    read_back = warehouse.read_sector_investor_flow(["5044"], "20260401", "20260408", market="KR")
    assert len(read_back) == 2
    assert set(read_back["investor_type"]) == {"외국인", "개인"}
    assert warehouse.read_dataset_status("investor_flow", market="KR")["provider"] == "PYKRX_UNOFFICIAL"
    assert warehouse.probe_dataset_mode("investor_flow", market="KR") == "CACHED"


def test_read_latest_sector_constituents_snapshot_uses_latest_date_per_sector():
    warehouse.upsert_sector_constituents_snapshot(
        [
            {
                "sector_code": "5044",
                "ticker": "005930",
                "reference_date": "20260408",
                "resolved_from": "20260408",
            }
        ],
        snapshot_date="20260408",
        provider="PYKRX_UNOFFICIAL",
        market="KR",
    )
    warehouse.upsert_sector_constituents_snapshot(
        [
            {
                "sector_code": "5045",
                "ticker": "000660",
                "reference_date": "20260409",
                "resolved_from": "20260409",
            }
        ],
        snapshot_date="20260409",
        provider="PYKRX_UNOFFICIAL",
        market="KR",
    )

    rows = warehouse.read_latest_sector_constituents_snapshot(["5044", "5045"], market="KR")

    assert set(rows["sector_code"]) == {"5044", "5045"}
    latest_dates = {row["sector_code"]: str(row["snapshot_date"].date()) for _, row in rows.iterrows()}
    assert latest_dates["5044"] == "2026-04-08"
    assert latest_dates["5045"] == "2026-04-09"


def _operational_summary(*, end: str, coverage_complete: bool, failed_days: list[str] | None = None) -> dict:
    return {
        "status": "LIVE",
        "provider": "PYKRX_UNOFFICIAL",
        "requested_start": "20260401",
        "requested_end": end,
        "coverage_complete": coverage_complete,
        "failed_days": list(failed_days or []),
        "failed_codes": {},
        "predicted_requests": 10,
        "processed_requests": 10,
        "rows": 2,
    }


def _raw_frame_for(date_str: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime([date_str, date_str]),
            "ticker": ["005930", "005930"],
            "ticker_name": ["삼성전자", "삼성전자"],
            "investor_type": ["외국인", "개인"],
            "buy_amount": [1200, 900],
            "sell_amount": [800, 1100],
            "net_buy_amount": [400, -200],
        }
    )


def _sector_frame_for(date_str: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime([date_str, date_str]),
            "sector_code": ["5044", "5044"],
            "sector_name": ["KRX 반도체", "KRX 반도체"],
            "investor_type": ["외국인", "개인"],
            "buy_amount": [1200, 900],
            "sell_amount": [800, 1100],
            "net_buy_amount": [400, -200],
            "net_flow_ratio": [0.2, -0.1],
        }
    )


def test_operational_complete_cursor_caps_product_reads():
    warehouse.write_investor_flow_operational_result(
        raw_frame=_raw_frame_for("2026-04-08"),
        sector_frame=_sector_frame_for("2026-04-08"),
        provider="PYKRX_UNOFFICIAL",
        requested_start="20260408",
        requested_end="20260408",
        reason="manual_refresh",
        summary=_operational_summary(end="20260408", coverage_complete=True),
        market="KR",
    )
    warehouse.upsert_investor_flow_sector(
        _sector_frame_for("2026-04-09"),
        provider="PYKRX_UNOFFICIAL",
        market="KR",
    )

    capped = warehouse.read_sector_investor_flow(
        ["5044"],
        "20260408",
        "20260409",
        market="KR",
        cap_to_operational_cursor=True,
    )
    uncapped = warehouse.read_sector_investor_flow(
        ["5044"],
        "20260408",
        "20260409",
        market="KR",
        cap_to_operational_cursor=False,
    )

    assert warehouse.read_investor_flow_operational_complete_cursor(market="KR") == "20260408"
    assert capped.index.max() == pd.Timestamp("2026-04-08")
    assert uncapped.index.max() == pd.Timestamp("2026-04-09")


def test_partial_operational_run_keeps_rows_but_not_complete_cursor():
    warehouse.write_investor_flow_operational_result(
        raw_frame=_raw_frame_for("2026-04-08"),
        sector_frame=_sector_frame_for("2026-04-08"),
        provider="PYKRX_UNOFFICIAL",
        requested_start="20260408",
        requested_end="20260408",
        reason="manual_refresh",
        summary=_operational_summary(end="20260408", coverage_complete=True),
        market="KR",
    )

    partial_summary = _operational_summary(
        end="20260410",
        coverage_complete=False,
        failed_days=["20260410"],
    )
    warehouse.write_investor_flow_operational_result(
        raw_frame=_raw_frame_for("2026-04-09"),
        sector_frame=_sector_frame_for("2026-04-09"),
        provider="PYKRX_UNOFFICIAL",
        requested_start="20260409",
        requested_end="20260410",
        reason="manual_refresh",
        summary=partial_summary,
        market="KR",
    )

    stored = warehouse.read_sector_investor_flow(
        ["5044"],
        "20260409",
        "20260409",
        market="KR",
        cap_to_operational_cursor=False,
    )
    latest_run = warehouse.read_latest_investor_flow_run(
        market="KR",
        reasons=("manual_refresh",),
    )

    assert not stored.empty
    assert warehouse.read_investor_flow_operational_complete_cursor(market="KR") == "20260408"
    assert latest_run["requested_end"] == "20260410"
    assert latest_run["coverage_complete"] is False
    assert latest_run["failed_days"] == ["20260410"]


def test_backfill_progress_cursor_is_independent_from_latest_fact_date():
    warehouse.write_investor_flow_backfill_chunk(
        raw_frame=_raw_frame_for("2025-01-02"),
        chunk_start="20250102",
        chunk_end="20250102",
        provider="PYKRX_UNOFFICIAL",
        summary={"status": "LIVE", "failed_days": [], "failed_codes": {}},
        oldest_collectable_date="20250102",
        target_end_date="20250103",
        market="KR",
    )
    warehouse.write_investor_flow_backfill_chunk(
        raw_frame=_raw_frame_for("2025-01-03"),
        chunk_start="20250103",
        chunk_end="20250103",
        provider="PYKRX_UNOFFICIAL",
        summary={"status": "LIVE", "failed_days": [], "failed_codes": {}},
        oldest_collectable_date="20250102",
        target_end_date="20250103",
        market="KR",
    )
    warehouse.write_investor_flow_operational_result(
        raw_frame=_raw_frame_for("2026-04-08"),
        sector_frame=_sector_frame_for("2026-04-08"),
        provider="PYKRX_UNOFFICIAL",
        requested_start="20260408",
        requested_end="20260408",
        reason="manual_refresh",
        summary=_operational_summary(end="20260408", coverage_complete=True),
        market="KR",
    )

    assert warehouse.read_investor_flow_backfill_progress_cursor(market="KR") == "20250103"
    assert warehouse.read_investor_flow_operational_complete_cursor(market="KR") == "20260408"


def test_latest_investor_flow_run_filters_out_backfill_reasons():
    warehouse.record_ingest_run(
        dataset="investor_flow",
        reason="manual_refresh",
        provider="PYKRX_UNOFFICIAL",
        requested_start="20260401",
        requested_end="20260408",
        status="LIVE",
        coverage_complete=False,
        failed_days=["20260408"],
        failed_codes={"20260408": "partial"},
        delta_keys=[],
        row_count=0,
        summary={"mode": "manual"},
        market="KR",
        created_at=pd.Timestamp("2026-04-10T00:00:00Z").to_pydatetime(),
    )
    warehouse.record_ingest_run(
        dataset="investor_flow",
        reason="historical_backfill",
        provider="PYKRX_UNOFFICIAL",
        requested_start="20250101",
        requested_end="20250102",
        status="LIVE",
        coverage_complete=True,
        failed_days=[],
        failed_codes={},
        delta_keys=[],
        row_count=2,
        summary={"mode": "raw_only_history"},
        market="KR",
        created_at=pd.Timestamp("2026-04-11T00:00:00Z").to_pydatetime(),
    )

    latest_operational = warehouse.read_latest_investor_flow_run(
        market="KR",
        reasons=("manual_refresh",),
    )

    assert latest_operational["reason"] == "manual_refresh"
    assert latest_operational["requested_end"] == "20260408"
