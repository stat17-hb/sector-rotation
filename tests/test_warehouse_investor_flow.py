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
