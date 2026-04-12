from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timezone
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

import scripts.backfill_investor_flow_history as backfill_flow_script
import scripts.bootstrap_warehouse as bootstrap_script
import scripts.sync_warehouse as sync_script
import src.data_sources.warehouse as warehouse


def _seed_market_status(*, provider: str = "OPENAPI") -> None:
    warehouse.ensure_warehouse_schema()
    warehouse.record_ingest_run(
        dataset="market_prices",
        reason="test_seed",
        provider=provider,
        requested_start="20240101",
        requested_end="20240131",
        status="LIVE",
        coverage_complete=True,
        failed_days=[],
        failed_codes={},
        delta_keys=["1001"],
        row_count=1,
        summary={"status": "LIVE"},
    )
    warehouse.update_ingest_watermark(
        dataset="market_prices",
        watermark_key="20240131",
        status="LIVE",
        coverage_complete=True,
        provider=provider,
        details={"reason": "test_seed"},
    )


def test_ensure_warehouse_schema_is_idempotent():
    warehouse.ensure_warehouse_schema()
    warehouse.ensure_warehouse_schema()

    con = duckdb.connect(str(warehouse.WAREHOUSE_PATH), read_only=True)
    try:
        tables = {
            row[0]
            for row in con.execute("SHOW TABLES").fetchall()
        }
    finally:
        con.close()

    assert {
        "dim_index",
        "fact_krx_index_daily",
        "dim_macro_series",
        "fact_macro_monthly",
        "ingest_runs",
        "ingest_watermarks",
    }.issubset(tables)


def test_read_dataset_status_skips_schema_write_when_read_schema_ready(monkeypatch):
    _seed_market_status()
    state = {"calls": 0}

    def _unexpected_schema_write():
        state["calls"] += 1
        raise AssertionError("read path should not attempt schema writes once the warehouse is ready")

    monkeypatch.setattr(warehouse, "ensure_warehouse_schema", _unexpected_schema_write)

    status = warehouse.read_dataset_status("market_prices")

    assert status["provider"] == "OPENAPI"
    assert status["status"] == "LIVE"
    assert state["calls"] == 0


def test_get_dataset_artifact_key_survives_external_read_only_connection():
    _seed_market_status()

    external_ro = duckdb.connect(str(warehouse.WAREHOUSE_PATH), read_only=True)
    try:
        artifact_key = warehouse.get_dataset_artifact_key("market_prices")
    finally:
        external_ro.close()

    assert artifact_key[2:] == ("20240131", "LIVE", "20240131")


def test_ensure_warehouse_schema_normalizes_connection_conflict():
    bootstrap = duckdb.connect(str(warehouse.WAREHOUSE_PATH))
    bootstrap.close()

    external_ro = duckdb.connect(str(warehouse.WAREHOUSE_PATH), read_only=True)
    try:
        with pytest.raises(RuntimeError, match="Cannot acquire write lock on warehouse.duckdb"):
            warehouse.ensure_warehouse_schema()
    finally:
        external_ro.close()


def test_warehouse_upserts_are_idempotent():
    market_idx = pd.date_range("2024-01-01", periods=2, freq="B")
    market_frame = pd.DataFrame(
        {
            "index_code": ["1001", "1001"],
            "index_name": ["KOSPI", "KOSPI"],
            "close": [100.0, 101.0],
        },
        index=market_idx,
    )
    macro_frame = pd.DataFrame(
        {
            "series_id": ["722Y001/0101000", "722Y001/0101000"],
            "value": [1.0, 2.0],
            "source": ["ECOS", "ECOS"],
            "fetched_at": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
            "is_provisional": [False, False],
        },
        index=pd.period_range("2024-01", periods=2, freq="M"),
    )

    warehouse.upsert_index_dimension(
        [
            {
                "index_code": "1001",
                "index_name": "KOSPI",
                "family": "KOSPI",
                "is_benchmark": True,
                "is_active": True,
                "export_sector": False,
            }
        ]
    )
    warehouse.upsert_market_prices(market_frame, provider="OPENAPI")
    warehouse.upsert_market_prices(market_frame, provider="OPENAPI")

    warehouse.upsert_macro_dimension(
        [
            {
                "series_alias": "base_rate",
                "provider": "ECOS",
                "provider_series_id": "722Y001/0101000",
                "enabled": True,
                "label": "Base Rate",
                "unit": "%",
            }
        ]
    )
    warehouse.upsert_macro_series_frame(
        series_alias="base_rate",
        provider="ECOS",
        provider_series_id="722Y001/0101000",
        frame=macro_frame,
    )
    warehouse.upsert_macro_series_frame(
        series_alias="base_rate",
        provider="ECOS",
        provider_series_id="722Y001/0101000",
        frame=macro_frame,
    )

    assert warehouse.market_row_count() == 2
    assert warehouse.macro_row_count() == 2


def test_bootstrap_warehouse_cli_reports_success(monkeypatch):
    warm_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        bootstrap_script,
        "_parse_args",
        lambda: Namespace(prices_years=5, macro_years=10, as_of="20260306"),
    )
    monkeypatch.setattr(
        bootstrap_script,
        "_load_configs",
        lambda: (
            {"benchmark": {"code": "1001"}, "regimes": {"Recovery": {"sectors": [{"code": "5044"}]}}},
            {"ecos": {}, "kosis": {}},
        ),
    )
    monkeypatch.setattr(bootstrap_script, "get_last_business_day", lambda **kwargs: datetime(2026, 3, 6).date())
    def _fake_warm(*args, **kwargs):
        warm_calls.append({"args": args, "kwargs": kwargs})
        return (
            ("LIVE", pd.DataFrame({"index_code": ["1001"], "index_name": ["KOSPI"], "close": [1.0]}, index=pd.DatetimeIndex(["2026-03-06"]))),
            {"coverage_complete": True},
        )

    monkeypatch.setattr(
        bootstrap_script,
        "warm_sector_price_cache",
        _fake_warm,
    )
    monkeypatch.setattr(
        bootstrap_script,
        "sync_macro_warehouse",
        lambda **kwargs: (
            "LIVE",
            pd.DataFrame(
                {
                    "series_id": ["722Y001/0101000"],
                    "value": [1.0],
                    "source": ["ECOS"],
                    "fetched_at": [datetime.now(timezone.utc)],
                    "is_provisional": [False],
                },
                index=pd.period_range("2026-03", periods=1, freq="M"),
            ),
            {"coverage_complete": True},
        ),
    )
    monkeypatch.setattr(
        bootstrap_script,
        "read_market_prices",
        lambda *args, **kwargs: pd.DataFrame(
            {"index_code": ["1001"], "index_name": ["KOSPI"], "close": [1.0]},
            index=pd.DatetimeIndex(["2026-03-06"]),
        ),
    )
    monkeypatch.setattr(
        bootstrap_script,
        "is_market_coverage_complete",
        lambda *args, **kwargs: True,
    )

    assert bootstrap_script.main() == 0
    assert warm_calls
    assert all(call["kwargs"]["force"] is True for call in warm_calls)


def test_sync_warehouse_cli_reports_success(monkeypatch):
    monkeypatch.setattr(
        sync_script,
        "_parse_args",
        lambda: Namespace(prices_years=5, macro_years=10, as_of="20260306"),
    )
    monkeypatch.setattr(
        sync_script,
        "_load_configs",
        lambda: (
            {"benchmark": {"code": "1001"}, "regimes": {"Recovery": {"sectors": [{"code": "5044"}]}}},
            {"ecos": {}, "kosis": {}},
        ),
    )
    monkeypatch.setattr(sync_script, "get_last_business_day", lambda **kwargs: datetime(2026, 3, 6).date())
    monkeypatch.setattr(sync_script, "get_market_latest_dates", lambda codes: {})
    monkeypatch.setattr(
        sync_script,
        "warm_sector_price_cache",
        lambda *args, **kwargs: (
            ("CACHED", pd.DataFrame({"index_code": ["1001"], "index_name": ["KOSPI"], "close": [1.0]}, index=pd.DatetimeIndex(["2026-03-06"]))),
            {"coverage_complete": True},
        ),
    )
    monkeypatch.setattr(
        sync_script,
        "read_market_prices",
        lambda *args, **kwargs: pd.DataFrame(
            {"index_code": ["1001"], "index_name": ["KOSPI"], "close": [1.0]},
            index=pd.DatetimeIndex(["2026-03-06"]),
        ),
    )
    monkeypatch.setattr(
        sync_script,
        "sync_macro_warehouse",
        lambda **kwargs: (
            "CACHED",
            pd.DataFrame(
                {
                    "series_id": ["722Y001/0101000"],
                    "value": [1.0],
                    "source": ["ECOS"],
                    "fetched_at": [datetime.now(timezone.utc)],
                    "is_provisional": [False],
                },
                index=pd.period_range("2026-03", periods=1, freq="M"),
            ),
            {"coverage_complete": True},
        ),
    )

    assert sync_script.main() == 0


def test_backfill_investor_flow_history_cli_reports_success(monkeypatch):
    monkeypatch.setattr(
        backfill_flow_script,
        "_parse_args",
        lambda: Namespace(
            market="KR",
            mode="full",
            end_date="20260306",
            oldest_date="20250102",
            earliest_candidate="19900101",
            chunk_business_days=20,
            retry_attempts=3,
            retry_sleep_sec=30.0,
            assume_non_regression_passed=True,
        ),
    )
    monkeypatch.setattr(
        backfill_flow_script,
        "_load_configs",
        lambda *args, **kwargs: (
            {"benchmark_code": "1001"},
            {"regimes": {"Recovery": {"sectors": [{"code": "5044", "name": "KRX 반도체"}]}}},
            {},
            SimpleNamespace(benchmark_code="1001"),
        ),
    )
    monkeypatch.setattr(
        backfill_flow_script,
        "_run_phase2_full_backfill",
        lambda **kwargs: {
            "status": "LIVE",
            "requested_end": "20260306",
            "passed": True,
        },
    )

    assert backfill_flow_script.main() == 0


def test_sync_warehouse_cli_uses_incremental_market_start(monkeypatch):
    calls: list[tuple[list[str], str, str, str, bool]] = []

    monkeypatch.setattr(
        sync_script,
        "_parse_args",
        lambda: Namespace(prices_years=5, macro_years=10, as_of="20260306"),
    )
    monkeypatch.setattr(
        sync_script,
        "_load_configs",
        lambda: (
            {"benchmark": {"code": "1001"}, "regimes": {"Recovery": {"sectors": [{"code": "5044"}]}}},
            {"ecos": {}, "kosis": {}},
        ),
    )
    monkeypatch.setattr(sync_script, "get_last_business_day", lambda **kwargs: datetime(2026, 3, 6).date())
    monkeypatch.setattr(
        sync_script,
        "get_market_latest_dates",
        lambda codes: {"1001": "20260305", "5044": "20260304"},
    )
    monkeypatch.setattr(
        sync_script,
        "warm_sector_price_cache",
        lambda index_codes, start, end, *, reason, force: (
            calls.append((list(index_codes), start, end, reason, force))
            or (
                ("LIVE", pd.DataFrame({"index_code": ["1001"], "index_name": ["KOSPI"], "close": [1.0]}, index=pd.DatetimeIndex(["2026-03-06"]))),
                {"coverage_complete": True},
            )
        ),
    )
    monkeypatch.setattr(
        sync_script,
        "read_market_prices",
        lambda *args, **kwargs: pd.DataFrame(
            {"index_code": ["1001"], "index_name": ["KOSPI"], "close": [1.0]},
            index=pd.DatetimeIndex(["2026-03-06"]),
        ),
    )
    monkeypatch.setattr(
        sync_script,
        "sync_macro_warehouse",
        lambda **kwargs: (
            "CACHED",
            pd.DataFrame(
                {
                    "series_id": ["722Y001/0101000"],
                    "value": [1.0],
                    "source": ["ECOS"],
                    "fetched_at": [datetime.now(timezone.utc)],
                    "is_provisional": [False],
                },
                index=pd.period_range("2026-03", periods=1, freq="M"),
            ),
            {"coverage_complete": True},
        ),
    )

    assert sync_script.main() == 0
    assert calls == [(["5044", "1001"], "20260305", "20260306", "sync_warehouse", False)]


def test_bootstrap_market_retries_failed_chunk(monkeypatch):
    calls: list[int] = []

    def _fake_warm(*args, **kwargs):
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            return ("SAMPLE", pd.DataFrame()), {"coverage_complete": False, "failed_days": ["20240102"]}
        return (
            ("LIVE", pd.DataFrame({"index_code": ["1001"], "index_name": ["KOSPI"], "close": [1.0]}, index=pd.DatetimeIndex(["2024-01-02"]))),
            {"coverage_complete": True},
        )

    monkeypatch.setattr(bootstrap_script, "warm_sector_price_cache", _fake_warm)
    monkeypatch.setattr(
        bootstrap_script,
        "read_market_prices",
        lambda *args, **kwargs: pd.DataFrame(
            {"index_code": ["1001"], "index_name": ["KOSPI"], "close": [1.0]},
            index=pd.DatetimeIndex(["2024-01-02"]),
        ),
    )
    monkeypatch.setattr(bootstrap_script, "is_market_coverage_complete", lambda *args, **kwargs: True)
    monkeypatch.setattr(bootstrap_script.time, "sleep", lambda *_args, **_kwargs: None)

    status, frame, summary = bootstrap_script._bootstrap_market(
        ["1001"],
        start_date=datetime(2024, 1, 1).date(),
        end_date=datetime(2024, 1, 31).date(),
        benchmark_code="1001",
        chunk_months=1,
        chunk_retries=2,
        chunk_retry_sleep_sec=0.0,
    )

    assert status == "LIVE"
    assert not frame.empty
    assert summary["coverage_complete"] is True
    assert calls == [1, 2]
