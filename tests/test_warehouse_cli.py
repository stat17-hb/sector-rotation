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


def test_get_dataset_artifact_key_survives_same_process_read_write_connection():
    _seed_market_status()

    external_rw = duckdb.connect(str(warehouse.WAREHOUSE_PATH), read_only=False)
    try:
        artifact_key = warehouse.get_dataset_artifact_key("market_prices")
    finally:
        external_rw.close()
        warehouse.close_cached_read_only_connection()

    assert artifact_key[2:] == ("20240131", "LIVE", "20240131")


def test_read_dataset_data_bounds_returns_actual_stored_ranges():
    market_frame = pd.DataFrame(
        {
            "index_code": ["1001", "1001"],
            "index_name": ["KOSPI", "KOSPI"],
            "close": [2700.0, 2710.0],
        },
        index=pd.to_datetime(["2026-05-01", "2026-05-03"]),
    )
    warehouse.upsert_market_prices(market_frame, provider="OPENAPI")
    warehouse.upsert_macro_dimension(
        [
            {
                "series_alias": "base_rate",
                "provider": "ECOS",
                "provider_series_id": "722Y001/0101000",
                "enabled": True,
            }
        ]
    )
    macro_frame = pd.DataFrame(
        {
            "value": [3.5, 3.5],
            "source": ["ECOS", "ECOS"],
            "fetched_at": [pd.Timestamp("2026-05-10T00:00:00Z")] * 2,
            "is_provisional": [False, False],
        },
        index=pd.period_range("2026-03", "2026-04", freq="M"),
    )
    warehouse.upsert_macro_series_frame(
        series_alias="base_rate",
        provider="ECOS",
        provider_series_id="722Y001/0101000",
        frame=macro_frame,
    )
    sector_frame = pd.DataFrame(
        {
            "sector_code": ["5044", "5044"],
            "sector_name": ["KRX 반도체", "KRX 반도체"],
            "investor_type": ["외국인", "외국인"],
            "buy_amount": [1200, 1300],
            "sell_amount": [800, 900],
            "net_buy_amount": [400, 400],
            "net_flow_ratio": [0.2, 0.2],
        },
        index=pd.to_datetime(["2026-04-08", "2026-04-10"]),
    )
    warehouse.upsert_investor_flow_sector(sector_frame, provider="PYKRX_UNOFFICIAL")

    assert warehouse.read_dataset_data_bounds("market_prices") == {
        "min_trade_date": "20260501",
        "max_trade_date": "20260503",
        "row_count": 2,
    }
    assert warehouse.read_dataset_data_bounds("macro_data") == {
        "min_period_month": "20260331",
        "max_period_month": "20260430",
        "row_count": 2,
    }
    assert warehouse.read_dataset_data_bounds("macro_data", provider="ECOS") == {
        "min_period_month": "20260331",
        "max_period_month": "20260430",
        "row_count": 2,
    }
    assert warehouse.read_dataset_data_bounds("investor_flow") == {
        "min_trade_date": "20260408",
        "max_trade_date": "20260410",
        "row_count": 2,
    }


def test_read_collection_run_history_returns_market_macro_and_flow_runs():
    warehouse.ensure_warehouse_schema()
    seed_runs = [
        ("market_prices", "OPENAPI", datetime(2026, 5, 5, 1, 0, tzinfo=timezone.utc)),
        ("macro_data", "ECOS", datetime(2026, 5, 5, 2, 0, tzinfo=timezone.utc)),
        ("investor_flow", "KRX_UNOFFICIAL", datetime(2026, 5, 5, 3, 0, tzinfo=timezone.utc)),
    ]
    for dataset, provider, created_at in seed_runs:
        warehouse.record_ingest_run(
            dataset=dataset,
            reason="manual_refresh",
            provider=provider,
            requested_start="20260501",
            requested_end="20260505",
            status="LIVE",
            coverage_complete=True,
            failed_days=["20260502"] if dataset == "market_prices" else [],
            failed_codes={"1001": "empty close"} if dataset == "market_prices" else {},
            delta_keys=[],
            row_count=10,
            predicted_requests=4,
            processed_requests=3,
            created_at=created_at,
        )

    history = warehouse.read_collection_run_history(market="KR", limit=15)

    assert history["dataset"].tolist() == ["investor_flow", "macro_data", "market_prices"]
    assert history["provider"].tolist() == ["KRX_UNOFFICIAL", "ECOS", "OPENAPI"]
    assert history["completion_pct"].tolist() == [75.0, 75.0, 75.0]
    market_row = history[history["dataset"].eq("market_prices")].iloc[0]
    assert market_row["failed_days"] == ["20260502"]
    assert market_row["failed_codes"] == {"1001": "empty close"}

    flow_history = warehouse.read_investor_flow_run_history(market="KR", limit=15)
    assert flow_history["reason"].tolist() == ["manual_refresh"]
    assert "dataset" not in flow_history.columns


def test_read_collection_run_history_uses_macro_alias_month_completion_rate():
    warehouse.ensure_warehouse_schema()
    warehouse.upsert_macro_dimension(
        [
            {
                "series_alias": "complete_alias",
                "provider": "ECOS",
                "provider_series_id": "COMPLETE",
                "enabled": True,
            },
            {
                "series_alias": "partial_alias",
                "provider": "ECOS",
                "provider_series_id": "PARTIAL",
                "enabled": True,
            },
        ]
    )
    fetched_at = pd.Timestamp("2026-05-10T00:00:00Z")
    complete_frame = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0],
            "source": ["ECOS"] * 3,
            "fetched_at": [fetched_at] * 3,
            "is_provisional": [False] * 3,
        },
        index=pd.period_range("2026-01", "2026-03", freq="M"),
    )
    partial_frame = complete_frame.iloc[:2].copy()
    warehouse.upsert_macro_series_frame(
        series_alias="complete_alias",
        provider="ECOS",
        provider_series_id="COMPLETE",
        frame=complete_frame,
    )
    warehouse.upsert_macro_series_frame(
        series_alias="partial_alias",
        provider="ECOS",
        provider_series_id="PARTIAL",
        frame=partial_frame,
    )
    warehouse.record_ingest_run(
        dataset="macro_data",
        reason="manual_refresh",
        provider="ECOS",
        requested_start="202601",
        requested_end="202603",
        status="LIVE",
        coverage_complete=False,
        failed_days=[],
        failed_codes={},
        delta_keys=["complete_alias", "partial_alias"],
        row_count=5,
        created_at=datetime(2026, 5, 10, 1, 0, tzinfo=timezone.utc),
    )

    history = warehouse.read_collection_run_history(market="KR", limit=15)

    assert history["dataset"].tolist() == ["macro_data"]
    assert history["completion_pct"].tolist() == [83.3]


def test_read_collection_run_history_does_not_zero_partial_rows_without_request_counters():
    warehouse.ensure_warehouse_schema()
    warehouse.record_ingest_run(
        dataset="investor_flow",
        reason="manual_refresh",
        provider="KRX_UNOFFICIAL",
        requested_start="20260401",
        requested_end="20260410",
        status="LIVE",
        coverage_complete=False,
        failed_days=[],
        failed_codes={},
        delta_keys=["5044"],
        row_count=20,
        predicted_requests=0,
        processed_requests=0,
        created_at=datetime(2026, 5, 10, 1, 0, tzinfo=timezone.utc),
    )

    history = warehouse.read_collection_run_history(market="KR", limit=15)

    assert history["dataset"].tolist() == ["investor_flow"]
    assert pd.isna(history["completion_pct"].iloc[0])


def test_read_collection_run_history_filters_reasons_before_sampling():
    warehouse.ensure_warehouse_schema()
    for reason, created_at in (
        ("manual_refresh", datetime(2026, 5, 10, 1, 0, tzinfo=timezone.utc)),
        ("load_ecos_macro", datetime(2026, 5, 10, 2, 0, tzinfo=timezone.utc)),
    ):
        warehouse.record_ingest_run(
            dataset="macro_data",
            reason=reason,
            provider="ECOS",
            requested_start="202601",
            requested_end="202603",
            status="LIVE",
            coverage_complete=True,
            failed_days=[],
            failed_codes={},
            delta_keys=[],
            row_count=10,
            predicted_requests=1,
            processed_requests=1,
            created_at=created_at,
        )

    history = warehouse.read_collection_run_history(
        market="KR",
        reasons=("manual_refresh",),
        sample_per_dataset=True,
        sample_size=10,
    )

    assert history["reason"].tolist() == ["manual_refresh"]


def test_read_collection_run_history_can_sample_latest_ten_per_dataset():
    warehouse.ensure_warehouse_schema()
    con = warehouse._connect()
    try:
        for dataset in ("market_prices", "macro_data", "investor_flow"):
            for idx in range(12):
                warehouse.record_ingest_run(
                    dataset=dataset,
                    reason="manual_refresh",
                    provider="TEST",
                    requested_start="20260501",
                    requested_end="20260505",
                    status="LIVE",
                    coverage_complete=True,
                    failed_days=[],
                    failed_codes={},
                    delta_keys=[],
                    row_count=idx,
                    predicted_requests=1,
                    processed_requests=1,
                    created_at=datetime(2026, 5, 5, idx, 0, tzinfo=timezone.utc),
                    connection=con,
                )
    finally:
        con.close()

    history = warehouse.read_collection_run_history(
        market="KR",
        sample_per_dataset=True,
        sample_size=10,
    )

    market_rows = history[history["dataset"].eq("market_prices")]
    assert len(market_rows) == 10
    assert market_rows["row_count"].tolist() == [11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    assert market_rows["sample_bucket"].tolist() == ["latest"] * 10


def test_read_collection_run_history_respects_explicit_empty_dataset_selection():
    warehouse.ensure_warehouse_schema()
    warehouse.record_ingest_run(
        dataset="market_prices",
        reason="manual_refresh",
        provider="OPENAPI",
        requested_start="20260501",
        requested_end="20260505",
        status="LIVE",
        coverage_complete=True,
        failed_days=[],
        failed_codes={},
        delta_keys=[],
        row_count=1,
        predicted_requests=1,
        processed_requests=1,
        created_at=datetime(2026, 5, 5, 1, 0, tzinfo=timezone.utc),
    )

    history = warehouse.read_collection_run_history(market="KR", datasets=[])

    assert history.empty


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


def test_bootstrap_warehouse_cli_fails_when_theme_taxonomy_incomplete(monkeypatch):
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
    monkeypatch.setattr(
        bootstrap_script,
        "sync_theme_taxonomy_warehouse",
        lambda **kwargs: (
            "LIVE",
            pd.DataFrame({"index_code": ["5044"]}),
            {"coverage_complete": False, "index_codes": ["5044"]},
        ),
    )
    monkeypatch.setattr(
        bootstrap_script,
        "warm_sector_price_cache",
        lambda *args, **kwargs: (
            ("LIVE", pd.DataFrame({"index_code": ["1001"], "index_name": ["KOSPI"], "close": [1.0]}, index=pd.DatetimeIndex(["2026-03-06"]))),
            {"coverage_complete": True},
        ),
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
    monkeypatch.setattr(bootstrap_script, "is_market_coverage_complete", lambda *args, **kwargs: True)

    assert bootstrap_script.main() == 1


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


def test_sync_warehouse_cli_fails_when_theme_taxonomy_incomplete(monkeypatch):
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
        "sync_theme_taxonomy_warehouse",
        lambda **kwargs: (
            "LIVE",
            pd.DataFrame({"index_code": ["5044"]}),
            {"coverage_complete": False, "index_codes": ["5044"]},
        ),
    )
    monkeypatch.setattr(
        sync_script,
        "warm_sector_price_cache",
        lambda *args, **kwargs: (
            ("LIVE", pd.DataFrame({"index_code": ["1001"], "index_name": ["KOSPI"], "close": [1.0]}, index=pd.DatetimeIndex(["2026-03-06"]))),
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

    assert sync_script.main() == 1


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
    monkeypatch.setattr(
        sync_script,
        "sync_theme_taxonomy_warehouse",
        lambda **kwargs: (
            "LIVE",
            pd.DataFrame({"index_code": ["5044"]}),
            {"coverage_complete": True, "index_codes": ["5044"]},
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
