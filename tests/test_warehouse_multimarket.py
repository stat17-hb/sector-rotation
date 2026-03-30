from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import pandas as pd

import src.data_sources.warehouse as warehouse


def test_market_and_macro_rows_are_scoped_by_market():
    market_kr = pd.DataFrame(
        {"index_code": ["1001"], "index_name": ["KOSPI"], "close": [1.0]},
        index=pd.DatetimeIndex(["2024-01-02"]),
    )
    market_us = pd.DataFrame(
        {"index_code": ["SPY"], "index_name": ["S&P 500"], "close": [2.0]},
        index=pd.DatetimeIndex(["2024-01-02"]),
    )
    warehouse.upsert_index_dimension(
        [{"index_code": "1001", "index_name": "KOSPI", "family": "IDX", "is_benchmark": True, "is_active": True, "export_sector": False}],
        market="KR",
    )
    warehouse.upsert_index_dimension(
        [{"index_code": "SPY", "index_name": "S&P 500", "family": "ETF", "is_benchmark": True, "is_active": True, "export_sector": False}],
        market="US",
    )
    warehouse.upsert_market_prices(market_kr, provider="PYKRX", market="KR")
    warehouse.upsert_market_prices(market_us, provider="YFINANCE", market="US")

    now = datetime.now(timezone.utc)
    macro_kr = pd.DataFrame(
        {
            "series_id": ["KR_SERIES"],
            "value": [1.0],
            "source": ["ECOS"],
            "fetched_at": [now],
            "is_provisional": [False],
        },
        index=pd.PeriodIndex(["2024-01"], freq="M"),
    )
    macro_us = pd.DataFrame(
        {
            "series_id": ["US_SERIES"],
            "value": [2.0],
            "source": ["FRED"],
            "fetched_at": [now],
            "is_provisional": [False],
        },
        index=pd.PeriodIndex(["2024-01"], freq="M"),
    )
    warehouse.upsert_macro_dimension(
        [{"series_alias": "leading_index", "provider": "ECOS", "provider_series_id": "KR_SERIES", "enabled": True, "label": "", "unit": ""}],
        market="KR",
    )
    warehouse.upsert_macro_dimension(
        [{"series_alias": "leading_index", "provider": "FRED", "provider_series_id": "US_SERIES", "enabled": True, "label": "", "unit": ""}],
        market="US",
    )
    warehouse.upsert_macro_series_frame(
        series_alias="leading_index",
        provider="ECOS",
        provider_series_id="KR_SERIES",
        frame=macro_kr,
        market="KR",
    )
    warehouse.upsert_macro_series_frame(
        series_alias="leading_index",
        provider="FRED",
        provider_series_id="US_SERIES",
        frame=macro_us,
        market="US",
    )

    assert len(warehouse.read_market_prices(["1001"], "20240101", "20240131", market="KR")) == 1
    assert len(warehouse.read_market_prices(["SPY"], "20240101", "20240131", market="US")) == 1
    assert warehouse.read_macro_data(series_aliases=["leading_index"], start_ym="202401", end_ym="202401", market="KR")["series_id"].iloc[0] == "KR_SERIES"
    assert warehouse.read_macro_data(series_aliases=["leading_index"], start_ym="202401", end_ym="202401", market="US")["series_id"].iloc[0] == "US_SERIES"


def test_legacy_schema_migrates_to_market_aware_tables(tmp_path):
    con = duckdb.connect(str(warehouse.WAREHOUSE_PATH))
    try:
        con.execute(
            """
            CREATE TABLE dim_index (
                index_code VARCHAR PRIMARY KEY,
                index_name VARCHAR,
                family VARCHAR,
                is_benchmark BOOLEAN,
                is_active BOOLEAN,
                export_sector BOOLEAN,
                updated_at TIMESTAMPTZ
            )
            """
        )
        con.execute(
            """
            CREATE TABLE fact_krx_index_daily (
                trade_date DATE,
                index_code VARCHAR,
                close DOUBLE,
                provider VARCHAR,
                loaded_at TIMESTAMPTZ,
                PRIMARY KEY (trade_date, index_code)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE dim_macro_series (
                series_alias VARCHAR PRIMARY KEY,
                provider VARCHAR,
                provider_series_id VARCHAR,
                enabled BOOLEAN,
                label VARCHAR,
                unit VARCHAR,
                updated_at TIMESTAMPTZ
            )
            """
        )
        con.execute(
            """
            CREATE TABLE fact_macro_monthly (
                period_month DATE,
                series_alias VARCHAR,
                provider VARCHAR,
                provider_series_id VARCHAR,
                value DOUBLE,
                is_provisional BOOLEAN,
                fetched_at TIMESTAMPTZ,
                PRIMARY KEY (period_month, series_alias)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE ingest_runs (
                run_id VARCHAR PRIMARY KEY,
                dataset VARCHAR,
                reason VARCHAR,
                provider VARCHAR,
                requested_start VARCHAR,
                requested_end VARCHAR,
                status VARCHAR,
                coverage_complete BOOLEAN,
                failed_days_json VARCHAR,
                failed_codes_json VARCHAR,
                delta_keys_json VARCHAR,
                row_count BIGINT,
                aborted BOOLEAN,
                abort_reason VARCHAR,
                predicted_requests INTEGER,
                processed_requests INTEGER,
                summary_json VARCHAR,
                created_at TIMESTAMPTZ
            )
            """
        )
        con.execute(
            """
            CREATE TABLE ingest_watermarks (
                dataset VARCHAR PRIMARY KEY,
                watermark_key VARCHAR,
                coverage_complete BOOLEAN,
                status VARCHAR,
                provider VARCHAR,
                details_json VARCHAR,
                updated_at TIMESTAMPTZ
            )
            """
        )
        con.execute("INSERT INTO dim_index VALUES ('1001', 'KOSPI', 'IDX', TRUE, TRUE, FALSE, now())")
        con.execute("INSERT INTO fact_krx_index_daily VALUES (DATE '2024-01-02', '1001', 1.0, 'PYKRX', now())")
        con.execute("INSERT INTO dim_macro_series VALUES ('leading_index', 'ECOS', 'KR_SERIES', TRUE, '', '', now())")
        con.execute("INSERT INTO fact_macro_monthly VALUES (DATE '2024-01-31', 'leading_index', 'ECOS', 'KR_SERIES', 1.0, FALSE, now())")
        con.execute(
            """
            INSERT INTO ingest_runs VALUES (
                'market_prices:test',
                'market_prices',
                'test',
                'PYKRX',
                '20240101',
                '20240102',
                'CACHED',
                TRUE,
                '[]',
                '{}',
                '[]',
                1,
                FALSE,
                '',
                0,
                0,
                '{}',
                now()
            )
            """
        )
        con.execute(
            """
            INSERT INTO ingest_watermarks VALUES (
                'market_prices',
                '20240102',
                TRUE,
                'CACHED',
                'PYKRX',
                '{}',
                now()
            )
            """
        )
    finally:
        con.close()

    warehouse.ensure_warehouse_schema()

    migrated = warehouse.read_market_prices(["1001"], "20240101", "20240131", market="KR")
    assert len(migrated) == 1
    assert warehouse.read_dataset_status("market_prices", market="KR")["provider"] == "PYKRX"
    assert warehouse.probe_dataset_mode("market_prices", market="US") == "SAMPLE"
