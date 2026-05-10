from __future__ import annotations

import duckdb
import pandas as pd
import pytest

import src.data_sources.warehouse as warehouse
import src.data_sources.yfinance_sectors as yf_sectors
from src.data_sources.warehouse import read_market_prices, upsert_index_dimension, upsert_market_prices


@pytest.fixture(autouse=True)
def _isolated_us_curated_path(tmp_path, monkeypatch):
    monkeypatch.setattr(yf_sectors, "CURATED_PATH", tmp_path / "curated" / "sector_prices_us.parquet")


def test_fetch_sector_prices_normalizes_yfinance_download(monkeypatch):
    raw = pd.DataFrame(
        {
            "ticker": ["SPY", "XLK", "SPY", "XLK"],
            "close": [100.0, 200.0, 101.0, 202.0],
            "volume": [1_000.0, 2_000.0, 1_100.0, 2_100.0],
        },
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
    )
    monkeypatch.setattr(yf_sectors, "fetch_yahoo_chart_history_batch", lambda *args, **kwargs: raw)

    frame = yf_sectors.fetch_sector_prices(["SPY", "XLK"], "20240102", "20240103")

    assert set(frame["index_code"].astype(str).unique()) == {"SPY", "XLK"}
    assert len(frame) == 4
    assert frame["close"].dtype == "float64"


def test_fetch_sector_prices_names_us_index_tickers(monkeypatch):
    raw = pd.DataFrame(
        {
            "ticker": ["^GSPC", "^IXIC"],
            "close": [7165.08, 24836.60],
            "volume": [3_179_035_878, 7_000_000_000],
        },
        index=pd.DatetimeIndex(["2026-04-24", "2026-04-24"]),
    )
    monkeypatch.setattr(yf_sectors, "fetch_yahoo_chart_history_batch", lambda *args, **kwargs: raw)

    frame = yf_sectors.fetch_sector_prices(["^GSPC", "^IXIC"], "20260424", "20260424")

    assert frame[["index_code", "index_name", "close"]].to_dict("records") == [
        {"index_code": "^GSPC", "index_name": "S&P 500", "close": 7165.08},
        {"index_code": "^IXIC", "index_name": "Nasdaq Composite", "close": 24836.60},
    ]


def test_load_sector_prices_reads_cached_us_rows():
    frame = pd.DataFrame(
        {"index_code": ["SPY"], "index_name": ["S&P 500"], "close": [500.0]},
        index=pd.DatetimeIndex(["2024-01-02"]),
    )
    upsert_index_dimension(
        [{"index_code": "SPY", "index_name": "S&P 500", "family": "ETF", "is_benchmark": True, "is_active": True, "export_sector": False}],
        market="US",
    )
    upsert_market_prices(frame, provider="YFINANCE", market="US")

    status, cached = yf_sectors.load_sector_prices(["SPY"], "20240101", "20240102")

    assert status == "CACHED"
    assert len(cached) == 1
    assert len(read_market_prices(["SPY"], "20240101", "20240102", market="US")) == 1


def test_load_sector_prices_persists_live_us_rows(monkeypatch):
    live = pd.DataFrame(
        {
            "index_code": ["SPY", "XLK"],
            "index_name": ["S&P 500", "Technology"],
            "close": [500.0, 200.0],
        },
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-02"]),
    )
    monkeypatch.setattr(yf_sectors, "fetch_sector_prices", lambda *args, **kwargs: live)

    status, frame = yf_sectors.load_sector_prices(["SPY", "XLK"], "20240101", "20240131")

    assert status == "LIVE"
    assert set(frame["index_code"].astype(str).unique()) == {"SPY", "XLK"}
    stored = read_market_prices(["SPY", "XLK"], "20240101", "20240131", market="US")
    assert set(stored["index_code"].astype(str).unique()) == {"SPY", "XLK"}


def test_load_sector_prices_refreshes_stale_cached_us_rows(monkeypatch):
    stale = pd.DataFrame(
        {
            "index_code": ["SPY", "XLK", "SPY", "XLK"],
            "index_name": ["S&P 500", "Technology", "S&P 500", "Technology"],
            "close": [500.0, 200.0, 501.0, 201.0],
        },
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
    )
    live = pd.DataFrame(
        {
            "index_code": ["SPY", "XLK", "SPY", "XLK"],
            "index_name": ["S&P 500", "Technology", "S&P 500", "Technology"],
            "close": [510.0, 210.0, 511.0, 211.0],
        },
        index=pd.DatetimeIndex(["2024-01-04", "2024-01-04", "2024-01-05", "2024-01-05"]),
    )
    upsert_index_dimension(
        [
            {"index_code": "SPY", "index_name": "S&P 500", "family": "ETF", "is_benchmark": True, "is_active": True, "export_sector": False},
            {"index_code": "XLK", "index_name": "Technology", "family": "ETF", "is_benchmark": False, "is_active": True, "export_sector": False},
        ],
        market="US",
    )
    upsert_market_prices(stale, provider="YFINANCE", market="US")
    monkeypatch.setattr(yf_sectors, "fetch_sector_prices", lambda *args, **kwargs: live)

    status, frame = yf_sectors.load_sector_prices(["SPY", "XLK"], "20240102", "20240105")

    assert status == "LIVE"
    assert frame.index.max() == pd.Timestamp("2024-01-05")
    stored = read_market_prices(["SPY", "XLK"], "20240102", "20240105", market="US")
    assert stored.index.max() == pd.Timestamp("2024-01-05")
    assert len(stored) == 8


def test_load_sector_prices_defers_live_refresh_when_cached_rows_exist_and_write_lock_is_unavailable(monkeypatch, caplog):
    stale = pd.DataFrame(
        {
            "index_code": ["SPY", "XLK", "SPY", "XLK"],
            "index_name": ["S&P 500", "Technology", "S&P 500", "Technology"],
            "close": [500.0, 200.0, 501.0, 201.0],
        },
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
    )
    live = pd.DataFrame(
        {
            "index_code": ["SPY", "XLK", "SPY", "XLK"],
            "index_name": ["S&P 500", "Technology", "S&P 500", "Technology"],
            "close": [510.0, 210.0, 511.0, 211.0],
        },
        index=pd.DatetimeIndex(["2024-01-04", "2024-01-04", "2024-01-05", "2024-01-05"]),
    )
    upsert_index_dimension(
        [
            {"index_code": "SPY", "index_name": "S&P 500", "family": "ETF", "is_benchmark": True, "is_active": True, "export_sector": False},
            {"index_code": "XLK", "index_name": "Technology", "family": "ETF", "is_benchmark": False, "is_active": True, "export_sector": False},
        ],
        market="US",
    )
    upsert_market_prices(stale, provider="YFINANCE", market="US")
    fetch_calls = 0

    def fake_fetch(*args, **kwargs):
        nonlocal fetch_calls
        fetch_calls += 1
        return live

    monkeypatch.setattr(yf_sectors, "fetch_sector_prices", fake_fetch)

    external_ro = duckdb.connect(str(warehouse.WAREHOUSE_PATH), read_only=True)
    try:
        with caplog.at_level("INFO", logger=yf_sectors.logger.name):
            status, frame = yf_sectors.load_sector_prices(["SPY", "XLK"], "20240102", "20240105")
    finally:
        external_ro.close()
        warehouse.close_cached_read_only_connection()

    assert status == "CACHED"
    assert fetch_calls == 0
    assert frame.index.max() == pd.Timestamp("2024-01-03")
    stored = read_market_prices(["SPY", "XLK"], "20240102", "20240105", market="US")
    assert stored.index.max() == pd.Timestamp("2024-01-03")
    assert "live refresh deferred" in caplog.text
    assert "US market live refresh skipped" not in caplog.text
    assert "US market bookkeeping skipped" not in caplog.text


def test_manual_price_refresh_attempts_live_refresh_even_when_write_lock_is_unavailable(monkeypatch):
    stale = pd.DataFrame(
        {
            "index_code": ["SPY", "XLK", "SPY", "XLK"],
            "index_name": ["S&P 500", "Technology", "S&P 500", "Technology"],
            "close": [500.0, 200.0, 501.0, 201.0],
        },
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
    )
    live = pd.DataFrame(
        {
            "index_code": ["SPY", "XLK", "SPY", "XLK"],
            "index_name": ["S&P 500", "Technology", "S&P 500", "Technology"],
            "close": [510.0, 210.0, 511.0, 211.0],
        },
        index=pd.DatetimeIndex(["2024-01-04", "2024-01-04", "2024-01-05", "2024-01-05"]),
    )
    upsert_index_dimension(
        [
            {"index_code": "SPY", "index_name": "S&P 500", "family": "ETF", "is_benchmark": True, "is_active": True, "export_sector": False},
            {"index_code": "XLK", "index_name": "Technology", "family": "ETF", "is_benchmark": False, "is_active": True, "export_sector": False},
        ],
        market="US",
    )
    upsert_market_prices(stale, provider="YFINANCE", market="US")
    fetch_calls = 0

    def fake_fetch(*args, **kwargs):
        nonlocal fetch_calls
        fetch_calls += 1
        return live

    monkeypatch.setattr(yf_sectors, "fetch_sector_prices", fake_fetch)

    external_ro = duckdb.connect(str(warehouse.WAREHOUSE_PATH), read_only=True)
    try:
        (status, frame), summary = yf_sectors.run_manual_price_refresh(["SPY", "XLK"], "20240102", "20240105")
    finally:
        external_ro.close()
        warehouse.close_cached_read_only_connection()

    assert status == "LIVE"
    assert fetch_calls == 1
    assert frame.index.max() == pd.Timestamp("2024-01-05")
    assert summary["status"] == "LIVE"
    assert summary["warehouse_write_skipped"] is True
    assert summary["coverage_complete"] is False


def test_load_sector_prices_returns_sample_when_write_lock_blocks_live_refresh_without_cache(monkeypatch):
    warehouse.ensure_warehouse_schema()
    live = pd.DataFrame(
        {
            "index_code": ["SPY"],
            "index_name": ["S&P 500"],
            "close": [510.0],
        },
        index=pd.DatetimeIndex(["2024-01-05"]),
    )
    monkeypatch.setattr(yf_sectors, "fetch_sector_prices", lambda *args, **kwargs: live)

    external_ro = duckdb.connect(str(warehouse.WAREHOUSE_PATH), read_only=True)
    try:
        status, frame = yf_sectors.load_sector_prices(["SPY"], "20240102", "20240105")
    finally:
        external_ro.close()
        warehouse.close_cached_read_only_connection()

    assert status == "SAMPLE"
    assert set(frame["index_code"].astype(str).unique()) == {"SPY"}
    assert read_market_prices(["SPY"], "20240102", "20240105", market="US").empty


def test_load_sector_prices_ignores_bookkeeping_failures_after_live_persist(monkeypatch):
    live = pd.DataFrame(
        {
            "index_code": ["SPY", "XLK"],
            "index_name": ["S&P 500", "Technology"],
            "close": [500.0, 200.0],
        },
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-02"]),
    )
    monkeypatch.setattr(yf_sectors, "fetch_sector_prices", lambda *args, **kwargs: live)
    monkeypatch.setattr(
        yf_sectors,
        "record_ingest_run",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("telemetry unavailable")),
    )

    status, frame = yf_sectors.load_sector_prices(["SPY", "XLK"], "20240101", "20240131")

    assert status == "LIVE"
    assert set(frame["index_code"].astype(str).unique()) == {"SPY", "XLK"}
    stored = read_market_prices(["SPY", "XLK"], "20240101", "20240131", market="US")
    assert set(stored["index_code"].astype(str).unique()) == {"SPY", "XLK"}


def test_load_sector_prices_uses_cached_rows_when_yahoo_batch_fails(monkeypatch):
    cached = pd.DataFrame(
        {
            "index_code": ["SPY", "XLV"],
            "index_name": ["S&P 500", "Health Care"],
            "close": [500.0, 130.0],
        },
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-02"]),
    )
    upsert_index_dimension(
        [
            {"index_code": "SPY", "index_name": "S&P 500", "family": "ETF", "is_benchmark": True, "is_active": True, "export_sector": False},
            {"index_code": "XLV", "index_name": "Health Care", "family": "ETF", "is_benchmark": False, "is_active": True, "export_sector": False},
        ],
        market="US",
    )
    upsert_market_prices(cached, provider="YFINANCE", market="US")
    monkeypatch.setattr(
        yf_sectors,
        "fetch_yahoo_chart_history_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Yahoo chart batch failed: XLV=timeout")),
    )

    status, frame = yf_sectors.load_sector_prices(["SPY", "XLV"], "20240101", "20240102")

    assert status == "CACHED"
    assert set(frame["index_code"].astype(str).unique()) == {"SPY", "XLV"}


def test_manual_price_refresh_reports_live_fetch_failure_when_using_cache(monkeypatch):
    cached = pd.DataFrame(
        {
            "index_code": ["SPY", "XLV"],
            "index_name": ["S&P 500", "Health Care"],
            "close": [500.0, 130.0],
        },
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-02"]),
    )
    upsert_index_dimension(
        [
            {"index_code": "SPY", "index_name": "S&P 500", "family": "ETF", "is_benchmark": True, "is_active": True, "export_sector": False},
            {"index_code": "XLV", "index_name": "Health Care", "family": "ETF", "is_benchmark": False, "is_active": True, "export_sector": False},
        ],
        market="US",
    )
    upsert_market_prices(cached, provider="YFINANCE", market="US")
    monkeypatch.setattr(
        yf_sectors,
        "fetch_yahoo_chart_history_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Yahoo chart batch failed: XLV=timeout")),
    )

    (status, frame), summary = yf_sectors.run_manual_price_refresh(["SPY", "XLV"], "20240101", "20240105")

    assert status == "CACHED"
    assert set(frame["index_code"].astype(str).unique()) == {"SPY", "XLV"}
    assert summary["failed_codes"]["live_fetch"] == "Yahoo chart batch failed: XLV=timeout"
