from __future__ import annotations

import sys
import types

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
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    columns = pd.MultiIndex.from_product([["SPY", "XLK"], ["Close"]])
    raw = pd.DataFrame([[100.0, 200.0], [101.0, 202.0]], index=index, columns=columns)
    fake_module = types.SimpleNamespace(download=lambda **kwargs: raw)
    monkeypatch.setitem(sys.modules, "yfinance", fake_module)

    frame = yf_sectors.fetch_sector_prices(["SPY", "XLK"], "20240102", "20240103")

    assert set(frame["index_code"].astype(str).unique()) == {"SPY", "XLK"}
    assert len(frame) == 4
    assert frame["close"].dtype == "float64"


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


def test_load_sector_prices_returns_cached_when_write_lock_blocks_live_refresh(monkeypatch):
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

    external_ro = duckdb.connect(str(warehouse.WAREHOUSE_PATH), read_only=True)
    try:
        status, frame = yf_sectors.load_sector_prices(["SPY", "XLK"], "20240102", "20240105")
    finally:
        external_ro.close()
        warehouse.close_cached_read_only_connection()

    assert status == "CACHED"
    assert frame.index.max() == pd.Timestamp("2024-01-03")
    stored = read_market_prices(["SPY", "XLK"], "20240102", "20240105", market="US")
    assert stored.index.max() == pd.Timestamp("2024-01-03")


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
