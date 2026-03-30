from __future__ import annotations

import sys
import types

import pandas as pd

import src.data_sources.yfinance_sectors as yf_sectors
from src.data_sources.warehouse import read_market_prices, upsert_index_dimension, upsert_market_prices


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

    status, cached = yf_sectors.load_sector_prices(["SPY"], "20240101", "20240131")

    assert status == "CACHED"
    assert len(cached) == 1
    assert len(read_market_prices(["SPY"], "20240101", "20240131", market="US")) == 1


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
