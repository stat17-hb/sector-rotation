from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta

import pandas as pd

from src.data_sources.krx_constituents import ConstituentLookupResult
import src.data_sources.krx_stock_screening as screening
from src.data_sources.krx_stock_screening import _get_constituents


class _DummyStockModule:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, bool]] = []

    def get_index_portfolio_deposit_file(self, ticker, date=None, alternative=False):
        self.calls.append((ticker, date, alternative))
        return ["005930", "000660"]


def test_get_constituents_calls_pykrx_with_ticker_first():
    stock_module = _DummyStockModule()

    tickers = _get_constituents(stock_module, "20260411", "5044")

    assert tickers == ["005930", "000660"]
    assert stock_module.calls == [("5044", "20260410", False)]


def test_get_constituents_accepts_dataframe_result():
    class _FrameStockModule(_DummyStockModule):
        def get_index_portfolio_deposit_file(self, ticker, date=None, alternative=False):
            self.calls.append((ticker, date, alternative))
            return pd.DataFrame({"weight": [0.5, 0.5]}, index=["005930", "000660"])

    stock_module = _FrameStockModule()

    tickers = _get_constituents(stock_module, "20260411", "5044")

    assert tickers == ["005930", "000660"]
    assert stock_module.calls == [("5044", "20260410", False)]


def test_get_constituents_returns_lookup_helper_result(monkeypatch):
    monkeypatch.setattr(
        screening,
        "lookup_index_constituents",
        lambda *args, **kwargs: ConstituentLookupResult(
            tickers=["035420", "005930"],
            resolved_from="20260410",
            source="krx_raw_payload",
        ),
    )

    tickers = screening._get_constituents(_DummyStockModule(), "20260411", "5044")

    assert tickers == ["035420", "005930"]


def test_load_screened_stocks_cache_only_skips_live_fetch_on_cache_miss(monkeypatch, tmp_path):
    monkeypatch.setattr(screening, "CACHE_PATH", tmp_path / "missing_screening_cache.pkl")
    monkeypatch.setattr(
        screening,
        "_fetch_and_score",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live fetch should not run")),
    )

    status, rows = screening.load_screened_stocks(
        strong_buy_sectors=[{"code": "5044", "name": "KRX 반도체"}],
        allow_live_fetch=False,
    )

    assert status == "UNAVAILABLE"
    assert rows == []


def test_load_screened_stocks_cache_only_returns_cached_rows(monkeypatch, tmp_path):
    cache_path = tmp_path / "screening_cache.pkl"
    monkeypatch.setattr(screening, "CACHE_PATH", cache_path)
    cached_rows = [{"ticker": "005930", "name": "삼성전자"}]
    screening._write_cache([{"code": "5044", "name": "KRX 반도체"}], cached_rows)
    monkeypatch.setattr(
        screening,
        "_fetch_and_score",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live fetch should not run")),
    )

    status, rows = screening.load_screened_stocks(
        strong_buy_sectors=[{"code": "5044", "name": "KRX 반도체"}],
        allow_live_fetch=False,
    )

    assert status == "CACHED"
    assert rows == cached_rows


def test_load_screened_stocks_cache_only_returns_stale_rows(monkeypatch, tmp_path):
    cache_path = tmp_path / "screening_cache.pkl"
    monkeypatch.setattr(screening, "CACHE_PATH", cache_path)
    cached_rows = [{"ticker": "005930", "name": "삼성전자"}]
    sectors = [{"code": "5044", "name": "KRX 반도체"}]
    screening._write_cache(sectors, cached_rows)
    cached = screening.pickle.load(open(cache_path, "rb"))
    cached["ts"] = datetime.now() - timedelta(hours=screening.CACHE_TTL_HOURS + 1)
    screening.pickle.dump(cached, open(cache_path, "wb"))
    monkeypatch.setattr(
        screening,
        "_fetch_and_score",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live fetch should not run")),
    )

    status, rows = screening.load_screened_stocks(
        strong_buy_sectors=sectors,
        allow_live_fetch=False,
    )

    assert status == "STALE_CACHE"
    assert rows == cached_rows


def test_load_screened_stocks_live_failure_falls_back_to_stale_rows(monkeypatch, tmp_path):
    cache_path = tmp_path / "screening_cache.pkl"
    monkeypatch.setattr(screening, "CACHE_PATH", cache_path)
    cached_rows = [{"ticker": "005930", "name": "삼성전자"}]
    sectors = [{"code": "5044", "name": "KRX 반도체"}]
    screening._write_cache(sectors, cached_rows)
    cached = screening.pickle.load(open(cache_path, "rb"))
    cached["ts"] = datetime.now() - timedelta(hours=screening.CACHE_TTL_HOURS + 1)
    screening.pickle.dump(cached, open(cache_path, "wb"))
    monkeypatch.setattr(
        screening,
        "_fetch_and_score",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("KRX unavailable")),
    )

    status, rows = screening.load_screened_stocks(
        strong_buy_sectors=sectors,
        force_refresh=True,
        allow_live_fetch=True,
    )

    assert status == "STALE_CACHE"
    assert rows == cached_rows


def test_fetch_and_score_reports_ticker_progress(monkeypatch):
    fake_stock = types.ModuleType("pykrx.stock")
    fake_pykrx = types.ModuleType("pykrx")
    fake_pykrx.stock = fake_stock
    monkeypatch.setitem(sys.modules, "pykrx", fake_pykrx)
    monkeypatch.setitem(sys.modules, "pykrx.stock", fake_stock)

    fake_compat = types.ModuleType("src.data_sources.pykrx_compat")
    fake_compat.ensure_pykrx_transport_compat = lambda: None
    monkeypatch.setitem(sys.modules, "src.data_sources.pykrx_compat", fake_compat)

    fake_stock.get_index_ohlcv_by_date = lambda *args, **kwargs: pd.DataFrame(
        {"종가": [100.0, 101.0, 102.0]},
        index=pd.date_range("2026-04-01", periods=3, freq="B"),
    )
    monkeypatch.setattr(screening, "_last_business_day", lambda: "20260403")
    monkeypatch.setattr(
        screening,
        "_get_constituents",
        lambda stock_module, trade_date, sector_code: ["005930", "000660"] if sector_code == "5044" else ["035420"],
    )

    def _score_stock(**kwargs):
        return {"ticker": kwargs["ticker"], "rs": 1.0}

    monkeypatch.setattr(screening, "_score_stock", _score_stock)
    events: list[dict[str, object]] = []

    rows = screening._fetch_and_score(
        [
            {"code": "5044", "name": "KRX 반도체"},
            {"code": "5046", "name": "KRX 인터넷"},
        ],
        benchmark_code="1001",
        settings={},
        progress_callback=events.append,
    )

    assert [row["ticker"] for row in rows] == ["005930", "000660", "035420"]
    assert events[0] == {"stage": "start", "current": 0, "total": 3}
    ticker_events = [event for event in events if event["stage"] == "ticker"]
    assert [event["current"] for event in ticker_events] == [1, 2, 3]
    assert [event["ticker"] for event in ticker_events] == ["005930", "000660", "035420"]
    assert events[-1] == {"stage": "done", "current": 3, "total": 3}


def test_load_representative_etf_context_cache_only_skips_live_fetch_on_cache_miss(monkeypatch, tmp_path):
    monkeypatch.setattr(screening, "ETF_CONTEXT_CACHE_PATH", tmp_path / "missing_etf_cache.pkl")
    monkeypatch.setattr(
        screening,
        "_fetch_representative_etf_context",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("live ETF fetch should not run")),
    )

    status, rows = screening.load_representative_etf_context(
        strong_buy_sectors=[{"code": "5044", "name": "KRX 반도체"}],
        etf_map={"5044": [{"code": "396500", "name": "TIGER 반도체TOP10"}]},
        allow_live_fetch=False,
    )

    assert status == "UNAVAILABLE"
    assert rows == []


def test_load_representative_etf_context_cache_only_returns_cached_rows(monkeypatch, tmp_path):
    cache_path = tmp_path / "etf_context_cache.pkl"
    monkeypatch.setattr(screening, "ETF_CONTEXT_CACHE_PATH", cache_path)
    sectors = [{"code": "5044", "name": "KRX 반도체"}]
    etf_map = {"5044": [{"code": "396500", "name": "TIGER 반도체TOP10"}]}
    cached_rows = [{"sector_code": "5044", "etf_code": "396500"}]
    screening._write_etf_context_cache(sectors, etf_map, cached_rows)
    monkeypatch.setattr(
        screening,
        "_fetch_representative_etf_context",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("live ETF fetch should not run")),
    )

    status, rows = screening.load_representative_etf_context(
        strong_buy_sectors=sectors,
        etf_map=etf_map,
        allow_live_fetch=False,
    )

    assert status == "CACHED"
    assert rows == cached_rows


def test_build_sector_etf_context_row_selects_highest_recent_trade_value(monkeypatch):
    snapshot = pd.DataFrame(
        {
            "name": ["TIGER 반도체TOP10", "KODEX 반도체"],
            "nav": [10200.0, 9900.0],
            "latest_trade_value": [1_200_000_000.0, 900_000_000.0],
            "market_cap": [0.0, 0.0],
            "net_assets": [220_000_000_000.0, 180_000_000_000.0],
            "listed_shares": [0.0, 0.0],
        },
        index=["396500", "091160"],
    )

    histories = {
        "396500": pd.DataFrame({"거래대금": [800_000_000.0] * 20}, index=pd.date_range("2026-04-01", periods=20, freq="B")),
        "091160": pd.DataFrame({"거래대금": [700_000_000.0] * 20}, index=pd.date_range("2026-04-01", periods=20, freq="B")),
    }

    monkeypatch.setattr(
        screening,
        "_fetch_etf_history",
        lambda **kwargs: histories[str(kwargs["ticker"])],
    )

    row = screening._build_sector_etf_context_row(
        stock_module=object(),
        sector_code="5044",
        sector_name="KRX 반도체",
        mapped_etfs=[
            {"code": "396500", "name": "TIGER 반도체TOP10"},
            {"code": "091160", "name": "KODEX 반도체"},
        ],
        snapshot_df=snapshot,
        snapshot_date="20260428",
        target_date="20260428",
        lookback_days=20,
        min_avg_trading_value=300_000_000,
    )

    assert row["etf_code"] == "396500"
    assert row["execution_state"] == "정상"
    assert row["freshness_label"] == "20260428"


def test_build_sector_etf_context_row_marks_illiquid_when_average_value_is_too_low(monkeypatch):
    snapshot = pd.DataFrame(
        {
            "name": ["TIGER 200 산업재"],
            "nav": [11500.0],
            "latest_trade_value": [120_000_000.0],
            "market_cap": [0.0],
            "net_assets": [40_000_000_000.0],
            "listed_shares": [0.0],
        },
        index=["227550"],
    )

    history = pd.DataFrame(
        {"거래대금": [100_000_000.0] * 20},
        index=pd.date_range("2026-04-01", periods=20, freq="B"),
    )
    monkeypatch.setattr(screening, "_fetch_etf_history", lambda **kwargs: history)

    row = screening._build_sector_etf_context_row(
        stock_module=object(),
        sector_code="5042",
        sector_name="KRX 산업재",
        mapped_etfs=[{"code": "227550", "name": "TIGER 200 산업재"}],
        snapshot_df=snapshot,
        snapshot_date="20260428",
        target_date="20260428",
        lookback_days=20,
        min_avg_trading_value=300_000_000,
    )

    assert row["execution_state"] == "실행 유동성 부족"
    assert "20일 평균 거래대금 기준 미달" in row["note"]


def test_build_sector_etf_context_row_handles_no_mapped_etf():
    row = screening._build_sector_etf_context_row(
        stock_module=object(),
        sector_code="1170",
        sector_name="KOSPI200 유틸리티",
        mapped_etfs=[],
        snapshot_df=pd.DataFrame(),
        snapshot_date="",
        target_date="20260428",
        lookback_days=20,
        min_avg_trading_value=300_000_000,
    )

    assert row["execution_state"] == "대표 ETF 없음"
    assert row["etf_name"] == "—"


def test_normalize_etf_snapshot_tolerates_missing_listed_shares():
    raw = pd.DataFrame(
        {
            "ISU_SRT_CD": ["396500"],
            "ISU_ABBRV": ["TIGER 반도체TOP10"],
            "NAV": ["10,200.00"],
            "ACC_TRDVAL": ["1,200,000,000"],
            "MKTCAP": ["220,000,000,000"],
            "INVSTASST_NETASST_TOTAMT": ["210,000,000,000"],
        }
    )

    normalized = screening._normalize_etf_snapshot(raw)

    assert list(normalized.index) == ["396500"]
    assert float(normalized.loc["396500", "nav"]) == 10200.0
    assert pd.isna(normalized.loc["396500", "listed_shares"])


def test_load_stock_ohlcv_uses_cached_warehouse_rows(monkeypatch):
    index = pd.bdate_range("2025-05-01", periods=230)
    cached = pd.DataFrame(
        {
            "ticker": ["005930"] * len(index),
            "ticker_name": ["삼성전자"] * len(index),
            "open": [100.0 + i for i in range(len(index))],
            "high": [101.0 + i for i in range(len(index))],
            "low": [99.0 + i for i in range(len(index))],
            "close": [100.0 + i for i in range(len(index))],
            "volume": [1_000_000] * len(index),
        },
        index=index,
    )
    screening_start = index.min().strftime("%Y%m%d")
    screening_end = index.max().strftime("%Y%m%d")

    from src.data_sources import warehouse

    warehouse.upsert_stock_ohlcv(cached, provider="PYKRX", market="KR")

    class _NoLiveStockModule:
        def get_market_ohlcv_by_date(self, *args, **kwargs):
            raise AssertionError("live stock OHLCV should not run on usable cache")

        def get_market_ticker_name(self, *args, **kwargs):
            raise AssertionError("live ticker name should not run on usable cache")

    loaded = screening._load_stock_ohlcv_cached_or_live(
        _NoLiveStockModule(),
        ticker="005930",
        start=screening_start,
        end=screening_end,
    )

    assert len(loaded) == len(cached)
    assert loaded["close"].iloc[-1] == cached["close"].iloc[-1]


def test_score_stock_computes_200dma_from_one_year_cached_history():
    index = pd.bdate_range("2025-05-01", periods=230)
    cached = pd.DataFrame(
        {
            "ticker": ["005930"] * len(index),
            "ticker_name": ["삼성전자"] * len(index),
            "open": [100.0 + i for i in range(len(index))],
            "high": [101.0 + i for i in range(len(index))],
            "low": [99.0 + i for i in range(len(index))],
            "close": [100.0 + i for i in range(len(index))],
            "volume": [1_000_000] * len(index),
        },
        index=index,
    )
    from src.data_sources import warehouse

    warehouse.upsert_stock_ohlcv(cached, provider="PYKRX", market="KR")
    bench_close = pd.Series([90.0 + i * 0.1 for i in range(len(index))], index=index)

    class _NoLiveStockModule:
        def get_market_ohlcv_by_date(self, *args, **kwargs):
            raise AssertionError("live stock OHLCV should not run on usable cache")

        def get_market_ticker_name(self, *args, **kwargs):
            raise AssertionError("live ticker name should not run on usable cache")

    row = screening._score_stock(
        stock=_NoLiveStockModule(),
        ticker="005930",
        sector_code="5044",
        sector_name="KRX 반도체",
        start_date=index.min().strftime("%Y%m%d"),
        trade_date=index.max().strftime("%Y%m%d"),
        bench_close=bench_close,
        rs_ma_period=20,
        rsi_period=14,
        ma_fast=20,
        ma_slow=60,
    )

    assert row is not None
    assert row["above_200dma"] is True
    assert row["trend_ok"] is True
    assert row["name"] == "삼성전자"
