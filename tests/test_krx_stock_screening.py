from __future__ import annotations

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
