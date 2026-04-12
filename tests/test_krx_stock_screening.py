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
