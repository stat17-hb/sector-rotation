"""Targeted regressions for KRX pykrx compatibility wiring."""
from __future__ import annotations

from datetime import date
import sys
import types

import pandas as pd
import pytest


def test_fetch_chunk_calls_transport_compat_before_pykrx(monkeypatch):
    import src.data_sources.krx_indices as krx_mod

    state = {"compat_called": False}

    def fake_ensure():
        state["compat_called"] = True

    monkeypatch.setattr(krx_mod, "ensure_pykrx_transport_compat", fake_ensure)

    fake_pykrx = types.ModuleType("pykrx")

    class _Stock:
        @staticmethod
        def get_index_ohlcv(start, end, index_code):
            assert state["compat_called"] is True
            idx = pd.date_range("2024-01-01", periods=2, freq="B")
            return pd.DataFrame({"\uc885\uac00": [1000.0, 1001.0]}, index=idx)

    fake_pykrx.stock = _Stock
    monkeypatch.setitem(sys.modules, "pykrx", fake_pykrx)

    df = krx_mod._fetch_chunk("1001", "20240101", "20240131")
    assert not df.empty
    assert df["\uc885\uac00"].iloc[-1] == 1001.0


def test_load_sector_prices_uses_positional_close_fallback(tmp_path, monkeypatch):
    import src.data_sources.krx_indices as krx_mod

    monkeypatch.setattr(krx_mod, "CURATED_DIR", tmp_path / "curated")
    monkeypatch.setattr(krx_mod, "RAW_DIR", tmp_path / "raw")

    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    ohlcv = pd.DataFrame(
        {
            "col0": [1.0, 1.0, 1.0],
            "col1": [2.0, 2.0, 2.0],
            "col2": [0.0, 0.0, 0.0],
            "close_pos": [10.0, 11.0, 12.0],
            "col4": [99.0, 99.0, 99.0],
        },
        index=idx,
    )
    monkeypatch.setattr(krx_mod, "fetch_index_ohlcv", lambda *args, **kwargs: ohlcv)

    status, result = krx_mod.load_sector_prices(["5044"], "20240101", "20240131")
    assert status == "LIVE"
    assert result["close"].tolist() == [10.0, 11.0, 12.0]
    assert set(result["index_code"].unique()) == {"5044"}


def test_load_sector_prices_replaces_stale_codes_before_fetch(tmp_path, monkeypatch):
    import src.data_sources.krx_indices as krx_mod

    monkeypatch.setattr(krx_mod, "CURATED_DIR", tmp_path / "curated")
    monkeypatch.setattr(krx_mod, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(
        krx_mod,
        "_get_index_universe",
        lambda: frozenset({"5049", "1157"}),
    )

    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    ohlcv = pd.DataFrame({"close": [101.0, 102.0]}, index=idx)
    calls: list[str] = []

    def _fake_fetch(index_code, start, end, chunk_years=2):
        calls.append(index_code)
        return ohlcv

    monkeypatch.setattr(krx_mod, "fetch_index_ohlcv", _fake_fetch)

    status, result = krx_mod.load_sector_prices(["5041", "1166"], "20240101", "20240131")
    assert status == "LIVE"
    assert calls == ["5049", "1157"]
    assert set(result["index_code"].unique()) == {"5049", "1157"}


def test_load_sector_prices_skips_unsupported_codes_before_fetch(tmp_path, monkeypatch):
    import src.data_sources.krx_indices as krx_mod

    monkeypatch.setattr(krx_mod, "CURATED_DIR", tmp_path / "curated")
    monkeypatch.setattr(krx_mod, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(
        krx_mod,
        "_get_index_universe",
        lambda: frozenset({"5049"}),
    )

    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    ohlcv = pd.DataFrame({"close": [201.0, 202.0]}, index=idx)
    calls: list[str] = []

    def _fake_fetch(index_code, start, end, chunk_years=2):
        calls.append(index_code)
        return ohlcv

    monkeypatch.setattr(krx_mod, "fetch_index_ohlcv", _fake_fetch)

    status, result = krx_mod.load_sector_prices(["5041", "9999"], "20240101", "20240131")
    assert status == "LIVE"
    assert calls == ["5049"]
    assert set(result["index_code"].unique()) == {"5049"}


def test_fetch_chunk_stops_retry_for_missing_code_keyerror(monkeypatch):
    import src.data_sources.krx_indices as krx_mod

    monkeypatch.setattr(krx_mod, "ensure_pykrx_transport_compat", lambda: None)

    state = {"calls": 0}
    fake_pykrx = types.ModuleType("pykrx")

    class _Stock:
        @staticmethod
        def get_index_ohlcv(start, end, index_code):
            state["calls"] += 1
            raise KeyError(index_code)

    fake_pykrx.stock = _Stock
    monkeypatch.setitem(sys.modules, "pykrx", fake_pykrx)

    with pytest.raises(KeyError, match="5041"):
        krx_mod._fetch_chunk("5041", "20240101", "20240131")

    assert state["calls"] == 1


def test_calendar_lookup_applies_transport_compat(monkeypatch):
    import src.transforms.calendar as calendar_mod

    state = {"compat_called": False}

    def fake_ensure():
        state["compat_called"] = True

    monkeypatch.setattr(calendar_mod, "ensure_pykrx_transport_compat", fake_ensure)

    fake_pykrx = types.ModuleType("pykrx")

    class _Stock:
        @staticmethod
        def get_index_ohlcv(start, end, ticker):
            assert state["compat_called"] is True
            idx = pd.DatetimeIndex(["2026-02-19", "2026-02-20"])
            return pd.DataFrame({"\uc885\uac00": [2990.0, 3010.0]}, index=idx)

    fake_pykrx.stock = _Stock
    monkeypatch.setitem(sys.modules, "pykrx", fake_pykrx)

    result = calendar_mod.get_last_business_day(as_of=date(2026, 2, 22))
    assert result == date(2026, 2, 20)


def test_calendar_weekend_fallback_unchanged_when_pykrx_fails(monkeypatch):
    import src.transforms.calendar as calendar_mod

    monkeypatch.setattr(calendar_mod, "ensure_pykrx_transport_compat", lambda: None)

    fake_pykrx = types.ModuleType("pykrx")

    class _Stock:
        @staticmethod
        def get_index_ohlcv(start, end, ticker):
            raise KeyError("\uc9c0\uc218\uba85")

    fake_pykrx.stock = _Stock
    monkeypatch.setitem(sys.modules, "pykrx", fake_pykrx)

    result = calendar_mod.get_last_business_day(as_of=date(2026, 2, 22))
    assert result == date(2026, 2, 20)
