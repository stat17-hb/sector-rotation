from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import src.data_sources.theme_lens as theme_lens
import src.data_sources.warehouse as warehouse


def _write_config(path: Path, *, duplicate: bool = False) -> None:
    duplicate_block = """
  - theme_id: "power_infra"
    name: "Duplicate"
    price_source: "ETF_OHLCV"
    representative_etfs:
      - code: "999999"
        name: "Duplicate ETF"
""" if duplicate else ""
    path.write_text(
        """
themes:
  - theme_id: "power_infra"
    name: "전력/AI전력인프라"
    price_source: "ETF_OHLCV"
    proxy_note: "proxy"
    classification_basis:
      - provider: "iSelect"
        label: "AI전력"
    representative_etfs:
      - code: "487240"
        name: "KODEX AI전력핵심설비"
  - theme_id: "shipbuilding"
    name: "조선"
    price_source: "ETF_OHLCV"
    representative_etfs:
      - code: "0141S0"
        name: "SOL 조선기자재"
"""
        + duplicate_block,
        encoding="utf-8",
    )


def test_load_theme_lens_config_preserves_string_etf_codes(tmp_path):
    config_path = tmp_path / "theme_lens.yml"
    _write_config(config_path)

    definitions = theme_lens.load_theme_lens_config(config_path)

    assert [definition.theme_id for definition in definitions] == ["power_infra", "shipbuilding"]
    assert definitions[1].representative_etfs[0]["code"] == "0141S0"
    assert isinstance(definitions[1].representative_etfs[0]["code"], str)


def test_load_theme_lens_config_rejects_duplicate_theme_id(tmp_path):
    config_path = tmp_path / "theme_lens.yml"
    _write_config(config_path, duplicate=True)

    with pytest.raises(ValueError, match="duplicate theme_id"):
        theme_lens.load_theme_lens_config(config_path)


def test_build_theme_proxy_returns_computes_period_returns():
    dates = pd.date_range("2026-01-01", periods=70, freq="B")
    frame = pd.DataFrame({"close": range(100, 170)}, index=dates)

    returns = theme_lens.build_theme_proxy_returns(frame)

    assert returns["return_1d"] == pytest.approx(169 / 168 - 1)
    assert returns["return_1m"] == pytest.approx(169 / 148 - 1)
    assert returns["return_3m"] == pytest.approx(169 / 106 - 1)


def test_load_theme_lens_cache_only_does_not_import_pykrx(monkeypatch, tmp_path):
    config_path = tmp_path / "theme_lens.yml"
    _write_config(config_path)
    sys.modules.pop("pykrx", None)

    cached = pd.DataFrame(
        {
            "ticker": ["487240", "487240", "0141S0", "0141S0"],
            "ticker_name": ["KODEX AI전력핵심설비", "KODEX AI전력핵심설비", "SOL 조선기자재", "SOL 조선기자재"],
            "close": [100.0, 110.0, 200.0, 220.0],
        },
        index=pd.to_datetime(["2026-05-01", "2026-05-15", "2026-05-01", "2026-05-15"]),
    )
    monkeypatch.setattr(theme_lens, "read_stock_ohlcv", lambda *args, **kwargs: cached)

    status, rows = theme_lens.load_theme_lens_cache_only(
        asof_date="20260516",
        config_path=config_path,
    )

    assert status == "CACHED"
    assert [row["theme_name"] for row in rows] == ["전력/AI전력인프라", "조선"]
    assert "pykrx" not in sys.modules


def test_load_theme_lens_cache_only_uses_first_available_representative_etf(monkeypatch, tmp_path):
    config_path = tmp_path / "theme_lens.yml"
    config_path.write_text(
        """
themes:
  - theme_id: "shipbuilding"
    name: "조선"
    representative_etfs:
      - code: "441540"
        name: "HANARO Fn조선해운"
      - code: "0141S0"
        name: "SOL 조선기자재"
""",
        encoding="utf-8",
    )
    cached = pd.DataFrame(
        {
            "ticker": ["0141S0", "0141S0"],
            "ticker_name": ["SOL 조선기자재", "SOL 조선기자재"],
            "close": [200.0, 220.0],
        },
        index=pd.to_datetime(["2026-05-01", "2026-05-15"]),
    )
    monkeypatch.setattr(theme_lens, "read_stock_ohlcv", lambda *args, **kwargs: cached)

    status, rows = theme_lens.load_theme_lens_cache_only(
        asof_date="20260516",
        config_path=config_path,
    )

    assert status == "CACHED"
    assert rows[0]["primary_proxy_code"] == "0141S0"
    assert rows[0]["primary_proxy_name"] == "SOL 조선기자재"


def test_load_theme_proxy_signal_inputs_builds_cache_only_signal_rows(monkeypatch, tmp_path):
    config_path = tmp_path / "theme_lens.yml"
    config_path.write_text(
        """
themes:
  - theme_id: "robotics"
    name: "로봇"
    representative_etfs:
      - code: "445290"
        name: "KODEX K-로봇액티브"
""",
        encoding="utf-8",
    )
    sys.modules.pop("pykrx", None)
    dates = pd.date_range("2025-03-21", periods=300, freq="B")
    cached = pd.DataFrame(
        {
            "ticker": ["445290"] * len(dates),
            "ticker_name": ["KODEX K-로봇액티브"] * len(dates),
            "close": [100.0 + idx for idx in range(len(dates))],
        },
        index=dates,
    )
    monkeypatch.setattr(theme_lens, "read_stock_ohlcv", lambda *args, **kwargs: cached)

    status, prices, universe_rows = theme_lens.load_theme_proxy_signal_inputs(
        asof_date=dates[-1].strftime("%Y%m%d"),
        config_path=config_path,
    )

    assert status == "CACHED"
    assert "pykrx" not in sys.modules
    assert prices["index_code"].unique().tolist() == ["445290"]
    assert prices["index_name"].unique().tolist() == ["로봇"]
    assert universe_rows == [
        {
            "index_code": "445290",
            "index_name": "로봇",
            "family": "theme_lens_etf_proxy",
            "is_benchmark": False,
            "is_active": True,
            "export_sector": False,
            "taxonomy_kind": "THEME",
            "taxonomy_label": "로봇",
            "theme_id": "robotics",
            "primary_proxy_code": "445290",
            "primary_proxy_name": "KODEX K-로봇액티브",
            "reference_only": True,
        }
    ]


def test_load_theme_proxy_signal_inputs_excludes_stale_cache(monkeypatch, tmp_path):
    config_path = tmp_path / "theme_lens.yml"
    _write_config(config_path)
    cached = pd.DataFrame(
        {
            "ticker": ["487240"],
            "ticker_name": ["KODEX AI전력핵심설비"],
            "close": [100.0],
        },
        index=pd.to_datetime(["2026-04-01"]),
    )
    monkeypatch.setattr(theme_lens, "read_stock_ohlcv", lambda *args, **kwargs: cached)

    status, prices, universe_rows = theme_lens.load_theme_proxy_signal_inputs(
        asof_date="20260516",
        config_path=config_path,
    )

    assert status == "UNAVAILABLE"
    assert prices.empty
    assert universe_rows == []


def test_refresh_theme_lens_etf_ohlcv_calls_pykrx_and_upserts(monkeypatch, tmp_path):
    config_path = tmp_path / "theme_lens.yml"
    _write_config(config_path)
    upserts: list[pd.DataFrame] = []

    class FakeStock:
        @staticmethod
        def get_market_ohlcv_by_date(start, end, ticker):
            assert start <= end
            return pd.DataFrame(
                {"시가": [100], "고가": [111], "저가": [99], "종가": [110], "거래량": [1000]},
                index=pd.to_datetime(["2026-05-15"]),
            )

    monkeypatch.setitem(sys.modules, "pykrx", SimpleNamespace(stock=FakeStock))
    import src.data_sources.pykrx_compat as compat

    monkeypatch.setattr(compat, "ensure_pykrx_transport_compat", lambda: None)
    monkeypatch.setattr(theme_lens, "upsert_stock_ohlcv", lambda frame, **kwargs: upserts.append(frame.copy()))

    summary = theme_lens.refresh_theme_lens_etf_ohlcv(
        asof_date="20260516",
        config_path=config_path,
    )

    assert summary["status"] == "LIVE"
    assert summary["live_status"] == "LIVE"
    assert summary["fetched_codes"] == ["487240", "0141S0"]
    assert summary["refreshed_codes"] == ["487240", "0141S0"]
    assert len(upserts) == 2
    assert upserts[1]["ticker"].iloc[0] == "0141S0"
    assert [row["status"] for row in summary["live_rows"]] == ["LIVE", "LIVE"]


def test_refresh_theme_lens_returns_live_rows_when_warehouse_upsert_is_locked(monkeypatch, tmp_path):
    config_path = tmp_path / "theme_lens.yml"
    _write_config(config_path)

    class FakeStock:
        @staticmethod
        def get_market_ohlcv_by_date(start, end, ticker):
            return pd.DataFrame(
                {"시가": [100], "고가": [111], "저가": [99], "종가": [110], "거래량": [1000]},
                index=pd.to_datetime(["2026-05-15"]),
            )

    monkeypatch.setitem(sys.modules, "pykrx", SimpleNamespace(stock=FakeStock))
    import src.data_sources.pykrx_compat as compat

    monkeypatch.setattr(compat, "ensure_pykrx_transport_compat", lambda: None)

    def _locked_upsert(*_args, **_kwargs):
        raise RuntimeError("Cannot acquire write lock on warehouse.duckdb")

    monkeypatch.setattr(theme_lens, "upsert_stock_ohlcv", _locked_upsert)

    summary = theme_lens.refresh_theme_lens_etf_ohlcv(
        asof_date="20260516",
        config_path=config_path,
    )

    assert summary["status"] == "PARTIAL"
    assert summary["live_status"] == "LIVE"
    assert summary["fetched_codes"] == ["487240", "0141S0"]
    assert summary["refreshed_codes"] == []
    assert summary["row_count"] == 2
    assert [row["status"] for row in summary["live_rows"]] == ["LIVE", "LIVE"]


def test_warehouse_stock_ohlcv_supports_alphanumeric_codes(monkeypatch, tmp_path):
    monkeypatch.setattr(warehouse, "WAREHOUSE_PATH", tmp_path / "warehouse.duckdb")
    warehouse.close_cached_read_only_connection()

    frame = pd.DataFrame(
        {"ticker": ["0141S0"], "ticker_name": ["SOL 조선기자재"], "close": [123.0]},
        index=pd.to_datetime(["2026-05-15"]),
    )
    warehouse.upsert_stock_ohlcv(frame, provider="PYKRX", market="KR")
    loaded = warehouse.read_stock_ohlcv(["0141S0"], "20260501", "20260516", market="KR")

    assert loaded["ticker"].tolist() == ["0141S0"]
    assert loaded["close"].tolist() == [123.0]
