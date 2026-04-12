from __future__ import annotations

import socket
import pandas as pd

from src.data_sources.krx_constituents import ConstituentLookupResult
import src.data_sources.krx_investor_flow as flow_source
import src.data_sources.warehouse as warehouse


_SECTOR_MAP = {
    "regimes": {
        "Recovery": {
            "sectors": [
                {"code": "5044", "name": "KRX 반도체"},
                {"code": "5045", "name": "KRX 헬스케어"},
            ]
        }
    }
}


def _single_day_raw_frame(day: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime([day]),
            "ticker": ["005930"],
            "ticker_name": ["삼성전자"],
            "investor_type": ["외국인"],
            "buy_amount": [100],
            "sell_amount": [50],
            "net_buy_amount": [50],
        }
    )


def _single_day_sector_frame(day: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime([day]),
            "sector_code": ["5044"],
            "sector_name": ["KRX 반도체"],
            "investor_type": ["외국인"],
            "buy_amount": [100],
            "sell_amount": [50],
            "net_buy_amount": [50],
            "net_flow_ratio": [0.2],
        }
    )


def test_collect_sector_investor_flow_aggregates_current_constituents(monkeypatch):
    monkeypatch.setattr(flow_source, "ensure_pykrx_transport_compat", lambda: None)
    monkeypatch.setattr(flow_source, "_check_socket_stack", lambda: None)
    monkeypatch.setattr(flow_source, "upsert_sector_constituents_snapshot", lambda *a, **kw: None)

    import pykrx.stock as stock

    constituent_calls: list[tuple[str, str, bool]] = []

    monkeypatch.setattr(
        stock,
        "get_index_portfolio_deposit_file",
        lambda ticker, date=None, alternative=False: (
            constituent_calls.append((ticker, date, alternative))
            or (["005930"] if ticker == "5044" else ["000660"])
        ),
    )

    monkeypatch.setattr(stock, "get_market_ticker_name", lambda ticker: f"Name {ticker}")

    def _trading_value_by_date(start, end, ticker, on="순매수", **kwargs):
        dates = pd.to_datetime(["2026-04-07", "2026-04-08"])
        ratios = {
            "005930": {"개인": -0.08, "외국인합계": 0.20, "기관합계": 0.10},
            "000660": {"개인": 0.02, "외국인합계": -0.10, "기관합계": 0.05},
        }[ticker]
        multiplier = {"매수": 1000, "매도": 900, "순매수": 100}[on]
        frame = pd.DataFrame(index=dates)
        for investor, ratio in ratios.items():
            if on == "매수":
                frame[investor] = [1000 + int(ratio * multiplier), 1100 + int(ratio * multiplier)]
            elif on == "매도":
                frame[investor] = [900 - int(ratio * multiplier), 950 - int(ratio * multiplier)]
            else:
                frame[investor] = [int(ratio * multiplier), int(ratio * multiplier * 1.2)]
        frame["전체"] = 0
        frame.index.name = "날짜"
        return frame

    monkeypatch.setattr(stock, "get_market_trading_value_by_date", _trading_value_by_date)

    raw_frame, sector_frame, summary = flow_source.collect_sector_investor_flow(
        sector_map=_SECTOR_MAP,
        start="20260407",
        end="20260408",
    )

    assert not raw_frame.empty
    assert not sector_frame.empty
    assert set(sector_frame["sector_code"]) == {"5044", "5045"}
    assert "net_flow_ratio" in sector_frame.columns
    assert summary["tracked_sectors"] == 2
    assert set(raw_frame["investor_type"]) == {"개인", "외국인", "기관합계"}
    assert constituent_calls
    assert constituent_calls[0] == ("5044", "20260408", False)


def test_collect_sector_investor_flow_uses_trading_day_calendar_for_failed_days(monkeypatch):
    monkeypatch.setattr(flow_source, "ensure_pykrx_transport_compat", lambda: None)
    monkeypatch.setattr(flow_source, "_check_socket_stack", lambda: None)
    monkeypatch.setattr(flow_source, "upsert_sector_constituents_snapshot", lambda *a, **kw: None)
    monkeypatch.setattr(flow_source, "_requested_trading_days", lambda *args, **kwargs: ["20260407", "20260408"])

    import pykrx.stock as stock

    monkeypatch.setattr(
        stock,
        "get_index_portfolio_deposit_file",
        lambda ticker, date=None, alternative=False: ["005930"],
    )
    monkeypatch.setattr(stock, "get_market_ticker_name", lambda ticker: f"Name {ticker}")

    dates = pd.to_datetime(["2026-04-07", "2026-04-08"])

    def _trading_value_by_date(start, end, ticker, on="순매수", **kwargs):
        frame = pd.DataFrame(
            {
                "개인": [100, 110],
                "외국인합계": [200, 210],
                "기관합계": [50, 55],
                "전체": [0, 0],
            },
            index=dates,
        )
        frame.index.name = "날짜"
        return frame

    monkeypatch.setattr(stock, "get_market_trading_value_by_date", _trading_value_by_date)

    _, _, summary = flow_source.collect_sector_investor_flow(
        sector_map={"regimes": {"Recovery": {"sectors": [{"code": "5044", "name": "KRX 반도체"}]}}},
        start="20260407",
        end="20260409",
    )

    assert summary["failed_days"] == []
    assert summary["coverage_complete"] is True


def test_run_manual_investor_flow_refresh_persists_to_warehouse(monkeypatch):
    raw_frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-04-08"] * 2),
            "ticker": ["005930", "005930"],
            "ticker_name": ["삼성전자", "삼성전자"],
            "investor_type": ["외국인", "개인"],
            "buy_amount": [1200, 900],
            "sell_amount": [800, 1100],
            "net_buy_amount": [400, -200],
        }
    )
    sector_frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-04-08"] * 2),
            "sector_code": ["5044", "5044"],
            "sector_name": ["KRX 반도체", "KRX 반도체"],
            "investor_type": ["외국인", "개인"],
            "buy_amount": [1200, 900],
            "sell_amount": [800, 1100],
            "net_buy_amount": [400, -200],
            "net_flow_ratio": [0.2, -0.1],
        }
    )
    monkeypatch.setattr(
        flow_source,
        "collect_sector_investor_flow",
        lambda **kwargs: (
            raw_frame,
            sector_frame,
            {
                "status": "LIVE",
                "provider": flow_source.FLOW_PROVIDER,
                "requested_start": "20260101",
                "requested_end": "20260408",
                "coverage_complete": True,
                "failed_days": [],
                "failed_codes": {},
                "predicted_requests": 10,
                "processed_requests": 10,
                "rows": 2,
            },
        ),
    )

    (status, cached), summary = flow_source.run_manual_investor_flow_refresh(
        sector_map=_SECTOR_MAP,
        end_date_str="20260408",
    )

    assert status == "LIVE"
    assert summary["coverage_complete"] is True
    assert not cached.empty
    cached_status, cached_frame = flow_source.load_sector_investor_flow(
        sector_map=_SECTOR_MAP,
        start="20260101",
        end="20260408",
    )
    assert cached_status == "CACHED"
    assert not cached_frame.empty


def test_run_manual_investor_flow_refresh_preserves_live_rows_on_warehouse_write_lock(monkeypatch):
    raw_frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-04-08"] * 2),
            "ticker": ["005930", "005930"],
            "ticker_name": ["삼성전자", "삼성전자"],
            "investor_type": ["외국인", "개인"],
            "buy_amount": [1200, 900],
            "sell_amount": [800, 1100],
            "net_buy_amount": [400, -200],
        }
    )
    sector_frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-04-08"] * 2),
            "sector_code": ["5044", "5044"],
            "sector_name": ["KRX 반도체", "KRX 반도체"],
            "investor_type": ["외국인", "개인"],
            "buy_amount": [1200, 900],
            "sell_amount": [800, 1100],
            "net_buy_amount": [400, -200],
            "net_flow_ratio": [0.2, -0.1],
        }
    )
    monkeypatch.setattr(
        flow_source,
        "collect_sector_investor_flow",
        lambda **kwargs: (
            raw_frame,
            sector_frame,
            {
                "status": "LIVE",
                "provider": flow_source.FLOW_PROVIDER,
                "requested_start": "20260101",
                "requested_end": "20260408",
                "coverage_complete": True,
                "failed_days": [],
                "failed_codes": {},
                "predicted_requests": 10,
                "processed_requests": 10,
                "rows": 2,
            },
        ),
    )
    monkeypatch.setattr(
        flow_source,
        "write_investor_flow_operational_result",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "Cannot acquire write lock on warehouse.duckdb. Underlying error: Connection Error: "
                "Can't open a connection to same database file with a different configuration than existing connections"
            )
        ),
    )

    (status, result_frame), summary = flow_source.run_manual_investor_flow_refresh(
        sector_map=_SECTOR_MAP,
        end_date_str="20260408",
    )

    assert status == "LIVE"
    assert result_frame.equals(sector_frame)
    assert summary["warehouse_write_skipped"] is True
    assert "warehouse_write_error" in summary
    assert summary["rows"] == 2
    assert summary["coverage_complete"] is True
    assert summary["failed_codes"]["warehouse"].startswith("Cannot acquire write lock")


def test_load_sector_investor_flow_returns_bootstrap_partial_preview_without_complete_cursor():
    partial_summary = {
        "status": "LIVE",
        "provider": flow_source.FLOW_PROVIDER,
        "requested_start": "20260408",
        "requested_end": "20260408",
        "coverage_complete": False,
        "failed_days": ["20260408"],
        "failed_codes": {"20260408": "partial"},
        "predicted_requests": 1,
        "processed_requests": 1,
        "rows": 1,
    }
    warehouse.write_investor_flow_operational_result(
        raw_frame=_single_day_raw_frame("2026-04-08"),
        sector_frame=_single_day_sector_frame("2026-04-08"),
        provider=flow_source.FLOW_PROVIDER,
        requested_start="20260408",
        requested_end="20260408",
        reason="manual_refresh",
        summary=partial_summary,
        market="KR",
    )

    default_status, default_frame = flow_source.load_sector_investor_flow(
        sector_map=_SECTOR_MAP,
        start="20260408",
        end="20260408",
        market="KR",
    )
    preview_status, preview_frame = flow_source.load_sector_investor_flow(
        sector_map=_SECTOR_MAP,
        start="20260408",
        end="20260408",
        market="KR",
        allow_bootstrap_partial_preview=True,
    )

    assert default_status == "SAMPLE"
    assert default_frame.empty
    assert preview_status == "CACHED"
    assert not preview_frame.empty


def test_collect_sector_investor_flow_marks_partial_ticker_frames_as_failure(monkeypatch):
    monkeypatch.setattr(flow_source, "ensure_pykrx_transport_compat", lambda: None)
    monkeypatch.setattr(flow_source, "_check_socket_stack", lambda: None)
    monkeypatch.setattr(flow_source, "upsert_sector_constituents_snapshot", lambda *a, **kw: None)
    monkeypatch.setattr(flow_source, "_fetch_ticker_trading_value_frame_raw", lambda **kwargs: pd.DataFrame())

    import pykrx.stock as stock

    monkeypatch.setattr(
        stock,
        "get_index_portfolio_deposit_file",
        lambda ticker, date=None, alternative=False: ["005930"],
    )
    monkeypatch.setattr(stock, "get_market_ticker_name", lambda ticker: f"Name {ticker}")

    def _trading_value_by_date(start, end, ticker, on="순매수", **kwargs):
        dates = pd.to_datetime(["2026-04-07", "2026-04-08"])
        if on == "매도":
            return pd.DataFrame()
        frame = pd.DataFrame(
            {
                "개인": [100, 110],
                "외국인합계": [200, 210],
                "기관합계": [50, 55],
                "전체": [0, 0],
            },
            index=dates,
        )
        frame.index.name = "날짜"
        return frame

    monkeypatch.setattr(stock, "get_market_trading_value_by_date", _trading_value_by_date)

    raw_frame, sector_frame, summary = flow_source.collect_sector_investor_flow(
        sector_map={"regimes": {"Recovery": {"sectors": [{"code": "5044", "name": "KRX 반도체"}]}}},
        start="20260407",
        end="20260408",
    )

    assert not raw_frame.empty
    assert not sector_frame.empty
    assert summary["coverage_complete"] is True
    assert summary["failed_codes"] == {}


def test_collect_sector_investor_flow_falls_back_to_detail_frames_when_general_frames_are_empty(monkeypatch):
    monkeypatch.setattr(flow_source, "ensure_pykrx_transport_compat", lambda: None)
    monkeypatch.setattr(flow_source, "_check_socket_stack", lambda: None)
    monkeypatch.setattr(flow_source, "upsert_sector_constituents_snapshot", lambda *a, **kw: None)

    import pykrx.stock as stock

    monkeypatch.setattr(
        stock,
        "get_index_portfolio_deposit_file",
        lambda ticker, date=None, alternative=False: ["005930"],
    )
    monkeypatch.setattr(stock, "get_market_ticker_name", lambda ticker: f"Name {ticker}")

    dates = pd.to_datetime(["2026-04-07", "2026-04-08"])

    def _trading_value_by_date(start, end, ticker, on="순매수", detail=False, **kwargs):
        if not detail:
            return pd.DataFrame()
        base = pd.DataFrame(
            {
                "금융투자": [10, 20],
                "보험": [5, 5],
                "투신": [15, 10],
                "사모": [1, 1],
                "은행": [0, 0],
                "기타금융": [0, 0],
                "연기금": [9, 9],
                "기타법인": [3, 4],
                "개인": [100, 120],
                "외국인": [50, 60],
                "기타외국인": [7, 8],
                "전체": [0, 0],
            },
            index=dates,
        )
        base.index.name = "날짜"
        if on == "매수":
            return base
        if on == "매도":
            return base * 0 + pd.DataFrame(
                {
                    "금융투자": [6, 7],
                    "보험": [2, 2],
                    "투신": [8, 8],
                    "사모": [1, 1],
                    "은행": [0, 0],
                    "기타금융": [0, 0],
                    "연기금": [4, 4],
                    "기타법인": [2, 2],
                    "개인": [70, 90],
                    "외국인": [20, 30],
                    "기타외국인": [3, 4],
                    "전체": [0, 0],
                },
                index=dates,
            )
        return base * 0 + pd.DataFrame(
            {
                "금융투자": [4, 13],
                "보험": [3, 3],
                "투신": [7, 2],
                "사모": [0, 0],
                "은행": [0, 0],
                "기타금융": [0, 0],
                "연기금": [5, 5],
                "기타법인": [1, 2],
                "개인": [30, 30],
                "외국인": [30, 30],
                "기타외국인": [4, 4],
                "전체": [0, 0],
            },
            index=dates,
        )

    monkeypatch.setattr(stock, "get_market_trading_value_by_date", _trading_value_by_date)

    raw_frame, sector_frame, summary = flow_source.collect_sector_investor_flow(
        sector_map={"regimes": {"Recovery": {"sectors": [{"code": "5044", "name": "KRX 반도체"}]}}},
        start="20260407",
        end="20260408",
    )

    assert not raw_frame.empty
    assert not sector_frame.empty
    assert summary["failed_codes"] == {}
    latest = raw_frame.sort_values(["trade_date", "investor_type"]).reset_index(drop=True)
    assert set(latest["investor_type"]) == {"개인", "외국인", "기관합계"}
    assert int(latest.loc[latest["investor_type"] == "기관합계", "buy_amount"].iloc[0]) == 40
    assert int(latest.loc[latest["investor_type"] == "외국인", "sell_amount"].iloc[0]) == 23


def test_collect_sector_investor_flow_recovers_missing_buy_frame_from_sell_and_net(monkeypatch):
    monkeypatch.setattr(flow_source, "ensure_pykrx_transport_compat", lambda: None)
    monkeypatch.setattr(flow_source, "_check_socket_stack", lambda: None)
    monkeypatch.setattr(flow_source, "upsert_sector_constituents_snapshot", lambda *a, **kw: None)
    monkeypatch.setattr(flow_source, "_fetch_ticker_trading_value_frame_raw", lambda **kwargs: pd.DataFrame())

    import pykrx.stock as stock

    monkeypatch.setattr(
        stock,
        "get_index_portfolio_deposit_file",
        lambda ticker, date=None, alternative=False: ["005930"],
    )
    monkeypatch.setattr(stock, "get_market_ticker_name", lambda ticker: "삼성전자")

    dates = pd.to_datetime(["2026-04-09", "2026-04-10"])

    def _trading_value_by_date(start, end, ticker, on="순매수", detail=False, **kwargs):
        if detail:
            return pd.DataFrame()
        if on == "매수":
            return pd.DataFrame()
        if on == "매도":
            frame = pd.DataFrame(
                {"개인": [70, 90], "외국인합계": [20, 30], "기관합계": [10, 15], "전체": [0, 0]},
                index=dates,
            )
            frame.index.name = "날짜"
            return frame
        frame = pd.DataFrame(
            {"개인": [30, 30], "외국인합계": [30, 30], "기관합계": [10, 12], "전체": [0, 0]},
            index=dates,
        )
        frame.index.name = "날짜"
        return frame

    monkeypatch.setattr(stock, "get_market_trading_value_by_date", _trading_value_by_date)

    raw_frame, sector_frame, summary = flow_source.collect_sector_investor_flow(
        sector_map={"regimes": {"Recovery": {"sectors": [{"code": "5044", "name": "KRX 반도체"}]}}},
        start="20260409",
        end="20260410",
    )

    assert summary["failed_codes"] == {}
    assert not raw_frame.empty
    assert not sector_frame.empty
    foreign_latest = raw_frame[
        (raw_frame["investor_type"] == "외국인")
        & (raw_frame["trade_date"] == pd.Timestamp("2026-04-10"))
    ]
    assert int(foreign_latest["buy_amount"].iloc[0]) == 60


def test_collect_sector_investor_flow_falls_back_to_raw_trading_value_payload(monkeypatch):
    monkeypatch.setattr(flow_source, "ensure_pykrx_transport_compat", lambda: None)
    monkeypatch.setattr(flow_source, "_check_socket_stack", lambda: None)
    monkeypatch.setattr(flow_source, "upsert_sector_constituents_snapshot", lambda *a, **kw: None)

    import pykrx.stock as stock

    monkeypatch.setattr(
        stock,
        "get_index_portfolio_deposit_file",
        lambda ticker, date=None, alternative=False: ["005930"],
    )
    monkeypatch.setattr(stock, "get_market_ticker_name", lambda ticker: "삼성전자")
    monkeypatch.setattr(stock, "get_market_trading_value_by_date", lambda *args, **kwargs: pd.DataFrame())

    def _raw_frame(**kwargs):
        dates = pd.to_datetime(["2026-04-09", "2026-04-10"])
        if kwargs["on"] == "매수":
            frame = pd.DataFrame(
                {"개인": [100, 120], "외국인합계": [50, 60], "기관합계": [40, 38], "전체": [0, 0]},
                index=dates,
            )
        elif kwargs["on"] == "매도":
            frame = pd.DataFrame(
                {"개인": [70, 90], "외국인합계": [20, 30], "기관합계": [10, 15], "전체": [0, 0]},
                index=dates,
            )
        else:
            frame = pd.DataFrame(
                {"개인": [30, 30], "외국인합계": [30, 30], "기관합계": [30, 23], "전체": [0, 0]},
                index=dates,
            )
        frame.index.name = "날짜"
        return frame

    monkeypatch.setattr(flow_source, "_fetch_ticker_trading_value_frame_raw", _raw_frame)

    raw_frame, sector_frame, summary = flow_source.collect_sector_investor_flow(
        sector_map={"regimes": {"Recovery": {"sectors": [{"code": "5044", "name": "KRX 반도체"}]}}},
        start="20260409",
        end="20260410",
    )

    assert not raw_frame.empty
    assert not sector_frame.empty
    assert summary["failed_codes"] == {}


def test_collect_sector_investor_flow_fails_fast_when_socket_stack_is_unavailable(monkeypatch):
    monkeypatch.setattr(flow_source, "ensure_pykrx_transport_compat", lambda: None)

    def _broken_socket(*args, **kwargs):
        raise OSError(10106, "service provider failed")

    monkeypatch.setattr(socket, "socket", _broken_socket)

    try:
        flow_source.collect_sector_investor_flow(
            sector_map=_SECTOR_MAP,
            start="20260407",
            end="20260408",
        )
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Windows socket stack is unavailable" in str(exc)


# ---------------------------------------------------------------------------
# _normalize_constituent_result
# ---------------------------------------------------------------------------

def test_normalize_constituent_result_handles_plain_list():
    result = flow_source._normalize_constituent_result(["005930", "000660"])
    assert result == ["005930", "000660"]


def test_normalize_constituent_result_handles_none_and_empty():
    assert flow_source._normalize_constituent_result(None) == []
    assert flow_source._normalize_constituent_result([]) == []


def test_normalize_constituent_result_handles_dataframe_object_index():
    df = pd.DataFrame({"weight": [0.3, 0.7]}, index=pd.Index(["005930", "000660"], dtype=object))
    result = flow_source._normalize_constituent_result(df)
    assert result == ["005930", "000660"]


def test_normalize_constituent_result_handles_dataframe_int_index():
    """Non-object index → fall back to first column values."""
    df = pd.DataFrame({"ticker": ["005930", "000660"]})  # RangeIndex (int64)
    result = flow_source._normalize_constituent_result(df)
    assert result == ["005930", "000660"]


def test_normalize_constituent_result_handles_empty_dataframe():
    result = flow_source._normalize_constituent_result(pd.DataFrame())
    assert result == []


# ---------------------------------------------------------------------------
# DataFrame return from pykrx (main bug fix: if result: ValueError)
# ---------------------------------------------------------------------------

def test_collect_sector_investor_flow_handles_dataframe_constituent_result(monkeypatch):
    """get_index_portfolio_deposit_file returning a DataFrame must NOT crash."""
    monkeypatch.setattr(flow_source, "ensure_pykrx_transport_compat", lambda: None)
    monkeypatch.setattr(flow_source, "_check_socket_stack", lambda: None)
    monkeypatch.setattr(flow_source, "upsert_sector_constituents_snapshot", lambda *a, **kw: None)

    import pykrx.stock as stock

    # Return a DataFrame with object index (ticker codes as index)
    df_result = pd.DataFrame(
        {"weight": [1.0, 1.0]},
        index=pd.Index(["005930", "000660"], dtype=object),
    )
    monkeypatch.setattr(
        stock,
        "get_index_portfolio_deposit_file",
        lambda ticker, date=None, alternative=False: df_result,
    )
    monkeypatch.setattr(stock, "get_market_ticker_name", lambda ticker: f"Name {ticker}")

    dates = pd.to_datetime(["2026-04-07", "2026-04-08"])

    def _trading_value(start, end, ticker, on="순매수", **kwargs):
        frame = pd.DataFrame(
            {"개인": [100, 110], "외국인합계": [200, 210], "기관합계": [50, 55], "전체": [0, 0]},
            index=dates,
        )
        frame.index.name = "날짜"
        return frame

    monkeypatch.setattr(stock, "get_market_trading_value_by_date", _trading_value)

    raw_frame, sector_frame, summary = flow_source.collect_sector_investor_flow(
        sector_map={"regimes": {"Recovery": {"sectors": [{"code": "5044", "name": "KRX 반도체"}]}}},
        start="20260407",
        end="20260408",
    )

    assert not raw_frame.empty, "Expected data even when pykrx returns DataFrame"
    assert not sector_frame.empty


# ---------------------------------------------------------------------------
# Snapshot fallback when ALL sector constituent lookups return empty
# ---------------------------------------------------------------------------

def test_collect_sector_investor_flow_uses_snapshot_fallback(monkeypatch):
    """When live constituent lookup returns empty, warehouse snapshot is used."""
    monkeypatch.setattr(flow_source, "ensure_pykrx_transport_compat", lambda: None)
    monkeypatch.setattr(flow_source, "_check_socket_stack", lambda: None)
    monkeypatch.setattr(flow_source, "upsert_sector_constituents_snapshot", lambda *a, **kw: None)

    import pykrx.stock as stock

    monkeypatch.setattr(
        flow_source,
        "lookup_index_constituents",
        lambda *args, **kwargs: ConstituentLookupResult(
            tickers=[],
            failure_detail="empty constituent list across candidate dates 20260407..20260408",
        ),
    )
    monkeypatch.setattr(stock, "get_market_ticker_name", lambda ticker: f"Name {ticker}")

    # Provide a cached snapshot
    cached_snap = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(["2026-04-07"]),
            "sector_code": ["5044"],
            "ticker": ["005930"],
            "reference_date": pd.to_datetime(["2026-04-07"]),
            "resolved_from": ["20260407"],
            "provider": ["PYKRX_UNOFFICIAL"],
            "is_fallback": [False],
        }
    )
    monkeypatch.setattr(
        flow_source,
        "read_latest_sector_constituents_snapshot",
        lambda sector_codes, market="KR": cached_snap,
    )

    dates = pd.to_datetime(["2026-04-07", "2026-04-08"])

    def _trading_value(start, end, ticker, on="순매수", **kwargs):
        frame = pd.DataFrame(
            {"개인": [100, 110], "외국인합계": [200, 210], "기관합계": [50, 55], "전체": [0, 0]},
            index=dates,
        )
        frame.index.name = "날짜"
        return frame

    monkeypatch.setattr(stock, "get_market_trading_value_by_date", _trading_value)

    raw_frame, sector_frame, summary = flow_source.collect_sector_investor_flow(
        sector_map={"regimes": {"Recovery": {"sectors": [{"code": "5044", "name": "KRX 반도체"}]}}},
        start="20260407",
        end="20260408",
    )

    # Snapshot fallback should allow data collection to proceed
    assert not raw_frame.empty, "Snapshot fallback should supply tickers for collection"
    assert not sector_frame.empty
    # sector:5044 should be marked as CACHED_FALLBACK failure (constituent lookup failed)
    assert any("CACHED_FALLBACK" in v for v in summary["failed_codes"].values()), (
        "Sector constituent failure should be tagged as CACHED_FALLBACK"
    )


# ---------------------------------------------------------------------------
# stage-based RuntimeError message in run_manual_investor_flow_refresh
# ---------------------------------------------------------------------------

def test_run_manual_investor_flow_refresh_raises_structured_error_on_total_failure(monkeypatch):
    """When sector_frame is empty, the RuntimeError must include stage classification."""
    monkeypatch.setattr(
        flow_source,
        "collect_sector_investor_flow",
        lambda **kwargs: (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "status": "SAMPLE",
                "provider": flow_source.FLOW_PROVIDER,
                "requested_start": "20260101",
                "requested_end": "20260408",
                "coverage_complete": False,
                "failed_days": [],
                "failed_codes": {
                    "sector:5044": "empty constituent list",
                    "sector:5045": "empty constituent list",
                },
                "predicted_requests": 0,
                "processed_requests": 0,
                "rows": 0,
                "tracked_sectors": 2,
                "tracked_tickers": 0,
            },
        ),
    )

    _, summary = flow_source.run_manual_investor_flow_refresh(
        sector_map=_SECTOR_MAP,
        end_date_str="20260408",
    )

    # Should fall back gracefully, not raise to caller
    assert summary["coverage_complete"] is False
    assert "refresh" in summary["failed_codes"]
    error_msg = summary["failed_codes"]["refresh"]
    assert "CONSTITUENT_ERROR" in error_msg


def test_run_manual_investor_flow_refresh_surfaces_auth_required(monkeypatch):
    monkeypatch.setattr(
        flow_source,
        "collect_sector_investor_flow",
        lambda **kwargs: (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "status": "SAMPLE",
                "provider": flow_source.FLOW_PROVIDER,
                "requested_start": "20260101",
                "requested_end": "20260408",
                "coverage_complete": False,
                "failed_days": [],
                "failed_codes": {
                    "sector:5044": "AUTH_REQUIRED: KRX Data Marketplace login is required",
                },
                "predicted_requests": 0,
                "processed_requests": 0,
                "rows": 0,
                "tracked_sectors": 1,
                "tracked_tickers": 0,
            },
        ),
    )

    _, summary = flow_source.run_manual_investor_flow_refresh(
        sector_map={"regimes": {"Recovery": {"sectors": [{"code": "5044", "name": "KRX 반도체"}]}}},
        end_date_str="20260408",
    )

    assert "AUTH_REQUIRED" in summary["failed_codes"]["refresh"]


def test_resolve_investor_flow_refresh_window_bootstraps_without_cursor():
    start, end, meta = flow_source.resolve_investor_flow_refresh_window(
        end_date_str="20260408",
        market="KR",
    )

    assert end == "20260408"
    assert start == (pd.Timestamp("2026-04-08") - pd.Timedelta(days=120)).strftime("%Y%m%d")
    assert meta["mode"] == "bootstrap_seed"


def test_resolve_investor_flow_refresh_window_uses_cursor_and_failed_days():
    warehouse.update_ingest_watermark(
        dataset="investor_flow_operational_complete",
        watermark_key="20260408",
        status="LIVE",
        coverage_complete=True,
        provider="PYKRX_UNOFFICIAL",
        details={"status": "LIVE"},
        market="KR",
    )
    warehouse.record_ingest_run(
        dataset="investor_flow",
        reason="manual_refresh",
        provider="PYKRX_UNOFFICIAL",
        requested_start="20260409",
        requested_end="20260410",
        status="LIVE",
        coverage_complete=False,
        failed_days=["20260410"],
        failed_codes={"20260410": "partial"},
        delta_keys=[],
        row_count=1,
        summary={"status": "LIVE"},
        market="KR",
    )

    start, end, meta = flow_source.resolve_investor_flow_refresh_window(
        end_date_str="20260410",
        market="KR",
    )

    assert start == "20260410"
    assert end == "20260410"
    assert meta["mode"] == "incremental"
    assert meta["complete_cursor"] == "20260408"
    assert meta["failed_days_repaired"] == ["20260410"]


def test_resolve_investor_flow_refresh_window_skips_non_trading_days_after_cursor(monkeypatch):
    warehouse.update_ingest_watermark(
        dataset="investor_flow_operational_complete",
        watermark_key="20260408",
        status="LIVE",
        coverage_complete=True,
        provider="PYKRX_UNOFFICIAL",
        details={"status": "LIVE"},
        market="KR",
    )
    monkeypatch.setattr(flow_source, "_requested_trading_days", lambda *args, **kwargs: ["20260410"])

    start, end, meta = flow_source.resolve_investor_flow_refresh_window(
        end_date_str="20260410",
        market="KR",
    )

    assert start == "20260410"
    assert end == "20260410"
    assert meta["mode"] == "incremental"


def test_run_manual_investor_flow_refresh_returns_cached_when_already_current(monkeypatch):
    warehouse.update_ingest_watermark(
        dataset="investor_flow_operational_complete",
        watermark_key="20260408",
        status="LIVE",
        coverage_complete=True,
        provider="PYKRX_UNOFFICIAL",
        details={"status": "LIVE"},
        market="KR",
    )
    monkeypatch.setattr(
        flow_source,
        "collect_sector_investor_flow",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("collector should not run when already current")),
    )

    (status, cached), summary = flow_source.run_manual_investor_flow_refresh(
        sector_map=_SECTOR_MAP,
        end_date_str="20260408",
    )

    assert status == "CACHED"
    assert cached.empty
    assert summary["coverage_complete"] is True


def test_discover_oldest_collectable_date_chooses_earliest_day_with_rows(monkeypatch):
    def _collect(**kwargs):
        start = kwargs["start"]
        end = kwargs["end"]
        if end < "20250103":
            return pd.DataFrame(), pd.DataFrame(), {"status": "SAMPLE", "predicted_requests": 1, "processed_requests": 1}
        if start <= "20250103" <= end:
            raw = _single_day_raw_frame("2025-01-03")
            return raw, pd.DataFrame(), {"status": "LIVE", "predicted_requests": 1, "processed_requests": 1}
        return pd.DataFrame(), pd.DataFrame(), {"status": "SAMPLE", "predicted_requests": 1, "processed_requests": 1}

    monkeypatch.setattr(flow_source, "collect_sector_investor_flow", _collect)

    discovered, details = flow_source.discover_oldest_collectable_date(
        sector_map=_SECTOR_MAP,
        end_date_str="20250105",
        earliest_candidate_str="20250101",
        discovery_window_days=3,
        market="KR",
    )

    assert discovered == "20250103"
    assert details["requested_day"] == "20250103"
    assert details["oldest_collectable_date"] == "20250103"
    assert details["requested_earliest_candidate"] == "20250101"
    assert details["collector_contract_floor"] == "20140502"
    assert details["used_cached_constituent_fallback"] is False
    assert details["status"] == "LIVE"
    assert set(details) == {
        "requested_day",
        "oldest_collectable_date",
        "requested_earliest_candidate",
        "collector_contract_floor",
        "discovery_window_start",
        "discovery_window_end",
        "discovery_window_days",
        "used_cached_constituent_fallback",
        "provider",
        "status",
    }


def test_discover_oldest_collectable_date_clamps_to_collector_contract_floor(monkeypatch):
    calls: list[tuple[str, str]] = []

    def _collect(**kwargs):
        calls.append((kwargs["start"], kwargs["end"]))
        day = kwargs["start"]
        raw = _single_day_raw_frame(day)
        return raw, pd.DataFrame(), {"status": "LIVE", "predicted_requests": 1, "processed_requests": 1}

    monkeypatch.setattr(flow_source, "collect_sector_investor_flow", _collect)

    discovered, details = flow_source.discover_oldest_collectable_date(
        sector_map=_SECTOR_MAP,
        end_date_str="20140510",
        earliest_candidate_str="19900101",
        discovery_window_days=3,
        market="KR",
    )

    assert discovered == "20140502"
    assert details["requested_earliest_candidate"] == "19900101"
    assert details["collector_contract_floor"] == "20140502"
    assert calls[0][0] == "20140502"


def test_discover_oldest_collectable_date_rejects_requested_day_mismatch(monkeypatch):
    def _collect(**kwargs):
        start = kwargs["start"]
        end = kwargs["end"]
        if start == end == "20250102":
            raw_day = "20250103"
        elif start <= "20250103" <= end:
            raw_day = "20250103"
        else:
            return pd.DataFrame(), pd.DataFrame(), {"status": "SAMPLE", "predicted_requests": 1, "processed_requests": 1}
        raw = _single_day_raw_frame(raw_day)
        return raw, pd.DataFrame(), {"status": "LIVE", "predicted_requests": 1, "processed_requests": 1}

    monkeypatch.setattr(flow_source, "collect_sector_investor_flow", _collect)

    discovered, details = flow_source.discover_oldest_collectable_date(
        sector_map=_SECTOR_MAP,
        end_date_str="20250104",
        earliest_candidate_str="20250102",
        discovery_window_days=3,
        market="KR",
    )

    assert discovered == "20250103"
    assert details["requested_day"] == "20250103"


def test_discover_oldest_collectable_date_allows_partial_coverage_when_requested_day_exists(monkeypatch):
    def _collect(**kwargs):
        start = kwargs["start"]
        end = kwargs["end"]
        if end < "20250103":
            return pd.DataFrame(), pd.DataFrame(), {"status": "SAMPLE", "predicted_requests": 1, "processed_requests": 1}
        if start <= "20250103" <= end:
            raw = _single_day_raw_frame("2025-01-03")
            return raw, pd.DataFrame(), {
                "status": "LIVE",
                "coverage_complete": False,
                "failed_days": [],
                "failed_codes": {"005930": "partial ticker failure outside discovery contract"},
                "predicted_requests": 1,
                "processed_requests": 1,
            }
        return pd.DataFrame(), pd.DataFrame(), {"status": "SAMPLE", "predicted_requests": 1, "processed_requests": 1}

    monkeypatch.setattr(flow_source, "collect_sector_investor_flow", _collect)

    discovered, details = flow_source.discover_oldest_collectable_date(
        sector_map=_SECTOR_MAP,
        end_date_str="20250105",
        earliest_candidate_str="20250101",
        discovery_window_days=3,
        market="KR",
    )

    assert discovered == "20250103"
    assert details["status"] == "LIVE"
    assert details["used_cached_constituent_fallback"] is False


def test_discover_oldest_collectable_date_excludes_cached_constituent_fallback(monkeypatch):
    def _collect(**kwargs):
        start = kwargs["start"]
        end = kwargs["end"]
        if start <= "20250102" <= end:
            raw = _single_day_raw_frame("2025-01-02")
            return raw, pd.DataFrame(), {
                "status": "LIVE",
                "coverage_complete": False,
                "failed_days": [],
                "failed_codes": {"sector:5044": "CACHED_FALLBACK(from 20250101): lookup failed"},
                "predicted_requests": 1,
                "processed_requests": 1,
            }
        if start <= "20250103" <= end:
            raw = _single_day_raw_frame("2025-01-03")
            return raw, pd.DataFrame(), {
                "status": "LIVE",
                "coverage_complete": True,
                "failed_days": [],
                "failed_codes": {},
                "predicted_requests": 1,
                "processed_requests": 1,
            }
        return pd.DataFrame(), pd.DataFrame(), {"status": "SAMPLE", "predicted_requests": 1, "processed_requests": 1}

    monkeypatch.setattr(flow_source, "collect_sector_investor_flow", _collect)

    discovered, details = flow_source.discover_oldest_collectable_date(
        sector_map=_SECTOR_MAP,
        end_date_str="20250105",
        earliest_candidate_str="20250102",
        discovery_window_days=3,
        market="KR",
    )

    assert discovered == "20250103"
    assert details["requested_day"] == "20250103"
    assert details["used_cached_constituent_fallback"] is False


def test_run_historical_investor_flow_backfill_updates_progress_cursor(monkeypatch):
    def _collect(**kwargs):
        day = kwargs["start"]
        raw = pd.DataFrame(
            {
                "trade_date": pd.to_datetime([day]),
                "ticker": ["005930"],
                "ticker_name": ["삼성전자"],
                "investor_type": ["외국인"],
                "buy_amount": [100],
                "sell_amount": [50],
                "net_buy_amount": [50],
            }
        )
        return raw, pd.DataFrame(), {"status": "LIVE", "failed_days": [], "failed_codes": {}}

    monkeypatch.setattr(flow_source, "collect_sector_investor_flow", _collect)

    summary = flow_source.run_historical_investor_flow_backfill(
        sector_map=_SECTOR_MAP,
        end_date_str="20250103",
        oldest_collectable_date="20250102",
        market="KR",
    )

    sector_rows = warehouse.read_sector_investor_flow(
        ["5044"],
        "20250102",
        "20250103",
        market="KR",
        cap_to_operational_cursor=False,
    )

    assert summary["status"] == "LIVE"
    assert warehouse.read_investor_flow_backfill_progress_cursor(market="KR") == "20250103"
    assert sector_rows.empty


def test_run_historical_investor_flow_backfill_validation_mode_does_not_advance_progress(monkeypatch):
    def _collect(**kwargs):
        day = kwargs["start"]
        raw = pd.DataFrame(
            {
                "trade_date": pd.to_datetime([day]),
                "ticker": ["005930"],
                "ticker_name": ["삼성전자"],
                "investor_type": ["외국인"],
                "buy_amount": [100],
                "sell_amount": [50],
                "net_buy_amount": [50],
            }
        )
        return raw, pd.DataFrame(), {"status": "LIVE", "failed_days": [], "failed_codes": {}}

    monkeypatch.setattr(flow_source, "collect_sector_investor_flow", _collect)

    summary = flow_source.run_historical_investor_flow_backfill(
        sector_map=_SECTOR_MAP,
        start_date_str="20250102",
        end_date_str="20250103",
        oldest_collectable_date="20250102",
        market="KR",
        track_progress=False,
        reason=flow_source.HISTORICAL_BACKFILL_VALIDATION_REASON,
    )

    latest_validation_run = warehouse.read_latest_investor_flow_run(
        market="KR",
        reasons=(flow_source.HISTORICAL_BACKFILL_VALIDATION_REASON,),
    )

    assert summary["status"] == "LIVE"
    assert summary["track_progress"] is False
    assert warehouse.read_investor_flow_backfill_progress_cursor(market="KR") is None
    assert latest_validation_run["reason"] == flow_source.HISTORICAL_BACKFILL_VALIDATION_REASON


def test_run_historical_investor_flow_backfill_preserves_failure_reason(monkeypatch):
    monkeypatch.setattr(
        flow_source,
        "collect_sector_investor_flow",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("transport timeout")),
    )

    summary = flow_source.run_historical_investor_flow_backfill(
        sector_map=_SECTOR_MAP,
        start_date_str="20250102",
        end_date_str="20250102",
        oldest_collectable_date="20250102",
        market="KR",
        track_progress=False,
        reason=flow_source.HISTORICAL_BACKFILL_VALIDATION_REASON,
    )

    latest_validation_run = warehouse.read_latest_investor_flow_run(
        market="KR",
        reasons=(flow_source.HISTORICAL_BACKFILL_VALIDATION_REASON,),
    )

    assert summary["status"] == "SAMPLE"
    assert latest_validation_run["reason"] == flow_source.HISTORICAL_BACKFILL_VALIDATION_REASON
    assert "20250102" in latest_validation_run["failed_codes"]


def test_read_warm_status_ignores_backfill_runs():
    warehouse.record_ingest_run(
        dataset="investor_flow",
        reason="manual_refresh",
        provider="PYKRX_UNOFFICIAL",
        requested_start="20260401",
        requested_end="20260408",
        status="LIVE",
        coverage_complete=False,
        failed_days=["20260408"],
        failed_codes={"20260408": "partial"},
        delta_keys=[],
        row_count=0,
        summary={"status": "LIVE"},
        market="KR",
        created_at=pd.Timestamp("2026-04-10T00:00:00Z").to_pydatetime(),
    )
    warehouse.record_ingest_run(
        dataset="investor_flow",
        reason=flow_source.HISTORICAL_BACKFILL_REASON,
        provider="PYKRX_UNOFFICIAL",
        requested_start="20250102",
        requested_end="20250131",
        status="SAMPLE",
        coverage_complete=False,
        failed_days=["20250115"],
        failed_codes={"20250115": "AUTH_REQUIRED: blocked"},
        delta_keys=[],
        row_count=0,
        summary={"status": "SAMPLE"},
        market="KR",
        created_at=pd.Timestamp("2026-04-11T00:00:00Z").to_pydatetime(),
    )
    warehouse.update_ingest_watermark(
        dataset="investor_flow_operational_complete",
        watermark_key="20260407",
        status="LIVE",
        coverage_complete=True,
        provider="PYKRX_UNOFFICIAL",
        details={"status": "LIVE"},
        market="KR",
    )

    summary = flow_source.read_warm_status()

    assert summary["end"] == "20260407"
    assert summary["coverage_complete"] is True
    assert summary["failed_codes"] == {"20260408": "partial"}
