from __future__ import annotations

import pandas as pd
import pytest
import requests

import src.data_sources.yahoo_chart as yahoo_chart


class _Response:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.url = "https://query1.finance.yahoo.com/v8/finance/chart/SPY"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payloads: dict[str, dict]):
        self._payloads = payloads

    def get(self, url, *args, **kwargs):
        ticker = str(url).rsplit("/", 1)[-1].upper()
        return _Response(self._payloads[ticker])


class _RetrySession:
    def __init__(self, responses: list[object]):
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, *args, **kwargs):
        _ = (url, args, kwargs)
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _payload(
    symbol: str,
    closes: list[float],
    volumes: list[int],
    *,
    timestamps: list[int] | None = None,
    meta_overrides: dict | None = None,
) -> dict:
    meta = {"symbol": symbol, "exchangeTimezoneName": "America/New_York"}
    meta.update(meta_overrides or {})
    return {
        "chart": {
            "result": [
                {
                    "meta": meta,
                    "timestamp": timestamps or [1704171600, 1704258000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [closes[0] - 1.0, closes[1] - 1.0],
                                "high": [closes[0] + 1.0, closes[1] + 1.0],
                                "low": [closes[0] - 2.0, closes[1] - 2.0],
                                "close": closes,
                                "volume": volumes,
                            }
                        ],
                        "adjclose": [{"adjclose": closes}],
                    },
                }
            ],
            "error": None,
        }
    }


def test_fetch_yahoo_chart_history_parses_daily_rows():
    frame = yahoo_chart.fetch_yahoo_chart_history(
        "SPY",
        "20240102",
        "20240103",
        session=_Session({"SPY": _payload("SPY", [100.0, 101.0], [1000, 1100])}),
    )

    assert list(frame["ticker"]) == ["SPY", "SPY"]
    assert list(frame["close"]) == [100.0, 101.0]
    assert frame.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_fetch_yahoo_chart_history_batch_collects_multiple_tickers():
    frame = yahoo_chart.fetch_yahoo_chart_history_batch(
        ["SPY", "XLV"],
        "20240102",
        "20240103",
        session=_Session(
            {
                "SPY": _payload("SPY", [100.0, 101.0], [1000, 1100]),
                "XLV": _payload("XLV", [130.0, 131.0], [2000, 2100]),
            }
        ),
    )

    assert set(frame["ticker"].astype(str).unique()) == {"SPY", "XLV"}
    assert len(frame) == 4


def test_fetch_yahoo_chart_history_batch_can_keep_partial_results():
    frame = yahoo_chart.fetch_yahoo_chart_history_batch(
        ["SPY", "XLV"],
        "20240102",
        "20240103",
        session=_Session(
            {
                "SPY": _payload("SPY", [100.0, 101.0], [1000, 1100]),
                "XLV": {"chart": {"result": None, "error": None}},
            }
        ),
        allow_partial=True,
    )

    assert set(frame["ticker"].astype(str).unique()) == {"SPY"}
    assert frame.attrs["failed_tickers"]["XLV"].startswith("Yahoo chart returned no result")


def test_fetch_yahoo_chart_history_batch_raises_when_all_fail():
    with pytest.raises(ValueError, match="Yahoo chart batch failed"):
        yahoo_chart.fetch_yahoo_chart_history_batch(
            ["SPY"],
            "20240102",
            "20240103",
            session=_Session({"SPY": {"chart": {"result": None, "error": None}}}),
        )


def test_fetch_yahoo_chart_history_uses_gmtoffset_when_timezone_name_is_missing():
    frame = yahoo_chart.fetch_yahoo_chart_history(
        "SPY",
        "20240102",
        "20240103",
        session=_Session(
            {
                "SPY": _payload(
                    "SPY",
                    [100.0, 101.0],
                    [1000, 1100],
                    timestamps=[1704243600, 1704330000],  # 2024-01-03/04 01:00 UTC
                    meta_overrides={"exchangeTimezoneName": "", "timezone": "", "gmtoffset": -18000},
                )
            }
        ),
    )

    assert frame.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_fetch_yahoo_chart_history_falls_back_from_invalid_timezone_token_to_gmtoffset():
    frame = yahoo_chart.fetch_yahoo_chart_history(
        "SPY",
        "20240102",
        "20240103",
        session=_Session(
            {
                "SPY": _payload(
                    "SPY",
                    [100.0, 101.0],
                    [1000, 1100],
                    timestamps=[1704243600, 1704330000],
                    meta_overrides={"exchangeTimezoneName": "EDT", "timezone": "EDT", "gmtoffset": -18000},
                )
            }
        ),
    )

    assert frame.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_fetch_yahoo_chart_history_preserves_local_dates_across_dst_boundary():
    frame = yahoo_chart.fetch_yahoo_chart_history(
        "SPY",
        "20240308",
        "20240311",
        session=_Session(
            {
                "SPY": _payload(
                    "SPY",
                    [100.0, 101.0],
                    [1000, 1100],
                    timestamps=[1709958600, 1710214200],  # local 2024-03-08 23:30 ET / 2024-03-11 23:30 ET
                )
            }
        ),
    )

    assert frame.index.tolist() == [pd.Timestamp("2024-03-08"), pd.Timestamp("2024-03-11")]


def test_fetch_yahoo_chart_history_retries_timeout_then_succeeds(monkeypatch):
    monkeypatch.setattr(yahoo_chart.time_module, "sleep", lambda seconds: None)
    session = _RetrySession(
        [
            requests.ReadTimeout("first timeout"),
            _Response(_payload("SPY", [100.0, 101.0], [1000, 1100])),
        ]
    )

    frame = yahoo_chart.fetch_yahoo_chart_history("SPY", "20240102", "20240103", session=session)

    assert session.calls == 2
    assert frame.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_fetch_yahoo_chart_history_raises_after_retry_budget_exhausted(monkeypatch):
    monkeypatch.setattr(yahoo_chart.time_module, "sleep", lambda seconds: None)
    session = _RetrySession(
        [
            requests.ReadTimeout("timeout 1"),
            requests.ReadTimeout("timeout 2"),
            requests.ReadTimeout("timeout 3"),
        ]
    )

    with pytest.raises(requests.ReadTimeout, match="timeout 3"):
        yahoo_chart.fetch_yahoo_chart_history("SPY", "20240102", "20240103", session=session)

    assert session.calls == 3


def test_fetch_yahoo_chart_history_retries_retryable_http_status_then_succeeds(monkeypatch):
    monkeypatch.setattr(yahoo_chart.time_module, "sleep", lambda seconds: None)
    session = _RetrySession(
        [
            _Response({}, status_code=503),
            _Response(_payload("SPY", [100.0, 101.0], [1000, 1100])),
        ]
    )

    frame = yahoo_chart.fetch_yahoo_chart_history("SPY", "20240102", "20240103", session=session)

    assert session.calls == 2
    assert frame.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_fetch_yahoo_chart_history_does_not_retry_non_retryable_http_status(monkeypatch):
    monkeypatch.setattr(yahoo_chart.time_module, "sleep", lambda seconds: None)
    session = _RetrySession([_Response({}, status_code=404)])

    with pytest.raises(requests.HTTPError, match="HTTP 404"):
        yahoo_chart.fetch_yahoo_chart_history("SPY", "20240102", "20240103", session=session)

    assert session.calls == 1
