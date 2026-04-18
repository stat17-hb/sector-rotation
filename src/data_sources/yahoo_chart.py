"""
Direct Yahoo Finance chart API helpers.

These helpers bypass the `yfinance` wrapper layer and normalize daily OHLCV
history into a small shared contract for US market loaders.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import logging
import time as time_module
from typing import Any

import pandas as pd
import requests


logger = logging.getLogger(__name__)

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
REQUEST_TIMEOUT_SEC = 15
REQUEST_CONNECT_TIMEOUT_SEC = 5
REQUEST_RETRY_DELAYS_SEC = (0.25, 0.75)
REQUEST_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
}


def _normalize_market_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) != 8:
        raise ValueError(f"Invalid market date: {value!r}")
    return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))


def _align_series(values: Any, size: int) -> list[Any]:
    data = list(values or [])
    if len(data) < size:
        data.extend([None] * (size - len(data)))
    return data[:size]


def _normalize_gmtoffset(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _timestamp_index(timestamps: list[Any], timezone_name: str, gmtoffset: int | None) -> pd.DatetimeIndex:
    index = pd.to_datetime(list(timestamps), unit="s", utc=True)
    if timezone_name:
        try:
            index = index.tz_convert(timezone_name)
            return pd.DatetimeIndex(index.tz_localize(None).normalize())
        except Exception:
            logger.debug("Yahoo chart timezone conversion failed for %s", timezone_name)
    if gmtoffset is not None:
        shifted = index.tz_localize(None) + pd.to_timedelta(gmtoffset, unit="s")
        return pd.DatetimeIndex(shifted.normalize())
    return pd.DatetimeIndex(index.tz_localize(None).normalize())


def _parse_chart_payload(payload: dict[str, Any], ticker: str) -> pd.DataFrame:
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        raise ValueError(f"Yahoo chart payload missing chart object for {ticker}")

    error = chart.get("error")
    if error:
        if isinstance(error, dict):
            detail = str(error.get("description") or error.get("code") or error)
        else:
            detail = str(error)
        raise ValueError(f"Yahoo chart returned an error for {ticker}: {detail}")

    results = chart.get("result") or []
    if not results or not isinstance(results[0], dict):
        raise ValueError(f"Yahoo chart returned no result for {ticker}")

    result = results[0]
    timestamps = list(result.get("timestamp") or [])
    if not timestamps:
        raise ValueError(f"Yahoo chart returned no timestamps for {ticker}")

    meta = dict(result.get("meta") or {})
    timezone_name = str(meta.get("exchangeTimezoneName") or meta.get("timezone") or "")
    gmtoffset = _normalize_gmtoffset(meta.get("gmtoffset"))
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0] or {}
    adjclose = (((result.get("indicators") or {}).get("adjclose") or [{}])[0] or {}).get("adjclose") or []

    trade_index = _timestamp_index(timestamps, timezone_name, gmtoffset)
    size = len(trade_index)
    frame = pd.DataFrame(
        {
            "open": _align_series(quote.get("open"), size),
            "high": _align_series(quote.get("high"), size),
            "low": _align_series(quote.get("low"), size),
            "close": _align_series(quote.get("close"), size),
            "adj_close": _align_series(adjclose, size),
            "volume": _align_series(quote.get("volume"), size),
        },
        index=trade_index,
    ).sort_index()
    frame.index.name = "trade_date"
    frame = frame.dropna(how="all")
    if frame.empty:
        raise ValueError(f"Yahoo chart returned no usable rows for {ticker}")
    return frame


def fetch_yahoo_chart_history(
    ticker: str,
    start: str | date,
    end: str | date,
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    normalized_ticker = str(ticker or "").strip().upper()
    if not normalized_ticker:
        raise ValueError("Ticker is required for Yahoo chart history")

    start_date = _normalize_market_date(start)
    end_date = _normalize_market_date(end)
    if end_date < start_date:
        raise ValueError(f"Invalid Yahoo chart range for {normalized_ticker}: {start_date}..{end_date}")

    http = session or requests.Session()
    request_params = {
        "period1": int(datetime.combine(start_date, time.min, tzinfo=timezone.utc).timestamp()),
        "period2": int(datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=timezone.utc).timestamp()),
        "interval": "1d",
        "includeAdjustedClose": "true",
        "events": "div,splits",
    }
    response = None
    last_exc: Exception | None = None
    for attempt, delay_sec in enumerate((0.0, *REQUEST_RETRY_DELAYS_SEC), start=1):
        if delay_sec > 0:
            time_module.sleep(delay_sec)
        try:
            response = http.get(
                YAHOO_CHART_URL.format(ticker=normalized_ticker),
                params=request_params,
                timeout=(REQUEST_CONNECT_TIMEOUT_SEC, REQUEST_TIMEOUT_SEC),
                headers=REQUEST_HEADERS,
                allow_redirects=True,
            )
            if response.status_code in REQUEST_RETRYABLE_STATUS_CODES:
                raise requests.HTTPError(
                    f"Yahoo chart HTTP {response.status_code} for {normalized_ticker}",
                    response=response,
                )
            break
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            if isinstance(exc, requests.HTTPError):
                response_obj = getattr(exc, "response", None)
                status_code = getattr(response_obj, "status_code", None)
                if status_code not in REQUEST_RETRYABLE_STATUS_CODES:
                    raise
            last_exc = exc
            logger.warning(
                "Yahoo chart request failed for %s on attempt %s/%s: %s",
                normalized_ticker,
                attempt,
                len(REQUEST_RETRY_DELAYS_SEC) + 1,
                exc,
            )
    if response is None:
        assert last_exc is not None
        raise last_exc
    response.raise_for_status()
    try:
        payload = response.json()
    except Exception as exc:
        raise ValueError(f"Yahoo chart returned non-JSON response for {normalized_ticker}") from exc

    frame = _parse_chart_payload(payload, normalized_ticker)
    frame["ticker"] = normalized_ticker
    return frame[["ticker", "open", "high", "low", "close", "adj_close", "volume"]]


def fetch_yahoo_chart_history_batch(
    tickers: list[str],
    start: str | date,
    end: str | date,
    *,
    session: requests.Session | None = None,
    allow_partial: bool = False,
) -> pd.DataFrame:
    normalized = [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
    if not normalized:
        return pd.DataFrame(columns=["ticker", "open", "high", "low", "close", "adj_close", "volume"])

    http = session or requests.Session()
    frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    for ticker in normalized:
        try:
            frames.append(fetch_yahoo_chart_history(ticker, start, end, session=http))
        except Exception as exc:
            failures[ticker] = str(exc)

    if failures and not allow_partial:
        detail = ", ".join(f"{ticker}={message}" for ticker, message in sorted(failures.items()))
        raise ValueError(f"Yahoo chart batch failed: {detail}")
    if not frames:
        detail = ", ".join(f"{ticker}={message}" for ticker, message in sorted(failures.items()))
        raise ValueError(f"Yahoo chart batch returned no rows: {detail}")

    frame = pd.concat(frames).sort_index()
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    frame.attrs["failed_tickers"] = failures
    return frame


__all__ = [
    "fetch_yahoo_chart_history",
    "fetch_yahoo_chart_history_batch",
]
