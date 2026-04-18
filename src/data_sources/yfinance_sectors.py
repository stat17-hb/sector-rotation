"""
US sector ETF price loader backed by Yahoo chart history and DuckDB cache.
"""
from __future__ import annotations

from datetime import date
import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.contracts.validators import normalize_then_validate
from src.data_sources.warehouse import (
    close_cached_read_only_connection,
    export_market_parquet,
    get_dataset_artifact_key,
    is_market_coverage_complete,
    probe_dataset_mode,
    read_dataset_status,
    read_market_prices,
    record_ingest_run,
    update_ingest_watermark,
    upsert_index_dimension,
    upsert_market_prices,
)
from src.data_sources.yahoo_chart import fetch_yahoo_chart_history_batch

logger = logging.getLogger(__name__)

DataStatus = Literal["LIVE", "CACHED", "SAMPLE"]
LoaderResult = tuple[DataStatus, pd.DataFrame]

CURATED_PATH = Path("data/curated/sector_prices_us.parquet")
MARKET_ID = "US"
DEFAULT_TICKER_NAMES: dict[str, str] = {
    "SPY": "S&P 500",
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}


def _run_best_effort_bookkeeping(label: str, callback, /, *args, **kwargs) -> None:
    try:
        callback(*args, **kwargs)
    except Exception as exc:
        logger.warning("US market bookkeeping skipped (%s): %s", label, exc)
def _to_long_frame(history_frame: pd.DataFrame) -> pd.DataFrame:
    if history_frame.empty:
        return pd.DataFrame()

    long = history_frame.copy()
    long["index_code"] = long["ticker"].astype(str)
    long["index_name"] = long["index_code"].map(
        lambda code: DEFAULT_TICKER_NAMES.get(str(code), str(code))
    )
    long.index = pd.DatetimeIndex(long.index).normalize()
    long["index_code"] = long["index_code"].astype("object")
    long["index_name"] = long["index_name"].astype("object")
    long["close"] = pd.to_numeric(long["close"], errors="coerce")
    long = long.dropna(subset=["close"])
    return long[["index_code", "index_name", "close"]]


def _index_dimension_rows(tickers: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "index_code": ticker,
            "index_name": DEFAULT_TICKER_NAMES.get(ticker, ticker),
            "family": "ETF",
            "is_benchmark": ticker == "SPY",
            "is_active": True,
            "export_sector": None,
        }
        for ticker in tickers
    ]


def fetch_sector_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    normalized = [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
    if not normalized:
        return pd.DataFrame()

    history = fetch_yahoo_chart_history_batch(normalized, start, end)
    if history.empty:
        raise ValueError("Yahoo chart returned no rows")
    return _to_long_frame(history)


def _make_sample_df(tickers: list[str]) -> pd.DataFrame:
    dates = pd.date_range(end=date.today(), periods=252 * 3, freq="B")
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        values = pd.Series(range(len(dates)), dtype="float64").add(100.0)
        values = values.pct_change().fillna(0.0).add(1.0).cumprod().mul(100.0)
        for trade_date, close in zip(dates, values):
            rows.append(
                {
                    "trade_date": trade_date,
                    "index_code": ticker,
                    "index_name": DEFAULT_TICKER_NAMES.get(ticker, ticker),
                    "close": float(close),
                }
            )
    frame = pd.DataFrame(rows).set_index("trade_date")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame[["index_code", "index_name", "close"]].astype(
        {"index_code": "object", "index_name": "object", "close": "float64"}
    )


def get_price_artifact_key() -> tuple[int, int, str, str, str]:
    return get_dataset_artifact_key("market_prices", market=MARKET_ID)


def probe_market_status() -> str:
    return probe_dataset_mode("market_prices", market=MARKET_ID)


def run_manual_price_refresh(
    tickers: list[str],
    start: str,
    end: str,
) -> tuple[LoaderResult, dict[str, Any]]:
    status, frame = load_sector_prices(tickers, start, end)
    summary = {
        "status": status,
        "coverage_complete": bool(
            not frame.empty and is_market_coverage_complete(tickers, start, end, benchmark_code="SPY", market=MARKET_ID)
        ),
        "delta_codes": sorted({str(code) for code in frame.get("index_code", pd.Series(dtype=object)).astype(str).unique()}) if not frame.empty else [],
        "failed_days": [],
        "failed_codes": {},
        "provider": "YFINANCE",
        "reason": "manual_refresh",
        "rows": int(len(frame)),
    }
    return (status, frame), summary


def load_sector_prices(tickers: list[str], start: str, end: str) -> LoaderResult:
    normalized = [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
    if not normalized:
        return ("SAMPLE", pd.DataFrame())

    coverage_code = "SPY" if "SPY" in normalized else normalized[0]
    cached = read_market_prices(normalized, start, end, market=MARKET_ID)
    if not cached.empty:
        coverage_complete = is_market_coverage_complete(
            normalized,
            start,
            end,
            benchmark_code=coverage_code,
            market=MARKET_ID,
        )
        if coverage_complete:
            return ("CACHED", normalize_then_validate(cached, "sector_prices"))

        digits = "".join(ch for ch in str(end or "") if ch.isdigit())
        requested_end = pd.Timestamp(date(int(digits[:4]), int(digits[4:6]), int(digits[6:8])))
        benchmark_cached = cached[cached["index_code"].astype(str) == coverage_code]
        if not benchmark_cached.empty:
            benchmark_latest = pd.Timestamp(benchmark_cached.index.max()).normalize()
            if benchmark_latest < requested_end:
                logger.info(
                    "US market cache stale for %s: latest benchmark date %s is before requested end %s; refreshing via yfinance.",
                    coverage_code,
                    benchmark_latest.strftime("%Y-%m-%d"),
                    requested_end.strftime("%Y-%m-%d"),
                )

    live: pd.DataFrame | None = None
    try:
        live = normalize_then_validate(fetch_sector_prices(normalized, start, end), "sector_prices")
    except Exception as exc:
        logger.warning("US market live fetch failed: %s", exc)
    else:
        try:
            close_cached_read_only_connection()
            upsert_index_dimension(_index_dimension_rows(normalized), market=MARKET_ID)
            upsert_market_prices(live, provider="YFINANCE", market=MARKET_ID)
        except RuntimeError as exc:
            logger.warning(
                "US market live refresh skipped, write lock unavailable; returning cached data. (%s)",
                exc,
            )
        else:
            _run_best_effort_bookkeeping(
                "export_market_parquet",
                export_market_parquet,
                CURATED_PATH,
                market=MARKET_ID,
            )
            coverage_complete = is_market_coverage_complete(
                normalized,
                start,
                end,
                benchmark_code=coverage_code,
                market=MARKET_ID,
            )
            _run_best_effort_bookkeeping(
                "update_ingest_watermark",
                update_ingest_watermark,
                dataset="market_prices",
                watermark_key=end,
                status="LIVE",
                coverage_complete=coverage_complete,
                provider="YFINANCE",
                details={"tickers": normalized},
                market=MARKET_ID,
            )
            _run_best_effort_bookkeeping(
                "record_ingest_run",
                record_ingest_run,
                dataset="market_prices",
                reason="load_sector_prices",
                provider="YFINANCE",
                requested_start=start,
                requested_end=end,
                status="LIVE",
                coverage_complete=coverage_complete,
                failed_days=[],
                failed_codes={},
                delta_keys=normalized,
                row_count=int(len(live)),
                summary={"status": "LIVE", "rows": int(len(live)), "coverage_complete": coverage_complete},
                market=MARKET_ID,
            )
            return ("LIVE", live)

    cached = read_market_prices(normalized, start, end, market=MARKET_ID)
    if not cached.empty:
        _run_best_effort_bookkeeping(
            "record_ingest_run",
            record_ingest_run,
            dataset="market_prices",
            reason="load_sector_prices_cache_fallback",
            provider="YFINANCE",
            requested_start=start,
            requested_end=end,
            status="CACHED",
            coverage_complete=False,
            failed_days=[],
            failed_codes={},
            delta_keys=[],
            row_count=int(len(cached)),
            summary={"status": "CACHED", "rows": int(len(cached)), "source": "warehouse"},
            market=MARKET_ID,
        )
        return ("CACHED", normalize_then_validate(cached, "sector_prices"))

    if CURATED_PATH.exists():
        try:
            curated = pd.read_parquet(CURATED_PATH)
            if not curated.empty:
                curated.index = pd.DatetimeIndex(curated.index)
                return ("CACHED", normalize_then_validate(curated, "sector_prices"))
        except Exception as exc:
            logger.warning("US curated cache load failed: %s", exc)

    sample = _make_sample_df(normalized)
    _run_best_effort_bookkeeping(
        "record_ingest_run",
        record_ingest_run,
        dataset="market_prices",
        reason="load_sector_prices_sample",
        provider="YFINANCE",
        requested_start=start,
        requested_end=end,
        status="SAMPLE",
        coverage_complete=False,
        failed_days=[],
        failed_codes={},
        delta_keys=[],
        row_count=int(len(sample)),
        summary={"status": "SAMPLE", "rows": int(len(sample))},
        market=MARKET_ID,
    )
    return ("SAMPLE", sample)


def read_warm_status() -> dict[str, Any]:
    return read_dataset_status("market_prices", market=MARKET_ID)
