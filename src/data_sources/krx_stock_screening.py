"""
KRX sector constituent stock screener.

Fetches top stocks from sector indices, scores by RS/RSI/SMA momentum,
and returns a ranked list of buy candidates.

Fallback behaviour:
- Weekend / API unavailable → returns empty list with status="UNAVAILABLE"
- Cache hit (< TTL) → returns cached result with status="CACHED"
- Live fetch success → returns scored list with status="LIVE"
"""
from __future__ import annotations

import logging
import math
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

DataStatus = Literal["LIVE", "CACHED", "UNAVAILABLE"]
CACHE_PATH = Path("data/curated/stock_screening_cache.pkl")
CACHE_TTL_HOURS = 24
MAX_STOCKS_PER_SECTOR = 15


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_screened_stocks(
    strong_buy_sectors: list[dict],
    benchmark_code: str = "1001",
    settings: dict | None = None,
    force_refresh: bool = False,
) -> tuple[DataStatus, list[dict]]:
    """Load momentum-screened stocks for Strong Buy sectors.

    Args:
        strong_buy_sectors: List of {"code": "5044", "name": "KRX 반도체"}.
        benchmark_code: KOSPI benchmark code (default "1001").
        settings: Dashboard settings dict (rs_ma_period, rsi_period, etc.).
        force_refresh: Bypass cache and re-fetch.

    Returns:
        (status, rows) where each row is a dict with scoring fields.
    """
    if not strong_buy_sectors:
        return "UNAVAILABLE", []

    if not force_refresh:
        cached = _read_cache(strong_buy_sectors)
        if cached is not None:
            return "CACHED", cached

    try:
        rows = _fetch_and_score(strong_buy_sectors, benchmark_code, settings or {})
        if rows:
            _write_cache(strong_buy_sectors, rows)
            return "LIVE", rows
        return "UNAVAILABLE", []
    except Exception as exc:
        logger.warning("Stock screening failed: %s", exc)
        return "UNAVAILABLE", []


# ---------------------------------------------------------------------------
# Internal: fetch & score
# ---------------------------------------------------------------------------

def _fetch_and_score(
    sectors: list[dict],
    benchmark_code: str,
    settings: dict,
) -> list[dict]:
    """Fetch constituent stocks for each sector and apply momentum scoring."""
    import pykrx.stock as stock
    from src.data_sources.pykrx_compat import ensure_pykrx_transport_compat

    ensure_pykrx_transport_compat()

    trade_date = _last_business_day()
    start_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")

    # Fetch benchmark
    benchmark_df = stock.get_index_ohlcv_by_date(start_date, trade_date, benchmark_code)
    if benchmark_df is None or benchmark_df.empty:
        logger.warning("Benchmark data unavailable for %s", benchmark_code)
        return []
    bench_close = benchmark_df["종가"] if "종가" in benchmark_df.columns else benchmark_df.iloc[:, 3]
    bench_close.index = pd.to_datetime(bench_close.index)

    rs_ma_period = int(settings.get("rs_ma_period", 20))
    rsi_period = int(settings.get("rsi_period", 14))
    ma_fast = int(settings.get("ma_fast", 20))
    ma_slow = int(settings.get("ma_slow", 60))

    rows: list[dict] = []
    for sector in sectors:
        sector_code = str(sector["code"])
        sector_name = sector["name"]

        # Get constituent tickers
        tickers = _get_constituents(stock, trade_date, sector_code)
        if not tickers:
            logger.info("No constituents for sector %s (%s)", sector_code, sector_name)
            continue

        for ticker in tickers[:MAX_STOCKS_PER_SECTOR]:
            try:
                row = _score_stock(
                    stock=stock,
                    ticker=ticker,
                    sector_code=sector_code,
                    sector_name=sector_name,
                    start_date=start_date,
                    trade_date=trade_date,
                    bench_close=bench_close,
                    rs_ma_period=rs_ma_period,
                    rsi_period=rsi_period,
                    ma_fast=ma_fast,
                    ma_slow=ma_slow,
                )
                if row is not None:
                    rows.append(row)
            except Exception as exc:
                logger.debug("Skipping ticker %s: %s", ticker, exc)

    return sorted(rows, key=lambda r: r.get("rs", 0), reverse=True)


def _get_constituents(stock_module, trade_date: str, sector_code: str) -> list[str]:
    """Get constituent ticker codes for a sector index."""
    try:
        result = stock_module.get_index_portfolio_deposit_file(trade_date, sector_code)
        if isinstance(result, pd.DataFrame) and not result.empty:
            return result.index.tolist() if result.index.dtype == object else result.iloc[:, 0].tolist()
        if isinstance(result, (list, tuple)) and result:
            return list(result)
    except Exception as exc:
        logger.debug("get_index_portfolio_deposit_file failed for %s: %s", sector_code, exc)

    # Fallback: try one business day earlier
    try:
        prev_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
        result = stock_module.get_index_portfolio_deposit_file(prev_date, sector_code)
        if isinstance(result, pd.DataFrame) and not result.empty:
            return result.index.tolist() if result.index.dtype == object else result.iloc[:, 0].tolist()
    except Exception:
        pass

    return []


def _score_stock(
    stock,
    ticker: str,
    sector_code: str,
    sector_name: str,
    start_date: str,
    trade_date: str,
    bench_close: pd.Series,
    rs_ma_period: int,
    rsi_period: int,
    ma_fast: int,
    ma_slow: int,
) -> dict | None:
    """Fetch OHLCV and compute momentum score for one stock."""
    from src.indicators.momentum import compute_rs, compute_rs_ma, is_rs_strong, is_trend_positive
    from src.indicators.rsi import compute_rsi

    df = stock.get_market_ohlcv_by_date(start_date, trade_date, ticker)
    if df is None or df.empty:
        return None

    close_col = "종가" if "종가" in df.columns else df.columns[3]
    close = df[close_col].copy()
    close.index = pd.to_datetime(close.index)
    close = close.sort_index().dropna()

    if len(close) < ma_slow:
        return None

    name = stock.get_market_ticker_name(ticker) or ticker

    # RS
    rs_series = compute_rs(close, bench_close)
    rs_ma_series = compute_rs_ma(rs_series, period=rs_ma_period)
    if rs_series.empty or rs_ma_series.empty:
        return None

    rs_val = float(rs_series.iloc[-1])
    rs_ma_val = float(rs_ma_series.iloc[-1])
    rs_strong = bool(is_rs_strong(rs_val, rs_ma_val))

    # SMA trend
    trend_ok = bool(is_trend_positive(close, fast=ma_fast, slow=ma_slow))

    # RSI
    rsi_series = compute_rsi(close, period=rsi_period)
    rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else float("nan")

    # Returns
    ret_1m = _pct_return(close, 21)
    ret_3m = _pct_return(close, 63)

    # Alerts
    alerts = []
    if not math.isnan(rsi_val):
        if rsi_val >= 70:
            alerts.append("과열")
        elif rsi_val <= 30:
            alerts.append("과매도")

    # Score: pass both filters = top candidate
    momentum_ok = rs_strong and trend_ok

    return {
        "ticker": ticker,
        "name": name,
        "sector_code": sector_code,
        "sector_name": sector_name,
        "rs": round(rs_val, 4),
        "rs_ma": round(rs_ma_val, 4),
        "rs_strong": rs_strong,
        "trend_ok": trend_ok,
        "momentum_ok": momentum_ok,
        "rsi": round(rsi_val, 1) if not math.isnan(rsi_val) else None,
        "ret_1m": round(ret_1m, 1) if ret_1m is not None else None,
        "ret_3m": round(ret_3m, 1) if ret_3m is not None else None,
        "alerts": ", ".join(alerts),
    }


def _pct_return(close: pd.Series, days: int) -> float | None:
    if len(close) < days + 1:
        return None
    start_price = float(close.iloc[-(days + 1)])
    end_price = float(close.iloc[-1])
    if start_price == 0:
        return None
    return (end_price / start_price - 1) * 100


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(sectors: list[dict]) -> str:
    return ",".join(sorted(s["code"] for s in sectors))


def _read_cache(sectors: list[dict]) -> list[dict] | None:
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        if cached.get("key") != _cache_key(sectors):
            return None
        age_hours = (datetime.now() - cached["ts"]).total_seconds() / 3600
        if age_hours > CACHE_TTL_HOURS:
            return None
        return cached["rows"]
    except Exception:
        return None


def _write_cache(sectors: list[dict], rows: list[dict]) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"key": _cache_key(sectors), "ts": datetime.now(), "rows": rows}, f)
    except Exception as exc:
        logger.debug("Cache write failed: %s", exc)


def _last_business_day() -> str:
    """Return the most recent business day (Mon-Fri) as YYYYMMDD."""
    d = datetime.now().date()
    while d.weekday() >= 5:  # Saturday=5, Sunday=6
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")
