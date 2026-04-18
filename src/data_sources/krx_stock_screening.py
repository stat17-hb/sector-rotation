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

from src.data_sources.krx_constituents import candidate_reference_dates, lookup_index_constituents

logger = logging.getLogger(__name__)

DataStatus = Literal["LIVE", "CACHED", "UNAVAILABLE"]
CACHE_PATH = Path("data/curated/stock_screening_cache.pkl")
ETF_CONTEXT_CACHE_PATH = Path("data/curated/stock_screening_etf_context_cache.pkl")
CACHE_TTL_HOURS = 24
MAX_STOCKS_PER_SECTOR = 15
ETF_LIQUIDITY_LOOKBACK_DAYS = 20
ETF_HISTORY_BUFFER_DAYS = 45
ETF_MIN_AVG_TRADING_VALUE = 300_000_000


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


def load_representative_etf_context(
    strong_buy_sectors: list[dict],
    etf_map: dict[str, list[dict]] | None,
    settings: dict | None = None,
    force_refresh: bool = False,
) -> tuple[DataStatus, list[dict]]:
    """Load representative ETF execution context for Strong Buy sectors.

    The result is intentionally execution-support only. It must not affect
    sector ranking, action, or stock-screening results.
    """
    if not strong_buy_sectors:
        return "UNAVAILABLE", []

    normalized_etf_map = {
        str(code): [dict(item) for item in items]
        for code, items in (etf_map or {}).items()
    }

    if not force_refresh:
        cached = _read_etf_context_cache(strong_buy_sectors, normalized_etf_map)
        if cached is not None:
            return "CACHED", cached

    try:
        rows = _fetch_representative_etf_context(
            strong_buy_sectors=strong_buy_sectors,
            etf_map=normalized_etf_map,
            settings=settings or {},
        )
        if rows:
            _write_etf_context_cache(strong_buy_sectors, normalized_etf_map, rows)
            return "LIVE", rows
        return "UNAVAILABLE", []
    except Exception as exc:
        logger.warning("Representative ETF context load failed: %s", exc)
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


def _fetch_representative_etf_context(
    *,
    strong_buy_sectors: list[dict],
    etf_map: dict[str, list[dict]],
    settings: dict,
) -> list[dict]:
    import pykrx.stock as stock
    from pykrx.website.krx.etx.core import 전종목시세_ETF

    from src.data_sources.pykrx_compat import ensure_pykrx_transport_compat

    ensure_pykrx_transport_compat()

    target_date = _last_business_day()
    candidate_dates = candidate_reference_dates(target_date, periods=5)
    min_avg_trading_value = int(
        settings.get("screening_etf_min_avg_trading_value", ETF_MIN_AVG_TRADING_VALUE)
    )
    lookback_days = int(
        settings.get("screening_etf_liquidity_lookback_days", ETF_LIQUIDITY_LOOKBACK_DAYS)
    )

    snapshot_df, snapshot_date = _fetch_etf_snapshot(candidate_dates, 전종목시세_ETF)
    rows: list[dict] = []

    for sector in strong_buy_sectors:
        sector_code = str(sector.get("code", "")).strip()
        sector_name = str(sector.get("name", "")).strip() or sector_code
        mapped_etfs = [dict(item) for item in etf_map.get(sector_code, [])]
        rows.append(
            _build_sector_etf_context_row(
                stock_module=stock,
                sector_code=sector_code,
                sector_name=sector_name,
                mapped_etfs=mapped_etfs,
                snapshot_df=snapshot_df,
                snapshot_date=snapshot_date,
                target_date=target_date,
                lookback_days=lookback_days,
                min_avg_trading_value=min_avg_trading_value,
            )
        )

    return rows


def _fetch_etf_snapshot(candidate_dates: list[str], fetcher_cls) -> tuple[pd.DataFrame, str]:
    fetcher = fetcher_cls()
    for candidate_date in candidate_dates:
        try:
            raw = fetcher.fetch(candidate_date)
        except Exception as exc:
            logger.debug("ETF snapshot fetch failed for %s: %s", candidate_date, exc)
            continue
        normalized = _normalize_etf_snapshot(raw)
        if not normalized.empty:
            return normalized, candidate_date
    return pd.DataFrame(), ""


def _normalize_etf_snapshot(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    required = {"ISU_SRT_CD", "ISU_ABBRV", "NAV", "ACC_TRDVAL", "MKTCAP", "INVSTASST_NETASST_TOTAMT"}
    if not required.issubset(set(raw.columns)):
        return pd.DataFrame()

    selected_columns = [
        "ISU_SRT_CD",
        "ISU_ABBRV",
        "NAV",
        "ACC_TRDVAL",
        "MKTCAP",
        "INVSTASST_NETASST_TOTAMT",
    ]
    if "LIST_SHRS" in raw.columns:
        selected_columns.append("LIST_SHRS")

    frame = raw[selected_columns].copy()
    frame = frame.rename(
        columns={
            "ISU_SRT_CD": "ticker",
            "ISU_ABBRV": "name",
            "NAV": "nav",
            "ACC_TRDVAL": "latest_trade_value",
            "MKTCAP": "market_cap",
            "INVSTASST_NETASST_TOTAMT": "net_assets",
            "LIST_SHRS": "listed_shares",
        }
    )
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame["name"] = frame["name"].astype(str).str.strip()
    if "listed_shares" not in frame.columns:
        frame["listed_shares"] = pd.NA

    for column in ("nav", "latest_trade_value", "market_cap", "net_assets", "listed_shares"):
        frame[column] = _to_numeric(frame[column])
    frame = frame.set_index("ticker")
    return frame


def _build_sector_etf_context_row(
    *,
    stock_module,
    sector_code: str,
    sector_name: str,
    mapped_etfs: list[dict],
    snapshot_df: pd.DataFrame,
    snapshot_date: str,
    target_date: str,
    lookback_days: int,
    min_avg_trading_value: int,
) -> dict:
    if not mapped_etfs:
        return {
            "sector_code": sector_code,
            "sector_name": sector_name,
            "etf_code": "",
            "etf_name": "—",
            "style_tags": "없음",
            "execution_state": "대표 ETF 없음",
            "latest_trade_value": None,
            "avg_trade_value_20d": None,
            "net_assets": None,
            "nav": None,
            "reference_date": "",
            "freshness_label": "매핑 없음",
            "note": "ETF 매핑이 없는 섹터입니다.",
        }

    candidates: list[dict] = []
    for order, etf in enumerate(mapped_etfs):
        ticker = str(etf.get("code", "")).strip()
        etf_name = str(etf.get("name", "")).strip() or ticker
        snapshot_row = snapshot_df.loc[ticker] if ticker and ticker in snapshot_df.index else None
        history = _fetch_etf_history(
            stock_module=stock_module,
            ticker=ticker,
            target_date=target_date,
            lookback_days=lookback_days,
        )
        latest_history_date = ""
        avg_trade_value_20d = None
        latest_trade_value = None
        nav_value = (
            float(snapshot_row.get("nav"))
            if snapshot_row is not None and pd.notna(snapshot_row.get("nav"))
            else None
        )
        net_assets = (
            float(snapshot_row.get("net_assets"))
            if snapshot_row is not None and pd.notna(snapshot_row.get("net_assets"))
            else None
        )

        if history is not None and not history.empty:
            history = history.sort_index()
            latest_history_date = history.index.max().strftime("%Y%m%d")
            latest_trade_value = float(history["거래대금"].iloc[-1])
            avg_trade_value_20d = float(history["거래대금"].tail(lookback_days).mean())
            if nav_value is None and "NAV" in history.columns and not history["NAV"].empty:
                nav_value = float(history["NAV"].iloc[-1])

        if (
            latest_trade_value is None
            and snapshot_row is not None
            and pd.notna(snapshot_row.get("latest_trade_value"))
        ):
            latest_trade_value = float(snapshot_row.get("latest_trade_value"))

        candidates.append(
            {
                "mapping_order": order,
                "etf_code": ticker,
                "etf_name": etf_name,
                "style_tags": _infer_etf_style_tags(etf_name),
                "latest_trade_value": latest_trade_value,
                "avg_trade_value_20d": avg_trade_value_20d,
                "net_assets": net_assets,
                "nav": nav_value,
                "reference_date": latest_history_date or (snapshot_date if snapshot_row is not None else ""),
                "liquidity_ok": (
                    avg_trade_value_20d is not None and avg_trade_value_20d >= float(min_avg_trading_value)
                ),
            }
        )

    chosen = sorted(
        candidates,
        key=lambda row: (
            1 if row["liquidity_ok"] else 0,
            float(row["latest_trade_value"] or 0.0),
            float(row["net_assets"] or 0.0),
            -int(row["mapping_order"]),
        ),
        reverse=True,
    )[0]

    reference_date = str(chosen.get("reference_date", "") or "")
    is_stale = bool(reference_date) and reference_date != target_date
    if not reference_date:
        freshness_label = "기준시각 불명"
    elif is_stale:
        freshness_label = f"저신뢰 · {reference_date}"
    else:
        freshness_label = reference_date

    if not chosen["liquidity_ok"]:
        execution_state = "실행 유동성 부족"
    elif not reference_date:
        execution_state = "기준시각 불명"
    else:
        execution_state = "정상"

    note_parts: list[str] = []
    if execution_state == "실행 유동성 부족":
        note_parts.append("20일 평균 거래대금 기준 미달")
    if not reference_date:
        note_parts.append("최근 기준일 확인 불가")
    elif is_stale:
        note_parts.append("최신 영업일 기준이 아님")

    return {
        "sector_code": sector_code,
        "sector_name": sector_name,
        "etf_code": str(chosen.get("etf_code", "")),
        "etf_name": str(chosen.get("etf_name", "")),
        "style_tags": str(chosen.get("style_tags", "")),
        "execution_state": execution_state,
        "latest_trade_value": chosen.get("latest_trade_value"),
        "avg_trade_value_20d": chosen.get("avg_trade_value_20d"),
        "net_assets": chosen.get("net_assets"),
        "nav": chosen.get("nav"),
        "reference_date": reference_date,
        "freshness_label": freshness_label,
        "note": " · ".join(note_parts),
    }


def _fetch_etf_history(*, stock_module, ticker: str, target_date: str, lookback_days: int) -> pd.DataFrame:
    try:
        end_ts = pd.Timestamp(target_date).normalize()
        start_ts = pd.bdate_range(end=end_ts, periods=max(lookback_days * 2, ETF_HISTORY_BUFFER_DAYS))[0]
        history = stock_module.get_etf_ohlcv_by_date(start_ts.strftime("%Y%m%d"), target_date, ticker)
    except Exception as exc:
        logger.debug("ETF history fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()
    if history is None or history.empty:
        return pd.DataFrame()
    return history.sort_index()


def _infer_etf_style_tags(name: str) -> str:
    text = str(name or "").strip()
    tags: list[str] = []
    for needle, tag in (
        ("고배당", "배당"),
        ("배당", "배당"),
        ("TR", "TR"),
        ("액티브", "액티브"),
        ("TOP10", "TOP10"),
        ("레버리지", "레버리지"),
        ("인버스", "인버스"),
    ):
        if needle in text and tag not in tags:
            tags.append(tag)
    if not tags:
        tags.append("일반")
    return ", ".join(tags[:3])


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.replace("-", "0", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _get_constituents(stock_module, trade_date: str, sector_code: str) -> list[str]:
    """Get constituent ticker codes for a sector index."""
    lookup = lookup_index_constituents(
        stock_module,
        sector_code=sector_code,
        candidate_dates=candidate_reference_dates(trade_date, periods=5),
        logger=logger,
    )
    return list(lookup.tickers)


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


def _etf_context_cache_key(sectors: list[dict], etf_map: dict[str, list[dict]]) -> str:
    items: list[str] = []
    for sector in sorted(sectors, key=lambda item: str(item.get("code", ""))):
        sector_code = str(sector.get("code", "")).strip()
        etf_codes = ",".join(str(item.get("code", "")).strip() for item in etf_map.get(sector_code, []))
        items.append(f"{sector_code}:{etf_codes}")
    return "|".join(items)


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


def _read_etf_context_cache(sectors: list[dict], etf_map: dict[str, list[dict]]) -> list[dict] | None:
    if not ETF_CONTEXT_CACHE_PATH.exists():
        return None
    try:
        with open(ETF_CONTEXT_CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        if cached.get("key") != _etf_context_cache_key(sectors, etf_map):
            return None
        age_hours = (datetime.now() - cached["ts"]).total_seconds() / 3600
        if age_hours > CACHE_TTL_HOURS:
            return None
        return cached["rows"]
    except Exception:
        return None


def _write_etf_context_cache(
    sectors: list[dict],
    etf_map: dict[str, list[dict]],
    rows: list[dict],
) -> None:
    try:
        ETF_CONTEXT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ETF_CONTEXT_CACHE_PATH, "wb") as f:
            pickle.dump(
                {
                    "key": _etf_context_cache_key(sectors, etf_map),
                    "ts": datetime.now(),
                    "rows": rows,
                },
                f,
            )
    except Exception as exc:
        logger.debug("ETF context cache write failed: %s", exc)


def _last_business_day() -> str:
    """Return the most recent business day (Mon-Fri) as YYYYMMDD."""
    d = datetime.now().date()
    while d.weekday() >= 5:  # Saturday=5, Sunday=6
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")
