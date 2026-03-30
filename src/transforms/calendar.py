"""
KRX business day calendar helpers.

OPENAPI is preferred when the runtime provider is OPENAPI with a configured
key. pykrx remains the fallback before weekend-only subtraction.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

from src.data_sources.krx_openapi import (
    fetch_index_ohlcv_openapi,
    get_krx_openapi_key,
    get_krx_provider,
)
from src.data_sources.pykrx_compat import ensure_pykrx_transport_compat

logger = logging.getLogger(__name__)

OPENAPI_LOOKBACK_WEEKDAYS = 10
PYKRX_LOOKBACK_DAYS = 14
YFINANCE_LOOKBACK_DAYS = 14


def _resolve_calendar_provider(provider: str | None = None) -> str:
    """Resolve the effective market-data provider for calendar lookup."""
    configured = get_krx_provider(provider)
    if configured == "AUTO":
        return "OPENAPI" if get_krx_openapi_key() else "PYKRX"
    return configured


def _weekday_candidates(ref: date, limit: int) -> list[date]:
    """Return up to *limit* weekdays in reverse order starting at *ref*."""
    candidates: list[date] = []
    current = ref
    scanned_days = 0
    while len(candidates) < limit and scanned_days < limit * 3:
        if current.weekday() < 5:
            candidates.append(current)
        current -= timedelta(days=1)
        scanned_days += 1
    return candidates


def _get_last_business_day_openapi(ref: date, benchmark_code: str) -> date:
    """Probe recent daily snapshots until one trading day returns rows."""
    key = get_krx_openapi_key()
    if not key:
        raise ValueError("KRX_OPENAPI_KEY not configured")

    last_exc: Exception | None = None
    for candidate in _weekday_candidates(ref, OPENAPI_LOOKBACK_WEEKDAYS):
        bas_dd = candidate.strftime("%Y%m%d")
        try:
            df = fetch_index_ohlcv_openapi(
                str(benchmark_code).strip() or "1001",
                bas_dd,
                bas_dd,
                auth_key=key,
            )
        except Exception as exc:
            last_exc = exc
            logger.debug("KRX calendar OPENAPI probe failed for %s: %s", bas_dd, exc)
            continue
        if df is not None and not df.empty:
            return df.index[-1].date()

    if last_exc is not None:
        raise last_exc
    raise ValueError("KRX OpenAPI calendar probe found no trading day")


def _get_last_business_day_pykrx(ref: date, benchmark_code: str) -> date:
    """Fetch recent benchmark history via pykrx and return the latest date."""
    start = (ref - timedelta(days=PYKRX_LOOKBACK_DAYS)).strftime("%Y%m%d")
    end = ref.strftime("%Y%m%d")

    ensure_pykrx_transport_compat()
    from pykrx import stock  # type: ignore[import]

    df = stock.get_index_ohlcv(start, end, str(benchmark_code).strip() or "1001", name_display=False)
    if df is None or df.empty:
        raise ValueError("Empty OHLCV response from pykrx")
    return df.index[-1].date()


def _get_last_business_day_yfinance(ref: date, benchmark_code: str) -> date:
    """Fetch recent ETF history via yfinance and return the latest trading day."""
    start = (ref - timedelta(days=YFINANCE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = (ref + timedelta(days=1)).strftime("%Y-%m-%d")

    import yfinance as yf  # type: ignore[import]

    df = yf.download(
        tickers=str(benchmark_code).strip() or "SPY",
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if df is None or df.empty:
        raise ValueError("Empty OHLCV response from yfinance")
    return df.index[-1].date()


def _weekend_only_fallback(ref: date) -> date:
    """Return the previous weekday without KRX holiday awareness."""
    d = ref
    for _ in range(10):
        d -= timedelta(days=1)
        if d.weekday() < 5:
            return d
    return ref - timedelta(days=1)


def _is_us_calendar_lookup(provider: str | None, benchmark_code: str) -> bool:
    provider_token = str(provider or "").strip().upper()
    if provider_token in {"YFINANCE", "US"}:
        return True
    code = str(benchmark_code or "").strip().upper()
    return bool(code and not code.isdigit())


def get_last_business_day(
    as_of: date | None = None,
    provider: str | None = None,
    benchmark_code: str = "1001",
) -> date:
    """Return the last KRX trading day on or before *as_of*.

    Priority:
      1. Yfinance probe for US ETF benchmarks
      2. OPENAPI daily snapshot probe when provider resolves to OPENAPI
      3. pykrx benchmark OHLCV lookup
      4. Weekend-only subtraction
    """
    ref = as_of or date.today()
    if _is_us_calendar_lookup(provider, benchmark_code):
        try:
            return _get_last_business_day_yfinance(ref, benchmark_code)
        except Exception as exc:
            logger.warning(
                "US market calendar lookup failed (%s); using weekend-only fallback.",
                exc,
            )
            return _weekend_only_fallback(ref)

    resolved_provider = _resolve_calendar_provider(provider)
    errors: list[str] = []

    if resolved_provider == "OPENAPI" and get_krx_openapi_key():
        try:
            return _get_last_business_day_openapi(ref, benchmark_code)
        except Exception as exc:
            errors.append(f"openapi={exc}")
            logger.debug("KRX calendar OPENAPI lookup failed; trying pykrx: %s", exc)

    try:
        return _get_last_business_day_pykrx(ref, benchmark_code)
    except Exception as exc:
        errors.append(f"pykrx={exc}")

    logger.warning(
        "KRX calendar lookup failed (%s); using weekend-only fallback "
        "(no KRX holiday awareness).",
        "; ".join(errors) or "no provider available",
    )
    return _weekend_only_fallback(ref)
