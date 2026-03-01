"""
KRX index price data loader.

Wraps pykrx with chunked fetching, retry/backoff, and LIVE/CACHED/SAMPLE fallback.
R11 ??public loader returns LoaderResult = tuple[DataStatus, pd.DataFrame].
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd

from src.data_sources.krx_openapi import (
    KRXProvider,
    fetch_index_ohlcv_openapi,
    get_krx_openapi_key,
    get_krx_provider,
)
from src.data_sources.pykrx_compat import (
    ensure_pykrx_transport_compat,
    resolve_ohlcv_close_column,
)

logger = logging.getLogger(__name__)

# Type aliases (R11)
DataStatus = Literal["LIVE", "CACHED", "SAMPLE"]
LoaderResult = tuple[DataStatus, pd.DataFrame]

# Monkeypatch-friendly module-level constants (R10)
RAW_DIR = Path("data/raw/krx")
CURATED_DIR = Path("data/curated")

# Retry policy (R11)
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
BACKOFF_BASE = 2  # waits 2, 4, 8 seconds

# Stale/index migration policy (2026-02-22).
STALE_INDEX_CODE_REPLACEMENTS: dict[str, str] = {
    "5041": "5049",
    "1166": "1157",
}
INDEX_TICKER_MARKETS: tuple[str, ...] = ("KRX", "KOSPI", "KOSDAQ", "테마")


def _resolve_provider_mode() -> KRXProvider:
    """Resolve effective provider mode for current process."""
    configured = get_krx_provider()
    if configured == "AUTO":
        return "OPENAPI" if get_krx_openapi_key() else "PYKRX"
    return configured


def _is_deterministic_pykrx_failure(exc: Exception) -> bool:
    """Return True for pykrx failure modes where retry/backoff is not useful."""
    if isinstance(exc, KeyError):
        key_text = str(exc.args[0]).strip() if exc.args else ""
        if key_text.isdigit():
            return True

    message = str(exc).lower()
    return (
        "line 1 column 1 (char 0)" in message
        or "logout" in message
        or "지수명" in message
    )


def _normalize_requested_codes(
    index_codes: list[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Apply stale-code replacements and remove duplicates (stable order)."""
    normalized: list[str] = []
    replacements: list[tuple[str, str]] = []
    seen_codes: set[str] = set()
    seen_replacements: set[tuple[str, str]] = set()

    for raw_code in index_codes:
        code = str(raw_code)
        mapped = STALE_INDEX_CODE_REPLACEMENTS.get(code, code)
        replacement_pair = (code, mapped)
        if code != mapped and replacement_pair not in seen_replacements:
            replacements.append(replacement_pair)
            seen_replacements.add(replacement_pair)
        if mapped not in seen_codes:
            normalized.append(mapped)
            seen_codes.add(mapped)

    return normalized, replacements


@lru_cache(maxsize=1)
def _get_index_universe() -> frozenset[str]:
    """Fetch and cache pykrx index ticker universe across markets.

    Raises ValueError if the universe is empty or pykrx fails, so that
    lru_cache does NOT persist the failure (lru_cache only caches return
    values, not exceptions).
    """
    ensure_pykrx_transport_compat()
    from pykrx import stock  # type: ignore[import]

    universe: set[str] = set()
    try:
        for market in INDEX_TICKER_MARKETS:
            tickers = stock.get_index_ticker_list(market=market)
            universe.update(str(code) for code in tickers)
    except Exception as exc:
        # IndexTicker singleton may have an empty df due to KRX server changes.
        # Reset it now so the next process restart can re-initialise cleanly.
        from src.data_sources.pykrx_compat import _reset_index_ticker_singleton
        _reset_index_ticker_singleton()
        raise ValueError(f"pykrx index ticker list failed: {exc}") from exc

    if not universe:
        from src.data_sources.pykrx_compat import _reset_index_ticker_singleton
        _reset_index_ticker_singleton()
        raise ValueError("pykrx returned empty index ticker universe")

    return frozenset(universe)


def _filter_supported_codes(index_codes: list[str]) -> tuple[list[str], list[str]]:
    """Return (supported, skipped) based on pykrx universe.

    If universe lookup fails, return unfiltered codes and no skipped list to keep
    behavior conservative.
    """
    if not index_codes:
        return ([], [])

    try:
        universe = _get_index_universe()
    except Exception as exc:
        logger.warning(
            "KRX index universe lookup failed: %s (falling back to unfiltered codes)",
            exc,
        )
        return (index_codes, [])

    supported = [code for code in index_codes if code in universe]
    skipped = [code for code in index_codes if code not in universe]
    return (supported, skipped)


def _filter_supported_codes_openapi(index_codes: list[str]) -> tuple[list[str], list[str]]:
    """OpenAPI path currently does not expose a cached universe lookup."""
    return (list(index_codes), [])


def _fetch_chunk(index_code: str, start: str, end: str) -> pd.DataFrame:
    """Fetch a single chunk from pykrx with retry/backoff."""
    ensure_pykrx_transport_compat()
    from pykrx import stock  # type: ignore[import]

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            df = stock.get_index_ohlcv(start, end, index_code, name_display=False)
            if df is None:
                raise ValueError(f"pykrx returned None for {index_code}")
            return df
        except Exception as exc:
            last_exc = exc
            is_deterministic = _is_deterministic_pykrx_failure(exc)
            if is_deterministic:
                logger.warning(
                    "pykrx fetch attempt %d/%d failed for %s: %s (deterministic failure, no retry)",
                    attempt + 1,
                    MAX_RETRIES,
                    index_code,
                    exc,
                )
                break

            logger.warning(
                "pykrx fetch attempt %d/%d failed for %s: %s",
                attempt + 1,
                MAX_RETRIES,
                index_code,
                exc,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** (attempt + 1))

    assert last_exc is not None
    raise last_exc


def fetch_index_ohlcv(
    index_code: str,
    start: str,
    end: str,
    chunk_years: int = 2,
) -> pd.DataFrame:
    """Fetch index OHLCV from pykrx in 2-year chunks.

    Args:
        index_code: KRX index code (e.g. "1001" for KOSPI).
        start: Start date in YYYYMMDD format.
        end: End date in YYYYMMDD format.
        chunk_years: Max years per API request (default 2).

    Returns:
        DataFrame with DatetimeIndex and OHLCV columns.
    """
    start_dt = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
    end_dt = date(int(end[:4]), int(end[4:6]), int(end[6:8]))

    chunks: list[pd.DataFrame] = []
    chunk_start = start_dt

    while chunk_start <= end_dt:
        chunk_end = min(
            chunk_start + timedelta(days=365 * chunk_years),
            end_dt,
        )
        df = _fetch_chunk(
            index_code,
            chunk_start.strftime("%Y%m%d"),
            chunk_end.strftime("%Y%m%d"),
        )
        if not df.empty:
            chunks.append(df)

        chunk_start = chunk_end + timedelta(days=1)

    if not chunks:
        return pd.DataFrame()

    combined = pd.concat(chunks)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)

    # Save raw parquet
    raw_path = RAW_DIR / index_code
    raw_path.mkdir(parents=True, exist_ok=True)
    raw_file = raw_path / f"{end}.parquet"
    combined.to_parquet(raw_file)

    return combined



def _make_sample_df(index_codes: list[str]) -> pd.DataFrame:
    """Generate synthetic sector price data for SAMPLE mode."""
    import numpy as np

    dates = pd.date_range(end=date.today(), periods=252 * 3, freq="B")
    rows = []
    rng = np.random.default_rng(42)
    for code in index_codes:
        prices = 1000.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates)))
        for d, p in zip(dates, prices):
            rows.append(
                {
                    "index_code": code,
                    "index_name": f"Sample-{code}",
                    "close": float(p),
                }
            )
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex([r for code in index_codes for r in dates])
    # Rebuild as multi-index friendly wide form for simplicity
    wide = pd.DataFrame(index=dates)
    for code in index_codes:
        prices = 1000.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates)))
        wide[code] = prices
    # Return long form matching schema
    long = wide.stack().reset_index()
    long.columns = pd.Index(["date", "index_code", "close"])
    long["index_name"] = long["index_code"].apply(lambda c: f"Sample-{c}")
    long = long.set_index("date")
    long.index = pd.DatetimeIndex(long.index)
    return long[["index_code", "index_name", "close"]].astype(
        {"close": "float64"}
    )


def _load_from_raw_cache(code: str, start: str, end: str) -> pd.DataFrame:
    """Load the most recent raw parquet for *code* and filter to [start, end].

    Returns an empty DataFrame if no raw cache exists or loading fails.
    """
    raw_code_dir = RAW_DIR / code
    if not raw_code_dir.exists():
        return pd.DataFrame()
    parquet_files = sorted(raw_code_dir.glob("*.parquet"))
    if not parquet_files:
        return pd.DataFrame()
    try:
        df = pd.read_parquet(parquet_files[-1])
        df.index = pd.DatetimeIndex(df.index)
        df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        return df
    except Exception as exc:
        logger.warning("Raw cache load failed for %s: %s", code, exc)
        return pd.DataFrame()


def _save_raw_cache(code: str, frame: pd.DataFrame, end: str) -> None:
    """Persist raw frame to per-code parquet cache."""
    raw_path = RAW_DIR / code
    raw_path.mkdir(parents=True, exist_ok=True)
    raw_file = raw_path / f"{end}.parquet"
    frame.to_parquet(raw_file)


def load_sector_prices(
    index_codes: list[str],
    start: str,
    end: str,
) -> LoaderResult:
    """Load sector price data with LIVE/CACHED/SAMPLE fallback.

    Args:
        index_codes: List of KRX index codes.
        start: Start date YYYYMMDD.
        end: End date YYYYMMDD.

    Returns:
        LoaderResult = (DataStatus, DataFrame).
        DataFrame schema: DatetimeIndex, columns [index_code, index_name, close].
    """
    curated_path = CURATED_DIR / "sector_prices.parquet"
    provider_mode = _resolve_provider_mode()
    normalized_codes, replacements = _normalize_requested_codes(index_codes)
    if replacements:
        replacement_summary = ", ".join(f"{src}->{dst}" for src, dst in replacements)
        logger.warning("Applied KRX index code replacements: %s", replacement_summary)

    if provider_mode == "PYKRX":
        live_codes, skipped_codes = _filter_supported_codes(normalized_codes)
    else:
        live_codes, skipped_codes = _filter_supported_codes_openapi(normalized_codes)

    if skipped_codes:
        skipped_summary = ", ".join(skipped_codes)
        logger.warning(
            "Skipping unsupported KRX index codes before live fetch: %s",
            skipped_summary,
        )

    # Try LIVE fetch (partial-success tolerant)
    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    if provider_mode == "OPENAPI" and not get_krx_openapi_key():
        logger.warning(
            "KRX provider is OPENAPI but KRX_OPENAPI_KEY is missing; falling back to cache."
        )
        failures = [(code, "KRX_OPENAPI_KEY not configured") for code in live_codes]
    else:
        for code in live_codes:
            try:
                if provider_mode == "OPENAPI":
                    df = fetch_index_ohlcv_openapi(code, start, end)
                else:
                    df = fetch_index_ohlcv(code, start, end)

                if df.empty:
                    failures.append((code, "empty response"))
                    logger.warning("Empty response for index %s", code)
                    continue

                if provider_mode == "OPENAPI":
                    try:
                        _save_raw_cache(code, df, end)
                    except Exception as cache_exc:
                        logger.warning("OpenAPI raw cache save failed for %s: %s", code, cache_exc)

                close_col = resolve_ohlcv_close_column(df)
                tmp = df[[close_col]].copy()
                tmp.columns = pd.Index(["close"])
                tmp["index_code"] = code
                tmp["index_name"] = code
                tmp["close"] = tmp["close"].astype(float)
                frames.append(tmp[["index_code", "index_name", "close"]])
            except Exception as exc:
                failures.append((code, str(exc)))
                logger.warning("Live fetch failed for index %s (%s): %s", code, provider_mode, exc)
                continue

    try:
        if frames:
            result = pd.concat(frames)
            result.index = pd.DatetimeIndex(result.index)
            CURATED_DIR.mkdir(parents=True, exist_ok=True)

            from src.contracts.validators import normalize_then_validate
            result = normalize_then_validate(result, "sector_prices")
            result.to_parquet(curated_path)

            if failures:
                failed_codes = ", ".join(code for code, _ in failures)
                logger.warning(
                    "KRX partial success: %d codes loaded, %d failed (%s)",
                    len(frames),
                    len(failures),
                    failed_codes,
                )
            return ("LIVE", result)

    except Exception as exc:
        logger.warning("Live fetch aggregation failed: %s", exc)

    if not frames and failures:
        if provider_mode == "OPENAPI":
            logger.warning(
                "OPENAPI live fetch failed for all requested codes: %s"
                " (falling back to cache)",
                failures,
            )
        else:
            logger.warning(
                "PYKRX live fetch failed for all requested codes: %s"
                " (KRX may require authentication — consider OPENAPI provider; falling back to cache)",
                failures,
            )

    # ── RAW CACHE pass (if live produced nothing) ─────────────────────────────
    if not frames and failures:
        raw_frames: list[pd.DataFrame] = []
        for code, _reason in failures:
            raw_df = _load_from_raw_cache(code, start, end)
            if raw_df.empty:
                continue
            try:
                close_col = resolve_ohlcv_close_column(raw_df)
                tmp = raw_df[[close_col]].copy()
                tmp.columns = pd.Index(["close"])
                tmp["index_code"] = code
                tmp["index_name"] = code
                tmp["close"] = tmp["close"].astype(float)
                raw_frames.append(tmp[["index_code", "index_name", "close"]])
            except Exception as exc:
                logger.warning("Raw cache column resolution failed for %s: %s", code, exc)

        if raw_frames:
            try:
                result = pd.concat(raw_frames)
                result.index = pd.DatetimeIndex(result.index)
                CURATED_DIR.mkdir(parents=True, exist_ok=True)
                from src.contracts.validators import normalize_then_validate
                result = normalize_then_validate(result, "sector_prices")
                result.to_parquet(curated_path)
                logger.info(
                    "Loaded %d codes from raw cache (%s unavailable)",
                    len(raw_frames),
                    provider_mode,
                )
                return ("CACHED", result)
            except Exception as exc:
                logger.error("Raw cache aggregation failed: %s", exc)

    # Try CACHED
    if curated_path.exists():
        try:
            cached = pd.read_parquet(curated_path)
            cached.index = pd.DatetimeIndex(cached.index)
            logger.info("Loaded sector prices from cache: %s", curated_path)
            return ("CACHED", cached)
        except Exception as exc:
            logger.error("Cache load failed: %s", exc)

    # SAMPLE fallback
    logger.warning("Using SAMPLE data for sector prices")
    sample = _make_sample_df(normalized_codes)
    return ("SAMPLE", sample)
