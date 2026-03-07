"""
KRX index price data loader.

Wraps pykrx with chunked fetching, retry/backoff, and LIVE/CACHED/SAMPLE fallback.
R11 ??public loader returns LoaderResult = tuple[DataStatus, pd.DataFrame].
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.data_sources.krx_openapi import (
    KRXOpenAPIAccessDeniedError,
    KRXProvider,
    fetch_index_ohlcv_openapi_batch_detailed,
    get_index_display_name,
    get_krx_openapi_key,
    get_krx_provider,
    reset_krx_openapi_health_cache,
    resolve_openapi_api_id,
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
WARM_STATUS_FILE = "_warm_status.json"
STALE_CACHE_TOLERANCE_BUSINESS_DAYS = 1
BACKGROUND_WARM_KEY = "default"
RAW_CACHE_CONTAMINATION_WINDOW = 60
INTERACTIVE_OPENAPI_REQUEST_LIMIT = 60

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
_BACKGROUND_WARM_LOCK = threading.Lock()
_BACKGROUND_WARM_THREADS: dict[str, threading.Thread] = {}


class KRXMarketDataError(RuntimeError):
    """Base fail-fast error for interactive market-data loads."""


class KRXInteractiveRangeLimitError(KRXMarketDataError):
    """Raised when an interactive OPENAPI request would exceed the request budget."""


class KRXMarketDataAccessDeniedError(KRXMarketDataError):
    """Raised when KRX OpenAPI denies interactive market-data access."""


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


def _business_gap_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Return the number of business-day steps between two timestamps."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts <= start_ts:
        return 0
    return max(0, len(pd.date_range(start_ts, end_ts, freq="B")) - 1)


def _warm_status_path() -> Path:
    """Return the JSON status file path for background/manual warms."""
    return RAW_DIR / WARM_STATUS_FILE


def _business_days_in_range(start: str, end: str) -> list[pd.Timestamp]:
    """Return business-day timestamps in the inclusive request range."""
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="B"))


def _predict_openapi_requests(index_codes: list[str], start: str, end: str) -> int:
    """Estimate the number of OpenAPI snapshot requests for one range."""
    codes = [str(code).strip() for code in index_codes if str(code).strip()]
    if not codes:
        return 0
    families = {resolve_openapi_api_id(code) for code in codes}
    return len(_business_days_in_range(start, end)) * len(families)


def _cap_openapi_interactive_range(
    index_codes: list[str],
    start: str,
    end: str,
    *,
    request_limit: int = INTERACTIVE_OPENAPI_REQUEST_LIMIT,
) -> tuple[str, str, int]:
    """Clamp an interactive OpenAPI refresh to the most recent request-budget window."""
    codes = [str(code).strip() for code in index_codes if str(code).strip()]
    predicted_requests = _predict_openapi_requests(codes, start, end)
    if predicted_requests <= request_limit:
        return start, end, predicted_requests

    family_count = max(1, len({resolve_openapi_api_id(code) for code in codes}))
    max_business_days = max(1, request_limit // family_count)
    business_days = _business_days_in_range(start, end)
    if not business_days:
        return start, end, predicted_requests

    capped_days = business_days[-max_business_days:]
    capped_start = capped_days[0].strftime("%Y%m%d")
    capped_end = capped_days[-1].strftime("%Y%m%d")
    return capped_start, capped_end, len(capped_days) * family_count


def _is_access_denied_reason(detail: str) -> bool:
    """Return True when a failure detail clearly indicates KRX edge denial."""
    return "access denied" in str(detail or "").lower()


def _summary_has_access_denied(summary: dict[str, Any]) -> bool:
    """Return True when a warm summary indicates OpenAPI edge denial."""
    if str(summary.get("abort_reason", "")).strip().upper() == "ACCESS_DENIED":
        return True
    if any(_is_access_denied_reason(detail) for detail in dict(summary.get("failed_codes") or {}).values()):
        return True
    snapshot_failures = dict(summary.get("snapshot_failures") or {})
    return any(
        _is_access_denied_reason(detail)
        for family_failures in snapshot_failures.values()
        if isinstance(family_failures, dict)
        for detail in family_failures.values()
    )


def _access_denied_detail(summary: dict[str, Any]) -> str:
    """Extract one representative access-denied message from a warm summary."""
    for detail in dict(summary.get("failed_codes") or {}).values():
        if _is_access_denied_reason(detail):
            return str(detail)
    snapshot_failures = dict(summary.get("snapshot_failures") or {})
    for family_failures in snapshot_failures.values():
        if not isinstance(family_failures, dict):
            continue
        for detail in family_failures.values():
            if _is_access_denied_reason(detail):
                return str(detail)
    return "KRX OpenAPI access denied during interactive refresh"


def _write_warm_status(payload: dict[str, Any]) -> None:
    """Persist warm job status for app cache invalidation and diagnostics."""
    path = _warm_status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_warm_status() -> dict[str, Any]:
    """Return the latest warm status summary for diagnostics/UI use."""
    path = _warm_status_path()
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable warm status %s: %s", path, exc)
        return {}

    if not isinstance(payload, dict):
        return {}

    def _digits(value: Any) -> str:
        raw = "".join(ch for ch in str(value or "") if ch.isdigit())
        return raw if len(raw) == 8 else ""

    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _string_map(value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {
            str(key).strip(): str(val).strip()
            for key, val in value.items()
            if str(key).strip() and str(val).strip()
        }

    def _int(value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    return {
        "status": str(payload.get("status", "")).strip().upper(),
        "provider": str(payload.get("provider", "")).strip().upper(),
        "end": _digits(payload.get("end")),
        "coverage_complete": bool(payload.get("coverage_complete")),
        "failed_days": _string_list(payload.get("failed_days")),
        "failed_codes": _string_map(payload.get("failed_codes")),
        "reason": str(payload.get("reason", "")).strip(),
        "delta_codes": _string_list(payload.get("delta_codes")),
        "aborted": bool(payload.get("aborted")),
        "abort_reason": str(payload.get("abort_reason", "")).strip(),
        "predicted_requests": _int(payload.get("predicted_requests")),
        "processed_requests": _int(payload.get("processed_requests")),
    }


def get_price_artifact_key() -> tuple[int, int, int, int, int, int]:
    """Return a cache-busting key for curated prices + warm status artifacts."""
    def _stat_tuple(path: Path) -> tuple[int, int]:
        if not path.exists():
            return (0, 0)
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size)

    curated = CURATED_DIR / "sector_prices.parquet"
    warm_status = _warm_status_path()
    return (*_stat_tuple(curated), *_stat_tuple(warm_status), *_stat_tuple(RAW_DIR))


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
        {"index_code": "object", "index_name": "object", "close": "float64"}
    )


def _latest_raw_cache_file(code: str) -> Path | None:
    """Return the most recent raw parquet path for one index code."""
    raw_code_dir = RAW_DIR / code
    if not raw_code_dir.exists():
        return None
    parquet_files = sorted(raw_code_dir.glob("*.parquet"))
    if not parquet_files:
        return None
    return parquet_files[-1]


def _load_latest_raw_cache(code: str) -> pd.DataFrame:
    """Load the most recent raw parquet for *code*."""
    raw_file = _latest_raw_cache_file(code)
    if raw_file is None:
        return pd.DataFrame()
    try:
        df = pd.read_parquet(raw_file)
        df.index = pd.DatetimeIndex(df.index)
        df = df.sort_index()
        return df
    except Exception as exc:
        logger.warning("Raw cache load failed for %s: %s", code, exc)
        return pd.DataFrame()


def _slice_raw_cache(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Filter a raw frame to the requested date window."""
    if frame.empty:
        return pd.DataFrame()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    sliced = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()
    sliced.index = pd.DatetimeIndex(sliced.index)
    return sliced


def _load_from_raw_cache(code: str, start: str, end: str) -> pd.DataFrame:
    """Load the latest raw frame filtered to [start, end]."""
    return _slice_raw_cache(_load_latest_raw_cache(code), start, end)


def _merge_price_frames(base: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge two raw OHLCV-like frames and keep the latest value per date."""
    if base.empty:
        merged = incoming.copy()
    elif incoming.empty:
        merged = base.copy()
    else:
        merged = pd.concat([base, incoming])
    if merged.empty:
        return pd.DataFrame()
    merged.index = pd.DatetimeIndex(merged.index)
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


def _save_raw_cache(code: str, frame: pd.DataFrame, end: str) -> None:
    """Persist raw frame to per-code parquet cache."""
    raw_path = RAW_DIR / code
    raw_path.mkdir(parents=True, exist_ok=True)
    raw_file = raw_path / f"{end}.parquet"
    frame.to_parquet(raw_file)


def _save_merged_raw_cache(code: str, frame: pd.DataFrame, end: str) -> None:
    """Merge incoming data into the latest raw cache snapshot and persist it."""
    merged = _merge_price_frames(_load_latest_raw_cache(code), frame)
    if merged.empty:
        return
    _save_raw_cache(code, merged, end)


def _compute_missing_ranges(frame: pd.DataFrame, start: str, end: str) -> list[tuple[str, str]]:
    """Return missing [start, end] subranges relative to one cached frame."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if frame.empty:
        return [(start_ts.strftime("%Y%m%d"), end_ts.strftime("%Y%m%d"))]

    frame = frame.sort_index()
    earliest = pd.Timestamp(frame.index.min()).normalize()
    latest = pd.Timestamp(frame.index.max()).normalize()
    ranges: list[tuple[str, str]] = []

    if earliest > start_ts:
        older_end = earliest - timedelta(days=1)
        ranges.append((start_ts.strftime("%Y%m%d"), older_end.strftime("%Y%m%d")))
    if latest < end_ts:
        newer_start = latest + timedelta(days=1)
        ranges.append((newer_start.strftime("%Y%m%d"), end_ts.strftime("%Y%m%d")))

    return [
        (range_start, range_end)
        for range_start, range_end in ranges
        if pd.Timestamp(range_start) <= pd.Timestamp(range_end)
    ]


def _build_sector_frame(code: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a close-history frame into the sector_prices contract shape."""
    close_col = resolve_ohlcv_close_column(frame)
    result = frame[[close_col]].copy()
    result.columns = pd.Index(["close"])
    result["index_code"] = code
    result["index_name"] = get_index_display_name(code)
    result["index_code"] = result["index_code"].astype("object")
    result["index_name"] = result["index_name"].astype("object")
    result["close"] = result["close"].astype(float)
    return result[["index_code", "index_name", "close"]]


def _validate_sector_prices(result: pd.DataFrame) -> pd.DataFrame:
    """Validate the sector_prices contract before persisting/returning."""
    from src.contracts.validators import normalize_then_validate

    normalized = result.copy()
    normalized.index = pd.DatetimeIndex(normalized.index)
    for col in ("index_code", "index_name"):
        if col in normalized.columns:
            normalized[col] = normalized[col].astype("object")
    return normalize_then_validate(normalized, "sector_prices")


def _persist_curated_sector_prices(result: pd.DataFrame) -> pd.DataFrame:
    """Validate and persist the curated sector_prices parquet."""
    curated_path = CURATED_DIR / "sector_prices.parquet"
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    validated = _validate_sector_prices(result)
    validated.to_parquet(curated_path)
    return validated


def _load_curated_sector_prices() -> pd.DataFrame:
    """Load the curated sector_prices parquet if it exists."""
    curated_path = CURATED_DIR / "sector_prices.parquet"
    if not curated_path.exists():
        return pd.DataFrame()
    try:
        cached = pd.read_parquet(curated_path)
        cached.index = pd.DatetimeIndex(cached.index)
        return cached
    except Exception as exc:
        logger.error("Cache load failed: %s", exc)
        return pd.DataFrame()


def _raw_cache_signature(
    frame: pd.DataFrame,
    start: str,
    end: str,
    *,
    window: int = RAW_CACHE_CONTAMINATION_WINDOW,
) -> tuple[tuple[str, float], ...]:
    """Return a stable trailing-close signature for contamination checks."""
    sliced = _slice_raw_cache(frame, start, end)
    if sliced.empty:
        return ()
    close_col = resolve_ohlcv_close_column(sliced)
    close = sliced[close_col].astype("float64").dropna().sort_index().tail(window)
    if len(close) < window:
        return ()
    return tuple((ts.strftime("%Y%m%d"), float(value)) for ts, value in close.items())


def _detect_contaminated_raw_cache_codes(
    index_codes: list[str],
    start: str,
    end: str,
    *,
    raw_frames_by_code: dict[str, pd.DataFrame] | None = None,
    window: int = RAW_CACHE_CONTAMINATION_WINDOW,
) -> list[str]:
    """Detect codes whose trailing raw-cache close series are exactly duplicated."""
    signature_groups: dict[tuple[tuple[str, float], ...], list[str]] = {}
    for code in index_codes:
        source = (
            raw_frames_by_code.get(code, pd.DataFrame())
            if raw_frames_by_code is not None
            else _load_latest_raw_cache(code)
        )
        signature = _raw_cache_signature(source, start, end, window=window)
        if signature:
            signature_groups.setdefault(signature, []).append(code)

    contaminated: list[str] = []
    for codes in signature_groups.values():
        if len(codes) > 1:
            contaminated.extend(sorted(codes))
    return sorted(dict.fromkeys(contaminated))


def _filter_sector_price_result(
    result: pd.DataFrame,
    excluded_codes: list[str] | set[str],
) -> pd.DataFrame:
    """Return sector_prices data with selected index codes removed."""
    if result.empty or not excluded_codes:
        return result
    excluded = {str(code) for code in excluded_codes}
    if not excluded:
        return result
    filtered = result[~result["index_code"].astype(str).isin(excluded)].copy()
    filtered.index = pd.DatetimeIndex(filtered.index)
    return filtered


def _collect_raw_cache_state(
    index_codes: list[str],
    start: str,
    end: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Load raw cache slices and per-code coverage metadata."""
    frames: dict[str, pd.DataFrame] = {}
    state: dict[str, dict[str, Any]] = {}
    end_ts = pd.Timestamp(end)

    for code in index_codes:
        raw_full = _load_latest_raw_cache(code)
        raw_slice = _slice_raw_cache(raw_full, start, end)
        latest = pd.Timestamp(raw_full.index.max()).normalize() if not raw_full.empty else None
        earliest = pd.Timestamp(raw_full.index.min()).normalize() if not raw_full.empty else None
        missing_ranges = _compute_missing_ranges(raw_full, start, end)
        newer_gap_days = _business_gap_days(latest, end_ts) if latest is not None else 9999
        has_older_gap = bool(missing_ranges and missing_ranges[0][0] == pd.Timestamp(start).strftime("%Y%m%d"))

        if not raw_slice.empty:
            frames[code] = raw_slice
        state[code] = {
            "has_cache": not raw_full.empty,
            "has_slice": not raw_slice.empty,
            "latest": latest,
            "earliest": earliest,
            "missing_ranges": missing_ranges,
            "newer_gap_days": newer_gap_days,
            "has_older_gap": has_older_gap,
        }

    return frames, state


def _is_requested_coverage_complete(
    frames_by_code: dict[str, pd.DataFrame],
    index_codes: list[str],
    start: str,
    end: str,
    *,
    failures: dict[str, str] | None = None,
    failed_days: list[str] | None = None,
) -> bool:
    """Return True when the requested range is fully covered without known failures."""
    if failures:
        return False
    if failed_days:
        return False
    if not index_codes:
        return False

    for code in index_codes:
        frame = frames_by_code.get(code, pd.DataFrame())
        if _slice_raw_cache(frame, start, end).empty:
            return False
        if _compute_missing_ranges(frame, start, end):
            return False
    return True


def _build_result_from_raw_frames(
    frames_by_code: dict[str, pd.DataFrame],
    index_codes: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Build one validated sector_prices frame from raw cache frames."""
    frames: list[pd.DataFrame] = []
    for code in index_codes:
        raw_slice = _slice_raw_cache(frames_by_code.get(code, pd.DataFrame()), start, end)
        if raw_slice.empty:
            continue
        frames.append(_build_sector_frame(code, raw_slice))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


def _refresh_openapi_raw_cache(
    index_codes: list[str],
    start: str,
    end: str,
    *,
    force: bool = False,
    force_codes: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, Any]]:
    """Fetch only missing raw-cache ranges via KRX OpenAPI and merge them."""
    forced = {str(code) for code in (force_codes or set())}
    frames_by_code = {
        code: (_load_latest_raw_cache(code) if not force and code not in forced else pd.DataFrame())
        for code in index_codes
    }
    failures: dict[str, str] = {}
    range_groups: dict[tuple[str, str], list[str]] = {}
    cache_hits = 0
    refreshed_codes: list[str] = []
    failed_days: list[str] = []
    snapshot_failures: dict[str, dict[str, str]] = {}
    predicted_requests = 0
    processed_requests = 0
    aborted = False
    abort_reason = ""

    for code, frame in frames_by_code.items():
        if not frame.empty:
            cache_hits += 1
        missing_ranges = (
            [(start, end)]
            if force or code in forced
            else _compute_missing_ranges(frame, start, end)
        )
        for range_start, range_end in missing_ranges:
            range_groups.setdefault((range_start, range_end), []).append(code)

    range_summaries: list[dict[str, Any]] = []
    for (range_start, range_end), codes in sorted(range_groups.items()):
        predicted_requests += _predict_openapi_requests(codes, range_start, range_end)
        if aborted:
            break
        batch_started = time.perf_counter()
        try:
            fetched_frames, batch_failures, batch_details = fetch_index_ohlcv_openapi_batch_detailed(
                codes,
                range_start,
                range_end,
                force=force,
            )
        except Exception as exc:
            detail = str(exc)
            for code in codes:
                failures[code] = detail
            if isinstance(exc, KRXOpenAPIAccessDeniedError):
                aborted = True
                abort_reason = "ACCESS_DENIED"
            logger.warning(
                "OPENAPI delta fetch failed for range %s~%s (%s): %s",
                range_start,
                range_end,
                codes,
                detail,
            )
            range_summaries.append(
                {
                    "range": f"{range_start}~{range_end}",
                    "codes": codes,
                    "status": "failed",
                    "duration_sec": round(time.perf_counter() - batch_started, 3),
                }
            )
            continue

        batch_failed_days = list(batch_details.get("failed_days", []))
        failed_days.extend(batch_failed_days)
        processed_requests += int(batch_details.get("processed_requests", 0) or 0)
        if batch_details.get("aborted"):
            aborted = True
            abort_reason = str(batch_details.get("abort_reason", "")).strip().upper()
        for api_id, family_failures in dict(batch_details.get("snapshot_failures", {})).items():
            snapshot_failures.setdefault(api_id, {}).update(family_failures)

        for code in codes:
            frame = fetched_frames.get(code, pd.DataFrame())
            if frame.empty:
                failures[code] = batch_failures.get(code, "empty response")
                continue
            merged = _merge_price_frames(frames_by_code.get(code, pd.DataFrame()), frame)
            frames_by_code[code] = merged
            _save_raw_cache(code, merged, end)
            refreshed_codes.append(code)

        range_summaries.append(
            {
                "range": f"{range_start}~{range_end}",
                "codes": codes,
                "status": (
                    "aborted"
                    if batch_details.get("aborted")
                    else "partial"
                    if batch_failed_days or any(code in batch_failures for code in codes)
                    else "ok"
                ),
                "failed_days": batch_failed_days,
                "failed_codes": sorted(code for code in codes if code in batch_failures),
                "predicted_requests": int(batch_details.get("request_count", 0) or 0),
                "processed_requests": int(batch_details.get("processed_requests", 0) or 0),
                "duration_sec": round(time.perf_counter() - batch_started, 3),
            }
        )
        if aborted:
            break

    summary = {
        "provider": "OPENAPI",
        "cache_hits": cache_hits,
        "delta_codes": sorted(set(refreshed_codes)),
        "failed_days": sorted(set(failed_days)),
        "snapshot_failures": snapshot_failures,
        "ranges": range_summaries,
        "predicted_requests": predicted_requests,
        "processed_requests": processed_requests,
        "aborted": aborted,
        "abort_reason": abort_reason,
    }
    return frames_by_code, failures, summary


def _refresh_pykrx_raw_cache(
    index_codes: list[str],
    start: str,
    end: str,
    *,
    force: bool = False,
    force_codes: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, Any]]:
    """Fetch missing raw-cache ranges via pykrx and merge them."""
    forced = {str(code) for code in (force_codes or set())}
    frames_by_code = {
        code: (_load_latest_raw_cache(code) if not force and code not in forced else pd.DataFrame())
        for code in index_codes
    }
    failures: dict[str, str] = {}
    cache_hits = sum(1 for frame in frames_by_code.values() if not frame.empty)
    refreshed_codes: list[str] = []

    for code in index_codes:
        missing_ranges = (
            [(start, end)]
            if force or code in forced
            else _compute_missing_ranges(frames_by_code[code], start, end)
        )
        for range_start, range_end in missing_ranges:
            try:
                frame = fetch_index_ohlcv(code, range_start, range_end)
                if frame.empty:
                    failures[code] = "empty response"
                    continue
                merged = _merge_price_frames(frames_by_code.get(code, pd.DataFrame()), frame)
                frames_by_code[code] = merged
                _save_raw_cache(code, merged, end)
                refreshed_codes.append(code)
            except Exception as exc:
                failures[code] = str(exc)

    summary = {
        "provider": "PYKRX",
        "cache_hits": cache_hits,
        "delta_codes": sorted(set(refreshed_codes)),
        "failed_days": [],
        "ranges": [],
        "predicted_requests": 0,
        "processed_requests": 0,
        "aborted": False,
        "abort_reason": "",
    }
    return frames_by_code, failures, summary


def _drop_unusable_contaminated_codes(
    *,
    contaminated_codes: list[str],
    frames_by_code: dict[str, pd.DataFrame],
    failures: dict[str, str],
    refresh_summary: dict[str, Any],
    provider_mode: KRXProvider,
    start: str,
    end: str,
) -> list[str]:
    """Remove contaminated codes that could not be fully rebuilt from live data."""
    unusable: list[str] = []
    snapshot_failures = dict(refresh_summary.get("snapshot_failures", {}))

    for code in contaminated_codes:
        frame = frames_by_code.get(code, pd.DataFrame())
        has_requested_range = not _slice_raw_cache(frame, start, end).empty
        has_full_range = has_requested_range and not _compute_missing_ranges(frame, start, end)
        family_failed = False
        family_failed_days: list[str] = []
        if provider_mode == "OPENAPI":
            family_failed_days = sorted(snapshot_failures.get(resolve_openapi_api_id(code), {}).keys())
            family_failed = bool(family_failed_days)

        if code in failures or not has_full_range or family_failed:
            if code not in failures:
                if family_failed_days:
                    preview = ", ".join(family_failed_days[:3])
                    suffix = "..." if len(family_failed_days) > 3 else ""
                    failures[code] = (
                        "contaminated raw cache refresh incomplete "
                        f"(failed days: {preview}{suffix})"
                    )
                else:
                    failures[code] = "contaminated raw cache refresh produced no full-range data"
            frames_by_code.pop(code, None)
            unusable.append(code)

    return sorted(unusable)


def warm_sector_price_cache(
    index_codes: list[str],
    start: str,
    end: str,
    *,
    reason: str = "manual",
    force: bool = False,
) -> tuple[LoaderResult, dict[str, Any]]:
    """Warm or incrementally refresh sector price cache for the requested range."""
    started = time.perf_counter()
    provider_mode = _resolve_provider_mode()
    normalized_codes, replacements = _normalize_requested_codes(index_codes)
    if provider_mode == "PYKRX":
        live_codes, skipped_codes = _filter_supported_codes(normalized_codes)
    else:
        live_codes, skipped_codes = _filter_supported_codes_openapi(normalized_codes)

    summary: dict[str, Any] = {
        "reason": reason,
        "provider": provider_mode,
        "start": start,
        "end": end,
        "requested_codes": normalized_codes,
        "replacements": [f"{src}->{dst}" for src, dst in replacements],
        "skipped_codes": skipped_codes,
        "status": "failed",
        "aborted": False,
        "abort_reason": "",
        "predicted_requests": 0,
        "processed_requests": 0,
    }
    contaminated_codes = [] if force else _detect_contaminated_raw_cache_codes(live_codes, start, end)
    if contaminated_codes:
        logger.warning(
            "Detected contaminated raw cache for codes=%s; forcing full-range rebuild",
            ",".join(contaminated_codes),
        )
    summary["contaminated_codes"] = contaminated_codes

    if provider_mode == "OPENAPI" and not get_krx_openapi_key():
        detail = "KRX_OPENAPI_KEY not configured"
        summary.update(
            {
                "error": detail,
                "failed_days": [],
                "coverage_complete": False,
                "predicted_requests": 0,
                "processed_requests": 0,
            }
        )
        _write_warm_status(summary)
        return ("CACHED", pd.DataFrame()), summary

    if provider_mode == "OPENAPI":
        frames_by_code, failures, refresh_summary = _refresh_openapi_raw_cache(
            live_codes,
            start,
            end,
            force=force,
            force_codes=set(contaminated_codes),
        )
    else:
        frames_by_code, failures, refresh_summary = _refresh_pykrx_raw_cache(
            live_codes,
            start,
            end,
            force=force,
            force_codes=set(contaminated_codes),
        )

    unavailable_contaminated_codes = _drop_unusable_contaminated_codes(
        contaminated_codes=contaminated_codes,
        frames_by_code=frames_by_code,
        failures=failures,
        refresh_summary=refresh_summary,
        provider_mode=provider_mode,
        start=start,
        end=end,
    )
    summary["contaminated_codes"] = contaminated_codes
    summary["unavailable_contaminated_codes"] = unavailable_contaminated_codes

    failed_days = list(refresh_summary.get("failed_days", []))
    coverage_complete = _is_requested_coverage_complete(
        frames_by_code,
        live_codes,
        start,
        end,
        failures=failures,
        failed_days=failed_days,
    )
    result = _build_result_from_raw_frames(frames_by_code, live_codes, start, end)
    if not result.empty:
        validated = _validate_sector_prices(result)
        status: DataStatus = (
            "LIVE" if coverage_complete and refresh_summary.get("delta_codes") else "CACHED"
        )
        if coverage_complete:
            validated = _persist_curated_sector_prices(validated)
        else:
            if not unavailable_contaminated_codes:
                cached = _load_curated_sector_prices()
                if not cached.empty:
                    validated = cached
        summary.update(
            {
                "status": status,
                "rows": int(len(validated)),
                "loaded_codes": sorted(validated["index_code"].unique().tolist()),
                "failed_codes": failures,
                "failed_days": failed_days,
                "coverage_complete": coverage_complete,
                "cache_hits": refresh_summary.get("cache_hits", 0),
                "delta_codes": refresh_summary.get("delta_codes", []),
                "ranges": refresh_summary.get("ranges", []),
                "aborted": bool(refresh_summary.get("aborted")),
                "abort_reason": str(refresh_summary.get("abort_reason", "")).strip(),
                "predicted_requests": int(refresh_summary.get("predicted_requests", 0) or 0),
                "processed_requests": int(refresh_summary.get("processed_requests", 0) or 0),
                "duration_sec": round(time.perf_counter() - started, 3),
            }
        )
        _write_warm_status(summary)
        logger.info(
            "KRX warm completed reason=%s provider=%s rows=%d coverage_complete=%s failed_days=%s cache_hits=%s delta_codes=%s duration_sec=%.2f",
            reason,
            provider_mode,
            len(validated),
            coverage_complete,
            failed_days,
            refresh_summary.get("cache_hits", 0),
            refresh_summary.get("delta_codes", []),
            time.perf_counter() - started,
        )
        return (status, validated), summary

    cached = _load_curated_sector_prices()
    if not cached.empty:
        cached = _filter_sector_price_result(cached, unavailable_contaminated_codes)
        summary.update(
            {
                "status": "CACHED",
                "rows": int(len(cached)),
                "failed_codes": failures,
                "failed_days": failed_days,
                "coverage_complete": coverage_complete,
                "cache_hits": refresh_summary.get("cache_hits", 0),
                "delta_codes": refresh_summary.get("delta_codes", []),
                "ranges": refresh_summary.get("ranges", []),
                "aborted": bool(refresh_summary.get("aborted")),
                "abort_reason": str(refresh_summary.get("abort_reason", "")).strip(),
                "predicted_requests": int(refresh_summary.get("predicted_requests", 0) or 0),
                "processed_requests": int(refresh_summary.get("processed_requests", 0) or 0),
                "duration_sec": round(time.perf_counter() - started, 3),
            }
        )
        _write_warm_status(summary)
        if not cached.empty:
            return ("CACHED", cached), summary

    summary.update(
        {
            "status": "failed",
            "failed_codes": failures,
            "failed_days": failed_days,
            "coverage_complete": coverage_complete,
            "cache_hits": refresh_summary.get("cache_hits", 0),
            "delta_codes": refresh_summary.get("delta_codes", []),
            "ranges": refresh_summary.get("ranges", []),
            "aborted": bool(refresh_summary.get("aborted")),
            "abort_reason": str(refresh_summary.get("abort_reason", "")).strip(),
            "predicted_requests": int(refresh_summary.get("predicted_requests", 0) or 0),
            "processed_requests": int(refresh_summary.get("processed_requests", 0) or 0),
            "duration_sec": round(time.perf_counter() - started, 3),
        }
    )
    _write_warm_status(summary)
    return ("SAMPLE", pd.DataFrame()), summary


def run_manual_price_refresh(
    index_codes: list[str],
    start: str,
    end: str,
) -> tuple[LoaderResult, dict[str, Any]]:
    """Refresh the requested range through the standard manual OpenAPI warm path."""
    provider_mode = _resolve_provider_mode()
    refresh_start = start
    refresh_end = end
    if provider_mode == "OPENAPI":
        reset_krx_openapi_health_cache()
        refresh_start, refresh_end, _predicted = _cap_openapi_interactive_range(index_codes, start, end)

    result, summary = warm_sector_price_cache(
        index_codes,
        refresh_start,
        refresh_end,
        reason="manual_refresh",
        force=False,
    )
    if provider_mode == "OPENAPI" and _summary_has_access_denied(summary):
        raise KRXMarketDataAccessDeniedError(_access_denied_detail(summary))
    return result, summary


def schedule_background_warm(
    index_codes: list[str],
    start: str,
    end: str,
    *,
    reason: str = "background",
    force: bool = False,
) -> bool:
    """Start one background warm thread per normalized request key."""
    provider_mode = _resolve_provider_mode()
    if provider_mode == "OPENAPI" and not get_krx_openapi_key():
        return False

    normalized_codes, _ = _normalize_requested_codes(index_codes)
    predicted_requests = _predict_openapi_requests(normalized_codes, start, end)
    if (
        provider_mode == "OPENAPI"
        and predicted_requests > INTERACTIVE_OPENAPI_REQUEST_LIMIT
    ):
        logger.info(
            "Skipping background OPENAPI warm for oversized interactive range %s~%s (%d predicted requests)",
            start,
            end,
            predicted_requests,
        )
        return False
    thread_key = "|".join([provider_mode, start, end, ",".join(normalized_codes), str(force)])

    with _BACKGROUND_WARM_LOCK:
        existing = _BACKGROUND_WARM_THREADS.get(thread_key)
        if existing is not None and existing.is_alive():
            return False

        def _runner() -> None:
            try:
                warm_sector_price_cache(
                    normalized_codes,
                    start,
                    end,
                    reason=reason,
                    force=force,
                )
            except Exception:
                logger.exception("Background KRX warm failed")
            finally:
                with _BACKGROUND_WARM_LOCK:
                    _BACKGROUND_WARM_THREADS.pop(thread_key, None)

        _write_warm_status(
            {
                "status": "scheduled",
                "reason": reason,
                "provider": provider_mode,
                "start": start,
                "end": end,
                "requested_codes": normalized_codes,
                "force": force,
                "predicted_requests": predicted_requests,
            }
        )
        thread = threading.Thread(
            target=_runner,
            name=f"krx-warm-{len(_BACKGROUND_WARM_THREADS)+1}",
            daemon=True,
        )
        _BACKGROUND_WARM_THREADS[thread_key] = thread
        thread.start()
        return True


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

    raw_frames, raw_state = _collect_raw_cache_state(live_codes, start, end)
    contaminated_codes = _detect_contaminated_raw_cache_codes(
        live_codes,
        start,
        end,
        raw_frames_by_code=raw_frames,
    )
    cache_safe_codes = [code for code in live_codes if code not in contaminated_codes]
    if contaminated_codes:
        logger.warning(
            "Detected contaminated raw cache during load for codes=%s; bypassing cache-only fast path",
            ",".join(contaminated_codes),
        )

    has_all_cache_slices = (not contaminated_codes) and bool(live_codes) and all(
        raw_state[code]["has_slice"] and not raw_state[code]["has_older_gap"]
        for code in live_codes
    )
    should_background_refresh = has_all_cache_slices and any(
        raw_state[code]["newer_gap_days"] > STALE_CACHE_TOLERANCE_BUSINESS_DAYS
        for code in live_codes
    )

    if has_all_cache_slices:
        cached_result = _build_result_from_raw_frames(raw_frames, live_codes, start, end)
        if not cached_result.empty:
            validated = _validate_sector_prices(cached_result)
            if not (CURATED_DIR / "sector_prices.parquet").exists():
                _persist_curated_sector_prices(validated)
            if should_background_refresh:
                schedule_background_warm(
                    live_codes,
                    start,
                    end,
                    reason="cache_delta_refresh",
                    force=False,
                )
            logger.info(
                "Loaded %d codes from raw cache first (provider=%s, background_refresh=%s)",
                len(live_codes),
                provider_mode,
                should_background_refresh,
            )
            return ("CACHED", validated)

    if provider_mode == "OPENAPI" and not get_krx_openapi_key():
        logger.warning(
            "KRX provider is OPENAPI but KRX_OPENAPI_KEY is missing; falling back to cache."
        )
        cached = _load_curated_sector_prices()
        if not cached.empty:
            filtered_cached = _filter_sector_price_result(cached, contaminated_codes)
            if not filtered_cached.empty:
                return ("CACHED", _validate_sector_prices(filtered_cached))
        if raw_frames and cache_safe_codes:
            partial = _build_result_from_raw_frames(raw_frames, cache_safe_codes, start, end)
            if not partial.empty:
                return ("CACHED", _validate_sector_prices(partial))
        sample = _make_sample_df(normalized_codes)
        return ("SAMPLE", sample)

    if provider_mode == "OPENAPI":
        predicted_requests = _predict_openapi_requests(live_codes, start, end)
        if predicted_requests > INTERACTIVE_OPENAPI_REQUEST_LIMIT:
            raise KRXInteractiveRangeLimitError(
                "Interactive OpenAPI refresh would require "
                f"{predicted_requests} snapshot requests, exceeding the limit of "
                f"{INTERACTIVE_OPENAPI_REQUEST_LIMIT}. Use the warm script for large backfills."
            )

    (status, result), summary = warm_sector_price_cache(
        live_codes,
        start,
        end,
        reason="blocking_refresh",
        force=False,
    )
    if provider_mode == "OPENAPI" and _summary_has_access_denied(summary):
        raise KRXMarketDataAccessDeniedError(_access_denied_detail(summary))
    if not result.empty:
        if summary.get("failed_codes"):
            logger.warning(
                "KRX partial success: loaded=%s failed=%s",
                summary.get("loaded_codes", []),
                summary.get("failed_codes", {}),
            )
        return (status, result)

    cached = _load_curated_sector_prices()
    if not cached.empty:
        cached = _filter_sector_price_result(cached, contaminated_codes)
        if cached.empty:
            logger.warning(
                "Curated cache only contained contaminated codes=%s; ignoring fallback",
                ",".join(contaminated_codes),
            )
        else:
            logger.info("Loaded sector prices from curated cache after warm fallback")
            return ("CACHED", cached)

    # SAMPLE fallback
    logger.warning("Using SAMPLE data for sector prices")
    sample = _make_sample_df(normalized_codes)
    return ("SAMPLE", sample)
