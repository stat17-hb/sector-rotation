"""
ECOS (한국은행 경제통계시스템) Open API wrapper.

R11 — HTTP retry policy: timeout 10s, retries 3, backoff 2/4/8s.
Public loaders return LoaderResult = tuple[DataStatus, pd.DataFrame].
"""
from __future__ import annotations

import logging
import os
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DataStatus = Literal["LIVE", "CACHED", "SAMPLE"]
LoaderResult = tuple[DataStatus, pd.DataFrame]

CURATED_DIR = Path("data/curated")
ECOS_BASE_URL = "https://ecos.bok.or.kr/api"

REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
BACKOFF_BASE = 2
MAX_ITEM_CODES = 3
VALID_CYCLES = {"A", "Q", "M", "D"}


def _normalize_cycle(cycle: str | None) -> str:
    cycle_norm = str(cycle or "M").strip().upper()
    if cycle_norm not in VALID_CYCLES:
        raise ValueError(f"Unsupported ECOS cycle: {cycle!r}")
    return cycle_norm


def _normalize_date_bounds(start_ym: str, end_ym: str, cycle: str) -> tuple[str, str]:
    """Convert app-level YYYYMM bounds to ECOS cycle-specific date strings."""
    cycle_norm = _normalize_cycle(cycle)
    start_clean = str(start_ym).strip()
    end_clean = str(end_ym).strip()

    if cycle_norm == "D":
        if len(start_clean) == 6:
            start_clean = f"{start_clean}01"
        if len(end_clean) == 6:
            end_period = pd.Period(end_clean, freq="M")
            end_clean = end_period.to_timestamp(how="end").strftime("%Y%m%d")
        return start_clean, end_clean

    if cycle_norm == "Q":
        s = pd.Period(start_clean[:6], freq="M").asfreq("Q")
        e = pd.Period(end_clean[:6], freq="M").asfreq("Q")
        return f"{s.year}Q{s.quarter}", f"{e.year}Q{e.quarter}"

    if cycle_norm == "A":
        return start_clean[:4], end_clean[:4]

    # Monthly
    return start_clean[:6], end_clean[:6]


def _time_to_month_period(time_str: str, cycle: str) -> pd.Period | None:
    """Convert ECOS TIME token to monthly Period for contract consistency."""
    token = str(time_str).strip()
    cycle_norm = _normalize_cycle(cycle)

    if cycle_norm == "M":
        if len(token) == 6 and token.isdigit():
            return pd.Period(token, freq="M")
        return None

    if cycle_norm == "D":
        if len(token) == 8 and token.isdigit():
            return pd.Period(token[:6], freq="M")
        return None

    if cycle_norm == "Q":
        # Expected format usually YYYYQn
        m = re.match(r"^(\d{4})Q([1-4])$", token)
        if m:
            y = int(m.group(1))
            q = int(m.group(2))
            return pd.Period(f"{y}Q{q}", freq="Q").asfreq("M", how="end")
        return None

    # Annual
    if len(token) == 4 and token.isdigit():
        return pd.Period(f"{token}12", freq="M")
    return None


def _get_api_key() -> str:
    """Get ECOS API key from Streamlit secrets or environment variable."""
    try:
        import streamlit as st  # type: ignore[import]
        key = st.secrets.get("ECOS_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("ECOS_API_KEY", "")


def _get_with_retry(url: str) -> dict:
    """HTTP GET with timeout, retry, and exponential backoff.

    Policy (R11): timeout=10s, retries=3, backoff 2/4/8s.
    Re-raises last exception on final failure.
    Does not retry 4xx client errors (except 429).
    """
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            try:
                return resp.json()
            except ValueError as exc:
                text_preview = (resp.text or "")[:200]
                raise ValueError(f"ECOS returned non-JSON response: {text_preview!r}") from exc
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            logger.warning("ECOS HTTP attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** (attempt + 1))
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in {429, 503}:
                last_exc = exc
                logger.warning("ECOS rate limit/503 attempt %d/%d: %s", attempt + 1, MAX_RETRIES, exc)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BACKOFF_BASE ** (attempt + 1))
            else:
                raise  # 4xx client errors: don't retry

    assert last_exc is not None
    raise last_exc


def _normalize_item_codes(
    item_code: str | None,
    item_codes: Iterable[str] | None,
) -> list[str]:
    """Return validated item code path (1..3 items)."""
    normalized: list[str] = []
    if item_codes:
        normalized = [str(code).strip() for code in item_codes if str(code).strip()]
    elif item_code:
        normalized = [str(item_code).strip()]

    if not normalized:
        raise ValueError("ECOS item code is missing (item_code or item_codes required)")
    if len(normalized) > MAX_ITEM_CODES:
        raise ValueError(f"ECOS item_codes supports up to {MAX_ITEM_CODES} levels")
    return normalized


def _build_statistic_url(
    api_key: str,
    stat_code: str,
    cycle: str,
    start_date: str,
    end_date: str,
    item_codes: list[str],
) -> str:
    path_parts = [
        ECOS_BASE_URL,
        "StatisticSearch",
        api_key,
        "json",
        "kr",
        "1",
        "1000",
        stat_code,
        cycle,
        start_date,
        end_date,
        *item_codes,
    ]
    return "/".join(path_parts)


def _extract_result_block(data: dict) -> dict:
    """Extract ECOS RESULT envelope from top-level or nested blocks."""
    if not isinstance(data, dict):
        return {}

    top_level = data.get("RESULT")
    if isinstance(top_level, dict) and top_level:
        return top_level

    nested = data.get("StatisticSearch")
    if isinstance(nested, dict):
        nested_result = nested.get("RESULT")
        if isinstance(nested_result, dict) and nested_result:
            return nested_result

    return {}


def fetch_series(
    stat_code: str,
    item_code: str | None,
    start_ym: str,
    end_ym: str,
    item_codes: list[str] | None = None,
    cycle: str = "M",
) -> pd.DataFrame:
    """Fetch a single ECOS statistical series.

    Args:
        stat_code: ECOS statistic table code (e.g. "722Y001").
        item_code: Single item code for backward compatibility.
        start_ym: Start year-month in YYYYMM format.
        end_ym: End year-month in YYYYMM format.
        item_codes: Optional list of item codes for multi-level series path.
        cycle: ECOS cycle code ("M", "D", "Q", "A"). Defaults to "M".

    Returns:
        DataFrame with PeriodIndex (monthly) and columns:
        [series_id, value, source, fetched_at, is_provisional].
    """
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("ECOS_API_KEY not configured")

    cycle_norm = _normalize_cycle(cycle)
    normalized_item_codes = _normalize_item_codes(item_code=item_code, item_codes=item_codes)
    start_date, end_date = _normalize_date_bounds(start_ym, end_ym, cycle_norm)
    url = _build_statistic_url(
        api_key=api_key,
        stat_code=stat_code,
        cycle=cycle_norm,
        start_date=start_date,
        end_date=end_date,
        item_codes=normalized_item_codes,
    )
    masked_url = url.replace(api_key, api_key[:4] + "****")
    logger.info("ECOS request: %s", masked_url)
    data = _get_with_retry(url)

    # ECOS returns RESULT envelopes for invalid key / bad schema / no data.
    result_block = _extract_result_block(data)
    if result_block:
        code = str(result_block.get("CODE", "")).strip()
        msg = str(result_block.get("MESSAGE", "")).strip()
        hint = ""
        if code == "ERROR-100":
            hint = " (invalid key OR missing/invalid request schema such as item code path)"
        raise ValueError(f"ECOS API error [{code}]: {msg}{hint}")

    rows_raw = data.get("StatisticSearch", {}).get("row", []) if isinstance(data, dict) else []
    if not rows_raw:
        keys = list(data.keys()) if isinstance(data, dict) else [type(data).__name__]
        logger.error("ECOS raw response keys: %s", keys)
        item_path = "/".join(normalized_item_codes)
        raise ValueError(f"No ECOS data returned for {stat_code}/{item_path}")

    rows = []
    now = datetime.now(timezone.utc)
    series_key = "/".join(normalized_item_codes)
    for row in rows_raw:
        time_str = row.get("TIME", "")
        period = _time_to_month_period(time_str, cycle_norm)
        if period is None:
            continue

        raw_value = row.get("DATA_VALUE", "")
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue

        rows.append(
            {
                "period": period,
                "series_id": f"{stat_code}/{series_key}",
                "value": value,
                "source": "ECOS",
                "fetched_at": now,
                "is_provisional": False,
            }
        )

    if not rows:
        item_path = "/".join(normalized_item_codes)
        raise ValueError(f"Could not parse ECOS rows for {stat_code}/{item_path}")

    df = pd.DataFrame(rows).set_index("period")
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df.astype({"value": "float64", "is_provisional": "bool"})


def fetch_base_rate(start_ym: str, end_ym: str) -> pd.DataFrame:
    """Fetch 기준금리 (base rate) from ECOS."""
    return fetch_series("722Y001", "0101000", start_ym, end_ym)


def fetch_bond_3y(start_ym: str, end_ym: str) -> pd.DataFrame:
    """Fetch 국고채 3년 yield from ECOS."""
    return fetch_series("817Y002", "010200000", start_ym, end_ym)


def fetch_usdkrw(start_ym: str, end_ym: str) -> pd.DataFrame:
    """Fetch USD/KRW exchange rate from ECOS."""
    return fetch_series("731Y004", "0000001", start_ym, end_ym)


def _make_sample_macro() -> pd.DataFrame:
    """Generate synthetic macro data for SAMPLE mode."""
    import numpy as np

    periods = pd.period_range(end=pd.Period.now("M"), periods=36, freq="M")
    rng = np.random.default_rng(42)
    now = datetime.now(timezone.utc)
    rows = []
    for p in periods:
        rows.append(
            {
                "series_id": "SAMPLE/base_rate",
                "value": float(rng.uniform(2.0, 4.0)),
                "source": "ECOS",
                "fetched_at": now,
                "is_provisional": False,
            }
        )
    df = pd.DataFrame(rows, index=pd.PeriodIndex(periods, freq="M"))
    return df.astype({"value": "float64", "is_provisional": "bool"})


def load_ecos_macro(
    start_ym: str,
    end_ym: str,
    series_config: dict | None = None,
) -> LoaderResult:
    """Load ECOS macro data with LIVE/CACHED/SAMPLE fallback.

    Args:
        start_ym: Start YYYYMM.
        end_ym: End YYYYMM.
        series_config: Dict of {name: {stat_code, item_code}} to fetch.

    Returns:
        LoaderResult = (DataStatus, DataFrame).
    """
    curated_path = CURATED_DIR / "macro_monthly.parquet"

    if series_config is None:
        series_config = {
            "base_rate": {
                "stat_code": "722Y001",
                "item_code": "0101000",
                "item_codes": ["0101000"],
                "cycle": "M",
                "enabled": True,
            },
            "bond_3y": {
                "stat_code": "817Y002",
                "item_code": "010200000",
                "item_codes": ["010200000"],
                "cycle": "D",
                "enabled": True,
            },
            "usdkrw": {
                "stat_code": "731Y004",
                "item_code": "0000001",
                "item_codes": ["0000001"],
                "cycle": "M",
                "enabled": True,
            },
        }

    active_config = {
        name: cfg for name, cfg in series_config.items() if bool(cfg.get("enabled", True))
    }
    if not active_config:
        logger.warning("No enabled ECOS series configured; returning empty LIVE macro frame.")
        empty = pd.DataFrame(
            columns=["series_id", "value", "source", "fetched_at", "is_provisional"]
        )
        empty.index = pd.PeriodIndex([], freq="M")
        return ("LIVE", empty)

    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    for name, cfg in active_config.items():
        try:
            df = fetch_series(
                stat_code=cfg["stat_code"],
                item_code=cfg.get("item_code"),
                start_ym=start_ym,
                end_ym=end_ym,
                item_codes=cfg.get("item_codes"),
                cycle=str(cfg.get("cycle", "M")),
            )
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            failures.append((name, str(exc)))
            logger.warning("ECOS series fetch failed (%s): %s", name, exc)

    if frames:
        result = pd.concat(frames)
        result = result.sort_index()
        CURATED_DIR.mkdir(parents=True, exist_ok=True)

        from src.contracts.validators import normalize_then_validate

        result = normalize_then_validate(result, "macro_monthly")
        result.to_parquet(curated_path)
        if failures:
            failed_names = ", ".join(name for name, _ in failures)
            logger.warning(
                "ECOS partial success: %d series loaded, %d failed (%s)",
                len(frames),
                len(failures),
                failed_names,
            )
        return ("LIVE", result)

    if failures:
        logger.error("ECOS live fetch failed for all series: %s", failures)

    # CACHED fallback
    if curated_path.exists():
        try:
            cached = pd.read_parquet(curated_path)
            logger.info("Loaded ECOS macro from cache")
            return ("CACHED", cached)
        except Exception as exc:
            logger.error("Cache load failed: %s", exc)

    # SAMPLE fallback
    logger.warning("Using SAMPLE data for ECOS macro")
    return ("SAMPLE", _make_sample_macro())
