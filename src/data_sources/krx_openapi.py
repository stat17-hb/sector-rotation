"""
KRX OpenAPI helpers for index daily close retrieval.

This module provides:
- Runtime config loading for KRX provider mode and OpenAPI key.
- OpenAPI request/response handling with auth-aware exceptions.
- Parsing utilities that normalize API responses into OHLCV-like DataFrames.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Literal

import pandas as pd
import requests

logger = logging.getLogger(__name__)

KRXProvider = Literal["AUTO", "OPENAPI", "PYKRX"]

DEFAULT_KRX_PROVIDER: KRXProvider = "AUTO"
OPENAPI_BASE_URL = "https://data-dbg.krx.co.kr"
OPENAPI_INDEX_DAILY_PATH = "/svc/apis/idx/krx_dd_trd"

REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
BACKOFF_BASE = 2

_SUCCESS_CODES = {"0", "00", "200", "0000"}
_AUTH_CODES = {"401"}
_PERMISSION_CODES = {"403"}


class KRXOpenAPIError(RuntimeError):
    """Base OpenAPI error."""


class KRXOpenAPIAuthError(KRXOpenAPIError):
    """Raised when AUTH_KEY is missing/invalid or rejected."""


class KRXOpenAPIPermissionError(KRXOpenAPIError):
    """Raised when key is valid but service permission is missing."""


class KRXOpenAPIResponseError(KRXOpenAPIError):
    """Raised when response payload is malformed or empty."""


def _load_secret_or_env(name: str) -> str:
    """Load a setting from Streamlit secrets with environment fallback."""
    try:
        import streamlit as st  # type: ignore[import]

        value = str(st.secrets.get(name, "")).strip()
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name, "").strip()


def get_krx_openapi_key() -> str:
    """Return KRX OpenAPI key from secrets/env."""
    return _load_secret_or_env("KRX_OPENAPI_KEY")


def get_krx_provider(raw: str | None = None) -> KRXProvider:
    """Return provider mode (AUTO/OPENAPI/PYKRX), defaulting to AUTO."""
    source = raw if raw is not None else _load_secret_or_env("KRX_PROVIDER")
    value = str(source or "").strip().upper()
    if not value:
        return DEFAULT_KRX_PROVIDER
    if value in {"AUTO", "OPENAPI", "PYKRX"}:
        return value  # type: ignore[return-value]

    logger.warning("Invalid KRX_PROVIDER=%r; falling back to AUTO.", source)
    return DEFAULT_KRX_PROVIDER


def get_krx_openapi_url() -> str:
    """Return OpenAPI endpoint URL with optional override."""
    custom = _load_secret_or_env("KRX_OPENAPI_URL")
    if custom:
        return custom
    return f"{OPENAPI_BASE_URL}{OPENAPI_INDEX_DAILY_PATH}"


def _normalize_resp_meta(payload: Any) -> tuple[str, str]:
    """Extract top-level respCode/respMsg (if present)."""
    if isinstance(payload, dict):
        resp_code = str(payload.get("respCode", "")).strip()
        resp_msg = str(payload.get("respMsg", "")).strip()
        return resp_code, resp_msg
    return "", ""


def _raise_by_status_or_code(status_code: int, resp_code: str, resp_msg: str) -> None:
    """Raise auth/permission errors based on HTTP status and payload code/message."""
    lowered_msg = resp_msg.lower()
    if status_code == 401 or resp_code in _AUTH_CODES or "unauthorized" in lowered_msg:
        raise KRXOpenAPIAuthError(f"KRX OpenAPI authentication failed: {resp_msg or status_code}")
    if status_code == 403 or resp_code in _PERMISSION_CODES or "forbidden" in lowered_msg:
        raise KRXOpenAPIPermissionError(
            f"KRX OpenAPI permission denied: {resp_msg or status_code}"
        )


def _request_with_retry(
    *,
    url: str,
    auth_key: str,
    params: dict[str, str],
    session: requests.Session | None = None,
) -> Any:
    """Issue OpenAPI request with retry policy and auth-aware error handling."""
    if not auth_key:
        raise ValueError("KRX_OPENAPI_KEY not configured")

    last_exc: Exception | None = None
    requester = session or requests
    headers = {"AUTH_KEY": auth_key, "User-Agent": "sector-rotation/1.0"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requester.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            payload: Any
            try:
                payload = resp.json()
            except ValueError as exc:
                text_preview = (resp.text or "")[:200]
                raise KRXOpenAPIResponseError(
                    f"KRX OpenAPI returned non-JSON payload: {text_preview!r}"
                ) from exc

            resp_code, resp_msg = _normalize_resp_meta(payload)
            _raise_by_status_or_code(resp.status_code, resp_code, resp_msg)

            if resp.status_code >= 500:
                raise KRXOpenAPIResponseError(f"KRX OpenAPI server error: HTTP {resp.status_code}")

            if resp.status_code >= 400:
                detail = resp_msg or f"HTTP {resp.status_code}"
                raise KRXOpenAPIResponseError(f"KRX OpenAPI request failed: {detail}")

            if resp_code and resp_code not in _SUCCESS_CODES:
                detail = resp_msg or f"respCode={resp_code}"
                _raise_by_status_or_code(resp.status_code, resp_code, detail)
                raise KRXOpenAPIResponseError(f"KRX OpenAPI error: {detail}")

            return payload
        except (KRXOpenAPIAuthError, KRXOpenAPIPermissionError):
            raise
        except (requests.Timeout, requests.ConnectionError, KRXOpenAPIResponseError) as exc:
            last_exc = exc
            is_last = attempt >= MAX_RETRIES - 1
            if is_last:
                break
            logger.warning(
                "KRX OpenAPI attempt %d/%d failed: %s",
                attempt + 1,
                MAX_RETRIES,
                exc,
            )
            time.sleep(BACKOFF_BASE ** (attempt + 1))
        except requests.RequestException as exc:
            raise KRXOpenAPIError(f"KRX OpenAPI request exception: {exc}") from exc

    assert last_exc is not None
    raise last_exc


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    """Extract row dictionaries from common OpenAPI payload shapes."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if not isinstance(payload, dict):
        return []

    list_keys = (
        "OutBlock_1",
        "outBlock_1",
        "output",
        "result",
        "results",
        "data",
        "items",
        "list",
    )
    for key in list_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            nested = _extract_rows(value)
            if nested:
                return nested

    response = payload.get("response")
    if isinstance(response, dict):
        body = response.get("body")
        if isinstance(body, dict):
            items = body.get("items")
            if isinstance(items, dict):
                item = items.get("item")
                if isinstance(item, list):
                    return [row for row in item if isinstance(row, dict)]
                if isinstance(item, dict):
                    return [item]
            if isinstance(items, list):
                return [row for row in items if isinstance(row, dict)]

    return []


def _row_get(row: dict[str, Any], candidates: tuple[str, ...]) -> str:
    """Return first matching non-empty value across multiple key candidates."""
    lowered = {str(k).lower(): v for k, v in row.items()}
    for key in candidates:
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
        lower_key = key.lower()
        if lower_key in lowered and str(lowered[lower_key]).strip():
            return str(lowered[lower_key]).strip()
    return ""


def _parse_row_date(row: dict[str, Any]) -> pd.Timestamp | None:
    raw = _row_get(
        row,
        (
            "BAS_DD",
            "basDd",
            "TRD_DD",
            "trdDd",
            "date",
            "DATE",
            "BAS_DT",
            "basDt",
        ),
    )
    if not raw:
        return None
    digits = "".join(ch for ch in raw if ch.isdigit())
    try:
        if len(digits) == 8:
            return pd.Timestamp(datetime.strptime(digits, "%Y%m%d"))
        if len(raw) == 10 and "-" in raw:
            return pd.Timestamp(raw)
    except Exception:
        return None
    return None


def _parse_row_index_code(row: dict[str, Any]) -> str:
    return _row_get(
        row,
        (
            "IDX_IND_CD",
            "idxIndCd",
            "IDX_CD",
            "idxCd",
            "index_code",
            "INDEX_CODE",
            "ISU_CD",
            "isuCd",
        ),
    )


def _parse_row_close(row: dict[str, Any]) -> float | None:
    raw = _row_get(
        row,
        (
            "CLSPRC_IDX",
            "clsprcIdx",
            "TDD_CLSPRC",
            "tddClsprc",
            "CLOSE",
            "close",
            "\uc885\uac00",
        ),
    )
    if not raw:
        return None
    cleaned = raw.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def fetch_index_ohlcv_openapi(
    index_code: str,
    start: str,
    end: str,
    *,
    auth_key: str | None = None,
    url: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch index close history via KRX OpenAPI and return DatetimeIndex DataFrame."""
    key = (auth_key or get_krx_openapi_key()).strip()
    if not key:
        raise ValueError("KRX_OPENAPI_KEY not configured")

    endpoint = (url or get_krx_openapi_url()).strip()
    if not endpoint:
        raise ValueError("KRX OpenAPI URL is empty")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    params = {
        "idxIndCd": str(index_code),
        "strtDd": str(start),
        "endDd": str(end),
        # Some KRX services use a single-date key. Keeping basDd as compatibility hint.
        "basDd": str(end),
    }

    payload = _request_with_retry(url=endpoint, auth_key=key, params=params, session=session)
    rows = _extract_rows(payload)
    if not rows:
        raise KRXOpenAPIResponseError("KRX OpenAPI returned no data rows")

    records: list[tuple[pd.Timestamp, float]] = []
    target_code = str(index_code).strip()
    for row in rows:
        row_code = _parse_row_index_code(row)
        if row_code and row_code != target_code:
            continue

        row_date = _parse_row_date(row)
        if row_date is None or row_date < start_ts or row_date > end_ts:
            continue

        close = _parse_row_close(row)
        if close is None:
            continue

        records.append((row_date, close))

    if not records:
        raise KRXOpenAPIResponseError(
            f"KRX OpenAPI rows did not contain parseable close data for {target_code}"
        )

    frame = pd.DataFrame(records, columns=["date", "close"]).drop_duplicates(
        subset=["date"], keep="last"
    )
    frame = frame.sort_values("date").set_index("date")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame.astype({"close": "float64"})
