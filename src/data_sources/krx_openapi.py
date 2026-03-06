"""
KRX OpenAPI helpers for index daily close retrieval.

This module provides:
- Runtime config loading for KRX provider mode and OpenAPI key.
- OpenAPI request/response handling with auth-aware exceptions.
- Parsing utilities that normalize API responses into OHLCV-like DataFrames.
- Family-based snapshot collection for KRX/KOSPI/KOSDAQ daily index services.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import pandas as pd
import requests

logger = logging.getLogger(__name__)

KRXProvider = Literal["AUTO", "OPENAPI", "PYKRX"]
OpenAPIFamily = Literal["KRX", "KOSPI", "KOSDAQ"]

DEFAULT_KRX_PROVIDER: KRXProvider = "AUTO"
OPENAPI_BASE_URL = "https://data-dbg.krx.co.kr"
OPENAPI_HOST = "data-dbg.krx.co.kr"
OPENAPI_PATH_PREFIX = "/svc/apis/idx"
OPENAPI_API_IDS: dict[OpenAPIFamily, str] = {
    "KRX": "krx_dd_trd",
    "KOSPI": "kospi_dd_trd",
    "KOSDAQ": "kosdaq_dd_trd",
}
DEFAULT_OPENAPI_API_ID = OPENAPI_API_IDS["KRX"]
OPENAPI_INDEX_DAILY_PATH = f"{OPENAPI_PATH_PREFIX}/{DEFAULT_OPENAPI_API_ID}"

OPENAPI_BATCH_WORKERS = 4

OPENAPI_CODE_FAMILY_OVERRIDES: dict[str, OpenAPIFamily] = {
    "1001": "KOSPI",
    "1155": "KOSPI",
    "1157": "KOSPI",
    "1165": "KOSPI",
    "1168": "KOSPI",
    "1170": "KOSPI",
    "5042": "KRX",
    "5044": "KRX",
    "5045": "KRX",
    "5046": "KRX",
    "5048": "KRX",
    "5049": "KRX",
}
MANUAL_INDEX_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "1001": ("\ucf54\uc2a4\ud53c",),
    "5042": ("KRX 300 \uc0b0\uc5c5\uc7ac",),
    "5046": (
        "KRX \ubc29\uc1a1\ud1b5\uc2e0",
        "KRX 300 \ucee4\ubba4\ub2c8\ucf00\uc774\uc158\uc11c\ube44\uc2a4",
    ),
    "1170": ("\uc804\uae30\u00b7\uac00\uc2a4",),
}
ROW_NAME_KEYS = ("IDX_NM", "idxNm", "name", "NAME")

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


def _build_openapi_url(api_id: str) -> str:
    """Return canonical OpenAPI URL for a supported API id."""
    return f"{OPENAPI_BASE_URL}{OPENAPI_PATH_PREFIX}/{api_id}"


def _validate_openapi_url_override(custom: str, api_id: str) -> str:
    """Accept only official KRX OpenAPI host + exact supported path."""
    parsed = urlparse(custom)
    expected = _build_openapi_url(api_id)
    if (
        parsed.scheme == "https"
        and parsed.netloc.lower() == OPENAPI_HOST
        and parsed.path.rstrip("/") == f"{OPENAPI_PATH_PREFIX}/{api_id}"
        and not parsed.query
        and not parsed.fragment
    ):
        return custom.rstrip("/")

    logger.warning(
        "Ignoring invalid KRX_OPENAPI_URL override %r; expected %s",
        custom,
        expected,
    )
    return expected


def get_krx_openapi_url(api_id: str | None = None) -> str:
    """Return validated OpenAPI endpoint URL for the requested family API."""
    resolved_api_id = str(api_id or DEFAULT_OPENAPI_API_ID).strip() or DEFAULT_OPENAPI_API_ID
    custom = _load_secret_or_env("KRX_OPENAPI_URL")
    if custom:
        return _validate_openapi_url_override(custom.strip(), resolved_api_id)
    return _build_openapi_url(resolved_api_id)


def resolve_openapi_family(index_code: str) -> OpenAPIFamily:
    """Resolve KRX OpenAPI family for a dashboard index code."""
    code = str(index_code).strip()
    if code in OPENAPI_CODE_FAMILY_OVERRIDES:
        return OPENAPI_CODE_FAMILY_OVERRIDES[code]
    if code.startswith("5"):
        return "KRX"
    if code.startswith("1"):
        return "KOSPI"
    if code.startswith("2"):
        return "KOSDAQ"
    raise KRXOpenAPIResponseError(f"Unable to resolve KRX OpenAPI family for index code {code}")


def resolve_openapi_api_id(index_code: str) -> str:
    """Resolve family API id for a dashboard index code."""
    return OPENAPI_API_IDS[resolve_openapi_family(index_code)]


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


def _repair_mojibake_text(value: str) -> str:
    """Repair common cp949/latin1 mojibake seen in KRX JSON strings."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return text.encode("latin1").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _normalize_index_name(value: str) -> str:
    """Normalize index display text for alias matching."""
    repaired = _repair_mojibake_text(value)
    return " ".join(repaired.split())


def _row_name(row: dict[str, Any]) -> str:
    """Return normalized row display name."""
    return _normalize_index_name(_row_get(row, ROW_NAME_KEYS))


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


@lru_cache(maxsize=1)
def _load_index_name_metadata() -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """Load display names and aliases from local sector-map config."""
    import yaml

    display_names: dict[str, str] = {}
    aliases: dict[str, set[str]] = {}
    config_path = Path("config/sector_map.yml")
    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            sector_map = yaml.safe_load(f) or {}

        benchmark = sector_map.get("benchmark", {})
        benchmark_code = str(benchmark.get("code", "")).strip()
        benchmark_name = _normalize_index_name(str(benchmark.get("name", "")).strip())
        if benchmark_code:
            if benchmark_name:
                display_names[benchmark_code] = benchmark_name
                aliases.setdefault(benchmark_code, set()).add(benchmark_name)
            if benchmark_code == "1001":
                aliases.setdefault(benchmark_code, set()).add("\ucf54\uc2a4\ud53c")

        for regime_data in sector_map.get("regimes", {}).values():
            for sector in regime_data.get("sectors", []):
                code = str(sector.get("code", "")).strip()
                name = _normalize_index_name(str(sector.get("name", "")).strip())
                if not code:
                    continue
                if name:
                    display_names.setdefault(code, name)
                    aliases.setdefault(code, set()).add(name)
                    if name.startswith("KOSPI200"):
                        suffix = name[len("KOSPI200"):].strip()
                        aliases.setdefault(code, set()).add("\ucf54\uc2a4\ud53c 200")
                        if suffix:
                            aliases.setdefault(code, set()).add(f"\ucf54\uc2a4\ud53c 200 {suffix}")

    for code, extra_aliases in MANUAL_INDEX_NAME_ALIASES.items():
        aliases.setdefault(code, set()).update(_normalize_index_name(alias) for alias in extra_aliases)

    frozen_aliases = {
        code: tuple(sorted(name for name in names if name))
        for code, names in aliases.items()
    }
    return display_names, frozen_aliases


def get_index_display_name(index_code: str) -> str:
    """Return display name for an index code when known."""
    display_names, _ = _load_index_name_metadata()
    code = str(index_code).strip()
    return display_names.get(code, code)


def resolve_index_name_aliases(index_code: str) -> tuple[str, ...]:
    """Return normalized candidate row names for a dashboard index code."""
    _, aliases = _load_index_name_metadata()
    code = str(index_code).strip()
    if code in aliases:
        return aliases[code]

    fallback = get_index_display_name(code)
    normalized = _normalize_index_name(fallback)
    return (normalized,) if normalized else (code,)


def _business_days(start: str, end: str) -> list[str]:
    """Return weekday-only YYYYMMDD dates between start and end."""
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="B")
    return [ts.strftime("%Y%m%d") for ts in idx]


def _fetch_snapshot_rows(
    *,
    api_id: str,
    bas_dd: str,
    auth_key: str,
    url: str | None = None,
    session: requests.Session | None = None,
) -> tuple[str, dict[str, float]]:
    """Fetch one daily snapshot and normalize it into name -> close mapping."""
    endpoint = (url or get_krx_openapi_url(api_id)).strip()
    payload = _request_with_retry(
        url=endpoint,
        auth_key=auth_key,
        params={"basDd": str(bas_dd)},
        session=session,
    )
    rows = _extract_rows(payload)
    if not rows:
        return bas_dd, {}

    bas_ts = pd.Timestamp(bas_dd)
    parsed: dict[str, float] = {}
    for row in rows:
        row_name = _row_name(row)
        row_date = _parse_row_date(row)
        close = _parse_row_close(row)
        if not row_name or row_date is None or row_date != bas_ts or close is None:
            continue
        parsed[row_name] = close

    if not parsed:
        raise KRXOpenAPIResponseError(
            f"KRX OpenAPI snapshot {api_id} for {bas_dd} did not contain parseable name/close rows"
        )
    return bas_dd, parsed


def fetch_index_ohlcv_openapi_batch(
    index_codes: list[str],
    start: str,
    end: str,
    *,
    auth_key: str | None = None,
    url: str | None = None,
    session: requests.Session | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Fetch historical close series by replaying daily family snapshots."""
    key = (auth_key or get_krx_openapi_key()).strip()
    if not key:
        raise ValueError("KRX_OPENAPI_KEY not configured")

    normalized_codes = [str(code).strip() for code in index_codes if str(code).strip()]
    if not normalized_codes:
        return {}, {}

    grouped_codes: dict[str, list[str]] = {}
    for code in normalized_codes:
        api_id = resolve_openapi_api_id(code)
        grouped_codes.setdefault(api_id, []).append(code)

    if url and len(grouped_codes) > 1:
        raise ValueError("Explicit url override can only be used for a single OpenAPI family")

    bas_days = _business_days(start, end)
    successes: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}

    for api_id, codes in grouped_codes.items():
        aliases_by_code = {
            code: tuple(_normalize_index_name(alias) for alias in resolve_index_name_aliases(code))
            for code in codes
        }
        records_by_code: dict[str, list[tuple[pd.Timestamp, float]]] = {code: [] for code in codes}
        endpoint = (url or get_krx_openapi_url(api_id)).strip()
        saw_non_empty_snapshot = False

        snapshots: list[tuple[str, dict[str, float]]] = []
        if session is not None or len(bas_days) <= 1:
            for bas_dd in bas_days:
                snapshots.append(
                    _fetch_snapshot_rows(
                        api_id=api_id,
                        bas_dd=bas_dd,
                        auth_key=key,
                        url=endpoint,
                        session=session,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=min(OPENAPI_BATCH_WORKERS, len(bas_days))) as executor:
                future_map = {
                    executor.submit(
                        _fetch_snapshot_rows,
                        api_id=api_id,
                        bas_dd=bas_dd,
                        auth_key=key,
                        url=endpoint,
                    ): bas_dd
                    for bas_dd in bas_days
                }
                for future in as_completed(future_map):
                    snapshots.append(future.result())

        for bas_dd, snapshot in sorted(snapshots, key=lambda item: item[0]):
            if not snapshot:
                continue
            saw_non_empty_snapshot = True
            bas_ts = pd.Timestamp(bas_dd)
            for code in codes:
                matched = False
                for alias in aliases_by_code[code]:
                    if alias in snapshot:
                        records_by_code[code].append((bas_ts, float(snapshot[alias])))
                        matched = True
                        break
                if not matched:
                    continue

        for code in codes:
            records = records_by_code[code]
            if not records:
                if not saw_non_empty_snapshot:
                    failures[code] = "KRX OpenAPI returned no data rows"
                else:
                    failures[code] = (
                        "KRX OpenAPI snapshot rows did not contain a matched series "
                        f"for {code} (aliases={aliases_by_code[code]!r})"
                    )
                continue
            frame = pd.DataFrame(records, columns=["date", "close"]).drop_duplicates(
                subset=["date"], keep="last"
            )
            frame = frame.sort_values("date").set_index("date")
            frame.index = pd.DatetimeIndex(frame.index)
            successes[code] = frame.astype({"close": "float64"})

    return successes, failures


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
    successes, failures = fetch_index_ohlcv_openapi_batch(
        [index_code],
        start,
        end,
        auth_key=auth_key,
        url=url,
        session=session,
    )
    code = str(index_code).strip()
    if code in successes:
        return successes[code]

    reason = failures.get(code, "KRX OpenAPI returned no data rows")
    raise KRXOpenAPIResponseError(reason)
