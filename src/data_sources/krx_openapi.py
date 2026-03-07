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
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import requests

logger = logging.getLogger(__name__)

KRXProvider = Literal["AUTO", "OPENAPI", "PYKRX"]
OpenAPIFamily = Literal["KRX", "KOSPI", "KOSDAQ"]
OpenAPIHealthStatus = Literal["OK", "AUTH_ERROR", "ACCESS_DENIED", "HTTP_ERROR"]

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
INDEX_NAME_METADATA_PATH = Path("data/raw/krx/index_name_metadata.json")
INDEX_NAME_METADATA_SCHEMA_VERSION = 1

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
EMERGENCY_INDEX_NAME_ALIASES: dict[str, tuple[str, ...]] = {
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
THROTTLE_RETRY_DELAY_SEC = 1.0
FORCE_BATCH_WORKERS = 4
OPENAPI_HEALTHCHECK_BAS_DD = "20240131"
OPENAPI_BATCH_CHUNK_DAYS = 10
OPENAPI_BATCH_MAX_CONCURRENCY = 2

_SUCCESS_CODES = {"0", "00", "200", "0000"}
_AUTH_CODES = {"401"}
_PERMISSION_CODES = {"403"}
_OPENAPI_THREAD_LOCAL = threading.local()
_OPENAPI_URL_OVERRIDE_WARNED = False
_OPENAPI_URL_OVERRIDE_LOCK = threading.Lock()


class KRXOpenAPIError(RuntimeError):
    """Base OpenAPI error."""


class KRXOpenAPIAuthError(KRXOpenAPIError):
    """Raised when AUTH_KEY is missing/invalid or rejected."""


class KRXOpenAPIPermissionError(KRXOpenAPIError):
    """Raised when key is valid but service permission is missing."""


class KRXOpenAPIResponseError(KRXOpenAPIError):
    """Raised when response payload is malformed or empty."""


class KRXOpenAPIAccessDeniedError(KRXOpenAPIResponseError):
    """Raised when KRX edge/CDN denies the request before JSON is returned."""


class KRXOpenAPIThrottleError(KRXOpenAPIAccessDeniedError):
    """Deprecated alias kept for backwards compatibility."""


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


def _load_int_setting(name: str, default: int) -> int:
    """Load an integer setting from secrets/env with a sane fallback."""
    raw = _load_secret_or_env(name)
    if not raw:
        return int(default)
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to %d.", name, raw, default)
        return int(default)


def get_openapi_batch_workers() -> int:
    """Return configured worker count for OpenAPI snapshot collection."""
    return _load_int_setting("KRX_OPENAPI_BATCH_WORKERS", 8)


def resolve_openapi_batch_workers(total_requests: int, *, force: bool = False) -> int:
    """Return effective worker count for an OpenAPI batch."""
    configured = get_openapi_batch_workers()
    limit = min(configured, FORCE_BATCH_WORKERS) if force else configured
    return max(1, min(limit, max(1, int(total_requests))))


def get_index_metadata_ttl_hours() -> int:
    """Return metadata cache TTL in hours."""
    return _load_int_setting("KRX_INDEX_METADATA_TTL_HOURS", 24)


def _get_openapi_session() -> requests.Session:
    """Return a thread-local requests.Session for OpenAPI calls."""
    session = getattr(_OPENAPI_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _OPENAPI_THREAD_LOCAL.session = session
    return session


def _build_openapi_url(api_id: str) -> str:
    """Return canonical OpenAPI URL for a supported API id."""
    return f"{OPENAPI_BASE_URL}{OPENAPI_PATH_PREFIX}/{api_id}"


def _warn_deprecated_openapi_url_override(custom: str) -> None:
    """Warn once per process that `KRX_OPENAPI_URL` is no longer used."""
    global _OPENAPI_URL_OVERRIDE_WARNED
    if _OPENAPI_URL_OVERRIDE_WARNED:
        return
    with _OPENAPI_URL_OVERRIDE_LOCK:
        if _OPENAPI_URL_OVERRIDE_WARNED:
            return
        logger.warning(
            "Ignoring deprecated KRX_OPENAPI_URL override %r; family endpoints are resolved internally.",
            custom,
        )
        _OPENAPI_URL_OVERRIDE_WARNED = True


def get_krx_openapi_url(api_id: str | None = None) -> str:
    """Return canonical OpenAPI endpoint URL for the requested family API."""
    resolved_api_id = str(api_id or DEFAULT_OPENAPI_API_ID).strip() or DEFAULT_OPENAPI_API_ID
    custom = _load_secret_or_env("KRX_OPENAPI_URL")
    if custom:
        _warn_deprecated_openapi_url_override(custom.strip())
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


def _looks_like_access_denied(text: str) -> bool:
    lowered = str(text or "").lower()
    return "access denied" in lowered or ("denied" in lowered and "<html" in lowered)


def _request_with_retry(
    *,
    url: str,
    auth_key: str,
    params: dict[str, str],
    session: requests.Session | None = None,
    timeout_sec: int = REQUEST_TIMEOUT,
) -> Any:
    """Issue OpenAPI request with retry policy and auth-aware error handling."""
    if not auth_key:
        raise ValueError("KRX_OPENAPI_KEY not configured")

    last_exc: Exception | None = None
    requester = session or requests
    headers = {"AUTH_KEY": auth_key, "User-Agent": "sector-rotation/1.0"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requester.get(url, params=params, headers=headers, timeout=timeout_sec)
            payload: Any
            try:
                payload = resp.json()
            except ValueError as exc:
                text_preview = (resp.text or "")[:200]
                if _looks_like_access_denied(text_preview):
                    raise KRXOpenAPIAccessDeniedError(
                        f"KRX OpenAPI returned Access Denied payload: {text_preview!r}"
                    ) from exc
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
        except (KRXOpenAPIAuthError, KRXOpenAPIPermissionError, KRXOpenAPIAccessDeniedError):
            raise
        except (requests.Timeout, requests.ConnectionError, KRXOpenAPIResponseError) as exc:
            last_exc = exc
            is_throttle = isinstance(exc, KRXOpenAPIThrottleError)
            max_attempts = 2 if is_throttle else MAX_RETRIES
            is_last = attempt >= max_attempts - 1
            if is_last:
                break
            logger.warning(
                "KRX OpenAPI attempt %d/%d failed: %s",
                attempt + 1,
                max_attempts,
                exc,
            )
            time.sleep(THROTTLE_RETRY_DELAY_SEC if is_throttle else BACKOFF_BASE ** (attempt + 1))
        except requests.RequestException as exc:
            raise KRXOpenAPIError(f"KRX OpenAPI request exception: {exc}") from exc

    assert last_exc is not None
    raise last_exc


@lru_cache(maxsize=1)
def _probe_krx_openapi_health_cached(timeout_sec: int) -> tuple[OpenAPIHealthStatus, str, str, str]:
    """Probe one known-good OpenAPI snapshot so UI can distinguish edge denial from connectivity."""
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    url = get_krx_openapi_url(DEFAULT_OPENAPI_API_ID)
    auth_key = get_krx_openapi_key()
    if not auth_key:
        return ("AUTH_ERROR", "KRX_OPENAPI_KEY not configured", url, checked_at)

    headers = {"AUTH_KEY": auth_key, "User-Agent": "sector-rotation/1.0"}
    params = {"basDd": OPENAPI_HEALTHCHECK_BAS_DD}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=max(1, int(timeout_sec)))
    except requests.RequestException as exc:
        return ("HTTP_ERROR", str(exc), url, checked_at)

    try:
        payload = resp.json()
    except ValueError:
        text_preview = (resp.text or "")[:200]
        if _looks_like_access_denied(text_preview):
            return (
                "ACCESS_DENIED",
                f"KRX OpenAPI returned Access Denied payload: {text_preview!r}",
                url,
                checked_at,
            )
        return ("HTTP_ERROR", f"KRX OpenAPI returned non-JSON payload: {text_preview!r}", url, checked_at)

    resp_code, resp_msg = _normalize_resp_meta(payload)
    try:
        _raise_by_status_or_code(resp.status_code, resp_code, resp_msg)
    except (KRXOpenAPIAuthError, KRXOpenAPIPermissionError) as exc:
        return ("AUTH_ERROR", str(exc), url, checked_at)

    if resp.status_code >= 400:
        detail = resp_msg or f"HTTP {resp.status_code}"
        return ("HTTP_ERROR", detail, url, checked_at)

    if resp_code and resp_code not in _SUCCESS_CODES:
        detail = resp_msg or f"respCode={resp_code}"
        return ("HTTP_ERROR", detail, url, checked_at)

    rows = _extract_rows(payload)
    if not rows:
        return ("HTTP_ERROR", "KRX OpenAPI returned no data rows", url, checked_at)

    return ("OK", f"HTTP {resp.status_code}", url, checked_at)


def probe_krx_openapi_health(timeout_sec: int = 3) -> dict[str, str]:
    """Return a cached health classification for the real KRX OpenAPI endpoint."""
    status, detail, url, checked_at = _probe_krx_openapi_health_cached(int(timeout_sec))
    return {
        "status": status,
        "detail": detail,
        "url": url,
        "checked_at": checked_at,
    }


def reset_krx_openapi_health_cache() -> None:
    """Clear cached OpenAPI health so manual refresh can force a fresh probe."""
    _probe_krx_openapi_health_cached.cache_clear()


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
def _load_config_index_display_names() -> dict[str, str]:
    """Load canonical UI display names from local sector-map config."""
    import yaml

    display_names: dict[str, str] = {}
    config_path = Path("config/sector_map.yml")
    if not config_path.exists():
        return display_names

    with config_path.open(encoding="utf-8") as f:
        sector_map = yaml.safe_load(f) or {}

    benchmark = sector_map.get("benchmark", {})
    benchmark_code = str(benchmark.get("code", "")).strip()
    benchmark_name = _normalize_index_name(str(benchmark.get("name", "")).strip())
    if benchmark_code and benchmark_name:
        display_names[benchmark_code] = benchmark_name

    for regime_data in sector_map.get("regimes", {}).values():
        for sector in regime_data.get("sectors", []):
            code = str(sector.get("code", "")).strip()
            name = _normalize_index_name(str(sector.get("name", "")).strip())
            if code and name:
                display_names.setdefault(code, name)

    return display_names


def _empty_index_name_metadata() -> dict[str, Any]:
    return {
        "schema_version": INDEX_NAME_METADATA_SCHEMA_VERSION,
        "updated_at": "",
        "codes": {},
    }


def _read_index_name_metadata() -> dict[str, Any]:
    """Load persisted index-name metadata, tolerating missing/corrupt cache."""
    path = INDEX_NAME_METADATA_PATH
    if not path.exists():
        return _empty_index_name_metadata()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Index metadata cache load failed: %s", exc)
        return _empty_index_name_metadata()

    if not isinstance(payload, dict):
        return _empty_index_name_metadata()
    if not isinstance(payload.get("codes"), dict):
        payload["codes"] = {}
    payload.setdefault("schema_version", INDEX_NAME_METADATA_SCHEMA_VERSION)
    payload.setdefault("updated_at", "")
    return payload


def _write_index_name_metadata(payload: dict[str, Any]) -> None:
    """Persist index-name metadata and clear in-process caches."""
    INDEX_NAME_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload["schema_version"] = INDEX_NAME_METADATA_SCHEMA_VERSION
    payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    INDEX_NAME_METADATA_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _load_index_name_metadata.cache_clear()


def is_index_name_metadata_stale() -> bool:
    """Return True when the persisted metadata cache is missing or stale."""
    path = INDEX_NAME_METADATA_PATH
    if not path.exists():
        return True
    age_seconds = max(0.0, time.time() - path.stat().st_mtime)
    return age_seconds > float(get_index_metadata_ttl_hours() * 3600)


def _bootstrap_aliases_from_display_name(display_name: str) -> tuple[str, ...]:
    """Return bootstrap aliases derived from the canonical UI display name."""
    normalized = _normalize_index_name(display_name)
    if not normalized:
        return ()

    aliases: list[str] = [normalized]
    if normalized == "KOSPI":
        aliases.append("\ucf54\uc2a4\ud53c")
    if normalized.startswith("KOSPI200"):
        suffix = normalized[len("KOSPI200"):].strip()
        if suffix:
            aliases.append(f"\ucf54\uc2a4\ud53c 200 {suffix}")
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _is_overbroad_alias(alias: str, display_name: str) -> bool:
    """Return True for aliases that are too broad to identify one configured code."""
    normalized_alias = _normalize_index_name(alias)
    normalized_display = _normalize_index_name(display_name)
    return (
        normalized_alias in {"KOSPI200", "\ucf54\uc2a4\ud53c 200"}
        and normalized_display.startswith("KOSPI200")
        and normalized_display != "KOSPI200"
    )


@lru_cache(maxsize=1)
def _load_index_name_metadata() -> dict[str, dict[str, Any]]:
    """Return normalized metadata rows keyed by dashboard code."""
    payload = _read_index_name_metadata()
    result: dict[str, dict[str, Any]] = {}
    for raw_code, raw_entry in payload.get("codes", {}).items():
        code = str(raw_code).strip()
        if not code or not isinstance(raw_entry, dict):
            continue
        official = _normalize_index_name(str(raw_entry.get("official_name", "")).strip())
        history_raw = raw_entry.get("alias_history", [])
        if not isinstance(history_raw, list):
            history_raw = []
        history = tuple(
            dict.fromkeys(
                alias
                for alias in (
                    _normalize_index_name(str(item).strip())
                    for item in history_raw
                )
                if alias and alias != official
            )
        )
        result[code] = {
            "official_name": official,
            "alias_history": history,
            "last_synced_at": str(raw_entry.get("last_synced_at", "")).strip(),
        }
    return result


def update_index_name_metadata(observed_names_by_code: dict[str, list[str] | tuple[str, ...]]) -> None:
    """Persist observed official names and keep previous names as alias history."""
    if not observed_names_by_code:
        return

    payload = _read_index_name_metadata()
    codes_payload = payload.setdefault("codes", {})
    changed = False
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    for raw_code, raw_names in observed_names_by_code.items():
        code = str(raw_code).strip()
        if not code:
            continue
        ordered_names = [
            alias
            for alias in (
                _normalize_index_name(str(name).strip())
                for name in (raw_names or [])
            )
            if alias
        ]
        ordered_names = list(dict.fromkeys(ordered_names))
        if not ordered_names:
            continue

        latest_name = ordered_names[-1]
        entry = codes_payload.get(code, {})
        if not isinstance(entry, dict):
            entry = {}
        previous_official = _normalize_index_name(str(entry.get("official_name", "")).strip())
        history_raw = entry.get("alias_history", [])
        if not isinstance(history_raw, list):
            history_raw = []
        history = [
            alias
            for alias in (
                _normalize_index_name(str(item).strip())
                for item in history_raw
            )
            if alias
        ]

        for alias in ordered_names[:-1]:
            if alias != latest_name and alias not in history:
                history.append(alias)
        if previous_official and previous_official != latest_name and previous_official not in history:
            history.append(previous_official)

        next_entry = {
            "official_name": latest_name,
            "alias_history": history[-12:],
            "last_synced_at": now,
        }
        if entry != next_entry:
            codes_payload[code] = next_entry
            changed = True
        else:
            entry["last_synced_at"] = now
            codes_payload[code] = entry
            changed = True

    if changed:
        _write_index_name_metadata(payload)


def get_index_display_name(index_code: str) -> str:
    """Return canonical UI display name for an index code when known."""
    code = str(index_code).strip()
    display_names = _load_config_index_display_names()
    if code in display_names:
        return display_names[code]

    metadata = _load_index_name_metadata()
    official = str(metadata.get(code, {}).get("official_name", "")).strip()
    return official or code


def resolve_index_name_aliases(index_code: str) -> tuple[str, ...]:
    """Return candidate row names in matching priority order."""
    code = str(index_code).strip()
    metadata = _load_index_name_metadata().get(code, {})
    display_name = get_index_display_name(code)
    aliases: list[str] = []

    official_name = _normalize_index_name(str(metadata.get("official_name", "")).strip())
    if official_name and not _is_overbroad_alias(official_name, display_name):
        aliases.append(official_name)

    for alias in metadata.get("alias_history", ()):
        normalized = _normalize_index_name(str(alias).strip())
        if normalized and not _is_overbroad_alias(normalized, display_name):
            aliases.append(normalized)

    for alias in EMERGENCY_INDEX_NAME_ALIASES.get(code, ()):
        normalized = _normalize_index_name(alias)
        if normalized:
            aliases.append(normalized)

    for alias in _bootstrap_aliases_from_display_name(display_name):
        aliases.append(alias)

    deduped = tuple(dict.fromkeys(alias for alias in aliases if alias))
    return deduped or (code,)


def _business_days(start: str, end: str) -> list[str]:
    """Return weekday-only YYYYMMDD dates between start and end."""
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="B")
    return [ts.strftime("%Y%m%d") for ts in idx]


def _chunk_business_days(bas_days: list[str], size: int = OPENAPI_BATCH_CHUNK_DAYS) -> list[list[str]]:
    """Split business days into fixed-size chunks for bounded OpenAPI replay."""
    chunk_size = max(1, int(size))
    return [bas_days[idx: idx + chunk_size] for idx in range(0, len(bas_days), chunk_size)]


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


def _fetch_snapshot_rows_threadsafe(
    *,
    api_id: str,
    bas_dd: str,
    auth_key: str,
    url: str,
) -> tuple[str, dict[str, float]]:
    """Fetch one snapshot using the thread-local OpenAPI session."""
    return _fetch_snapshot_rows(
        api_id=api_id,
        bas_dd=bas_dd,
        auth_key=auth_key,
        url=url,
        session=_get_openapi_session(),
    )


def fetch_index_ohlcv_openapi_batch_detailed(
    index_codes: list[str],
    start: str,
    end: str,
    *,
    auth_key: str | None = None,
    url: str | None = None,
    session: requests.Session | None = None,
    force: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, Any]]:
    """Fetch close series by replaying daily family snapshots with partial-failure details."""
    key = (auth_key or get_krx_openapi_key()).strip()
    if not key:
        raise ValueError("KRX_OPENAPI_KEY not configured")

    normalized_codes = [str(code).strip() for code in index_codes if str(code).strip()]
    if not normalized_codes:
        return (
            {},
            {},
            {
                "failed_days": [],
                "snapshot_failures": {},
                "request_count": 0,
                "processed_requests": 0,
                "aborted": False,
                "abort_reason": "",
            },
        )

    grouped_codes: dict[str, list[str]] = {}
    for code in normalized_codes:
        api_id = resolve_openapi_api_id(code)
        grouped_codes.setdefault(api_id, []).append(code)

    if url and len(grouped_codes) > 1:
        raise ValueError("Explicit url override can only be used for a single OpenAPI family")

    bas_days = _business_days(start, end)
    successes: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    family_snapshots: dict[str, list[tuple[str, dict[str, float]]]] = {
        api_id: [] for api_id in grouped_codes
    }
    snapshot_failures: dict[str, dict[str, str]] = {api_id: {} for api_id in grouped_codes}
    endpoints = {api_id: (url or get_krx_openapi_url(api_id)).strip() for api_id in grouped_codes}
    total_requests = len(bas_days) * len(grouped_codes)
    processed_requests = 0
    aborted = False
    abort_reason = ""
    abort_detail = ""

    day_chunks = _chunk_business_days(bas_days)

    def _record_failure(api_id: str, bas_dd: str, exc: Exception) -> None:
        nonlocal aborted, abort_reason, abort_detail
        snapshot_failures[api_id][bas_dd] = str(exc)
        if isinstance(exc, KRXOpenAPIAccessDeniedError) and not aborted:
            aborted = True
            abort_reason = "ACCESS_DENIED"
            abort_detail = str(exc)

    def _chunk_tasks(day_chunk: list[str]) -> list[tuple[str, str]]:
        return [
            (api_id, bas_dd)
            for api_id in grouped_codes
            for bas_dd in day_chunk
        ]

    if session is not None or total_requests <= 1:
        for day_chunk in day_chunks:
            if aborted:
                break
            for api_id, bas_dd in _chunk_tasks(day_chunk):
                processed_requests += 1
                try:
                    family_snapshots[api_id].append(
                        _fetch_snapshot_rows(
                            api_id=api_id,
                            bas_dd=bas_dd,
                            auth_key=key,
                            url=endpoints[api_id],
                            session=session,
                        )
                    )
                except Exception as exc:
                    _record_failure(api_id, bas_dd, exc)
                    if aborted:
                        break
            if aborted:
                break
    else:
        worker_count = min(
            resolve_openapi_batch_workers(total_requests, force=force),
            OPENAPI_BATCH_MAX_CONCURRENCY,
        )
        for day_chunk in day_chunks:
            if aborted:
                break

            tasks = iter(_chunk_tasks(day_chunk))
            with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
                future_map: dict[Any, tuple[str, str]] = {}

                def _submit_next() -> bool:
                    nonlocal processed_requests
                    try:
                        api_id, bas_dd = next(tasks)
                    except StopIteration:
                        return False
                    future = executor.submit(
                        _fetch_snapshot_rows_threadsafe,
                        api_id=api_id,
                        bas_dd=bas_dd,
                        auth_key=key,
                        url=endpoints[api_id],
                    )
                    future_map[future] = (api_id, bas_dd)
                    processed_requests += 1
                    return True

                while len(future_map) < max(1, worker_count) and _submit_next():
                    pass

                while future_map:
                    future = next(as_completed(tuple(future_map)))
                    api_id, bas_dd = future_map.pop(future)
                    try:
                        family_snapshots[api_id].append(future.result())
                    except Exception as exc:
                        _record_failure(api_id, bas_dd, exc)
                        if aborted:
                            for pending in list(future_map):
                                pending.cancel()
                            break

                    if not aborted:
                        _submit_next()

            if aborted:
                break

    if aborted:
        logger.warning(
            "KRX OpenAPI batch aborted start=%s end=%s processed=%d/%d reason=%s detail=%s",
            start,
            end,
            processed_requests,
            total_requests,
            abort_reason or "unknown",
            abort_detail,
        )

    observed_names_by_code: dict[str, list[str]] = {}
    for api_id, codes in grouped_codes.items():
        started_at = time.perf_counter()
        aliases_by_code = {
            code: tuple(_normalize_index_name(alias) for alias in resolve_index_name_aliases(code))
            for code in codes
        }
        records_by_code: dict[str, list[tuple[pd.Timestamp, float]]] = {code: [] for code in codes}
        matched_names_by_code: dict[str, list[str]] = {code: [] for code in codes}
        saw_non_empty_snapshot = False

        for bas_dd, snapshot in sorted(family_snapshots[api_id], key=lambda item: item[0]):
            if not snapshot:
                continue
            saw_non_empty_snapshot = True
            bas_ts = pd.Timestamp(bas_dd)
            for code in codes:
                for alias in aliases_by_code[code]:
                    if alias in snapshot:
                        records_by_code[code].append((bas_ts, float(snapshot[alias])))
                        matched_names_by_code[code].append(alias)
                        break

        for code in codes:
            matched_names = list(dict.fromkeys(name for name in matched_names_by_code[code] if name))
            if matched_names:
                observed_names_by_code[code] = matched_names

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

        logger.info(
            "KRX OpenAPI batch family=%s start=%s end=%s requests=%d codes=%d matched=%d failed_days=%d duration_sec=%.2f metadata_stale=%s",
            api_id,
            start,
            end,
            len(bas_days),
            len(codes),
            sum(1 for code in codes if code in successes),
            len(snapshot_failures.get(api_id, {})),
            time.perf_counter() - started_at,
            is_index_name_metadata_stale(),
        )

    if observed_names_by_code:
        update_index_name_metadata(observed_names_by_code)

    failed_days = sorted(
        {
            bas_dd
            for family_failures in snapshot_failures.values()
            for bas_dd in family_failures
        }
    )
    details = {
        "failed_days": failed_days,
        "snapshot_failures": snapshot_failures,
        "request_count": total_requests,
        "processed_requests": processed_requests,
        "aborted": aborted,
        "abort_reason": abort_reason,
    }
    return successes, failures, details


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
    successes, failures, _details = fetch_index_ohlcv_openapi_batch_detailed(
        index_codes,
        start,
        end,
        auth_key=auth_key,
        url=url,
        session=session,
        force=False,
    )
    return successes, failures


def audit_index_name_aliases(
    index_codes: list[str],
    audit_date: str,
    *,
    auth_key: str | None = None,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Audit whether configured aliases can match one known-good snapshot date."""
    key = (auth_key or get_krx_openapi_key()).strip()
    if not key:
        raise ValueError("KRX_OPENAPI_KEY not configured")
    audit_date = "".join(ch for ch in str(audit_date).strip() if ch.isdigit()) or str(audit_date).strip()

    normalized_codes = [str(code).strip() for code in index_codes if str(code).strip()]
    grouped_codes: dict[str, list[str]] = {}
    for code in normalized_codes:
        grouped_codes.setdefault(resolve_openapi_api_id(code), []).append(code)

    results: list[dict[str, Any]] = []
    observed_names_by_code: dict[str, list[str]] = {}
    for api_id, codes in grouped_codes.items():
        _bas_dd, snapshot = _fetch_snapshot_rows(
            api_id=api_id,
            bas_dd=audit_date,
            auth_key=key,
            url=get_krx_openapi_url(api_id),
            session=session,
        )
        for code in codes:
            aliases = resolve_index_name_aliases(code)
            matched_name = next((alias for alias in aliases if alias in snapshot), "")
            if matched_name:
                observed_names_by_code.setdefault(code, []).append(matched_name)
            results.append(
                {
                    "code": code,
                    "family": api_id,
                    "audit_date": audit_date,
                    "matched": bool(matched_name),
                    "matched_name": matched_name,
                    "aliases": aliases,
                    "snapshot_rows": len(snapshot),
                }
            )

    if observed_names_by_code:
        update_index_name_metadata(observed_names_by_code)
    return results


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
    successes, failures, details = fetch_index_ohlcv_openapi_batch_detailed(
        [index_code],
        start,
        end,
        auth_key=auth_key,
        url=url,
        session=session,
        force=False,
    )
    code = str(index_code).strip()
    if code in successes:
        return successes[code]

    snapshot_failures = details.get("snapshot_failures", {})
    if details.get("abort_reason") == "ACCESS_DENIED":
        first_access_denied = next(
            (
                reason
                for family_failures in snapshot_failures.values()
                for reason in family_failures.values()
                if "access denied" in reason.lower()
            ),
            "",
        )
        raise KRXOpenAPIAccessDeniedError(
            first_access_denied or "KRX OpenAPI batch aborted due to Access Denied"
        )

    first_reason = next(
        (
            reason
            for family_failures in snapshot_failures.values()
            for reason in family_failures.values()
        ),
        "",
    )
    lowered = first_reason.lower()
    if "authentication failed" in lowered:
        raise KRXOpenAPIAuthError(first_reason)
    if "permission denied" in lowered or "forbidden" in lowered:
        raise KRXOpenAPIPermissionError(first_reason)
    if first_reason:
        raise KRXOpenAPIResponseError(first_reason)
    if code in failures:
        raise KRXOpenAPIResponseError(failures[code])
    raise KRXOpenAPIResponseError("KRX OpenAPI returned no data rows")
