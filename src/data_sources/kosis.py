"""
KOSIS (통계청 국가통계포털) Open API wrapper.

R11 — HTTP retry policy: timeout 10s, retries 3, backoff 2/4/8s.
Public loaders return LoaderResult = tuple[DataStatus, pd.DataFrame].
KOSIS data for recent 3 months is marked is_provisional=True.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from src.data_sources.common import load_secret_or_env, request_json_with_retry
from src.data_sources.macro_sync import sync_provider_macro

logger = logging.getLogger(__name__)

DataStatus = Literal["LIVE", "CACHED", "SAMPLE"]
LoaderResult = tuple[DataStatus, pd.DataFrame]

CURATED_DIR = Path("data/curated")
KOSIS_BASE_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"

PROVISIONAL_MONTHS = 3  # most recent N months are marked provisional
REQUEST_VARIABLE_ERROR_CODES = {"20", "21"}


def _get_api_key() -> str:
    """Get KOSIS API key from Streamlit secrets or environment variable."""
    return load_secret_or_env("KOSIS_API_KEY")


def _get_with_retry(url: str, params: dict | None = None) -> object:
    """HTTP GET with timeout, retry, and exponential backoff.

    Policy (R11): timeout=10s, retries=3, backoff 2/4/8s.
    Re-raises last exception on final failure.
    """
    return request_json_with_retry(url, params=params, client_name="KOSIS")


def _is_provisional(period: pd.Period) -> bool:
    """Mark the most recent PROVISIONAL_MONTHS periods as provisional."""
    now = pd.Period.now("M")
    return (now - period).n < PROVISIONAL_MONTHS


def _normalize_obj_params(obj_params: dict | None) -> dict[str, str]:
    if not obj_params:
        return {}

    normalized: dict[str, str] = {}
    for key, value in obj_params.items():
        key_str = str(key).strip()
        if not key_str:
            continue
        if not key_str.startswith("objL") or not key_str[4:].isdigit():
            raise ValueError(f"Invalid KOSIS obj param key: {key_str!r}")
        val_str = str(value).strip()
        if val_str:
            normalized[key_str] = val_str
    return normalized


def _build_obj_param_candidates(obj_params: dict | None) -> list[dict[str, str]]:
    """Build ordered parameter candidates for err=20/21 recovery."""
    candidates: list[dict[str, str]] = []
    normalized = _normalize_obj_params(obj_params)

    if normalized:
        candidates.append(normalized)
        candidates.append({})
    else:
        candidates.append({})

    for depth in range(1, 5):
        fallback = {f"objL{i}": "ALL" for i in range(1, depth + 1)}
        candidates.append(fallback)

    deduped: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for candidate in candidates:
        signature = tuple(sorted(candidate.items()))
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(candidate)
    return deduped


def _parse_kosis_error_payload(data: object) -> tuple[str, str] | None:
    if isinstance(data, dict):
        code = str(data.get("err") or data.get("ERR") or "").strip()
        if code:
            msg = str(data.get("errMsg") or data.get("ERR_MSG") or data.get("msg") or "").strip()
            return (code, msg)

    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        first = data[0]
        code = str(first.get("err") or first.get("ERR") or "").strip()
        if code:
            msg = str(first.get("errMsg") or first.get("ERR_MSG") or first.get("msg") or "").strip()
            return (code, msg)

    return None


def _build_kosis_error(code: str, msg: str) -> ValueError:
    hint = ""
    if code in REQUEST_VARIABLE_ERROR_CODES:
        hint = " (request variable mismatch; verify table-specific objL* params, not only API key)"
    return ValueError(f"KOSIS API error [{code}]: {msg}{hint}")


def fetch_kosis_series(
    org_id: str,
    tbl_id: str,
    item_id: str,
    start_ym: str,
    end_ym: str,
    obj_params: dict | None = None,
) -> pd.DataFrame:
    """Fetch a single KOSIS statistical series.

    Args:
        org_id: KOSIS organization ID (e.g. "101").
        tbl_id: Table ID (e.g. "DT_1J22003").
        item_id: Item code within table.
        start_ym: Start year-month YYYYMM.
        end_ym: End year-month YYYYMM.
        obj_params: Optional KOSIS object params (objL1..objL8 etc).

    Returns:
        DataFrame with PeriodIndex (monthly) and columns:
        [series_id, value, source, fetched_at, is_provisional].
    """
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("KOSIS_API_KEY not configured")

    base_params = {
        "method": "getList",
        "apiKey": api_key,
        "itmId": item_id,
        "format": "json",
        "jsonVD": "Y",
        "prdSe": "M",
        "startPrdDe": start_ym,
        "endPrdDe": end_ym,
        "orgId": org_id,
        "tblId": tbl_id,
    }
    param_candidates = _build_obj_param_candidates(obj_params)
    last_exc: Exception | None = None

    for idx, obj_candidate in enumerate(param_candidates, start=1):
        params = {**base_params, **obj_candidate}
        masked_params = {**params, "apiKey": api_key[:4] + "****"}
        logger.info(
            "KOSIS request params (attempt %d/%d): %s",
            idx,
            len(param_candidates),
            masked_params,
        )

        data = _get_with_retry(KOSIS_BASE_URL, params=params)
        err_payload = _parse_kosis_error_payload(data)
        if err_payload:
            code, msg = err_payload
            exc = _build_kosis_error(code, msg)
            last_exc = exc

            has_more = idx < len(param_candidates)
            if code in REQUEST_VARIABLE_ERROR_CODES and has_more:
                logger.warning(
                    "KOSIS request variable mismatch for %s/%s/%s with params=%s; trying next candidate",
                    org_id,
                    tbl_id,
                    item_id,
                    obj_candidate,
                )
                continue
            raise exc

        if not isinstance(data, list) or not data:
            logger.error(
                "KOSIS unexpected response type=%s value=%s",
                type(data).__name__,
                str(data)[:300],
            )
            raise ValueError(f"No KOSIS data for {org_id}/{tbl_id}/{item_id}")

        now = datetime.now(timezone.utc)
        rows = []
        for row in data:
            time_str = str(row.get("PRD_DE", ""))
            if len(time_str) < 6:
                continue

            raw_value = row.get("DT", "")
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue

            period = pd.Period(time_str[:6], freq="M")
            _l1 = (obj_params or {}).get("objL1") if isinstance(obj_params, dict) else None
            _sid = f"{org_id}/{tbl_id}/{item_id}" + (f"/{_l1}" if _l1 else "")
            rows.append(
                {
                    "period": period,
                    "series_id": _sid,
                    "value": value,
                    "source": "KOSIS",
                    "fetched_at": now,
                    "is_provisional": _is_provisional(period),
                }
            )

        if not rows:
            raise ValueError(f"Could not parse KOSIS rows for {org_id}/{tbl_id}")

        df = pd.DataFrame(rows).set_index("period")
        df.index = pd.PeriodIndex(df.index, freq="M")
        return df.astype({"value": "float64", "is_provisional": "bool"})

    if last_exc is not None:
        raise last_exc
    raise ValueError(f"No KOSIS data for {org_id}/{tbl_id}/{item_id}")


def _make_sample_kosis() -> pd.DataFrame:
    """Generate synthetic KOSIS data for SAMPLE mode."""
    import numpy as np

    periods = pd.period_range(end=pd.Period.now("M"), periods=36, freq="M")
    rng = np.random.default_rng(99)
    now = datetime.now(timezone.utc)
    rows = []
    for p in periods:
        rows.append(
            {
                "series_id": "SAMPLE/cpi_yoy",
                "value": float(rng.uniform(1.0, 5.0)),
                "source": "KOSIS",
                "fetched_at": now,
                "is_provisional": _is_provisional(p),
            }
        )
    df = pd.DataFrame(rows, index=pd.PeriodIndex(periods, freq="M"))
    return df.astype({"value": "float64", "is_provisional": "bool"})


def load_kosis_macro(
    start_ym: str,
    end_ym: str,
    series_config: dict | None = None,
) -> LoaderResult:
    """Load KOSIS macro data with LIVE/CACHED/SAMPLE fallback.

    Args:
        start_ym: Start YYYYMM.
        end_ym: End YYYYMM.
        series_config: Dict of {name: {org_id, tbl_id, item_id, obj_params}}.

    Returns:
        LoaderResult = (DataStatus, DataFrame).
    """
    if series_config is None:
        series_config = {
            "cpi_yoy": {
                "org_id": "101",
                "tbl_id": "DT_1J22042",
                "item_id": "T03",
                "obj_params": {"objL1": "0"},  # 총지수 전년동월비(%)
                "enabled": True,
            },
            "cpi_mom": {
                "org_id": "101",
                "tbl_id": "DT_1J22042",
                "item_id": "T02",
                "obj_params": {"objL1": "0"},  # 총지수 전월비(%)
                "enabled": True,
            },
            "leading_index": {
                "org_id": "101",
                "tbl_id": "DT_1C8015",
                "item_id": "T1",
                "obj_params": {"objL1": "A03"},
                "enabled": True,
            },
            "export_growth": {
                "org_id": "142",
                "tbl_id": "DT_142N_0200",
                "item_id": "T10",
                "obj_params": {"objL1": "ALL"},
                "enabled": False,
            },
        }

    def _fetch(alias: str, cfg: dict, provider_start: str, provider_end: str) -> pd.DataFrame:
        _ = alias
        return fetch_kosis_series(
            cfg["org_id"],
            cfg["tbl_id"],
            cfg["item_id"],
            provider_start,
            provider_end,
            obj_params=cfg.get("obj_params"),
        )

    status, result, summary = sync_provider_macro(
        provider="KOSIS",
        start_ym=start_ym,
        end_ym=end_ym,
        series_config=series_config,
        fetch_fn=_fetch,
        reason="load_kosis_macro",
        force=False,
    )
    failed_aliases = dict(summary.get("failed_aliases") or {})
    if failed_aliases and status != "SAMPLE":
        logger.warning(
            "KOSIS partial success: %d aliases loaded, %d failed (%s)",
            len(summary.get("delta_aliases", [])),
            len(failed_aliases),
            ", ".join(sorted(failed_aliases)),
        )
    if status == "SAMPLE":
        logger.warning("Using SAMPLE data for KOSIS macro")
        return ("SAMPLE", _make_sample_kosis())
    return status, result
