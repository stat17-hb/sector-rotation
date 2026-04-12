"""Shared KRX index-constituent lookup helpers.

The primary pykrx wrapper can silently collapse schema drift into an empty list
because its inner fetch path assumes the raw KRX payload still uses `output`.
This module keeps the wrapper as the first attempt, then falls back to the raw
payload reader so callers can survive response-key changes such as
`OutBlock_1`.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping

import pandas as pd

from src.data_sources.pykrx_compat import (
    KRX_REFERER,
    ensure_pykrx_transport_compat,
    get_krx_login_state,
    request_krx_data,
)

CONSTITUENT_TICKER_KEYS: tuple[str, ...] = (
    "ISU_SRT_CD",
    "isuSrtCd",
    "short_code",
    "ticker",
    "종목코드",
)
CONSTITUENT_PAYLOAD_KEYS: tuple[str, ...] = (
    "block1",
    "output",
    "OutBlock_1",
    "outBlock_1",
    "OutBlock1",
    "result",
    "results",
    "data",
    "items",
    "list",
)
PYKRX_CONSTITUENT_HISTORY_FLOOR = "20140502"


@dataclass(frozen=True)
class ConstituentLookupResult:
    tickers: list[str]
    resolved_from: str = ""
    source: str = ""
    failure_detail: str = ""


def normalize_constituent_result(result: Any) -> list[str]:
    """Normalize pykrx constituent results to a flat ticker list."""
    if result is None:
        return []
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return []
        if result.index.dtype == object:
            return [str(item) for item in result.index.tolist()]
        return [str(item) for item in result.iloc[:, 0].tolist()]
    try:
        items = list(result)
    except TypeError:
        return []
    return [str(item) for item in items if item is not None]


def candidate_reference_dates(reference_date: str, *, periods: int = 5) -> list[str]:
    """Return recent business dates in descending order."""
    end_ts = pd.Timestamp(reference_date).normalize()
    values = [ts.strftime("%Y%m%d") for ts in pd.bdate_range(end=end_ts, periods=periods)]
    values.reverse()
    return [value for value in values if str(value) >= PYKRX_CONSTITUENT_HISTORY_FLOOR]


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []

    for key in CONSTITUENT_PAYLOAD_KEYS:
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return [row for row in candidate if isinstance(row, Mapping)]
        if isinstance(candidate, Mapping):
            nested = _extract_rows(candidate)
            if nested:
                return nested

    for value in payload.values():
        nested = _extract_rows(value)
        if nested:
            return nested
    return []


def _first_present(row: Mapping[str, Any], candidates: tuple[str, ...]) -> str:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in candidates:
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
        lower_key = key.lower()
        if lower_key in lowered and str(lowered[lower_key]).strip():
            return str(lowered[lower_key]).strip()
    return ""


def _normalize_ticker(value: str) -> str:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 6:
        return digits
    return text


def _extract_tickers(payload: Any) -> list[str]:
    rows = _extract_rows(payload)
    tickers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        ticker = _normalize_ticker(_first_present(row, CONSTITUENT_TICKER_KEYS))
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
    return tickers


def _response_snippet(text: str, *, limit: int = 180) -> str:
    compact = " ".join(str(text or "").split())
    return compact[:limit]


def _classify_non_json_response(response) -> str:
    status = int(getattr(response, "status_code", 0) or 0)
    content_type = str(getattr(response, "headers", {}).get("content-type", "")).lower()
    text = getattr(response, "text", "") or ""
    snippet = _response_snippet(text)
    lowered = text.lower()
    login_state = get_krx_login_state()
    auth_state_blocked = (
        not login_state["configured"] or login_state["authenticated"] is False
    )

    if status in {401, 403} or "access denied" in lowered:
        return (
            "ACCESS_DENIED: "
            f"KRX constituent endpoint denied access (status={status}, snippet={snippet!r})"
        )

    login_markers = ("login.jsp", "mdccoms001", "로그인", "회원가입", "mbrid", "password")
    if any(marker in lowered for marker in login_markers):
        detail = "KRX Data Marketplace login is required for constituent lookups"
        if not login_state["configured"]:
            detail += " (KRX_ID/KRX_PW not configured)"
        elif login_state["authenticated"] is False:
            detail += f" (login failed: {login_state['detail']})"
        return (
            "AUTH_REQUIRED: "
            f"{detail}; status={status}; content_type={content_type or 'unknown'}; snippet={snippet!r}"
        )

    if auth_state_blocked and (
        not text.strip()
        or "<html" in lowered
        or "<!doctype" in lowered
        or content_type.startswith("text/html")
    ):
        detail = "KRX Data Marketplace login is required for constituent lookups"
        if not login_state["configured"]:
            detail += " (KRX_ID/KRX_PW not configured)"
        else:
            detail += f" (login failed: {login_state['detail']})"
        return (
            "AUTH_REQUIRED: "
            f"{detail}; status={status}; content_type={content_type or 'unknown'}; snippet={snippet!r}"
        )

    if not text.strip():
        return (
            "NON_JSON_EMPTY: "
            f"KRX constituent endpoint returned an empty body (status={status}, content_type={content_type or 'unknown'})"
        )

    return (
        "NON_JSON_RESPONSE: "
        f"KRX constituent endpoint returned non-JSON content "
        f"(status={status}, content_type={content_type or 'unknown'}, snippet={snippet!r})"
    )


def read_index_constituent_payload(*, trade_date: str, sector_code: str) -> Any:
    """Return raw KRX payload for one sector/date lookup via pykrx transport."""
    ensure_pykrx_transport_compat()
    code = str(sector_code).strip()
    if len(code) < 2:
        raise ValueError(f"Invalid sector code for constituent lookup: {sector_code!r}")
    response = request_krx_data(
        {
            "bld": "dbms/MDC/STAT/standard/MDCSTAT00601",
            "locale": "ko_KR",
            "indIdx2": code[1:],
            "indIdx": code[0],
            "trdDd": str(trade_date),
        },
        referer=KRX_REFERER,
    )
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(_classify_non_json_response(response)) from exc


def lookup_index_constituents(
    stock_module,
    *,
    sector_code: str,
    candidate_dates: list[str],
    logger: logging.Logger | None = None,
) -> ConstituentLookupResult:
    """Resolve sector constituents through pykrx first, then raw KRX payload."""
    log = logger or logging.getLogger(__name__)
    failure_notes: list[str] = []
    code = str(sector_code).strip()
    filtered_candidate_dates = [
        str(candidate_date)
        for candidate_date in candidate_dates
        if str(candidate_date).strip() >= PYKRX_CONSTITUENT_HISTORY_FLOOR
    ]
    if not filtered_candidate_dates:
        return ConstituentLookupResult(
            tickers=[],
            failure_detail=(
                "no candidate dates were available for constituent lookup after "
                f"applying history floor {PYKRX_CONSTITUENT_HISTORY_FLOOR}"
            ),
        )

    for candidate_date in filtered_candidate_dates:
        try:
            result = stock_module.get_index_portfolio_deposit_file(
                ticker=code,
                date=candidate_date,
                alternative=False,
            )
            tickers = normalize_constituent_result(result)
            log.debug(
                "Constituent wrapper lookup %s on %s -> %d tickers (type=%s)",
                code,
                candidate_date,
                len(tickers),
                type(result).__name__,
            )
            if tickers:
                return ConstituentLookupResult(
                    tickers=tickers,
                    resolved_from=candidate_date,
                    source="pykrx_wrapper",
                )
        except Exception as exc:
            failure_notes.append(f"{candidate_date}:pykrx={exc}")
            log.debug("Constituent wrapper lookup failed for %s on %s: %s", code, candidate_date, exc)

        try:
            payload = read_index_constituent_payload(
                trade_date=candidate_date,
                sector_code=code,
            )
            tickers = _extract_tickers(payload)
            payload_keys = list(payload.keys()) if isinstance(payload, Mapping) else [type(payload).__name__]
            log.debug(
                "Constituent raw lookup %s on %s -> %d tickers (keys=%s)",
                code,
                candidate_date,
                len(tickers),
                payload_keys,
            )
            if tickers:
                return ConstituentLookupResult(
                    tickers=tickers,
                    resolved_from=candidate_date,
                    source="krx_raw_payload",
                )
            failure_notes.append(f"{candidate_date}:raw-empty")
        except Exception as exc:
            failure_detail = str(exc)
            if failure_detail.startswith(("AUTH_REQUIRED:", "ACCESS_DENIED:")):
                return ConstituentLookupResult(
                    tickers=[],
                    failure_detail=failure_detail,
                )
            failure_notes.append(f"{candidate_date}:raw={failure_detail}")
            log.debug("Constituent raw lookup failed for %s on %s: %s", code, candidate_date, exc)

    if filtered_candidate_dates:
        default_detail = (
            f"empty constituent list across candidate dates "
            f"{filtered_candidate_dates[-1]}..{filtered_candidate_dates[0]}"
        )
    else:
        default_detail = "no candidate dates were available for constituent lookup"

    note_text = ", ".join(failure_notes[:6])
    if len(failure_notes) > 6:
        note_text += ", ..."
    return ConstituentLookupResult(
        tickers=[],
        failure_detail=f"{default_detail} [{note_text}]" if note_text else default_detail,
    )


__all__ = [
    "ConstituentLookupResult",
    "candidate_reference_dates",
    "lookup_index_constituents",
    "normalize_constituent_result",
    "read_index_constituent_payload",
]
