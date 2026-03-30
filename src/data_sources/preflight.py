"""
API preflight diagnostics for macro/market providers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import requests

from src.data_sources.krx_openapi import probe_krx_openapi_health

PreflightStatus = Literal[
    "OK",
    "TIMEOUT",
    "DNS_FAIL",
    "SOCKET_BLOCKED",
    "AUTH_ERROR",
    "ACCESS_DENIED",
    "HTTP_ERROR",
]

ENDPOINTS: dict[str, str] = {
    "ECOS": "https://ecos.bok.or.kr/api",
    "KOSIS": "https://kosis.kr/openapi",
}
US_ENDPOINTS: dict[str, str] = {
    "FRED": "https://api.stlouisfed.org/fred",
}


def _classify_connection_error(exc: requests.ConnectionError) -> tuple[PreflightStatus, str]:
    msg = str(exc)
    lowered = msg.lower()

    if "10013" in lowered or "access permissions" in lowered or "wsaeacces" in lowered or "액세스" in msg:
        return ("SOCKET_BLOCKED", msg)

    dns_signals = (
        "name or service not known",
        "temporary failure in name resolution",
        "nodename nor servname provided",
        "getaddrinfo failed",
    )
    if any(signal in lowered for signal in dns_signals):
        return ("DNS_FAIL", msg)

    return ("HTTP_ERROR", msg)


def _probe_endpoint(url: str, timeout_sec: int) -> tuple[PreflightStatus, str]:
    try:
        resp = requests.get(url, timeout=timeout_sec)
    except requests.Timeout as exc:
        return ("TIMEOUT", str(exc))
    except requests.ConnectionError as exc:
        return _classify_connection_error(exc)
    except requests.RequestException as exc:
        return ("HTTP_ERROR", str(exc))

    if resp.status_code >= 500:
        return ("HTTP_ERROR", f"HTTP {resp.status_code}")
    return ("OK", f"HTTP {resp.status_code}")


def run_api_preflight(timeout_sec: int = 3, market_id: str = "KR") -> dict[str, dict[str, str]]:
    """Run lightweight endpoint checks for active providers."""
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results: dict[str, dict[str, str]] = {}
    market = str(market_id or "KR").strip().upper()
    endpoints = US_ENDPOINTS if market == "US" else ENDPOINTS
    for name, url in endpoints.items():
        status, detail = _probe_endpoint(url, timeout_sec=timeout_sec)
        results[name] = {
            "status": status,
            "detail": detail,
            "url": url,
            "checked_at": checked_at,
        }

    if market != "US":
        try:
            krx_health = probe_krx_openapi_health(timeout_sec=timeout_sec)
        except Exception as exc:
            krx_health = {
                "status": "HTTP_ERROR",
                "detail": str(exc),
                "url": "",
                "checked_at": checked_at,
            }
        results["KRX"] = {
            "status": str(krx_health.get("status", "HTTP_ERROR")),
            "detail": str(krx_health.get("detail", "")).strip(),
            "url": str(krx_health.get("url", "")).strip(),
            "checked_at": str(krx_health.get("checked_at", checked_at)).strip() or checked_at,
        }
    return results
