"""
API preflight diagnostics for macro/market providers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import requests

PreflightStatus = Literal["OK", "TIMEOUT", "DNS_FAIL", "SOCKET_BLOCKED", "HTTP_ERROR"]

ENDPOINTS: dict[str, str] = {
    "ECOS": "https://ecos.bok.or.kr/api",
    "KOSIS": "https://kosis.kr/openapi",
    "KRX": "https://data.krx.co.kr",
}


def _classify_connection_error(exc: requests.ConnectionError) -> tuple[PreflightStatus, str]:
    msg = str(exc)
    lowered = msg.lower()

    # Windows socket-block policy errors often show WinError 10013.
    if "10013" in lowered or "access permissions" in lowered or "액세스 권한" in msg:
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

    # 4xx from root endpoints still proves connectivity; treat as OK.
    if resp.status_code >= 500:
        return ("HTTP_ERROR", f"HTTP {resp.status_code}")
    return ("OK", f"HTTP {resp.status_code}")


def run_api_preflight(timeout_sec: int = 3) -> dict[str, dict[str, str]]:
    """Run lightweight endpoint checks for ECOS/KOSIS/KRX providers."""
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results: dict[str, dict[str, str]] = {}
    for name, url in ENDPOINTS.items():
        status, detail = _probe_endpoint(url, timeout_sec=timeout_sec)
        results[name] = {
            "status": status,
            "detail": detail,
            "url": url,
            "checked_at": checked_at,
        }
    return results
