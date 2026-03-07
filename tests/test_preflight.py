"""Tests for API preflight diagnostics."""
from __future__ import annotations

import requests

from src.data_sources import preflight


class _Response:
    def __init__(self, status_code: int):
        self.status_code = status_code


def test_preflight_classifies_timeout(monkeypatch):
    monkeypatch.setattr(preflight, "ENDPOINTS", {"ECOS": "https://example.test"})

    def fake_get(url, timeout):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(preflight.requests, "get", fake_get)

    result = preflight.run_api_preflight(timeout_sec=1)
    assert result["ECOS"]["status"] == "TIMEOUT"


def test_preflight_classifies_socket_blocked(monkeypatch):
    monkeypatch.setattr(preflight, "ENDPOINTS", {"KOSIS": "https://example.test"})

    def fake_get(url, timeout):
        raise requests.ConnectionError("[WinError 10013] blocked by access permissions")

    monkeypatch.setattr(preflight.requests, "get", fake_get)

    result = preflight.run_api_preflight(timeout_sec=1)
    assert result["KOSIS"]["status"] == "SOCKET_BLOCKED"


def test_preflight_ok_on_reachable_4xx(monkeypatch):
    monkeypatch.setattr(preflight, "ENDPOINTS", {"ECOS": "https://example.test"})

    def fake_get(url, timeout):
        return _Response(403)

    monkeypatch.setattr(preflight.requests, "get", fake_get)
    monkeypatch.setattr(
        preflight,
        "probe_krx_openapi_health",
        lambda timeout_sec=3: {
            "status": "OK",
            "detail": "HTTP 200",
            "url": "https://data-dbg.krx.co.kr/svc/apis/idx/krx_dd_trd",
            "checked_at": "2026-03-07T00:00:00+00:00",
        },
    )

    result = preflight.run_api_preflight(timeout_sec=1)
    assert result["ECOS"]["status"] == "OK"


def test_run_api_preflight_uses_real_krx_health_probe(monkeypatch):
    monkeypatch.setattr(preflight, "_probe_endpoint", lambda url, timeout_sec: ("OK", f"HTTP 200 {url}"))
    monkeypatch.setattr(
        preflight,
        "probe_krx_openapi_health",
        lambda timeout_sec=3: {
            "status": "ACCESS_DENIED",
            "detail": "KRX OpenAPI returned Access Denied payload",
            "url": "https://data-dbg.krx.co.kr/svc/apis/idx/krx_dd_trd",
            "checked_at": "2026-03-07T00:00:00+00:00",
        },
    )

    results = preflight.run_api_preflight(timeout_sec=1)

    assert results["ECOS"]["status"] == "OK"
    assert results["KOSIS"]["status"] == "OK"
    assert results["KRX"]["status"] == "ACCESS_DENIED"
    assert "Access Denied" in results["KRX"]["detail"]
