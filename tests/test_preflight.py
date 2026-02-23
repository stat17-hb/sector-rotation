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
    monkeypatch.setattr(preflight, "ENDPOINTS", {"KRX": "https://example.test"})

    def fake_get(url, timeout):
        return _Response(403)

    monkeypatch.setattr(preflight.requests, "get", fake_get)

    result = preflight.run_api_preflight(timeout_sec=1)
    assert result["KRX"]["status"] == "OK"
