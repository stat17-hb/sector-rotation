"""Tests for KRX OpenAPI provider helpers."""
from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from src.data_sources import krx_openapi


class _FakeResponse:
    def __init__(self, status_code: int, payload: object, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeSession:
    def __init__(self, responses_by_basdd: dict[str, _FakeResponse] | _FakeResponse):
        self._responses_by_basdd = responses_by_basdd
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url, params=None, headers=None, timeout=None):
        _ = (headers, timeout)
        self.calls.append((url, params))
        if isinstance(self._responses_by_basdd, _FakeResponse):
            return self._responses_by_basdd
        bas_dd = str((params or {}).get("basDd", ""))
        return self._responses_by_basdd[bas_dd]


@pytest.fixture(autouse=True)
def _isolate_streamlit_runtime(monkeypatch):
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.secrets = {}
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.delenv("KRX_OPENAPI_URL", raising=False)
    monkeypatch.delenv("KRX_OPENAPI_KEY", raising=False)
    monkeypatch.delenv("KRX_PROVIDER", raising=False)
    monkeypatch.setattr(krx_openapi, "_OPENAPI_URL_OVERRIDE_WARNED", False)


def test_get_krx_provider_parsing():
    assert krx_openapi.get_krx_provider("AUTO") == "AUTO"
    assert krx_openapi.get_krx_provider("openapi") == "OPENAPI"
    assert krx_openapi.get_krx_provider("pykrx") == "PYKRX"
    assert krx_openapi.get_krx_provider("invalid") == "AUTO"


def test_openapi_key_prefers_streamlit_secrets(monkeypatch):
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.secrets = {"KRX_OPENAPI_KEY": "SECRET_KEY"}
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.setenv("KRX_OPENAPI_KEY", "ENV_KEY")

    assert krx_openapi.get_krx_openapi_key() == "SECRET_KEY"


def test_get_krx_openapi_url_ignores_deprecated_override_warns_once(caplog, monkeypatch):
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.secrets = {"KRX_OPENAPI_URL": "https://data.krx.co.kr/svc/apis/idx/krx_dd_trd"}
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    caplog.set_level("WARNING")

    url_krx = krx_openapi.get_krx_openapi_url("krx_dd_trd")
    url_kospi = krx_openapi.get_krx_openapi_url("kospi_dd_trd")

    assert url_krx == "https://data-dbg.krx.co.kr/svc/apis/idx/krx_dd_trd"
    assert url_kospi == "https://data-dbg.krx.co.kr/svc/apis/idx/kospi_dd_trd"
    warnings = [
        rec.message for rec in caplog.records if "deprecated KRX_OPENAPI_URL override" in rec.message
    ]
    assert len(warnings) == 1


def test_resolve_openapi_batch_workers_caps_force_mode(monkeypatch):
    monkeypatch.setenv("KRX_OPENAPI_BATCH_WORKERS", "8")

    assert krx_openapi.resolve_openapi_batch_workers(20, force=True) == 4
    assert krx_openapi.resolve_openapi_batch_workers(3, force=True) == 3


def test_resolve_openapi_api_id_routes_dashboard_codes():
    assert krx_openapi.resolve_openapi_api_id("5044") == "krx_dd_trd"
    assert krx_openapi.resolve_openapi_api_id("1155") == "kospi_dd_trd"
    assert krx_openapi.resolve_openapi_api_id("2001") == "kosdaq_dd_trd"


def test_fetch_index_ohlcv_openapi_parses_rows():
    session = _FakeSession(
        {
            "20260226": _FakeResponse(
                status_code=200,
                payload={
                    "OutBlock_1": [
                        {"BAS_DD": "20260226", "IDX_NM": "\ucf54\uc2a4\ud53c", "CLSPRC_IDX": "2,600.12"},
                        {"BAS_DD": "20260226", "IDX_NM": "\ucf54\uc2a4\ud53c 200", "CLSPRC_IDX": "360.10"},
                    ]
                },
            ),
            "20260227": _FakeResponse(
                status_code=200,
                payload={
                    "OutBlock_1": [
                        {"BAS_DD": "20260227", "IDX_NM": "\ucf54\uc2a4\ud53c", "CLSPRC_IDX": "2,620.50"},
                        {"BAS_DD": "20260227", "IDX_NM": "\ucf54\uc2a4\ud53c 200", "CLSPRC_IDX": "362.22"},
                    ]
                },
            ),
        }
    )

    result = krx_openapi.fetch_index_ohlcv_openapi(
        "1001",
        "20260226",
        "20260227",
        auth_key="DUMMY",
        session=session,
    )

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result["close"].tolist() == [2600.12, 2620.5]
    assert [params["basDd"] for _, params in session.calls] == ["20260226", "20260227"]


def test_fetch_index_ohlcv_openapi_auth_error():
    session = _FakeSession(
        _FakeResponse(
            status_code=401,
            payload={"respMsg": "Unauthorized Key", "respCode": "401"},
        )
    )

    with pytest.raises(krx_openapi.KRXOpenAPIAuthError):
        krx_openapi.fetch_index_ohlcv_openapi(
            "1001",
            "20260226",
            "20260227",
            auth_key="BAD",
            session=session,
        )


def test_request_with_retry_fails_fast_on_access_denied():
    session = _FakeSession(_FakeResponse(status_code=200, payload={"respCode": "0", "respMsg": "OK"}))
    state = {"calls": 0}

    def _get(url, params=None, headers=None, timeout=None):
        _ = (url, params, headers, timeout)
        state["calls"] += 1
        if state["calls"] == 1:
            return _FakeResponse(
                status_code=200,
                payload=ValueError("not json"),
                text="<html><body>Access Denied</body></html>",
            )
        return _FakeResponse(status_code=200, payload={"respCode": "0", "respMsg": "OK"})

    session.get = _get  # type: ignore[method-assign]

    with pytest.raises(krx_openapi.KRXOpenAPIAccessDeniedError):
        krx_openapi._request_with_retry(
            url="https://data-dbg.krx.co.kr/svc/apis/idx/kospi_dd_trd",
            auth_key="DUMMY",
            params={"basDd": "20260227"},
            session=session,
        )

    assert state["calls"] == 1


def test_probe_krx_openapi_health_classifies_ok(monkeypatch):
    krx_openapi.reset_krx_openapi_health_cache()
    monkeypatch.setenv("KRX_OPENAPI_KEY", "DUMMY")

    def _fake_get(url, params=None, headers=None, timeout=None):
        _ = (url, params, headers, timeout)
        return _FakeResponse(
            status_code=200,
            payload={
                "OutBlock_1": [
                    {
                        "BAS_DD": "20240131",
                        "IDX_NM": "\ucf54\uc2a4\ud53c",
                        "CLSPRC_IDX": "2500.00",
                    }
                ]
            },
        )

    monkeypatch.setattr(krx_openapi.requests, "get", _fake_get)

    health = krx_openapi.probe_krx_openapi_health(timeout_sec=1)

    assert health["status"] == "OK"
    assert "HTTP 200" in health["detail"]


def test_probe_krx_openapi_health_classifies_access_denied(monkeypatch):
    krx_openapi.reset_krx_openapi_health_cache()
    monkeypatch.setenv("KRX_OPENAPI_KEY", "DUMMY")

    def _fake_get(url, params=None, headers=None, timeout=None):
        _ = (url, params, headers, timeout)
        return _FakeResponse(
            status_code=200,
            payload=ValueError("not json"),
            text="<html><body>Access Denied</body></html>",
        )

    monkeypatch.setattr(krx_openapi.requests, "get", _fake_get)

    health = krx_openapi.probe_krx_openapi_health(timeout_sec=1)

    assert health["status"] == "ACCESS_DENIED"
    assert "Access Denied" in health["detail"]


def test_fetch_index_ohlcv_openapi_rejects_snapshot_without_series_identifier():
    session = _FakeSession(
        {
            "20260226": _FakeResponse(
                status_code=200,
                payload={"OutBlock_1": [{"BAS_DD": "20260226", "CLSPRC_IDX": "2,600.12"}]},
            )
        }
    )

    with pytest.raises(krx_openapi.KRXOpenAPIResponseError, match="parseable name/close rows"):
        krx_openapi.fetch_index_ohlcv_openapi(
            "1001",
            "20260226",
            "20260226",
            auth_key="DUMMY",
            session=session,
        )


def test_fetch_index_ohlcv_openapi_empty_response_raises():
    session = _FakeSession(
        {
            "20260226": _FakeResponse(status_code=200, payload={"respCode": "0", "respMsg": "OK"}),
            "20260227": _FakeResponse(status_code=200, payload={"respCode": "0", "respMsg": "OK"}),
        }
    )

    with pytest.raises(krx_openapi.KRXOpenAPIResponseError, match="no data rows"):
        krx_openapi.fetch_index_ohlcv_openapi(
            "1001",
            "20260226",
            "20260227",
            auth_key="DUMMY",
            session=session,
        )


def test_fetch_index_ohlcv_openapi_batch_detailed_collects_failed_days_and_preserves_successes():
    session = _FakeSession(
        {
            "20260226": _FakeResponse(
                status_code=200,
                payload={
                    "OutBlock_1": [
                        {"BAS_DD": "20260226", "IDX_NM": "\ucf54\uc2a4\ud53c", "CLSPRC_IDX": "2600.12"},
                    ]
                },
            ),
            "20260227": _FakeResponse(
                status_code=200,
                payload=ValueError("not json"),
                text="<html><body>Access Denied</body></html>",
            ),
        }
    )

    successes, failures, details = krx_openapi.fetch_index_ohlcv_openapi_batch_detailed(
        ["1001"],
        "20260226",
        "20260227",
        auth_key="DUMMY",
        session=session,
        force=True,
    )

    assert failures == {}
    assert list(successes["1001"].index.strftime("%Y%m%d")) == ["20260226"]
    assert details["failed_days"] == ["20260227"]
    assert details["aborted"] is True
    assert details["abort_reason"] == "ACCESS_DENIED"
    assert details["processed_requests"] == 2


def test_update_index_name_metadata_moves_previous_official_into_history(tmp_path, monkeypatch):
    metadata_path = tmp_path / "index_name_metadata.json"
    monkeypatch.setattr(krx_openapi, "INDEX_NAME_METADATA_PATH", metadata_path)
    krx_openapi._load_index_name_metadata.cache_clear()

    krx_openapi.update_index_name_metadata({"5046": ["KRX \ubbf8\ub514\uc5b4\ud1b5\uc2e0"]})
    krx_openapi.update_index_name_metadata({"5046": ["KRX \ubc29\uc1a1\ud1b5\uc2e0"]})

    aliases = krx_openapi.resolve_index_name_aliases("5046")

    assert aliases[0] == "KRX \ubc29\uc1a1\ud1b5\uc2e0"
    assert "KRX \ubbf8\ub514\uc5b4\ud1b5\uc2e0" in aliases
    payload = krx_openapi._read_index_name_metadata()
    assert payload["codes"]["5046"]["official_name"] == "KRX \ubc29\uc1a1\ud1b5\uc2e0"


def test_audit_index_name_aliases_uses_emergency_fallback_and_persists_official_name(
    tmp_path,
    monkeypatch,
):
    metadata_path = tmp_path / "index_name_metadata.json"
    monkeypatch.setattr(krx_openapi, "INDEX_NAME_METADATA_PATH", metadata_path)
    krx_openapi._load_index_name_metadata.cache_clear()

    session = _FakeSession(
        {
            "20240131": _FakeResponse(
                status_code=200,
                payload={
                    "OutBlock_1": [
                        {
                            "BAS_DD": "20240131",
                            "IDX_NM": "KRX \ubc29\uc1a1\ud1b5\uc2e0",
                            "CLSPRC_IDX": "657.90",
                        },
                    ]
                },
            )
        }
    )

    results = krx_openapi.audit_index_name_aliases(
        ["5046"],
        "2024-01-31",
        auth_key="DUMMY",
        session=session,
    )

    assert results[0]["matched"] is True
    assert results[0]["matched_name"] == "KRX \ubc29\uc1a1\ud1b5\uc2e0"
    assert krx_openapi.resolve_index_name_aliases("5046")[0] == "KRX \ubc29\uc1a1\ud1b5\uc2e0"


def test_resolve_index_name_aliases_skips_overbroad_kospi200_official_name(tmp_path, monkeypatch):
    metadata_path = tmp_path / "index_name_metadata.json"
    monkeypatch.setattr(krx_openapi, "INDEX_NAME_METADATA_PATH", metadata_path)
    krx_openapi._load_index_name_metadata.cache_clear()

    krx_openapi.update_index_name_metadata({"1155": ["\ucf54\uc2a4\ud53c 200"]})

    aliases = krx_openapi.resolve_index_name_aliases("1155")

    assert "\ucf54\uc2a4\ud53c 200" not in aliases
    assert "\ucf54\uc2a4\ud53c 200 \uc815\ubcf4\uae30\uc220" in aliases
