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


def test_get_krx_openapi_url_ignores_invalid_override(monkeypatch):
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.secrets = {"KRX_OPENAPI_URL": "https://data.krx.co.kr/svc/apis/idx/krx_dd_trd"}
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)

    url = krx_openapi.get_krx_openapi_url("kospi_dd_trd")

    assert url == "https://data-dbg.krx.co.kr/svc/apis/idx/kospi_dd_trd"


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
