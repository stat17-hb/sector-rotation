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
    def __init__(self, response: _FakeResponse):
        self._response = response

    def get(self, url, params=None, headers=None, timeout=None):
        _ = (url, params, headers, timeout)
        return self._response


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


def test_fetch_index_ohlcv_openapi_parses_rows():
    payload = {
        "OutBlock_1": [
            {"BAS_DD": "20260226", "IDX_IND_CD": "1001", "CLSPRC_IDX": "2,600.12"},
            {"BAS_DD": "20260227", "IDX_IND_CD": "1001", "CLSPRC_IDX": "2,620.50"},
        ]
    }
    session = _FakeSession(_FakeResponse(status_code=200, payload=payload))

    result = krx_openapi.fetch_index_ohlcv_openapi(
        "1001",
        "20260226",
        "20260227",
        auth_key="DUMMY",
        session=session,
    )

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result["close"].tolist() == [2600.12, 2620.5]


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


def test_fetch_index_ohlcv_openapi_empty_response_raises():
    session = _FakeSession(_FakeResponse(status_code=200, payload={"respCode": "0", "respMsg": "OK"}))

    with pytest.raises(krx_openapi.KRXOpenAPIResponseError, match="no data rows"):
        krx_openapi.fetch_index_ohlcv_openapi(
            "1001",
            "20260226",
            "20260227",
            auth_key="DUMMY",
            session=session,
        )

