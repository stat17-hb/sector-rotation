"""Tests for pykrx compatibility helpers."""
from __future__ import annotations

import importlib
import sys
import types

import pandas as pd
import pytest


def _install_fake_pykrx(monkeypatch):
    """Install a minimal fake pykrx module tree for transport patch tests."""
    pykrx_mod = types.ModuleType("pykrx")
    website_mod = types.ModuleType("pykrx.website")
    website_mod.__path__ = []  # package marker
    comm_mod = types.ModuleType("pykrx.website.comm")
    comm_mod.__path__ = []  # package marker
    webio_mod = types.ModuleType("pykrx.website.comm.webio")
    krx_mod = types.ModuleType("pykrx.website.krx")
    krx_mod.__path__ = []  # package marker
    krxio_mod = types.ModuleType("pykrx.website.krx.krxio")

    class Get:
        def __init__(self):
            self.headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "http://data.krx.co.kr/",
            }

    class Post:
        def __init__(self, headers=None):
            self.headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "http://data.krx.co.kr/",
            }
            if headers is not None:
                self.headers.update(headers)

    class KrxWebIo:
        @property
        def url(self):
            return "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"

    class KrxFutureIo:
        @property
        def url(self):
            return "http://data.krx.co.kr/comm/bldAttendant/executeForResourceBundle.cmd"

    webio_mod.Get = Get
    webio_mod.Post = Post
    krxio_mod.KrxWebIo = KrxWebIo
    krxio_mod.KrxFutureIo = KrxFutureIo

    pykrx_mod.website = website_mod
    website_mod.comm = comm_mod
    website_mod.krx = krx_mod
    comm_mod.webio = webio_mod
    krx_mod.krxio = krxio_mod

    module_map = {
        "pykrx": pykrx_mod,
        "pykrx.website": website_mod,
        "pykrx.website.comm": comm_mod,
        "pykrx.website.comm.webio": webio_mod,
        "pykrx.website.krx": krx_mod,
        "pykrx.website.krx.krxio": krxio_mod,
    }
    for name, module in module_map.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_transport_patch_updates_referer_and_urls(monkeypatch):
    _install_fake_pykrx(monkeypatch)
    import src.data_sources.pykrx_compat as compat

    importlib.reload(compat)
    compat.ensure_pykrx_transport_compat()

    from pykrx.website.comm import webio
    from pykrx.website.krx import krxio

    assert webio.Get().headers["Referer"] == compat.KRX_REFERER
    assert webio.Post().headers["Referer"] == compat.KRX_REFERER
    assert krxio.KrxWebIo().url == compat.KRX_JSON_URL
    assert krxio.KrxFutureIo().url == compat.KRX_RESOURCE_URL


def test_transport_patch_is_idempotent(monkeypatch):
    _install_fake_pykrx(monkeypatch)
    import src.data_sources.pykrx_compat as compat

    importlib.reload(compat)
    compat.ensure_pykrx_transport_compat()
    from pykrx.website.comm import webio

    first_get_init = webio.Get.__init__
    first_post_init = webio.Post.__init__

    compat.ensure_pykrx_transport_compat()

    assert webio.Get.__init__ is first_get_init
    assert webio.Post.__init__ is first_post_init


def test_parse_login_response_handles_non_json_html():
    import src.data_sources.pykrx_compat as compat

    class _Resp:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = "<html><title>로그인</title></html>"

        @staticmethod
        def json():
            raise ValueError("no json")

    ok, detail = compat._parse_login_response(_Resp())
    assert ok is False
    assert "HTML" in detail or "login" in detail.lower()


def test_request_krx_data_uses_shared_session_headers(monkeypatch):
    import src.data_sources.pykrx_compat as compat

    captured: dict[str, object] = {}

    class _Session:
        def post(self, url, headers=None, data=None, timeout=None):
            captured["url"] = url
            captured["headers"] = dict(headers or {})
            captured["data"] = dict(data or {})
            captured["timeout"] = timeout
            return object()

    monkeypatch.setattr(compat, "_get_shared_session", lambda: _Session())
    response = compat.request_krx_data({"foo": "bar"})
    assert response is not None
    assert captured["url"] == compat.KRX_JSON_URL
    assert captured["headers"]["X-Requested-With"] == "XMLHttpRequest"
    assert captured["data"] == {"foo": "bar"}


def test_patched_webio_reads_pick_up_fresh_session_after_reset(monkeypatch):
    _install_fake_pykrx(monkeypatch)
    import src.data_sources.pykrx_compat as compat

    importlib.reload(compat)

    sessions: list[object] = []

    class _Session:
        def __init__(self, label):
            self.label = label

        def post(self, url, headers=None, data=None, timeout=None):
            return {"label": self.label, "url": url, "data": dict(data or {})}

        def get(self, url, headers=None, params=None, timeout=None):
            return {"label": self.label, "url": url, "params": dict(params or {})}

        def close(self):
            return None

    def _get_shared_session():
        if compat._SHARED_SESSION is None:
            compat._SHARED_SESSION = _Session(f"s{len(sessions) + 1}")
            sessions.append(compat._SHARED_SESSION)
        return compat._SHARED_SESSION

    monkeypatch.setattr(compat, "_get_shared_session", _get_shared_session)
    compat.ensure_pykrx_transport_compat()

    from pykrx.website.comm import webio

    post = webio.Post()
    post.url = compat.KRX_JSON_URL
    first = post.read(foo="bar")
    compat.reset_krx_shared_session()
    post_after_reset = webio.Post()
    post_after_reset.url = compat.KRX_JSON_URL
    second = post_after_reset.read(foo="baz")

    assert first["label"] == "s1"
    assert second["label"] == "s2"


def test_get_shared_session_records_login_failure_detail(monkeypatch):
    import src.data_sources.pykrx_compat as compat

    monkeypatch.setattr(compat, "_SHARED_SESSION", None)
    monkeypatch.setattr(compat, "_SESSION_AUTHENTICATED", None)
    monkeypatch.setattr(compat, "_SESSION_AUTH_DETAIL", "KRX_ID/KRX_PW not configured")
    monkeypatch.setattr(compat, "_load_secret_or_env", lambda name: "x" if name in {"KRX_ID", "KRX_PW"} else "")
    monkeypatch.setattr(compat, "_warmup_krx_login_session", lambda session: (_ for _ in ()).throw(RuntimeError("dns fail")))

    session = compat._get_shared_session()

    assert session is not None
    state = compat.get_krx_login_state()
    assert state["configured"] is True
    assert state["authenticated"] is False
    assert "dns fail" in state["detail"]


def test_login_krx_session_retries_after_access_denied(monkeypatch):
    import src.data_sources.pykrx_compat as compat

    class _RespDenied:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = "<html><body>Access Denied</body></html>"

        @staticmethod
        def json():
            raise ValueError("not json")

    class _RespOk:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = '{"_error_code":"CD001"}'

        @staticmethod
        def json():
            return {"_error_code": "CD001"}

    class _Session:
        def __init__(self):
            self.calls = 0

        def post(self, url, data=None, headers=None, timeout=None):
            self.calls += 1
            return _RespDenied() if self.calls == 1 else _RespOk()

    session = _Session()
    ok, detail = compat._login_krx_session(session, "id", "pw")
    assert ok is True
    assert detail == "authenticated"
    assert session.calls == 2


def test_resolve_close_column_prefers_named_column():
    from src.data_sources.pykrx_compat import resolve_ohlcv_close_column

    df = pd.DataFrame(
        columns=[
            "\uc2dc\uac00",
            "\uace0\uac00",
            "\uc800\uac00",
            "\uc885\uac00",
            "\uac70\ub798\ub7c9",
        ]
    )
    assert resolve_ohlcv_close_column(df) == "\uc885\uac00"


def test_resolve_close_column_falls_back_to_4th_column():
    from src.data_sources.pykrx_compat import resolve_ohlcv_close_column

    df = pd.DataFrame(columns=["c0", "c1", "c2", "c3", "c4"])
    assert resolve_ohlcv_close_column(df) == "c3"


def test_resolve_close_column_raises_for_invalid_shape():
    from src.data_sources.pykrx_compat import resolve_ohlcv_close_column

    df = pd.DataFrame(columns=["c0", "c1", "c2"])
    with pytest.raises(ValueError, match="Unable to resolve OHLCV close column"):
        resolve_ohlcv_close_column(df)
