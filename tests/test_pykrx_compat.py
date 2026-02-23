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
