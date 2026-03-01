"""
Compatibility helpers for pykrx index transport behavior.

Some pykrx versions ship with legacy HTTP/Referer defaults that can trigger
`LOGOUT` responses from KRX endpoints. These helpers patch transport defaults
at runtime without changing dependency versions.
"""
from __future__ import annotations

import logging
import threading
from functools import wraps

import pandas as pd
import requests as _requests

logger = logging.getLogger(__name__)

KRX_REFERER = "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd"
KRX_JSON_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
KRX_RESOURCE_URL = (
    "https://data.krx.co.kr/comm/bldAttendant/executeForResourceBundle.cmd"
)

_PATCH_APPLIED = False
_PATCH_LOCK = threading.Lock()
_PATCH_ATTR = "__sector_rotation_pykrx_transport_patched__"

_SHARED_SESSION: _requests.Session | None = None
_SESSION_LOCK = threading.Lock()

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def _get_shared_session() -> _requests.Session:
    """Return (and lazily create) the shared requests.Session for KRX calls.

    On first call the session is pre-warmed by fetching the KRX index page so
    that any session cookie KRX sets is captured and forwarded with every
    subsequent pykrx API request.
    """
    global _SHARED_SESSION
    if _SHARED_SESSION is not None:
        return _SHARED_SESSION
    with _SESSION_LOCK:
        if _SHARED_SESSION is not None:
            return _SHARED_SESSION
        session = _requests.Session()
        try:
            session.get(
                KRX_REFERER,
                headers={"User-Agent": _BROWSER_UA, "Referer": "https://data.krx.co.kr"},
                timeout=10,
            )
            logger.debug("KRX shared session pre-warmed (cookies: %s).", list(session.cookies.keys()))
        except Exception as exc:
            logger.warning("KRX session pre-warm failed (will proceed anyway): %s", exc)
        _SHARED_SESSION = session
    return _SHARED_SESSION


def _wrap_init_with_referer(init_fn):
    """Wrap pykrx transport initializer to enforce current KRX Referer."""
    if getattr(init_fn, _PATCH_ATTR, False):
        return init_fn

    @wraps(init_fn)
    def wrapped(self, *args, **kwargs):
        init_fn(self, *args, **kwargs)
        headers = getattr(self, "headers", None)
        if isinstance(headers, dict):
            headers["Referer"] = KRX_REFERER

    setattr(wrapped, _PATCH_ATTR, True)
    return wrapped


def _reset_index_ticker_singleton() -> bool:
    """Reset pykrx IndexTicker singleton if its internal DataFrame is empty.

    When KRX server changes its response format, pykrx's IndexTicker singleton
    initialises with an empty DataFrame (missing '시장'/'지수명' columns).
    Since it's a singleton (sealed after first __init__), subsequent calls
    return the same broken instance. This function detects and resets it.

    Returns:
        True if singleton was reset, False if it was healthy or not yet initialised.
    """
    try:
        from pykrx.website.krx.market.ticker import IndexTicker  # type: ignore[import]

        instance = IndexTicker._instance  # type: ignore[attr-defined]
        if instance is None:
            return False
        df = getattr(instance, "df", None)
        if df is None or (hasattr(df, "empty") and df.empty):
            IndexTicker._instance = None  # type: ignore[attr-defined]
            logger.warning(
                "pykrx IndexTicker singleton had empty df (KRX server response "
                "changed); reset singleton so next call re-initialises."
            )
            return True
    except Exception:
        pass
    return False


def ensure_pykrx_transport_compat() -> None:
    """Patch pykrx transport defaults once per process."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    with _PATCH_LOCK:
        if _PATCH_APPLIED:
            return

        from pykrx.website.comm import webio  # type: ignore[import]
        from pykrx.website.krx import krxio  # type: ignore[import]

        webio.Get.__init__ = _wrap_init_with_referer(webio.Get.__init__)
        webio.Post.__init__ = _wrap_init_with_referer(webio.Post.__init__)
        krxio.KrxWebIo.url = property(lambda self: KRX_JSON_URL)
        krxio.KrxFutureIo.url = property(lambda self: KRX_RESOURCE_URL)

        # ── Shared-session patch (KRX 2026-02-27 세션 쿠키 의무화 대응) ──
        # pykrx는 요청마다 새 requests 객체를 생성해 쿠키가 보존되지 않는다.
        # Post.read / Get.read 를 공유 Session 을 사용하도록 교체한다.
        _session = _get_shared_session()

        def _shared_post_read(self, **params):
            return _session.post(self.url, headers=self.headers, data=params)

        def _shared_get_read(self, **params):
            return _session.get(self.url, headers=self.headers, params=params)

        webio.Post.read = _shared_post_read
        webio.Get.read = _shared_get_read

        _PATCH_APPLIED = True
        logger.info("Applied pykrx transport compatibility patch (shared session).")

    # Reset IndexTicker singleton if it was initialised before the patch,
    # or if KRX server returned an empty/broken response previously.
    _reset_index_ticker_singleton()


def resolve_ohlcv_close_column(df: pd.DataFrame) -> str:
    """Resolve the close-price column name from a pykrx OHLCV DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected pandas DataFrame for OHLCV column resolution.")

    columns = list(df.columns)
    if not columns:
        raise ValueError("Unable to resolve OHLCV close column: no columns available.")

    known_names = ("\uc885\uac00", "close", "Close", "CLOSE")
    for col in columns:
        col_text = str(col).strip()
        if col_text in known_names:
            return str(col)
        if "\uc885\uac00" in col_text:
            return str(col)

    if len(columns) <= 3:
        raise ValueError(
            "Unable to resolve OHLCV close column: expected at least 4 OHLCV "
            f"columns, got {len(columns)} ({columns!r})."
        )

    return str(columns[3])
