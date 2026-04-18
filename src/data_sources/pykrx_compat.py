"""
Compatibility helpers for pykrx index transport behavior.

Some pykrx versions ship with legacy HTTP/Referer defaults that can trigger
`LOGOUT` responses from KRX endpoints. These helpers patch transport defaults
at runtime without changing dependency versions.
"""
from __future__ import annotations

import logging
import os
import threading
from functools import wraps
from typing import Any

import pandas as pd
import requests as _requests

logger = logging.getLogger(__name__)

KRX_REFERER = "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd"
KRX_JSON_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
KRX_RESOURCE_URL = (
    "https://data.krx.co.kr/comm/bldAttendant/executeForResourceBundle.cmd"
)
KRX_LOGIN_PAGE = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
KRX_LOGIN_JSP = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
KRX_LOGIN_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"

_PATCH_APPLIED = False
_PATCH_LOCK = threading.Lock()
_PATCH_ATTR = "__sector_rotation_pykrx_transport_patched__"

_SHARED_SESSION: _requests.Session | None = None
_SESSION_LOCK = threading.Lock()
_SESSION_AUTHENTICATED: bool | None = None
_SESSION_AUTH_DETAIL = "KRX_ID/KRX_PW not configured"

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def _load_secret_or_env(name: str) -> str:
    try:
        import streamlit as st  # type: ignore[import]

        value = str(st.secrets.get(name, "")).strip()
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name, "").strip()


def get_krx_login_state() -> dict[str, Any]:
    """Return the latest KRX login/session state for diagnostics."""
    return {
        "configured": bool(_load_secret_or_env("KRX_ID") and _load_secret_or_env("KRX_PW")),
        "authenticated": _SESSION_AUTHENTICATED,
        "detail": _SESSION_AUTH_DETAIL,
    }


def _set_krx_login_state(authenticated: bool | None, detail: str) -> None:
    global _SESSION_AUTHENTICATED, _SESSION_AUTH_DETAIL
    _SESSION_AUTHENTICATED = authenticated
    _SESSION_AUTH_DETAIL = str(detail or "").strip() or "unknown"


def _warmup_krx_login_session(session: _requests.Session) -> None:
    session.get(
        KRX_REFERER,
        headers={
            "User-Agent": _BROWSER_UA,
            "Referer": "https://data.krx.co.kr",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        },
        timeout=15,
    )
    session.get(
        KRX_LOGIN_PAGE,
        headers={
            "User-Agent": _BROWSER_UA,
            "Referer": KRX_REFERER,
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        },
        timeout=15,
    )
    session.get(
        KRX_LOGIN_JSP,
        headers={
            "User-Agent": _BROWSER_UA,
            "Referer": KRX_LOGIN_PAGE,
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        },
        timeout=15,
    )


def _parse_login_response(response: _requests.Response) -> tuple[bool, str]:
    content_type = str(response.headers.get("content-type", "")).lower()
    text = response.text or ""
    try:
        payload = response.json()
    except ValueError:
        snippet = " ".join(text.split())[:160]
        if "access denied" in text.lower():
            return False, f"access denied during KRX login ({snippet})"
        if "login" in text.lower() or "로그인" in text:
            return False, f"login HTML returned instead of JSON ({snippet})"
        return False, (
            "non-JSON login response "
            f"(status={response.status_code}, content_type={content_type or 'unknown'}, snippet={snippet!r})"
        )

    error_code = str(payload.get("_error_code", "")).strip()
    error_msg = str(payload.get("_error_msg", "")).strip()
    if error_code == "CD001":
        return True, "authenticated"
    if error_code:
        return False, f"{error_code}: {error_msg or 'login failed'}"
    return False, error_msg or "KRX login failed"


def _login_krx_session(session: _requests.Session, login_id: str, login_pw: str) -> tuple[bool, str]:
    payload = {
        "mbrNm": "",
        "telNo": "",
        "di": "",
        "certType": "",
        "mbrId": login_id,
        "pw": login_pw,
    }
    header_variants = [
        {
            "User-Agent": _BROWSER_UA,
            "Referer": KRX_LOGIN_JSP,
            "Origin": "https://data.krx.co.kr",
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        },
        {
            "User-Agent": _BROWSER_UA,
            "Referer": KRX_LOGIN_PAGE,
            "Origin": "https://data.krx.co.kr",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        },
        {
            "User-Agent": _BROWSER_UA,
            "Referer": KRX_LOGIN_PAGE,
        },
    ]

    last_detail = "KRX login failed"
    for headers in header_variants:
        response = session.post(KRX_LOGIN_URL, data=payload, headers=headers, timeout=15)
        ok, detail = _parse_login_response(response)
        if ok:
            return True, detail

        if detail.startswith("CD011"):
            payload["skipDup"] = "Y"
            response = session.post(KRX_LOGIN_URL, data=payload, headers=headers, timeout=15)
            ok, detail = _parse_login_response(response)
            if ok:
                return True, detail

        last_detail = detail
        if "access denied" not in detail.lower():
            break

    return False, last_detail


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
        login_id = _load_secret_or_env("KRX_ID")
        login_pw = _load_secret_or_env("KRX_PW")
        try:
            if login_id and login_pw:
                _set_krx_login_state(False, "KRX login warmup not completed yet")
                _warmup_krx_login_session(session)
                authenticated, detail = _login_krx_session(session, login_id, login_pw)
                _set_krx_login_state(authenticated, detail)
                if not authenticated:
                    logger.warning("KRX login failed; proceeding without authenticated session: %s", detail)
            else:
                _set_krx_login_state(None, "KRX_ID/KRX_PW not configured")
            session.get(
                KRX_REFERER,
                headers={"User-Agent": _BROWSER_UA, "Referer": "https://data.krx.co.kr"},
                timeout=10,
            )
            logger.debug("KRX shared session ready (cookies: %s).", list(session.cookies.keys()))
        except Exception as exc:
            if login_id and login_pw:
                _set_krx_login_state(False, f"session warmup/login failed: {exc}")
            logger.warning("KRX session pre-warm failed (will proceed anyway): %s", exc)
        _SHARED_SESSION = session
    return _SHARED_SESSION


def reset_krx_shared_session() -> None:
    """Drop the shared KRX session so the next raw request re-warms a new one."""
    global _SHARED_SESSION
    with _SESSION_LOCK:
        session = _SHARED_SESSION
        _SHARED_SESSION = None
    if session is not None:
        try:
            session.close()
        except Exception:
            logger.debug("KRX shared session close failed during reset.", exc_info=True)
    login_id = _load_secret_or_env("KRX_ID")
    login_pw = _load_secret_or_env("KRX_PW")
    if login_id and login_pw:
        _set_krx_login_state(False, "KRX session reset; next request will re-authenticate")
    else:
        _set_krx_login_state(None, "KRX session reset; KRX_ID/KRX_PW not configured")


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
            headers["X-Requested-With"] = "XMLHttpRequest"
            headers.setdefault("Origin", "https://data.krx.co.kr")

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
        def _shared_post_read(self, **params):
            session = _get_shared_session()
            headers = dict(getattr(self, "headers", {}) or {})
            headers.setdefault("Referer", KRX_REFERER)
            headers.setdefault("User-Agent", _BROWSER_UA)
            headers.setdefault("X-Requested-With", "XMLHttpRequest")
            headers.setdefault("Origin", "https://data.krx.co.kr")
            return session.post(self.url, headers=headers, data=params)

        def _shared_get_read(self, **params):
            session = _get_shared_session()
            headers = dict(getattr(self, "headers", {}) or {})
            headers.setdefault("Referer", KRX_REFERER)
            headers.setdefault("User-Agent", _BROWSER_UA)
            headers.setdefault("X-Requested-With", "XMLHttpRequest")
            return session.get(self.url, headers=headers, params=params)

        webio.Post.read = _shared_post_read
        webio.Get.read = _shared_get_read

        _PATCH_APPLIED = True
        logger.info("Applied pykrx transport compatibility patch (shared session).")

    # Reset IndexTicker singleton if it was initialised before the patch,
    # or if KRX server returned an empty/broken response previously.
    _reset_index_ticker_singleton()


def request_krx_data(
    payload: dict[str, Any],
    *,
    referer: str = KRX_REFERER,
    timeout: int = 15,
) -> _requests.Response:
    """Issue a raw KRX JSON-endpoint POST using the shared session."""
    session = _get_shared_session()
    headers = {
        "User-Agent": _BROWSER_UA,
        "Referer": referer,
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://data.krx.co.kr",
    }
    return session.post(KRX_JSON_URL, headers=headers, data=payload, timeout=timeout)


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
