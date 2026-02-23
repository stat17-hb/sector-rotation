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

logger = logging.getLogger(__name__)

KRX_REFERER = "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd"
KRX_JSON_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
KRX_RESOURCE_URL = (
    "https://data.krx.co.kr/comm/bldAttendant/executeForResourceBundle.cmd"
)

_PATCH_APPLIED = False
_PATCH_LOCK = threading.Lock()
_PATCH_ATTR = "__sector_rotation_pykrx_transport_patched__"


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

        _PATCH_APPLIED = True
        logger.info("Applied pykrx transport compatibility patch.")


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
