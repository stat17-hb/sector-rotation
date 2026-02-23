"""
KRX business day calendar.

R6 — pykrx is the sole calendar authority. No static holiday table.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

from src.data_sources.pykrx_compat import ensure_pykrx_transport_compat

logger = logging.getLogger(__name__)


def get_last_business_day(as_of: date | None = None) -> date:
    """Return the last KRX trading day on or before as_of.

    Uses pykrx get_index_ohlcv (KOSPI=1001) as the sole KRX calendar source.
    Falls back to weekend-only subtraction as best-effort if pykrx fails.
    No static holiday table — holiday awareness comes entirely from pykrx.

    Args:
        as_of: Reference date. Defaults to today.

    Returns:
        Last KRX business day as a date object.
    """
    ref = as_of or date.today()

    try:
        ensure_pykrx_transport_compat()
        from pykrx import stock  # type: ignore[import]

        start = (ref - timedelta(days=14)).strftime("%Y%m%d")
        end = ref.strftime("%Y%m%d")
        df = stock.get_index_ohlcv(start, end, "1001")
        if df is None or df.empty:
            raise ValueError("Empty OHLCV response from pykrx")
        return df.index[-1].date()
    except Exception as exc:
        logger.warning(
            "pykrx calendar lookup failed (%s); using weekend-only fallback "
            "(no KRX holiday awareness).",
            exc,
        )
        # Explicit weekend-only fallback — no KRX holiday awareness
        d = ref
        for _ in range(10):
            d -= timedelta(days=1)
            if d.weekday() < 5:  # 0=Mon .. 4=Fri
                return d
        return ref - timedelta(days=1)
