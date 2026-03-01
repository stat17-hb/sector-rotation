"""
KRX business day calendar.

R6 — pykrx (KOSPI index) is the primary calendar source.
     Weekend-only subtraction is the fallback.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

from src.data_sources.pykrx_compat import ensure_pykrx_transport_compat

logger = logging.getLogger(__name__)


def get_last_business_day(as_of: date | None = None) -> date:
    """Return the last KRX trading day on or before as_of.

    Priority:
      1. pykrx get_index_ohlcv (KOSPI)
      2. Weekend-only subtraction (no KRX holiday awareness)

    Args:
        as_of: Reference date. Defaults to today.

    Returns:
        Last KRX business day as a date object.
    """
    ref = as_of or date.today()
    start = (ref - timedelta(days=14)).strftime("%Y%m%d")
    end = ref.strftime("%Y%m%d")

    # ── 1. pykrx (primary) ────────────────────────────────────────────────
    try:
        ensure_pykrx_transport_compat()
        from pykrx import stock  # type: ignore[import]

        df = stock.get_index_ohlcv(start, end, "1001", name_display=False)
        if df is None or df.empty:
            raise ValueError("Empty OHLCV response from pykrx")
        return df.index[-1].date()
    except Exception as exc:
        logger.warning(
            "pykrx calendar lookup failed (%s); using weekend-only fallback "
            "(no KRX holiday awareness).",
            exc,
        )

    # ── 3. Weekend-only fallback ────────────────────────────────────────────
    d = ref
    for _ in range(10):
        d -= timedelta(days=1)
        if d.weekday() < 5:  # 0=Mon .. 4=Fri
            return d
    return ref - timedelta(days=1)

