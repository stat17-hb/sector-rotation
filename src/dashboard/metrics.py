"""Shared numeric dashboard metrics helpers."""
from __future__ import annotations

import math
from typing import Any


def safe_float(value: float | int | None) -> float | None:
    """Convert numeric-ish values to float, returning None for missing values."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def compute_rs_divergence_pct(signal: Any) -> float | None:
    """Return the percent gap between RS and RS MA."""
    rs = safe_float(getattr(signal, "rs", None))
    rs_ma = safe_float(getattr(signal, "rs_ma", None))
    if rs is None or rs_ma in {None, 0.0}:
        return None
    return (rs - rs_ma) / rs_ma * 100
