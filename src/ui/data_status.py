"""
Pure functions for data status checking and button state management.

R9 — SAMPLE mode: pure helper functions, no Streamlit dependency.
These can be tested without a Streamlit runtime.
"""
from __future__ import annotations

from typing import Any, Literal, Mapping


PriceCacheBannerCase = Literal[
    "fresh_cache",
    "retryable_cache_fallback",
    "missing_openapi_key",
    "pykrx_cache_fallback",
]


def _normalize_yyyymmdd(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits if len(digits) == 8 else ""


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key).strip(): str(val).strip()
        for key, val in value.items()
        if str(key).strip() and str(val).strip()
    }


def is_sample_mode(data_status: dict) -> bool:
    """Return True if any data source is in SAMPLE mode.

    Args:
        data_status: Dict mapping source names to DataStatus strings.
                     Example: {"price": "SAMPLE", "ecos": "LIVE"}

    Returns:
        True if any value equals "SAMPLE", False otherwise.
    """
    return any(v == "SAMPLE" for v in data_status.values())


def get_button_states(data_status: dict) -> dict[str, bool]:
    """Return enabled/disabled state for each dashboard button.

    Button rules (R9):
    - 시장데이터 갱신 (refresh_market):  always enabled — attempts to escape SAMPLE
    - 매크로데이터 갱신 (refresh_macro):  always enabled — attempts to escape SAMPLE
    - 전체 재계산 (recompute):           disabled in SAMPLE (meaningless on synthetic data)

    Args:
        data_status: Dict mapping source names to DataStatus strings.

    Returns:
        dict with keys "refresh_market", "refresh_macro", "recompute".
    """
    sample = is_sample_mode(data_status)
    return {
        "refresh_market": True,
        "refresh_macro": True,
        "recompute": not sample,
    }


def resolve_price_cache_banner_case(
    *,
    price_status: str,
    provider_mode: str,
    openapi_key_present: bool,
    market_end_date_str: str,
    warm_status: Mapping[str, Any] | None,
) -> PriceCacheBannerCase | None:
    """Resolve how the UI should describe cached KRX market data."""
    if str(price_status).strip().upper() != "CACHED":
        return None

    provider = str(provider_mode or "").strip().upper()
    if provider != "OPENAPI":
        return "pykrx_cache_fallback"
    if not openapi_key_present:
        return "missing_openapi_key"

    warm = dict(warm_status or {})
    warm_state = str(warm.get("status", "")).strip().upper()
    warm_end = _normalize_yyyymmdd(warm.get("end") or warm.get("watermark_key"))
    requested_end = _normalize_yyyymmdd(market_end_date_str)
    coverage_complete = bool(warm.get("coverage_complete"))
    failed_days = _normalize_string_list(warm.get("failed_days"))
    failed_codes = _normalize_string_map(warm.get("failed_codes"))

    if (
        warm_state in {"LIVE", "CACHED"}
        and coverage_complete
        and warm_end
        and warm_end == requested_end
        and not failed_days
        and not failed_codes
    ):
        return "fresh_cache"

    return "retryable_cache_fallback"
