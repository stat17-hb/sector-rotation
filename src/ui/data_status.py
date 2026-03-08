"""
Pure helpers for dashboard data-status messaging.

These helpers stay Streamlit-free so the app can unit test banner priority
and button-state behavior without a runtime.
"""
from __future__ import annotations

from typing import Any, Literal, Mapping, TypedDict


PriceCacheBannerCase = Literal[
    "fresh_cache",
    "retryable_cache_fallback",
    "missing_openapi_key",
    "pykrx_cache_fallback",
]


class DashboardStatusBanner(TypedDict):
    level: Literal["error", "warning", "info"]
    title: str
    message: str
    details: list[str]


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


def is_sample_mode(data_status: Mapping[str, str]) -> bool:
    """Return True if any data source is in SAMPLE mode."""
    return any(str(value).strip().upper() == "SAMPLE" for value in data_status.values())


def get_button_states(data_status: Mapping[str, str]) -> dict[str, bool]:
    """Return enabled/disabled state for each dashboard action button."""
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


def _summarize_preflight_issues(
    preflight_status: Mapping[str, Mapping[str, Any]] | None,
) -> list[str]:
    details: list[str] = []
    if not preflight_status:
        return details

    for name, raw_info in preflight_status.items():
        info = dict(raw_info or {})
        status = str(info.get("status", "")).strip().upper()
        if not status or status == "OK":
            continue
        detail = str(info.get("detail", "")).strip()
        if detail:
            details.append(f"{name}: {status} ({detail})")
        else:
            details.append(f"{name}: {status}")
    return details


def _summarize_warm_failures(warm_status: Mapping[str, Any] | None) -> list[str]:
    warm = dict(warm_status or {})
    details: list[str] = []

    failed_days = _normalize_string_list(warm.get("failed_days"))
    if failed_days:
        preview = ", ".join(failed_days[:5])
        suffix = "" if len(failed_days) <= 5 else ", ..."
        details.append(f"실패 일자: {preview}{suffix}")

    failed_codes = _normalize_string_map(warm.get("failed_codes"))
    if failed_codes:
        preview_items = list(sorted(failed_codes.items()))[:5]
        preview = ", ".join(f"{code}={reason}" for code, reason in preview_items)
        suffix = "" if len(failed_codes) <= 5 else ", ..."
        details.append(f"실패 종목: {preview}{suffix}")

    return details


def resolve_dashboard_status_banner(
    *,
    data_status: Mapping[str, str],
    market_blocking_error: str = "",
    price_cache_case: PriceCacheBannerCase | None = None,
    openapi_key_warning: bool = False,
    preflight_status: Mapping[str, Mapping[str, Any]] | None = None,
    price_warm_status: Mapping[str, Any] | None = None,
) -> DashboardStatusBanner | None:
    """Return the highest-priority dashboard status banner payload."""
    price_status = str(data_status.get("price", "")).strip().upper()
    macro_status = str(data_status.get("macro", "")).strip().upper()
    preflight_details = _summarize_preflight_issues(preflight_status)
    warm_details = _summarize_warm_failures(price_warm_status)

    if price_status == "BLOCKED" and market_blocking_error:
        return {
            "level": "error",
            "title": "시장 데이터 접근 차단",
            "message": "시장 데이터 새로고침이 차단되었습니다. 권한 또는 공급자 상태를 확인하세요.",
            "details": [market_blocking_error],
        }

    if is_sample_mode(data_status):
        details = [
            "실시간 시장 데이터를 불러오지 못해 SAMPLE 데이터로 표시 중입니다.",
            "API 설정 또는 공급자 상태를 확인한 뒤 새로고침을 다시 실행하세요.",
        ]
        details.extend(preflight_details)
        return {
            "level": "error",
            "title": "SAMPLE 데이터 모드",
            "message": "현재 화면은 합성 데이터 또는 대체 데이터에 기반합니다.",
            "details": details,
        }

    if openapi_key_warning or price_cache_case == "missing_openapi_key":
        return {
            "level": "warning",
            "title": "KRX OpenAPI 키 누락",
            "message": "시장 데이터는 캐시 또는 대체 경로를 사용하고 있습니다.",
            "details": [
                "KRX_PROVIDER가 OPENAPI로 설정되어 있지만 KRX_OPENAPI_KEY가 없습니다.",
                "OpenAPI 키를 추가한 뒤 시장 데이터 새로고침을 다시 실행하세요.",
            ],
        }

    if price_cache_case == "retryable_cache_fallback":
        details = [
            "시장 데이터는 현재 로컬 warehouse cache를 사용 중입니다.",
            "잠시 후 새로고침을 다시 시도하거나 현재 캐시로 계속 진행할 수 있습니다.",
        ]
        details.extend(warm_details)
        return {
            "level": "warning",
            "title": "시장 데이터 캐시 사용 중",
            "message": "최근 OpenAPI warm이 현재 커버리지를 완전히 확인하지 못했습니다.",
            "details": details,
        }

    if price_cache_case == "pykrx_cache_fallback":
        return {
            "level": "warning",
            "title": "시장 데이터 캐시 사용 중",
            "message": "pykrx 경로가 실시간 응답을 반환하지 않아 캐시를 사용하고 있습니다.",
            "details": [
                "pykrx는 인증 세션 상태에 따라 응답이 불안정할 수 있습니다.",
                "가능하면 OPENAPI 경로를 사용하거나 나중에 다시 시도하세요.",
            ],
        }

    if macro_status == "CACHED":
        details = ["매크로 데이터는 현재 로컬 warehouse cache를 사용 중입니다."]
        details.extend(preflight_details)
        return {
            "level": "warning",
            "title": "매크로 데이터 캐시 사용 중",
            "message": "매크로 공급자 최신 응답을 확인하지 못해 캐시 데이터를 보여주고 있습니다.",
            "details": details,
        }

    if preflight_details:
        return {
            "level": "info",
            "title": "API 사전 점검 참고",
            "message": "일부 엔드포인트 연결 상태를 확인하지 못했지만 현재 화면은 계속 사용할 수 있습니다.",
            "details": preflight_details,
        }

    return None
