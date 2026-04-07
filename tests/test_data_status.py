"""Pure function tests for data_status module."""
from __future__ import annotations

import pytest

from src.ui.data_status import (
    get_button_states,
    is_sample_mode,
    resolve_dashboard_status_banner,
    resolve_price_cache_banner_case,
)


class TestDataStatus:
    def test_is_sample_mode_true_when_any_sample(self):
        """is_sample_mode returns True when any source is SAMPLE."""
        assert is_sample_mode({"price": "SAMPLE", "ecos": "LIVE"}) is True

    def test_is_sample_mode_false_when_no_sample(self):
        """is_sample_mode returns False when no source is SAMPLE."""
        assert is_sample_mode({"price": "LIVE", "ecos": "CACHED"}) is False

    def test_get_button_states_recompute_disabled_in_sample(self):
        """recompute button is disabled when any source is SAMPLE."""
        states = get_button_states({"price": "SAMPLE"})
        assert states["recompute"] is False

    def test_get_button_states_recompute_enabled_in_live(self):
        """recompute button is enabled when no source is SAMPLE."""
        states = get_button_states({"price": "LIVE"})
        assert states["recompute"] is True
        assert states["refresh_market"] is True
        assert states["refresh_macro"] is True

    def test_resolve_price_cache_banner_case_fresh_openapi_cache(self):
        """A current successful warm should be presented as fresh cache, not failure."""
        case = resolve_price_cache_banner_case(
            price_status="CACHED",
            provider_mode="OPENAPI",
            openapi_key_present=True,
            market_end_date_str="20260306",
            warm_status={
                "status": "LIVE",
                "end": "20260306",
                "coverage_complete": True,
                "failed_days": [],
                "failed_codes": {},
            },
        )
        assert case == "fresh_cache"

    def test_resolve_price_cache_banner_case_fresh_openapi_noop_cache(self):
        """A complete no-op warm should stay on the fresh-cache path."""
        case = resolve_price_cache_banner_case(
            price_status="CACHED",
            provider_mode="OPENAPI",
            openapi_key_present=True,
            market_end_date_str="20260306",
            warm_status={
                "status": "CACHED",
                "watermark_key": "20260306",
                "coverage_complete": True,
                "failed_days": [],
                "failed_codes": {},
            },
        )
        assert case == "fresh_cache"

    def test_resolve_price_cache_banner_case_retryable_openapi_fallback(self):
        """Incomplete warm evidence should keep the retryable warning path."""
        case = resolve_price_cache_banner_case(
            price_status="CACHED",
            provider_mode="OPENAPI",
            openapi_key_present=True,
            market_end_date_str="20260306",
            warm_status={
                "status": "CACHED",
                "end": "20260306",
                "coverage_complete": False,
                "failed_days": ["20260306"],
                "failed_codes": {},
            },
        )
        assert case == "retryable_cache_fallback"

    def test_resolve_price_cache_banner_case_missing_openapi_key(self):
        """Explicit OPENAPI-without-key should keep the key-missing path."""
        case = resolve_price_cache_banner_case(
            price_status="CACHED",
            provider_mode="OPENAPI",
            openapi_key_present=False,
            market_end_date_str="20260306",
            warm_status={},
        )
        assert case == "missing_openapi_key"

    def test_resolve_price_cache_banner_case_pykrx_fallback(self):
        """PYKRX with incomplete coverage should still return pykrx_cache_fallback."""
        case = resolve_price_cache_banner_case(
            price_status="CACHED",
            provider_mode="PYKRX",
            openapi_key_present=False,
            market_end_date_str="20260306",
            warm_status={},
        )
        assert case == "pykrx_cache_fallback"

    def test_resolve_price_cache_banner_case_pykrx_fresh_cache(self):
        """PYKRX with complete coverage and matching end date should return fresh_cache."""
        case = resolve_price_cache_banner_case(
            price_status="CACHED",
            provider_mode="PYKRX",
            openapi_key_present=False,
            market_end_date_str="20260306",
            warm_status={
                "status": "CACHED",
                "coverage_complete": True,
                "end": "20260306",
                "failed_days": [],
                "failed_codes": {},
            },
        )
        assert case == "fresh_cache"

    def test_resolve_price_cache_banner_case_pykrx_fallback_on_failed_codes(self):
        """PYKRX with failed codes should still return pykrx_cache_fallback."""
        case = resolve_price_cache_banner_case(
            price_status="CACHED",
            provider_mode="PYKRX",
            openapi_key_present=False,
            market_end_date_str="20260306",
            warm_status={
                "status": "CACHED",
                "coverage_complete": True,
                "end": "20260306",
                "failed_days": [],
                "failed_codes": {"1028": "empty response"},
            },
        )
        assert case == "pykrx_cache_fallback"

    def test_resolve_dashboard_status_banner_prefers_sample_over_lower_priority_items(self):
        banner = resolve_dashboard_status_banner(
            data_status={"price": "SAMPLE", "macro": "CACHED"},
            price_cache_case="retryable_cache_fallback",
            preflight_status={
                "KRX": {"status": "HTTP_ERROR", "detail": "timeout"},
            },
            price_warm_status={"failed_days": ["20260306"]},
        )

        assert banner is not None
        assert banner["level"] == "error"
        assert banner["title"] == "SAMPLE 데이터 모드"

    def test_resolve_dashboard_status_banner_prefers_blocked_over_general_warning(self):
        banner = resolve_dashboard_status_banner(
            data_status={"price": "BLOCKED", "macro": "LIVE"},
            market_blocking_error="Access denied",
            price_cache_case="retryable_cache_fallback",
            preflight_status={
                "KRX": {"status": "ACCESS_DENIED", "detail": "auth"},
            },
        )

        assert banner is not None
        assert banner["level"] == "error"
        assert banner["title"] == "시장 데이터 접근 차단"
        assert "Access denied" in banner["details"][0]

    def test_resolve_dashboard_status_banner_uses_single_missing_key_message(self):
        banner = resolve_dashboard_status_banner(
            data_status={"price": "CACHED", "macro": "LIVE"},
            price_cache_case="missing_openapi_key",
            openapi_key_warning=True,
            preflight_status={
                "KRX": {"status": "OK", "detail": ""},
            },
        )

        assert banner is not None
        assert banner["level"] == "warning"
        assert banner["title"] == "KRX OpenAPI 키 누락"
        assert len(banner["details"]) == 2

    def test_resolve_dashboard_status_banner_surfaces_preflight_when_no_higher_priority(self):
        banner = resolve_dashboard_status_banner(
            data_status={"price": "LIVE", "macro": "LIVE"},
            preflight_status={
                "ECOS": {"status": "HTTP_ERROR", "detail": "timeout"},
            },
        )

        assert banner is not None
        assert banner["level"] == "info"
        assert banner["title"] == "API 사전 점검 참고"
        assert "ECOS: HTTP_ERROR (timeout)" in banner["details"]
