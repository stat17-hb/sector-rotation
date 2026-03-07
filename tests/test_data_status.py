"""Pure function tests for data_status module."""
from __future__ import annotations

import pytest

from src.ui.data_status import (
    get_button_states,
    is_sample_mode,
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
        """PYKRX cached paths should not be mislabeled as OpenAPI failure."""
        case = resolve_price_cache_banner_case(
            price_status="CACHED",
            provider_mode="PYKRX",
            openapi_key_present=False,
            market_end_date_str="20260306",
            warm_status={},
        )
        assert case == "pykrx_cache_fallback"
