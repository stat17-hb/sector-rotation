from __future__ import annotations

from src.data_sources.macro_freshness import build_macro_freshness_payload, summarize_macro_freshness


def test_build_macro_freshness_marks_source_lag_and_not_configured_imports():
    payload = build_macro_freshness_payload(
        requested_end="202606",
        provider_configs={
            "ECOS": {
                "export_amount": {"enabled": True},
                "usdkrw": {"enabled": True},
            },
            "KOSIS": {
                "cpi_yoy": {"enabled": True},
                "leading_index": {"enabled": True},
            },
        },
        latest_periods={
            "export_amount": "202604",
            "usdkrw": "202606",
            "cpi_yoy": "202606",
            "leading_index": "202606",
        },
    )

    freshness = payload["freshness"]

    assert freshness["requested_end"] == "202606"
    assert freshness["overall_reason"] == "NOT_CONFIGURED"
    assert freshness["groups"]["aggregate_exports"]["reason"] == "SOURCE_LAG"
    assert freshness["groups"]["aggregate_exports"]["latest_period"] == "202604"
    assert freshness["groups"]["aggregate_imports"]["reason"] == "NOT_CONFIGURED"
    assert freshness["aliases"]["export_amount"]["reason"] == "SOURCE_LAG"


def test_build_macro_freshness_reason_precedence_prefers_write_lock():
    payload = build_macro_freshness_payload(
        requested_end="202606",
        provider_configs={"ECOS": {"export_amount": {"enabled": True}}},
        latest_periods={"export_amount": "202606"},
        failed_aliases={"export_amount": "timeout"},
        write_lock_fallback=True,
    )

    assert payload["freshness"]["overall_reason"] == "WRITE_LOCK_FALLBACK"
    assert payload["freshness"]["aliases"]["export_amount"]["reason"] == "WRITE_LOCK_FALLBACK"


def test_summarize_macro_freshness_returns_user_facing_details():
    payload = build_macro_freshness_payload(
        requested_end="202606",
        provider_configs={"ECOS": {"export_amount": {"enabled": True}}},
        latest_periods={"export_amount": "202604"},
    )

    details = summarize_macro_freshness(payload)

    assert "수출: 소스 최신월 지연 최신 2026-04" in details
    assert "수입: 미설정" in details
