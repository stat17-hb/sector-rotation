from __future__ import annotations

from src.data_sources.cache_keys import build_macro_cache_token, build_price_cache_token


def test_cache_token_changes_when_key_changes():
    cfg = {"ecos": {"base_rate": {"stat_code": "722Y001", "item_code": "0101000"}}}

    token_a = build_macro_cache_token(cfg, "ECOS_A", "KOSIS_A", 100)
    token_b = build_macro_cache_token(cfg, "ECOS_B", "KOSIS_A", 100)

    assert token_a != token_b


def test_cache_token_stable_for_same_inputs():
    cfg = {"kosis": {"cpi_yoy": {"org_id": "101", "tbl_id": "DT_1J22003", "item_id": "T10"}}}

    token_a = build_macro_cache_token(cfg, "ECOS_A", "KOSIS_A", 200)
    token_b = build_macro_cache_token(cfg, "ECOS_A", "KOSIS_A", 200)

    assert token_a == token_b


def test_cache_token_does_not_expose_raw_keys():
    raw_ecos = "VERY_SECRET_ECOS"
    raw_kosis = "VERY_SECRET_KOSIS"
    cfg = {}

    token = build_macro_cache_token(cfg, raw_ecos, raw_kosis, 1)

    assert raw_ecos not in token
    assert raw_kosis not in token


def test_price_cache_token_changes_when_provider_or_key_changes():
    token_auto = build_price_cache_token(
        krx_provider="AUTO",
        krx_openapi_key="KEY_A",
        secrets_mtime_ns=10,
    )
    token_openapi = build_price_cache_token(
        krx_provider="OPENAPI",
        krx_openapi_key="KEY_A",
        secrets_mtime_ns=10,
    )
    token_key_changed = build_price_cache_token(
        krx_provider="OPENAPI",
        krx_openapi_key="KEY_B",
        secrets_mtime_ns=10,
    )

    assert token_auto != token_openapi
    assert token_openapi != token_key_changed


def test_price_cache_token_is_stable_and_not_leaking_key():
    token_a = build_price_cache_token(
        krx_provider="OPENAPI",
        krx_openapi_key="VERY_SECRET_KRX_KEY",
        secrets_mtime_ns=99,
    )
    token_b = build_price_cache_token(
        krx_provider="OPENAPI",
        krx_openapi_key="VERY_SECRET_KRX_KEY",
        secrets_mtime_ns=99,
    )

    assert token_a == token_b
    assert "VERY_SECRET_KRX_KEY" not in token_a
