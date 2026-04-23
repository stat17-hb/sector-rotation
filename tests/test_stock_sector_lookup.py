from __future__ import annotations

import pandas as pd

import src.data_sources.stock_sector_lookup as lookup


def _kr_sector_map(*, priorities: dict[str, int | None] | None = None) -> dict:
    if priorities is None:
        priorities = {"5044": 10, "5045": 20}
    return {
        "regimes": {
            "Recovery": {
                "sectors": [
                    {"code": "5044", "name": "KRX 반도체", "lookup_priority": priorities.get("5044")},
                    {"code": "5045", "name": "KRX 헬스케어", "lookup_priority": priorities.get("5045")},
                ]
            }
        }
    }


def _us_sector_map() -> dict:
    return {
        "regimes": {
            "Recovery": {
                "sectors": [
                    {"code": "XLK", "name": "Technology"},
                    {"code": "XLF", "name": "Financials"},
                    {"code": "XLV", "name": "Health Care"},
                ]
            }
        }
    }


def test_resolve_kr_exact_ticker_from_constituent_snapshot(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "read_latest_sector_constituents_snapshot",
        lambda sector_codes, market="KR": pd.DataFrame([{"sector_code": "5044", "ticker": "005930"}]),
    )
    monkeypatch.setattr(
        lookup,
        "read_latest_kr_ticker_names",
        lambda: pd.DataFrame([{"ticker": "005930", "ticker_name": "삼성전자"}]),
    )
    monkeypatch.setattr(lookup, "_fetch_kr_live_ticker_name", lambda ticker: "삼성전자")

    result = lookup.resolve_stock_to_sector("005930", "KR", _kr_sector_map(), asof_date="20260419")

    assert result["status"] == "success"
    assert result["sector_code"] == "5044"
    assert result["sector_name"] == "KRX 반도체"
    assert result["resolution_kind"] == "constituent_membership"
    assert result["confidence"] == "high"
    assert result["canonicalization_applied"] is False
    assert result["canonicalization_basis"] == "single_match"
    assert result["match_date_mode"] == "not_applicable"


def test_resolve_kr_exact_name_via_latest_ticker_names(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "read_latest_sector_constituents_snapshot",
        lambda sector_codes, market="KR": pd.DataFrame([{"sector_code": "5044", "ticker": "005930"}]),
    )
    monkeypatch.setattr(
        lookup,
        "read_latest_kr_ticker_names",
        lambda: pd.DataFrame([{"ticker": "005930", "ticker_name": "삼성전자"}]),
    )

    result = lookup.resolve_stock_to_sector("삼성전자", "KR", _kr_sector_map(), asof_date="20260419")

    assert result["status"] == "success"
    assert result["matched_symbol"] == "005930"
    assert result["sector_name"] == "KRX 반도체"
    assert result["confidence"] == "medium"
    assert result["canonicalization_basis"] == "single_match"


def test_resolve_kr_uses_live_name_fallback_after_warehouse_miss(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "read_latest_sector_constituents_snapshot",
        lambda sector_codes, market="KR": pd.DataFrame([{"sector_code": "5044", "ticker": "005930"}]),
    )
    monkeypatch.setattr(lookup, "read_latest_kr_ticker_names", lambda: pd.DataFrame(columns=["ticker", "ticker_name"]))
    monkeypatch.setattr(
        lookup,
        "_read_live_kr_constituents",
        lambda sector_codes, asof_date: pd.DataFrame(
            [{"sector_code": "5044", "ticker": "005930", "resolved_from": "20260419", "source": "krx_raw_payload", "snapshot_date": "20260419"}]
        ),
    )
    monkeypatch.setattr(lookup, "_fetch_kr_live_ticker_name", lambda ticker: "삼성전자")

    result = lookup.resolve_stock_to_sector("삼성전자", "KR", _kr_sector_map(), asof_date="20260419")

    assert result["status"] == "success"
    assert result["source"] == "krx_raw_payload"


def test_resolve_kr_uses_live_name_fallback_when_snapshot_is_partial(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "read_latest_sector_constituents_snapshot",
        lambda sector_codes, market="KR": pd.DataFrame([{"sector_code": "5045", "ticker": "000660"}]),
    )
    monkeypatch.setattr(lookup, "read_latest_kr_ticker_names", lambda: pd.DataFrame(columns=["ticker", "ticker_name"]))
    monkeypatch.setattr(
        lookup,
        "_read_live_kr_constituents",
        lambda sector_codes, asof_date: pd.DataFrame(
            [{"sector_code": "5044", "ticker": "005930", "resolved_from": "20260419", "source": "krx_raw_payload", "snapshot_date": "20260419"}]
        ),
    )
    monkeypatch.setattr(
        lookup,
        "_fetch_kr_live_ticker_name",
        lambda ticker: "삼성전자" if ticker == "005930" else "SK하이닉스",
    )

    result = lookup.resolve_stock_to_sector("삼성전자", "KR", _kr_sector_map(), asof_date="20260419")

    assert result["status"] == "success"
    assert result["sector_code"] == "5044"
    assert result["source"] == "krx_raw_payload"


def test_resolve_kr_ambiguous_name_returns_ambiguous(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "read_latest_sector_constituents_snapshot",
        lambda sector_codes, market="KR": pd.DataFrame(
            [{"sector_code": "5044", "ticker": "005930"}, {"sector_code": "5045", "ticker": "000001"}]
        ),
    )
    monkeypatch.setattr(
        lookup,
        "read_latest_kr_ticker_names",
        lambda: pd.DataFrame(
            [
                {"ticker": "005930", "ticker_name": "삼성전자"},
                {"ticker": "000001", "ticker_name": "삼성전자"},
            ]
        ),
    )

    result = lookup.resolve_stock_to_sector("삼성전자", "KR", _kr_sector_map(), asof_date="20260419")

    assert result["status"] == "ambiguous"
    assert result["sector_name"] == ""


def test_resolve_kr_same_date_unique_lowest_priority_wins(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "read_latest_sector_constituents_snapshot",
        lambda sector_codes, market="KR": pd.DataFrame([
            {"sector_code": "5044", "ticker": "005930", "snapshot_date": "2026-04-17", "resolved_from": "20260417"},
            {"sector_code": "5045", "ticker": "005930", "snapshot_date": "2026-04-17", "resolved_from": "20260417"},
        ]),
    )
    monkeypatch.setattr(lookup, "read_latest_kr_ticker_names", lambda: pd.DataFrame(columns=["ticker", "ticker_name"]))
    monkeypatch.setattr(lookup, "_fetch_kr_live_ticker_name", lambda ticker: "삼성전자")

    result = lookup.resolve_stock_to_sector("005930", "KR", _kr_sector_map(), asof_date="20260419")

    assert result["status"] == "success"
    assert result["sector_code"] == "5044"
    assert result["canonicalization_applied"] is True
    assert result["canonicalization_basis"] == "lookup_priority_same_date"
    assert result["match_date_mode"] == "same_date"
    assert result["match_effective_date"] == "2026-04-17"
    assert len(result["matched_sector_candidates"]) == 2


def test_resolve_kr_equal_lowest_priority_stays_ambiguous(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "read_latest_sector_constituents_snapshot",
        lambda sector_codes, market="KR": pd.DataFrame([
            {"sector_code": "5044", "ticker": "005930", "snapshot_date": "2026-04-17", "resolved_from": "20260417"},
            {"sector_code": "5045", "ticker": "005930", "snapshot_date": "2026-04-17", "resolved_from": "20260417"},
        ]),
    )
    monkeypatch.setattr(lookup, "read_latest_kr_ticker_names", lambda: pd.DataFrame(columns=["ticker", "ticker_name"]))
    monkeypatch.setattr(lookup, "_fetch_kr_live_ticker_name", lambda ticker: "삼성전자")

    result = lookup.resolve_stock_to_sector(
        "005930",
        "KR",
        _kr_sector_map(priorities={"5044": 10, "5045": 10}),
        asof_date="20260419",
    )

    assert result["status"] == "ambiguous"
    assert result["canonicalization_basis"] == "equal_lowest_priority"
    assert result["match_date_mode"] == "same_date"


def test_resolve_kr_missing_priorities_stays_ambiguous(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "read_latest_sector_constituents_snapshot",
        lambda sector_codes, market="KR": pd.DataFrame([
            {"sector_code": "5044", "ticker": "005930", "snapshot_date": "2026-04-17", "resolved_from": "20260417"},
            {"sector_code": "5045", "ticker": "005930", "snapshot_date": "2026-04-17", "resolved_from": "20260417"},
        ]),
    )
    monkeypatch.setattr(lookup, "read_latest_kr_ticker_names", lambda: pd.DataFrame(columns=["ticker", "ticker_name"]))
    monkeypatch.setattr(lookup, "_fetch_kr_live_ticker_name", lambda ticker: "삼성전자")

    result = lookup.resolve_stock_to_sector(
        "005930",
        "KR",
        _kr_sector_map(priorities={}),
        asof_date="20260419",
    )

    assert result["status"] == "ambiguous"
    assert result["canonicalization_basis"] == "missing_lookup_priority"
    assert result["match_date_mode"] == "same_date"


def test_resolve_kr_mixed_effective_dates_stays_ambiguous(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "read_latest_sector_constituents_snapshot",
        lambda sector_codes, market="KR": pd.DataFrame([
            {"sector_code": "5044", "ticker": "005930", "snapshot_date": "2026-04-17", "resolved_from": "20260417"},
            {"sector_code": "5045", "ticker": "005930", "snapshot_date": "2026-04-16", "resolved_from": "20260416"},
        ]),
    )
    monkeypatch.setattr(lookup, "read_latest_kr_ticker_names", lambda: pd.DataFrame(columns=["ticker", "ticker_name"]))
    monkeypatch.setattr(lookup, "_fetch_kr_live_ticker_name", lambda ticker: "삼성전자")

    result = lookup.resolve_stock_to_sector("005930", "KR", _kr_sector_map(), asof_date="20260419")

    assert result["status"] == "ambiguous"
    assert result["canonicalization_applied"] is False
    assert result["canonicalization_basis"] == "mixed_effective_dates"
    assert result["match_date_mode"] == "mixed"


def test_resolve_kr_non_success_when_snapshot_and_live_both_miss(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "read_latest_sector_constituents_snapshot",
        lambda sector_codes, market="KR": pd.DataFrame(columns=["sector_code", "ticker", "snapshot_date", "resolved_from"]),
    )
    monkeypatch.setattr(lookup, "read_latest_kr_ticker_names", lambda: pd.DataFrame(columns=["ticker", "ticker_name"]))
    monkeypatch.setattr(
        lookup,
        "_read_live_kr_constituents",
        lambda sector_codes, asof_date: pd.DataFrame(columns=["sector_code", "ticker", "resolved_from", "source", "snapshot_date"]),
    )
    monkeypatch.setattr(lookup, "_fetch_kr_live_ticker_name", lambda ticker: "")

    result = lookup.resolve_stock_to_sector("005930", "KR", _kr_sector_map(), asof_date="20260419")

    assert result["status"] == "not_found"


def test_resolve_kr_prefers_official_sector_universe_rows_over_subset_map(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "read_latest_sector_constituents_snapshot",
        lambda sector_codes, market="KR": pd.DataFrame([{"sector_code": "5064", "ticker": "005930"}]),
    )
    monkeypatch.setattr(
        lookup,
        "read_latest_kr_ticker_names",
        lambda: pd.DataFrame([{"ticker": "005930", "ticker_name": "삼성전자"}]),
    )
    monkeypatch.setattr(lookup, "_fetch_kr_live_ticker_name", lambda ticker: "삼성전자")

    result = lookup.resolve_stock_to_sector(
        "005930",
        "KR",
        _kr_sector_map(),
        asof_date="20260419",
        sector_universe_rows=[
            {"index_code": "1001", "index_name": "코스피", "family": "kospi_dd_trd"},
            {"index_code": "5042", "index_name": "KRX 100", "family": "krx_dd_trd"},
            {"index_code": "5064", "index_name": "KRX 정보기술", "family": "krx_dd_trd"},
            {"index_code": "5351", "index_name": "KRX 300 정보기술", "family": "krx_dd_trd"},
            {"index_code": "1155", "index_name": "코스피 200 정보기술", "family": "kospi_dd_trd"},
        ],
    )

    assert result["status"] == "success"
    assert result["sector_code"] == "5064"
    assert result["sector_name"] == "KRX 정보기술"
    assert result["canonicalization_basis"] == "single_match"


def test_resolve_kr_official_universe_keeps_config_names_for_overlap_codes(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "read_latest_sector_constituents_snapshot",
        lambda sector_codes, market="KR": pd.DataFrame([
            {"sector_code": "1155", "ticker": "005930", "snapshot_date": "2026-04-17", "resolved_from": "20260417"},
            {"sector_code": "5044", "ticker": "005930", "snapshot_date": "2026-04-17", "resolved_from": "20260417"},
        ]),
    )
    monkeypatch.setattr(
        lookup,
        "read_latest_kr_ticker_names",
        lambda: pd.DataFrame([{"ticker": "005930", "ticker_name": "삼성전자"}]),
    )
    monkeypatch.setattr(lookup, "_fetch_kr_live_ticker_name", lambda ticker: "삼성전자")

    result = lookup.resolve_stock_to_sector(
        "005930",
        "KR",
        {
            "regimes": {
                "Recovery": {"sectors": [{"code": "1155", "name": "KOSPI200 정보기술", "lookup_priority": 20}]},
                "Expansion": {"sectors": [{"code": "5044", "name": "KRX 반도체", "lookup_priority": 10}]},
            }
        },
        asof_date="20260419",
        sector_universe_rows=[
            {"index_code": "5044", "index_name": "KRX 반도체"},
            {"index_code": "5064", "index_name": "KRX 정보기술"},
        ],
    )

    assert result["status"] == "success"
    assert result["sector_code"] == "5044"
    assert result["sector_name"] == "KRX 반도체"
    assert [item["sector_name"] for item in result["matched_sector_candidates"]] == [
        "KOSPI200 정보기술",
        "KRX 반도체",
    ]


def test_resolve_kr_official_universe_overrides_stale_config_name_for_same_code(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "read_latest_sector_constituents_snapshot",
        lambda sector_codes, market="KR": pd.DataFrame([
            {"sector_code": "5044", "ticker": "005930", "snapshot_date": "2026-04-17", "resolved_from": "20260417"},
        ]),
    )
    monkeypatch.setattr(
        lookup,
        "read_latest_kr_ticker_names",
        lambda: pd.DataFrame([{"ticker": "005930", "ticker_name": "삼성전자"}]),
    )
    monkeypatch.setattr(lookup, "_fetch_kr_live_ticker_name", lambda ticker: "삼성전자")

    result = lookup.resolve_stock_to_sector(
        "005930",
        "KR",
        {
            "regimes": {
                "Expansion": {
                    "sectors": [{"code": "5044", "name": "Stale config name", "lookup_priority": 10}]
                }
            }
        },
        asof_date="20260419",
        sector_universe_rows=[
            {"index_code": "5044", "index_name": "KRX 반도체"},
        ],
    )

    assert result["status"] == "success"
    assert result["sector_name"] == "KRX 반도체"
    assert result["matched_sector_candidates"][0]["sector_name"] == "KRX 반도체"


def test_resolve_us_exact_ticker_via_sec_and_yfinance_sector(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "fetch_sec_company_tickers",
        lambda: pd.DataFrame([{"ticker": "MSFT", "title": "Microsoft Corp", "normalized_title": "MICROSOFT"}]),
    )
    monkeypatch.setattr(
        lookup,
        "_fetch_us_issuer_metadata",
        lambda symbol: {"sector": "Technology", "sectorKey": "", "industryKey": ""},
    )

    result = lookup.resolve_stock_to_sector("MSFT", "US", _us_sector_map())

    assert result["status"] == "success"
    assert result["sector_code"] == "XLK"
    assert result["sector_name"] == "Technology"
    assert result["resolution_kind"] == "issuer_classification"
    assert result["confidence"] == "high"


def test_resolve_us_exact_name_via_normalized_title(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "fetch_sec_company_tickers",
        lambda: pd.DataFrame([{"ticker": "MSFT", "title": "Microsoft Corp", "normalized_title": "MICROSOFT"}]),
    )
    monkeypatch.setattr(
        lookup,
        "_fetch_us_issuer_metadata",
        lambda symbol: {"sector": "", "sectorKey": "technology", "industryKey": ""},
    )

    result = lookup.resolve_stock_to_sector("Microsoft", "US", _us_sector_map())

    assert result["status"] == "success"
    assert result["matched_symbol"] == "MSFT"
    assert result["confidence"] == "medium"
    assert result["source"] == "yfinance_sectorKey"


def test_resolve_us_uses_industry_key_only_when_deterministic(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "fetch_sec_company_tickers",
        lambda: pd.DataFrame([{"ticker": "BIOT", "title": "Bio Tech Inc", "normalized_title": "BIO TECH"}]),
    )
    monkeypatch.setattr(
        lookup,
        "_fetch_us_issuer_metadata",
        lambda symbol: {"sector": "", "sectorKey": "", "industryKey": "biotechnology"},
    )

    result = lookup.resolve_stock_to_sector("BIOT", "US", _us_sector_map())

    assert result["status"] == "success"
    assert result["sector_code"] == "XLV"
    assert result["source"] == "yfinance_industryKey"


def test_resolve_us_unmappable_provider_metadata_returns_non_success(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "fetch_sec_company_tickers",
        lambda: pd.DataFrame([{"ticker": "MSFT", "title": "Microsoft Corp", "normalized_title": "MICROSOFT"}]),
    )
    monkeypatch.setattr(
        lookup,
        "_fetch_us_issuer_metadata",
        lambda symbol: {"sector": "Unknown Sector", "sectorKey": "unknown", "industryKey": "unknown"},
    )

    result = lookup.resolve_stock_to_sector("MSFT", "US", _us_sector_map())

    assert result["status"] == "unsupported"
    assert result["sector_name"] == ""


def test_resolve_us_ambiguous_name_returns_ambiguous(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "fetch_sec_company_tickers",
        lambda: pd.DataFrame(
            [
                {"ticker": "AAA", "title": "Acme Corp", "normalized_title": "ACME"},
                {"ticker": "BBB", "title": "Acme Holdings", "normalized_title": "ACME"},
            ]
        ),
    )

    result = lookup.resolve_stock_to_sector("Acme", "US", _us_sector_map())

    assert result["status"] == "ambiguous"


def test_resolve_us_prefers_exact_ticker_over_name_match(monkeypatch):
    monkeypatch.setattr(
        lookup,
        "fetch_sec_company_tickers",
        lambda: pd.DataFrame(
            [
                {"ticker": "MSFT", "title": "Microsoft Corp", "normalized_title": "MICROSOFT"},
                {"ticker": "MICROSOFT", "title": "Microsoft Holdings", "normalized_title": "MICROSOFT HOLDINGS"},
            ]
        ),
    )
    monkeypatch.setattr(
        lookup,
        "_fetch_us_issuer_metadata",
        lambda symbol: {"sector": "Technology", "sectorKey": "", "industryKey": ""},
    )

    result = lookup.resolve_stock_to_sector("MSFT", "US", _us_sector_map())

    assert result["status"] == "success"
    assert result["matched_symbol"] == "MSFT"
    assert result["confidence"] == "high"
