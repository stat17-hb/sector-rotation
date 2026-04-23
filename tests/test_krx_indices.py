from __future__ import annotations

import pandas as pd

import src.data_sources.krx_indices as krx_indices


def test_discover_kr_index_rows_via_finder_builds_prefixed_codes(monkeypatch):
    class _FakeFinder:
        def fetch(self, mktsel: str) -> pd.DataFrame:
            if mktsel == "2":
                return pd.DataFrame(
                    [
                        {"full_code": "5", "short_code": "064", "codeName": "KRX 정보기술"},
                        {"full_code": "5", "short_code": "046", "codeName": "KRX 은행"},
                    ]
                )
            if mktsel == "3":
                return pd.DataFrame(
                    [
                        {"full_code": "1", "short_code": "001", "codeName": "코스피"},
                    ]
                )
            return pd.DataFrame(columns=["full_code", "short_code", "codeName"])

    monkeypatch.setattr(krx_indices, "ensure_pykrx_transport_compat", lambda: None)

    import pykrx.website.krx.market.core as core

    monkeypatch.setattr(core, "주가지수검색", _FakeFinder)

    rows = krx_indices._discover_kr_index_rows_via_finder()

    assert rows == [
        {"index_code": "5064", "index_name": "KRX 정보기술", "source_market": "KRX"},
        {"index_code": "5046", "index_name": "KRX 은행", "source_market": "KRX"},
        {"index_code": "1001", "index_name": "코스피", "source_market": "KOSPI"},
    ]


def test_discover_kr_index_rows_prefers_official_names_for_non_benchmark(monkeypatch):
    monkeypatch.setattr(
        krx_indices,
        "_discover_kr_index_rows_via_finder",
        lambda: [
            {"index_code": "1001", "index_name": "코스피", "source_market": "KOSPI"},
            {"index_code": "5046", "index_name": "KRX 은행", "source_market": "KRX"},
        ],
    )
    monkeypatch.setattr(
        krx_indices,
        "_configured_index_metadata",
        lambda: {
            "1001": {
                "index_code": "1001",
                "index_name": "KOSPI",
                "family": "kospi_dd_trd",
                "is_benchmark": True,
                "is_active": True,
                "export_sector": False,
                "taxonomy_kind": "BENCHMARK",
                "taxonomy_label": "KOSPI",
            },
            "5046": {
                "index_code": "5046",
                "index_name": "KRX 미디어통신",
                "family": "krx_dd_trd",
                "is_benchmark": False,
                "is_active": True,
                "export_sector": False,
            },
        },
    )

    rows = {row["index_code"]: row for row in krx_indices.discover_kr_index_rows("1001")}

    assert rows["1001"]["index_name"] == "KOSPI"
    assert rows["5046"]["index_name"] == "KRX 은행"
    assert rows["5046"]["taxonomy_kind"] == "INDEX"
    assert rows["5046"]["taxonomy_label"] == "은행"


def test_discover_kr_index_rows_prefers_krx_source_over_theme_duplicate(monkeypatch):
    monkeypatch.setattr(
        krx_indices,
        "_discover_kr_index_rows_via_finder",
        lambda: [
            {"index_code": "1001", "index_name": "코스피", "source_market": "KOSPI"},
            {"index_code": "5064", "index_name": "KRX 정보기술", "source_market": "테마"},
            {"index_code": "5064", "index_name": "KRX 정보기술", "source_market": "KRX"},
        ],
    )
    monkeypatch.setattr(
        krx_indices,
        "_configured_index_metadata",
        lambda: {
            "1001": {
                "index_code": "1001",
                "index_name": "KOSPI",
                "family": "kospi_dd_trd",
                "is_benchmark": True,
                "is_active": True,
                "export_sector": False,
                "taxonomy_kind": "BENCHMARK",
                "taxonomy_label": "KOSPI",
            }
        },
    )

    rows = {row["index_code"]: row for row in krx_indices.discover_kr_index_rows("1001")}

    assert rows["5064"]["index_name"] == "KRX 정보기술"
    assert rows["5064"]["taxonomy_kind"] == "INDEX"
    assert rows["5064"]["taxonomy_label"] == "정보기술"


def test_repair_stale_kr_index_dimension_names_updates_code_like_rows(monkeypatch):
    stale_rows = pd.DataFrame(
        [
            {
                "index_code": "1002",
                "index_name": "1002",
                "family": "kospi_dd_trd",
                "is_benchmark": False,
                "is_active": True,
                "export_sector": False,
                "taxonomy_kind": "INDEX",
                "taxonomy_label": "1002",
            },
            {
                "index_code": "5044",
                "index_name": "KRX 반도체",
                "family": "krx_dd_trd",
                "is_benchmark": False,
                "is_active": True,
                "export_sector": False,
                "taxonomy_kind": "INDEX",
                "taxonomy_label": "반도체",
            },
        ]
    )
    repaired_rows = pd.DataFrame(
        [
            {
                "index_code": "1002",
                "index_name": "코스피 대형주",
                "family": "kospi_dd_trd",
                "is_benchmark": False,
                "is_active": True,
                "export_sector": False,
                "taxonomy_kind": "INDEX",
                "taxonomy_label": "대형주",
            },
            {
                "index_code": "5044",
                "index_name": "KRX 반도체",
                "family": "krx_dd_trd",
                "is_benchmark": False,
                "is_active": True,
                "export_sector": False,
                "taxonomy_kind": "INDEX",
                "taxonomy_label": "반도체",
            },
        ]
    )
    state = {"rows": stale_rows.copy()}
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(krx_indices, "read_active_index_dimension", lambda market="KR": state["rows"].copy())

    def _upsert(rows, market="KR"):
        captured.extend(rows)
        state["rows"] = repaired_rows.copy()

    monkeypatch.setattr(krx_indices, "upsert_index_dimension", _upsert)
    monkeypatch.setattr(
        krx_indices,
        "_discovered_index_rows_by_code",
        lambda benchmark_code=None: {
            "1002": {
                "index_code": "1002",
                "index_name": "코스피 대형주",
                "family": "kospi_dd_trd",
                "is_benchmark": False,
                "is_active": True,
                "export_sector": False,
                "taxonomy_kind": "INDEX",
                "taxonomy_label": "대형주",
            }
        },
    )

    result = krx_indices.repair_stale_kr_index_dimension_names("1001")

    assert [row["index_code"] for row in captured] == ["1002"]
    assert result.loc[result["index_code"] == "1002", "index_name"].iloc[0] == "코스피 대형주"
