"""Tests for ECOS/KOSIS API error handling and parameter behavior."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import pytest

import src.data_sources.ecos as ecos
import src.data_sources.kosis as kosis
from src.macro.series_utils import to_plotly_time_index


def _valid_macro_df(series_id: str, source: str = "KOSIS") -> pd.DataFrame:
    idx = pd.period_range("2024-01", periods=2, freq="M")
    now = datetime.now(timezone.utc)
    return pd.DataFrame(
        {
            "series_id": [series_id] * 2,
            "value": [1.0, 2.0],
            "source": [source] * 2,
            "fetched_at": [now, now],
            "is_provisional": [False, False],
        },
        index=idx,
    ).astype({"value": "float64", "is_provisional": "bool"})


def test_ecos_error_100_mentions_schema_hint(monkeypatch):
    monkeypatch.setattr(ecos, "_get_api_key", lambda: "DUMMY_KEY")
    monkeypatch.setattr(
        ecos,
        "_get_with_retry",
        lambda url: {"RESULT": {"CODE": "ERROR-100", "MESSAGE": "required value missing"}},
    )

    with pytest.raises(ValueError) as exc:
        ecos.fetch_series("722Y001", "0101000", "202401", "202412")

    message = str(exc.value)
    assert "ERROR-100" in message
    assert "request schema" in message.lower()


def test_ecos_nested_result_is_detected(monkeypatch):
    monkeypatch.setattr(ecos, "_get_api_key", lambda: "DUMMY_KEY")
    monkeypatch.setattr(
        ecos,
        "_get_with_retry",
        lambda url: {"StatisticSearch": {"RESULT": {"CODE": "ERROR-100", "MESSAGE": "nested"}}},
    )

    with pytest.raises(ValueError) as exc:
        ecos.fetch_series("722Y001", "0101000", "202401", "202412")

    assert "ERROR-100" in str(exc.value)


def test_ecos_fetch_series_uses_cycle_path_in_url(monkeypatch):
    monkeypatch.setattr(ecos, "_get_api_key", lambda: "DUMMY_KEY")

    seen: dict[str, str] = {}

    def fake_get(url: str):
        seen["url"] = url
        return {"StatisticSearch": {"row": [{"TIME": "202401", "DATA_VALUE": "1.23"}]}}

    monkeypatch.setattr(ecos, "_get_with_retry", fake_get)

    df = ecos.fetch_series(
        stat_code="722Y001",
        item_code="0101000",
        start_ym="202401",
        end_ym="202412",
        cycle="M",
    )

    assert not df.empty
    assert "/M/" in seen["url"]
    assert "/MM/" not in seen["url"]


def test_ecos_load_partial_success(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    curated = tmp_path / "curated"
    curated.mkdir()
    monkeypatch.setattr(ecos, "CURATED_DIR", curated)

    def fake_fetch(stat_code, item_code, start_ym, end_ym, item_codes=None, cycle="M"):
        if stat_code == "BAD":
            raise ValueError("mock ecos failure")
        return _valid_macro_df(f"{stat_code}/0101000", source="ECOS")

    monkeypatch.setattr(ecos, "fetch_series", fake_fetch)

    status, df = ecos.load_ecos_macro(
        "202401",
        "202412",
        series_config={
            "ok": {
                "stat_code": "722Y001",
                "item_code": "0101000",
                "item_codes": ["0101000"],
                "cycle": "M",
                "enabled": True,
            },
            "bad": {
                "stat_code": "BAD",
                "item_code": "X",
                "item_codes": ["X"],
                "cycle": "M",
                "enabled": True,
            },
        },
    )

    assert status == "LIVE"
    assert not df.empty
    assert "ECOS partial success" in caplog.text


def test_kosis_err21_is_structured_error(monkeypatch):
    monkeypatch.setattr(kosis, "_get_api_key", lambda: "DUMMY_KEY")
    monkeypatch.setattr(
        kosis,
        "_get_with_retry",
        lambda url, params=None: {"err": "21", "errMsg": "invalid request variable"},
    )

    with pytest.raises(ValueError) as exc:
        kosis.fetch_kosis_series("101", "DT_1J22003", "T10", "202401", "202412")

    message = str(exc.value)
    assert "KOSIS API error [21]" in message
    assert "request variable mismatch" in message


@pytest.mark.parametrize(
    ("item_id", "obj_l1"),
    [("T", "T10"), ("T1", "A03")],
)
def test_kosis_validated_item_obj_params_are_forwarded(monkeypatch, item_id, obj_l1):
    monkeypatch.setattr(kosis, "_get_api_key", lambda: "DUMMY_KEY")
    captured: list[dict] = []

    def fake_get(url, params=None):
        captured.append(dict(params))
        return [{"PRD_DE": "202401", "DT": "1.0"}]

    monkeypatch.setattr(kosis, "_get_with_retry", fake_get)

    df = kosis.fetch_kosis_series(
        "101",
        "DT_1J22003",
        item_id,
        "202401",
        "202412",
        obj_params={"objL1": obj_l1},
    )

    assert not df.empty
    assert captured[0]["itmId"] == item_id
    assert captured[0]["objL1"] == obj_l1


def test_kosis_obj_params_are_forwarded(monkeypatch):
    monkeypatch.setattr(kosis, "_get_api_key", lambda: "DUMMY_KEY")

    captured: list[dict] = []

    def fake_get(url, params=None):
        captured.append(dict(params))
        return [{"PRD_DE": "202401", "DT": "1.23"}]

    monkeypatch.setattr(kosis, "_get_with_retry", fake_get)

    df = kosis.fetch_kosis_series(
        "101",
        "DT_1J22003",
        "T10",
        "202401",
        "202412",
        obj_params={"objL1": "A01", "objL2": "B02"},
    )

    assert not df.empty
    assert captured[0]["objL1"] == "A01"
    assert captured[0]["objL2"] == "B02"
    assert "objL3" not in captured[0]


def test_kosis_err21_retries_with_all_obj_params(monkeypatch):
    monkeypatch.setattr(kosis, "_get_api_key", lambda: "DUMMY_KEY")

    calls: list[dict] = []

    def fake_get(url, params=None):
        calls.append(dict(params))
        if "objL1" not in params:
            return {"err": "21", "errMsg": "missing obj params"}
        return [{"PRD_DE": "202401", "DT": "10"}]

    monkeypatch.setattr(kosis, "_get_with_retry", fake_get)

    df = kosis.fetch_kosis_series("101", "DT_1J22003", "T10", "202401", "202412")

    assert not df.empty
    assert len(calls) >= 2
    assert calls[1]["objL1"] == "ALL"


def test_kosis_load_skips_disabled_series(tmp_path, monkeypatch):
    curated = tmp_path / "curated"
    curated.mkdir()
    monkeypatch.setattr(kosis, "CURATED_DIR", curated)

    calls: list[str] = []

    def fake_fetch(org_id, tbl_id, item_id, start_ym, end_ym, obj_params=None):
        calls.append(item_id)
        return _valid_macro_df(f"{org_id}/{tbl_id}/{item_id}")

    monkeypatch.setattr(kosis, "fetch_kosis_series", fake_fetch)

    status, df = kosis.load_kosis_macro(
        "202401",
        "202412",
        series_config={
            "enabled": {
                "org_id": "101",
                "tbl_id": "DT_1J22003",
                "item_id": "T",
                "obj_params": {"objL1": "T10"},
                "enabled": True,
            },
            "disabled": {
                "org_id": "999",
                "tbl_id": "BAD",
                "item_id": "BAD",
                "enabled": False,
            },
        },
    )

    assert status == "LIVE"
    assert not df.empty
    assert calls == ["T"]


def test_ecos_load_skips_disabled_series(tmp_path, monkeypatch):
    curated = tmp_path / "curated"
    curated.mkdir()
    monkeypatch.setattr(ecos, "CURATED_DIR", curated)

    calls: list[str] = []

    def fake_fetch(stat_code, item_code, start_ym, end_ym, item_codes=None, cycle="M"):
        calls.append(stat_code)
        return _valid_macro_df(f"{stat_code}/0101000", source="ECOS")

    monkeypatch.setattr(ecos, "fetch_series", fake_fetch)

    status, df = ecos.load_ecos_macro(
        "202401",
        "202412",
        series_config={
            "enabled": {
                "stat_code": "722Y001",
                "item_code": "0101000",
                "item_codes": ["0101000"],
                "cycle": "M",
                "enabled": True,
            },
            "disabled": {
                "stat_code": "BAD",
                "item_code": "X",
                "item_codes": ["X"],
                "cycle": "M",
                "enabled": False,
            },
        },
    )

    assert status == "LIVE"
    assert not df.empty
    assert calls == ["722Y001"]


def test_to_plotly_time_index_converts_period_index():
    idx = pd.period_range("2024-01", periods=3, freq="M")
    df = pd.DataFrame({"regime": ["Recovery", "Expansion", "Slowdown"]}, index=idx)

    converted = to_plotly_time_index(df)

    assert isinstance(converted.index, pd.DatetimeIndex)
    assert converted.index[0] > pd.Timestamp("2024-01-01")


def test_load_kosis_macro_partial_success(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    curated = tmp_path / "curated"
    curated.mkdir()
    monkeypatch.setattr(kosis, "CURATED_DIR", curated)

    def fake_fetch(org_id, tbl_id, item_id, start_ym, end_ym, obj_params=None):
        if item_id == "BAD":
            raise ValueError("mock failure")
        return _valid_macro_df(f"{org_id}/{tbl_id}/{item_id}")

    monkeypatch.setattr(kosis, "fetch_kosis_series", fake_fetch)

    status, df = kosis.load_kosis_macro(
        "202401",
        "202412",
        series_config={
            "good": {
                "org_id": "101",
                "tbl_id": "DT_1J22003",
                "item_id": "T10",
                "obj_params": {"objL1": "ALL"},
            },
            "bad": {
                "org_id": "999",
                "tbl_id": "BAD_TABLE",
                "item_id": "BAD",
            },
        },
    )

    assert status == "LIVE"
    assert not df.empty
    assert "T10" in "".join(df["series_id"].astype(str).tolist())
    assert "partial success" in caplog.text.lower()
