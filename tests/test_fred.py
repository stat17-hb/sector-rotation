from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

import src.data_sources.fred as fred
import src.data_sources.macro_sync as macro_sync
from src.data_sources.warehouse import read_macro_data, upsert_macro_dimension, upsert_macro_series_frame


def _make_observations(values: list[tuple[str, str]]) -> dict[str, object]:
    return {"observations": [{"date": date_str, "value": value} for date_str, value in values]}


def test_fetch_fred_series_pct_change_12m(monkeypatch):
    monkeypatch.setattr(fred, "_get_api_key", lambda: "SECRET")
    payload = _make_observations(
        [
            ("2023-01-01", "100"),
            ("2023-02-01", "101"),
            ("2023-03-01", "102"),
            ("2023-04-01", "103"),
            ("2023-05-01", "104"),
            ("2023-06-01", "105"),
            ("2023-07-01", "106"),
            ("2023-08-01", "107"),
            ("2023-09-01", "108"),
            ("2023-10-01", "109"),
            ("2023-11-01", "110"),
            ("2023-12-01", "111"),
            ("2024-01-01", "112"),
        ]
    )
    monkeypatch.setattr(fred, "request_json_with_retry", lambda *args, **kwargs: payload)

    frame = fred.fetch_fred_series("CPIAUCSL", "202301", "202401", transform="pct_change_12m")

    assert frame.index[-1] == pd.Period("2024-01", freq="M")
    assert frame["series_id"].iloc[-1] == "CPIAUCSL"
    assert round(float(frame["value"].iloc[-1]), 2) == 12.00
    assert set(frame["source"].astype(str)) == {"FRED"}


def test_load_fred_macro_persists_market_scoped_rows(monkeypatch):
    now = datetime.now(timezone.utc)

    def _fake_fetch(series_id: str, start_ym: str, end_ym: str, transform: str = "none") -> pd.DataFrame:
        _ = (start_ym, end_ym, transform)
        return pd.DataFrame(
            {
                "series_id": [series_id],
                "value": [1.0],
                "source": ["FRED"],
                "fetched_at": [now],
                "is_provisional": [False],
            },
            index=pd.PeriodIndex(["2024-01"], freq="M"),
        )

    monkeypatch.setattr(fred, "fetch_fred_series", _fake_fetch)

    status, frame = fred.load_fred_macro(
        "202401",
        "202401",
        series_config={
            "leading_index": {"series_id": "USALOLITONOSTSAM", "transform": "none", "enabled": True},
            "cpi_mom": {"series_id": "CPIAUCSL", "transform": "pct_change_1m", "enabled": True},
        },
        market="US",
    )

    assert status == "LIVE"
    assert sorted(frame["series_id"].astype(str).unique()) == ["CPIAUCSL", "USALOLITONOSTSAM"]

    stored = read_macro_data(
        series_aliases=["leading_index", "cpi_mom"],
        start_ym="202401",
        end_ym="202401",
        market="US",
    )
    assert sorted(stored["series_alias"].astype(str).unique()) == ["cpi_mom", "leading_index"]


def test_load_fred_macro_uses_cached_rows_without_write(monkeypatch):
    now = datetime.now(timezone.utc)
    upsert_macro_dimension(
        [
            {
                "series_alias": "leading_index",
                "provider": "FRED",
                "provider_series_id": "USALOLITONOSTSAM",
                "enabled": True,
                "label": "",
                "unit": "",
            }
        ],
        market="US",
    )
    upsert_macro_series_frame(
        series_alias="leading_index",
        provider="FRED",
        provider_series_id="USALOLITONOSTSAM",
        frame=pd.DataFrame(
            {
                "series_id": ["USALOLITONOSTSAM"],
                "value": [1.0],
                "source": ["FRED"],
                "fetched_at": [now],
                "is_provisional": [False],
            },
            index=pd.PeriodIndex(["2024-01"], freq="M"),
        ),
        market="US",
    )

    read_macro_data(
        series_aliases=["leading_index"],
        start_ym="202401",
        end_ym="202401",
        market="US",
    )

    monkeypatch.setattr(
        macro_sync,
        "upsert_macro_dimension",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache hit should not open a write path")),
    )
    monkeypatch.setattr(
        fred,
        "fetch_fred_series",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache hit should not trigger live fetch")),
    )

    cached_status, cached_result = fred.load_fred_macro(
        "202401",
        "202401",
        series_config={
            "leading_index": {"series_id": "USALOLITONOSTSAM", "transform": "none", "enabled": True},
        },
        market="US",
    )

    assert cached_status == "CACHED"
    assert list(cached_result["series_id"].astype(str)) == ["USALOLITONOSTSAM"]


def test_load_fred_macro_persists_market_scoped_rows_after_fetch(monkeypatch):
    now = datetime.now(timezone.utc)
    cached_frame = pd.DataFrame(
        {
            "series_id": ["USALOLITONOSTSAM"],
            "value": [1.0],
            "source": ["FRED"],
            "fetched_at": [now],
            "is_provisional": [False],
        },
        index=pd.PeriodIndex(["2024-01"], freq="M"),
    )
    monkeypatch.setattr(fred, "fetch_fred_series", lambda *args, **kwargs: cached_frame)
    status, frame = fred.load_fred_macro(
        "202401",
        "202401",
        series_config={
            "leading_index": {"series_id": "USALOLITONOSTSAM", "transform": "none", "enabled": True},
        },
        market="US",
    )

    assert status == "LIVE"
    assert list(frame["series_id"].astype(str)) == ["USALOLITONOSTSAM"]
