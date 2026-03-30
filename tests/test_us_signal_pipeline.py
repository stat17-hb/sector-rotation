from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.markets import load_market_configs
from src.dashboard import data as dashboard_data


def _make_long_prices(benchmark_code: str, sector_code: str) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=90, freq="B")
    spy = pd.Series(range(len(dates)), dtype="float64").mul(0.1).add(100.0)
    sector = pd.Series(range(len(dates)), dtype="float64").mul(0.5).add(100.0)
    rows = []
    for trade_date, bench_close, sector_close in zip(dates, spy, sector):
        rows.append({"trade_date": trade_date, "index_code": benchmark_code, "index_name": "S&P 500", "close": float(bench_close)})
        rows.append({"trade_date": trade_date, "index_code": sector_code, "index_name": "Energy", "close": float(sector_close)})
    frame = pd.DataFrame(rows).set_index("trade_date")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame[["index_code", "index_name", "close"]].astype(
        {"index_code": "object", "index_name": "object", "close": "float64"}
    )


def _make_us_macro_frame() -> pd.DataFrame:
    periods = pd.PeriodIndex(["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"], freq="M")
    now = datetime.now(timezone.utc)
    rows = []
    payload = {
        "leading_index": ("USALOLITONOSTSAM", [105.0, 104.0, 103.0, 102.0, 101.0]),
        "cpi_mom": ("CPIAUCSL", [1.0, 1.5, 2.0, 2.5, 3.0]),
        "cpi_yoy": ("CPIAUCSL", [2.0, 2.1, 2.2, 2.3, 2.4]),
        "dxy": ("DTWEXBGS", [100.0, 101.0, 102.0, 103.0, 107.0]),
        "treasury_10y": ("GS10", [4.0, 4.1, 4.2, 4.3, 4.4]),
        "treasury_2y": ("GS2", [3.0, 3.2, 3.4, 3.6, 3.8]),
    }
    for alias, (series_id, values) in payload.items():
        for period, value in zip(periods, values):
            rows.append(
                {
                    "period": period,
                    "series_alias": alias,
                    "series_id": series_id,
                    "value": float(value),
                    "source": "FRED",
                    "fetched_at": now,
                    "is_provisional": False,
                }
            )
    frame = pd.DataFrame(rows).set_index("period")
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame[["series_alias", "series_id", "value", "source", "fetched_at", "is_provisional"]]


def test_cached_signals_supports_us_market(monkeypatch):
    settings, sector_map, macro_series_cfg, market_profile = load_market_configs("US")
    settings = dict(settings)
    settings["use_adaptive_epsilon"] = False
    settings["confirmation_periods"] = 1

    dashboard_data.configure_dashboard_env(
        settings_obj=settings,
        sector_map_obj=sector_map,
        macro_series_cfg_obj=macro_series_cfg,
        market_id_obj="US",
        market_profile_obj=market_profile,
        cache_ttl=60,
        curated_sector_prices_path=Path("data/curated/sector_prices_us.parquet"),
    )
    dashboard_data._cached_signals.clear()

    sector_prices = _make_long_prices("SPY", "XLE")
    macro_df = _make_us_macro_frame()
    monkeypatch.setattr(
        dashboard_data,
        "_cached_sector_prices",
        lambda *args, **kwargs: ("LIVE", sector_prices),
    )
    monkeypatch.setattr(
        dashboard_data,
        "_cached_macro",
        lambda *args, **kwargs: ("LIVE", macro_df),
    )

    signals, macro_result, price_status, macro_status, blocking_error = dashboard_data._cached_signals(
        "US",
        "20240531",
        (0, 0),
        (0, 0),
        "params",
        "macro",
        "price",
        (0, 0),
        0.0,
        20,
        20,
        60,
        3,
    )

    by_code = {signal.index_code: signal for signal in signals}
    assert price_status == "LIVE"
    assert macro_status == "LIVE"
    assert blocking_error == ""
    assert not macro_result.empty
    assert by_code["XLE"].action == "Watch"
    assert "FX Shock" in by_code["XLE"].alerts
