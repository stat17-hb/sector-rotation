from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.dashboard import data as dashboard_data
from src.macro import series_utils


def test_build_regime_inflation_series_kr_uses_homogeneous_yoy_backfill(monkeypatch):
    direct_yoy = pd.Series(
        [2.1, 2.3],
        index=pd.PeriodIndex(["2024-03", "2024-04"], freq="M"),
        dtype="float64",
    )
    direct_mom = pd.Series(
        [0.3, 0.4],
        index=direct_yoy.index,
        dtype="float64",
    )
    legacy_index = pd.Series(
        [100.0 + idx for idx in range(28)],
        index=pd.period_range("2022-01", periods=28, freq="M"),
        dtype="float64",
    )

    def _fake_extract(*, alias: str, **kwargs) -> pd.Series:
        if alias == "cpi_yoy":
            return direct_yoy
        if alias == "cpi_mom":
            return direct_mom
        if alias == "cpi_index_legacy":
            return legacy_index
        return pd.Series(dtype="float64")

    monkeypatch.setattr(series_utils, "extract_macro_series", _fake_extract)

    result = dashboard_data._build_regime_inflation_series(
        macro_df=pd.DataFrame(),
        macro_series_cfg_obj={},
        market_id_arg="KR",
    )

    assert result.index.min() < direct_yoy.index.min()
    assert result.loc[direct_yoy.index].equals(direct_yoy)
    assert not result.loc[direct_yoy.index].equals(direct_mom)


def test_build_regime_inflation_series_kr_falls_back_to_mom_when_yoy_missing(monkeypatch):
    direct_mom = pd.Series(
        [0.1, 0.2, 0.3],
        index=pd.PeriodIndex(["2024-01", "2024-02", "2024-03"], freq="M"),
        dtype="float64",
    )

    def _fake_extract(*, alias: str, **kwargs) -> pd.Series:
        if alias == "cpi_mom":
            return direct_mom
        return pd.Series(dtype="float64")

    monkeypatch.setattr(series_utils, "extract_macro_series", _fake_extract)

    result = dashboard_data._build_regime_inflation_series(
        macro_df=pd.DataFrame(),
        macro_series_cfg_obj={},
        market_id_arg="KR",
    )

    assert result.equals(direct_mom)


def test_build_regime_inflation_series_us_keeps_mom_priority(monkeypatch):
    direct_mom = pd.Series(
        [0.2, 0.4],
        index=pd.PeriodIndex(["2024-03", "2024-04"], freq="M"),
        dtype="float64",
    )
    direct_yoy = pd.Series(
        [2.5, 2.7],
        index=direct_mom.index,
        dtype="float64",
    )

    def _fake_extract(*, alias: str, **kwargs) -> pd.Series:
        if alias == "cpi_mom":
            return direct_mom
        if alias == "cpi_yoy":
            return direct_yoy
        return pd.Series(dtype="float64")

    monkeypatch.setattr(series_utils, "extract_macro_series", _fake_extract)

    result = dashboard_data._build_regime_inflation_series(
        macro_df=pd.DataFrame(),
        macro_series_cfg_obj={},
        market_id_arg="US",
    )

    assert result.equals(direct_mom)


def test_cached_signals_uses_shared_regime_builder_with_dashboard_window(monkeypatch):
    captured: dict[str, object] = {}
    idx = pd.to_datetime(["2024-01-31", "2024-02-29"])
    sector_prices = pd.DataFrame(
        {
            "index_code": ["1001", "5044"],
            "index_name": ["KOSPI", "KRX 반도체"],
            "close": [100.0, 110.0],
        },
        index=idx,
    )
    macro_df = pd.DataFrame({"value": [1.0]}, index=pd.PeriodIndex(["2024-01"], freq="M"))

    def _fake_builder(*, macro_df, macro_series_cfg, settings, market_id, include_provisional, window_months):
        captured["market_id"] = market_id
        captured["include_provisional"] = include_provisional
        captured["window_months"] = window_months
        captured["confirmation_periods"] = settings["confirmation_periods"]
        return pd.DataFrame(
            {
                "growth_dir": ["Up"],
                "inflation_dir": ["Down"],
                "regime": ["Recovery"],
                "confirmed_regime": ["Recovery"],
            },
            index=pd.to_datetime(["2024-01-31"]),
        )

    monkeypatch.setattr(dashboard_data, "build_regime_history_from_macro", _fake_builder)
    monkeypatch.setattr(dashboard_data, "_cached_sector_prices", lambda *args, **kwargs: ("LIVE", sector_prices))
    monkeypatch.setattr(dashboard_data, "_cached_investor_flow", lambda *args, **kwargs: ("SAMPLE", False, {}, pd.DataFrame()))
    monkeypatch.setattr(dashboard_data, "_cached_macro", lambda *args, **kwargs: ("LIVE", macro_df))

    import src.signals.matrix as matrix_mod

    monkeypatch.setattr(matrix_mod, "build_signal_table", lambda **kwargs: [])

    dashboard_data.configure_dashboard_env(
        settings_obj={
            "benchmark_code": "1001",
            "epsilon": 0.0,
            "confirmation_periods": 2,
            "use_adaptive_epsilon": False,
            "epsilon_factor": 0.5,
            "yield_curve_spread_threshold": 0.0,
            "price_years": 3,
        },
        sector_map_obj={"regimes": {}},
        macro_series_cfg_obj={},
        market_id_obj="KR",
        market_profile_obj=None,
        cache_ttl=1,
        curated_sector_prices_path=Path("data/curated/sector_prices.parquet"),
    )
    dashboard_data._cached_signals.clear()

    dashboard_data._cached_signals(
        "KR",
        "20240229",
        (0, 0),
        (0, 0),
        "params",
        "macro-token",
        "price-token",
        (0, 0),
        (),
        epsilon=0.0,
        rs_ma_period=20,
        ma_fast=20,
        ma_slow=60,
        price_years=3,
        flow_profile="foreign_lead",
    )

    assert captured == {
        "market_id": "KR",
        "include_provisional": True,
        "window_months": 60,
        "confirmation_periods": 2,
    }
