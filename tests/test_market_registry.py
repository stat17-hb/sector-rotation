from __future__ import annotations

from config.markets import get_market_profile, load_market_configs


def test_market_registry_exposes_us_profile():
    profile = get_market_profile("US")
    assert profile.market_id == "US"
    assert profile.benchmark_code == "SPY"
    assert profile.price_provider == "YFINANCE"
    assert profile.macro_provider == "FRED"
    assert profile.sector_map_path.exists()
    assert profile.macro_series_path.exists()


def test_load_market_configs_merges_us_overrides():
    settings, sector_map, macro_series_cfg, profile = load_market_configs("US")

    assert profile.market_id == "US"
    assert settings["benchmark_code"] == "SPY"
    assert settings["benchmark_label"] == "S&P 500"
    assert settings["fx_series_alias"] == "dxy"
    assert settings["yield_curve_long"] == "treasury_10y"
    assert sector_map["benchmark"]["code"] == "SPY"
    assert "fred" in macro_series_cfg
    assert macro_series_cfg["fred"]["leading_index"]["series_id"] == "USALOLITONOSTSAM"
