from __future__ import annotations

from datetime import date
import pandas as pd
from pathlib import Path

from src.dashboard import data as dashboard_data
from src.macro import series_utils


def test_resolve_market_end_date_uses_newer_warehouse_benchmark_date(monkeypatch):
    monkeypatch.setattr(dashboard_data, "market_id", "KR")
    monkeypatch.setattr(
        "src.transforms.calendar.get_last_business_day",
        lambda **_kwargs: date(2026, 5, 14),
    )
    monkeypatch.setattr(
        "src.data_sources.warehouse.get_market_latest_dates",
        lambda codes, *, market: {"1001": "20260515"},
    )

    assert dashboard_data._resolve_market_end_date("1001") == date(2026, 5, 15)


def test_resolve_market_end_date_ignores_older_warehouse_benchmark_date(monkeypatch):
    monkeypatch.setattr(dashboard_data, "market_id", "KR")
    monkeypatch.setattr(
        "src.transforms.calendar.get_last_business_day",
        lambda **_kwargs: date(2026, 5, 15),
    )
    monkeypatch.setattr(
        "src.data_sources.warehouse.get_market_latest_dates",
        lambda codes, *, market: {"1001": "20260514"},
    )

    assert dashboard_data._resolve_market_end_date("1001") == date(2026, 5, 15)


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


def test_load_dashboard_runtime_data_emits_progress_and_returns_flow_payload(monkeypatch):
    events: list[dict[str, object]] = []
    idx = pd.to_datetime(["2024-01-31", "2024-02-29"])
    sector_prices = pd.DataFrame(
        {
            "index_code": ["1001", "5044"],
            "index_name": ["KOSPI", "KRX 반도체"],
            "close": [100.0, 110.0],
        },
        index=idx,
    )
    flow_frame = pd.DataFrame(
        {
            "sector_code": ["5044"],
            "sector_name": ["KRX 반도체"],
            "investor_type": ["외국인"],
            "buy_amount": [100],
            "sell_amount": [50],
            "net_buy_amount": [50],
            "net_flow_ratio": [0.2],
        },
        index=pd.to_datetime(["2024-02-29"]),
    )
    macro_df = pd.DataFrame({"value": [1.0]}, index=pd.PeriodIndex(["2024-02"], freq="M"))

    monkeypatch.setattr(
        dashboard_data,
        "_cached_sector_prices",
        lambda *args, **kwargs: ("LIVE", sector_prices),
    )
    monkeypatch.setattr(
        dashboard_data,
        "_cached_investor_flow",
        lambda *args, **kwargs: ("CACHED", True, {"coverage_complete": True}, flow_frame),
    )
    monkeypatch.setattr(
        dashboard_data,
        "_cached_macro",
        lambda *args, **kwargs: ("LIVE", macro_df),
    )
    monkeypatch.setattr(
        dashboard_data,
        "build_regime_history_from_macro",
        lambda **kwargs: pd.DataFrame(
            {
                "growth_dir": ["Up"],
                "inflation_dir": ["Down"],
                "regime": ["Recovery"],
                "confirmed_regime": ["Recovery"],
            },
            index=pd.to_datetime(["2024-02-29"]),
        ),
    )

    import src.signals.matrix as matrix_mod

    monkeypatch.setattr(matrix_mod, "build_signal_table", lambda **kwargs: ["signal-a"])

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

    payload = dashboard_data.load_dashboard_runtime_data(
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
        progress_callback=events.append,
    )

    assert [event["phase"] for event in events] == [
        "준비 중",
        "시장 데이터 로드 완료",
        "수급 데이터 로드 완료",
        "매크로 데이터 로드 완료",
        "신호 계산 완료",
        "표시 데이터 준비 완료",
    ]
    assert events[-1]["pct"] == 100
    assert events[-1]["status"] == "complete"
    assert payload["signals"] == ["signal-a"]
    assert payload["market_data_reference_date"] == "2024-01-31"
    assert payload["investor_flow_status"] == "CACHED"
    assert payload["investor_flow_frame"].equals(flow_frame)
    assert "5044" in payload["shared_flow_summary_map"]
    assert payload["shared_flow_summary_map"]["5044"].flow_profile == "foreign_lead"


def test_cached_signals_merges_cached_theme_proxy_inputs(monkeypatch):
    captured: dict[str, object] = {}
    idx = pd.to_datetime(["2024-02-28", "2024-02-29"])
    sector_prices = pd.DataFrame(
        {
            "index_code": ["1001", "5044"],
            "index_name": ["KOSPI", "KRX 반도체"],
            "close": [100.0, 110.0],
        },
        index=idx,
    )
    theme_prices = pd.DataFrame(
        {
            "index_code": ["445290", "445290"],
            "index_name": ["로봇", "로봇"],
            "close": [100.0, 125.0],
        },
        index=idx,
    )
    theme_rows = [
        {
            "index_code": "445290",
            "index_name": "로봇",
            "family": "theme_lens_etf_proxy",
            "taxonomy_kind": "THEME",
            "taxonomy_label": "로봇",
        }
    ]

    monkeypatch.setattr(dashboard_data, "_cached_sector_prices", lambda *args, **kwargs: ("LIVE", sector_prices))
    monkeypatch.setattr(dashboard_data, "_cached_investor_flow", lambda *args, **kwargs: ("SAMPLE", False, {}, pd.DataFrame()))
    monkeypatch.setattr(dashboard_data, "_cached_macro", lambda *args, **kwargs: ("LIVE", pd.DataFrame()))
    monkeypatch.setattr(dashboard_data, "load_theme_proxy_signal_inputs", lambda **kwargs: ("CACHED", theme_prices, theme_rows))
    monkeypatch.setattr(
        dashboard_data,
        "read_active_index_dimension",
        lambda market="KR": pd.DataFrame(
            [
                {
                    "index_code": "1001",
                    "index_name": "KOSPI",
                    "family": "kospi_dd_trd",
                    "is_benchmark": True,
                    "taxonomy_label": "KOSPI",
                },
                {
                    "index_code": "5044",
                    "index_name": "KRX 반도체",
                    "family": "krx_dd_trd",
                    "is_benchmark": False,
                    "taxonomy_label": "반도체",
                },
            ]
        ),
    )

    import src.signals.matrix as matrix_mod

    monkeypatch.setattr(matrix_mod, "build_signal_table", lambda **kwargs: captured.update(kwargs) or [])

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
        theme_lens_artifact_key=("theme",),
    )

    merged_prices = captured["sector_prices"]
    assert sorted(merged_prices["index_code"].astype(str).unique().tolist()) == ["1001", "445290", "5044"]
    universe_rows = captured["sector_universe_rows"]
    assert [row["index_code"] for row in universe_rows] == ["5044", "445290"]
    assert universe_rows[-1]["taxonomy_kind"] == "THEME"
    assert universe_rows[-1]["taxonomy_label"] == "로봇"


def test_normalize_kr_named_frame_replaces_code_like_names_from_active_dimension(monkeypatch):
    monkeypatch.setattr(
        dashboard_data,
        "read_active_index_dimension",
        lambda market="KR": pd.DataFrame(
            [
                {"index_code": "1002", "index_name": "코스피 대형주"},
                {"index_code": "1013", "index_name": "전기전자"},
            ]
        ),
    )

    frame = pd.DataFrame(
        {
            "sector_code": ["1002", "1013", "5044"],
            "sector_name": ["1002", "1013", "KRX 반도체"],
        }
    )

    normalized = dashboard_data._normalize_kr_named_frame(
        frame,
        code_col="sector_code",
        name_col="sector_name",
    )

    assert list(normalized["sector_name"]) == ["코스피 대형주", "전기전자", "KRX 반도체"]


def test_kr_active_index_name_lookup_falls_back_to_official_discovery_for_placeholder_rows(monkeypatch):
    monkeypatch.setattr(
        dashboard_data,
        "read_active_index_dimension",
        lambda market="KR": pd.DataFrame(
            [
                {"index_code": "1002", "index_name": "1002", "taxonomy_label": "1002"},
                {"index_code": "1013", "index_name": "1013", "taxonomy_label": "1013"},
            ]
        ),
    )
    monkeypatch.setattr(
        dashboard_data,
        "settings",
        {"benchmark_code": "1001"},
    )

    import src.data_sources.krx_indices as krx_indices

    monkeypatch.setattr(
        krx_indices,
        "discover_kr_index_rows",
        lambda benchmark_code=None: [
            {"index_code": "1002", "index_name": "코스피 대형주", "taxonomy_label": "대형주"},
            {"index_code": "1013", "index_name": "전기전자", "taxonomy_label": "전기전자"},
        ],
    )

    lookup = dashboard_data._kr_active_index_name_lookup()

    assert lookup["1002"] == "코스피 대형주"
    assert lookup["1013"] == "전기전자"


def test_kr_active_index_name_lookup_logs_official_discovery_failure_once(monkeypatch, caplog):
    dashboard_data._KR_OFFICIAL_NAME_LOOKUP_WARNING_KEYS.clear()
    monkeypatch.setattr(
        dashboard_data,
        "read_active_index_dimension",
        lambda market="KR": pd.DataFrame(
            [
                {"index_code": "1002", "index_name": "1002", "taxonomy_label": "1002"},
            ]
        ),
    )
    monkeypatch.setattr(dashboard_data, "settings", {"benchmark_code": "1001"})

    import src.data_sources.krx_indices as krx_indices

    monkeypatch.setattr(
        krx_indices,
        "discover_kr_index_rows",
        lambda benchmark_code=None: (_ for _ in ()).throw(ValueError("KRX official index finder failed")),
    )
    caplog.set_level("WARNING", logger="src.dashboard.data")

    first = dashboard_data._kr_active_index_name_lookup()
    second = dashboard_data._kr_active_index_name_lookup()

    assert first == {}
    assert second == {}
    warnings = [
        rec
        for rec in caplog.records
        if "KR official name lookup fallback failed" in rec.getMessage()
    ]
    assert len(warnings) == 1


def test_normalize_kr_sector_universe_rows_replaces_placeholder_taxonomy_via_official_fallback(monkeypatch):
    monkeypatch.setattr(
        dashboard_data,
        "read_active_index_dimension",
        lambda market="KR": pd.DataFrame(
            [
                {"index_code": "1002", "index_name": "1002", "taxonomy_label": "1002"},
            ]
        ),
    )
    monkeypatch.setattr(
        dashboard_data,
        "settings",
        {"benchmark_code": "1001"},
    )

    import src.data_sources.krx_indices as krx_indices

    monkeypatch.setattr(
        krx_indices,
        "discover_kr_index_rows",
        lambda benchmark_code=None: [
            {"index_code": "1002", "index_name": "코스피 대형주", "taxonomy_label": "대형주"},
        ],
    )

    normalized = dashboard_data._normalize_kr_sector_universe_rows(
        [{"index_code": "1002", "index_name": "1002", "taxonomy_label": "1002"}]
    )

    assert normalized == [
        {"index_code": "1002", "index_name": "코스피 대형주", "taxonomy_label": "대형주"}
    ]


def test_canonicalize_kr_sector_universe_rows_keeps_only_plain_krx_sector_family():
    rows = [
        {"index_code": "1001", "index_name": "코스피", "family": "kospi_dd_trd", "taxonomy_label": "코스피"},
        {"index_code": "5042", "index_name": "KRX 100", "family": "krx_dd_trd", "taxonomy_label": "KRX 100"},
        {"index_code": "5044", "index_name": "KRX 반도체", "family": "krx_dd_trd", "taxonomy_label": "반도체"},
        {"index_code": "5064", "index_name": "KRX 정보기술", "family": "krx_dd_trd", "taxonomy_label": "정보기술"},
        {"index_code": "5351", "index_name": "KRX 300 정보기술", "family": "krx_dd_trd", "taxonomy_label": "KRX 300 정보기술"},
        {"index_code": "1155", "index_name": "코스피 200 정보기술", "family": "kospi_dd_trd", "taxonomy_label": "정보기술"},
        {"index_code": "2216", "index_name": "코스닥 150 정보기술", "family": "kosdaq_dd_trd", "taxonomy_label": "150 정보기술"},
    ]

    canonical = dashboard_data._canonicalize_kr_sector_universe_rows(
        rows,
        benchmark_code="1001",
        include_benchmark=True,
    )

    assert [row["index_code"] for row in canonical] == ["1001", "5044", "5064"]


def test_get_market_index_universe_codes_returns_canonical_kr_family_plus_benchmark(monkeypatch):
    import src.data_sources.krx_indices as krx_indices

    monkeypatch.setattr(
        krx_indices,
        "get_active_kr_index_universe_codes",
        lambda benchmark_code: ["1001", "5044", "5064"],
    )

    codes = dashboard_data.get_market_index_universe_codes("1001", "KR")

    assert codes == ["1001", "5044", "5064"]


def test_get_market_index_universe_codes_includes_us_reference_indexes(monkeypatch):
    monkeypatch.setattr(
        dashboard_data,
        "sector_map",
        {
            "regimes": {
                "Recovery": {
                    "sectors": [
                        {"code": "XLK", "name": "Technology"},
                        {"code": "XLE", "name": "Energy"},
                    ]
                }
            }
        },
    )
    monkeypatch.setattr(
        dashboard_data,
        "settings",
        {"reference_index_codes": ["^IXIC"]},
    )

    codes = dashboard_data.get_market_index_universe_codes("^GSPC", "US")

    assert codes == ["XLK", "XLE", "^GSPC", "^IXIC"]


def test_cached_sector_prices_prefers_transient_market_preview(monkeypatch):
    dashboard_data.cached_sector_prices.clear()
    frame = pd.DataFrame(
        {
            "index_code": ["SPY"],
            "index_name": ["S&P 500"],
            "close": [500.0],
        },
        index=pd.to_datetime(["2026-04-07"]),
    )
    monkeypatch.setattr(dashboard_data, "get_market_index_universe_codes", lambda benchmark_code, market_id_arg: ["SPY"])
    monkeypatch.setattr(
        dashboard_data,
        "_load_market_price_transient_override",
        lambda **kwargs: ("LIVE", frame),
    )
    monkeypatch.setattr(
        "src.data_sources.yfinance_sectors.load_sector_prices",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("warehouse path should not run")),
    )

    status, loaded = dashboard_data.cached_sector_prices(
        "US",
        "20260407",
        "^GSPC",
        3,
        "price-token",
        ("artifact",),
    )

    assert status == "LIVE"
    assert loaded.equals(frame)
