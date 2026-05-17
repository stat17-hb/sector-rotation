from __future__ import annotations

from datetime import date

import pandas as pd
from pathlib import Path

from config.markets import load_market_configs

from src.dashboard import data as dashboard_data
from src.dashboard import runtime
from src.dashboard.types import DashboardContext


def test_app_sidebar_uses_auto_initial_state_for_responsive_first_viewport():
    app_source = Path("app.py").read_text(encoding="utf-8")

    assert 'initial_sidebar_state="auto"' in app_source
    assert 'initial_sidebar_state="expanded"' not in app_source


def test_app_header_displays_dashboard_data_reference_date():
    app_source = Path("app.py").read_text(encoding="utf-8")

    assert "market_data_reference_date" in app_source
    assert '"조회 기준일"' in app_source
    assert '"목표일"' in app_source
    assert "선택 기준일" in app_source


class _DummyCache:
    def __init__(self) -> None:
        self.clears = 0

    def clear(self) -> None:
        self.clears += 1


def _kr_sector_prices() -> pd.DataFrame:
    dates = pd.to_datetime(["2026-04-07", "2026-04-08"])
    return pd.DataFrame(
        {
            "index_code": ["1001", "5044"],
            "index_name": ["KOSPI", "KRX 반도체"],
            "close": [100.0, 101.0],
        },
        index=dates[:2],
    )


def _kr_flow_frame(day: str = "2026-04-08") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sector_code": ["5044"],
            "sector_name": ["KRX 반도체"],
            "investor_type": ["외국인"],
            "buy_amount": [100],
            "sell_amount": [50],
            "net_buy_amount": [50],
            "net_flow_ratio": [0.2],
        },
        index=pd.to_datetime([day]),
    )


def _run_kr_cached_signals(monkeypatch, investor_flow_result: tuple[object, object, dict[str, object], pd.DataFrame]) -> tuple[dict[str, object], pd.DataFrame]:
    settings, sector_map, macro_series_cfg, market_profile = load_market_configs("KR")
    dashboard_data.configure_dashboard_env(
        settings_obj=settings,
        sector_map_obj=sector_map,
        macro_series_cfg_obj=macro_series_cfg,
        market_id_obj="KR",
        market_profile_obj=market_profile,
        cache_ttl=60,
        curated_sector_prices_path=Path("data/curated/sector_prices.parquet"),
    )
    dashboard_data._cached_signals.clear()

    sector_prices = _kr_sector_prices()
    flow_frame = investor_flow_result[3]
    captured: dict[str, object] = {}

    monkeypatch.setattr(dashboard_data, "_cached_sector_prices", lambda *args, **kwargs: ("LIVE", sector_prices))
    monkeypatch.setattr(dashboard_data, "_cached_macro", lambda *args, **kwargs: ("SAMPLE", pd.DataFrame()))
    monkeypatch.setattr(dashboard_data, "_cached_investor_flow", lambda *args, **kwargs: investor_flow_result)
    monkeypatch.setattr(
        "src.signals.matrix.build_signal_table",
        lambda **kwargs: captured.update(kwargs) or [],
    )

    dashboard_data._cached_signals(
        "KR",
        "20260408",
        (0, 0),
        (0, 0),
        "params",
        "macro",
        "price",
        ("flow",),
        0.0,
        20,
        20,
        60,
        3,
    )

    return captured, flow_frame


def _context(market_id: str = "KR") -> DashboardContext:
    return DashboardContext(
        market_id=market_id,
        market_profile=None,
        settings={},
        sector_map={},
        macro_series_cfg={},
        benchmark_code="1001" if market_id == "KR" else "^GSPC",
        market_end_date=date(2026, 4, 7),
        market_end_date_str="20260407",
        macro_cache_token="macro",
        price_cache_token="price",
        price_artifact_key=("price",),
        macro_artifact_key=("macro",),
        provider_configured="AUTO",
        provider_effective="PYKRX",
        openapi_key_present=False,
        theme_mode="dark",
        analysis_heatmap_palette="classic",
        ui_locale="ko",
    )


def test_invalidate_dashboard_caches_scopes(monkeypatch):
    api = _DummyCache()
    sector = _DummyCache()
    analysis = _DummyCache()
    flow = _DummyCache()
    macro = _DummyCache()
    signals = _DummyCache()
    monitoring_clears: list[str] = []

    monkeypatch.setattr(runtime, "cached_api_preflight", api)
    monkeypatch.setattr(runtime, "cached_sector_prices", sector)
    monkeypatch.setattr(runtime, "cached_analysis_sector_prices", analysis)
    monkeypatch.setattr(runtime, "cached_investor_flow", flow)
    monkeypatch.setattr(runtime, "cached_macro", macro)
    monkeypatch.setattr(runtime, "cached_signals", signals)
    monkeypatch.setattr(runtime, "_clear_monitoring_data_cache", lambda: monitoring_clears.append("clear"))

    runtime.invalidate_dashboard_caches("market")
    assert api.clears == 1
    assert sector.clears == 1
    assert analysis.clears == 1
    assert flow.clears == 1
    assert macro.clears == 0
    assert signals.clears == 1
    assert monitoring_clears == ["clear"]

    runtime.invalidate_dashboard_caches("macro")
    assert macro.clears == 1
    assert signals.clears == 2
    assert monitoring_clears == ["clear", "clear"]

    runtime.invalidate_dashboard_caches("flow")
    assert flow.clears == 2
    assert signals.clears == 3
    assert monitoring_clears == ["clear", "clear", "clear"]

    runtime.invalidate_dashboard_caches("all")
    assert api.clears == 2
    assert sector.clears == 2
    assert analysis.clears == 2
    assert flow.clears == 3
    assert macro.clears == 2
    assert signals.clears == 4
    assert monitoring_clears == ["clear", "clear", "clear", "clear"]


def test_run_market_refresh_returns_notice_and_invalidates(monkeypatch):
    close_calls: list[str] = []
    invalidate_calls: list[str] = []
    events: list[dict[str, object]] = []

    monkeypatch.setattr(runtime, "close_cached_read_only_connection", lambda: close_calls.append("close"))
    monkeypatch.setattr(runtime, "get_all_sector_codes", lambda benchmark_code: [benchmark_code, "5044"])
    monkeypatch.setattr(runtime, "get_market_range_strings", lambda end_date_str, price_years: ("20230101", end_date_str))
    monkeypatch.setattr(runtime, "invalidate_dashboard_caches", lambda scope: invalidate_calls.append(scope))
    monkeypatch.setattr(runtime, "build_market_refresh_notice", lambda summary: ("success", summary["status"]))
    monkeypatch.setattr(
        runtime,
        "_resolve_market_refresh_runner",
        lambda market_id: lambda codes, start, end: ((start, end), {"status": f"{market_id}:{','.join(codes)}"}),
    )

    notice = runtime.run_market_refresh(_context("US"), price_years=3, progress_callback=events.append)

    assert notice == ("success", "US:^GSPC,5044")
    assert close_calls == ["close"]
    assert invalidate_calls == ["market"]
    assert [event["phase"] for event in events] == ["준비 중", "데이터 요청 중", "캐시 정리 중", "완료"]
    assert events[-1]["pct"] == 100
    assert events[-1]["status"] == "complete"


def test_run_market_refresh_uses_incremental_start_from_warehouse_latest_dates(monkeypatch):
    calls: list[tuple[list[str], str, str]] = []

    monkeypatch.setattr(runtime, "close_cached_read_only_connection", lambda: None)
    monkeypatch.setattr(runtime, "get_market_index_universe_codes", lambda benchmark_code, market_id: ["1001", "5044"])
    monkeypatch.setattr(runtime, "get_market_range_strings", lambda end_date_str, price_years: ("20230101", end_date_str))
    monkeypatch.setattr(runtime, "get_market_latest_dates", lambda codes, *, market: {"1001": "20260405", "5044": "20260404"})
    monkeypatch.setattr(runtime, "invalidate_dashboard_caches", lambda scope: None)
    monkeypatch.setattr(runtime, "build_market_refresh_notice", lambda summary: ("success", summary["status"]))
    monkeypatch.setattr(
        runtime,
        "_resolve_market_refresh_runner",
        lambda market_id: lambda codes, start, end: (
            calls.append((list(codes), start, end))
            or (("LIVE", pd.DataFrame()), {"status": "LIVE", "coverage_complete": True})
        ),
    )

    notice = runtime.run_market_refresh(_context("KR"), price_years=3)

    assert notice == ("success", "LIVE")
    assert calls == [(["1001", "5044"], "20260405", "20260407")]


def test_run_market_refresh_skips_runner_when_warehouse_is_current(monkeypatch):
    invalidate_calls: list[str] = []
    transient_clears: list[str] = []
    close_calls: list[str] = []
    recorded_runs: list[dict[str, object]] = []

    monkeypatch.setattr(runtime, "get_market_index_universe_codes", lambda benchmark_code, market_id: ["1001", "5044"])
    monkeypatch.setattr(runtime, "get_market_range_strings", lambda end_date_str, price_years: ("20230101", end_date_str))
    monkeypatch.setattr(runtime, "get_market_latest_dates", lambda codes, *, market: {"1001": "20260407", "5044": "20260407"})
    monkeypatch.setattr(runtime, "close_cached_read_only_connection", lambda: close_calls.append("close"))
    monkeypatch.setattr(runtime, "read_dataset_data_bounds", lambda dataset, *, market: {"row_count": 1234})
    monkeypatch.setattr(runtime, "invalidate_dashboard_caches", lambda scope: invalidate_calls.append(scope))
    monkeypatch.setattr(runtime, "clear_market_price_transient_override", lambda: transient_clears.append("clear"))
    monkeypatch.setattr(runtime, "record_ingest_run", lambda **kwargs: recorded_runs.append(kwargs))
    monkeypatch.setattr(runtime, "build_market_refresh_notice", lambda summary: ("info", summary["status"]))
    monkeypatch.setattr(
        runtime,
        "_resolve_market_refresh_runner",
        lambda market_id: lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("runner should not run when warehouse is current")),
    )

    notice = runtime.run_market_refresh(_context("KR"), price_years=3)

    assert notice == ("info", "CACHED")
    assert close_calls == ["close"]
    assert invalidate_calls == ["market"]
    assert transient_clears == ["clear"]
    assert recorded_runs[-1]["dataset"] == "market_prices"
    assert recorded_runs[-1]["reason"] == "manual_refresh"
    assert recorded_runs[-1]["requested_start"] == "20260407"
    assert recorded_runs[-1]["requested_end"] == "20260407"
    assert recorded_runs[-1]["summary"]["next_start"] == "20260408"
    assert recorded_runs[-1]["status"] == "CACHED"
    assert recorded_runs[-1]["row_count"] == 1234


def test_run_market_refresh_sets_transient_override_for_write_lock_preview(monkeypatch):
    close_calls: list[str] = []
    invalidate_calls: list[str] = []
    transient_sets: list[dict[str, object]] = []
    transient_clears: list[str] = []
    frame = pd.DataFrame(
        {
            "index_code": ["SPY"],
            "index_name": ["S&P 500"],
            "close": [500.0],
        },
        index=pd.to_datetime(["2026-04-07"]),
    )

    monkeypatch.setattr(runtime, "close_cached_read_only_connection", lambda: close_calls.append("close"))
    monkeypatch.setattr(runtime, "get_all_sector_codes", lambda benchmark_code: [benchmark_code])
    monkeypatch.setattr(runtime, "get_market_range_strings", lambda end_date_str, price_years: ("20260401", end_date_str))
    monkeypatch.setattr(runtime, "invalidate_dashboard_caches", lambda scope: invalidate_calls.append(scope))
    monkeypatch.setattr(runtime, "set_market_price_transient_override", lambda **kwargs: transient_sets.append(kwargs))
    monkeypatch.setattr(runtime, "clear_market_price_transient_override", lambda: transient_clears.append("clear"))
    monkeypatch.setattr(
        runtime,
        "build_market_refresh_notice",
        lambda summary: ("warning", "preview") if summary["warehouse_write_skipped"] else ("success", "saved"),
    )
    monkeypatch.setattr(
        runtime,
        "_resolve_market_refresh_runner",
        lambda market_id: lambda codes, start, end: (
            ("LIVE", frame),
            {
                "status": "LIVE",
                "coverage_complete": False,
                "warehouse_write_skipped": True,
                "rows": len(frame),
            },
        ),
    )

    notice = runtime.run_market_refresh(_context("US"), price_years=3)

    assert notice == ("warning", "preview")
    assert close_calls == ["close"]
    assert invalidate_calls == ["market"]
    assert transient_clears == []
    assert len(transient_sets) == 1
    assert transient_sets[0]["market_id_arg"] == "US"
    assert transient_sets[0]["requested_end"] == "20260407"
    assert transient_sets[0]["status"] == "LIVE"
    assert transient_sets[0]["frame"].equals(frame)


def test_run_macro_refresh_returns_notice_and_invalidates(monkeypatch):
    close_calls: list[str] = []
    invalidate_calls: list[str] = []
    events: list[dict[str, object]] = []

    monkeypatch.setattr(runtime, "close_cached_read_only_connection", lambda: close_calls.append("close"))
    monkeypatch.setattr(runtime, "invalidate_dashboard_caches", lambda scope: invalidate_calls.append(scope))
    monkeypatch.setattr(runtime, "build_macro_refresh_notice", lambda summary: ("info", summary["status"]))
    monkeypatch.setattr(runtime, "_sync_macro_warehouse", lambda **kwargs: (None, None, {"status": kwargs["market"]}))

    notice = runtime.run_macro_refresh(_context("KR"), {"series": []}, progress_callback=events.append)

    assert notice == ("info", "KR")
    assert close_calls == ["close"]
    assert invalidate_calls == ["macro"]
    assert [event["phase"] for event in events] == ["준비 중", "공급자 동기화 중", "캐시 정리 중", "완료"]
    assert events[-1]["status"] == "complete"


def test_run_macro_refresh_keeps_configured_window_for_macro_sync_policy(monkeypatch):
    calls: list[dict[str, object]] = []
    cfg = {
        "ecos": {"leading_index": {"enabled": True}},
        "kosis": {"cpi_yoy": {"enabled": True}},
    }

    monkeypatch.setattr(runtime, "close_cached_read_only_connection", lambda: None)
    monkeypatch.setattr(runtime, "invalidate_dashboard_caches", lambda scope: None)
    monkeypatch.setattr(runtime, "build_macro_refresh_notice", lambda summary: ("success", summary["status"]))
    monkeypatch.setattr(
        runtime,
        "_sync_macro_warehouse",
        lambda **kwargs: calls.append(dict(kwargs)) or (None, pd.DataFrame({"value": [1]}), {"status": "LIVE", "rows": 1}),
    )

    notice = runtime.run_macro_refresh(_context("KR"), cfg)

    assert notice == ("success", "LIVE")
    assert calls
    assert calls[0]["start_ym"] == "201605"
    assert calls[0]["end_ym"] == "202604"
    assert calls[0]["market"] == "KR"


def test_run_macro_refresh_does_not_bypass_provider_series_policy(monkeypatch):
    calls: list[dict[str, object]] = []
    cfg = {"ecos": {"leading_index": {"enabled": True}}}

    monkeypatch.setattr(runtime, "close_cached_read_only_connection", lambda: None)
    monkeypatch.setattr(runtime, "invalidate_dashboard_caches", lambda scope: None)
    monkeypatch.setattr(runtime, "build_macro_refresh_notice", lambda summary: ("info", summary["status"]))
    monkeypatch.setattr(
        runtime,
        "_sync_macro_warehouse",
        lambda **kwargs: calls.append(dict(kwargs)) or (None, pd.DataFrame(), {"status": "CACHED", "coverage_complete": True}),
    )

    notice = runtime.run_macro_refresh(_context("KR"), cfg)

    assert notice == ("info", "CACHED")
    assert calls
    assert calls[0]["start_ym"] == "201605"


def test_run_investor_flow_refresh_returns_notice_and_invalidates(monkeypatch):
    close_calls: list[str] = []
    invalidate_calls: list[str] = []
    transient_sets: list[dict[str, object]] = []
    transient_clears: list[str] = []
    events: list[dict[str, object]] = []

    monkeypatch.setattr(runtime, "close_cached_read_only_connection", lambda: close_calls.append("close"))
    monkeypatch.setattr(runtime, "invalidate_dashboard_caches", lambda scope: invalidate_calls.append(scope))
    monkeypatch.setattr(runtime, "build_investor_flow_refresh_notice", lambda summary: ("warning", summary["status"]))
    monkeypatch.setattr(runtime, "set_investor_flow_transient_override", lambda **kwargs: transient_sets.append(kwargs))
    monkeypatch.setattr(runtime, "clear_investor_flow_transient_override", lambda: transient_clears.append("clear"))

    def _runner(**kwargs):
        assert kwargs["market"] == "KR"
        assert kwargs["end_date_str"] == "20260407"
        assert "start_date_str" not in kwargs
        return (("LIVE", pd.DataFrame()), {"status": "LIVE", "coverage_complete": True})

    monkeypatch.setattr(
        "src.data_sources.krx_investor_flow.run_manual_investor_flow_refresh",
        _runner,
    )

    notice = runtime.run_investor_flow_refresh(_context("KR"), progress_callback=events.append)

    assert notice == ("warning", "LIVE")
    assert close_calls == ["close"]
    assert invalidate_calls == ["flow"]
    assert transient_sets == []
    assert transient_clears == ["clear"]
    assert [event["phase"] for event in events] == ["준비 중", "완료"]
    assert events[-1]["status"] == "complete"


def test_run_investor_flow_refresh_does_not_append_outer_terminal_progress_after_inner_updates(monkeypatch):
    events: list[dict[str, object]] = []

    monkeypatch.setattr(runtime, "close_cached_read_only_connection", lambda: None)
    monkeypatch.setattr(runtime, "invalidate_dashboard_caches", lambda scope: None)
    monkeypatch.setattr(runtime, "build_investor_flow_refresh_notice", lambda summary: ("warning", summary["status"]))
    monkeypatch.setattr(runtime, "set_investor_flow_transient_override", lambda **kwargs: None)
    monkeypatch.setattr(runtime, "clear_investor_flow_transient_override", lambda: None)

    def _runner(**kwargs):
        kwargs["progress_callback"](
            {
                "task": "투자자수급 갱신",
                "phase": "1차 수집 중",
                "pct": 60,
                "detail": "180/300 logical steps",
                "status": "running",
            }
        )
        kwargs["progress_callback"](
            {
                "task": "투자자수급 갱신",
                "phase": "캐시 재로드 완료",
                "pct": 100,
                "detail": "120 rows",
                "status": "complete",
            }
        )
        return (("LIVE", pd.DataFrame()), {"status": "LIVE", "coverage_complete": True})

    monkeypatch.setattr(
        "src.data_sources.krx_investor_flow.run_manual_investor_flow_refresh",
        _runner,
    )

    notice = runtime.run_investor_flow_refresh(_context("KR"), progress_callback=events.append)

    assert notice == ("warning", "LIVE")
    assert [(event["phase"], event["pct"]) for event in events] == [
        ("준비 중", 5),
        ("1차 수집 중", 60),
        ("캐시 재로드 완료", 100),
    ]


def test_run_investor_flow_refresh_sets_transient_override_for_warehouse_lock_preview(monkeypatch):
    close_calls: list[str] = []
    invalidate_calls: list[str] = []
    transient_sets: list[dict[str, object]] = []
    transient_clears: list[str] = []
    frame = pd.DataFrame(
        {
            "sector_code": ["5044"],
            "sector_name": ["KRX 반도체"],
            "investor_type": ["외국인"],
            "buy_amount": [100],
            "sell_amount": [50],
            "net_buy_amount": [50],
            "net_flow_ratio": [0.2],
        },
        index=pd.to_datetime(["2026-04-07"]),
    )

    monkeypatch.setattr(runtime, "close_cached_read_only_connection", lambda: close_calls.append("close"))
    monkeypatch.setattr(runtime, "invalidate_dashboard_caches", lambda scope: invalidate_calls.append(scope))
    monkeypatch.setattr(runtime, "build_investor_flow_refresh_notice", lambda summary: ("warning", summary["status"]))
    monkeypatch.setattr(runtime, "set_investor_flow_transient_override", lambda **kwargs: transient_sets.append(kwargs))
    monkeypatch.setattr(runtime, "clear_investor_flow_transient_override", lambda: transient_clears.append("clear"))
    monkeypatch.setattr(
        "src.data_sources.krx_investor_flow.run_manual_investor_flow_refresh",
        lambda **kwargs: (
            ("LIVE", frame),
            {"status": "LIVE", "coverage_complete": True, "warehouse_write_skipped": True},
        ),
    )

    notice = runtime.run_investor_flow_refresh(_context("KR"))

    assert notice == ("warning", "LIVE")
    assert close_calls == ["close"]
    assert invalidate_calls == ["flow"]
    assert transient_clears == []
    assert len(transient_sets) == 1
    assert transient_sets[0]["market_id_arg"] == "KR"
    assert transient_sets[0]["requested_end"] == "20260407"
    assert transient_sets[0]["status"] == "LIVE"
    assert transient_sets[0]["frame"].equals(frame)


def test_build_investor_flow_refresh_notice_returns_info_when_already_current():
    notice = dashboard_data.build_investor_flow_refresh_notice(
        {
            "status": "CACHED",
            "coverage_complete": True,
            "rows": 0,
            "failed_codes": {},
        }
    )

    assert notice[0] == "info"


def test_build_investor_flow_refresh_notice_surfaces_auth_required():
    notice = dashboard_data.build_investor_flow_refresh_notice(
        {
            "status": "CACHED",
            "coverage_complete": False,
            "rows": 0,
            "failed_codes": {
                "refresh": "AUTH_REQUIRED: KRX Data Marketplace login is required",
            },
        }
    )

    assert notice[0] == "error"
    assert "KRX_ID / KRX_PW" in notice[1]


def test_build_investor_flow_refresh_notice_surfaces_access_denied():
    notice = dashboard_data.build_investor_flow_refresh_notice(
        {
            "status": "CACHED",
            "coverage_complete": False,
            "rows": 0,
            "failed_codes": {
                "refresh": "ACCESS_DENIED: KRX Data Marketplace blocked the investor-flow trading-value endpoint.",
            },
        }
    )

    assert notice[0] == "error"
    assert "접근을 차단" in notice[1]
    assert "KRX_ID / KRX_PW" in notice[1]


def test_build_investor_flow_refresh_notice_mentions_bootstrap_partial_preview():
    notice = dashboard_data.build_investor_flow_refresh_notice(
        {
            "status": "LIVE",
            "coverage_complete": False,
            "rows": 12,
            "failed_codes": {},
            "window": {"complete_cursor": ""},
        }
    )

    assert notice[0] == "warning"
    assert "partial preview" in notice[1]


def test_build_investor_flow_refresh_notice_exposes_resolver_window_details():
    notice = dashboard_data.build_investor_flow_refresh_notice(
        {
            "status": "CACHED",
            "coverage_complete": False,
            "rows": 0,
            "failed_codes": {"005930": "buy frame empty"},
            "window": {
                "mode": "replay_uncertain_span",
                "anchor_start": "20251211",
                "anchor_reason": "earliest_unresolved_requested_start",
            },
        }
    )

    assert notice[0] == "warning"
    assert "resolver=replay_uncertain_span" in notice[1]
    assert "anchor=2025-12-11" in notice[1]


def test_build_investor_flow_refresh_notice_surfaces_warehouse_write_lock():
    notice = dashboard_data.build_investor_flow_refresh_notice(
        {
            "status": "LIVE",
            "coverage_complete": True,
            "rows": 12,
            "failed_codes": {},
            "warehouse_write_skipped": True,
        }
    )

    assert notice[0] == "warning"
    assert "warehouse write lock" in notice[1]
    assert "임시 preview" in notice[1]
    assert "warehouse에 반영되지 않았습니다" in notice[1]


def test_build_market_refresh_notice_surfaces_warehouse_write_lock_preview():
    notice = dashboard_data.build_market_refresh_notice(
        {
            "status": "LIVE",
            "coverage_complete": False,
            "warehouse_write_skipped": True,
            "provider": "YFINANCE",
            "rows": 12,
        }
    )

    assert notice[0] == "warning"
    assert "warehouse write lock" in notice[1]
    assert "temporary preview" in notice[1]


def test_build_market_refresh_notice_includes_provider_failure_detail():
    notice = dashboard_data.build_market_refresh_notice(
        {
            "status": "CACHED",
            "coverage_complete": False,
            "provider": "OPENAPI",
            "failed_days": [],
            "failed_codes": {"1001": "KRX OpenAPI returned no data rows"},
            "rows": 10,
        }
    )

    assert notice[0] == "warning"
    assert notice[1] != "Market data refresh fell back to cache."
    assert "1001=KRX OpenAPI returned no data rows" in notice[1]


def test_cached_investor_flow_uses_bootstrap_partial_preview(monkeypatch):
    dashboard_data.cached_investor_flow.clear()
    frame = pd.DataFrame(
        {
            "sector_code": ["5044"],
            "sector_name": ["KRX 반도체"],
            "investor_type": ["외국인"],
            "buy_amount": [100],
            "sell_amount": [50],
            "net_buy_amount": [50],
            "net_flow_ratio": [0.2],
        },
        index=pd.to_datetime(["2026-04-07"]),
    )
    monkeypatch.setattr(
        "src.data_sources.krx_investor_flow.load_sector_investor_flow",
        lambda **kwargs: ("CACHED", frame),
    )
    monkeypatch.setattr(
        "src.data_sources.krx_investor_flow.read_warm_status",
        lambda: {"status": "LIVE", "coverage_complete": False, "end": "", "failed_codes": {}},
    )

    status, fresh, detail, loaded = dashboard_data.cached_investor_flow("KR", "20260407", ("flow",))

    assert status == "CACHED"
    assert fresh is False
    assert detail["bootstrap_partial_preview"] is True
    assert not loaded.empty


def test_cached_investor_flow_prefers_transient_live_preview(monkeypatch):
    dashboard_data.cached_investor_flow.clear()
    frame = pd.DataFrame(
        {
            "sector_code": ["5044"],
            "sector_name": ["KRX 반도체"],
            "investor_type": ["외국인"],
            "buy_amount": [100],
            "sell_amount": [50],
            "net_buy_amount": [50],
            "net_flow_ratio": [0.2],
        },
        index=pd.to_datetime(["2026-04-07"]),
    )
    monkeypatch.setattr(
        dashboard_data,
        "_load_investor_flow_transient_override",
        lambda **kwargs: (
            "LIVE",
            {"coverage_complete": True, "end": "20260407", "warehouse_write_skipped": True},
            frame,
        ),
    )
    monkeypatch.setattr(
        "src.data_sources.krx_investor_flow.load_sector_investor_flow",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("warehouse path should not run")),
    )

    status, fresh, detail, loaded = dashboard_data.cached_investor_flow("KR", "20260407", ("flow",))

    assert status == "LIVE"
    assert fresh is True
    assert detail["warehouse_write_skipped"] is True
    assert loaded.equals(frame)


def test_cached_investor_flow_loads_us_flow_proxies(monkeypatch):
    dashboard_data.cached_investor_flow.clear()
    frame = pd.DataFrame(
        {
            "sector_code": ["XLF"],
            "sector_name": ["Financials"],
            "activity_state": ["elevated"],
            "activity_zscore": [1.2],
            "dollar_volume": [100_000_000.0],
        },
        index=pd.to_datetime(["2026-04-07"]),
    )
    monkeypatch.setattr(
        "src.data_sources.us_flow_proxies.load_us_flow_proxies",
        lambda **kwargs: (
            "LIVE",
            frame,
            {"provider": "YFINANCE+SSGA", "coverage_complete": True, "watermark_key": "20260407"},
        ),
    )

    status, fresh, detail, loaded = dashboard_data.cached_investor_flow("US", "20260407", ("price",))

    assert status == "LIVE"
    assert fresh is True
    assert detail["provider"] == "YFINANCE+SSGA"
    assert loaded.equals(frame)


def test_cached_signals_enables_flow_for_bootstrap_partial_preview(monkeypatch):
    flow_frame = _kr_flow_frame()
    captured, loaded_frame = _run_kr_cached_signals(
        monkeypatch,
        ("CACHED", False, {"bootstrap_partial_preview": True}, flow_frame),
    )

    assert captured["flow_enabled"] is False
    assert captured["sector_investor_flow"].equals(loaded_frame)


def test_cached_signals_enables_flow_for_complete_fresh_kr(monkeypatch):
    flow_frame = _kr_flow_frame()
    captured, loaded_frame = _run_kr_cached_signals(
        monkeypatch,
        ("LIVE", True, {"coverage_complete": True, "end": "20260408"}, flow_frame),
    )

    assert captured["flow_enabled"] is True
    assert captured["sector_investor_flow"].equals(loaded_frame)


def test_cached_signals_keeps_flow_display_only_for_warehouse_write_skipped_preview(monkeypatch):
    flow_frame = _kr_flow_frame()
    captured, loaded_frame = _run_kr_cached_signals(
        monkeypatch,
        (
            "LIVE",
            True,
            {"coverage_complete": True, "end": "20260408", "warehouse_write_skipped": True},
            flow_frame,
        ),
    )

    assert captured["flow_enabled"] is False
    assert captured["sector_investor_flow"].equals(loaded_frame)


def test_cached_signals_keeps_stale_cached_flow_display_only(monkeypatch):
    flow_frame = _kr_flow_frame(day="2026-04-07")
    captured, loaded_frame = _run_kr_cached_signals(
        monkeypatch,
        ("CACHED", False, {"coverage_complete": False, "end": "20260407"}, flow_frame),
    )

    assert captured["flow_enabled"] is False
    assert captured["sector_investor_flow"].equals(loaded_frame)


def test_cached_signals_keeps_us_flow_overlay_disabled(monkeypatch):
    settings, sector_map, macro_series_cfg, market_profile = load_market_configs("US")
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

    dates = pd.to_datetime(["2026-04-07", "2026-04-08"])
    sector_prices = pd.DataFrame(
        {
            "index_code": ["^GSPC", "XLK"],
            "index_name": ["S&P 500", "Technology"],
            "close": [100.0, 101.0],
        },
        index=dates[:2],
    )
    flow_frame = pd.DataFrame(
        {
            "sector_code": ["XLK"],
            "sector_name": ["Technology"],
            "activity_state": ["elevated"],
            "activity_zscore": [1.1],
        },
        index=pd.to_datetime(["2026-04-08"]),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(dashboard_data, "_cached_sector_prices", lambda *args, **kwargs: ("LIVE", sector_prices))
    monkeypatch.setattr(dashboard_data, "_cached_macro", lambda *args, **kwargs: ("SAMPLE", pd.DataFrame()))
    monkeypatch.setattr(
        dashboard_data,
        "_cached_investor_flow",
        lambda *args, **kwargs: ("LIVE", True, {"watermark_key": "20260408"}, flow_frame),
    )

    def _fake_build_signal_table(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("src.signals.matrix.build_signal_table", _fake_build_signal_table)

    dashboard_data._cached_signals(
        "US",
        "20260408",
        (0, 0),
        (0, 0),
        "params",
        "macro",
        "price",
        ("flow",),
        0.0,
        20,
        20,
        60,
        3,
    )

    assert captured["flow_enabled"] is False
    assert captured["sector_investor_flow"].equals(flow_frame)
