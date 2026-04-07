from __future__ import annotations

from datetime import date

from src.dashboard import runtime
from src.dashboard.types import DashboardContext


class _DummyCache:
    def __init__(self) -> None:
        self.clears = 0

    def clear(self) -> None:
        self.clears += 1


def _context(market_id: str = "KR") -> DashboardContext:
    return DashboardContext(
        market_id=market_id,
        market_profile=None,
        settings={},
        sector_map={},
        macro_series_cfg={},
        benchmark_code="1001" if market_id == "KR" else "SPY",
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
    macro = _DummyCache()
    signals = _DummyCache()

    monkeypatch.setattr(runtime, "cached_api_preflight", api)
    monkeypatch.setattr(runtime, "cached_sector_prices", sector)
    monkeypatch.setattr(runtime, "cached_analysis_sector_prices", analysis)
    monkeypatch.setattr(runtime, "cached_macro", macro)
    monkeypatch.setattr(runtime, "cached_signals", signals)

    runtime.invalidate_dashboard_caches("market")
    assert api.clears == 1
    assert sector.clears == 1
    assert analysis.clears == 1
    assert macro.clears == 0
    assert signals.clears == 1

    runtime.invalidate_dashboard_caches("macro")
    assert macro.clears == 1
    assert signals.clears == 2

    runtime.invalidate_dashboard_caches("all")
    assert api.clears == 2
    assert sector.clears == 2
    assert analysis.clears == 2
    assert macro.clears == 2
    assert signals.clears == 3


def test_run_market_refresh_returns_notice_and_invalidates(monkeypatch):
    close_calls: list[str] = []
    invalidate_calls: list[str] = []

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

    notice = runtime.run_market_refresh(_context("US"), price_years=3)

    assert notice == ("success", "US:SPY,5044")
    assert close_calls == ["close"]
    assert invalidate_calls == ["market"]


def test_run_macro_refresh_returns_notice_and_invalidates(monkeypatch):
    close_calls: list[str] = []
    invalidate_calls: list[str] = []

    monkeypatch.setattr(runtime, "close_cached_read_only_connection", lambda: close_calls.append("close"))
    monkeypatch.setattr(runtime, "invalidate_dashboard_caches", lambda scope: invalidate_calls.append(scope))
    monkeypatch.setattr(runtime, "build_macro_refresh_notice", lambda summary: ("info", summary["status"]))
    monkeypatch.setattr(runtime, "_sync_macro_warehouse", lambda **kwargs: (None, None, {"status": kwargs["market"]}))

    notice = runtime.run_macro_refresh(_context("KR"), {"series": []})

    assert notice == ("info", "KR")
    assert close_calls == ["close"]
    assert invalidate_calls == ["macro"]


def test_run_feature_recompute_clears_features_and_invalidates(tmp_path, monkeypatch):
    close_calls: list[str] = []
    invalidate_calls: list[str] = []
    features_dir = tmp_path / "data" / "features"
    features_dir.mkdir(parents=True)
    (features_dir / "artifact.txt").write_text("x", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime, "close_cached_read_only_connection", lambda: close_calls.append("close"))
    monkeypatch.setattr(runtime, "invalidate_dashboard_caches", lambda scope: invalidate_calls.append(scope))

    runtime.run_feature_recompute()

    assert close_calls == ["close"]
    assert invalidate_calls == ["signals"]
    assert features_dir.exists()
    assert list(features_dir.iterdir()) == []
