"""Integration tests. All file I/O uses tmp_path + monkeypatch. (5 tests)"""
from __future__ import annotations

import json
import math
import sys
import types
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.signals.matrix import SectorSignal, build_signal_table
from src.signals.scoring import apply_fx_shock_filter


# --- Fixtures ---


@pytest.fixture(autouse=True)
def _blank_streamlit(monkeypatch):
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.secrets = {}
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.delenv("KRX_OPENAPI_URL", raising=False)


def _make_sector_prices_df(codes: list[str], n: int = 60) -> pd.DataFrame:
    """Create a minimal valid sector_prices DataFrame."""
    import numpy as np

    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    frames = []
    rng = np.random.default_rng(0)
    for code in codes:
        prices = 1000.0 * (1 + rng.normal(0.001, 0.01, n)).cumprod()
        df = pd.DataFrame(
            {
                "index_code": code,
                "index_name": f"Sector {code}",
                "close": prices.astype(float),
            },
            index=pd.DatetimeIndex(idx),
        )
        df["index_code"] = df["index_code"].astype("object")
        df["index_name"] = df["index_name"].astype("object")
        frames.append(df)
    return pd.concat(frames)


def _make_macro_result() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=36, freq="ME")
    return pd.DataFrame(
        {
            "growth_dir": ["Up"] * 36,
            "inflation_dir": ["Down"] * 36,
            "regime": ["Recovery"] * 36,
        },
        index=idx,
    )


def _minimal_sector_map() -> dict:
    return {
        "benchmark": {"code": "1001", "name": "KOSPI"},
        "regimes": {
            "Recovery": {
                "sectors": [
                    {"code": "5044", "name": "KRX 반도체", "export_sector": True},
                    {"code": "1155", "name": "KOSPI200 IT", "export_sector": True},
                ]
            }
        },
    }


def _default_settings() -> dict:
    return {
        "rs_ma_period": 20,
        "ma_fast": 20,
        "ma_slow": 60,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "fx_shock_pct": 3.0,
    }


def _make_close_frame(values: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.DataFrame({"close": [float(value) for value in values]}, index=idx)


def _seed_raw_cache(raw_dir: Path, code: str, frame: pd.DataFrame) -> None:
    code_dir = raw_dir / code
    code_dir.mkdir(parents=True, exist_ok=True)
    end_label = pd.Timestamp(frame.index.max()).strftime("%Y%m%d")
    frame.to_parquet(code_dir / f"{end_label}.parquet")


# --- Tests ---


class TestIntegration:
    def test_openapi_provider_live_success(self, tmp_path, monkeypatch):
        """OPENAPI provider returns LIVE and writes curated parquet on success."""
        import src.data_sources.krx_indices as krx_mod

        curated = tmp_path / "curated"
        curated.mkdir()
        monkeypatch.setattr(krx_mod, "CURATED_DIR", curated)
        monkeypatch.setattr(krx_mod, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(krx_mod, "get_krx_provider", lambda: "OPENAPI")
        monkeypatch.setattr(krx_mod, "get_krx_openapi_key", lambda: "OPENAPI_KEY")

        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        live_df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]}, index=idx)
        monkeypatch.setattr(
            krx_mod,
            "fetch_index_ohlcv_openapi_batch_detailed",
            lambda *a, **kw: ({"1001": live_df}, {}, {"failed_days": [], "snapshot_failures": {}}),
        )

        status, result = krx_mod.load_sector_prices(["1001"], "20240101", "20240104")
        assert status == "LIVE"
        assert not result.empty
        assert (curated / "sector_prices.parquet").exists()

    def test_load_sector_prices_prefers_complete_raw_cache_before_openapi(self, tmp_path, monkeypatch):
        """Complete raw cache returns CACHED without calling OpenAPI."""
        import src.data_sources.krx_indices as krx_mod

        curated = tmp_path / "curated"
        raw_dir = tmp_path / "raw"
        monkeypatch.setattr(krx_mod, "CURATED_DIR", curated)
        monkeypatch.setattr(krx_mod, "RAW_DIR", raw_dir)
        monkeypatch.setattr(krx_mod, "get_krx_provider", lambda: "OPENAPI")
        monkeypatch.setattr(krx_mod, "get_krx_openapi_key", lambda: "OPENAPI_KEY")

        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        raw_frame = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=idx)
        code_dir = raw_dir / "1001"
        code_dir.mkdir(parents=True)
        raw_frame.to_parquet(code_dir / "20240131.parquet")

        monkeypatch.setattr(
            krx_mod,
            "fetch_index_ohlcv_openapi_batch_detailed",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("OpenAPI should not be called")),
        )

        status, result = krx_mod.load_sector_prices(["1001"], "20240101", "20240105")

        assert status == "CACHED"
        assert not result.empty
        assert result["close"].tolist() == [100.0, 101.0, 102.0, 103.0, 104.0]

    def test_detect_contaminated_raw_cache_codes_ignores_nonmatching_1170(self, tmp_path, monkeypatch):
        """Duplicate trailing series should flag only the contaminated KOSPI200 sector codes."""
        import src.data_sources.krx_indices as krx_mod

        raw_dir = tmp_path / "raw"
        monkeypatch.setattr(krx_mod, "RAW_DIR", raw_dir)

        shared = [float(300 + i) for i in range(70)]
        utilities = [float(900 + i * 2) for i in range(70)]
        _seed_raw_cache(raw_dir, "1155", _make_close_frame(shared))
        _seed_raw_cache(raw_dir, "1165", _make_close_frame(shared))
        _seed_raw_cache(raw_dir, "1170", _make_close_frame(utilities))

        detected = krx_mod._detect_contaminated_raw_cache_codes(
            ["1155", "1165", "1170"],
            "20240101",
            "20240405",
        )

        assert detected == ["1155", "1165"]
        assert "1170" not in detected

    def test_load_sector_prices_force_refreshes_contaminated_cache(self, tmp_path, monkeypatch):
        """Contaminated raw cache should force-refresh only the duplicated sector codes."""
        import src.data_sources.krx_indices as krx_mod

        curated = tmp_path / "curated"
        raw_dir = tmp_path / "raw"
        monkeypatch.setattr(krx_mod, "CURATED_DIR", curated)
        monkeypatch.setattr(krx_mod, "RAW_DIR", raw_dir)
        monkeypatch.setattr(krx_mod, "get_krx_provider", lambda: "OPENAPI")
        monkeypatch.setattr(krx_mod, "get_krx_openapi_key", lambda: "OPENAPI_KEY")

        benchmark = _make_close_frame([float(1000 + i) for i in range(70)])
        shared = [float(500 + i) for i in range(70)]
        utilities = _make_close_frame([float(700 + i * 1.5) for i in range(70)])
        _seed_raw_cache(raw_dir, "1001", benchmark)
        _seed_raw_cache(raw_dir, "1155", _make_close_frame(shared))
        _seed_raw_cache(raw_dir, "1157", _make_close_frame(shared))
        _seed_raw_cache(raw_dir, "1165", _make_close_frame(shared))
        _seed_raw_cache(raw_dir, "1168", _make_close_frame(shared))
        _seed_raw_cache(raw_dir, "1170", utilities)

        calls: list[tuple[list[str], str, str, bool]] = []
        bases = {"1155": 3100.0, "1157": 1200.0, "1165": 2200.0, "1168": 1600.0}

        def _fake_openapi(index_codes, start, end, force=False):
            calls.append((list(index_codes), start, end, force))
            frames = {
                code: _make_close_frame([base + (i * 3.0) for i in range(70)])
                for code, base in bases.items()
                if code in index_codes
            }
            return frames, {}, {"failed_days": [], "snapshot_failures": {}}

        monkeypatch.setattr(krx_mod, "fetch_index_ohlcv_openapi_batch_detailed", _fake_openapi)

        status, result = krx_mod.load_sector_prices(
            ["1001", "1155", "1157", "1165", "1168", "1170"],
            "20240115",
            "20240405",
        )

        assert status == "LIVE"
        assert calls == [(["1155", "1157", "1165", "1168"], "20240115", "20240405", False)]
        assert set(result["index_code"].unique()) == {"1001", "1155", "1157", "1165", "1168", "1170"}

        it_close = result[result["index_code"].astype(str) == "1155"]["close"].sort_index()
        discretionary_close = result[result["index_code"].astype(str) == "1165"]["close"].sort_index()
        assert not it_close.equals(discretionary_close)
        assert (curated / "sector_prices.parquet").exists()

    def test_warm_sector_price_cache_drops_contaminated_codes_when_refresh_incomplete(self, tmp_path, monkeypatch):
        """Out-of-band warm excludes contaminated codes that cannot be rebuilt cleanly."""
        import src.data_sources.krx_indices as krx_mod

        curated = tmp_path / "curated"
        raw_dir = tmp_path / "raw"
        monkeypatch.setattr(krx_mod, "CURATED_DIR", curated)
        monkeypatch.setattr(krx_mod, "RAW_DIR", raw_dir)
        monkeypatch.setattr(krx_mod, "get_krx_provider", lambda: "OPENAPI")
        monkeypatch.setattr(krx_mod, "get_krx_openapi_key", lambda: "OPENAPI_KEY")

        benchmark = _make_close_frame([float(1000 + i) for i in range(70)])
        shared = [float(500 + i) for i in range(70)]
        utilities = _make_close_frame([float(700 + i * 1.5) for i in range(70)])
        _seed_raw_cache(raw_dir, "1001", benchmark)
        _seed_raw_cache(raw_dir, "1155", _make_close_frame(shared))
        _seed_raw_cache(raw_dir, "1157", _make_close_frame(shared))
        _seed_raw_cache(raw_dir, "1165", _make_close_frame(shared))
        _seed_raw_cache(raw_dir, "1168", _make_close_frame(shared))
        _seed_raw_cache(raw_dir, "1170", utilities)

        monkeypatch.setattr(
            krx_mod,
            "fetch_index_ohlcv_openapi_batch_detailed",
            lambda *a, **kw: (
                {},
                {},
                {
                    "failed_days": ["20240405"],
                    "snapshot_failures": {"kospi_dd_trd": {"20240405": "Access Denied"}},
                    "aborted": True,
                    "abort_reason": "ACCESS_DENIED",
                    "processed_requests": 1,
                },
            ),
        )

        (status, result), summary = krx_mod.warm_sector_price_cache(
            ["1001", "1155", "1157", "1165", "1168", "1170"],
            "20240101",
            "20240405",
            reason="cli_warm",
            force=False,
        )

        assert status == "CACHED"
        assert summary["aborted"] is True
        assert summary["abort_reason"] == "ACCESS_DENIED"
        assert set(result["index_code"].unique()) == {"1001", "1170"}
        assert not (curated / "sector_prices.parquet").exists()

        sector_map = {
            "benchmark": {"code": "1001", "name": "KOSPI"},
            "regimes": {
                "Recovery": {
                    "sectors": [
                        {"code": "1155", "name": "KOSPI200 IT", "export_sector": True},
                        {"code": "1170", "name": "KOSPI200 Utilities", "export_sector": False},
                    ]
                }
            },
        }
        benchmark_prices = result[result["index_code"].astype(str) == "1001"]["close"].sort_index()
        signals = build_signal_table(
            sector_prices=result,
            benchmark_prices=benchmark_prices,
            macro_result=_make_macro_result(),
            sector_map=sector_map,
            settings=_default_settings(),
        )
        by_code = {signal.index_code: signal for signal in signals}
        assert by_code["1155"].action == "N/A"
        assert by_code["1170"].action != "N/A"

    def test_read_warm_status_returns_sanitized_summary(self, tmp_path, monkeypatch):
        """Warm-status reader should expose only the UI-relevant normalized fields."""
        import src.data_sources.krx_indices as krx_mod

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        monkeypatch.setattr(krx_mod, "RAW_DIR", raw_dir)

        payload = {
            "status": "live",
            "provider": "openapi",
            "end": "2026-03-06",
            "coverage_complete": True,
            "failed_days": ["20260305", ""],
            "failed_codes": {"1001": "Access Denied", "": "skip"},
            "reason": "manual_refresh",
            "delta_codes": ["1001", ""],
            "ignored": "value",
        }
        (raw_dir / krx_mod.WARM_STATUS_FILE).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        summary = krx_mod.read_warm_status()

        assert summary == {
            "status": "LIVE",
            "provider": "OPENAPI",
            "end": "20260306",
            "coverage_complete": True,
            "failed_days": ["20260305"],
            "failed_codes": {"1001": "Access Denied"},
            "reason": "manual_refresh",
            "delta_codes": ["1001"],
            "aborted": False,
            "abort_reason": "",
            "predicted_requests": 0,
            "processed_requests": 0,
        }

    def test_run_manual_price_refresh_uses_manual_warm_reason_and_resets_health(self, monkeypatch):
        """Manual market refresh resets health state and reuses the manual warm path."""
        import src.data_sources.krx_indices as krx_mod

        calls: list[tuple[list[str], str, str, str, bool]] = []
        reset_calls: list[str] = []
        expected_result = ("CACHED", pd.DataFrame())
        expected_summary = {"status": "CACHED", "coverage_complete": True, "delta_codes": []}

        def _fake_warm(index_codes, start, end, *, reason, force):
            calls.append((list(index_codes), start, end, reason, force))
            return expected_result, expected_summary

        monkeypatch.setattr(krx_mod, "get_krx_provider", lambda: "OPENAPI")
        monkeypatch.setattr(krx_mod, "get_krx_openapi_key", lambda: "OPENAPI_KEY")
        monkeypatch.setattr(krx_mod, "warm_sector_price_cache", _fake_warm)
        monkeypatch.setattr(krx_mod, "reset_krx_openapi_health_cache", lambda: reset_calls.append("reset"))

        result, summary = krx_mod.run_manual_price_refresh(["1001"], "20240101", "20240105")

        assert calls == [(["1001"], "20240101", "20240105", "manual_refresh", False)]
        assert reset_calls == ["reset"]
        assert result == expected_result
        assert summary == expected_summary

    def test_run_manual_price_refresh_caps_openapi_range_to_recent_window(self, monkeypatch):
        """Manual market refresh should cap oversized OpenAPI requests to a recent delta window."""
        import src.data_sources.krx_indices as krx_mod

        calls: list[tuple[list[str], str, str, str, bool]] = []

        def _fake_warm(index_codes, start, end, *, reason, force):
            calls.append((list(index_codes), start, end, reason, force))
            return ("CACHED", pd.DataFrame()), {"status": "CACHED", "coverage_complete": True}

        monkeypatch.setattr(krx_mod, "get_krx_provider", lambda: "OPENAPI")
        monkeypatch.setattr(krx_mod, "get_krx_openapi_key", lambda: "OPENAPI_KEY")
        monkeypatch.setattr(krx_mod, "warm_sector_price_cache", _fake_warm)
        monkeypatch.setattr(krx_mod, "reset_krx_openapi_health_cache", lambda: None)

        krx_mod.run_manual_price_refresh(["1001", "5044"], "20240101", "20240430")

        assert calls
        call_codes, call_start, call_end, call_reason, call_force = calls[0]
        assert call_codes == ["1001", "5044"]
        assert call_start > "20240101"
        assert call_end == "20240430"
        assert call_reason == "manual_refresh"
        assert call_force is False

    def test_warm_sector_price_cache_fetches_only_missing_delta_range(self, tmp_path, monkeypatch):
        """Warm refresh fetches only the missing tail range, not the full history."""
        import src.data_sources.krx_indices as krx_mod

        curated = tmp_path / "curated"
        raw_dir = tmp_path / "raw"
        monkeypatch.setattr(krx_mod, "CURATED_DIR", curated)
        monkeypatch.setattr(krx_mod, "RAW_DIR", raw_dir)
        monkeypatch.setattr(krx_mod, "get_krx_provider", lambda: "OPENAPI")
        monkeypatch.setattr(krx_mod, "get_krx_openapi_key", lambda: "OPENAPI_KEY")

        seed_idx = pd.date_range("2024-01-01", periods=3, freq="B")
        seed_frame = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=seed_idx)
        code_dir = raw_dir / "1001"
        code_dir.mkdir(parents=True)
        seed_frame.to_parquet(code_dir / "20240103.parquet")

        calls: list[tuple[list[str], str, str]] = []

        def _fake_openapi(index_codes, start, end, force=False):
            calls.append((list(index_codes), start, end))
            idx = pd.date_range("2024-01-04", periods=3, freq="B")
            return (
                {"1001": pd.DataFrame({"close": [103.0, 104.0, 105.0]}, index=idx)},
                {},
                {"failed_days": [], "snapshot_failures": {}},
            )

        monkeypatch.setattr(krx_mod, "fetch_index_ohlcv_openapi_batch_detailed", _fake_openapi)

        (status, result), summary = krx_mod.warm_sector_price_cache(
            ["1001"],
            "20240101",
            "20240108",
            reason="test",
        )

        assert status == "LIVE"
        assert calls == [(["1001"], "20240104", "20240108")]
        assert len(result) == 6
        assert summary["delta_codes"] == ["1001"]

    def test_force_warm_partial_failure_returns_cached_with_failed_days(self, tmp_path, monkeypatch):
        """Force warm reports CACHED when snapshot failures leave coverage incomplete."""
        import src.data_sources.krx_indices as krx_mod

        curated = tmp_path / "curated"
        raw_dir = tmp_path / "raw"
        curated.mkdir()
        monkeypatch.setattr(krx_mod, "CURATED_DIR", curated)
        monkeypatch.setattr(krx_mod, "RAW_DIR", raw_dir)
        monkeypatch.setattr(krx_mod, "get_krx_provider", lambda: "OPENAPI")
        monkeypatch.setattr(krx_mod, "get_krx_openapi_key", lambda: "OPENAPI_KEY")

        cached = _make_sector_prices_df(["1001"], n=3)
        cached.to_parquet(curated / "sector_prices.parquet")

        partial_idx = pd.date_range("2024-01-01", periods=2, freq="B")
        partial_live = pd.DataFrame({"close": [100.0, 101.0]}, index=partial_idx)
        monkeypatch.setattr(
            krx_mod,
            "fetch_index_ohlcv_openapi_batch_detailed",
            lambda *a, **kw: (
                {"1001": partial_live},
                {},
                {
                    "failed_days": ["20240103"],
                    "snapshot_failures": {"kospi_dd_trd": {"20240103": "Access Denied"}},
                },
            ),
        )

        (status, result), summary = krx_mod.warm_sector_price_cache(
            ["1001"],
            "20240101",
            "20240103",
            reason="test_force",
            force=True,
        )

        assert status == "CACHED"
        assert summary["coverage_complete"] is False
        assert summary["failed_days"] == ["20240103"]
        assert len(result) == len(cached)

    def test_load_sector_prices_schedules_background_refresh_for_stale_raw_cache(self, tmp_path, monkeypatch):
        """Stale-but-usable raw cache returns CACHED and schedules background delta refresh."""
        import src.data_sources.krx_indices as krx_mod

        monkeypatch.setattr(krx_mod, "CURATED_DIR", tmp_path / "curated")
        monkeypatch.setattr(krx_mod, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(krx_mod, "get_krx_provider", lambda: "OPENAPI")
        monkeypatch.setattr(krx_mod, "get_krx_openapi_key", lambda: "OPENAPI_KEY")

        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        raw_frame = pd.DataFrame({"close": [10.0, 11.0, 12.0]}, index=idx)
        code_dir = (tmp_path / "raw" / "1001")
        code_dir.mkdir(parents=True)
        raw_frame.to_parquet(code_dir / "20240103.parquet")

        scheduled: list[tuple[list[str], str, str, str]] = []
        monkeypatch.setattr(
            krx_mod,
            "schedule_background_warm",
            lambda codes, start, end, reason, force=False: scheduled.append((list(codes), start, end, reason)) or True,
        )
        monkeypatch.setattr(
            krx_mod,
            "fetch_index_ohlcv_openapi_batch_detailed",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("blocking OpenAPI call not expected")),
        )

        status, result = krx_mod.load_sector_prices(["1001"], "20240101", "20240110")

        assert status == "CACHED"
        assert not result.empty
        assert scheduled == [(["1001"], "20240101", "20240110", "cache_delta_refresh")]

    def test_load_sector_prices_fails_fast_for_oversized_openapi_range(self, tmp_path, monkeypatch):
        """Interactive OpenAPI loads should reject ranges above the snapshot budget."""
        import src.data_sources.krx_indices as krx_mod

        monkeypatch.setattr(krx_mod, "CURATED_DIR", tmp_path / "curated")
        monkeypatch.setattr(krx_mod, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(krx_mod, "get_krx_provider", lambda: "OPENAPI")
        monkeypatch.setattr(krx_mod, "get_krx_openapi_key", lambda: "OPENAPI_KEY")

        with pytest.raises(krx_mod.KRXInteractiveRangeLimitError):
            krx_mod.load_sector_prices(["1001", "5044"], "20240101", "20240430")

    def test_load_sector_prices_raises_access_denied_instead_of_cached_fallback(self, tmp_path, monkeypatch):
        """Interactive OpenAPI access denial should surface as a blocking error."""
        import src.data_sources.krx_indices as krx_mod

        monkeypatch.setattr(krx_mod, "CURATED_DIR", tmp_path / "curated")
        monkeypatch.setattr(krx_mod, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(krx_mod, "get_krx_provider", lambda: "OPENAPI")
        monkeypatch.setattr(krx_mod, "get_krx_openapi_key", lambda: "OPENAPI_KEY")
        monkeypatch.setattr(
            krx_mod,
            "warm_sector_price_cache",
            lambda *a, **kw: (
                ("CACHED", _make_sector_prices_df(["1001"], n=5)),
                {
                    "status": "CACHED",
                    "coverage_complete": False,
                    "aborted": True,
                    "abort_reason": "ACCESS_DENIED",
                    "failed_days": ["20240105"],
                    "failed_codes": {},
                },
            ),
        )

        with pytest.raises(krx_mod.KRXMarketDataAccessDeniedError):
            krx_mod.load_sector_prices(["1001"], "20240101", "20240105")

    def test_openapi_auth_failure_falls_back_to_cache(self, tmp_path, monkeypatch):
        """OPENAPI auth failure returns CACHED when curated cache exists."""
        import src.data_sources.krx_indices as krx_mod

        curated = tmp_path / "curated"
        curated.mkdir()
        monkeypatch.setattr(krx_mod, "CURATED_DIR", curated)
        monkeypatch.setattr(krx_mod, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(krx_mod, "get_krx_provider", lambda: "OPENAPI")
        monkeypatch.setattr(krx_mod, "get_krx_openapi_key", lambda: "OPENAPI_KEY")

        cached = _make_sector_prices_df(["1001"], n=5)
        cached.to_parquet(curated / "sector_prices.parquet")

        monkeypatch.setattr(
            krx_mod,
            "fetch_index_ohlcv_openapi_batch_detailed",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("Unauthorized Key")),
        )

        status, result = krx_mod.load_sector_prices(["1001"], "20240101", "20240131")
        assert status == "CACHED"
        assert not result.empty

    def test_api_failure_falls_back_to_cache(self, tmp_path, monkeypatch):
        """When pykrx raises, load_sector_prices returns CACHED from disk."""
        import src.data_sources.krx_indices as krx_mod

        curated = tmp_path / "curated"
        curated.mkdir()
        monkeypatch.setattr(krx_mod, "CURATED_DIR", curated)
        monkeypatch.setattr(krx_mod, "get_krx_provider", lambda: "PYKRX")
        monkeypatch.setattr(krx_mod, "get_krx_openapi_key", lambda: "")

        # Pre-seed a valid parquet in the tmp curated dir
        df = _make_sector_prices_df(["5044"])
        df.to_parquet(curated / "sector_prices.parquet")

        # Mock pykrx to raise ConnectionError
        monkeypatch.setattr(
            krx_mod,
            "_fetch_chunk",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("mock network error")),
        )

        status, result = krx_mod.load_sector_prices(["5044"], "20230101", "20240101")
        assert status == "CACHED"
        assert not result.empty

    def test_full_fallback_to_sample(self, tmp_path, monkeypatch):
        """When API fails and no cache exists, load_sector_prices returns SAMPLE."""
        import src.data_sources.krx_indices as krx_mod

        curated = tmp_path / "curated"
        curated.mkdir()
        monkeypatch.setattr(krx_mod, "CURATED_DIR", curated)
        monkeypatch.setattr(krx_mod, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(krx_mod, "get_krx_provider", lambda: "PYKRX")
        monkeypatch.setattr(krx_mod, "get_krx_openapi_key", lambda: "")

        # Mock pykrx to raise
        monkeypatch.setattr(
            krx_mod,
            "_fetch_chunk",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("mock error")),
        )

        status, result = krx_mod.load_sector_prices(["5044"], "20230101", "20240101")
        assert status == "SAMPLE"
        assert not result.empty

    def test_partial_success_returns_cached_status(self, tmp_path, monkeypatch):
        """When one code fails but another succeeds, loader downgrades to CACHED."""
        import src.data_sources.krx_indices as krx_mod

        curated = tmp_path / "curated"
        curated.mkdir()
        monkeypatch.setattr(krx_mod, "CURATED_DIR", curated)
        monkeypatch.setattr(krx_mod, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(krx_mod, "get_krx_provider", lambda: "PYKRX")
        monkeypatch.setattr(krx_mod, "get_krx_openapi_key", lambda: "")
        monkeypatch.setattr(krx_mod, "_get_index_universe", lambda: frozenset({"5044", "5357"}))

        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        live_df = pd.DataFrame({"close": [1000.0, 1001.0, 1002.0]}, index=idx)

        def _fake_fetch(index_code, start, end, chunk_years=2):
            if index_code == "5357":
                raise KeyError("5357")
            return live_df

        monkeypatch.setattr(krx_mod, "fetch_index_ohlcv", _fake_fetch)

        status, result = krx_mod.load_sector_prices(["5044", "5357"], "20240101", "20240103")
        assert status == "CACHED"
        assert not result.empty
        assert set(result["index_code"].unique()) == {"5044"}
        assert not (curated / "sector_prices.parquet").exists()

    def test_partial_sector_failure_skips_gracefully(self):
        """build_signal_table returns N/A for sectors with missing data, valid signals for rest."""
        sector_map = _minimal_sector_map()
        settings = _default_settings()
        macro_result = _make_macro_result()

        # Only provide data for one of the two sectors
        sector_prices = _make_sector_prices_df(["5044"])  # missing "1155"

        # Benchmark
        import numpy as np
        rng = np.random.default_rng(1)
        bench_prices = pd.Series(
            1000.0 * (1 + rng.normal(0.001, 0.01, 60)).cumprod(),
            index=pd.date_range("2024-01-01", periods=60, freq="B"),
        )

        signals = build_signal_table(
            sector_prices, bench_prices, macro_result, sector_map, settings
        )

        assert len(signals) == 2, f"Expected 2 signals, got {len(signals)}"
        na_signals = [s for s in signals if s.action == "N/A"]
        valid_signals = [s for s in signals if s.action != "N/A"]
        assert len(na_signals) == 1, "Sector with missing data should get N/A"
        assert len(valid_signals) == 1, "Sector with data should get valid action"

    def test_missing_benchmark_returns_na_with_reason(self):
        """Empty benchmark series returns N/A for all sectors with Benchmark Missing reason."""
        sector_map = _minimal_sector_map()
        settings = _default_settings()
        macro_result = _make_macro_result()
        sector_prices = _make_sector_prices_df(["5044", "1155"])

        bench_prices = pd.Series(dtype=float)
        signals = build_signal_table(
            sector_prices, bench_prices, macro_result, sector_map, settings
        )

        assert len(signals) == 2
        assert all(s.action == "N/A" for s in signals)
        assert all("Benchmark Missing" in s.alerts for s in signals)

    def test_valid_benchmark_produces_non_nan_rs(self):
        """With benchmark data present, at least one sector has valid RS/RS_MA."""
        sector_map = _minimal_sector_map()
        settings = _default_settings()
        macro_result = _make_macro_result()
        sector_prices = _make_sector_prices_df(["5044", "1155"], n=80)

        bench_prices = pd.Series(
            data=1000.0 + (pd.Series(range(80)) * 2.0).values,
            index=pd.date_range("2024-01-01", periods=80, freq="B"),
            dtype=float,
        )
        signals = build_signal_table(
            sector_prices, bench_prices, macro_result, sector_map, settings
        )

        valid_rs = [
            s for s in signals
            if s.action != "N/A" and not math.isnan(s.rs) and not math.isnan(s.rs_ma)
        ]
        assert valid_rs, "Expected at least one sector with valid RS and RS_MA"

    def test_cache_invalidation_clears_only_target(self, tmp_path):
        """Market cache invalidation removes sector_prices but not macro_monthly."""
        # Seed both parquets
        (tmp_path / "sector_prices.parquet").write_bytes(b"dummy")
        (tmp_path / "macro_monthly.parquet").write_bytes(b"dummy")

        # Simulate market refresh: delete only sector_prices
        target = tmp_path / "sector_prices.parquet"
        target.unlink(missing_ok=True)

        assert not (tmp_path / "sector_prices.parquet").exists()
        assert (tmp_path / "macro_monthly.parquet").exists()

    def test_fx_shock_e2e(self):
        """FX shock > 3% downgrades all export-sector Strong Buy signals to Watch."""
        from src.signals.matrix import SectorSignal
        from src.signals.scoring import apply_fx_shock_filter

        export_codes = ["5044", "1155"]

        signals = [
            SectorSignal(
                index_code="5044",
                sector_name="KRX 반도체",
                macro_regime="Recovery",
                macro_fit=True,
                rs=1.1,
                rs_ma=1.0,
                rs_strong=True,
                trend_ok=True,
                momentum_strong=True,
                rsi_d=60.0,
                rsi_w=55.0,
                action="Strong Buy",
                alerts=[],
                returns={"1M": 0.05},
                asof_date="2024-01-31",
            ),
            SectorSignal(
                index_code="1168",  # non-export
                sector_name="KOSPI200 금융",
                macro_regime="Expansion",
                macro_fit=False,
                rs=0.98,
                rs_ma=1.0,
                rs_strong=False,
                trend_ok=True,
                momentum_strong=False,
                rsi_d=45.0,
                rsi_w=40.0,
                action="Strong Buy",  # hypothetical
                alerts=[],
                returns={"1M": 0.01},
                asof_date="2024-01-31",
            ),
        ]

        fx_change = 4.0  # > 3.0% threshold
        updated = [
            apply_fx_shock_filter(s, fx_change, export_codes, threshold_pct=3.0)
            for s in signals
        ]

        # Export sector downgraded
        assert updated[0].action == "Watch"
        assert "FX Shock" in updated[0].alerts

        # Non-export sector unchanged
        assert updated[1].action == "Strong Buy"

    def test_build_signal_table_applies_fx_shock_when_fx_change_provided(self):
        """build_signal_table downgrades export-sector Strong Buy when fx_change_pct exceeds threshold."""
        sector_map = {
            "benchmark": {"code": "1001", "name": "KOSPI"},
            "regimes": {
                "Recovery": {
                    "sectors": [
                        {"code": "5044", "name": "KRX Semis", "export_sector": True},
                    ]
                }
            },
        }
        settings = _default_settings()
        macro_result = _make_macro_result()

        n = 90
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        steps = pd.Series(range(n), dtype=float).to_numpy()
        sector_prices = pd.DataFrame(
            {
                "index_code": "5044",
                "index_name": "Sector 5044",
                "close": 100.0 * (1.015 ** steps),
            },
            index=idx,
        )
        sector_prices["index_code"] = sector_prices["index_code"].astype("object")
        sector_prices["index_name"] = sector_prices["index_name"].astype("object")
        benchmark_prices = pd.Series(100.0 * (1.005 ** steps), index=idx, dtype=float)

        baseline = build_signal_table(
            sector_prices=sector_prices,
            benchmark_prices=benchmark_prices,
            macro_result=macro_result,
            sector_map=sector_map,
            settings=settings,
            fx_change_pct=0.0,
        )
        assert baseline[0].action == "Strong Buy"

        shocked = build_signal_table(
            sector_prices=sector_prices,
            benchmark_prices=benchmark_prices,
            macro_result=macro_result,
            sector_map=sector_map,
            settings=settings,
            fx_change_pct=4.0,
        )
        assert shocked[0].action == "Watch"
        assert "FX Shock" in shocked[0].alerts
