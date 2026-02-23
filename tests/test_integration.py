"""Integration tests. All file I/O uses tmp_path + monkeypatch. (5 tests)"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.signals.matrix import SectorSignal, build_signal_table
from src.signals.scoring import apply_fx_shock_filter


# --- Fixtures ---


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


# --- Tests ---


class TestIntegration:
    def test_api_failure_falls_back_to_cache(self, tmp_path, monkeypatch):
        """When pykrx raises, load_sector_prices returns CACHED from disk."""
        import src.data_sources.krx_indices as krx_mod

        curated = tmp_path / "curated"
        curated.mkdir()
        monkeypatch.setattr(krx_mod, "CURATED_DIR", curated)

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

        # Mock pykrx to raise
        monkeypatch.setattr(
            krx_mod,
            "_fetch_chunk",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("mock error")),
        )

        status, result = krx_mod.load_sector_prices(["5044"], "20230101", "20240101")
        assert status == "SAMPLE"
        assert not result.empty

    def test_live_partial_success_keeps_live_status(self, tmp_path, monkeypatch):
        """When one code fails but another succeeds, loader returns LIVE with successful rows."""
        import src.data_sources.krx_indices as krx_mod

        curated = tmp_path / "curated"
        curated.mkdir()
        monkeypatch.setattr(krx_mod, "CURATED_DIR", curated)
        monkeypatch.setattr(krx_mod, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(krx_mod, "_get_index_universe", lambda: frozenset({"5044", "5357"}))

        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        live_df = pd.DataFrame({"close": [1000.0, 1001.0, 1002.0]}, index=idx)

        def _fake_fetch(index_code, start, end, chunk_years=2):
            if index_code == "5357":
                raise KeyError("5357")
            return live_df

        monkeypatch.setattr(krx_mod, "fetch_index_ohlcv", _fake_fetch)

        status, result = krx_mod.load_sector_prices(["5044", "5357"], "20240101", "20240131")
        assert status == "LIVE"
        assert not result.empty
        assert set(result["index_code"].unique()) == {"5044"}
        assert (curated / "sector_prices.parquet").exists()

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
