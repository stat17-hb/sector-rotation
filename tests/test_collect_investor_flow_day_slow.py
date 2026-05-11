from __future__ import annotations

from types import SimpleNamespace
import sys

import pandas as pd

import scripts.collect_investor_flow_day_slow as slow
from src.data_sources.krx_investor_flow import SectorUniverse


def _args(**overrides):
    values = {
        "date": "20260422",
        "market": "KR",
        "sectors": "all",
        "sector_codes": "",
        "sleep_sec": 0.0,
        "max_tickers": 0,
        "execute": False,
        "reset_state": False,
        "stop_on_access_denied": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _sector_map():
    return {
        "regimes": {
            "Expansion": {
                "sectors": [
                    {"code": "5044", "name": "KRX 반도체", "export_sector": True},
                    {"code": "5045", "name": "KRX 헬스케어", "export_sector": False},
                ]
            }
        }
    }


def _universe(tickers: list[str] | None = None) -> SectorUniverse:
    ticker_list = tickers or ["005930", "000660"]
    return SectorUniverse(
        sector_codes=["5044"],
        sector_names={"5044": "KRX 반도체"},
        ticker_to_sector_codes={ticker: ["5044"] for ticker in ticker_list},
        failed_sector_codes={},
    )


def test_dry_run_does_not_touch_krx(monkeypatch, tmp_path):
    monkeypatch.setattr(slow, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(slow, "load_market_configs", lambda market: ({}, _sector_map(), {}, object()))
    monkeypatch.setattr(slow, "_build_cached_universe", lambda entries, *, market: (_universe(), {"snapshot_rows": 2}))
    monkeypatch.setattr(
        slow,
        "ensure_pykrx_transport_compat",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run must not initialize KRX transport")),
    )

    summary = slow.run(_args(execute=False))

    assert summary["status"] == "DRY_RUN"
    assert summary["stopped_reason"] == "dry_run_no_network"
    assert summary["tracked_tickers"] == 2
    assert summary["processed_requests"] == 0


def test_execute_stops_and_persists_state_on_access_denied(monkeypatch, tmp_path):
    monkeypatch.setattr(slow, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(slow, "load_market_configs", lambda market: ({}, _sector_map(), {}, object()))
    monkeypatch.setattr(slow, "_build_cached_universe", lambda entries, *, market: (_universe(), {"snapshot_rows": 2}))
    monkeypatch.setattr(slow, "_get_access_denied_cooldown", lambda: None)
    monkeypatch.setattr(slow, "ensure_pykrx_transport_compat", lambda: None)
    monkeypatch.setitem(sys.modules, "pykrx", SimpleNamespace(stock=SimpleNamespace()))

    def _blocked(*_args, **_kwargs):
        raise RuntimeError("status=403 access denied")

    monkeypatch.setattr(slow, "_collect_one_ticker", _blocked)

    summary = slow.run(_args(execute=True, sectors="export"))

    assert summary["persisted"] is False
    assert summary["completed_tickers"] == 0
    assert summary["failed_tickers"] == 1
    assert "ACCESS_DENIED" in summary["stopped_reason"]
    state_path = tmp_path / "KR" / "20260422" / "5044" / "state.json"
    assert state_path.exists()
    assert "failed_tickers" in state_path.read_text(encoding="utf-8")


def test_execute_with_ticker_limit_spools_without_operational_persist(monkeypatch, tmp_path):
    monkeypatch.setattr(slow, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(slow, "load_market_configs", lambda market: ({}, _sector_map(), {}, object()))
    monkeypatch.setattr(slow, "_build_cached_universe", lambda entries, *, market: (_universe(), {"snapshot_rows": 2}))
    monkeypatch.setattr(slow, "_get_access_denied_cooldown", lambda: None)
    monkeypatch.setattr(slow, "ensure_pykrx_transport_compat", lambda: None)
    monkeypatch.setitem(sys.modules, "pykrx", SimpleNamespace(stock=SimpleNamespace()))
    monkeypatch.setattr(
        slow,
        "write_investor_flow_operational_result",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("limited runs must not update operational cursor")),
    )

    def _one(_stock, *, ticker: str, day: str):
        return pd.DataFrame(
            {
                "trade_date": [pd.Timestamp(day)] * 3,
                "ticker": [ticker] * 3,
                "ticker_name": ["삼성전자"] * 3,
                "investor_type": ["개인", "외국인", "기관합계"],
                "buy_amount": [10, 20, 30],
                "sell_amount": [5, 10, 15],
                "net_buy_amount": [5, 10, 15],
            }
        )

    monkeypatch.setattr(slow, "_collect_one_ticker", _one)

    summary = slow.run(_args(execute=True, sectors="export", max_tickers=1))

    assert summary["persisted"] is False
    assert summary["completed_tickers"] == 1
    assert summary["rows"] == 3
    raw_path = tmp_path / "KR" / "20260422" / "5044" / "raw.parquet"
    assert raw_path.exists()


def test_build_cached_universe_uses_snapshot_without_live_lookup(monkeypatch):
    snapshot = pd.DataFrame(
        {
            "sector_code": ["5044", "5044", "5045"],
            "ticker": ["005930", "000660", "068270"],
            "snapshot_date": ["2026-04-21", "2026-04-21", "2026-04-21"],
        }
    )
    monkeypatch.setattr(slow, "read_latest_sector_constituents_snapshot", lambda codes, *, market: snapshot)

    universe, meta = slow._build_cached_universe(
        [
            slow.SectorEntry(code="5044", name="KRX 반도체", export_sector=True),
            slow.SectorEntry(code="5045", name="KRX 헬스케어", export_sector=False),
        ],
        market="KR",
    )

    assert universe.ticker_to_sector_codes["005930"] == ["5044"]
    assert universe.ticker_to_sector_codes["068270"] == ["5045"]
    assert meta["snapshot_rows"] == 3
