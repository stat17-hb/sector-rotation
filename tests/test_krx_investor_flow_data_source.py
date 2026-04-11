from __future__ import annotations

import pandas as pd

import src.data_sources.krx_investor_flow as flow_source


_SECTOR_MAP = {
    "regimes": {
        "Recovery": {
            "sectors": [
                {"code": "5044", "name": "KRX 반도체"},
                {"code": "5045", "name": "KRX 헬스케어"},
            ]
        }
    }
}


def test_collect_sector_investor_flow_aggregates_current_constituents(monkeypatch):
    monkeypatch.setattr(flow_source, "ensure_pykrx_transport_compat", lambda: None)

    import pykrx.stock as stock

    monkeypatch.setattr(
        stock,
        "get_index_portfolio_deposit_file",
        lambda trade_date, code: ["005930"] if code == "5044" else ["000660"],
    )

    def _net_purchase(trade_date, *_args, **kwargs):
        investor = kwargs["investor"]
        base = {
            "외국인": {"005930": 0.20, "000660": -0.10},
            "기관합계": {"005930": 0.10, "000660": 0.05},
            "개인": {"005930": -0.08, "000660": 0.02},
        }[investor]
        rows = []
        index = []
        for ticker, ratio in base.items():
            rows.append(
                {
                    "종목명": f"Name {ticker}",
                    "매도거래대금": 1000,
                    "매수거래대금": 1000 + int(ratio * 1000),
                    "순매수거래대금": int(ratio * 2000),
                }
            )
            index.append(ticker)
        return pd.DataFrame(rows, index=index)

    monkeypatch.setattr(stock, "get_market_net_purchases_of_equities_by_ticker", _net_purchase)

    raw_frame, sector_frame, summary = flow_source.collect_sector_investor_flow(
        sector_map=_SECTOR_MAP,
        start="20260407",
        end="20260408",
    )

    assert not raw_frame.empty
    assert not sector_frame.empty
    assert set(sector_frame["sector_code"]) == {"5044", "5045"}
    assert "net_flow_ratio" in sector_frame.columns
    assert summary["tracked_sectors"] == 2


def test_run_manual_investor_flow_refresh_persists_to_warehouse(monkeypatch):
    raw_frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-04-08"] * 2),
            "ticker": ["005930", "005930"],
            "ticker_name": ["삼성전자", "삼성전자"],
            "investor_type": ["외국인", "개인"],
            "buy_amount": [1200, 900],
            "sell_amount": [800, 1100],
            "net_buy_amount": [400, -200],
        }
    )
    sector_frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-04-08"] * 2),
            "sector_code": ["5044", "5044"],
            "sector_name": ["KRX 반도체", "KRX 반도체"],
            "investor_type": ["외국인", "개인"],
            "buy_amount": [1200, 900],
            "sell_amount": [800, 1100],
            "net_buy_amount": [400, -200],
            "net_flow_ratio": [0.2, -0.1],
        }
    )
    monkeypatch.setattr(
        flow_source,
        "collect_sector_investor_flow",
        lambda **kwargs: (
            raw_frame,
            sector_frame,
            {
                "status": "LIVE",
                "provider": flow_source.FLOW_PROVIDER,
                "requested_start": "20260101",
                "requested_end": "20260408",
                "coverage_complete": True,
                "failed_days": [],
                "failed_codes": {},
                "predicted_requests": 10,
                "processed_requests": 10,
                "rows": 2,
            },
        ),
    )

    (status, cached), summary = flow_source.run_manual_investor_flow_refresh(
        sector_map=_SECTOR_MAP,
        end_date_str="20260408",
    )

    assert status == "LIVE"
    assert summary["coverage_complete"] is True
    assert not cached.empty
    cached_status, cached_frame = flow_source.load_sector_investor_flow(
        sector_map=_SECTOR_MAP,
        start="20260101",
        end="20260408",
    )
    assert cached_status == "CACHED"
    assert not cached_frame.empty
