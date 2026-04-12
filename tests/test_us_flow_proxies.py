from __future__ import annotations

import sys
import types
from io import BytesIO

import pandas as pd

import src.data_sources.us_ownership_context as ownership
import src.data_sources.us_flow_proxies as us_flow


def test_fetch_sector_flow_history_normalizes_close_and_volume(monkeypatch):
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    columns = pd.MultiIndex.from_product([["XLF", "XLK"], ["Close", "Volume"]])
    raw = pd.DataFrame(
        [
            [40.0, 1_000_000, 200.0, 500_000],
            [41.0, 1_200_000, 201.0, 600_000],
        ],
        index=index,
        columns=columns,
    )
    fake_module = types.SimpleNamespace(download=lambda **kwargs: raw)
    monkeypatch.setitem(sys.modules, "yfinance", fake_module)

    frame = us_flow.fetch_sector_flow_history(["XLF", "XLK"], "20240102", "20240103")

    assert set(frame["sector_code"].astype(str).unique()) == {"XLF", "XLK"}
    assert "dollar_volume" in frame.columns
    assert float(frame.loc[pd.Timestamp("2024-01-02")]["dollar_volume"].max()) == 100_000_000.0


def test_fetch_ssga_fund_snapshot_parses_core_metrics():
    class _Response:
        status_code = 200
        url = "https://www.ssga.com/us/en/intermediary/etfs/state-street-financial-select-sector-spdr-etf-xlf"
        text = """
        <html>
          <body>
            <h2>Fund Net Cash Amount as of Apr 09 2026</h2>
            <div>Net Cash Amount</div><div>$92,163,529.64</div>
            <h2>Fund Net Asset Value as of Apr 09 2026</h2>
            <div>NAV</div><div>$51.32</div>
            <div>Shares Outstanding</div><div>985.65 M</div>
            <div>Assets Under Management</div><div>$50,588.09 M</div>
          </body>
        </html>
        """

        def raise_for_status(self) -> None:
            return None

    class _Session:
        def get(self, *args, **kwargs):
            return _Response()

    snapshot = us_flow.fetch_ssga_fund_snapshot("XLF", session=_Session())

    assert snapshot["sector_code"] == "XLF"
    assert snapshot["snapshot_date"] == "Apr 09 2026"
    assert snapshot["nav"] == 51.32
    assert snapshot["shares_outstanding"] == 985_650_000.0
    assert snapshot["assets_under_management"] == 50_588_090_000.0
    assert snapshot["net_cash_amount"] == 92_163_529.64


def test_fetch_ssga_fund_snapshot_falls_back_to_xlsx():
    workbook = BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                ["Fund Net Cash Amount as of Apr 09 2026", ""],
                ["Net Cash Amount", "$92,163,529.64"],
                ["Fund Net Asset Value as of Apr 09 2026", ""],
                ["NAV", "$51.32"],
                ["Shares Outstanding", "985.65 M"],
                ["Assets Under Management", "$50,588.09 M"],
            ]
        ).to_excel(writer, index=False, header=False)

    class _HtmlResponse:
        status_code = 200
        url = "https://www.ssga.com/us/en/intermediary/etfs/state-street-financial-select-sector-spdr-etf-xlf"
        text = """
        <html>
          <body>
            <a href="/library-content/products/fund-data/etfs/us/fund-data-xlf.xlsx">Fund Data</a>
          </body>
        </html>
        """

        def raise_for_status(self) -> None:
            return None

    class _WorkbookResponse:
        status_code = 200
        url = "https://www.ssga.com/library-content/products/fund-data/etfs/us/fund-data-xlf.xlsx"
        content = workbook.getvalue()

        def raise_for_status(self) -> None:
            return None

    class _Session:
        def get(self, url, *args, **kwargs):
            if str(url).endswith(".xlsx"):
                return _WorkbookResponse()
            return _HtmlResponse()

    snapshot = us_flow.fetch_ssga_fund_snapshot("XLF", session=_Session())

    assert snapshot["sector_code"] == "XLF"
    assert snapshot["snapshot_url"].endswith(".xlsx")
    assert snapshot["snapshot_date"] == "Apr 09 2026"
    assert snapshot["nav"] == 51.32
    assert snapshot["shares_outstanding"] == 985_650_000.0


def test_fetch_ssga_fund_profile_extracts_cusip_and_top_holdings():
    class _Response:
        status_code = 200
        url = "https://www.ssga.com/us/en/intermediary/etfs/state-street-financial-select-sector-spdr-etf-xlf"
        text = """
        <html>
          <body>
            <div>NYSE ARCA Dec 22 1998 USD XLF 81369Y605 US81369Y6059</div>
            <h2>Fund Net Asset Value as of Apr 09 2026</h2>
            <div>NAV</div><div>$51.32</div>
            <div>Shares Outstanding</div><div>985.65 M</div>
            <table>
              <thead><tr><th>Name</th><th>Shares Held</th><th>Weight</th></tr></thead>
              <tbody>
                <tr><td>BERKSHIRE HATHAWAY INC CL B</td><td>12,663,705</td><td>12.04%</td></tr>
                <tr><td>JPMORGAN CHASE + CO</td><td>18,619,876</td><td>11.37%</td></tr>
              </tbody>
            </table>
          </body>
        </html>
        """

        def raise_for_status(self) -> None:
            return None

    class _Session:
        def get(self, *args, **kwargs):
            return _Response()

    profile = us_flow.fetch_ssga_fund_profile("XLF", session=_Session())

    assert profile["cusip"] == "81369Y605"
    assert len(profile["top_holdings"]) == 2
    assert profile["top_holdings"][0]["name"] == "BERKSHIRE HATHAWAY INC CL B"


def test_summarize_us_flow_proxies_uses_history_and_snapshot():
    dates = pd.date_range("2024-01-02", periods=25, freq="B")
    history = pd.DataFrame(
        {
            "sector_code": ["XLF"] * len(dates),
            "close": [40.0 + idx for idx in range(len(dates))],
            "volume": [1_000_000.0] * 20 + [3_000_000.0] * 5,
            "dollar_volume": [40_000_000.0] * 20 + [150_000_000.0] * 5,
        },
        index=dates,
    )
    frame = us_flow.summarize_us_flow_proxies(
        history,
        sector_map={
            "regimes": {
                "Expansion": {
                    "sectors": [{"code": "XLF", "name": "Financials"}],
                }
            }
        },
        snapshots={
            "XLF": {
                "snapshot_date": "Apr 09 2026",
                "nav": 51.32,
                "shares_outstanding": 985_650_000.0,
                "assets_under_management": 50_588_090_000.0,
                "net_cash_amount": 92_163_529.64,
                "snapshot_url": "https://example.test/xlf",
            }
        },
    )

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["sector_name"] == "Financials"
    assert row["activity_state"] == "elevated"
    assert row["activity_zscore"] > 0
    assert row["shares_outstanding"] == 985_650_000.0


def test_load_us_flow_proxies_degrades_when_snapshot_fetch_fails(monkeypatch):
    dates = pd.date_range("2024-01-02", periods=25, freq="B")
    history = pd.DataFrame(
        {
            "sector_code": ["XLF"] * len(dates),
            "close": [40.0] * len(dates),
            "volume": [1_000_000.0] * len(dates),
            "dollar_volume": [40_000_000.0] * len(dates),
        },
        index=dates,
    )
    monkeypatch.setattr(us_flow, "fetch_sector_flow_history", lambda *args, **kwargs: history)
    monkeypatch.setattr(
        us_flow,
        "fetch_ssga_fund_profile",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("snapshot unavailable")),
    )
    monkeypatch.setattr(
        "src.data_sources.us_ownership_context.load_us_ownership_context",
        lambda *args, **kwargs: {"errors": {}},
    )

    status, frame, detail = us_flow.load_us_flow_proxies(
        sector_map={"regimes": {"Expansion": {"sectors": [{"code": "XLF", "name": "Financials"}]}}},
        start="20240102",
        end="20240205",
    )

    assert status == "LIVE"
    assert len(frame) == 1
    assert "XLF" in detail["snapshot_failures"]
    assert pd.isna(frame.iloc[0]["shares_outstanding"])


def test_load_us_flow_proxies_marks_missing_tickers_incomplete(monkeypatch):
    dates = pd.date_range("2024-01-02", periods=25, freq="B")
    history = pd.DataFrame(
        {
            "sector_code": ["XLF"] * len(dates),
            "close": [40.0] * len(dates),
            "volume": [1_000_000.0] * len(dates),
            "dollar_volume": [40_000_000.0] * len(dates),
        },
        index=dates,
    )
    monkeypatch.setattr(us_flow, "fetch_sector_flow_history", lambda *args, **kwargs: history)
    monkeypatch.setattr(
        us_flow,
        "fetch_ssga_fund_profile",
        lambda ticker, **kwargs: {"sector_code": ticker, "sector_name": ticker, "snapshot_date": "", "nav": float("nan"), "cusip": "", "top_holdings": []},
    )
    monkeypatch.setattr(
        "src.data_sources.us_ownership_context.load_us_ownership_context",
        lambda *args, **kwargs: {"errors": {}},
    )

    status, frame, detail = us_flow.load_us_flow_proxies(
        sector_map={
            "regimes": {
                "Expansion": {
                    "sectors": [
                        {"code": "XLF", "name": "Financials"},
                        {"code": "XLK", "name": "Technology"},
                    ]
                }
            }
        },
        start="20240102",
        end="20240205",
    )

    assert status == "LIVE"
    assert len(frame) == 1
    assert detail["coverage_complete"] is False
    assert detail["missing_tickers"] == ["XLK"]


def test_fetch_ici_weekly_etf_flows_parses_official_table():
    class _Response:
        status_code = 200
        url = "https://www.ici.org/research/stats/etf_flows"
        text = """
        <html><body>
          <table>
            <thead><tr><th></th><th>3/4/2026</th><th>2/25/2026</th></tr></thead>
            <tbody>
              <tr><td>Equity</td><td>12154</td><td>27334</td></tr>
              <tr><td>Domestic</td><td>-725</td><td>8279</td></tr>
              <tr><td>Bond</td><td>14234</td><td>29591</td></tr>
            </tbody>
          </table>
        </body></html>
        """

        def raise_for_status(self) -> None:
            return None

    class _Session:
        def get(self, *args, **kwargs):
            return _Response()

    result = ownership.fetch_ici_weekly_etf_flows(session=_Session())

    assert result["as_of"] == "3/4/2026"
    table = result["table"]
    assert list(table["category"]) == ["Equity", "Domestic", "Bond"]
    assert float(table.loc[0, "value"]) == 12154.0


def test_fetch_latest_13f_sector_etf_positions_parses_sec_zip():
    page_html = """
    <html><body>
      <a href="/files/13f/latest_13f.zip">2025 December 2026 January February 13F</a>
    </body></html>
    """
    raw_zip = BytesIO()
    with __import__("zipfile").ZipFile(raw_zip, "w") as zf:
        zf.writestr(
            "INFOTABLE.tsv",
            "ACCESSION_NUMBER\tCUSIP\tVALUE\tSSHPRNAMT\n"
            "0001\t81369Y605\t1000\t20000\n"
            "0002\t81369Y605\t2000\t30000\n"
            "0003\t81369Y803\t500\t10000\n",
        )
    zip_payload = raw_zip.getvalue()

    class _HtmlResponse:
        status_code = 200
        url = ownership.SEC_13F_DATASETS_URL
        text = page_html

        def raise_for_status(self) -> None:
            return None

    class _ZipResponse:
        status_code = 200
        url = "https://www.sec.gov/files/13f/latest_13f.zip"
        content = zip_payload

        def raise_for_status(self) -> None:
            return None

    class _Session:
        def get(self, url, *args, **kwargs):
            if str(url).endswith(".zip"):
                return _ZipResponse()
            return _HtmlResponse()

    result = ownership.fetch_latest_13f_sector_etf_positions(
        [
            {"sector_code": "XLF", "sector_name": "Financials", "cusip": "81369Y605"},
            {"sector_code": "XLK", "sector_name": "Technology", "cusip": "81369Y803"},
        ],
        session=_Session(),
    )

    table = result["table"]
    assert list(table["sector_code"]) == ["XLF", "XLK"]
    assert float(table.loc[0, "manager_value_total_usd"]) == 3_000_000.0
    assert int(table.loc[0, "filing_count"]) == 2


def test_fetch_recent_13dg_sector_events_parses_submissions_json():
    company_payload = {
        "0": {"cik_str": 1001, "ticker": "JPM", "title": "JPMORGAN CHASE & CO"},
        "1": {"cik_str": 1002, "ticker": "BRK.B", "title": "BERKSHIRE HATHAWAY INC"},
    }
    submissions_payload = {
        "filings": {
            "recent": {
                "form": ["SC 13G", "10-K", "SC 13D/A"],
                "filingDate": ["2026-03-01", "2026-02-01", "2026-01-15"],
            }
        }
    }

    class _CompanyResponse:
        status_code = 200
        url = ownership.SEC_COMPANY_TICKERS_URL
        text = __import__("json").dumps(company_payload)

        def raise_for_status(self) -> None:
            return None

    class _SubmissionResponse:
        status_code = 200
        url = ownership.SEC_SUBMISSIONS_URL.format(cik="0000001001")

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return submissions_payload

    class _Session:
        def get(self, url, *args, **kwargs):
            if "company_tickers" in str(url):
                return _CompanyResponse()
            return _SubmissionResponse()

    result = ownership.fetch_recent_13dg_sector_events(
        [
            {
                "sector_code": "XLF",
                "sector_name": "Financials",
                "top_holdings": [
                    {"name": "JPMORGAN CHASE + CO"},
                    {"name": "UNKNOWN HOLDING"},
                ],
            }
        ],
        session=_Session(),
        trailing_days=180,
    )

    table = result["table"]
    assert int(table.loc[0, "matched_top_holdings"]) == 1
    assert int(table.loc[0, "recent_13dg_events"]) == 2
