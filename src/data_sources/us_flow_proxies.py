"""
US sector ETF flow-proxy loader.

This module intentionally does not claim to reproduce Korean-style participant
flow. It surfaces a public-data proxy layer built from:
- sector ETF price/volume history from yfinance
- current official fund snapshot fields from SSGA product pages
"""
from __future__ import annotations

from io import BytesIO
from datetime import date, timedelta
import html
import logging
import re
from typing import Any, Literal
from urllib.parse import urljoin

import pandas as pd
import requests


logger = logging.getLogger(__name__)

DataStatus = Literal["LIVE", "SAMPLE"]
MARKET_ID = "US"
DEFAULT_SHORT_WINDOW = 5
DEFAULT_LONG_WINDOW = 20
SSGA_MAINFUND_URL = "https://www.ssga.com/mainfund/{ticker}"
REQUEST_TIMEOUT_SEC = 15
REQUEST_HEADERS = {
    "User-Agent": "sector-rotation/1.0 (+https://github.com/)",
}

def _normalize_market_date(value: str) -> date:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) != 8:
        raise ValueError(f"Invalid market date: {value!r}")
    return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))


def _sector_entries(sector_map: dict[str, Any]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for regime_data in (sector_map or {}).get("regimes", {}).values():
        for sector in regime_data.get("sectors", []):
            code = str(sector.get("code", "")).strip().upper()
            if not code or code in seen:
                continue
            seen.add(code)
            entries.append(
                {
                    "sector_code": code,
                    "sector_name": str(sector.get("name", code)).strip() or code,
                }
            )
    return entries


def _to_long_history(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError("yfinance returned no rows for US flow proxy history")

    normalized = [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
    if not normalized:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(-1) or "Volume" not in raw.columns.get_level_values(-1):
            raise ValueError("yfinance response did not include Close/Volume fields")
        close = raw.xs("Close", axis=1, level=-1)
        volume = raw.xs("Volume", axis=1, level=-1)
    else:
        if "Close" not in raw.columns or "Volume" not in raw.columns:
            raise ValueError("yfinance response did not include Close/Volume columns")
        close = raw[["Close"]].rename(columns={"Close": normalized[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": normalized[0]})

    close.index = pd.DatetimeIndex(close.index).normalize()
    volume.index = pd.DatetimeIndex(volume.index).normalize()
    close = close.sort_index().dropna(how="all")
    volume = volume.sort_index().dropna(how="all")

    rows: list[dict[str, object]] = []
    for ticker in normalized:
        close_series = pd.to_numeric(close.get(ticker), errors="coerce")
        volume_series = pd.to_numeric(volume.get(ticker), errors="coerce")
        if close_series is None or volume_series is None:
            continue
        combined = pd.concat(
            {"close": close_series, "volume": volume_series},
            axis=1,
        ).dropna()
        if combined.empty:
            continue
        combined["dollar_volume"] = combined["close"] * combined["volume"]
        for trade_date, item in combined.iterrows():
            rows.append(
                {
                    "trade_date": pd.Timestamp(trade_date).normalize(),
                    "sector_code": ticker,
                    "close": float(item["close"]),
                    "volume": float(item["volume"]),
                    "dollar_volume": float(item["dollar_volume"]),
                }
            )

    if not rows:
        raise ValueError("US flow proxy history was empty after Close/Volume normalization")

    frame = pd.DataFrame(rows).set_index("trade_date").sort_index()
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def fetch_sector_flow_history(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    normalized = [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
    if not normalized:
        return pd.DataFrame()

    start_date = _normalize_market_date(start)
    end_date = _normalize_market_date(end)
    end_exclusive = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

    import yfinance as yf  # type: ignore[import]

    raw = yf.download(
        tickers=normalized,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_exclusive,
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=False,
    )
    return _to_long_history(raw, normalized)


def _html_to_text(raw_html: str) -> str:
    stripped = re.sub(r"<script\b[^>]*>.*?</script>", " ", raw_html, flags=re.IGNORECASE | re.DOTALL)
    stripped = re.sub(r"<style\b[^>]*>.*?</style>", " ", stripped, flags=re.IGNORECASE | re.DOTALL)
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    stripped = html.unescape(stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def _parse_scaled_number(raw_value: str) -> float:
    cleaned = str(raw_value or "").strip().replace("$", "").replace(",", "")
    if not cleaned:
        return float("nan")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(?:\s*([KMBT]))?", cleaned, flags=re.IGNORECASE)
    if not match:
        return float("nan")
    value = float(match.group(1))
    suffix = (match.group(2) or "").upper()
    multiplier = {
        "": 1.0,
        "K": 1_000.0,
        "M": 1_000_000.0,
        "B": 1_000_000_000.0,
        "T": 1_000_000_000_000.0,
    }[suffix]
    return value * multiplier


def _extract_metric(text: str, label: str) -> float:
    match = re.search(
        rf"{re.escape(label)}\s+\$?([0-9,]+(?:\.[0-9]+)?(?:\s*[KMBT])?)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return float("nan")
    return _parse_scaled_number(match.group(1))


def _extract_metric_date(text: str, label: str) -> str:
    match = re.search(
        rf"{re.escape(label)}\s+as of\s+([A-Za-z]{{3}}\s+\d{{2}}\s+\d{{4}})",
        text,
        flags=re.IGNORECASE,
    )
    return str(match.group(1)) if match else ""


def _extract_xlsx_links(raw_html: str, base_url: str) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r'href=["\']([^"\']+\.xlsx(?:\?[^"\']*)?)["\']', raw_html, flags=re.IGNORECASE):
        href = str(match.group(1)).strip()
        if not href:
            continue
        resolved = urljoin(base_url, href)
        if resolved in seen:
            continue
        seen.add(resolved)
        links.append(resolved)
    return links


def _workbook_to_text(content: bytes) -> str:
    workbook = pd.read_excel(BytesIO(content), sheet_name=None, header=None, engine="openpyxl")
    chunks: list[str] = []
    for sheet in workbook.values():
        if sheet is None or sheet.empty:
            continue
        normalized = sheet.fillna("").astype(str)
        for row in normalized.itertuples(index=False):
            text = " ".join(item.strip() for item in row if str(item).strip())
            if text:
                chunks.append(text)
    return " ".join(chunks)


def _snapshot_from_text(text: str, *, sector_code: str, snapshot_url: str) -> dict[str, Any]:
    return {
        "sector_code": sector_code,
        "snapshot_url": snapshot_url,
        "snapshot_date": _extract_metric_date(text, "Fund Net Asset Value"),
        "nav": _extract_metric(text, "NAV"),
        "shares_outstanding": _extract_metric(text, "Shares Outstanding"),
        "assets_under_management": _extract_metric(text, "Assets Under Management"),
        "net_cash_amount": _extract_metric(text, "Net Cash Amount"),
    }


def _extract_cusip(text: str, *, ticker: str) -> str:
    match = re.search(
        rf"\b{re.escape(str(ticker).upper())}\b\s+([0-9A-Z]{{9}})\s+US[0-9A-Z]{{10}}",
        text,
        flags=re.IGNORECASE,
    )
    return str(match.group(1)).upper() if match else ""


def _extract_top_holdings(raw_html: str) -> list[dict[str, Any]]:
    tables: list[pd.DataFrame] = []
    for table_html in re.findall(r"<table\b[^>]*>.*?</table>", raw_html, flags=re.IGNORECASE | re.DOTALL):
        rows: list[list[str]] = []
        for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table_html, flags=re.IGNORECASE | re.DOTALL):
            cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row_html, flags=re.IGNORECASE | re.DOTALL)
            if not cells:
                continue
            normalized_cells = [
                re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(str(cell)))).strip()
                for cell in cells
            ]
            rows.append(normalized_cells)
        if len(rows) >= 2:
            header = rows[0]
            body = rows[1:]
            if all(len(item) == len(header) for item in body):
                tables.append(pd.DataFrame(body, columns=header))

    for normalized in tables:
        columns = [str(column).strip() for column in normalized.columns]
        if not any("Name" in column for column in columns):
            continue
        if not any("Shares Held" in column for column in columns):
            continue
        if not any("Weight" in column for column in columns):
            continue
        normalized.columns = columns
        name_col = next(column for column in normalized.columns if "Name" in column)
        shares_col = next(column for column in normalized.columns if "Shares Held" in column)
        weight_col = next(column for column in normalized.columns if "Weight" in column)
        rows: list[dict[str, Any]] = []
        for row in normalized[[name_col, shares_col, weight_col]].itertuples(index=False):
            name = str(row[0]).strip()
            if not name:
                continue
            rows.append(
                {
                    "name": name,
                    "shares_held": _parse_scaled_number(str(row[1])),
                    "weight_pct": _parse_scaled_number(str(row[2]).replace("%", "")),
                }
            )
        if rows:
            return rows
    return []


def _snapshot_has_core_metrics(snapshot: dict[str, Any]) -> bool:
    return any(
        pd.notna(snapshot.get(field))
        for field in ("nav", "shares_outstanding", "assets_under_management", "net_cash_amount")
    )


def fetch_ssga_fund_snapshot(ticker: str, *, session: requests.Session | None = None) -> dict[str, Any]:
    normalized = str(ticker or "").strip().upper()
    if not normalized:
        raise ValueError("Ticker is required for SSGA snapshot fetch")

    http = session or requests.Session()
    response = http.get(
        SSGA_MAINFUND_URL.format(ticker=normalized.lower()),
        timeout=REQUEST_TIMEOUT_SEC,
        headers=REQUEST_HEADERS,
        allow_redirects=True,
    )
    response.raise_for_status()

    page_url = str(response.url)
    snapshot = _snapshot_from_text(_html_to_text(response.text), sector_code=normalized, snapshot_url=page_url)
    if _snapshot_has_core_metrics(snapshot):
        return snapshot

    for workbook_url in _extract_xlsx_links(response.text, page_url):
        try:
            workbook_response = http.get(
                workbook_url,
                timeout=REQUEST_TIMEOUT_SEC,
                headers=REQUEST_HEADERS,
                allow_redirects=True,
            )
            workbook_response.raise_for_status()
            workbook_snapshot = _snapshot_from_text(
                _workbook_to_text(workbook_response.content),
                sector_code=normalized,
                snapshot_url=workbook_url,
            )
            if _snapshot_has_core_metrics(workbook_snapshot):
                return workbook_snapshot
        except Exception as exc:
            logger.debug("SSGA workbook snapshot fallback failed for %s (%s): %s", normalized, workbook_url, exc)

    return snapshot


def fetch_ssga_fund_profile(ticker: str, *, session: requests.Session | None = None) -> dict[str, Any]:
    normalized = str(ticker or "").strip().upper()
    if not normalized:
        raise ValueError("Ticker is required for SSGA profile fetch")

    http = session or requests.Session()
    response = http.get(
        SSGA_MAINFUND_URL.format(ticker=normalized.lower()),
        timeout=REQUEST_TIMEOUT_SEC,
        headers=REQUEST_HEADERS,
        allow_redirects=True,
    )
    response.raise_for_status()
    page_url = str(response.url)
    text = _html_to_text(response.text)
    snapshot = _snapshot_from_text(text, sector_code=normalized, snapshot_url=page_url)
    if not _snapshot_has_core_metrics(snapshot):
        for workbook_url in _extract_xlsx_links(response.text, page_url):
            try:
                workbook_response = http.get(
                    workbook_url,
                    timeout=REQUEST_TIMEOUT_SEC,
                    headers=REQUEST_HEADERS,
                    allow_redirects=True,
                )
                workbook_response.raise_for_status()
                workbook_snapshot = _snapshot_from_text(
                    _workbook_to_text(workbook_response.content),
                    sector_code=normalized,
                    snapshot_url=workbook_url,
                )
                if _snapshot_has_core_metrics(workbook_snapshot):
                    snapshot = workbook_snapshot
                    break
            except Exception as exc:
                logger.debug("SSGA workbook profile fallback failed for %s (%s): %s", normalized, workbook_url, exc)

    return {
        **snapshot,
        "cusip": _extract_cusip(text, ticker=normalized),
        "top_holdings": _extract_top_holdings(response.text),
    }


def _classify_activity_proxy(series: pd.Series, *, short_window: int, long_window: int) -> tuple[str, float, float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < long_window:
        return "unavailable", 0.0, float("nan"), float("nan")

    short_mean = float(values.tail(short_window).mean())
    long_sample = values.tail(long_window)
    long_mean = float(long_sample.mean())
    long_std = float(long_sample.std(ddof=0))
    zscore = 0.0 if long_std == 0 else (short_mean - long_mean) / long_std
    if zscore >= 0.75:
        state = "elevated"
    elif zscore <= -0.75:
        state = "subdued"
    else:
        state = "normal"
    return state, float(zscore), short_mean, long_mean


def summarize_us_flow_proxies(
    history_frame: pd.DataFrame,
    *,
    sector_map: dict[str, Any],
    snapshots: dict[str, dict[str, Any]] | None = None,
    short_window: int = DEFAULT_SHORT_WINDOW,
    long_window: int = DEFAULT_LONG_WINDOW,
) -> pd.DataFrame:
    if history_frame.empty:
        return pd.DataFrame()

    entries = {item["sector_code"]: item["sector_name"] for item in _sector_entries(sector_map)}
    snapshots = dict(snapshots or {})
    rows: list[dict[str, object]] = []
    expected_tickers = set(entries)
    realized_tickers: set[str] = set()

    for sector_code, sector_rows in history_frame.groupby(history_frame["sector_code"].astype(str)):
        ordered = sector_rows.sort_index()
        latest = ordered.iloc[-1]
        realized_tickers.add(str(sector_code))
        activity_state, activity_zscore, short_mean, long_mean = _classify_activity_proxy(
            ordered["dollar_volume"],
            short_window=short_window,
            long_window=long_window,
        )
        snapshot = snapshots.get(str(sector_code), {})
        rows.append(
            {
                "trade_date": pd.Timestamp(ordered.index.max()).normalize(),
                "sector_code": str(sector_code),
                "sector_name": entries.get(str(sector_code), str(sector_code)),
                "activity_state": activity_state,
                "activity_zscore": float(activity_zscore),
                "activity_reason": (
                    f"dollar_volume_short={short_mean:,.0f}, "
                    f"dollar_volume_long={long_mean:,.0f}, z={activity_zscore:+.2f}"
                    if pd.notna(short_mean) and pd.notna(long_mean)
                    else "insufficient dollar-volume history"
                ),
                "close": float(latest["close"]),
                "volume": float(latest["volume"]),
                "dollar_volume": float(latest["dollar_volume"]),
                "dollar_volume_short_mean": float(short_mean) if pd.notna(short_mean) else float("nan"),
                "dollar_volume_long_mean": float(long_mean) if pd.notna(long_mean) else float("nan"),
                "snapshot_date": str(snapshot.get("snapshot_date", "")),
                "nav": float(snapshot.get("nav", float("nan"))),
                "shares_outstanding": float(snapshot.get("shares_outstanding", float("nan"))),
                "assets_under_management": float(snapshot.get("assets_under_management", float("nan"))),
                "net_cash_amount": float(snapshot.get("net_cash_amount", float("nan"))),
                "snapshot_url": str(snapshot.get("snapshot_url", "")),
                "source": "YFINANCE+SSGA",
            }
        )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).set_index("trade_date").sort_index()
    frame.index = pd.DatetimeIndex(frame.index)
    frame.attrs["expected_tickers"] = sorted(expected_tickers)
    frame.attrs["realized_tickers"] = sorted(realized_tickers)
    return frame


def _sample_proxy_frame(sector_map: dict[str, Any], end: str) -> pd.DataFrame:
    end_ts = pd.Timestamp(_normalize_market_date(end))
    rows: list[dict[str, object]] = []
    for sector in _sector_entries(sector_map):
        rows.append(
            {
                "trade_date": end_ts,
                "sector_code": sector["sector_code"],
                "sector_name": sector["sector_name"],
                "activity_state": "unavailable",
                "activity_zscore": 0.0,
                "activity_reason": "sample fallback",
                "close": float("nan"),
                "volume": float("nan"),
                "dollar_volume": float("nan"),
                "dollar_volume_short_mean": float("nan"),
                "dollar_volume_long_mean": float("nan"),
                "snapshot_date": "",
                "nav": float("nan"),
                "shares_outstanding": float("nan"),
                "assets_under_management": float("nan"),
                "net_cash_amount": float("nan"),
                "snapshot_url": "",
                "source": "SAMPLE",
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index("trade_date")
    frame.index = pd.DatetimeIndex(frame.index)
    frame.attrs["expected_tickers"] = [sector["sector_code"] for sector in _sector_entries(sector_map)]
    frame.attrs["realized_tickers"] = []
    return frame


def load_us_flow_proxies(
    *,
    sector_map: dict[str, Any],
    start: str,
    end: str,
    session: requests.Session | None = None,
    short_window: int = DEFAULT_SHORT_WINDOW,
    long_window: int = DEFAULT_LONG_WINDOW,
) -> tuple[DataStatus, pd.DataFrame, dict[str, Any]]:
    entries = _sector_entries(sector_map)
    tickers = [item["sector_code"] for item in entries]
    if not tickers:
        return "SAMPLE", pd.DataFrame(), {"provider": "YFINANCE+SSGA", "watermark_key": str(end)}

    try:
        history_frame = fetch_sector_flow_history(tickers, start, end)
    except Exception as exc:
        logger.warning("US flow proxy history fetch failed: %s", exc)
        sample = _sample_proxy_frame(sector_map, end)
        return (
            "SAMPLE",
            sample,
            {
                "provider": "YFINANCE+SSGA",
                "watermark_key": str(end),
                "coverage_complete": False,
                "snapshot_failures": {},
                "history_error": str(exc),
            },
        )

    snapshot_failures: dict[str, str] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    sector_profiles: list[dict[str, Any]] = []
    for ticker in tickers:
        try:
            profile = fetch_ssga_fund_profile(ticker, session=session)
            snapshots[ticker] = profile
            sector_profiles.append(
                {
                    "sector_code": str(profile.get("sector_code", ticker)),
                    "sector_name": next((item["sector_name"] for item in entries if item["sector_code"] == ticker), ticker),
                    "cusip": str(profile.get("cusip", "")),
                    "top_holdings": list(profile.get("top_holdings", [])),
                    "snapshot_url": str(profile.get("snapshot_url", "")),
                }
            )
        except Exception as exc:
            snapshot_failures[ticker] = str(exc)

    frame = summarize_us_flow_proxies(
        history_frame,
        sector_map=sector_map,
        snapshots=snapshots,
        short_window=short_window,
        long_window=long_window,
    )
    expected_tickers = set(frame.attrs.get("expected_tickers", tickers))
    realized_tickers = set(frame.attrs.get("realized_tickers", []))
    missing_tickers = sorted(expected_tickers - realized_tickers)
    frame.attrs["missing_tickers"] = missing_tickers
    latest_date = frame.index.max().strftime("%Y%m%d") if not frame.empty else str(end)
    detail = {
        "provider": "YFINANCE+SSGA",
        "watermark_key": latest_date,
        "coverage_complete": bool(not frame.empty and latest_date == str(end) and not missing_tickers),
        "snapshot_failures": snapshot_failures,
        "missing_tickers": missing_tickers,
        "history_rows": int(len(history_frame)),
        "proxy_layer": "wrapper_flow",
        "sector_profiles": sector_profiles,
    }
    try:
        from src.data_sources.us_ownership_context import load_us_ownership_context

        detail["ownership_context"] = load_us_ownership_context(sector_profiles, session=session)
    except Exception as exc:
        detail["ownership_context"] = {"errors": {"load": str(exc)}}
    if frame.empty:
        return "SAMPLE", _sample_proxy_frame(sector_map, end), detail
    return "LIVE", frame, detail


__all__ = [
    "MARKET_ID",
    "fetch_sector_flow_history",
    "fetch_ssga_fund_profile",
    "fetch_ssga_fund_snapshot",
    "load_us_flow_proxies",
    "summarize_us_flow_proxies",
]
