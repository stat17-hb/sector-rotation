"""
Official US ownership / short-context loaders for the dashboard.

These layers complement the existing ETF activity proxy but intentionally do
not claim to be participant-segmented cash-equity flow.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
import html
import json
import logging
import re
from typing import Any
from urllib.parse import urljoin
from zipfile import ZipFile

import pandas as pd
import requests


logger = logging.getLogger(__name__)

ICI_ETF_FLOWS_URL = "https://www.ici.org/research/stats/etf_flows"
SEC_13F_DATASETS_URL = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FORM_SHO_RULE_URL = "https://www.sec.gov/newsroom/press-releases/2025-37"
REQUEST_TIMEOUT_SEC = 20
SEC_HEADERS = {
    "User-Agent": "sector-rotation/1.0 (local research client)",
    "Accept-Encoding": "gzip, deflate",
}
ICI_HEADERS = {
    "User-Agent": "sector-rotation/1.0",
}
RECENT_13DG_DAYS = 180
TOP_HOLDINGS_LIMIT = 5
THIRTEEN_D_FORMS = {"SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"}


def _get(url: str, *, session: requests.Session | None = None, headers: dict[str, str] | None = None) -> requests.Response:
    http = session or requests.Session()
    response = http.get(
        url,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SEC,
        allow_redirects=True,
    )
    response.raise_for_status()
    return response


def _read_html_tables(raw_html: str) -> list[pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    for table_html in re.findall(r"<table\b[^>]*>.*?</table>", raw_html, flags=re.IGNORECASE | re.DOTALL):
        rows: list[list[str]] = []
        for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table_html, flags=re.IGNORECASE | re.DOTALL):
            cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row_html, flags=re.IGNORECASE | re.DOTALL)
            if not cells:
                continue
            normalized = [
                re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(str(cell)))).strip()
                for cell in cells
            ]
            rows.append(normalized)
        if len(rows) >= 2:
            header = rows[0]
            body = rows[1:]
            if all(len(item) == len(header) for item in body):
                tables.append(pd.DataFrame(body, columns=header))
    return tables


def _normalize_company_name(value: str) -> str:
    cleaned = str(value or "").upper().replace("+", " AND ").replace("&", " AND ")
    cleaned = re.sub(r"[^A-Z0-9 ]+", " ", cleaned)
    tokens = [token for token in cleaned.split() if token]
    stop = {
        "INC",
        "INCORPORATED",
        "CORP",
        "CORPORATION",
        "CO",
        "COMPANY",
        "HOLDINGS",
        "HLDGS",
        "GROUP",
        "PLC",
        "NV",
        "SA",
        "SE",
        "CL",
        "CLASS",
        "SHARES",
        "SHARE",
        "COMMON",
        "COM",
        "ORDINARY",
        "ORD",
        "NEW",
    }
    return " ".join(token for token in tokens if token not in stop)


def fetch_ici_weekly_etf_flows(session: requests.Session | None = None) -> dict[str, Any]:
    response = _get(ICI_ETF_FLOWS_URL, session=session, headers=ICI_HEADERS)
    tables = _read_html_tables(response.text)
    if not tables:
        raise ValueError("ICI weekly ETF flows page did not contain a readable table")

    flow_table = next(
        (
            table
            for table in tables
            if any(str(value).strip() == "Equity" for value in table.iloc[:, 0].tolist())
        ),
        None,
    )
    if flow_table is None or flow_table.shape[1] < 2:
        raise ValueError("ICI weekly ETF flows table was not found")

    latest_column = str(flow_table.columns[1])
    normalized = flow_table.iloc[:, :2].copy()
    normalized.columns = ["category", "value"]
    normalized["category"] = normalized["category"].astype(str).str.strip()
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    normalized = normalized.dropna(subset=["value"])
    normalized = normalized[normalized["category"] != ""].reset_index(drop=True)

    return {
        "as_of": latest_column,
        "source_url": str(response.url),
        "table": normalized,
    }


def fetch_latest_13f_dataset_metadata(session: requests.Session | None = None) -> dict[str, str]:
    response = _get(SEC_13F_DATASETS_URL, session=session, headers=SEC_HEADERS)
    matches = list(
        re.finditer(
            r'<a[^>]+href=["\']([^"\']+\.zip)["\'][^>]*>(.*?)</a>',
            response.text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if not matches:
        raise ValueError("SEC 13F datasets page did not expose a ZIP link")
    href = str(matches[0].group(1)).strip()
    label = re.sub(r"\s+", " ", str(matches[0].group(2))).strip()
    return {
        "label": label or "Latest 13F ZIP",
        "zip_url": urljoin(str(response.url), href),
        "source_url": str(response.url),
    }


def _read_latest_13f_infotable(cusips: list[str], *, session: requests.Session | None = None) -> pd.DataFrame:
    metadata = fetch_latest_13f_dataset_metadata(session=session)
    response = _get(metadata["zip_url"], session=session, headers=SEC_HEADERS)
    targets = {str(cusip).strip().upper() for cusip in cusips if str(cusip).strip()}
    if not targets:
        return pd.DataFrame()

    with ZipFile(BytesIO(response.content)) as zf:
        info_name = next((name for name in zf.namelist() if "INFOTABLE" in name.upper() and name.lower().endswith((".tsv", ".txt"))), "")
        if not info_name:
            raise ValueError("SEC 13F ZIP did not contain INFOTABLE data")
        with zf.open(info_name) as handle:
            table = pd.read_csv(
                handle,
                sep="\t",
                dtype=str,
                usecols=lambda column: str(column).upper() in {"CUSIP", "VALUE", "SSHPRNAMT", "ACCESSION_NUMBER"},
            )

    table.columns = [str(column).upper() for column in table.columns]
    table["CUSIP"] = table["CUSIP"].astype(str).str.upper().str.strip()
    filtered = table[table["CUSIP"].isin(targets)].copy()
    if filtered.empty:
        return filtered
    filtered["VALUE"] = pd.to_numeric(filtered["VALUE"], errors="coerce")
    filtered["SSHPRNAMT"] = pd.to_numeric(filtered["SSHPRNAMT"], errors="coerce")
    return filtered


def fetch_latest_13f_sector_etf_positions(
    sector_profiles: list[dict[str, Any]],
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    profiles = [profile for profile in sector_profiles if str(profile.get("cusip", "")).strip()]
    if not profiles:
        return {"table": pd.DataFrame(), "dataset_label": "", "dataset_url": ""}

    cusip_to_sector = {str(profile["cusip"]).strip().upper(): profile for profile in profiles}
    filtered = _read_latest_13f_infotable(list(cusip_to_sector), session=session)
    metadata = fetch_latest_13f_dataset_metadata(session=session)
    if filtered.empty:
        return {
            "table": pd.DataFrame(),
            "dataset_label": metadata["label"],
            "dataset_url": metadata["zip_url"],
        }

    grouped = (
        filtered.groupby("CUSIP", dropna=False)
        .agg(
            filing_count=("ACCESSION_NUMBER", "nunique"),
            manager_value_total_kusd=("VALUE", "sum"),
            manager_shares_total=("SSHPRNAMT", "sum"),
        )
        .reset_index()
    )
    grouped["sector_code"] = grouped["CUSIP"].map(lambda value: cusip_to_sector.get(str(value), {}).get("sector_code", ""))
    grouped["sector_name"] = grouped["CUSIP"].map(lambda value: cusip_to_sector.get(str(value), {}).get("sector_name", ""))
    grouped["manager_value_total_usd"] = grouped["manager_value_total_kusd"] * 1000.0
    table = grouped[
        ["sector_code", "sector_name", "CUSIP", "filing_count", "manager_value_total_usd", "manager_shares_total"]
    ].rename(columns={"CUSIP": "cusip"})

    return {
        "table": table.sort_values(by="manager_value_total_usd", ascending=False).reset_index(drop=True),
        "dataset_label": metadata["label"],
        "dataset_url": metadata["zip_url"],
    }


def fetch_sec_company_tickers(session: requests.Session | None = None) -> pd.DataFrame:
    response = _get(SEC_COMPANY_TICKERS_URL, session=session, headers=SEC_HEADERS)
    payload = json.loads(response.text)
    records = list(payload.values()) if isinstance(payload, dict) else list(payload)
    frame = pd.DataFrame(records)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["title"] = frame["title"].astype(str).str.strip()
    frame["normalized_title"] = frame["title"].map(_normalize_company_name)
    frame["cik_str"] = frame["cik_str"].map(lambda value: str(value).zfill(10))
    return frame


def _match_holding_to_company(name: str, mapping: dict[str, dict[str, str]]) -> dict[str, str] | None:
    normalized = _normalize_company_name(name)
    if not normalized:
        return None
    return mapping.get(normalized)


def _recent_13dg_count(submissions: dict[str, Any], *, cutoff_date: datetime) -> tuple[int, list[str]]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = list(recent.get("form", []))
    dates = list(recent.get("filingDate", []))
    descriptions: list[str] = []
    count = 0
    for form, filed in zip(forms, dates):
        form_name = str(form).strip().upper()
        if form_name not in THIRTEEN_D_FORMS:
            continue
        try:
            filed_dt = datetime.strptime(str(filed), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if filed_dt < cutoff_date:
            continue
        count += 1
        descriptions.append(f"{form_name} {filed_dt.date().isoformat()}")
    return count, descriptions


def fetch_recent_13dg_sector_events(
    sector_profiles: list[dict[str, Any]],
    *,
    session: requests.Session | None = None,
    trailing_days: int = RECENT_13DG_DAYS,
) -> dict[str, Any]:
    company_map = fetch_sec_company_tickers(session=session)
    by_name: dict[str, dict[str, str]] = {}
    for row in company_map.itertuples(index=False):
        key = str(getattr(row, "normalized_title", "")).strip()
        if key and key not in by_name:
            by_name[key] = {
                "ticker": str(getattr(row, "ticker", "")).strip(),
                "cik": str(getattr(row, "cik_str", "")).strip(),
                "title": str(getattr(row, "title", "")).strip(),
            }

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=int(trailing_days))
    cik_cache: dict[str, tuple[int, list[str]]] = {}
    rows: list[dict[str, Any]] = []

    for profile in sector_profiles:
        matches: list[dict[str, str]] = []
        for holding in list(profile.get("top_holdings", []))[:TOP_HOLDINGS_LIMIT]:
            matched = _match_holding_to_company(str(holding.get("name", "")), by_name)
            if matched is not None:
                matches.append(matched)

        event_count = 0
        sample_events: list[str] = []
        unique_ciks = {match["cik"] for match in matches if match.get("cik")}
        for cik in unique_ciks:
            if cik not in cik_cache:
                submissions = _get(SEC_SUBMISSIONS_URL.format(cik=cik), session=session, headers=SEC_HEADERS).json()
                cik_cache[cik] = _recent_13dg_count(submissions, cutoff_date=cutoff_date)
            count, descriptions = cik_cache[cik]
            event_count += count
            sample_events.extend(descriptions[:2])

        rows.append(
            {
                "sector_code": str(profile.get("sector_code", "")),
                "sector_name": str(profile.get("sector_name", "")),
                "matched_top_holdings": int(len(unique_ciks)),
                "recent_13dg_events": int(event_count),
                "sample_events": ", ".join(sample_events[:3]),
            }
        )

    return {
        "table": pd.DataFrame(rows).sort_values(by=["recent_13dg_events", "matched_top_holdings"], ascending=[False, False]).reset_index(drop=True),
        "lookback_days": int(trailing_days),
        "source_url": SEC_SUBMISSIONS_URL.format(cik="##########"),
    }


def fetch_form_sho_context() -> dict[str, Any]:
    return {
        "status": "policy_only",
        "source_url": FORM_SHO_RULE_URL,
        "note": (
            "Rule 13f-2 / Form SHO는 2026-02-17부터 월별 보고가 시작됐고, "
            "SEC는 종목별 aggregated short-position 정보를 공개한다고 안내한다. "
            "현재 이 repo는 public aggregated feed 경로를 아직 확정하지 못해 availability/context만 표시한다."
        ),
    }


def load_us_ownership_context(
    sector_profiles: list[dict[str, Any]],
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    errors: dict[str, str] = {}

    try:
        context["ici_weekly_flows"] = fetch_ici_weekly_etf_flows(session=session)
    except Exception as exc:
        errors["ici_weekly_flows"] = str(exc)

    try:
        context["sec_13f_positions"] = fetch_latest_13f_sector_etf_positions(sector_profiles, session=session)
    except Exception as exc:
        errors["sec_13f_positions"] = str(exc)

    try:
        context["sec_13dg_events"] = fetch_recent_13dg_sector_events(sector_profiles, session=session)
    except Exception as exc:
        errors["sec_13dg_events"] = str(exc)

    context["form_sho_context"] = fetch_form_sho_context()
    context["errors"] = errors
    return context


__all__ = [
    "fetch_form_sho_context",
    "fetch_ici_weekly_etf_flows",
    "fetch_latest_13f_dataset_metadata",
    "fetch_latest_13f_sector_etf_positions",
    "fetch_recent_13dg_sector_events",
    "fetch_sec_company_tickers",
    "load_us_ownership_context",
]
