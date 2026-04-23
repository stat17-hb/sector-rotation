"""Market-aware stock -> dashboard sector resolution helpers."""
from __future__ import annotations

from functools import lru_cache
import re
import unicodedata
from typing import Any

import pandas as pd
import yfinance as yf

from src.data_sources.krx_sector_authority import canonicalize_kr_sector_universe_rows
from src.data_sources.us_ownership_context import fetch_sec_company_tickers
from src.data_sources.warehouse import (
    read_latest_kr_ticker_names,
    read_latest_sector_constituents_snapshot,
)


_US_SECTOR_TO_DASHBOARD: dict[str, tuple[str, str]] = {
    "technology": ("XLK", "Technology"),
    "communication services": ("XLC", "Communication Services"),
    "consumer cyclical": ("XLY", "Consumer Discretionary"),
    "consumer discretionary": ("XLY", "Consumer Discretionary"),
    "consumer defensive": ("XLP", "Consumer Staples"),
    "consumer staples": ("XLP", "Consumer Staples"),
    "industrials": ("XLI", "Industrials"),
    "financial services": ("XLF", "Financials"),
    "financial": ("XLF", "Financials"),
    "real estate": ("XLRE", "Real Estate"),
    "energy": ("XLE", "Energy"),
    "basic materials": ("XLB", "Materials"),
    "materials": ("XLB", "Materials"),
    "healthcare": ("XLV", "Health Care"),
    "health care": ("XLV", "Health Care"),
    "utilities": ("XLU", "Utilities"),
}

_US_SECTOR_KEY_TO_DASHBOARD: dict[str, tuple[str, str]] = {
    "technology": ("XLK", "Technology"),
    "communication-services": ("XLC", "Communication Services"),
    "consumer-cyclical": ("XLY", "Consumer Discretionary"),
    "consumer-defensive": ("XLP", "Consumer Staples"),
    "industrials": ("XLI", "Industrials"),
    "financial-services": ("XLF", "Financials"),
    "real-estate": ("XLRE", "Real Estate"),
    "energy": ("XLE", "Energy"),
    "basic-materials": ("XLB", "Materials"),
    "healthcare": ("XLV", "Health Care"),
    "utilities": ("XLU", "Utilities"),
}

_US_INDUSTRY_KEY_TO_DASHBOARD: dict[str, tuple[str, str]] = {
    "software-infrastructure": ("XLK", "Technology"),
    "software-application": ("XLK", "Technology"),
    "semiconductors": ("XLK", "Technology"),
    "consumer-electronics": ("XLK", "Technology"),
    "internet-content-information": ("XLC", "Communication Services"),
    "telecom-services": ("XLC", "Communication Services"),
    "restaurants": ("XLY", "Consumer Discretionary"),
    "specialty-retail": ("XLY", "Consumer Discretionary"),
    "internet-retail": ("XLY", "Consumer Discretionary"),
    "packaged-foods": ("XLP", "Consumer Staples"),
    "beverages-non-alcoholic": ("XLP", "Consumer Staples"),
    "aerospace-defense": ("XLI", "Industrials"),
    "integrated-freight-logistics": ("XLI", "Industrials"),
    "banks-diversified": ("XLF", "Financials"),
    "banks-regional": ("XLF", "Financials"),
    "asset-management": ("XLF", "Financials"),
    "insurance-diversified": ("XLF", "Financials"),
    "reit-specialty": ("XLRE", "Real Estate"),
    "reit-industrial": ("XLRE", "Real Estate"),
    "oil-gas-integrated": ("XLE", "Energy"),
    "oil-gas-ep": ("XLE", "Energy"),
    "specialty-chemicals": ("XLB", "Materials"),
    "steel": ("XLB", "Materials"),
    "biotechnology": ("XLV", "Health Care"),
    "drug-manufacturers-general": ("XLV", "Health Care"),
    "medical-devices": ("XLV", "Health Care"),
    "utilities-regulated-electric": ("XLU", "Utilities"),
}

_US_NORMALIZATION_STOPWORDS = {
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


def _normalize_lookup_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text:
        return ""
    compact = re.sub(r"\s+", "", text).upper()
    return "".join(ch for ch in compact if ch.isalnum())


def _normalize_us_company_name(value: str) -> str:
    cleaned = str(value or "").upper().replace("+", " AND ").replace("&", " AND ")
    cleaned = re.sub(r"[^A-Z0-9 ]+", " ", cleaned)
    tokens = [token for token in cleaned.split() if token and token not in _US_NORMALIZATION_STOPWORDS]
    return " ".join(tokens)


def _iter_sector_entries(sector_map: dict[str, Any]) -> list[dict[str, str]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for regime_payload in dict(sector_map or {}).get("regimes", {}).values():
        for sector in list(dict(regime_payload or {}).get("sectors", [])):
            code = str(dict(sector or {}).get("code", "")).strip()
            name = str(dict(sector or {}).get("name", "")).strip()
            if not code or not name or code in seen:
                continue
            seen.add(code)
            raw_priority = dict(sector or {}).get("lookup_priority")
            try:
                lookup_priority = int(raw_priority) if raw_priority is not None and str(raw_priority).strip() != "" else None
            except (TypeError, ValueError):
                lookup_priority = None
            entries.append({"code": code, "name": name, "lookup_priority": lookup_priority})
    return entries


def _iter_kr_official_sector_entries(
    sector_universe_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    canonical_rows = canonicalize_kr_sector_universe_rows(list(sector_universe_rows or []))
    for row in canonical_rows:
        code = str(dict(row or {}).get("index_code", "")).strip()
        name = str(dict(row or {}).get("index_name", "")).strip()
        if not code or not name:
            continue
        entries.append({"code": code, "name": name, "lookup_priority": None})
    return entries


def _merge_sector_entries(
    primary_entries: list[dict[str, Any]],
    fallback_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge sector entries by code while keeping primary names and fallback metadata."""
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for entry in fallback_entries:
        code = str(dict(entry or {}).get("code", "")).strip()
        if not code:
            continue
        if code not in merged:
            order.append(code)
        merged[code] = dict(entry)

    for entry in primary_entries:
        code = str(dict(entry or {}).get("code", "")).strip()
        if not code:
            continue
        current = dict(merged.get(code, {}))
        if code not in merged:
            order.append(code)
        current["code"] = code
        primary_name = str(dict(entry).get("name", "")).strip()
        if primary_name:
            current["name"] = primary_name
        primary_priority = dict(entry).get("lookup_priority")
        if primary_priority is not None:
            current["lookup_priority"] = primary_priority
        merged[code] = current

    return [merged[code] for code in order]


def _dashboard_maps(sector_entries: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str], dict[str, int | None]]:
    by_code = {entry["code"]: entry["name"] for entry in sector_entries}
    by_name = {entry["name"]: entry["code"] for entry in sector_entries}
    by_priority = {entry["code"]: entry.get("lookup_priority") for entry in sector_entries}
    return by_code, by_name, by_priority


def _base_result_fields() -> dict[str, object]:
    return {
        "canonicalization_applied": False,
        "canonicalization_basis": "not_applicable",
        "match_effective_date": "",
        "match_date_mode": "not_applicable",
        "matched_sector_candidates": [],
    }


def _empty_result(
    *,
    market: str,
    query: str,
    status: str,
    explanation: str,
) -> dict[str, object]:
    return {
        "status": status,
        "market": market,
        "query": query,
        "normalized_query": _normalize_lookup_text(query),
        "matched_symbol": "",
        "matched_name": "",
        "sector_code": "",
        "sector_name": "",
        "resolution_kind": "",
        "source": "",
        "confidence": "",
        "explanation": explanation,
        **_base_result_fields(),
    }


def _success_result(
    *,
    market: str,
    query: str,
    matched_symbol: str,
    matched_name: str,
    sector_code: str,
    sector_name: str,
    resolution_kind: str,
    source: str,
    confidence: str,
    explanation: str,
    canonicalization_applied: bool = False,
    canonicalization_basis: str = "",
    match_effective_date: str = "",
    match_date_mode: str = "not_applicable",
    matched_sector_candidates: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "status": "success",
        "market": market,
        "query": query,
        "normalized_query": _normalize_lookup_text(query),
        "matched_symbol": matched_symbol,
        "matched_name": matched_name,
        "sector_code": sector_code,
        "sector_name": sector_name,
        "resolution_kind": resolution_kind,
        "source": source,
        "confidence": confidence,
        "explanation": explanation,
        "canonicalization_applied": bool(canonicalization_applied),
        "canonicalization_basis": canonicalization_basis,
        "match_effective_date": match_effective_date,
        "match_date_mode": match_date_mode,
        "matched_sector_candidates": list(matched_sector_candidates or []),
    }


def _normalize_asof_date(asof_date: Any) -> str:
    if asof_date is None or str(asof_date).strip() == "":
        return pd.Timestamp.today().strftime("%Y%m%d")
    return pd.Timestamp(asof_date).strftime("%Y%m%d")


def _match_single_sector(
    *,
    frame: pd.DataFrame,
    by_code: dict[str, str],
    market: str,
    query: str,
    matched_symbol: str,
    matched_name: str,
    source: str,
    confidence: str,
) -> dict[str, object]:
    if frame.empty:
        return _empty_result(
            market=market,
            query=query,
            status="not_found",
            explanation=f"No dashboard sector mapping was found for {matched_symbol or query}.",
        )
    sector_codes = sorted({str(value).strip() for value in frame["sector_code"].tolist() if str(value).strip()})
    if len(sector_codes) != 1:
        return _empty_result(
            market=market,
            query=query,
            status="ambiguous",
            explanation=f"{matched_symbol or query} maps to multiple dashboard sectors, so no selection was applied.",
        )
    sector_code = sector_codes[0]
    sector_name = by_code.get(sector_code, "")
    if not sector_name:
        return _empty_result(
            market=market,
            query=query,
            status="unsupported",
            explanation=f"Resolved sector {sector_code} is not part of the active dashboard universe.",
        )
    return _success_result(
        market=market,
        query=query,
        matched_symbol=matched_symbol,
        matched_name=matched_name,
        sector_code=sector_code,
        sector_name=sector_name,
        resolution_kind="constituent_membership",
        source=source,
        confidence=confidence,
        explanation=f"{matched_name or matched_symbol} ({matched_symbol}) is a constituent of {sector_name}.",
        canonicalization_applied=False,
        canonicalization_basis="single_match",
        match_date_mode="not_applicable",
        matched_sector_candidates=[
            {
                "sector_code": sector_code,
                "sector_name": sector_name,
                "lookup_priority": None,
                "source": source,
                "resolved_from": "",
                "snapshot_date": "",
            }
        ],
    )


@lru_cache(maxsize=8)
def _read_live_kr_constituents(sector_codes: tuple[str, ...], asof_date: str) -> pd.DataFrame:
    from pykrx import stock as stock_module

    from src.data_sources.krx_constituents import candidate_reference_dates, lookup_index_constituents

    rows: list[dict[str, str]] = []
    candidate_dates = candidate_reference_dates(asof_date, periods=5)
    for sector_code in sector_codes:
        result = lookup_index_constituents(
            stock_module,
            sector_code=sector_code,
            candidate_dates=candidate_dates,
        )
        for ticker in result.tickers:
            rows.append(
                {
                    "sector_code": str(sector_code),
                    "ticker": str(ticker),
                    "resolved_from": str(result.resolved_from or asof_date),
                    "source": str(result.source or "kr_live_constituents"),
                    "snapshot_date": str(result.resolved_from or asof_date),
                }
            )
    return pd.DataFrame(rows, columns=["sector_code", "ticker", "resolved_from", "source", "snapshot_date"])


@lru_cache(maxsize=512)
def _fetch_kr_live_ticker_name(ticker: str) -> str:
    from pykrx import stock as stock_module

    return str(stock_module.get_market_ticker_name(str(ticker)) or "").strip()


def _prepare_snapshot_candidate_rows(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame(columns=["sector_code", "ticker", "resolved_from", "source", "snapshot_date"])
    frame = snapshot.copy()
    frame["sector_code"] = frame["sector_code"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str)
    if "resolved_from" in frame.columns:
        frame["resolved_from"] = frame["resolved_from"].astype(str)
    elif "reference_date" in frame.columns:
        frame["resolved_from"] = frame["reference_date"].astype(str)
    else:
        frame["resolved_from"] = ""
    frame["source"] = "warehouse_sector_constituents_snapshot"
    if "snapshot_date" in frame.columns:
        frame["snapshot_date"] = frame["snapshot_date"].astype(str)
    else:
        frame["snapshot_date"] = ""
    return frame[["sector_code", "ticker", "resolved_from", "source", "snapshot_date"]]


def _build_matched_sector_candidates(
    *,
    candidate_rows: pd.DataFrame,
    by_code: dict[str, str],
    by_priority: dict[str, int | None],
) -> list[dict[str, object]]:
    if candidate_rows.empty:
        return []
    rows: list[dict[str, object]] = []
    for _, row in candidate_rows.iterrows():
        sector_code = str(row["sector_code"]).strip()
        if not sector_code:
            continue
        rows.append(
            {
                "sector_code": sector_code,
                "sector_name": by_code.get(sector_code, sector_code),
                "lookup_priority": by_priority.get(sector_code),
                "source": str(row.get("source", "")).strip(),
                "resolved_from": str(row.get("resolved_from", "")).strip(),
                "snapshot_date": str(row.get("snapshot_date", "")).strip(),
            }
        )
    deduped: dict[str, dict[str, object]] = {}
    for row in rows:
        deduped.setdefault(str(row["sector_code"]), row)
    return list(deduped.values())


def _candidate_effective_dates(candidates: list[dict[str, object]]) -> set[str]:
    dates: set[str] = set()
    for candidate in candidates:
        effective = str(candidate.get("snapshot_date") or candidate.get("resolved_from") or "").strip()
        if effective:
            dates.add(effective)
    return dates


def _resolve_kr_sector_candidates(
    *,
    query: str,
    matched_symbol: str,
    matched_name: str,
    confidence: str,
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    if not candidates:
        return _empty_result(
            market="KR",
            query=query,
            status="not_found",
            explanation=f"No KR dashboard sector match was found for {matched_name or matched_symbol or query}.",
        )
    if len(candidates) == 1:
        candidate = candidates[0]
        return _success_result(
            market="KR",
            query=query,
            matched_symbol=matched_symbol,
            matched_name=matched_name,
            sector_code=str(candidate["sector_code"]),
            sector_name=str(candidate["sector_name"]),
            resolution_kind="constituent_membership",
            source=str(candidate.get("source", "")),
            confidence=confidence,
            explanation=f"{matched_name or matched_symbol} ({matched_symbol}) maps to {candidate['sector_name']}.",
            canonicalization_applied=False,
            canonicalization_basis="single_match",
            match_date_mode="not_applicable",
            matched_sector_candidates=candidates,
        )

    effective_dates = _candidate_effective_dates(candidates)
    if len(effective_dates) != 1:
        return {
            **_empty_result(
                market="KR",
                query=query,
                status="ambiguous",
                explanation=f"{matched_name or matched_symbol or query} has mixed effective dates across KR sector candidates, so no canonical sector was selected.",
            ),
            "canonicalization_applied": False,
            "canonicalization_basis": "mixed_effective_dates",
            "match_date_mode": "mixed",
            "matched_sector_candidates": candidates,
        }

    prioritized = [candidate for candidate in candidates if candidate.get("lookup_priority") is not None]
    if not prioritized:
        return {
            **_empty_result(
                market="KR",
                query=query,
                status="ambiguous",
                explanation=f"{matched_name or matched_symbol or query} maps to multiple KR sectors and none has a canonical lookup priority.",
            ),
            "canonicalization_applied": False,
            "canonicalization_basis": "missing_lookup_priority",
            "match_date_mode": "same_date",
            "match_effective_date": next(iter(effective_dates)),
            "matched_sector_candidates": candidates,
        }

    best_priority = min(int(candidate["lookup_priority"]) for candidate in prioritized)
    best_candidates = [candidate for candidate in prioritized if int(candidate["lookup_priority"]) == best_priority]
    if len(best_candidates) != 1:
        return {
            **_empty_result(
                market="KR",
                query=query,
                status="ambiguous",
                explanation=f"{matched_name or matched_symbol or query} maps to multiple KR sectors with the same highest lookup priority.",
            ),
            "canonicalization_applied": False,
            "canonicalization_basis": "equal_lowest_priority",
            "match_date_mode": "same_date",
            "match_effective_date": next(iter(effective_dates)),
            "matched_sector_candidates": candidates,
        }

    winner = best_candidates[0]
    return _success_result(
        market="KR",
        query=query,
        matched_symbol=matched_symbol,
        matched_name=matched_name,
        sector_code=str(winner["sector_code"]),
        sector_name=str(winner["sector_name"]),
        resolution_kind="constituent_membership",
        source=str(winner.get("source", "")),
        confidence=confidence,
        explanation=(
            f"{matched_name or matched_symbol} ({matched_symbol}) matched multiple KR sectors on {next(iter(effective_dates))}; "
            f"{winner['sector_name']} won by lookup_priority."
        ),
        canonicalization_applied=True,
        canonicalization_basis="lookup_priority_same_date",
        match_effective_date=next(iter(effective_dates)),
        match_date_mode="same_date",
        matched_sector_candidates=candidates,
    )


def _resolve_kr_stock(
    *,
    query: str,
    sector_entries: list[dict[str, str]],
    asof_date: str,
) -> dict[str, object]:
    by_code, _, by_priority = _dashboard_maps(sector_entries)
    sector_codes = [entry["code"] for entry in sector_entries]
    snapshot = read_latest_sector_constituents_snapshot(sector_codes, market="KR")
    snapshot_candidates = _prepare_snapshot_candidate_rows(snapshot)
    exact_query = str(query or "").strip()
    normalized_query = _normalize_lookup_text(query)

    if re.fullmatch(r"\d{6}", exact_query):
        ticker_match = snapshot_candidates.loc[snapshot_candidates["ticker"] == exact_query]
        if ticker_match.empty:
            live_snapshot = _read_live_kr_constituents(tuple(sector_codes), asof_date)
            ticker_match = live_snapshot.loc[live_snapshot["ticker"] == exact_query]

        matched_name = ""
        latest_names = read_latest_kr_ticker_names()
        if not latest_names.empty:
            matched = latest_names.loc[latest_names["ticker"].astype(str) == exact_query]
            if not matched.empty:
                matched_name = str(matched.iloc[0]["ticker_name"]).strip()
        if not matched_name:
            matched_name = _fetch_kr_live_ticker_name(exact_query)
        return _resolve_kr_sector_candidates(
            query=query,
            matched_symbol=exact_query,
            matched_name=matched_name,
            confidence="high",
            candidates=_build_matched_sector_candidates(
                candidate_rows=ticker_match,
                by_code=by_code,
                by_priority=by_priority,
            ),
        )

    latest_names = read_latest_kr_ticker_names()
    candidate_tickers: list[str] = []
    if not latest_names.empty:
        latest_names = latest_names.copy()
        latest_names["ticker"] = latest_names["ticker"].astype(str)
        latest_names["ticker_name"] = latest_names["ticker_name"].astype(str)
        latest_names["normalized_name"] = latest_names["ticker_name"].map(_normalize_lookup_text)
        warehouse_matches = latest_names.loc[latest_names["normalized_name"] == normalized_query]
        candidate_tickers = sorted({str(value).strip() for value in warehouse_matches["ticker"].tolist() if str(value).strip()})
        if len(candidate_tickers) > 1:
            return _empty_result(
                market="KR",
                query=query,
                status="ambiguous",
                explanation=f"{query} matches multiple KR tickers, so no sector was selected.",
            )
        if len(candidate_tickers) == 1:
            matched_symbol = candidate_tickers[0]
            matched_name = str(warehouse_matches.iloc[0]["ticker_name"]).strip()
            ticker_match = snapshot_candidates.loc[snapshot_candidates["ticker"] == matched_symbol]
            if ticker_match.empty:
                live_snapshot = _read_live_kr_constituents(tuple(sector_codes), asof_date)
                ticker_match = live_snapshot.loc[live_snapshot["ticker"] == matched_symbol]
            return _resolve_kr_sector_candidates(
                query=query,
                matched_symbol=matched_symbol,
                matched_name=matched_name,
                confidence="medium",
                candidates=_build_matched_sector_candidates(
                    candidate_rows=ticker_match,
                    by_code=by_code,
                    by_priority=by_priority,
                ),
            )

    live_snapshot = _read_live_kr_constituents(tuple(sector_codes), asof_date)
    snapshot_rows = snapshot_candidates[["ticker"]].drop_duplicates() if "ticker" in snapshot_candidates.columns else pd.DataFrame(columns=["ticker"])
    live_rows = live_snapshot[["ticker"]].drop_duplicates() if "ticker" in live_snapshot.columns else pd.DataFrame(columns=["ticker"])
    candidate_rows = pd.concat([snapshot_rows, live_rows], ignore_index=True).drop_duplicates()
    live_name_matches: list[tuple[str, str]] = []
    for ticker in candidate_rows["ticker"].astype(str).tolist():
        ticker_name = _fetch_kr_live_ticker_name(ticker)
        if ticker_name and _normalize_lookup_text(ticker_name) == normalized_query:
            live_name_matches.append((ticker, ticker_name))
    if len(live_name_matches) > 1:
        return _empty_result(
            market="KR",
            query=query,
            status="ambiguous",
            explanation=f"{query} matches multiple KR tickers, so no sector was selected.",
        )
    if len(live_name_matches) == 1:
        matched_symbol, matched_name = live_name_matches[0]
        ticker_match = live_snapshot.loc[live_snapshot["ticker"] == matched_symbol]
        return _resolve_kr_sector_candidates(
            query=query,
            matched_symbol=matched_symbol,
            matched_name=matched_name,
            confidence="medium",
            candidates=_build_matched_sector_candidates(
                candidate_rows=ticker_match,
                by_code=by_code,
                by_priority=by_priority,
            ),
        )

    return _empty_result(
        market="KR",
        query=query,
        status="not_found",
        explanation=f"No KR dashboard sector match was found for {query}.",
    )


def _map_us_sector(metadata: dict[str, str]) -> tuple[str, str, str] | None:
    sector_value = str(metadata.get("sector", "")).strip()
    sector_key = str(metadata.get("sectorKey", "")).strip()
    industry_key = str(metadata.get("industryKey", "")).strip()

    sector_match = _US_SECTOR_TO_DASHBOARD.get(sector_value.lower())
    if sector_match:
        return sector_match[0], sector_match[1], "yfinance_sector"

    sector_key_match = _US_SECTOR_KEY_TO_DASHBOARD.get(sector_key)
    if sector_key_match:
        return sector_key_match[0], sector_key_match[1], "yfinance_sectorKey"

    industry_key_match = _US_INDUSTRY_KEY_TO_DASHBOARD.get(industry_key)
    if industry_key_match:
        return industry_key_match[0], industry_key_match[1], "yfinance_industryKey"

    return None


def _fetch_us_issuer_metadata(symbol: str) -> dict[str, str]:
    ticker = yf.Ticker(str(symbol))
    payload: dict[str, Any] = {}
    try:
        if hasattr(ticker, "get_info"):
            payload = dict(ticker.get_info() or {})
        else:
            payload = dict(getattr(ticker, "info", {}) or {})
    except Exception:
        payload = dict(getattr(ticker, "info", {}) or {})
    return {
        "sector": str(payload.get("sector", "")).strip(),
        "sectorKey": str(payload.get("sectorKey", "")).strip(),
        "industryKey": str(payload.get("industryKey", "")).strip(),
    }


def _resolve_us_stock(
    *,
    query: str,
    sector_entries: list[dict[str, str]],
) -> dict[str, object]:
    by_code, by_name, _ = _dashboard_maps(sector_entries)
    universe_pairs = {(entry["code"], entry["name"]) for entry in sector_entries}
    company_map = fetch_sec_company_tickers()
    exact_query = str(query or "").strip().upper()
    normalized_name = _normalize_us_company_name(query)

    matches = company_map.loc[company_map["ticker"].astype(str).str.upper() == exact_query]
    confidence = "high"
    if matches.empty:
        matches = company_map.loc[company_map["normalized_title"].astype(str) == normalized_name]
        confidence = "medium"
    if matches.empty:
        return _empty_result(
            market="US",
            query=query,
            status="not_found",
            explanation=f"No US issuer match was found for {query}.",
        )
    if len(matches.index) > 1:
        return _empty_result(
            market="US",
            query=query,
            status="ambiguous",
            explanation=f"{query} matches multiple US issuers, so no sector was selected.",
        )

    matched = matches.iloc[0]
    matched_symbol = str(matched.get("ticker", "")).strip().upper()
    matched_name = str(matched.get("title", "")).strip()
    metadata = _fetch_us_issuer_metadata(matched_symbol)
    translated = _map_us_sector(metadata)
    if translated is None:
        return _empty_result(
            market="US",
            query=query,
            status="unsupported",
            explanation=f"{matched_symbol} did not return a dashboard-mappable US issuer sector.",
        )
    sector_code, sector_name, source = translated
    if (sector_code, sector_name) not in universe_pairs or by_name.get(sector_name) != sector_code or by_code.get(sector_code) != sector_name:
        return _empty_result(
            market="US",
            query=query,
            status="unsupported",
            explanation=f"Resolved US sector {sector_name} is not part of the active dashboard universe.",
        )
    return _success_result(
        market="US",
        query=query,
        matched_symbol=matched_symbol,
        matched_name=matched_name,
        sector_code=sector_code,
        sector_name=sector_name,
        resolution_kind="issuer_classification",
        source=source,
        confidence=confidence,
        explanation=f"{matched_name or matched_symbol} ({matched_symbol}) maps to {sector_name} by issuer classification.",
    )


def resolve_stock_to_sector(
    query: str,
    market: str,
    sector_map: dict[str, Any],
    asof_date: Any = None,
    sector_universe_rows: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    exact_query = str(query or "").strip()
    normalized_market = str(market or "KR").strip().upper() or "KR"
    if not exact_query:
        return _empty_result(
            market=normalized_market,
            query=exact_query,
            status="error",
            explanation="Enter a stock name or ticker/code before running the lookup.",
        )

    if normalized_market == "KR":
        config_entries = _iter_sector_entries(sector_map)
        sector_entries = _merge_sector_entries(
            _iter_kr_official_sector_entries(sector_universe_rows),
            config_entries,
        )
        if not sector_entries:
            sector_entries = config_entries
        if not sector_entries:
            return _empty_result(
                market=normalized_market,
                query=exact_query,
                status="unsupported",
                explanation="The active KR sector universe is unavailable.",
            )
        return _resolve_kr_stock(
            query=exact_query,
            sector_entries=sector_entries,
            asof_date=_normalize_asof_date(asof_date),
        )
    if normalized_market == "US":
        sector_entries = _iter_sector_entries(sector_map)
        if not sector_entries:
            return _empty_result(
                market=normalized_market,
                query=exact_query,
                status="unsupported",
                explanation="The active dashboard sector universe is unavailable.",
            )
        return _resolve_us_stock(
            query=exact_query,
            sector_entries=sector_entries,
        )
    return _empty_result(
        market=normalized_market,
        query=exact_query,
        status="unsupported",
        explanation=f"Market {normalized_market} is not supported for stock-sector lookup.",
    )
