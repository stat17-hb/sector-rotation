"""Experimental KRX investor-flow refresh and warehouse readers.

The dashboard never probes the live KRX path during page load. Manual refresh
uses pykrx's investor-by-ticker view, which itself depends on the same KRX web
backend family as the research prototype in ``python.collectors.krx_investor_flow``.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
import logging
from typing import Any, Literal

import pandas as pd

from src.data_sources.pykrx_compat import ensure_pykrx_transport_compat
from src.data_sources.warehouse import (
    get_dataset_artifact_key,
    probe_dataset_mode,
    read_dataset_status,
    read_sector_investor_flow,
    record_ingest_run,
    update_ingest_watermark,
    upsert_investor_flow_raw,
    upsert_investor_flow_sector,
)


logger = logging.getLogger(__name__)

DataStatus = Literal["LIVE", "CACHED", "SAMPLE"]
LoaderResult = tuple[DataStatus, pd.DataFrame]

FLOW_PROVIDER = "PYKRX_UNOFFICIAL"
DEFAULT_LOOKBACK_CALENDAR_DAYS = 120
DEFAULT_INVESTOR_TYPES: tuple[str, ...] = ("개인", "외국인", "기관합계")


@dataclass(frozen=True)
class SectorUniverse:
    sector_codes: list[str]
    sector_names: dict[str, str]
    ticker_to_sector_codes: dict[str, list[str]]


def _tracked_sector_entries(sector_map: dict[str, Any]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for regime in sector_map.get("regimes", {}).values():
        for sector in regime.get("sectors", []):
            code = str(sector.get("code", "")).strip()
            name = str(sector.get("name", code)).strip() or code
            if not code or code in seen:
                continue
            seen.add(code)
            entries.append({"code": code, "name": name})
    return entries


def _build_sector_universe(
    sector_map: dict[str, Any],
    *,
    reference_date: str,
) -> SectorUniverse:
    ensure_pykrx_transport_compat()
    from pykrx import stock  # type: ignore[import]

    entries = _tracked_sector_entries(sector_map)
    ticker_to_sector_codes: dict[str, list[str]] = defaultdict(list)
    sector_names: dict[str, str] = {}
    sector_codes: list[str] = []

    for entry in entries:
        sector_code = str(entry["code"])
        sector_name = str(entry["name"])
        sector_names[sector_code] = sector_name
        sector_codes.append(sector_code)
        try:
            constituents = stock.get_index_portfolio_deposit_file(reference_date, sector_code)
        except Exception as exc:
            logger.warning("Investor-flow constituent lookup failed for %s: %s", sector_code, exc)
            constituents = []

        ticker_list: list[str]
        if isinstance(constituents, pd.DataFrame):
            if constituents.empty:
                ticker_list = []
            elif constituents.index.dtype == object:
                ticker_list = [str(item) for item in constituents.index.tolist()]
            else:
                ticker_list = [str(item) for item in constituents.iloc[:, 0].tolist()]
        else:
            ticker_list = [str(item) for item in list(constituents or [])]

        for ticker in ticker_list:
            if sector_code not in ticker_to_sector_codes[ticker]:
                ticker_to_sector_codes[ticker].append(sector_code)

    return SectorUniverse(
        sector_codes=sector_codes,
        sector_names=sector_names,
        ticker_to_sector_codes={ticker: sorted(codes) for ticker, codes in ticker_to_sector_codes.items()},
    )


def _resolve_column(frame: pd.DataFrame, *candidates: str) -> str | None:
    available = {str(column).strip(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in available:
            return available[candidate]
    return None


def _normalize_net_purchase_frame(
    frame: pd.DataFrame,
    *,
    trade_date: str,
    investor_type: str,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    normalized = frame.copy()
    normalized.index = normalized.index.map(str)
    name_col = _resolve_column(normalized, "종목명", "name")
    sell_col = _resolve_column(normalized, "매도거래대금", "sell_amount")
    buy_col = _resolve_column(normalized, "매수거래대금", "buy_amount")
    net_col = _resolve_column(normalized, "순매수거래대금", "net_buy_amount")
    if sell_col is None or buy_col is None or net_col is None:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(trade_date),
            "ticker": normalized.index.astype(str),
            "ticker_name": normalized[name_col].astype(str) if name_col else normalized.index.astype(str),
            "investor_type": str(investor_type),
            "buy_amount": pd.to_numeric(normalized[buy_col], errors="coerce").fillna(0).astype("int64"),
            "sell_amount": pd.to_numeric(normalized[sell_col], errors="coerce").fillna(0).astype("int64"),
            "net_buy_amount": pd.to_numeric(normalized[net_col], errors="coerce").fillna(0).astype("int64"),
        }
    )
    return out


def _aggregate_sector_flow(raw_frame: pd.DataFrame, universe: SectorUniverse) -> pd.DataFrame:
    if raw_frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in raw_frame.iterrows():
        sector_codes = universe.ticker_to_sector_codes.get(str(row["ticker"]), [])
        if not sector_codes:
            continue
        for sector_code in sector_codes:
            rows.append(
                {
                    "trade_date": row["trade_date"],
                    "sector_code": sector_code,
                    "sector_name": universe.sector_names.get(sector_code, sector_code),
                    "investor_type": row["investor_type"],
                    "buy_amount": int(row["buy_amount"]),
                    "sell_amount": int(row["sell_amount"]),
                    "net_buy_amount": int(row["net_buy_amount"]),
                }
            )

    if not rows:
        return pd.DataFrame()

    aggregated = (
        pd.DataFrame(rows)
        .groupby(["trade_date", "sector_code", "sector_name", "investor_type"], as_index=False)
        .agg(
            buy_amount=("buy_amount", "sum"),
            sell_amount=("sell_amount", "sum"),
            net_buy_amount=("net_buy_amount", "sum"),
        )
    )
    turnover = aggregated["buy_amount"].abs() + aggregated["sell_amount"].abs()
    aggregated["net_flow_ratio"] = aggregated["net_buy_amount"] / turnover.clip(lower=1)
    return aggregated


def collect_sector_investor_flow(
    *,
    sector_map: dict[str, Any],
    start: str,
    end: str,
    investor_types: tuple[str, ...] = DEFAULT_INVESTOR_TYPES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ensure_pykrx_transport_compat()
    from pykrx import stock  # type: ignore[import]

    reference_date = str(end)
    universe = _build_sector_universe(sector_map, reference_date=reference_date)
    business_days = [ts.strftime("%Y%m%d") for ts in pd.bdate_range(start, end)]
    relevant_tickers = set(universe.ticker_to_sector_codes)

    raw_frames: list[pd.DataFrame] = []
    failed_days: list[str] = []
    failed_codes: dict[str, str] = {}
    processed_requests = 0
    predicted_requests = len(business_days) * len(investor_types)

    for trade_date in business_days:
        day_success = False
        for investor_type in investor_types:
            processed_requests += 1
            try:
                raw = stock.get_market_net_purchases_of_equities_by_ticker(
                    trade_date,
                    trade_date,
                    market="ALL",
                    investor=investor_type,
                )
            except Exception as exc:
                failed_codes[f"{trade_date}:{investor_type}"] = str(exc)
                continue

            normalized = _normalize_net_purchase_frame(
                raw,
                trade_date=trade_date,
                investor_type=investor_type,
            )
            if normalized.empty:
                continue
            normalized = normalized[normalized["ticker"].isin(relevant_tickers)].copy()
            if normalized.empty:
                continue
            raw_frames.append(normalized)
            day_success = True

        if not day_success:
            failed_days.append(trade_date)

    raw_frame = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    sector_frame = _aggregate_sector_flow(raw_frame, universe)
    summary = {
        "status": "LIVE" if not sector_frame.empty else "SAMPLE",
        "provider": FLOW_PROVIDER,
        "requested_start": str(start),
        "requested_end": str(end),
        "coverage_complete": not failed_days and not sector_frame.empty,
        "failed_days": failed_days,
        "failed_codes": failed_codes,
        "predicted_requests": predicted_requests,
        "processed_requests": processed_requests,
        "rows": int(len(sector_frame)),
        "tracked_sectors": len(universe.sector_codes),
        "tracked_tickers": len(relevant_tickers),
    }
    return raw_frame, sector_frame, summary


def _sector_codes_from_map(sector_map: dict[str, Any]) -> list[str]:
    return [entry["code"] for entry in _tracked_sector_entries(sector_map)]


def load_sector_investor_flow(
    *,
    sector_map: dict[str, Any],
    start: str,
    end: str,
    market: str = "KR",
) -> LoaderResult:
    sector_codes = _sector_codes_from_map(sector_map)
    cached = read_sector_investor_flow(sector_codes, start, end, market=market)
    if cached.empty:
        return ("SAMPLE", pd.DataFrame())
    return ("CACHED", cached)


def run_manual_investor_flow_refresh(
    *,
    sector_map: dict[str, Any],
    end_date_str: str,
    lookback_days: int = DEFAULT_LOOKBACK_CALENDAR_DAYS,
    market: str = "KR",
) -> tuple[LoaderResult, dict[str, Any]]:
    requested_end = pd.Timestamp(end_date_str).normalize()
    requested_start = (requested_end - timedelta(days=int(lookback_days))).normalize()
    start = requested_start.strftime("%Y%m%d")
    end = requested_end.strftime("%Y%m%d")

    try:
        raw_frame, sector_frame, summary = collect_sector_investor_flow(
            sector_map=sector_map,
            start=start,
            end=end,
        )
        if sector_frame.empty:
            raise RuntimeError("No investor-flow rows were collected for the tracked sectors.")

        upsert_investor_flow_raw(raw_frame, provider=FLOW_PROVIDER, market=market)
        upsert_investor_flow_sector(sector_frame, provider=FLOW_PROVIDER, market=market)
        update_ingest_watermark(
            dataset="investor_flow",
            watermark_key=end,
            status=str(summary["status"]),
            coverage_complete=bool(summary["coverage_complete"]),
            provider=FLOW_PROVIDER,
            details=summary,
            market=market,
        )
        record_ingest_run(
            dataset="investor_flow",
            reason="manual_refresh",
            provider=FLOW_PROVIDER,
            requested_start=start,
            requested_end=end,
            status=str(summary["status"]),
            coverage_complete=bool(summary["coverage_complete"]),
            failed_days=list(summary.get("failed_days", [])),
            failed_codes=dict(summary.get("failed_codes", {})),
            delta_keys=_sector_codes_from_map(sector_map),
            row_count=int(summary.get("rows", 0) or 0),
            predicted_requests=int(summary.get("predicted_requests", 0) or 0),
            processed_requests=int(summary.get("processed_requests", 0) or 0),
            summary=summary,
            market=market,
        )
        cached = read_sector_investor_flow(_sector_codes_from_map(sector_map), start, end, market=market)
        return ("LIVE", cached), summary
    except Exception as exc:
        logger.exception("Investor-flow refresh failed")
        fallback_status, fallback_frame = load_sector_investor_flow(
            sector_map=sector_map,
            start=start,
            end=end,
            market=market,
        )
        summary = {
            "status": fallback_status,
            "provider": FLOW_PROVIDER,
            "requested_start": start,
            "requested_end": end,
            "coverage_complete": False,
            "failed_days": [],
            "failed_codes": {"refresh": str(exc)},
            "predicted_requests": 0,
            "processed_requests": 0,
            "rows": int(len(fallback_frame)),
        }
        try:
            record_ingest_run(
                dataset="investor_flow",
                reason="manual_refresh",
                provider=FLOW_PROVIDER,
                requested_start=start,
                requested_end=end,
                status=fallback_status,
                coverage_complete=False,
                failed_days=[],
                failed_codes={"refresh": str(exc)},
                delta_keys=[],
                row_count=int(len(fallback_frame)),
                summary=summary,
                market=market,
            )
        except Exception:
            logger.debug("Investor-flow failure bookkeeping skipped", exc_info=True)
        return (fallback_status, fallback_frame), summary


def read_warm_status() -> dict[str, Any]:
    return read_dataset_status("investor_flow")


def get_investor_flow_artifact_key() -> tuple[int, int, str, str, str]:
    return get_dataset_artifact_key("investor_flow")


def probe_investor_flow_status() -> str:
    return probe_dataset_mode("investor_flow")


__all__ = [
    "DEFAULT_INVESTOR_TYPES",
    "DEFAULT_LOOKBACK_CALENDAR_DAYS",
    "FLOW_PROVIDER",
    "DataStatus",
    "LoaderResult",
    "collect_sector_investor_flow",
    "get_investor_flow_artifact_key",
    "load_sector_investor_flow",
    "probe_investor_flow_status",
    "read_warm_status",
    "run_manual_investor_flow_refresh",
]
