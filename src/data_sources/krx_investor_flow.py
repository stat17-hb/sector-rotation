"""Experimental KRX investor-flow refresh and warehouse readers.

The dashboard never probes the live KRX path during page load. Manual refresh
uses pykrx's investor-by-ticker view, which itself depends on the same KRX web
backend family as the research prototype in ``python.collectors.krx_investor_flow``.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
import logging
import socket
from typing import Any, Callable, Literal, Mapping, TypedDict

import pandas as pd

from src.data_sources.krx_constituents import (
    PYKRX_CONSTITUENT_HISTORY_FLOOR,
    candidate_reference_dates as _candidate_reference_dates,
    lookup_index_constituents,
    normalize_constituent_result as _normalize_constituent_result,
)
from src.data_sources.pykrx_compat import (
    ensure_pykrx_transport_compat,
    request_krx_data,
    reset_krx_shared_session,
)
from src.data_sources.warehouse import (
    get_dataset_artifact_key,
    read_investor_flow_backfill_progress_cursor,
    read_investor_flow_operational_complete_cursor,
    read_latest_investor_flow_failed_days,
    read_latest_investor_flow_run,
    read_investor_flow_runs,
    read_market_prices,
    probe_dataset_mode,
    read_dataset_status,
    read_latest_sector_constituents_snapshot,
    read_sector_investor_flow,
    record_investor_flow_run_failure,
    record_ingest_run,
    upsert_sector_constituents_snapshot,
    write_investor_flow_backfill_chunk,
    write_investor_flow_operational_result,
)


logger = logging.getLogger(__name__)
KRX_DATA_HOST = "data.krx.co.kr"

DataStatus = Literal["LIVE", "CACHED", "SAMPLE"]
LoaderResult = tuple[DataStatus, pd.DataFrame]
ProgressCallback = Callable[[dict[str, Any]], None]


class DiscoveryDetails(TypedDict):
    requested_day: str
    oldest_collectable_date: str
    requested_earliest_candidate: str
    collector_contract_floor: str
    discovery_window_start: str
    discovery_window_end: str
    discovery_window_days: int
    used_cached_constituent_fallback: bool
    provider: str
    status: str

FLOW_PROVIDER = "PYKRX_UNOFFICIAL"
DEFAULT_LOOKBACK_CALENDAR_DAYS = 120
KRX_INVESTOR_FLOW_WEB_HISTORY_FLOOR = "20140501"
DEFAULT_BACKFILL_EARLIEST_CANDIDATE = KRX_INVESTOR_FLOW_WEB_HISTORY_FLOOR
DEFAULT_BACKFILL_DISCOVERY_WINDOW_DAYS = 31
DEFAULT_INVESTOR_TYPES: tuple[str, ...] = ("개인", "외국인", "기관합계")
OPERATIONAL_INVESTOR_FLOW_REASONS: tuple[str, ...] = ("manual_refresh",)
HISTORICAL_BACKFILL_REASON = "historical_backfill"
HISTORICAL_BACKFILL_DISCOVERY_REASON = "historical_backfill_discovery"
HISTORICAL_BACKFILL_VALIDATION_REASON = "historical_backfill_validation"
CACHED_CONSTITUENT_FALLBACK_PREFIX = "CACHED_FALLBACK("
DEFAULT_KR_CALENDAR_BENCHMARK_CODE = "1001"
RETRY_SESSION_RESET_BATCH_SIZE = 25
INVESTOR_FLOW_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "개인": ("개인",),
    "외국인": ("외국인합계", "외국인"),
    "기관합계": ("기관합계",),
}
DETAIL_INSTITUTION_COLUMNS: tuple[str, ...] = (
    "금융투자",
    "보험",
    "투신",
    "사모",
    "은행",
    "기타금융",
    "연기금",
)
DETAIL_FOREIGN_COLUMNS: tuple[str, ...] = ("외국인", "기타외국인")
TRADING_VALUE_GENERAL_COLUMNS: tuple[str, ...] = (
    "기관합계",
    "기타법인",
    "개인",
    "외국인합계",
    "전체",
)
TRADING_VALUE_DETAIL_COLUMNS: tuple[str, ...] = (
    "금융투자",
    "보험",
    "투신",
    "사모",
    "은행",
    "기타금융",
    "연기금",
    "기타법인",
    "개인",
    "외국인",
    "기타외국인",
    "전체",
)


def _emit_progress(
    progress_callback: ProgressCallback | None,
    *,
    task: str,
    phase: str,
    pct: float | int,
    detail: str = "",
    status: str = "running",
    meta: dict[str, Any] | None = None,
) -> None:
    if progress_callback is None:
        return
    try:
        pct_value = int(round(float(pct)))
    except Exception:
        pct_value = 0
    event = {
        "task": str(task).strip(),
        "phase": str(phase).strip(),
        "pct": max(0, min(100, pct_value)),
        "detail": str(detail).strip(),
        "status": str(status).strip().lower() or "running",
    }
    if meta:
        event["meta"] = dict(meta)
    progress_callback(event)


@dataclass(frozen=True)
class SectorUniverse:
    sector_codes: list[str]
    sector_names: dict[str, str]
    ticker_to_sector_codes: dict[str, list[str]]
    failed_sector_codes: dict[str, str] = field(default_factory=dict)


def _is_warehouse_write_lock_error(exc: BaseException) -> bool:
    lowered = str(exc).lower()
    return (
        "cannot acquire write lock on warehouse.duckdb" in lowered
        or "same database file with a different configuration than existing connections" in lowered
        or "file is already open" in lowered
    )


def _normalize_yyyymmdd(value: str | None) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _weekday_range_strings(start: str, end: str) -> list[str]:
    return [ts.strftime("%Y%m%d") for ts in pd.bdate_range(start, end)]


def _requested_trading_days(
    start: str,
    end: str,
    *,
    market: str = "KR",
    stock_module: Any | None = None,
    benchmark_code: str = DEFAULT_KR_CALENDAR_BENCHMARK_CODE,
) -> list[str]:
    return _requested_trading_days_with_source(
        start,
        end,
        market=market,
        stock_module=stock_module,
        benchmark_code=benchmark_code,
    )[0]


def _requested_trading_days_with_source(
    start: str,
    end: str,
    *,
    market: str = "KR",
    stock_module: Any | None = None,
    benchmark_code: str = DEFAULT_KR_CALENDAR_BENCHMARK_CODE,
) -> tuple[list[str], str]:
    weekday_days = _weekday_range_strings(start, end)
    if str(market).strip().upper() != "KR":
        return weekday_days, "weekday_range"
    try:
        if stock_module is None:
            ensure_pykrx_transport_compat()
            from pykrx import stock as stock_module  # type: ignore[import]
        calendar_frame = stock_module.get_index_ohlcv(
            _normalize_yyyymmdd(start),
            _normalize_yyyymmdd(end),
            str(benchmark_code).strip() or DEFAULT_KR_CALENDAR_BENCHMARK_CODE,
            name_display=False,
        )
        if calendar_frame is not None and not calendar_frame.empty:
            return [
                pd.Timestamp(item).strftime("%Y%m%d")
                for item in pd.DatetimeIndex(calendar_frame.index)
            ], "pykrx_calendar"
    except Exception as exc:
        logger.debug("Investor-flow calendar probe failed for %s..%s: %s", start, end, exc)
    return weekday_days, "weekday_fallback"


def _warehouse_trading_days(
    start: str,
    end: str,
    *,
    market: str = "KR",
    benchmark_code: str = DEFAULT_KR_CALENDAR_BENCHMARK_CODE,
) -> list[str]:
    if str(market).strip().upper() != "KR":
        return []
    try:
        frame = read_market_prices([str(benchmark_code).strip() or DEFAULT_KR_CALENDAR_BENCHMARK_CODE], start, end, market=market)
    except Exception as exc:
        logger.debug("Investor-flow warehouse trading-day lookup failed for %s..%s: %s", start, end, exc)
        return []
    if frame is None or frame.empty:
        return []
    return sorted({pd.Timestamp(item).strftime("%Y%m%d") for item in pd.DatetimeIndex(frame.index)})


def _resolve_frozen_trading_day_truth(
    start: str,
    end: str,
    *,
    market: str = "KR",
    stock_module: Any | None = None,
    benchmark_code: str = DEFAULT_KR_CALENDAR_BENCHMARK_CODE,
) -> tuple[list[str], str, bool]:
    requested_days, source = _requested_trading_days_with_source(
        start,
        end,
        market=market,
        stock_module=stock_module,
        benchmark_code=benchmark_code,
    )
    if source != "weekday_fallback":
        return requested_days, source, True

    warehouse_days = _warehouse_trading_days(
        start,
        end,
        market=market,
        benchmark_code=benchmark_code,
    )
    if warehouse_days:
        return warehouse_days, "warehouse_market_prices", True

    return [], source, False


def _next_business_day_str(value: str, *, market: str = "KR") -> str:
    current = pd.Timestamp(_normalize_yyyymmdd(value)).normalize()
    next_start = (current + timedelta(days=1)).strftime("%Y%m%d")
    next_end = (current + timedelta(days=14)).strftime("%Y%m%d")
    next_days = _requested_trading_days(next_start, next_end, market=market)
    if not next_days:
        raise RuntimeError(f"Cannot resolve next business day after {value}.")
    return next_days[0]


def _check_socket_stack() -> None:
    """Fail fast when the socket stack or KRX DNS resolution is unavailable.

    In the affected Windows sessions, both Python requests and even `curl.exe`
    fail before DNS resolution with WinError 10106. Detect that upfront so the
    UI does not report a misleading "no investor-flow rows" message.
    """
    test_socket = None
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except OSError as exc:
        raise RuntimeError(
            f"Windows socket stack is unavailable for live investor-flow refresh: {exc}. "
            "This is an environment/Winsock issue, not a KRX data issue."
        ) from exc
    try:
        socket.getaddrinfo(KRX_DATA_HOST, 443)
    except OSError as exc:
        raise RuntimeError(
            f"KRX host resolution failed for live investor-flow refresh: {exc}. "
            "This is a connectivity/DNS issue, not a constituent-data issue."
        ) from exc
    finally:
        if test_socket is not None:
            try:
                test_socket.close()
            except OSError:
                pass
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
    market: str = "KR",
) -> SectorUniverse:
    """Build ticker→sector mapping by querying KRX constituent lists.

    ``reference_date`` must already be a resolved (non-weekend) trading day —
    the caller is responsible for the calendar lookup so that this function
    never calls pykrx OHLCV internally.

    When live KRX queries return empty for all sectors the function falls back
    to the most recent ``sector_constituents_snapshot`` stored in the warehouse,
    marking returned tickers with ``is_fallback=True`` in the summary.
    """
    ensure_pykrx_transport_compat()
    from pykrx import stock  # type: ignore[import]

    entries = _tracked_sector_entries(sector_map)
    ticker_to_sector_codes: dict[str, list[str]] = defaultdict(list)
    sector_names: dict[str, str] = {}
    sector_codes: list[str] = []
    failed_sector_codes: dict[str, str] = {}
    # Use the passed reference_date directly — no internal pykrx calendar call.
    candidate_dates = _candidate_reference_dates(reference_date, periods=20)

    snapshot_rows: list[dict[str, Any]] = []

    for entry in entries:
        sector_code = str(entry["code"])
        sector_name = str(entry["name"])
        sector_names[sector_code] = sector_name
        sector_codes.append(sector_code)

        lookup = lookup_index_constituents(
            stock,
            sector_code=sector_code,
            candidate_dates=candidate_dates,
            logger=logger,
        )
        ticker_list = list(lookup.tickers)

        if not ticker_list:
            failed_sector_codes[sector_code] = lookup.failure_detail
        else:
            if lookup.source == "krx_raw_payload":
                logger.info(
                    "Recovered constituent lookup for %s via raw KRX payload on %s.",
                    sector_code,
                    lookup.resolved_from,
                )
            for ticker in ticker_list:
                snapshot_rows.append(
                    {
                        "sector_code": sector_code,
                        "ticker": ticker,
                        "reference_date": lookup.resolved_from,
                        "resolved_from": lookup.resolved_from,
                        "is_fallback": False,
                    }
                )

        for ticker in ticker_list:
            if sector_code not in ticker_to_sector_codes[ticker]:
                ticker_to_sector_codes[ticker].append(sector_code)

    # Save successful live constituent snapshot so future fallbacks can use it
    if snapshot_rows:
        try:
            upsert_sector_constituents_snapshot(
                snapshot_rows,
                snapshot_date=reference_date,
                provider="PYKRX_UNOFFICIAL",
                market=market,
            )
        except Exception as exc:
            logger.warning("Failed to persist constituent snapshot: %s", exc)

    # If ALL sectors failed live lookup, attempt warehouse snapshot fallback
    if not ticker_to_sector_codes and entries:
        logger.warning(
            "All %d sector constituent lookups failed; attempting cached snapshot fallback.",
            len(entries),
        )
        cached_snap = read_latest_sector_constituents_snapshot(
            [e["code"] for e in entries], market=market
        )
        if not cached_snap.empty:
            snap_date = str(cached_snap["snapshot_date"].iloc[0])
            logger.info("Using cached constituent snapshot from %s.", snap_date)
            for _, row in cached_snap.iterrows():
                sc = str(row["sector_code"])
                tk = str(row["ticker"])
                if sc not in sector_names:
                    continue
                ticker_to_sector_codes[tk].append(sc)
            # Mark sectors that were recovered from fallback
            for sc in sector_names:
                if sc in failed_sector_codes:
                    failed_sector_codes[sc] = (
                        f"{CACHED_CONSTITUENT_FALLBACK_PREFIX}from {snap_date}): "
                        + failed_sector_codes[sc]
                    )

    return SectorUniverse(
        sector_codes=sector_codes,
        sector_names=sector_names,
        ticker_to_sector_codes={
            ticker: sorted(codes) for ticker, codes in ticker_to_sector_codes.items()
        },
        failed_sector_codes=dict(failed_sector_codes),
    )


def _resolve_frame_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    available = {str(column).strip(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in available:
            return available[candidate]
    return None


def _validate_trading_value_frames(
    *,
    buy_frame: pd.DataFrame,
    sell_frame: pd.DataFrame,
    net_frame: pd.DataFrame,
    investor_types: tuple[str, ...],
    leg_details: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    frames = {"buy": buy_frame, "sell": sell_frame, "net": net_frame}
    empty_labels = [label for label, frame in frames.items() if frame is None or frame.empty]
    if len(empty_labels) >= 2:
        error_legs = sorted(
            leg
            for leg in empty_labels
            if any(
                str(attempt.get("status", "")).strip() == "error"
                for attempt in list(dict((leg_details or {})).get(leg, {}).get("attempts") or [])
            )
        )
        family = "exception_backed_empty" if error_legs else "all_builders_empty"
        joined = "/".join(empty_labels)
        reason = f"{joined} frames empty [{family}]"
        return {
            "family": family,
            "empty_legs": list(empty_labels),
            "error_legs": error_legs,
            "final_reason": reason,
        }
    for investor_type in investor_types:
        candidates = INVESTOR_FLOW_COLUMN_CANDIDATES.get(str(investor_type), (str(investor_type),))
        missing = [
            label
            for label, frame in frames.items()
            if frame is not None and not frame.empty and _resolve_frame_column(frame, candidates) is None
        ]
        if missing:
            return {
                "family": "missing_investor_columns",
                "missing_legs": list(missing),
                "investor_type": str(investor_type),
                "final_reason": f"missing investor columns for {investor_type}: {', '.join(missing)}",
            }
    return None


def _short_exception_detail(exc: BaseException) -> str:
    detail = f"{type(exc).__name__}: {exc}".strip()
    return detail[:200]


def _fetch_first_non_empty_frame(*builders) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    attempts: list[dict[str, Any]] = []
    for name, builder in builders:
        try:
            frame = builder()
        except Exception as exc:
            attempts.append(
                {
                    "builder": str(name),
                    "status": "error",
                    "detail": _short_exception_detail(exc),
                }
            )
            frame = pd.DataFrame()
        else:
            if frame is None or frame.empty:
                attempts.append({"builder": str(name), "status": "empty"})
                continue
            attempts.append(
                {
                    "builder": str(name),
                    "status": "success",
                    "rows": int(len(frame)),
                }
            )
            return frame, attempts, str(name)
    return pd.DataFrame(), attempts, ""


def _fetch_ticker_trading_value_frames(
    stock_module,
    *,
    ticker: str,
    start: str,
    end: str,
    allow_wrapper_fallback: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    def _wrapper_frame(on: str) -> pd.DataFrame:
        return stock_module.get_market_trading_value_by_date(start, end, ticker, on=on)

    def _resolve_for(on: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        builders = [
            (
                "raw_general",
                lambda: _fetch_ticker_trading_value_frame_raw(
                    ticker=ticker,
                    start=start,
                    end=end,
                    on=on,
                    detail=False,
                ),
            ),
            (
                "raw_detail",
                lambda: _fetch_ticker_trading_value_frame_raw(
                    ticker=ticker,
                    start=start,
                    end=end,
                    on=on,
                    detail=True,
                ),
            ),
        ]
        if allow_wrapper_fallback:
            builders.extend(
                [
                    (
                        "wrapper_detail",
                        lambda: _fetch_ticker_trading_value_frame_detailed(
                            stock_module,
                            ticker=ticker,
                            start=start,
                            end=end,
                            on=on,
                        ),
                    ),
                    (
                        "wrapper_general",
                        lambda: _wrapper_frame(on),
                    ),
                ]
            )
        frame, attempts, selected_builder = _fetch_first_non_empty_frame(*builders)
        return frame, {
            "selected_builder": selected_builder,
            "attempts": attempts,
        }

    buy_frame, buy_detail = _resolve_for("매수")
    sell_frame, sell_detail = _resolve_for("매도")
    net_frame, net_detail = _resolve_for("순매수")
    return buy_frame, sell_frame, net_frame, {
        "buy": buy_detail,
        "sell": sell_detail,
        "net": net_detail,
    }


def _classify_failed_ticker(
    *,
    ticker: str,
    validation_detail: Mapping[str, Any] | None,
    leg_details: Mapping[str, Any],
    normalized_empty: bool = False,
) -> tuple[str, dict[str, Any]]:
    if validation_detail is not None:
        family = str(validation_detail.get("family", "")).strip() or "validation_error"
        final_reason = str(validation_detail.get("final_reason", "")).strip() or family
    elif normalized_empty:
        family = "normalized_empty"
        final_reason = "normalized investor-flow frame empty"
    else:
        family = "unknown"
        final_reason = "unknown investor-flow failure"

    detail = {
        "ticker": str(ticker),
        "family": family,
        "final_reason": final_reason,
        "legs": dict(leg_details),
    }
    if validation_detail is not None:
        for key in ("empty_legs", "error_legs", "missing_legs", "investor_type"):
            if key in validation_detail:
                detail[key] = validation_detail.get(key)
    return final_reason, detail


def _build_failed_ticker_family_counts(failed_ticker_detail: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for detail in dict(failed_ticker_detail or {}).values():
        family = str(dict(detail or {}).get("family", "")).strip() or "unknown"
        counts[family] = counts.get(family, 0) + 1
    return counts


def _collapse_detail_trading_value_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    collapsed = pd.DataFrame(index=pd.DatetimeIndex(frame.index))
    institution_cols = [col for col in DETAIL_INSTITUTION_COLUMNS if col in frame.columns]
    foreign_cols = [col for col in DETAIL_FOREIGN_COLUMNS if col in frame.columns]

    if institution_cols:
        collapsed["기관합계"] = (
            frame[institution_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        )
    if "개인" in frame.columns:
        collapsed["개인"] = pd.to_numeric(frame["개인"], errors="coerce").fillna(0)
    if foreign_cols:
        collapsed["외국인합계"] = (
            frame[foreign_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        )
    collapsed["전체"] = 0
    return collapsed


def _extract_krx_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("output", "OutBlock_1", "outBlock_1", "block1", "result", "data", "items", "list"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            nested = _extract_krx_rows(value)
            if nested:
                return nested
    for value in payload.values():
        if isinstance(value, dict):
            nested = _extract_krx_rows(value)
            if nested:
                return nested
    return []


def _clean_krx_int(value: Any) -> int:
    text = str(value or "").strip()
    if not text or text == "-":
        return 0
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    if text.startswith("-"):
        negative = True
        text = text[1:]
    digits = "".join(ch for ch in text if ch.isdigit())
    number = int(digits or "0")
    return -number if negative else number


def _parse_raw_trading_value_payload(payload: Any, *, detail: bool) -> pd.DataFrame:
    rows = _extract_krx_rows(payload)
    if not rows:
        return pd.DataFrame()

    columns = TRADING_VALUE_DETAIL_COLUMNS if detail else TRADING_VALUE_GENERAL_COLUMNS
    records: list[dict[str, Any]] = []
    for row in rows:
        trade_date = str(row.get("TRD_DD", "")).strip()
        if not trade_date:
            continue
        record: dict[str, Any] = {"날짜": trade_date}
        for index, column in enumerate(columns, start=1):
            key = "TRDVAL_TOT" if column == "전체" else f"TRDVAL{index}"
            record[column] = _clean_krx_int(row.get(key))
        records.append(record)

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records).set_index("날짜")
    frame.index = pd.to_datetime(frame.index, format="%Y/%m/%d", errors="coerce")
    frame = frame[frame.index.notna()]
    frame.index.name = "날짜"
    return frame.sort_index()


def _fetch_ticker_trading_value_frame_detailed(
    stock_module,
    *,
    ticker: str,
    start: str,
    end: str,
    on: str,
) -> pd.DataFrame:
    detail_frame = stock_module.get_market_trading_value_by_date(
        start,
        end,
        ticker,
        on=on,
        detail=True,
    )
    return _collapse_detail_trading_value_frame(detail_frame)


def _fetch_ticker_trading_value_frame_raw(
    *,
    ticker: str,
    start: str,
    end: str,
    on: str,
    detail: bool,
) -> pd.DataFrame:
    ensure_pykrx_transport_compat()
    from pykrx.website.krx.market.ticker import get_stock_ticker_isin  # type: ignore[import]

    ask_bid = {"매도": 1, "매수": 2, "순매수": 3}.get(on, 3)
    payload = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT02303" if detail else "dbms/MDC/STAT/standard/MDCSTAT02302",
        "locale": "ko_KR",
        "isuCd": get_stock_ticker_isin(ticker),
        "inqTpCd": 2,
        "trdVolVal": 2,
        "askBid": ask_bid,
        "strtDd": start,
        "endDd": end,
    }
    if detail:
        payload["detailView"] = 1

    response = request_krx_data(payload)
    parsed = _parse_raw_trading_value_payload(response.json(), detail=detail)
    return _collapse_detail_trading_value_frame(parsed) if detail else parsed


def _normalize_ticker_trading_value_frames(
    *,
    ticker: str,
    ticker_name: str,
    buy_frame: pd.DataFrame,
    sell_frame: pd.DataFrame,
    net_frame: pd.DataFrame,
    investor_types: tuple[str, ...],
) -> pd.DataFrame:
    dates = pd.Index([])
    for frame in (buy_frame, sell_frame, net_frame):
        if frame is not None and not frame.empty:
            dates = dates.union(pd.DatetimeIndex(frame.index))
    if dates.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for investor_type in investor_types:
        candidates = INVESTOR_FLOW_COLUMN_CANDIDATES.get(str(investor_type), (str(investor_type),))
        buy_col = _resolve_frame_column(buy_frame, candidates) if buy_frame is not None and not buy_frame.empty else None
        sell_col = _resolve_frame_column(sell_frame, candidates) if sell_frame is not None and not sell_frame.empty else None
        net_col = _resolve_frame_column(net_frame, candidates) if net_frame is not None and not net_frame.empty else None
        if buy_col is None and sell_col is None and net_col is None:
            continue

        buy_series_raw = (
            pd.to_numeric(buy_frame[buy_col], errors="coerce").reindex(dates).fillna(0)
            if buy_col is not None
            else None
        )
        sell_series_raw = (
            pd.to_numeric(sell_frame[sell_col], errors="coerce").reindex(dates).fillna(0)
            if sell_col is not None
            else None
        )
        net_series_raw = (
            pd.to_numeric(net_frame[net_col], errors="coerce").reindex(dates).fillna(0)
            if net_col is not None
            else None
        )

        if buy_series_raw is None and sell_series_raw is not None and net_series_raw is not None:
            buy_series_raw = sell_series_raw + net_series_raw
        if sell_series_raw is None and buy_series_raw is not None and net_series_raw is not None:
            sell_series_raw = buy_series_raw - net_series_raw
        if net_series_raw is None and buy_series_raw is not None and sell_series_raw is not None:
            net_series_raw = buy_series_raw - sell_series_raw

        if buy_series_raw is None or sell_series_raw is None or net_series_raw is None:
            continue

        buy_series = buy_series_raw
        sell_series = sell_series_raw
        net_series = net_series_raw

        rows.append(
            pd.DataFrame(
                {
                    "trade_date": dates,
                    "ticker": str(ticker),
                    "ticker_name": str(ticker_name or ticker),
                    "investor_type": str(investor_type),
                    "buy_amount": buy_series.astype("int64").values,
                    "sell_amount": sell_series.astype("int64").values,
                    "net_buy_amount": net_series.astype("int64").values,
                }
            )
        )

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


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


def _split_failed_code_map(failed_codes: Mapping[str, str] | None) -> tuple[dict[str, str], dict[str, str]]:
    normalized = {
        str(key).strip(): str(value).strip()
        for key, value in dict(failed_codes or {}).items()
        if str(key).strip() and str(value).strip()
    }
    sector_failures = {key: value for key, value in normalized.items() if key.startswith("sector:")}
    ticker_failures = {
        key: value
        for key, value in normalized.items()
        if key not in sector_failures and key not in {"refresh", "warehouse"}
    }
    return sector_failures, ticker_failures


def _collect_ticker_flow_rows(
    *,
    stock_module,
    tickers: list[str],
    start: str,
    end: str,
    investor_types: tuple[str, ...],
    reset_session_between_batches: bool,
    allow_wrapper_fallback: bool,
    ticker_name_resolver,
    progress_callback: ProgressCallback | None = None,
    progress_floor: int = 0,
    progress_ceiling: int = 100,
    progress_phase: str = "ticker collection",
) -> tuple[list[pd.DataFrame], dict[str, str], int, dict[str, dict[str, Any]]]:
    raw_frames: list[pd.DataFrame] = []
    failed_codes: dict[str, str] = {}
    failed_detail: dict[str, dict[str, Any]] = {}
    processed_requests = 0
    predicted_requests = max(0, len(tickers) * 3)
    last_pct = -1

    def _emit_collection_progress(current_ticker: str) -> None:
        nonlocal last_pct
        if progress_callback is None or predicted_requests <= 0:
            return
        ratio = processed_requests / predicted_requests
        pct = progress_floor + (progress_ceiling - progress_floor) * ratio
        pct_value = max(progress_floor, min(progress_ceiling, int(round(pct))))
        if pct_value == last_pct:
            return
        last_pct = pct_value
        _emit_progress(
            progress_callback,
            task="투자자수급 갱신",
            phase=progress_phase,
            pct=pct_value,
            detail=f"{processed_requests}/{predicted_requests} logical steps",
            meta={
                "processed_requests": processed_requests,
                "predicted_requests": predicted_requests,
                "current_ticker": str(current_ticker),
                "retry": bool(reset_session_between_batches),
                "metric_semantics": "logical_ticker_steps",
            },
        )

    for index, ticker in enumerate(tickers):
        if reset_session_between_batches and index % RETRY_SESSION_RESET_BATCH_SIZE == 0:
            reset_krx_shared_session()
        processed_requests += 3
        try:
            buy_frame, sell_frame, net_frame, leg_details = _fetch_ticker_trading_value_frames(
                stock_module,
                ticker=ticker,
                start=start,
                end=end,
                allow_wrapper_fallback=allow_wrapper_fallback,
            )
        except Exception as exc:
            failed_codes[str(ticker)] = _short_exception_detail(exc)
            failed_detail[str(ticker)] = {
                "ticker": str(ticker),
                "family": "collector_exception",
                "final_reason": _short_exception_detail(exc),
                "legs": {},
            }
            _emit_collection_progress(str(ticker))
            continue

        validation_detail = _validate_trading_value_frames(
            buy_frame=buy_frame,
            sell_frame=sell_frame,
            net_frame=net_frame,
            investor_types=investor_types,
            leg_details=leg_details,
        )
        if validation_detail is not None:
            final_reason, detail = _classify_failed_ticker(
                ticker=str(ticker),
                validation_detail=validation_detail,
                leg_details=leg_details,
            )
            failed_codes[str(ticker)] = final_reason
            failed_detail[str(ticker)] = detail
            _emit_collection_progress(str(ticker))
            continue

        ticker_name = ""
        try:
            ticker_name = str(ticker_name_resolver(ticker) or ticker)
        except Exception:
            ticker_name = str(ticker)

        normalized = _normalize_ticker_trading_value_frames(
            ticker=str(ticker),
            ticker_name=ticker_name,
            buy_frame=buy_frame,
            sell_frame=sell_frame,
            net_frame=net_frame,
            investor_types=investor_types,
        )
        if normalized.empty:
            final_reason, detail = _classify_failed_ticker(
                ticker=str(ticker),
                validation_detail=None,
                leg_details=leg_details,
                normalized_empty=True,
            )
            failed_codes[str(ticker)] = final_reason
            failed_detail[str(ticker)] = detail
            _emit_collection_progress(str(ticker))
            continue
        raw_frames.append(normalized)
        _emit_collection_progress(str(ticker))

    return raw_frames, failed_codes, processed_requests, failed_detail


def collect_sector_investor_flow(
    *,
    sector_map: dict[str, Any],
    start: str,
    end: str,
    investor_types: tuple[str, ...] = DEFAULT_INVESTOR_TYPES,
    market: str = "KR",
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _emit_progress(
        progress_callback,
        task="투자자수급 갱신",
        phase="거래일 범위 확인 중",
        pct=10,
        detail=f"{start} ~ {end}",
    )
    _check_socket_stack()
    ensure_pykrx_transport_compat()
    from pykrx import stock  # type: ignore[import]
    requested_days, failed_day_source, failed_day_truth_confident = _resolve_frozen_trading_day_truth(
        start,
        end,
        market=market,
        stock_module=stock,
    )

    reference_date = str(end)
    universe = _build_sector_universe(sector_map, reference_date=reference_date, market=market)
    relevant_tickers = set(universe.ticker_to_sector_codes)
    sorted_tickers = sorted(relevant_tickers)
    _emit_progress(
        progress_callback,
        task="투자자수급 갱신",
        phase="구성종목 확인 완료",
        pct=15,
        detail=f"{len(universe.sector_codes)} sectors · {len(relevant_tickers)} tickers",
    )

    failed_days: list[str] = []
    failed_codes: dict[str, str] = {
        f"sector:{code}": detail
        for code, detail in universe.failed_sector_codes.items()
    }
    request_metric_semantics = "logical_ticker_steps"
    initial_raw_frames, initial_failed_tickers, processed_requests, initial_failed_ticker_detail = _collect_ticker_flow_rows(
        stock_module=stock,
        tickers=sorted_tickers,
        start=start,
        end=end,
        investor_types=investor_types,
        reset_session_between_batches=False,
        allow_wrapper_fallback=True,
        ticker_name_resolver=stock.get_market_ticker_name,
        progress_callback=progress_callback,
        progress_floor=20,
        progress_ceiling=70,
        progress_phase="1차 수집 중",
    )
    raw_frames = list(initial_raw_frames)
    failed_codes.update(initial_failed_tickers)
    failed_ticker_detail = dict(initial_failed_ticker_detail)

    retry_targets = sorted(initial_failed_tickers)
    retried_ticker_count = len(retry_targets)
    retried_recovered_ticker_count = 0
    retry_processed_requests = 0
    if retry_targets:
        _emit_progress(
            progress_callback,
            task="투자자수급 갱신",
            phase="재시도 준비 중",
            pct=70,
            detail=f"{len(retry_targets)} tickers",
        )
        retry_raw_frames, retry_failed_tickers, retry_processed_requests, retry_failed_ticker_detail = _collect_ticker_flow_rows(
            stock_module=stock,
            tickers=retry_targets,
            start=start,
            end=end,
            investor_types=investor_types,
            reset_session_between_batches=True,
            allow_wrapper_fallback=True,
            ticker_name_resolver=stock.get_market_ticker_name,
            progress_callback=progress_callback,
            progress_floor=70,
            progress_ceiling=90,
            progress_phase="재시도 수집 중",
        )
        retry_recovered_tickers = sorted(set(retry_targets) - set(retry_failed_tickers))
        retried_recovered_ticker_count = len(retry_recovered_tickers)
        raw_frames.extend(retry_raw_frames)
        for ticker in retry_recovered_tickers:
            failed_codes.pop(ticker, None)
            failed_ticker_detail.pop(ticker, None)
        for ticker, detail in retry_failed_tickers.items():
            failed_codes[ticker] = detail
        for ticker, detail in retry_failed_ticker_detail.items():
            failed_ticker_detail[ticker] = detail

    processed_requests += retry_processed_requests
    predicted_requests = (len(sorted_tickers) + retried_ticker_count) * 3

    raw_frame = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    if not raw_frame.empty:
        successful_days = {
            pd.Timestamp(item).strftime("%Y%m%d")
            for item in pd.DatetimeIndex(raw_frame["trade_date"])
        }
        failed_days = (
            sorted(set(requested_days) - successful_days)
            if failed_day_truth_confident
            else []
        )
    else:
        failed_days = list(requested_days) if failed_day_truth_confident else []

    sector_frame = _aggregate_sector_flow(raw_frame, universe)
    _emit_progress(
        progress_callback,
        task="투자자수급 갱신",
        phase="섹터 집계 완료",
        pct=90,
        detail=f"{len(sector_frame)} rows",
    )
    failed_sector_codes, failed_ticker_codes = _split_failed_code_map(failed_codes)
    failed_ticker_family_counts = _build_failed_ticker_family_counts(failed_ticker_detail)
    summary = {
        "status": "LIVE" if not sector_frame.empty else "SAMPLE",
        "provider": FLOW_PROVIDER,
        "requested_start": str(start),
        "requested_end": str(end),
        "coverage_complete": (
            not failed_codes
            and not failed_days
            and not sector_frame.empty
            and failed_day_truth_confident
        ),
        "failed_days": failed_days,
        "failed_codes": failed_codes,
        "failed_sector_codes": failed_sector_codes,
        "failed_ticker_codes": failed_ticker_codes,
        "failed_ticker_detail": failed_ticker_detail,
        "failed_ticker_family_counts": failed_ticker_family_counts,
        "failed_sector_count": len(failed_sector_codes),
        "failed_ticker_count": len(failed_ticker_codes),
        "failed_day_source": failed_day_source,
        "failed_day_truth_confident": failed_day_truth_confident,
        "request_metric_semantics": request_metric_semantics,
        "predicted_requests": predicted_requests,
        "processed_requests": processed_requests,
        "retried_ticker_count": retried_ticker_count,
        "retried_recovered_ticker_count": retried_recovered_ticker_count,
        "retry_processed_requests": retry_processed_requests,
        "rows": int(len(sector_frame)),
        "tracked_sectors": len(universe.sector_codes),
        "tracked_tickers": len(relevant_tickers),
    }
    return raw_frame, sector_frame, summary


def _sector_codes_from_map(sector_map: dict[str, Any]) -> list[str]:
    return [entry["code"] for entry in _tracked_sector_entries(sector_map)]


def _collector_contract_floor() -> str:
    return max(KRX_INVESTOR_FLOW_WEB_HISTORY_FLOOR, PYKRX_CONSTITUENT_HISTORY_FLOOR)


def _used_cached_constituent_fallback(summary: Mapping[str, Any]) -> bool:
    values = [str(value) for value in dict(summary.get("failed_codes") or {}).values()]
    return any(value.startswith(CACHED_CONSTITUENT_FALLBACK_PREFIX) for value in values)


def _raw_frame_contains_requested_day(raw_frame: pd.DataFrame, *, requested_day: str) -> bool:
    if raw_frame.empty or "trade_date" not in raw_frame.columns:
        return False
    requested = _normalize_yyyymmdd(requested_day)
    if not requested:
        return False
    trade_dates = {
        pd.Timestamp(item).strftime("%Y%m%d")
        for item in pd.to_datetime(raw_frame["trade_date"], errors="coerce").dropna()
    }
    return requested in trade_dates


def _build_discovery_details(
    *,
    requested_day: str,
    requested_earliest_candidate: str,
    discovery_window_start: str,
    discovery_window_end: str,
    discovery_window_days: int,
    used_cached_constituent_fallback: bool,
    status: str,
) -> DiscoveryDetails:
    return {
        "requested_day": requested_day,
        "oldest_collectable_date": requested_day,
        "requested_earliest_candidate": requested_earliest_candidate,
        "collector_contract_floor": _collector_contract_floor(),
        "discovery_window_start": discovery_window_start,
        "discovery_window_end": discovery_window_end,
        "discovery_window_days": int(discovery_window_days),
        "used_cached_constituent_fallback": bool(used_cached_constituent_fallback),
        "provider": FLOW_PROVIDER,
        "status": str(status).strip().upper() or "LIVE",
    }


def _build_historical_backfill_failure_summary(
    *,
    day_str: str,
    failed_days: list[str],
    failed_codes: dict[str, str],
    written_rows: int,
    successful_days: list[str],
    oldest_collectable_date: str,
    track_progress: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "status": "SAMPLE",
        "provider": FLOW_PROVIDER,
        "requested_start": day_str,
        "requested_end": day_str,
        "coverage_complete": False,
        "failed_days": list(failed_days),
        "failed_codes": dict(failed_codes),
        "rows": int(written_rows),
        "successful_days": list(successful_days),
        "oldest_collectable_date": oldest_collectable_date,
        "mode": "raw_only_history",
        "track_progress": bool(track_progress),
        "reason": str(reason),
    }


def _summarize_operational_failure_shape(run: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(run.get("summary") or {})
    failed_codes = {
        str(key).strip(): str(value).strip()
        for key, value in dict(run.get("failed_codes") or {}).items()
        if str(key).strip() and str(value).strip()
    }
    failed_days = sorted(
        {
            normalized
            for normalized in (
                _normalize_yyyymmdd(day)
                for day in list(run.get("failed_days") or [])
            )
            if normalized
        }
    )
    date_scoped_code_days = sorted(
        {
            normalized
            for normalized in (_normalize_yyyymmdd(key) for key in failed_codes)
            if normalized
        }
    )
    failed_days = sorted(set(failed_days).union(date_scoped_code_days))
    sector_failures = sum(1 for key in failed_codes if key.startswith("sector:"))
    ticker_failures = sum(
        1
        for key in failed_codes
        if not key.startswith("sector:") and not _normalize_yyyymmdd(key)
    )
    sector_failures = max(sector_failures, int(summary.get("failed_sector_count", 0) or 0))
    ticker_failures = max(ticker_failures, int(summary.get("failed_ticker_count", 0) or 0))
    return {
        "failed_days": failed_days,
        "failed_day_truth_confident": bool(summary.get("failed_day_truth_confident", False)),
        "failed_sector_count": sector_failures,
        "failed_ticker_count": ticker_failures,
    }


def _resolve_unresolved_operational_replay(
    *,
    market: str,
    complete_cursor: str,
) -> dict[str, Any]:
    """Return a durable replay plan when unresolved manual-refresh history exists."""
    unresolved_runs: list[dict[str, Any]] = []
    for run in read_investor_flow_runs(market=market, reasons=OPERATIONAL_INVESTOR_FLOW_REASONS):
        requested_end = _normalize_yyyymmdd(str(run.get("requested_end", "")))
        if complete_cursor and requested_end and requested_end <= complete_cursor:
            break
        if bool(run.get("coverage_complete")):
            if complete_cursor:
                break
            continue
        unresolved_runs.append(run)

    if not unresolved_runs:
        return {}

    anchor_candidates = [
        _normalize_yyyymmdd(str(run.get("requested_start", "")))
        for run in unresolved_runs
    ]
    anchor_candidates = [value for value in anchor_candidates if value]
    if not anchor_candidates:
        return {}

    shapes = [_summarize_operational_failure_shape(run) for run in unresolved_runs]
    failed_days = sorted({day for shape in shapes for day in shape["failed_days"]})
    has_ticker_failures = any(int(shape["failed_ticker_count"]) > 0 for shape in shapes)
    has_sector_failures = any(int(shape["failed_sector_count"]) > 0 for shape in shapes)
    truth_confident = all(bool(shape["failed_day_truth_confident"]) for shape in shapes)
    anchor_start = min(anchor_candidates)
    latest_run = unresolved_runs[0]
    latest_run_end = _normalize_yyyymmdd(str(latest_run.get("requested_end", "")))
    strictly_day_scoped = bool(failed_days) and truth_confident and not has_ticker_failures and not has_sector_failures
    if strictly_day_scoped:
        start = failed_days[0]
        mode = "repair_from_failed_day"
        anchor_reason = "earliest_failed_day_in_unresolved_history"
        repaired_days = list(failed_days)
    else:
        start = anchor_start
        mode = "replay_uncertain_span"
        anchor_reason = "earliest_unresolved_requested_start"
        repaired_days = []
    return {
        "mode": mode,
        "start": start,
        "complete_cursor": complete_cursor,
        "anchor_start": anchor_start,
        "anchor_reason": anchor_reason,
        "metadata_durable": True,
        "latest_run_reason": str(latest_run.get("reason", "")),
        "latest_run_end": latest_run_end,
        "failed_days_repaired": repaired_days,
        "failed_day_truth_confident": truth_confident,
        "has_ticker_failures": has_ticker_failures,
        "has_sector_failures": has_sector_failures,
        "history_run_count": len(unresolved_runs),
    }


def resolve_investor_flow_refresh_window(
    *,
    end_date_str: str,
    lookback_days: int = DEFAULT_LOOKBACK_CALENDAR_DAYS,
    market: str = "KR",
) -> tuple[str, str, dict[str, Any]]:
    requested_end = pd.Timestamp(end_date_str).normalize()
    end = requested_end.strftime("%Y%m%d")
    complete_cursor = read_investor_flow_operational_complete_cursor(market=market)
    latest_run = read_latest_investor_flow_run(
        market=market,
        reasons=OPERATIONAL_INVESTOR_FLOW_REASONS,
    )

    if not complete_cursor:
        unresolved = _resolve_unresolved_operational_replay(
            market=market,
            complete_cursor="",
        )
        if unresolved:
            start = str(unresolved.get("start", "")).strip()
            window_meta = dict(unresolved)
            window_meta.pop("start", None)
            return start, end, window_meta
        requested_start = (requested_end - timedelta(days=int(lookback_days))).normalize()
        start = requested_start.strftime("%Y%m%d")
        return start, end, {
            "mode": "bootstrap_seed",
            "complete_cursor": "",
            "anchor_start": start,
            "anchor_reason": "bootstrap_lookback",
            "metadata_durable": False,
            "latest_run_reason": str(latest_run.get("reason", "")),
            "latest_run_end": str(latest_run.get("requested_end", "")),
            "failed_days_repaired": [],
        }

    start = _next_business_day_str(complete_cursor, market=market)
    repaired_days: list[str] = []
    if latest_run and not bool(latest_run.get("coverage_complete")):
        repaired_days = sorted(
            day
            for day in read_latest_investor_flow_failed_days(
                market=market,
                reasons=OPERATIONAL_INVESTOR_FLOW_REASONS,
            )
            if _normalize_yyyymmdd(day) and _normalize_yyyymmdd(day) > complete_cursor
        )
        if repaired_days:
            start = repaired_days[0]

    return start, end, {
        "mode": "incremental",
        "complete_cursor": complete_cursor,
        "anchor_start": start,
        "anchor_reason": "next_business_day_after_complete_cursor" if not repaired_days else "failed_days_after_complete_cursor",
        "metadata_durable": True,
        "latest_run_reason": str(latest_run.get("reason", "")),
        "latest_run_end": str(latest_run.get("requested_end", "")),
        "failed_days_repaired": repaired_days,
    }


def load_sector_investor_flow(
    *,
    sector_map: dict[str, Any],
    start: str,
    end: str,
    market: str = "KR",
    allow_bootstrap_partial_preview: bool = False,
) -> LoaderResult:
    sector_codes = _sector_codes_from_map(sector_map)
    cached = read_sector_investor_flow(
        sector_codes,
        start,
        end,
        market=market,
        cap_to_operational_cursor=True,
    )
    if cached.empty:
        if allow_bootstrap_partial_preview and not read_investor_flow_operational_complete_cursor(market=market):
            partial = read_sector_investor_flow(
                sector_codes,
                start,
                end,
                market=market,
                cap_to_operational_cursor=False,
            )
            if not partial.empty:
                return ("CACHED", partial)
        return ("SAMPLE", pd.DataFrame())
    return ("CACHED", cached)


def discover_oldest_collectable_date(
    *,
    sector_map: dict[str, Any],
    end_date_str: str,
    market: str = "KR",
    earliest_candidate_str: str = DEFAULT_BACKFILL_EARLIEST_CANDIDATE,
    discovery_window_days: int = DEFAULT_BACKFILL_DISCOVERY_WINDOW_DAYS,
) -> tuple[str, dict[str, Any]]:
    requested_candidate = pd.Timestamp(_normalize_yyyymmdd(earliest_candidate_str)).normalize()
    history_floor = pd.Timestamp(_collector_contract_floor()).normalize()
    earliest_candidate = max(requested_candidate, history_floor)
    target_end = pd.Timestamp(_normalize_yyyymmdd(end_date_str)).normalize()
    if earliest_candidate > target_end:
        raise RuntimeError("Historical backfill discovery window is invalid.")

    coarse_cursor = earliest_candidate
    window_days = max(1, int(discovery_window_days))
    while coarse_cursor <= target_end:
        window_end = min(coarse_cursor + timedelta(days=window_days - 1), target_end)
        window_start_str = coarse_cursor.strftime("%Y%m%d")
        window_end_str = window_end.strftime("%Y%m%d")
        raw_frame, _, summary = collect_sector_investor_flow(
            sector_map=sector_map,
            start=window_start_str,
            end=window_end_str,
            market=market,
        )
        if not raw_frame.empty:
            for day_str in _requested_trading_days(window_start_str, window_end_str, market=market):
                day_raw, _, day_summary = collect_sector_investor_flow(
                    sector_map=sector_map,
                    start=day_str,
                    end=day_str,
                    market=market,
                )
                used_cached_fallback = _used_cached_constituent_fallback(day_summary)
                if _raw_frame_contains_requested_day(day_raw, requested_day=day_str) and not used_cached_fallback:
                    details = _build_discovery_details(
                        requested_day=day_str,
                        requested_earliest_candidate=requested_candidate.strftime("%Y%m%d"),
                        discovery_window_start=window_start_str,
                        discovery_window_end=window_end_str,
                        discovery_window_days=window_days,
                        used_cached_constituent_fallback=used_cached_fallback,
                        status=str(day_summary.get("status", "LIVE")),
                    )
                    record_ingest_run(
                        dataset="investor_flow",
                        reason=HISTORICAL_BACKFILL_DISCOVERY_REASON,
                        provider=FLOW_PROVIDER,
                        requested_start=window_start_str,
                        requested_end=window_end_str,
                        status=str(day_summary.get("status", "LIVE")).strip().upper() or "LIVE",
                        coverage_complete=bool(day_summary.get("coverage_complete")),
                        failed_days=list(day_summary.get("failed_days", [])),
                        failed_codes=dict(day_summary.get("failed_codes") or {}),
                        delta_keys=[],
                        row_count=int(len(day_raw)),
                        predicted_requests=int(summary.get("predicted_requests", 0) or 0),
                        processed_requests=int(summary.get("processed_requests", 0) or 0),
                        summary=details,
                        market=market,
                    )
                    return day_str, details
        coarse_cursor = window_end + timedelta(days=1)

    raise RuntimeError(
        "Unable to discover a collectable historical investor-flow date through the current KRX path."
    )


def run_historical_investor_flow_backfill(
    *,
    sector_map: dict[str, Any],
    end_date_str: str,
    market: str = "KR",
    start_date_str: str | None = None,
    oldest_collectable_date: str | None = None,
    earliest_candidate_str: str = DEFAULT_BACKFILL_EARLIEST_CANDIDATE,
    track_progress: bool = True,
    reason: str = HISTORICAL_BACKFILL_REASON,
) -> dict[str, Any]:
    target_end = pd.Timestamp(_normalize_yyyymmdd(end_date_str)).normalize()
    discovery_details: dict[str, Any] = {}
    discovered_start = _normalize_yyyymmdd(oldest_collectable_date)
    explicit_start = _normalize_yyyymmdd(start_date_str)
    if not discovered_start:
        discovered_start, discovery_details = discover_oldest_collectable_date(
            sector_map=sector_map,
            end_date_str=target_end.strftime("%Y%m%d"),
            market=market,
            earliest_candidate_str=earliest_candidate_str,
        )

    backfill_cursor = read_investor_flow_backfill_progress_cursor(market=market) if track_progress else None
    start_str = explicit_start or (_next_business_day_str(backfill_cursor, market=market) if backfill_cursor else discovered_start)
    start_ts = pd.Timestamp(start_str).normalize()
    if start_ts > target_end:
        return {
            "status": "CACHED",
            "provider": FLOW_PROVIDER,
            "requested_start": start_str,
            "requested_end": target_end.strftime("%Y%m%d"),
            "coverage_complete": True,
            "failed_days": [],
            "failed_codes": {},
            "rows": 0,
            "oldest_collectable_date": discovered_start,
            "mode": "raw_only_history",
            "discovery": discovery_details,
        }

    written_rows = 0
    successful_days: list[str] = []
    failed_days: list[str] = []
    failed_codes: dict[str, str] = {}
    for day_str in _requested_trading_days(start_ts.strftime("%Y%m%d"), target_end.strftime("%Y%m%d"), market=market):
        try:
            raw_frame, _, day_summary = collect_sector_investor_flow(
                sector_map=sector_map,
                start=day_str,
                end=day_str,
                market=market,
            )
        except Exception as exc:
            failed_days = [day_str]
            failed_codes = {day_str: str(exc)}
            failure_summary = _build_historical_backfill_failure_summary(
                day_str=day_str,
                failed_days=failed_days,
                failed_codes=failed_codes,
                written_rows=written_rows,
                successful_days=successful_days,
                oldest_collectable_date=discovered_start,
                track_progress=track_progress,
                reason=reason,
            )
            record_investor_flow_run_failure(
                reason=reason,
                provider=FLOW_PROVIDER,
                requested_start=day_str,
                requested_end=day_str,
                summary=failure_summary,
                market=market,
            )
            return failure_summary

        day_failed_days = [str(item).strip() for item in day_summary.get("failed_days", []) if str(item).strip()]
        day_failed_codes = {
            str(key).strip(): str(value).strip()
            for key, value in dict(day_summary.get("failed_codes") or {}).items()
            if str(key).strip() and str(value).strip()
        }
        if day_failed_days or day_failed_codes or str(day_summary.get("status", "")).strip().upper() != "LIVE":
            failure_summary = _build_historical_backfill_failure_summary(
                day_str=day_str,
                failed_days=day_failed_days or [day_str],
                failed_codes=day_failed_codes or {day_str: "Historical backfill day did not reach complete coverage."},
                written_rows=written_rows,
                successful_days=successful_days,
                oldest_collectable_date=discovered_start,
                track_progress=track_progress,
                reason=reason,
            )
            record_investor_flow_run_failure(
                reason=reason,
                provider=FLOW_PROVIDER,
                requested_start=day_str,
                requested_end=day_str,
                summary=failure_summary,
                market=market,
            )
            return failure_summary

        if raw_frame.empty:
            failed_days = [day_str]
            failed_codes = {day_str: "No normalized raw investor-flow rows were collected."}
            failure_summary = _build_historical_backfill_failure_summary(
                day_str=day_str,
                failed_days=failed_days,
                failed_codes=failed_codes,
                written_rows=written_rows,
                successful_days=successful_days,
                oldest_collectable_date=discovered_start,
                track_progress=track_progress,
                reason=reason,
            )
            record_investor_flow_run_failure(
                reason=reason,
                provider=FLOW_PROVIDER,
                requested_start=day_str,
                requested_end=day_str,
                summary=failure_summary,
                market=market,
            )
            return failure_summary

        write_investor_flow_backfill_chunk(
            raw_frame=raw_frame,
            chunk_start=day_str,
            chunk_end=day_str,
            provider=FLOW_PROVIDER,
            summary=day_summary,
            oldest_collectable_date=discovered_start,
            target_end_date=target_end.strftime("%Y%m%d"),
            market=market,
            reason=reason,
            update_progress=track_progress,
        )
        successful_days.append(day_str)
        written_rows += int(len(raw_frame))

    return {
        "status": "LIVE",
        "provider": FLOW_PROVIDER,
        "requested_start": start_str,
        "requested_end": target_end.strftime("%Y%m%d"),
        "coverage_complete": True,
        "failed_days": [],
        "failed_codes": {},
        "rows": written_rows,
        "successful_days": successful_days,
        "oldest_collectable_date": discovered_start,
        "mode": "raw_only_history",
        "discovery": discovery_details,
        "track_progress": bool(track_progress),
        "reason": str(reason),
    }


def run_manual_investor_flow_refresh(
    *,
    sector_map: dict[str, Any],
    end_date_str: str,
    start_date_str: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_CALENDAR_DAYS,
    market: str = "KR",
    progress_callback: ProgressCallback | None = None,
) -> tuple[LoaderResult, dict[str, Any]]:
    resolved_end = _normalize_yyyymmdd(end_date_str)
    if not resolved_end:
        raise ValueError(f"Invalid investor-flow end date: {end_date_str}")
    if start_date_str:
        start = _normalize_yyyymmdd(start_date_str)
        if not start:
            raise ValueError(f"Invalid investor-flow start date: {start_date_str}")
        window_meta = {
            "mode": "explicit",
            "complete_cursor": read_investor_flow_operational_complete_cursor(market=market) or "",
            "latest_run_reason": "",
            "latest_run_end": "",
            "failed_days_repaired": [],
        }
    else:
        start, resolved_end, window_meta = resolve_investor_flow_refresh_window(
            end_date_str=resolved_end,
            lookback_days=lookback_days,
            market=market,
        )
    _emit_progress(
        progress_callback,
        task="투자자수급 갱신",
        phase="갱신 구간 확정",
        pct=8,
        detail=f"{start} ~ {resolved_end}",
    )
    end = resolved_end
    if pd.Timestamp(start) > pd.Timestamp(end):
        cached = load_sector_investor_flow(
            sector_map=sector_map,
            start=end,
            end=end,
            market=market,
        )[1]
        summary = {
            "status": "CACHED",
            "provider": FLOW_PROVIDER,
            "requested_start": start,
            "requested_end": end,
            "coverage_complete": True,
            "failed_days": [],
            "failed_codes": {},
            "rows": int(len(cached)),
            "window": window_meta,
        }
        _emit_progress(
            progress_callback,
            task="투자자수급 갱신",
            phase="이미 최신 구간",
            pct=100,
            detail="complete cursor 기준 이미 최신입니다.",
            status="complete",
        )
        return ("CACHED", cached), summary

    try:
        raw_frame, sector_frame, summary = collect_sector_investor_flow(
            sector_map=sector_map,
            start=start,
            end=end,
            market=market,
            progress_callback=progress_callback,
        )
        summary["window"] = window_meta
        if sector_frame.empty:
            failed_codes = dict(summary.get("failed_codes") or {})
            sector_failures, ticker_failures = _split_failed_code_map(failed_codes)
            auth_failures = {
                k: v for k, v in sector_failures.items() if str(v).startswith("AUTH_REQUIRED:")
            }
            access_denied_failures = {
                k: v for k, v in sector_failures.items() if str(v).startswith("ACCESS_DENIED:")
            }
            parts: list[str] = []
            if auth_failures:
                parts.append(
                    f"AUTH_REQUIRED: {len(auth_failures)}/{summary.get('tracked_sectors', '?')} sectors blocked by KRX login gate"
                )
            if access_denied_failures:
                parts.append(
                    f"ACCESS_DENIED: {len(access_denied_failures)}/{summary.get('tracked_sectors', '?')} sectors denied by KRX"
                )
            if sector_failures:
                parts.append(
                    f"CONSTITUENT_ERROR: {len(sector_failures)}/{summary.get('tracked_sectors', '?')} sectors failed"
                )
            if ticker_failures:
                parts.append(
                    f"TRADING_VALUE_ERROR: {len(ticker_failures)}/{summary.get('tracked_tickers', '?')} tickers failed"
                )
            if not parts:
                parts.append("No data returned and no explicit failure recorded — check pykrx/KRX connectivity")
            preview_items = list(failed_codes.items())[:5]
            preview = ", ".join(f"{k}={v}" for k, v in preview_items)
            suffix = ", ..." if len(failed_codes) > 5 else ""
            detail = f" [{'; '.join(parts)}] Detail: {preview}{suffix}"
            raise RuntimeError(
                "No investor-flow rows were collected for the tracked sectors." + detail
            )

        try:
            _emit_progress(
                progress_callback,
                task="투자자수급 갱신",
                phase="웨어하우스 저장 중",
                pct=94,
                detail=f"{len(sector_frame)} rows",
            )
            write_investor_flow_operational_result(
                raw_frame=raw_frame,
                sector_frame=sector_frame,
                provider=FLOW_PROVIDER,
                requested_start=start,
                requested_end=end,
                reason="manual_refresh",
                summary=summary,
                market=market,
            )
        except RuntimeError as exc:
            if not _is_warehouse_write_lock_error(exc):
                raise
            logger.warning(
                "Investor-flow live rows collected but warehouse write-back was skipped because the warehouse is locked: %s",
                exc,
            )
            summary = dict(summary)
            summary["collected_rows"] = int(len(sector_frame))
            failed_codes = dict(summary.get("failed_codes") or {})
            failed_codes["warehouse"] = str(exc)
            summary["failed_codes"] = failed_codes
            summary["warehouse_write_skipped"] = True
            summary["warehouse_write_error"] = str(exc)
            summary["rows"] = int(len(sector_frame))
            _emit_progress(
                progress_callback,
                task="투자자수급 갱신",
                phase="미리보기 완료",
                pct=100,
                detail="웨어하우스 쓰기는 건너뛰고 메모리 미리보기를 유지합니다.",
                status="complete",
            )
            return ("LIVE", sector_frame.copy()), summary
        cached = load_sector_investor_flow(
            sector_map=sector_map,
            start=start,
            end=end,
            market=market,
            allow_bootstrap_partial_preview=True,
        )[1]
        _emit_progress(
            progress_callback,
            task="투자자수급 갱신",
            phase="캐시 재로드 완료",
            pct=100,
            detail=f"{len(cached)} rows",
            status="complete",
        )
        return ("LIVE", cached), summary
    except Exception as exc:
        logger.exception("Investor-flow refresh failed")
        fallback_status, fallback_frame = load_sector_investor_flow(
            sector_map=sector_map,
            start=start,
            end=end,
            market=market,
            allow_bootstrap_partial_preview=True,
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
            "window": window_meta,
        }
        try:
            record_investor_flow_run_failure(
                reason="manual_refresh",
                provider=FLOW_PROVIDER,
                requested_start=start,
                requested_end=end,
                summary=summary,
                market=market,
            )
        except Exception:
            logger.debug("Investor-flow failure bookkeeping skipped", exc_info=True)
        _emit_progress(
            progress_callback,
            task="투자자수급 갱신",
            phase="갱신 실패",
            pct=100,
            detail=str(exc),
            status="error",
        )
        return (fallback_status, fallback_frame), summary


def read_warm_status() -> dict[str, Any]:
    latest_operational = read_latest_investor_flow_run(
        reasons=OPERATIONAL_INVESTOR_FLOW_REASONS,
    )
    latest_summary = dict(latest_operational.get("summary") or {})
    failed_codes = dict(latest_operational.get("failed_codes") or {})
    failed_sector_codes, failed_ticker_codes = _split_failed_code_map(failed_codes)
    status = {
        "status": str(latest_operational.get("status", "")).strip().upper(),
        "provider": str(latest_operational.get("provider", "")).strip().upper(),
        "coverage_complete": bool(latest_operational.get("coverage_complete", False)),
        "failed_days": list(latest_operational.get("failed_days", [])),
        "failed_codes": failed_codes,
        "failed_sector_codes": failed_sector_codes,
        "failed_ticker_codes": failed_ticker_codes,
        "failed_ticker_detail": dict(latest_summary.get("failed_ticker_detail") or {}),
        "failed_ticker_family_counts": {
            str(key).strip(): int(value or 0)
            for key, value in dict(latest_summary.get("failed_ticker_family_counts") or {}).items()
            if str(key).strip()
        },
        "failed_sector_count": len(failed_sector_codes),
        "failed_ticker_count": len(failed_ticker_codes),
        "reason": str(latest_operational.get("reason", "")).strip(),
        "delta_codes": list(latest_operational.get("delta_codes", [])),
        "aborted": bool(latest_operational.get("aborted")),
        "abort_reason": str(latest_operational.get("abort_reason", "")).strip(),
        "predicted_requests": int(latest_operational.get("predicted_requests", 0) or 0),
        "processed_requests": int(latest_operational.get("processed_requests", 0) or 0),
        "retried_ticker_count": int(latest_summary.get("retried_ticker_count", 0) or 0),
        "retried_recovered_ticker_count": int(latest_summary.get("retried_recovered_ticker_count", 0) or 0),
        "retry_processed_requests": int(latest_summary.get("retry_processed_requests", 0) or 0),
        "failed_day_source": str(latest_summary.get("failed_day_source", "")).strip(),
        "failed_day_truth_confident": bool(latest_summary.get("failed_day_truth_confident", False)),
        "request_metric_semantics": str(latest_summary.get("request_metric_semantics", "")).strip(),
    }
    if status["failed_days"] and not status["failed_day_truth_confident"]:
        requested_start = _normalize_yyyymmdd(str(latest_operational.get("requested_start", "")))
        requested_end = _normalize_yyyymmdd(str(latest_operational.get("requested_end", "")))
        if requested_start and requested_end:
            requested_days, source, confident = _resolve_frozen_trading_day_truth(
                requested_start,
                requested_end,
                market="KR",
            )
            if confident:
                requested_day_set = set(requested_days)
                status["failed_days"] = [day for day in status["failed_days"] if day in requested_day_set]
                status["failed_day_source"] = source
                status["failed_day_truth_confident"] = True
    window = dict(latest_summary.get("window") or {})
    cursor = read_investor_flow_operational_complete_cursor()
    if cursor:
        status["watermark_key"] = cursor
        status["end"] = cursor
    else:
        status["end"] = ""
    if not window and not cursor:
        window = _resolve_unresolved_operational_replay(market="KR", complete_cursor="")
        window.pop("start", None)
    if window:
        status["window"] = window
        status["window_mode"] = str(window.get("mode", "")).strip()
        status["anchor_start"] = str(window.get("anchor_start", "")).strip()
        status["anchor_reason"] = str(window.get("anchor_reason", "")).strip()
        status["metadata_durable"] = bool(window.get("metadata_durable"))
    return status


def get_investor_flow_artifact_key() -> tuple[int, int, str, str, str]:
    return get_dataset_artifact_key("investor_flow")


def probe_investor_flow_status() -> str:
    return probe_dataset_mode("investor_flow")


__all__ = [
    "DEFAULT_INVESTOR_TYPES",
    "DEFAULT_LOOKBACK_CALENDAR_DAYS",
    "FLOW_PROVIDER",
    "HISTORICAL_BACKFILL_DISCOVERY_REASON",
    "HISTORICAL_BACKFILL_REASON",
    "HISTORICAL_BACKFILL_VALIDATION_REASON",
    "KRX_INVESTOR_FLOW_WEB_HISTORY_FLOOR",
    "DataStatus",
    "LoaderResult",
    "collect_sector_investor_flow",
    "discover_oldest_collectable_date",
    "get_investor_flow_artifact_key",
    "load_sector_investor_flow",
    "probe_investor_flow_status",
    "read_warm_status",
    "resolve_investor_flow_refresh_window",
    "run_historical_investor_flow_backfill",
    "run_manual_investor_flow_refresh",
]
