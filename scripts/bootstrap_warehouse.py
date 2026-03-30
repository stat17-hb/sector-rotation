"""Bootstrap the local DuckDB warehouse with full initial market/macro history."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from config.markets import load_market_configs
from src.data_sources.krx_indices import warm_sector_price_cache
from src.data_sources.macro_sync import sync_macro_warehouse
from src.data_sources.warehouse import (
    is_market_coverage_complete,
    macro_row_count,
    market_row_count,
    read_dataset_status,
    read_market_prices,
)
from src.transforms.calendar import get_last_business_day


def _load_configs(market: str) -> tuple[dict, dict, dict, object]:
    return load_market_configs(market)


def _all_sector_codes(sector_map: dict) -> list[str]:
    codes: list[str] = []
    for regime_data in (sector_map.get("regimes", {}) or {}).values():
        for sector in regime_data.get("sectors", []) or []:
            code = str(sector.get("code", "")).strip()
            if code and code not in codes:
                codes.append(code)
    benchmark_code = str(sector_map.get("benchmark", {}).get("code", "")).strip()
    if benchmark_code and benchmark_code not in codes:
        codes.append(benchmark_code)
    return codes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=["KR", "US"], default="KR")
    parser.add_argument("--prices-years", type=int, default=5)
    parser.add_argument("--macro-years", type=int, default=10)
    parser.add_argument("--market-chunk-months", type=int, default=1)
    parser.add_argument("--market-chunk-retries", type=int, default=3)
    parser.add_argument("--market-chunk-retry-sleep-sec", type=float, default=5.0)
    parser.add_argument("--as-of", default="", help="End date in YYYYMMDD or YYYY-MM-DD")
    return parser.parse_args()


def _last_complete_month(as_of: date, *, lag_months: int = 2) -> str:
    return (pd.Period(as_of.strftime("%Y%m"), freq="M") - max(1, int(lag_months))).strftime("%Y%m")


def _chunk_market_windows(start_date: date, end_date: date, *, chunk_months: int) -> list[tuple[str, str]]:
    windows: list[tuple[str, str]] = []
    cursor = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    chunk_size = max(1, int(chunk_months))
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.DateOffset(months=chunk_size) - pd.Timedelta(days=1), end_ts)
        windows.append((cursor.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        cursor = chunk_end + pd.Timedelta(days=1)
    return windows


def _bootstrap_market(
    codes: list[str],
    *,
    start_date: date,
    end_date: date,
    benchmark_code: str,
    chunk_months: int,
    chunk_retries: int,
    chunk_retry_sleep_sec: float,
) -> tuple[str, pd.DataFrame, dict]:
    windows = _chunk_market_windows(start_date, end_date, chunk_months=chunk_months)
    chunk_results: list[dict[str, object]] = []
    last_status = "SAMPLE"
    last_frame = pd.DataFrame()

    for window_start, window_end in windows:
        chunk_status = "SAMPLE"
        chunk_frame = pd.DataFrame()
        chunk_summary: dict[str, object] = {}
        max_attempts = max(1, int(chunk_retries))
        attempt = 0
        for attempt in range(1, max_attempts + 1):
            (chunk_status, chunk_frame), chunk_summary = warm_sector_price_cache(
                codes,
                window_start,
                window_end,
                reason="bootstrap_warehouse",
                force=True,
            )
            if chunk_status != "SAMPLE" and bool(chunk_summary.get("coverage_complete")):
                break
            if attempt < max_attempts:
                time.sleep(max(0.0, float(chunk_retry_sleep_sec)))

        chunk_results.append(
            {
                "start": window_start,
                "end": window_end,
                "attempts": attempt,
                "status": chunk_status,
                "coverage_complete": bool(chunk_summary.get("coverage_complete")),
                "failed_days": list(chunk_summary.get("failed_days", [])),
                "failed_codes": dict(chunk_summary.get("failed_codes") or {}),
                "abort_reason": str(chunk_summary.get("abort_reason", "")),
            }
        )
        last_status = chunk_status
        last_frame = chunk_frame

        if chunk_status == "SAMPLE" or not bool(chunk_summary.get("coverage_complete")):
            aggregate = {
                "status": chunk_status,
                "coverage_complete": False,
                "start": start_date.strftime("%Y%m%d"),
                "end": end_date.strftime("%Y%m%d"),
                "chunks": chunk_results,
                "failed_chunk": chunk_results[-1],
            }
            return chunk_status, chunk_frame, aggregate

    final_frame = read_market_prices(
        codes,
        start_date.strftime("%Y%m%d"),
        end_date.strftime("%Y%m%d"),
    )
    coverage_complete = is_market_coverage_complete(
        codes,
        start_date.strftime("%Y%m%d"),
        end_date.strftime("%Y%m%d"),
        benchmark_code=benchmark_code,
    )
    aggregate = {
        "status": last_status,
        "coverage_complete": coverage_complete,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "chunks": chunk_results,
        "rows": int(len(final_frame)),
    }
    return last_status, final_frame, aggregate


def main() -> int:
    args = _parse_args()
    market_id = str(getattr(args, "market", "KR") or "KR").strip().upper()
    try:
        loaded = _load_configs(market_id)
    except TypeError:
        loaded = _load_configs()
    if len(loaded) == 4:
        settings, sector_map, macro_series_cfg, market_profile = loaded
    else:
        sector_map, macro_series_cfg = loaded
        settings = {"benchmark_code": str(sector_map.get("benchmark", {}).get("code", "1001"))}
        market_profile = SimpleNamespace(benchmark_code=settings["benchmark_code"])
    benchmark_code = str(settings.get("benchmark_code", getattr(market_profile, "benchmark_code", "1001")))
    if args.as_of:
        digits = "".join(ch for ch in str(args.as_of) if ch.isdigit())
        end_date = datetime.strptime(digits, "%Y%m%d").date()
    else:
        calendar_provider = "YFINANCE" if market_id == "US" else "OPENAPI"
        end_date = get_last_business_day(provider=calendar_provider, benchmark_code=benchmark_code)

    market_start = end_date - timedelta(days=365 * int(args.prices_years))
    macro_end = _last_complete_month(end_date)
    macro_start = (end_date - timedelta(days=365 * int(args.macro_years))).strftime("%Y%m")

    codes = _all_sector_codes(sector_map)
    if market_id == "US":
        from src.data_sources.yfinance_sectors import load_sector_prices

        market_status, market_frame = load_sector_prices(
            codes,
            market_start.strftime("%Y%m%d"),
            end_date.strftime("%Y%m%d"),
        )
        market_summary = {
            "status": market_status,
            "coverage_complete": bool(
                not market_frame.empty
                and is_market_coverage_complete(
                    codes,
                    market_start.strftime("%Y%m%d"),
                    end_date.strftime("%Y%m%d"),
                    benchmark_code=benchmark_code,
                    market=market_id,
                )
            ),
            "start": market_start.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"),
            "rows": int(len(market_frame)),
            "provider": "YFINANCE",
        }
    else:
        market_status, market_frame, market_summary = _bootstrap_market(
            codes,
            start_date=market_start,
            end_date=end_date,
            benchmark_code=benchmark_code,
            chunk_months=int(getattr(args, "market_chunk_months", 6)),
            chunk_retries=int(getattr(args, "market_chunk_retries", 3)),
            chunk_retry_sleep_sec=float(getattr(args, "market_chunk_retry_sleep_sec", 5.0)),
        )
    macro_status, macro_frame, macro_summary = sync_macro_warehouse(
        start_ym=macro_start,
        end_ym=macro_end,
        macro_series_cfg=macro_series_cfg,
        reason="bootstrap_warehouse",
        force=True,
        market=market_id,
    )

    success = (
        market_status in {"LIVE", "CACHED"}
        and not market_frame.empty
        and bool(market_summary.get("coverage_complete"))
        and macro_status in {"LIVE", "CACHED"}
        and not macro_frame.empty
        and bool(macro_summary.get("coverage_complete"))
    )

    output = {
        "success": success,
        "market": {
            "status": market_status,
            "rows": int(len(market_frame)),
            "summary": market_summary,
            "warehouse_rows": market_row_count(market=market_id),
            "warehouse_status": read_dataset_status("market_prices", market=market_id),
        },
        "macro": {
            "status": macro_status,
            "rows": int(len(macro_frame)),
            "summary": macro_summary,
            "warehouse_rows": macro_row_count(market=market_id),
            "warehouse_status": read_dataset_status("macro_data", market=market_id),
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
