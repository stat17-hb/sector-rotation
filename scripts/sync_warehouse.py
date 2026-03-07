"""Incrementally sync the local DuckDB warehouse for market and macro data."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd

from src.data_sources.krx_indices import warm_sector_price_cache
from src.data_sources.macro_sync import sync_macro_warehouse
from src.data_sources.warehouse import (
    get_market_latest_dates,
    macro_row_count,
    market_row_count,
    read_dataset_status,
    read_market_prices,
)
from src.transforms.calendar import get_last_business_day


def _load_configs() -> tuple[dict, dict]:
    with open(ROOT / "config" / "sector_map.yml", encoding="utf-8") as fh:
        sector_map = yaml.safe_load(fh) or {}
    with open(ROOT / "config" / "macro_series.yml", encoding="utf-8") as fh:
        macro_series_cfg = yaml.safe_load(fh) or {}
    return sector_map, macro_series_cfg


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
    parser.add_argument("--prices-years", type=int, default=5)
    parser.add_argument("--macro-years", type=int, default=10)
    parser.add_argument("--as-of", default="", help="End date in YYYYMMDD or YYYY-MM-DD")
    return parser.parse_args()


def _last_complete_month(as_of, *, lag_months: int = 2) -> str:
    return (pd.Period(as_of.strftime("%Y%m"), freq="M") - max(1, int(lag_months))).strftime("%Y%m")


def _resolve_market_sync_start(index_codes: list[str], fallback_start: date) -> date:
    latest_dates = get_market_latest_dates(index_codes)
    if not latest_dates:
        return fallback_start

    parsed_dates = [
        datetime.strptime(str(value), "%Y%m%d").date()
        for value in latest_dates.values()
        if str(value).strip()
    ]
    if not parsed_dates:
        return fallback_start
    return min(parsed_dates) + timedelta(days=1)


def _resolve_existing_market_end(index_codes: list[str]) -> date | None:
    latest_dates = get_market_latest_dates(index_codes)
    parsed_dates = [
        datetime.strptime(str(value), "%Y%m%d").date()
        for value in latest_dates.values()
        if str(value).strip()
    ]
    if not parsed_dates:
        return None
    return max(parsed_dates)


def main() -> int:
    args = _parse_args()
    sector_map, macro_series_cfg = _load_configs()
    benchmark_code = str(sector_map.get("benchmark", {}).get("code", "1001"))
    if args.as_of:
        digits = "".join(ch for ch in str(args.as_of) if ch.isdigit())
        end_date = datetime.strptime(digits, "%Y%m%d").date()
    else:
        end_date = get_last_business_day(provider="OPENAPI", benchmark_code=benchmark_code)

    codes = _all_sector_codes(sector_map)
    existing_market_end = _resolve_existing_market_end(codes)
    if existing_market_end is not None and existing_market_end > end_date:
        end_date = existing_market_end

    market_window_start = end_date - timedelta(days=365 * int(args.prices_years))
    macro_end = _last_complete_month(end_date)
    macro_start = (end_date - timedelta(days=365 * int(args.macro_years))).strftime("%Y%m")

    market_sync_start = _resolve_market_sync_start(codes, market_window_start)
    if market_sync_start <= end_date:
        (market_status, _market_delta_frame), market_summary = warm_sector_price_cache(
            codes,
            market_sync_start.strftime("%Y%m%d"),
            end_date.strftime("%Y%m%d"),
            reason="sync_warehouse",
            force=False,
        )
    else:
        market_status = "CACHED"
        market_summary = {
            "status": "CACHED",
            "coverage_complete": True,
            "start": market_sync_start.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"),
            "delta_codes": [],
            "failed_days": [],
            "failed_codes": {},
            "provider": "OPENAPI",
            "reason": "sync_warehouse",
        }
    market_frame = read_market_prices(
        codes,
        market_window_start.strftime("%Y%m%d"),
        end_date.strftime("%Y%m%d"),
    )
    macro_status, macro_frame, macro_summary = sync_macro_warehouse(
        start_ym=macro_start,
        end_ym=macro_end,
        macro_series_cfg=macro_series_cfg,
        reason="sync_warehouse",
        force=False,
    )

    success = (
        market_status in {"LIVE", "CACHED"}
        and not market_frame.empty
        and macro_status in {"LIVE", "CACHED"}
        and not macro_frame.empty
    )

    output = {
        "success": success,
        "market": {
            "status": market_status,
            "rows": int(len(market_frame)),
            "summary": market_summary,
            "warehouse_rows": market_row_count(),
            "warehouse_status": read_dataset_status("market_prices"),
        },
        "macro": {
            "status": macro_status,
            "rows": int(len(macro_frame)),
            "summary": macro_summary,
            "warehouse_rows": macro_row_count(),
            "warehouse_status": read_dataset_status("macro_data"),
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
