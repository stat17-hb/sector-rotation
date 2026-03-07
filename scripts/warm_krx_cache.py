"""Out-of-band KRX sector-price warm/backfill CLI."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.data_sources.krx_indices import warm_sector_price_cache
from src.transforms.calendar import get_last_business_day


def _load_sector_codes() -> list[str]:
    with open("config/sector_map.yml", encoding="utf-8") as f:
        sector_map = yaml.safe_load(f) or {}

    codes: list[str] = []
    for regime_data in sector_map.get("regimes", {}).values():
        for sector in regime_data.get("sectors", []):
            code = str(sector.get("code", "")).strip()
            if code and code not in codes:
                codes.append(code)

    benchmark_code = str(sector_map.get("benchmark", {}).get("code", "")).strip()
    if benchmark_code and benchmark_code not in codes:
        codes.append(benchmark_code)
    return codes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=3, help="Lookback years to warm (default: 3)")
    parser.add_argument(
        "--as-of",
        dest="as_of",
        default="",
        help="End date in YYYYMMDD or YYYY-MM-DD. Defaults to last business day.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore raw cache coverage and refetch the full requested range.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.as_of:
        digits = "".join(ch for ch in str(args.as_of) if ch.isdigit())
        end_date = datetime.strptime(digits, "%Y%m%d").date()
    else:
        end_date = get_last_business_day()

    start_date = end_date - timedelta(days=365 * int(args.years))
    codes = _load_sector_codes()
    (status, result), summary = warm_sector_price_cache(
        codes,
        start_date.strftime("%Y%m%d"),
        end_date.strftime("%Y%m%d"),
        reason="cli_warm",
        force=bool(args.force),
    )

    output = {
        "status": status,
        "rows": int(len(result)),
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "summary": summary,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status in {"LIVE", "CACHED"} and not result.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
