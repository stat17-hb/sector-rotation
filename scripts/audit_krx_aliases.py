"""Audit KRX index alias matching against one known-good snapshot date."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.data_sources.krx_openapi import audit_index_name_aliases


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
    parser.add_argument(
        "--date",
        required=True,
        help="Known-good snapshot date in YYYYMMDD or YYYY-MM-DD format.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results = audit_index_name_aliases(_load_sector_codes(), args.date)
    matched = [row for row in results if row["matched"]]
    unmatched = [row for row in results if not row["matched"]]

    print(
        json.dumps(
            {
                "date": "".join(ch for ch in args.date if ch.isdigit()),
                "matched": len(matched),
                "unmatched": len(unmatched),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not unmatched else 1


if __name__ == "__main__":
    raise SystemExit(main())
