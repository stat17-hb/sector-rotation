"""CLI entry point for the unofficial KRX investor-flow prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Allow `python scripts/krx_investor_flow_probe.py ...` to import the repo package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.collectors.krx_investor_flow.probe import KrxInvestorFlowClient


def _write_json(path: str | None, payload: Any) -> None:
    if not path:
        return
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--skip-prewarm", action="store_true")
    parser.add_argument("--dump-raw", help="Optional JSON file for raw probe artifacts.")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    market_parser = subparsers.add_parser("market", help="Probe market-level investor flow.")
    market_parser.add_argument("--market-id", default="ALL", help="KRX market id such as ALL/STK/KSQ.")

    stock_parser = subparsers.add_parser("stock", help="Probe stock-level investor flow.")
    stock_parser.add_argument("--isu-cd", required=True, help="Full isuCd value for the stock.")
    stock_parser.add_argument("--ticker", help="Optional short ticker symbol for finder-style fields.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    client = KrxInvestorFlowClient(timeout=args.timeout)

    if args.mode == "market":
        artifacts = client.fetch_market(
            start_date=args.start_date,
            end_date=args.end_date,
            market_id=args.market_id,
            prewarm=not args.skip_prewarm,
        )
    else:
        artifacts = client.fetch_stock(
            start_date=args.start_date,
            end_date=args.end_date,
            isu_cd=args.isu_cd,
            ticker=args.ticker,
            prewarm=not args.skip_prewarm,
        )

    payload = artifacts.to_dict()
    _write_json(args.dump_raw, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
