"""Slow, resumable one-day KR investor-flow collector.

Default mode is a dry run. Pass ``--execute`` to make KRX requests.
Partial or filtered runs are spooled under ``data/runtime`` and do not update
the operational investor-flow cursor.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from config.markets import load_market_configs
from src.data_sources.krx_investor_flow import (
    DEFAULT_INVESTOR_TYPES,
    FLOW_PROVIDER,
    SectorUniverse,
    _aggregate_sector_flow,
    _fetch_ticker_trading_value_frames,
    _format_refresh_failure_reason,
    _get_access_denied_cooldown,
    _normalize_ticker_trading_value_frames,
)
from src.data_sources.pykrx_compat import ensure_pykrx_transport_compat
from src.data_sources.warehouse import (
    read_latest_sector_constituents_snapshot,
    write_investor_flow_operational_result,
)


DEFAULT_SLEEP_SEC = 8.0
STATE_ROOT = Path("data/runtime/investor_flow_slow")


@dataclass(frozen=True)
class SectorEntry:
    code: str
    name: str
    export_sector: bool


def _normalize_yyyymmdd(value: str | None) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) < 8:
        raise ValueError(f"Expected YYYYMMDD date, got {value!r}")
    return digits[:8]


def _load_sector_entries(sector_map: dict[str, Any]) -> list[SectorEntry]:
    entries: list[SectorEntry] = []
    seen: set[str] = set()
    for regime in dict(sector_map.get("regimes") or {}).values():
        for sector in list(dict(regime or {}).get("sectors") or []):
            code = str(dict(sector).get("code", "")).strip()
            if not code or code in seen:
                continue
            seen.add(code)
            entries.append(
                SectorEntry(
                    code=code,
                    name=str(dict(sector).get("name", code)).strip() or code,
                    export_sector=bool(dict(sector).get("export_sector", False)),
                )
            )
    return entries


def _select_sector_entries(
    entries: list[SectorEntry],
    *,
    selector: str,
    sector_codes: str,
) -> list[SectorEntry]:
    requested_codes = {
        item.strip()
        for item in str(sector_codes or "").split(",")
        if item.strip()
    }
    if requested_codes:
        selected = [entry for entry in entries if entry.code in requested_codes]
    elif selector == "export":
        selected = [entry for entry in entries if entry.export_sector]
    else:
        selected = list(entries)
    if not selected:
        raise ValueError("No target sectors selected.")
    return selected


def _state_dir(*, market: str, day: str, sector_codes: list[str]) -> Path:
    scope = "all" if not sector_codes else "-".join(sector_codes)
    return STATE_ROOT / str(market).upper() / day / scope


def _load_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os_getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def os_getpid() -> int:
    import os

    return os.getpid()


def _load_spooled_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _append_spooled_raw(path: Path, incoming: pd.DataFrame) -> pd.DataFrame:
    if incoming.empty:
        return _load_spooled_raw(path)
    existing = _load_spooled_raw(path)
    combined = pd.concat([existing, incoming], ignore_index=True) if not existing.empty else incoming.copy()
    combined = combined.drop_duplicates(["trade_date", "ticker", "investor_type"], keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    return combined


def _save_progress_state(
    path: Path,
    *,
    day: str,
    market: str,
    sector_codes: list[str],
    completed: list[str],
    failed: dict[str, str],
) -> None:
    _write_state(
        path,
        {
            "date": day,
            "market": market,
            "sector_codes": list(sector_codes),
            "completed_tickers": sorted(set(completed)),
            "failed_tickers": dict(sorted(failed.items())),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _build_cached_universe(entries: list[SectorEntry], *, market: str) -> tuple[SectorUniverse, dict[str, Any]]:
    sector_codes = [entry.code for entry in entries]
    sector_names = {entry.code: entry.name for entry in entries}
    snapshot = read_latest_sector_constituents_snapshot(sector_codes, market=market)
    if snapshot.empty:
        raise RuntimeError(
            "No cached sector constituent snapshot is available. "
            "Slow collector intentionally avoids live constituent lookup."
        )
    ticker_to_sector_codes: dict[str, list[str]] = {}
    for _, row in snapshot.iterrows():
        sector_code = str(row.get("sector_code", "")).strip()
        ticker = str(row.get("ticker", "")).strip()
        if sector_code not in sector_names or not ticker:
            continue
        ticker_to_sector_codes.setdefault(ticker, [])
        if sector_code not in ticker_to_sector_codes[ticker]:
            ticker_to_sector_codes[ticker].append(sector_code)
    if not ticker_to_sector_codes:
        raise RuntimeError("Cached constituent snapshot did not contain any target tickers.")
    meta = {
        "snapshot_rows": int(len(snapshot)),
        "snapshot_date": str(snapshot.get("snapshot_date", pd.Series([""])).iloc[0]),
    }
    return (
        SectorUniverse(
            sector_codes=sector_codes,
            sector_names=sector_names,
            ticker_to_sector_codes={ticker: sorted(codes) for ticker, codes in ticker_to_sector_codes.items()},
            failed_sector_codes={},
        ),
        meta,
    )


def _collect_one_ticker(stock_module: Any, *, ticker: str, day: str) -> pd.DataFrame:
    buy_frame, sell_frame, net_frame, _details = _fetch_ticker_trading_value_frames(
        stock_module,
        ticker=ticker,
        start=day,
        end=day,
        allow_wrapper_fallback=False,
    )
    return _normalize_ticker_trading_value_frames(
        ticker=ticker,
        ticker_name=stock_module.get_market_ticker_name(ticker),
        buy_frame=buy_frame,
        sell_frame=sell_frame,
        net_frame=net_frame,
        investor_types=DEFAULT_INVESTOR_TYPES,
    )


def _build_summary(
    *,
    day: str,
    universe: SectorUniverse,
    completed: list[str],
    failed: dict[str, str],
    skipped: list[str],
    rows: int,
    dry_run: bool,
    persisted: bool,
    stopped_reason: str,
    snapshot_meta: dict[str, Any],
) -> dict[str, Any]:
    target_tickers = sorted(universe.ticker_to_sector_codes)
    return {
        "status": "DRY_RUN" if dry_run else ("LIVE" if rows else "SAMPLE"),
        "provider": FLOW_PROVIDER,
        "requested_start": day,
        "requested_end": day,
        "coverage_complete": bool(not dry_run and not failed and len(completed) == len(target_tickers)),
        "rows": int(rows),
        "tracked_sectors": len(universe.sector_codes),
        "tracked_tickers": len(target_tickers),
        "completed_tickers": len(completed),
        "failed_tickers": len(failed),
        "skipped_tickers": len(skipped),
        "failed_codes": dict(failed),
        "failed_days": [] if not failed else [day],
        "predicted_requests": len(target_tickers) * 3,
        "processed_requests": len(completed) * 3,
        "persisted": bool(persisted),
        "stopped_reason": str(stopped_reason),
        "snapshot": dict(snapshot_meta),
        "mode": "slow_day_resume",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Target trading day, YYYYMMDD or YYYY-MM-DD.")
    parser.add_argument("--market", choices=["KR"], default="KR")
    parser.add_argument("--sectors", choices=["all", "export"], default="all")
    parser.add_argument("--sector-codes", default="", help="Comma-separated sector code override.")
    parser.add_argument("--sleep-sec", type=float, default=DEFAULT_SLEEP_SEC)
    parser.add_argument("--max-tickers", type=int, default=0, help="Limit live ticker requests for staged execution.")
    parser.add_argument("--execute", action="store_true", help="Actually call KRX. Without this, only prints the plan.")
    parser.add_argument("--reset-state", action="store_true", help="Ignore previous runtime spool for this scope.")
    parser.add_argument("--stop-on-access-denied", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    day = _normalize_yyyymmdd(args.date)
    market = str(args.market).upper()
    _settings, sector_map, _macro_series_cfg, _market_profile = load_market_configs(market)
    entries = _select_sector_entries(
        _load_sector_entries(sector_map),
        selector=str(args.sectors),
        sector_codes=str(args.sector_codes),
    )
    universe, snapshot_meta = _build_cached_universe(entries, market=market)
    target_tickers = sorted(universe.ticker_to_sector_codes)
    state_dir = _state_dir(market=market, day=day, sector_codes=[entry.code for entry in entries])
    state_path = state_dir / "state.json"
    raw_path = state_dir / "raw.parquet"
    if bool(args.reset_state):
        state = {}
        if state_path.exists():
            state_path.unlink()
        if raw_path.exists():
            raw_path.unlink()
    else:
        state = _load_state(state_path)
    completed = sorted({str(item) for item in state.get("completed_tickers", [])})
    failed = {str(k): str(v) for k, v in dict(state.get("failed_tickers") or {}).items()}
    pending = [ticker for ticker in target_tickers if ticker not in set(completed) and ticker not in failed]
    if int(args.max_tickers or 0) > 0:
        pending = pending[: int(args.max_tickers)]

    if not bool(args.execute):
        summary = _build_summary(
            day=day,
            universe=universe,
            completed=completed,
            failed=failed,
            skipped=[ticker for ticker in target_tickers if ticker not in set(completed) and ticker not in failed],
            rows=len(_load_spooled_raw(raw_path)),
            dry_run=True,
            persisted=False,
            stopped_reason="dry_run_no_network",
            snapshot_meta=snapshot_meta,
        )
        summary["state_dir"] = str(state_dir)
        summary["planned_tickers_this_run"] = len(pending)
        return summary

    cooldown = _get_access_denied_cooldown()
    if cooldown:
        summary = _build_summary(
            day=day,
            universe=universe,
            completed=completed,
            failed=failed,
            skipped=pending,
            rows=len(_load_spooled_raw(raw_path)),
            dry_run=False,
            persisted=False,
            stopped_reason="access_denied_cooldown_active",
            snapshot_meta=snapshot_meta,
        )
        summary["cooldown"] = dict(cooldown)
        summary["state_dir"] = str(state_dir)
        summary["planned_tickers_this_run"] = 0
        return summary

    ensure_pykrx_transport_compat()
    from pykrx import stock  # type: ignore[import]

    stopped_reason = ""
    raw_frame = _load_spooled_raw(raw_path)
    for index, ticker in enumerate(pending):
        if index > 0 and float(args.sleep_sec or 0) > 0:
            time.sleep(float(args.sleep_sec))
        try:
            ticker_frame = _collect_one_ticker(stock, ticker=ticker, day=day)
        except Exception as exc:
            reason = _format_refresh_failure_reason(exc)
            failed[ticker] = reason
            stopped_reason = reason
            _save_progress_state(
                state_path,
                day=day,
                market=market,
                sector_codes=universe.sector_codes,
                completed=completed,
                failed=failed,
            )
            if bool(args.stop_on_access_denied) and "ACCESS_DENIED" in reason:
                break
            continue
        if ticker_frame.empty:
            failed[ticker] = "normalized investor-flow frame empty"
            _save_progress_state(
                state_path,
                day=day,
                market=market,
                sector_codes=universe.sector_codes,
                completed=completed,
                failed=failed,
            )
            continue
        raw_frame = _append_spooled_raw(raw_path, ticker_frame)
        completed = sorted(set(completed).union({ticker}))
        _save_progress_state(
            state_path,
            day=day,
            market=market,
            sector_codes=universe.sector_codes,
            completed=completed,
            failed=failed,
        )

    all_targeted = len(completed) == len(target_tickers) and not failed
    unrestricted_scope = str(args.sectors) == "all" and not str(args.sector_codes or "").strip() and int(args.max_tickers or 0) <= 0
    persisted = False
    sector_frame = pd.DataFrame()
    if all_targeted and unrestricted_scope and not raw_frame.empty:
        sector_frame = _aggregate_sector_flow(raw_frame, universe)
        summary_for_write = _build_summary(
            day=day,
            universe=universe,
            completed=completed,
            failed=failed,
            skipped=[],
            rows=len(sector_frame),
            dry_run=False,
            persisted=False,
            stopped_reason=stopped_reason,
            snapshot_meta=snapshot_meta,
        )
        write_investor_flow_operational_result(
            raw_frame=raw_frame,
            sector_frame=sector_frame,
            provider=FLOW_PROVIDER,
            requested_start=day,
            requested_end=day,
            reason="manual_refresh",
            summary=summary_for_write,
            market=market,
        )
        persisted = True

    summary = _build_summary(
        day=day,
        universe=universe,
        completed=completed,
        failed=failed,
        skipped=[ticker for ticker in target_tickers if ticker not in set(completed) and ticker not in failed],
        rows=len(sector_frame) if persisted else len(raw_frame),
        dry_run=False,
        persisted=persisted,
        stopped_reason=stopped_reason,
        snapshot_meta=snapshot_meta,
    )
    summary["state_dir"] = str(state_dir)
    summary["planned_tickers_this_run"] = len(pending)
    return summary


def main() -> None:
    summary = run(_parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
