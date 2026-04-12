"""Validate and backfill historical KR investor-flow raw daily facts."""
from __future__ import annotations

import argparse
from datetime import datetime
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
from src.data_sources.krx_investor_flow import (
    DEFAULT_BACKFILL_EARLIEST_CANDIDATE,
    FLOW_PROVIDER,
    HISTORICAL_BACKFILL_REASON,
    run_historical_investor_flow_backfill,
    discover_oldest_collectable_date,
)
from src.data_sources.warehouse import (
    read_investor_flow_backfill_progress_cursor,
    read_investor_flow_raw_date_bounds,
)
from src.transforms.calendar import get_last_business_day

DEFAULT_CHUNK_BUSINESS_DAYS = 20
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 30.0
DEFAULT_SAMPLE_FAILED_DAY_RATIO = 0.05
DEFAULT_ABORT_FAILED_DAY_RATIO = 0.10
DEFAULT_FULL_FAILED_DAY_RATIO = 0.02
HISTORICAL_BACKFILL_VALIDATION_REASON = "historical_backfill_validation"


def _load_configs(market: str) -> tuple[dict, dict, dict, object]:
    return load_market_configs(market)


def _normalize_yyyymmdd(value: str | None) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=["KR"], default="KR")
    parser.add_argument(
        "--mode",
        choices=["validate-samples", "full"],
        default="validate-samples",
        help="Run sample-window validation or the full raw-only backfill.",
    )
    parser.add_argument("--end-date", default="", help="Backfill end date in YYYYMMDD or YYYY-MM-DD")
    parser.add_argument(
        "--oldest-date",
        default="",
        help="Optional explicit oldest collectable date override in YYYYMMDD or YYYY-MM-DD",
    )
    parser.add_argument(
        "--earliest-candidate",
        default=DEFAULT_BACKFILL_EARLIEST_CANDIDATE,
        help="Earliest date candidate used when discovery is required.",
    )
    parser.add_argument(
        "--chunk-business-days",
        type=int,
        default=DEFAULT_CHUNK_BUSINESS_DAYS,
        help="Business-day chunk size for validation/full backfill orchestration.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help="Maximum retry attempts per chunk/date.",
    )
    parser.add_argument(
        "--retry-sleep-sec",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help="Fixed retry backoff for transient failures.",
    )
    parser.add_argument(
        "--assume-non-regression-passed",
        action="store_true",
        help="Mark the required manual non-regression gate as already satisfied for this run.",
    )
    return parser.parse_args()


def _business_windows(start_date_str: str, end_date_str: str, *, chunk_business_days: int) -> list[tuple[str, str]]:
    dates = list(pd.bdate_range(_normalize_yyyymmdd(start_date_str), _normalize_yyyymmdd(end_date_str)))
    if not dates:
        return []
    windows: list[tuple[str, str]] = []
    chunk_size = max(1, int(chunk_business_days))
    for index in range(0, len(dates), chunk_size):
        chunk = dates[index : index + chunk_size]
        windows.append((chunk[0].strftime("%Y%m%d"), chunk[-1].strftime("%Y%m%d")))
    return windows


def _select_sample_windows(windows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not windows:
        return []
    if len(windows) <= 5:
        return windows
    indices = [0, 1, 2, len(windows) // 2, len(windows) - 1]
    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index in indices:
        window = windows[index]
        if window in seen:
            continue
        seen.add(window)
        selected.append(window)
    return selected


def _failed_day_ratio(summary: dict[str, object], *, start_date_str: str, end_date_str: str) -> float:
    requested_days = pd.bdate_range(_normalize_yyyymmdd(start_date_str), _normalize_yyyymmdd(end_date_str))
    denominator = max(1, len(requested_days))
    failed_days = {
        _normalize_yyyymmdd(day)
        for day in summary.get("failed_days", [])
        if _normalize_yyyymmdd(day)
    }
    return len(failed_days) / float(denominator)


def _classify_failure(summary: dict[str, object]) -> str:
    failed_codes = dict(summary.get("failed_codes") or {})
    if not failed_codes and str(summary.get("status", "")).strip().upper() in {"LIVE", "CACHED"}:
        return "ok"

    values = [str(value) for value in failed_codes.values()]
    if any(value.startswith("AUTH_REQUIRED") or value.startswith("ACCESS_DENIED") for value in values):
        return "hard_auth"
    if any("schema" in value.lower() or "contract" in value.lower() or "non-json" in value.lower() for value in values):
        return "hard_contract"
    if any(
        token in value.lower()
        for value in values
        for token in ("timeout", "timed out", "connection", "winsock", "transport", "dns", "socket")
    ):
        return "transient_transport"
    if any("no normalized raw investor-flow rows" in value.lower() for value in values):
        return "data_gap"
    if summary.get("failed_days"):
        return "data_gap"
    return "unknown_failure"


def _all_targeted_sectors_hard_failed(summary: dict[str, object]) -> bool:
    failed_codes = {
        str(key): str(value)
        for key, value in dict(summary.get("failed_codes") or {}).items()
        if str(key).startswith("sector:")
    }
    tracked = int(summary.get("tracked_sectors", 0) or 0)
    if tracked <= 0 or len(failed_codes) < tracked:
        return False
    return all(
        value.startswith("AUTH_REQUIRED") or value.startswith("ACCESS_DENIED") or "schema" in value.lower()
        for value in failed_codes.values()
    )


def _retryable_failure(classification: str) -> bool:
    return classification in {"transient_transport", "data_gap", "unknown_failure"}


def _classify_error_text(error_text: str) -> str:
    message = str(error_text or "")
    lowered = message.lower()
    if message.startswith("AUTH_REQUIRED") or message.startswith("ACCESS_DENIED"):
        return "hard_auth"
    if any(token in lowered for token in ("schema", "contract", "non-json", "jsondecodeerror")):
        return "hard_contract"
    if any(token in lowered for token in ("timeout", "timed out", "connection", "winsock", "transport", "dns", "socket")):
        return "transient_transport"
    return "unknown_failure"


def _discover_oldest_with_retry(
    *,
    sector_map: dict[str, object],
    end_date_str: str,
    earliest_candidate_str: str,
    market: str,
    retry_attempts: int,
    retry_sleep_sec: float,
) -> tuple[str, dict[str, object], list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    last_exc: Exception | None = None
    max_attempts = max(1, int(retry_attempts))
    for attempt in range(1, max_attempts + 1):
        try:
            discovered, details = discover_oldest_collectable_date(
                sector_map=sector_map,
                end_date_str=end_date_str,
                market=market,
                earliest_candidate_str=earliest_candidate_str,
            )
            attempts.append({"attempt": attempt, "status": "LIVE", "oldest_collectable_date": discovered})
            return discovered, details, attempts
        except Exception as exc:  # pragma: no cover - network/error path covered via tests by monkeypatch
            last_exc = exc
            classification = _classify_error_text(str(exc))
            attempts.append({"attempt": attempt, "status": "FAILED", "classification": classification, "error": str(exc)})
            if not _retryable_failure(classification):
                break
            if attempt < max_attempts:
                time.sleep(max(0.0, float(retry_sleep_sec)))
    raise RuntimeError(
        "Failed to validate oldest_collectable_date under the current collector contract."
    ) from last_exc


def _run_chunk_with_retry(
    *,
    sector_map: dict[str, object],
    start_date_str: str,
    end_date_str: str,
    oldest_collectable_date: str,
    market: str,
    retry_attempts: int,
    retry_sleep_sec: float,
    track_progress: bool,
    reason: str,
    pass_failed_day_ratio: float,
) -> dict[str, object]:
    attempts: list[dict[str, object]] = []
    max_attempts = max(1, int(retry_attempts))
    for attempt in range(1, max_attempts + 1):
        summary = run_historical_investor_flow_backfill(
            sector_map=sector_map,
            end_date_str=end_date_str,
            start_date_str=start_date_str,
            oldest_collectable_date=oldest_collectable_date,
            market=market,
            track_progress=track_progress,
            reason=reason,
        )
        ratio = _failed_day_ratio(summary, start_date_str=start_date_str, end_date_str=end_date_str)
        classification = _classify_failure(summary)
        passed = (
            str(summary.get("status", "")).strip().upper() == "LIVE"
            and ratio <= float(pass_failed_day_ratio)
            and classification == "ok"
        )
        attempt_report = {
            "attempt": attempt,
            "status": str(summary.get("status", "")).strip().upper(),
            "classification": classification,
            "failed_day_ratio": ratio,
            "rows": int(summary.get("rows", 0) or 0),
            "failed_days": list(summary.get("failed_days", [])),
            "failed_codes": dict(summary.get("failed_codes") or {}),
            "passed": passed,
        }
        attempts.append(attempt_report)
        if passed:
            return {
                "start": start_date_str,
                "end": end_date_str,
                "passed": True,
                "attempts": attempts,
                "summary": summary,
                "classification": classification,
                "failed_day_ratio": ratio,
            }
        if not _retryable_failure(classification) or attempt >= max_attempts:
            return {
                "start": start_date_str,
                "end": end_date_str,
                "passed": False,
                "attempts": attempts,
                "summary": summary,
                "classification": classification,
                "failed_day_ratio": ratio,
            }
        time.sleep(max(0.0, float(retry_sleep_sec)))
    raise AssertionError("Unreachable retry loop termination.")


def _build_spot_checks(
    *,
    oldest_collectable_date: str,
    end_date_str: str,
    windows: list[tuple[str, str]],
) -> list[dict[str, str]]:
    if not windows:
        return []
    selected = _select_sample_windows(windows)
    labels: list[str] = []
    if selected:
        labels.extend([f"earliest-{idx + 1}" for idx in range(min(3, len(selected)))])
        if len(selected) > 3:
            labels.append("middle")
        if len(selected) > 4:
            labels.append("recent")
    checks: list[dict[str, str]] = []
    for idx, (start, end) in enumerate(selected):
        label = labels[idx] if idx < len(labels) else f"sample-{idx + 1}"
        checks.append(
            {
                "label": label,
                "start": start,
                "end": end,
                "expected": "row-count sanity + expected ticker presence where applicable",
            }
        )
    return checks


def _run_phase1_validation(
    *,
    sector_map: dict[str, object],
    end_date_str: str,
    market: str,
    oldest_date: str,
    earliest_candidate_str: str,
    chunk_business_days: int,
    retry_attempts: int,
    retry_sleep_sec: float,
) -> dict[str, object]:
    discovery_attempts: list[dict[str, object]] = []
    if oldest_date:
        validated_oldest = oldest_date
        discovery_details: dict[str, object] = {"oldest_collectable_date": oldest_date, "mode": "explicit_override"}
    else:
        validated_oldest, discovery_details, discovery_attempts = _discover_oldest_with_retry(
            sector_map=sector_map,
            end_date_str=end_date_str,
            earliest_candidate_str=earliest_candidate_str,
            market=market,
            retry_attempts=retry_attempts,
            retry_sleep_sec=retry_sleep_sec,
        )

    windows = _business_windows(validated_oldest, end_date_str, chunk_business_days=chunk_business_days)
    sample_windows = _select_sample_windows(windows)
    chunk_reports: list[dict[str, object]] = []
    consecutive_failures = 0
    aborted = False
    abort_reason = ""

    for start, end in sample_windows:
        report = _run_chunk_with_retry(
            sector_map=sector_map,
            start_date_str=start,
            end_date_str=end,
            oldest_collectable_date=validated_oldest,
            market=market,
            retry_attempts=retry_attempts,
            retry_sleep_sec=retry_sleep_sec,
            track_progress=False,
            reason=HISTORICAL_BACKFILL_VALIDATION_REASON,
            pass_failed_day_ratio=DEFAULT_SAMPLE_FAILED_DAY_RATIO,
        )
        chunk_reports.append(report)
        if report["passed"]:
            consecutive_failures = 0
            continue
        consecutive_failures += 1
        if float(report["failed_day_ratio"]) > DEFAULT_ABORT_FAILED_DAY_RATIO:
            aborted = True
            abort_reason = f"Abort: {start}..{end} failed day ratio exceeded {DEFAULT_ABORT_FAILED_DAY_RATIO:.0%}."
            break
        if _all_targeted_sectors_hard_failed(dict(report["summary"])):
            aborted = True
            abort_reason = f"Abort: {start}..{end} ended with all-sector auth/contract hard failure."
            break
        if consecutive_failures >= 2:
            aborted = True
            abort_reason = "Abort: two consecutive sample chunks failed after all retries."
            break

    passed = bool(sample_windows) and all(bool(item["passed"]) for item in chunk_reports) and not aborted
    return {
        "status": "LIVE" if passed else "SAMPLE",
        "mode": "validate_samples",
        "provider": FLOW_PROVIDER,
        "oldest_collectable_date": validated_oldest,
        "requested_end": end_date_str,
        "retry_attempts": int(retry_attempts),
        "retry_sleep_sec": float(retry_sleep_sec),
        "chunk_business_days": int(chunk_business_days),
        "sample_window_count": len(sample_windows),
        "discovery": discovery_details,
        "discovery_attempts": discovery_attempts,
        "sample_reports": chunk_reports,
        "spot_checks": _build_spot_checks(
            oldest_collectable_date=validated_oldest,
            end_date_str=end_date_str,
            windows=windows,
        ),
        "passed": passed,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "manual_non_regression_required": True,
        "required_non_regression_command": (
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q "
            "tests\\test_warehouse_investor_flow.py tests\\test_krx_investor_flow_data_source.py "
            "tests\\test_dashboard_runtime.py tests\\test_dashboard_tabs.py tests\\test_data_status.py"
        ),
    }


def _run_phase2_full_backfill(
    *,
    sector_map: dict[str, object],
    end_date_str: str,
    market: str,
    oldest_collectable_date: str,
    chunk_business_days: int,
    retry_attempts: int,
    retry_sleep_sec: float,
) -> dict[str, object]:
    windows = _business_windows(oldest_collectable_date, end_date_str, chunk_business_days=chunk_business_days)
    chunk_reports: list[dict[str, object]] = []
    total_failed_days = 0
    total_requested_days = 0
    aborted = False
    abort_reason = ""
    last_success_end = ""

    for start, end in windows:
        report = _run_chunk_with_retry(
            sector_map=sector_map,
            start_date_str=start,
            end_date_str=end,
            oldest_collectable_date=oldest_collectable_date,
            market=market,
            retry_attempts=retry_attempts,
            retry_sleep_sec=retry_sleep_sec,
            track_progress=True,
            reason=HISTORICAL_BACKFILL_REASON,
            pass_failed_day_ratio=1.0,
        )
        chunk_reports.append(report)
        requested_days = len(pd.bdate_range(start, end))
        total_requested_days += requested_days
        total_failed_days += len(
            {
                _normalize_yyyymmdd(day)
                for day in report["summary"].get("failed_days", [])
                if _normalize_yyyymmdd(day)
            }
        )

        classification = str(report["classification"])
        if report["passed"] or classification == "ok":
            last_success_end = end
            continue
        if classification in {"hard_auth", "hard_contract"}:
            aborted = True
            abort_reason = f"Abort: {start}..{end} ended with unresolved {classification}."
            break
        if float(report["failed_day_ratio"]) > DEFAULT_ABORT_FAILED_DAY_RATIO:
            aborted = True
            abort_reason = f"Abort: {start}..{end} failed day ratio exceeded {DEFAULT_ABORT_FAILED_DAY_RATIO:.0%}."
            break
        aborted = True
        abort_reason = f"Abort: {start}..{end} did not pass after retries."
        break

    cumulative_failed_ratio = (total_failed_days / total_requested_days) if total_requested_days else 0.0
    raw_date_bounds = read_investor_flow_raw_date_bounds(market=market)
    progress_cursor = read_investor_flow_backfill_progress_cursor(market=market)
    hard_failures = [
        report for report in chunk_reports if str(report.get("classification")) in {"hard_auth", "hard_contract"}
    ]
    closure_checks = {
        "progress_reached_target_end": str(progress_cursor or "") == str(end_date_str),
        "failed_day_ratio_within_threshold": cumulative_failed_ratio <= DEFAULT_FULL_FAILED_DAY_RATIO,
        "no_unresolved_hard_failures": not hard_failures,
        "min_trade_date_matches_oldest": str(raw_date_bounds.get("min_trade_date", "")) == str(oldest_collectable_date),
    }
    passed = (
        not aborted
        and last_success_end == end_date_str
        and all(bool(value) for value in closure_checks.values())
    )
    return {
        "status": "LIVE" if passed else "SAMPLE",
        "mode": "full_backfill",
        "provider": FLOW_PROVIDER,
        "oldest_collectable_date": oldest_collectable_date,
        "requested_end": end_date_str,
        "retry_attempts": int(retry_attempts),
        "retry_sleep_sec": float(retry_sleep_sec),
        "chunk_business_days": int(chunk_business_days),
        "chunk_reports": chunk_reports,
        "cumulative_failed_day_ratio": cumulative_failed_ratio,
        "raw_date_bounds": raw_date_bounds,
        "backfill_progress_cursor": progress_cursor or "",
        "closure_checks": closure_checks,
        "passed": passed,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "required_closure_checks": [
            "warehouse min(trade_date) matches validated oldest_collectable_date",
            "no unresolved hard failures remain",
            "current product non-regression checks pass",
        ],
        "manual_non_regression_required": True,
    }


def main() -> int:
    args = _parse_args()
    market_id = str(getattr(args, "market", "KR") or "KR").strip().upper()
    if market_id != "KR":
        print(json.dumps({"success": False, "error": "Investor-flow historical backfill is KR-only."}, ensure_ascii=False))
        return 1

    try:
        loaded = _load_configs(market_id)
    except TypeError:
        loaded = _load_configs()
    if len(loaded) == 4:
        _settings, sector_map, _macro_series_cfg, market_profile = loaded
    else:
        sector_map, _macro_series_cfg = loaded
        market_profile = SimpleNamespace(benchmark_code=str(sector_map.get("benchmark", {}).get("code", "1001")))

    if args.end_date:
        digits = "".join(ch for ch in str(args.end_date) if ch.isdigit())
        end_date_str = datetime.strptime(digits, "%Y%m%d").strftime("%Y%m%d")
    else:
        end_date_str = get_last_business_day(
            provider="OPENAPI",
            benchmark_code=str(getattr(market_profile, "benchmark_code", "1001")),
        ).strftime("%Y%m%d")

    oldest_date = _normalize_yyyymmdd(args.oldest_date)
    if args.mode == "validate-samples":
        report = _run_phase1_validation(
            sector_map=sector_map,
            end_date_str=end_date_str,
            market=market_id,
            oldest_date=oldest_date,
            earliest_candidate_str=str(args.earliest_candidate),
            chunk_business_days=int(args.chunk_business_days),
            retry_attempts=int(args.retry_attempts),
            retry_sleep_sec=float(args.retry_sleep_sec),
        )
    else:
        if not oldest_date:
            raise SystemExit("--mode full requires --oldest-date or a previously validated oldest collectable date.")
        report = _run_phase2_full_backfill(
            sector_map=sector_map,
            end_date_str=end_date_str,
            market=market_id,
            oldest_collectable_date=oldest_date,
            chunk_business_days=int(args.chunk_business_days),
            retry_attempts=int(args.retry_attempts),
            retry_sleep_sec=float(args.retry_sleep_sec),
        )

    manual_gate_ok = bool(args.assume_non_regression_passed)
    if report.get("manual_non_regression_required"):
        report["non_regression_gate_satisfied"] = manual_gate_ok
        report["phase_passed_without_non_regression"] = bool(report.get("passed"))
        report["passed"] = bool(report.get("passed")) and manual_gate_ok
    else:
        report["non_regression_gate_satisfied"] = True

    success = bool(report.get("passed"))
    print(json.dumps({"success": success, "report": report}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
