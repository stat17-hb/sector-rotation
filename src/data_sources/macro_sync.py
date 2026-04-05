"""
Shared DuckDB-backed macro warehouse sync helpers.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Literal

import pandas as pd

from src.contracts.validators import normalize_then_validate
from src.data_sources.warehouse import (
    close_cached_read_only_connection,
    export_macro_parquet,
    get_dataset_artifact_key,
    get_macro_latest_periods,
    is_macro_coverage_complete,
    probe_dataset_mode,
    read_macro_data,
    record_ingest_run,
    update_ingest_watermark,
    upsert_macro_dimension,
    upsert_macro_series_frame,
)

logger = logging.getLogger(__name__)

MacroProvider = Literal["ECOS", "KOSIS", "FRED"]
FetchFn = Callable[[str, dict[str, Any], str, str], pd.DataFrame]


def _active_series_config(series_config: dict | None) -> dict[str, dict[str, Any]]:
    return {
        str(alias): dict(cfg)
        for alias, cfg in (series_config or {}).items()
        if bool((cfg or {}).get("enabled", True))
    }


def _provider_series_id(provider: MacroProvider, cfg: dict[str, Any]) -> str:
    if provider == "ECOS":
        item_codes = cfg.get("item_codes") or []
        if not item_codes and cfg.get("item_code"):
            item_codes = [cfg["item_code"]]
        normalized = [str(value).strip() for value in item_codes if str(value).strip()]
        series_key = "/".join(normalized)
        return f"{cfg['stat_code']}/{series_key}"

    if provider == "FRED":
        return str(cfg["series_id"]).strip()

    obj_params = cfg.get("obj_params") or {}
    obj_l1 = obj_params.get("objL1") if isinstance(obj_params, dict) else None
    base = f"{cfg['org_id']}/{cfg['tbl_id']}/{cfg['item_id']}"
    return base + (f"/{obj_l1}" if obj_l1 else "")


def _series_dimension_rows(provider: MacroProvider, series_config: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alias, cfg in series_config.items():
        rows.append(
            {
                "series_alias": alias,
                "provider": provider,
                "provider_series_id": _provider_series_id(provider, cfg),
                "enabled": bool(cfg.get("enabled", True)),
                "label": str(cfg.get("label", "")).strip(),
                "unit": str(cfg.get("unit", "")).strip(),
            }
        )
    return rows


def _normalize_month_token(value: str) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:6]


def _shift_months(value: str, months: int) -> str:
    period = pd.Period(_normalize_month_token(value), freq="M")
    return str((period + months).strftime("%Y%m"))


def _warehouse_provider_frame(
    provider: MacroProvider,
    *,
    series_aliases: list[str],
    start_ym: str,
    end_ym: str,
    market: str,
) -> pd.DataFrame:
    frame = read_macro_data(
        series_aliases=series_aliases,
        start_ym=start_ym,
        end_ym=end_ym,
        market=market,
    )
    if frame.empty:
        return pd.DataFrame()
    filtered = frame[frame["source"].astype(str).str.upper() == provider].copy()
    if filtered.empty:
        return pd.DataFrame()
    return filtered.drop(columns=["series_alias"], errors="ignore")


def get_macro_artifact_key(*, market: str = "KR") -> tuple[int, int, str, str, str]:
    return get_dataset_artifact_key("macro_data", market=market)


def probe_macro_status(*, market: str = "KR") -> str:
    return probe_dataset_mode("macro_data", market=market)


def sync_provider_macro(
    *,
    provider: MacroProvider,
    start_ym: str,
    end_ym: str,
    series_config: dict | None,
    fetch_fn: FetchFn,
    reason: str,
    force: bool = False,
    market: str = "KR",
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    active_config = _active_series_config(series_config)
    aliases = sorted(active_config)
    if not aliases:
        empty = pd.DataFrame(
            columns=["series_id", "value", "source", "fetched_at", "is_provisional"]
        )
        empty.index = pd.PeriodIndex([], freq="M")
        summary = {
            "provider": provider,
            "status": "LIVE",
            "coverage_complete": True,
            "delta_aliases": [],
            "failed_aliases": {},
            "start": _normalize_month_token(start_ym),
            "end": _normalize_month_token(end_ym),
            "reason": reason,
        }
        return "LIVE", empty, summary

    normalized_start = _normalize_month_token(start_ym)
    normalized_end = _normalize_month_token(end_ym)

    cached = _warehouse_provider_frame(
        provider,
        series_aliases=aliases,
        start_ym=normalized_start,
        end_ym=normalized_end,
        market=market,
    )
    if (
        not force
        and not cached.empty
        and is_macro_coverage_complete(
            series_aliases=aliases,
            start_ym=normalized_start,
            end_ym=normalized_end,
            market=market,
        )
    ):
        validated = normalize_then_validate(cached, "macro_monthly")
        summary = {
            "provider": provider,
            "status": "CACHED",
            "coverage_complete": True,
            "delta_aliases": [],
            "failed_aliases": {},
            "start": normalized_start,
            "end": normalized_end,
            "reason": reason,
        }
        return "CACHED", validated, summary

    latest_periods = {} if force else get_macro_latest_periods(aliases, market=market)
    close_cached_read_only_connection()
    try:
        upsert_macro_dimension(_series_dimension_rows(provider, active_config), market=market)
    except RuntimeError as exc:
        # Write lock unavailable (e.g. another process holds warehouse.duckdb).
        # Fall back to whatever cached data already exists in the warehouse.
        logger.warning(
            "%s sync skipped — write lock unavailable, returning cached data. (%s)",
            provider,
            exc,
        )
        fallback = _warehouse_provider_frame(
            provider,
            series_aliases=aliases,
            start_ym=normalized_start,
            end_ym=normalized_end,
            market=market,
        )
        if not fallback.empty:
            validated = normalize_then_validate(fallback, "macro_monthly")
            return "CACHED", validated, {
                "provider": provider,
                "status": "CACHED",
                "coverage_complete": False,
                "delta_aliases": [],
                "failed_aliases": {},
                "start": normalized_start,
                "end": normalized_end,
                "reason": reason,
            }
        raise
    delta_aliases: list[str] = []
    failed_aliases: dict[str, str] = {}

    _TRANSFORM_LOOKBACK = {"pct_change_12m": 14}
    _DEFAULT_LOOKBACK = 6

    for alias, cfg in active_config.items():
        fetch_start = normalized_start
        last_period = latest_periods.get(alias, "")
        if last_period:
            transform = cfg.get("transform", "none")
            lookback = _TRANSFORM_LOOKBACK.get(transform, _DEFAULT_LOOKBACK)
            fetch_start = max(normalized_start, _shift_months(last_period, -lookback))

        try:
            frame = fetch_fn(alias, cfg, fetch_start, normalized_end)
            if frame.empty:
                continue
            provider_series_id = _provider_series_id(provider, cfg)
            upsert_macro_series_frame(
                series_alias=alias,
                provider=provider,
                provider_series_id=provider_series_id,
                frame=frame,
                market=market,
            )
            delta_aliases.append(alias)
        except Exception as exc:
            failed_aliases[alias] = str(exc)
            logger.warning("%s series fetch failed (%s): %s", provider, alias, exc)

    final_frame = _warehouse_provider_frame(
        provider,
        series_aliases=aliases,
        start_ym=normalized_start,
        end_ym=normalized_end,
        market=market,
    )
    coverage_complete = is_macro_coverage_complete(
        series_aliases=aliases,
        start_ym=normalized_start,
        end_ym=normalized_end,
        market=market,
    )

    if not final_frame.empty:
        validated = normalize_then_validate(final_frame, "macro_monthly")
        export_macro_parquet(market=market)
        status = "LIVE" if delta_aliases else "CACHED"
        summary = {
            "provider": provider,
            "status": status,
            "coverage_complete": coverage_complete,
            "delta_aliases": delta_aliases,
            "failed_aliases": failed_aliases,
            "start": normalized_start,
            "end": normalized_end,
            "reason": reason,
            "rows": int(len(validated)),
        }
        record_ingest_run(
            dataset="macro_data",
            reason=reason,
            provider=provider,
            requested_start=normalized_start,
            requested_end=normalized_end,
            status=status,
            coverage_complete=coverage_complete,
            failed_days=[],
            failed_codes=failed_aliases,
            delta_keys=delta_aliases,
            row_count=int(len(validated)),
            summary=summary,
            market=market,
        )
        # Always update the watermark after a successful ingest so it never
        # stalls.  When not all series reach normalized_end (e.g. leading_index
        # lags 2 months), use the minimum latest period actually present as the
        # watermark key — coverage IS complete through that minimum period.
        if coverage_complete:
            _watermark_key = normalized_end
            _wm_complete = True
        else:
            actual_latest = get_macro_latest_periods(aliases, market=market)
            if actual_latest:
                _watermark_key = min(actual_latest.values())
                _wm_complete = True  # all series have data through this minimum
            else:
                _watermark_key = normalized_end
                _wm_complete = False
        update_ingest_watermark(
            dataset="macro_data",
            watermark_key=_watermark_key,
            status=status,
            coverage_complete=_wm_complete,
            provider=provider,
            details={
                "reason": reason,
                "delta_aliases": delta_aliases,
                "failed_aliases": failed_aliases,
            },
            market=market,
        )
        return status, validated, summary

    summary = {
        "provider": provider,
        "status": "SAMPLE",
        "coverage_complete": False,
        "delta_aliases": delta_aliases,
        "failed_aliases": failed_aliases,
        "start": normalized_start,
        "end": normalized_end,
        "reason": reason,
        "rows": 0,
    }
    record_ingest_run(
        dataset="macro_data",
        reason=reason,
        provider=provider,
        requested_start=normalized_start,
        requested_end=normalized_end,
        status="FAILED",
        coverage_complete=False,
        failed_days=[],
        failed_codes=failed_aliases,
        delta_keys=delta_aliases,
        row_count=0,
        summary=summary,
        market=market,
    )
    sample = pd.DataFrame(
        columns=["series_id", "value", "source", "fetched_at", "is_provisional"]
    )
    sample.index = pd.PeriodIndex([], freq="M")
    return "SAMPLE", sample, summary


def sync_macro_warehouse(
    *,
    start_ym: str,
    end_ym: str,
    macro_series_cfg: dict[str, Any],
    reason: str,
    force: bool = False,
    market: str = "KR",
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    from src.data_sources.ecos import fetch_series
    from src.data_sources.fred import fetch_fred_series
    from src.data_sources.kosis import fetch_kosis_series

    ecos_cfg = _active_series_config((macro_series_cfg or {}).get("ecos"))
    kosis_cfg = _active_series_config((macro_series_cfg or {}).get("kosis"))
    fred_cfg = _active_series_config((macro_series_cfg or {}).get("fred"))

    def _fetch_ecos(alias: str, cfg: dict[str, Any], provider_start: str, provider_end: str) -> pd.DataFrame:
        _ = alias
        return fetch_series(
            stat_code=cfg["stat_code"],
            item_code=cfg.get("item_code"),
            start_ym=provider_start,
            end_ym=provider_end,
            item_codes=cfg.get("item_codes"),
            cycle=str(cfg.get("cycle", "M")),
        )

    def _fetch_kosis(alias: str, cfg: dict[str, Any], provider_start: str, provider_end: str) -> pd.DataFrame:
        _ = alias
        return fetch_kosis_series(
            cfg["org_id"],
            cfg["tbl_id"],
            cfg["item_id"],
            provider_start,
            provider_end,
            obj_params=cfg.get("obj_params"),
        )

    def _fetch_fred(alias: str, cfg: dict[str, Any], provider_start: str, provider_end: str) -> pd.DataFrame:
        _ = alias
        return fetch_fred_series(
            cfg["series_id"],
            provider_start,
            provider_end,
            transform=str(cfg.get("transform", "none")),
        )

    provider_results: dict[str, tuple[str, pd.DataFrame, dict[str, Any]]] = {}
    if ecos_cfg:
        provider_results["ECOS"] = sync_provider_macro(
            provider="ECOS",
            start_ym=start_ym,
            end_ym=end_ym,
            series_config=ecos_cfg,
            fetch_fn=_fetch_ecos,
            reason=reason,
            force=force,
            market=market,
        )
    if kosis_cfg:
        provider_results["KOSIS"] = sync_provider_macro(
            provider="KOSIS",
            start_ym=start_ym,
            end_ym=end_ym,
            series_config=kosis_cfg,
            fetch_fn=_fetch_kosis,
            reason=reason,
            force=force,
            market=market,
        )
    if fred_cfg:
        provider_results["FRED"] = sync_provider_macro(
            provider="FRED",
            start_ym=start_ym,
            end_ym=end_ym,
            series_config=fred_cfg,
            fetch_fn=_fetch_fred,
            reason=reason,
            force=force,
            market=market,
        )

    if not provider_results:
        empty = pd.DataFrame()
        return (
            "LIVE",
            empty,
            {
                "status": "LIVE",
                "coverage_complete": True,
                "providers": {},
                "rows": 0,
                "start": _normalize_month_token(start_ym),
                "end": _normalize_month_token(end_ym),
                "reason": reason,
            },
        )

    def _worst(left: str, right: str) -> str:
        order = {"LIVE": 0, "CACHED": 1, "SAMPLE": 2}
        return left if order.get(left, 2) >= order.get(right, 2) else right

    statuses = [payload[0] for payload in provider_results.values()]
    status = statuses[0]
    for value in statuses[1:]:
        status = _worst(status, value)

    frames = [payload[1] for payload in provider_results.values() if not payload[1].empty]
    combined = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    provider_summaries = {name: payload[2] for name, payload in provider_results.items()}
    summary = {
        "status": status,
        "coverage_complete": all(bool(item.get("coverage_complete")) for item in provider_summaries.values()),
        "providers": provider_summaries,
        "rows": int(len(combined)) if not combined.empty else 0,
        "start": _normalize_month_token(start_ym),
        "end": _normalize_month_token(end_ym),
        "reason": reason,
    }
    return status, combined, summary
