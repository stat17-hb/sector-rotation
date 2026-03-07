"""
Shared DuckDB-backed macro warehouse sync helpers.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Literal

import pandas as pd

from src.contracts.validators import normalize_then_validate
from src.data_sources.warehouse import (
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

MacroProvider = Literal["ECOS", "KOSIS"]
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
) -> pd.DataFrame:
    frame = read_macro_data(
        series_aliases=series_aliases,
        start_ym=start_ym,
        end_ym=end_ym,
    )
    if frame.empty:
        return pd.DataFrame()
    filtered = frame[frame["source"].astype(str).str.upper() == provider].copy()
    if filtered.empty:
        return pd.DataFrame()
    return filtered.drop(columns=["series_alias"], errors="ignore")


def get_macro_artifact_key() -> tuple[int, int, str, str, str]:
    return get_dataset_artifact_key("macro_data")


def probe_macro_status() -> str:
    return probe_dataset_mode("macro_data")


def sync_provider_macro(
    *,
    provider: MacroProvider,
    start_ym: str,
    end_ym: str,
    series_config: dict | None,
    fetch_fn: FetchFn,
    reason: str,
    force: bool = False,
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
    upsert_macro_dimension(_series_dimension_rows(provider, active_config))

    cached = _warehouse_provider_frame(
        provider,
        series_aliases=aliases,
        start_ym=normalized_start,
        end_ym=normalized_end,
    )
    if (
        not force
        and not cached.empty
        and is_macro_coverage_complete(
            series_aliases=aliases,
            start_ym=normalized_start,
            end_ym=normalized_end,
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

    latest_periods = {} if force else get_macro_latest_periods(aliases)
    delta_aliases: list[str] = []
    failed_aliases: dict[str, str] = {}

    for alias, cfg in active_config.items():
        fetch_start = normalized_start
        last_period = latest_periods.get(alias, "")
        if last_period:
            fetch_start = max(normalized_start, _shift_months(last_period, -6))

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
    )
    coverage_complete = is_macro_coverage_complete(
        series_aliases=aliases,
        start_ym=normalized_start,
        end_ym=normalized_end,
    )

    if not final_frame.empty:
        validated = normalize_then_validate(final_frame, "macro_monthly")
        export_macro_parquet()
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
        )
        if coverage_complete:
            update_ingest_watermark(
                dataset="macro_data",
                watermark_key=normalized_end,
                status=status,
                coverage_complete=True,
                provider=provider,
                details={
                    "reason": reason,
                    "delta_aliases": delta_aliases,
                    "failed_aliases": failed_aliases,
                },
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
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    from src.data_sources.ecos import fetch_series
    from src.data_sources.kosis import fetch_kosis_series

    ecos_cfg = _active_series_config((macro_series_cfg or {}).get("ecos"))
    kosis_cfg = _active_series_config((macro_series_cfg or {}).get("kosis"))

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

    ecos_status, ecos_frame, ecos_summary = sync_provider_macro(
        provider="ECOS",
        start_ym=start_ym,
        end_ym=end_ym,
        series_config=ecos_cfg,
        fetch_fn=_fetch_ecos,
        reason=reason,
        force=force,
    )
    kosis_status, kosis_frame, kosis_summary = sync_provider_macro(
        provider="KOSIS",
        start_ym=start_ym,
        end_ym=end_ym,
        series_config=kosis_cfg,
        fetch_fn=_fetch_kosis,
        reason=reason,
        force=force,
    )

    def _worst(left: str, right: str) -> str:
        order = {"LIVE": 0, "CACHED": 1, "SAMPLE": 2}
        return left if order.get(left, 2) >= order.get(right, 2) else right

    status = _worst(ecos_status, kosis_status)
    frames = [frame for frame in (ecos_frame, kosis_frame) if not frame.empty]
    combined = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    summary = {
        "status": status,
        "coverage_complete": bool(ecos_summary.get("coverage_complete")) and bool(kosis_summary.get("coverage_complete")),
        "providers": {
            "ECOS": ecos_summary,
            "KOSIS": kosis_summary,
        },
        "rows": int(len(combined)) if not combined.empty else 0,
        "start": _normalize_month_token(start_ym),
        "end": _normalize_month_token(end_ym),
        "reason": reason,
    }
    return status, combined, summary
