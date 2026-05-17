"""Warehouse sync helpers for the static theme taxonomy layer."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_sources.theme_taxonomy import (
    THEME_TAXONOMY_CONFIG_PATH,
    ThemeTaxonomy,
    load_theme_taxonomy_config,
)


THEME_TAXONOMY_DATASET = "theme_taxonomy"
THEME_TAXONOMY_PROVIDER = "THEME_TAXONOMY"


def _warehouse_module():
    return importlib.import_module("src.data_sources.warehouse")


def _normalize_market(market: str | None) -> str:
    return str(market or "KR").strip().upper() or "KR"


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        token = str(value or "").strip()
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def build_theme_taxonomy_collection_frame(
    taxonomy: ThemeTaxonomy,
    *,
    market: str = "KR",
) -> pd.DataFrame:
    """Flatten taxonomy execution mappings into the collected runtime rows."""
    normalized_market = _normalize_market(market)
    rows: list[dict[str, Any]] = []
    for mapping in taxonomy.execution_mappings:
        for runtime_ref in mapping.runtime_index_refs:
            if _normalize_market(runtime_ref.market) != normalized_market:
                continue
            rows.append(
                {
                    "mapping_id": mapping.mapping_id,
                    "runtime_role": mapping.runtime_role,
                    "label": mapping.label,
                    "market": _normalize_market(runtime_ref.market),
                    "index_code": runtime_ref.index_code,
                    "index_name": runtime_ref.index_name,
                    "source": runtime_ref.source,
                    "base_tag_refs": list(mapping.base_tag_refs),
                    "cross_tag_refs": list(mapping.cross_tag_refs),
                    "theme_refs": list(mapping.theme_refs),
                    "flow_sector_code": mapping.flow_join.sector_code if mapping.flow_join else "",
                    "export_series_alias": mapping.export_series_alias,
                    "representative_etf_codes": [item.code for item in mapping.representative_etfs],
                }
            )
    return pd.DataFrame(rows)


def get_theme_taxonomy_index_codes(
    *,
    market: str = "KR",
    config_path: Path = THEME_TAXONOMY_CONFIG_PATH,
) -> list[str]:
    """Return runtime index codes declared by theme_taxonomy for one market."""
    taxonomy = load_theme_taxonomy_config(config_path)
    frame = build_theme_taxonomy_collection_frame(taxonomy, market=market)
    if frame.empty or "index_code" not in frame.columns:
        return []
    return _ordered_unique(frame["index_code"].astype(str).tolist())


def _taxonomy_summary(taxonomy: ThemeTaxonomy, frame: pd.DataFrame, *, market: str) -> dict[str, Any]:
    index_codes = _ordered_unique(frame["index_code"].astype(str).tolist()) if not frame.empty else []
    mapping_ids = _ordered_unique(frame["mapping_id"].astype(str).tolist()) if not frame.empty else []
    expected_mapping_ids = (
        [mapping.mapping_id for mapping in taxonomy.execution_mappings]
        if _normalize_market(market) == _normalize_market(taxonomy.market)
        else []
    )
    missing_mapping_ids = [
        mapping_id for mapping_id in expected_mapping_ids
        if mapping_id not in set(mapping_ids)
    ]
    last_verified = taxonomy.verification.last_verified_at.isoformat()
    return {
        "market": _normalize_market(market),
        "taxonomy_version": int(taxonomy.taxonomy_version),
        "last_verified_at": last_verified,
        "verification_status": taxonomy.verification.verification_status,
        "stale_after_days": int(taxonomy.verification.stale_after_days),
        "theme_count": len(taxonomy.themes),
        "theme_mapping_count": len(taxonomy.theme_mappings),
        "base_axis_count": len(taxonomy.classification_axes.base_industries),
        "cross_axis_count": len(taxonomy.classification_axes.cross_themes),
        "execution_mapping_count": len(taxonomy.execution_mappings),
        "runtime_index_count": int(len(frame)),
        "row_count": int(len(frame)),
        "index_codes": index_codes,
        "mapping_ids": mapping_ids,
        "expected_mapping_ids": expected_mapping_ids,
        "expected_mapping_count": len(expected_mapping_ids),
        "missing_mapping_ids": missing_mapping_ids,
        "missing_mapping_count": len(missing_mapping_ids),
        "coverage_complete": bool(expected_mapping_ids) and not missing_mapping_ids and bool(index_codes),
    }


def sync_theme_taxonomy_warehouse(
    *,
    reason: str = "manual_refresh",
    market: str = "KR",
    config_path: Path = THEME_TAXONOMY_CONFIG_PATH,
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    """Record theme_taxonomy as a first-class warehouse collection dataset."""
    normalized_market = _normalize_market(market)
    taxonomy = load_theme_taxonomy_config(config_path)
    frame = build_theme_taxonomy_collection_frame(taxonomy, market=normalized_market)
    summary = _taxonomy_summary(taxonomy, frame, market=normalized_market)

    if normalized_market != _normalize_market(taxonomy.market):
        summary.update({"status": "SKIPPED", "coverage_complete": True, "skip_reason": "market_not_supported"})
        return "SKIPPED", frame, summary

    status = "LIVE" if not frame.empty else "SAMPLE"
    coverage_complete = bool(summary["coverage_complete"])
    failed_codes = {
        f"mapping:{mapping_id}": f"no runtime index reference for {normalized_market}"
        for mapping_id in summary["missing_mapping_ids"]
    }
    if not failed_codes and not coverage_complete:
        failed_codes["theme_taxonomy"] = "no runtime index mappings"
    verified_key = taxonomy.verification.last_verified_at.strftime("%Y%m%d")
    watermark_key = f"{verified_key}-v{taxonomy.taxonomy_version}"

    warehouse = _warehouse_module()
    warehouse.record_ingest_run(
        dataset=THEME_TAXONOMY_DATASET,
        reason=reason,
        provider=THEME_TAXONOMY_PROVIDER,
        requested_start=verified_key,
        requested_end=verified_key,
        status=status,
        coverage_complete=coverage_complete,
        failed_days=[],
        failed_codes=failed_codes,
        delta_keys=list(summary["mapping_ids"]),
        row_count=int(summary["row_count"]),
        predicted_requests=int(summary["expected_mapping_count"]),
        processed_requests=int(len(summary["mapping_ids"])),
        summary=summary,
        market=normalized_market,
    )
    warehouse.update_ingest_watermark(
        dataset=THEME_TAXONOMY_DATASET,
        watermark_key=watermark_key,
        status=status,
        coverage_complete=coverage_complete,
        provider=THEME_TAXONOMY_PROVIDER,
        details=summary,
        market=normalized_market,
    )
    summary["status"] = status
    return status, frame, summary
