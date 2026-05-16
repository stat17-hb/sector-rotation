"""Representative ETF mapping helpers.

The hand-authored sector map owns macro regime assignment.  Generated ETF
overlays own execution metadata for dynamically discovered KRX sectors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


GENERATED_KR_ETF_MAP_PATH = Path("data/curated/sector_etf_map.generated.yml")


def _normalize_etf_items(items: object) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        if not isinstance(item, Mapping):
            continue
        code = str(item.get("code", "")).strip()
        name = str(item.get("name", "")).strip()
        if code and name:
            normalized.append({"code": code, "name": name})
    return normalized


def build_config_etf_map(sector_map: Mapping[str, Any] | None) -> dict[str, list[dict[str, str]]]:
    """Build an ETF map from hand-authored sector_map.yml content."""
    if not sector_map:
        return {}
    etf_map: dict[str, list[dict[str, str]]] = {}
    for regime_data in dict(sector_map.get("regimes", {}) or {}).values():
        for sector in dict(regime_data or {}).get("sectors", []) or []:
            if not isinstance(sector, Mapping):
                continue
            code = str(sector.get("code", "")).strip()
            etfs = _normalize_etf_items(sector.get("etfs") or [])
            if code and etfs:
                etf_map[code] = etfs
    return etf_map


def load_generated_etf_map(path: Path = GENERATED_KR_ETF_MAP_PATH) -> dict[str, list[dict[str, str]]]:
    """Load a generated ETF overlay without requiring the file to exist."""
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mappings = payload.get("mappings", payload) if isinstance(payload, Mapping) else {}
    if not isinstance(mappings, Mapping):
        return {}

    etf_map: dict[str, list[dict[str, str]]] = {}
    for code, value in mappings.items():
        sector_code = str(code).strip()
        if not sector_code:
            continue
        if isinstance(value, Mapping):
            etfs = _normalize_etf_items(value.get("etfs") or [])
        else:
            etfs = _normalize_etf_items(value)
        if etfs:
            etf_map[sector_code] = etfs
    return etf_map


def merge_etf_maps(
    generated_map: Mapping[str, list[dict[str, str]]] | None,
    config_map: Mapping[str, list[dict[str, str]]] | None,
) -> dict[str, list[dict[str, str]]]:
    """Merge generated + config ETF maps, with explicit config entries winning."""
    merged: dict[str, list[dict[str, str]]] = {
        str(code): [dict(item) for item in items]
        for code, items in (generated_map or {}).items()
        if items
    }
    for code, items in (config_map or {}).items():
        if items:
            merged[str(code)] = [dict(item) for item in items]
    return merged


def build_effective_etf_map(
    sector_map: Mapping[str, Any] | None,
    *,
    generated_path: Path = GENERATED_KR_ETF_MAP_PATH,
) -> dict[str, list[dict[str, str]]]:
    """Return the ETF map used by dashboard execution-reference views."""
    return merge_etf_maps(
        load_generated_etf_map(generated_path),
        build_config_etf_map(sector_map),
    )
