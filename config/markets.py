"""Market registry and config-loading helpers for KR/US dashboard support."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MarketProfile:
    """Resolved market metadata used across the app."""

    market_id: str
    settings_base_path: Path
    settings_override_path: Path | None
    sector_map_path: Path
    macro_series_path: Path
    price_provider: str
    macro_provider: str
    benchmark_code: str
    benchmark_label: str
    page_header: str
    source_badges: tuple[str, ...]
    ui_labels: dict[str, str]


def _profile(
    market_id: str,
    *,
    settings_base_path: str,
    settings_override_path: str | None,
    sector_map_path: str,
    macro_series_path: str,
    price_provider: str,
    macro_provider: str,
    benchmark_code: str,
    benchmark_label: str,
    page_header: str,
    source_badges: tuple[str, ...],
    ui_labels: dict[str, str],
) -> MarketProfile:
    return MarketProfile(
        market_id=market_id,
        settings_base_path=CONFIG_DIR / settings_base_path,
        settings_override_path=(CONFIG_DIR / settings_override_path) if settings_override_path else None,
        sector_map_path=CONFIG_DIR / sector_map_path,
        macro_series_path=CONFIG_DIR / macro_series_path,
        price_provider=price_provider,
        macro_provider=macro_provider,
        benchmark_code=benchmark_code,
        benchmark_label=benchmark_label,
        page_header=page_header,
        source_badges=source_badges,
        ui_labels=dict(ui_labels),
    )


MARKET_REGISTRY: dict[str, MarketProfile] = {
    "KR": _profile(
        "KR",
        settings_base_path="settings.yml",
        settings_override_path=None,
        sector_map_path="sector_map.yml",
        macro_series_path="macro_series.yml",
        price_provider="KRX",
        macro_provider="ECOS_KOSIS",
        benchmark_code="1001",
        benchmark_label="KOSPI",
        page_header="Korea Sector Rotation",
        source_badges=("KRX", "ECOS", "KOSIS"),
        ui_labels={
            "market_selector": "Market",
            "sidebar_title": "Korea Sector Rotation",
            "sidebar_caption": "Theme, data actions, and parameters",
            "market_data_refresh": "Refresh market data",
            "macro_data_refresh": "Refresh macro data",
            "recompute": "Recompute signals",
            "asof_date": "As of date",
            "benchmark_strength_title": "Monthly sector strength vs KOSPI",
            "fx_metric_label": "USD/KRW move",
        },
    ),
    "US": _profile(
        "US",
        settings_base_path="settings.yml",
        settings_override_path="settings_us.yml",
        sector_map_path="sector_map_us.yml",
        macro_series_path="macro_series_us.yml",
        price_provider="YFINANCE",
        macro_provider="FRED",
        benchmark_code="SPY",
        benchmark_label="S&P 500",
        page_header="US Sector Rotation",
        source_badges=("Yahoo Finance", "FRED"),
        ui_labels={
            "market_selector": "Market",
            "sidebar_title": "US Sector Rotation",
            "sidebar_caption": "Theme, data actions, and parameters",
            "market_data_refresh": "Refresh market data",
            "macro_data_refresh": "Refresh macro data",
            "recompute": "Recompute signals",
            "asof_date": "As of date",
            "benchmark_strength_title": "Monthly sector strength vs S&P 500",
            "fx_metric_label": "Dollar index move",
        },
    ),
}


def get_market_profile(market_id: str) -> MarketProfile:
    """Return the registry entry for a market id, defaulting to KR."""
    normalized = str(market_id or "KR").strip().upper()
    return MARKET_REGISTRY.get(normalized, MARKET_REGISTRY["KR"])


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge dictionaries without mutating the inputs."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_market_configs(market_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], MarketProfile]:
    """Load settings, sector map, and macro-series config for one market."""
    profile = get_market_profile(market_id)
    settings = _load_yaml(profile.settings_base_path)
    if profile.settings_override_path is not None and profile.settings_override_path.exists():
        settings = deep_merge_dicts(settings, _load_yaml(profile.settings_override_path))
    sector_map = _load_yaml(profile.sector_map_path)
    macro_series = _load_yaml(profile.macro_series_path)
    settings.setdefault("benchmark_code", profile.benchmark_code)
    settings.setdefault("benchmark_label", profile.benchmark_label)
    return settings, sector_map, macro_series, profile
