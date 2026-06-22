"""Macro freshness classification helpers.

The sync status (LIVE/CACHED/SAMPLE) describes how a fetch ran.  Freshness
describes whether the configured and required signal families reached the
requested month.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal


FreshnessReason = Literal[
    "CURRENT",
    "SOURCE_LAG",
    "NOT_CONFIGURED",
    "PARTIAL_PROVIDER_COVERAGE",
    "FETCH_ERROR",
    "WRITE_LOCK_FALLBACK",
]

REASON_PRECEDENCE: tuple[FreshnessReason, ...] = (
    "WRITE_LOCK_FALLBACK",
    "FETCH_ERROR",
    "NOT_CONFIGURED",
    "SOURCE_LAG",
    "PARTIAL_PROVIDER_COVERAGE",
    "CURRENT",
)


DEFAULT_REQUIRED_SIGNAL_GROUPS: dict[str, dict[str, Any]] = {
    "aggregate_exports": {
        "required": True,
        "aliases": ["export_amount", "trade_exports_yoy"],
    },
    "aggregate_imports": {
        "required": True,
        "aliases": ["import_amount", "trade_imports_yoy"],
    },
    "sector_export_proxies": {
        "required": False,
        "aliases": [
            "export_it",
            "export_semiconductor",
            "export_chemicals",
            "export_steel",
            "export_auto",
            "export_machinery",
            "export_pharma",
        ],
    },
    "cpi": {
        "required": True,
        "aliases": ["cpi_yoy", "cpi_mom", "cpi_index_legacy"],
    },
    "fx": {
        "required": True,
        "aliases": ["usdkrw"],
    },
    "leading_index": {
        "required": True,
        "aliases": ["leading_index"],
    },
    "yield_curve": {
        "required": False,
        "aliases": ["bond_3y", "base_rate"],
    },
}


def _normalize_month(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:6]


def _active_alias_provider_map(provider_configs: Mapping[str, Mapping[str, Any]] | None) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for provider, raw_config in (provider_configs or {}).items():
        if not isinstance(raw_config, Mapping):
            continue
        for alias, cfg in raw_config.items():
            if isinstance(cfg, Mapping) and not bool(cfg.get("enabled", True)):
                continue
            aliases[str(alias)] = str(provider).strip().upper()
    return aliases


def _choose_reason(reasons: list[FreshnessReason]) -> FreshnessReason:
    if not reasons:
        return "CURRENT"
    for reason in REASON_PRECEDENCE:
        if reason in reasons:
            return reason
    return "CURRENT"


def _alias_reason(
    *,
    alias: str,
    provider: str,
    requested_end: str,
    latest_periods: Mapping[str, Any],
    failed_aliases: Mapping[str, Any],
    write_lock_fallback: bool,
) -> FreshnessReason:
    if write_lock_fallback:
        return "WRITE_LOCK_FALLBACK"
    if alias in failed_aliases:
        return "FETCH_ERROR"
    latest = _normalize_month(latest_periods.get(alias, ""))
    if not latest or latest < requested_end:
        return "SOURCE_LAG"
    return "CURRENT"


def build_macro_freshness_payload(
    *,
    requested_end: str,
    provider_configs: Mapping[str, Mapping[str, Any]] | None,
    latest_periods: Mapping[str, Any] | None,
    failed_aliases: Mapping[str, Any] | None = None,
    required_signal_groups: Mapping[str, Mapping[str, Any]] | None = None,
    write_lock_fallback: bool = False,
) -> dict[str, Any]:
    """Return a durable macro freshness payload for ingest summaries."""
    requested = _normalize_month(requested_end)
    active_aliases = _active_alias_provider_map(provider_configs)
    latest = {str(k): _normalize_month(v) for k, v in dict(latest_periods or {}).items()}
    failures = {str(k): str(v) for k, v in dict(failed_aliases or {}).items()}
    groups_cfg = dict(required_signal_groups or DEFAULT_REQUIRED_SIGNAL_GROUPS)

    alias_rows: dict[str, dict[str, Any]] = {}
    group_rows: dict[str, dict[str, Any]] = {}
    group_reasons: list[FreshnessReason] = []

    for group_name, raw_group in groups_cfg.items():
        group = dict(raw_group or {})
        candidate_aliases = [str(alias) for alias in group.get("aliases", []) if str(alias).strip()]
        configured_aliases = [alias for alias in candidate_aliases if alias in active_aliases]
        required = bool(group.get("required", False))

        if not configured_aliases:
            reason: FreshnessReason = "NOT_CONFIGURED" if required else "CURRENT"
            group_rows[str(group_name)] = {
                "required": required,
                "configured": False,
                "latest_period": "",
                "reason": reason,
                "aliases": [],
            }
            group_reasons.append(reason)
            continue

        alias_reasons: list[FreshnessReason] = []
        for alias in configured_aliases:
            reason = _alias_reason(
                alias=alias,
                provider=active_aliases[alias],
                requested_end=requested,
                latest_periods=latest,
                failed_aliases=failures,
                write_lock_fallback=write_lock_fallback,
            )
            alias_reasons.append(reason)
            alias_rows[alias] = {
                "group": str(group_name),
                "provider": active_aliases[alias],
                "configured": True,
                "latest_period": latest.get(alias, ""),
                "requested_end": requested,
                "reason": reason,
            }

        unique_reasons = set(alias_reasons)
        if len(unique_reasons) > 1 and "CURRENT" in unique_reasons:
            group_reason: FreshnessReason = "PARTIAL_PROVIDER_COVERAGE"
        else:
            group_reason = _choose_reason(alias_reasons)
        group_rows[str(group_name)] = {
            "required": required,
            "configured": True,
            "latest_period": max([latest.get(alias, "") for alias in configured_aliases] or [""]),
            "reason": group_reason,
            "aliases": configured_aliases,
        }
        group_reasons.append(group_reason)

    overall_reason = _choose_reason(group_reasons)
    return {
        "freshness": {
            "requested_end": requested,
            "overall_reason": overall_reason,
            "groups": group_rows,
            "aliases": alias_rows,
        }
    }


def summarize_macro_freshness(freshness_payload: Mapping[str, Any] | None) -> list[str]:
    """Return short Korean details for UI notices and banners."""
    payload = dict(freshness_payload or {})
    freshness = dict(payload.get("freshness") or {})
    if not freshness and isinstance(payload.get("details"), Mapping):
        freshness = dict(dict(payload.get("details") or {}).get("freshness") or {})
    groups = dict(freshness.get("groups") or {})
    details: list[str] = []
    labels = {
        "aggregate_exports": "수출",
        "aggregate_imports": "수입",
        "sector_export_proxies": "섹터 수출 proxy",
        "cpi": "물가",
        "fx": "환율",
        "leading_index": "경기선행",
        "yield_curve": "금리",
    }
    reason_labels = {
        "SOURCE_LAG": "소스 최신월 지연",
        "NOT_CONFIGURED": "미설정",
        "PARTIAL_PROVIDER_COVERAGE": "부분 커버리지",
        "FETCH_ERROR": "조회 실패",
        "WRITE_LOCK_FALLBACK": "저장 잠금",
    }
    for key, row in groups.items():
        reason = str(dict(row or {}).get("reason", "") or "").strip().upper()
        if reason in {"", "CURRENT"}:
            continue
        label = labels.get(str(key), str(key))
        reason_text = reason_labels.get(reason, reason)
        latest = _normalize_month(dict(row or {}).get("latest_period", ""))
        suffix = f" 최신 {latest[:4]}-{latest[4:6]}" if latest else ""
        details.append(f"{label}: {reason_text}{suffix}")
    return details
