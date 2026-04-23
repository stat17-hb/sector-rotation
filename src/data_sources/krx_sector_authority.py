"""Shared KR canonical sector-family helpers."""
from __future__ import annotations

from typing import Any


_KR_NON_CANONICAL_BROAD_EXACT_NAMES = {
    "KRX 100",
    "KRX 300",
    "KRX TMI",
    "KRX Mid 200",
    "KRX 중대형 TMI",
    "KRX 중형 TMI",
    "KRX 소형 TMI",
    "KRX 초소형 TMI",
}

_KR_NON_CANONICAL_FAMILY_PREFIXES = (
    "KRX 300 ",
)


def is_kr_canonical_sector_row(row: dict[str, Any]) -> bool:
    name = str(dict(row or {}).get("index_name", "")).strip()
    family = str(dict(row or {}).get("family", "")).strip()
    if family and family != "krx_dd_trd":
        return False
    if not name.startswith("KRX "):
        return False
    if name in _KR_NON_CANONICAL_BROAD_EXACT_NAMES:
        return False
    if any(name.startswith(prefix) for prefix in _KR_NON_CANONICAL_FAMILY_PREFIXES):
        return False
    return True


def canonicalize_kr_sector_universe_rows(
    rows: list[dict[str, Any]],
    *,
    benchmark_code: str = "",
    include_benchmark: bool = False,
) -> list[dict[str, Any]]:
    resolved_benchmark = str(benchmark_code or "").strip()
    selected: list[dict[str, Any]] = []
    seen_codes: set[str] = set()

    for row in rows:
        code = str(dict(row or {}).get("index_code", "")).strip()
        if not code or code in seen_codes:
            continue
        if include_benchmark and resolved_benchmark and code == resolved_benchmark:
            selected.append(dict(row))
            seen_codes.add(code)
            continue
        if is_kr_canonical_sector_row(dict(row)):
            selected.append(dict(row))
            seen_codes.add(code)

    return selected


__all__ = [
    "canonicalize_kr_sector_universe_rows",
    "is_kr_canonical_sector_row",
]
