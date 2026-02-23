"""
Cache key utilities for macro data loading.

This module intentionally fingerprints secrets and never returns raw key values.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _fingerprint(secret: str) -> str:
    """Return an 8-char SHA-256 fingerprint for a secret value."""
    normalized = (secret or "").strip()
    if not normalized:
        return "none"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]


def build_macro_cache_token(
    macro_series_cfg: dict[str, Any],
    ecos_key: str,
    kosis_key: str,
    secrets_mtime_ns: int,
) -> str:
    """Build a deterministic cache token for macro loader invalidation."""
    payload = {
        "macro_series_cfg": macro_series_cfg,
        "ecos_fp": _fingerprint(ecos_key),
        "kosis_fp": _fingerprint(kosis_key),
        "secrets_mtime_ns": int(secrets_mtime_ns),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
