"""Dashboard data/cache helpers extracted from app.py."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

from src.data_sources.common import shift_month_token
from src.data_sources.krx_sector_authority import canonicalize_kr_sector_universe_rows
from src.signals.flow import summarize_sector_investor_flow
from src.macro.series_utils import (
    build_regime_history_from_macro,
    build_regime_inflation_series,
    build_enabled_ecos_config,
    build_enabled_fred_config,
    build_enabled_kosis_config,
    extract_macro_series,
    to_plotly_time_index,
)
from src.data_sources.warehouse import (
    read_active_index_dimension,
    read_market_prices,
)
from src.data_sources.theme_lens import (
    get_theme_lens_artifact_key,
    load_theme_lens_cache_only,
    load_theme_proxy_signal_inputs,
)
from src.ui.data_status import (
    resolve_dashboard_status_banner,
    resolve_price_cache_banner_case,
)
from src.ui.components import render_status_strip

logger = logging.getLogger(__name__)

_KR_OFFICIAL_NAME_LOOKUP_WARNING_KEYS: set[str] = set()

settings: dict[str, Any] = {}
sector_map: dict[str, Any] = {}
macro_series_cfg: dict[str, Any] = {}
market_id = "KR"
market_profile: Any | None = None
CACHE_TTL = 21600
CURATED_SECTOR_PRICES_PATH = Path("data/curated/sector_prices.parquet")
MARKET_PRICE_TRANSIENT_OVERRIDE_KEY = "_market_price_transient_override"
INVESTOR_FLOW_TRANSIENT_OVERRIDE_KEY = "_investor_flow_transient_override"
DashboardProgressCallback = Callable[[dict[str, Any]], None]


def _clamp_progress_pct(pct: float | int) -> int:
    try:
        value = int(round(float(pct)))
    except Exception:
        return 0
    return max(0, min(100, value))


def _emit_progress(
    progress_callback: DashboardProgressCallback | None,
    *,
    task: str,
    phase: str,
    pct: float | int,
    detail: str = "",
    status: str = "running",
    meta: dict[str, Any] | None = None,
) -> None:
    if progress_callback is None:
        return
    event = {
        "task": str(task).strip(),
        "phase": str(phase).strip(),
        "pct": _clamp_progress_pct(pct),
        "detail": str(detail).strip(),
        "status": str(status).strip().lower() or "running",
    }
    if meta:
        event["meta"] = dict(meta)
    progress_callback(event)


def configure_dashboard_env(
    *,
    settings_obj: dict[str, Any],
    sector_map_obj: dict[str, Any],
    macro_series_cfg_obj: dict[str, Any],
    market_id_obj: str,
    market_profile_obj: Any,
    cache_ttl: int,
    curated_sector_prices_path: Path,
) -> None:
    """Configure module-level dashboard settings used by cached helpers."""
    global settings, sector_map, macro_series_cfg, market_id, market_profile, CACHE_TTL, CURATED_SECTOR_PRICES_PATH
    settings = dict(settings_obj)
    sector_map = dict(sector_map_obj)
    macro_series_cfg = dict(macro_series_cfg_obj)
    market_id = str(market_id_obj or "KR").strip().upper() or "KR"
    market_profile = market_profile_obj
    CACHE_TTL = int(cache_ttl)
    CURATED_SECTOR_PRICES_PATH = Path(curated_sector_prices_path)


def clear_market_price_transient_override() -> None:
    try:
        st.session_state.pop(MARKET_PRICE_TRANSIENT_OVERRIDE_KEY, None)
    except Exception:
        pass


def set_market_price_transient_override(
    *,
    market_id_arg: str,
    requested_end: str,
    status: str,
    summary: dict[str, Any],
    frame: pd.DataFrame,
) -> None:
    try:
        st.session_state[MARKET_PRICE_TRANSIENT_OVERRIDE_KEY] = {
            "market": str(market_id_arg or market_id).strip().upper() or market_id,
            "requested_end": str(requested_end or "").strip(),
            "status": str(status or "LIVE").strip().upper() or "LIVE",
            "summary": dict(summary),
            "frame": frame.copy(),
        }
    except Exception:
        pass


def _load_market_price_transient_override(
    *,
    market_id_arg: str,
    start_str: str,
    end_str: str,
    index_codes: list[str],
) -> tuple[str, pd.DataFrame] | None:
    try:
        payload = st.session_state.get(MARKET_PRICE_TRANSIENT_OVERRIDE_KEY)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    normalized_market = str(market_id_arg or market_id).strip().upper() or market_id
    if str(payload.get("market", "")).strip().upper() != normalized_market:
        return None

    requested_end = "".join(ch for ch in str(payload.get("requested_end", "")) if ch.isdigit())[:8]
    if requested_end != str(end_str):
        return None

    frame = payload.get("frame")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None

    filtered = _filter_cached_sector_prices(
        frame,
        index_codes=index_codes,
        end_date_str=end_str,
    )
    if filtered.empty:
        return None
    filtered = filtered.loc[pd.DatetimeIndex(filtered.index) >= pd.Timestamp(start_str)].copy()
    if filtered.empty:
        return None
    return str(payload.get("status", "LIVE")).strip().upper() or "LIVE", filtered


def clear_investor_flow_transient_override() -> None:
    try:
        st.session_state.pop(INVESTOR_FLOW_TRANSIENT_OVERRIDE_KEY, None)
    except Exception:
        pass


def set_investor_flow_transient_override(
    *,
    market_id_arg: str,
    requested_end: str,
    status: str,
    summary: dict[str, Any],
    frame: pd.DataFrame,
) -> None:
    try:
        st.session_state[INVESTOR_FLOW_TRANSIENT_OVERRIDE_KEY] = {
            "market": str(market_id_arg or market_id).strip().upper() or market_id,
            "requested_end": str(requested_end or "").strip(),
            "status": str(status or "LIVE").strip().upper() or "LIVE",
            "summary": dict(summary),
            "frame": frame.copy(),
        }
    except Exception:
        pass


def _load_investor_flow_transient_override(
    *,
    market_id_arg: str,
    start_str: str,
    end_str: str,
) -> tuple[str, dict[str, Any], pd.DataFrame] | None:
    try:
        payload = st.session_state.get(INVESTOR_FLOW_TRANSIENT_OVERRIDE_KEY)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    normalized_market = str(market_id_arg or market_id).strip().upper() or market_id
    if str(payload.get("market", "")).strip().upper() != normalized_market:
        return None

    requested_end = "".join(ch for ch in str(payload.get("requested_end", "")) if ch.isdigit())[:8]
    if requested_end != str(end_str):
        return None

    frame = payload.get("frame")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None

    filtered = frame.copy()
    filtered.index = pd.DatetimeIndex(filtered.index)
    filtered = filtered.loc[
        (filtered.index >= pd.Timestamp(start_str)) & (filtered.index <= pd.Timestamp(end_str))
    ].copy()
    if filtered.empty:
        return None

    status_detail = dict(payload.get("summary") or {})
    status_detail["warehouse_write_skipped"] = True
    status_detail.setdefault("end", requested_end)
    status_detail.setdefault("requested_end", requested_end)
    return str(payload.get("status", "LIVE")).strip().upper() or "LIVE", status_detail, filtered

def _parquet_key(path: str) -> tuple:
    """Return (mtime_ns, size) for a parquet file, or (0, 0) if missing.

    ns precision + size avoids mtime collision.
    """
    p = Path(path)
    if not p.exists():
        return (0, 0)
    s = p.stat()
    return (s.st_mtime_ns, s.st_size)


def _load_api_key(name: str) -> str:
    """Load API key from Streamlit secrets with environment fallback."""
    try:
        value = str(st.secrets.get(name, "")).strip()
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name, "").strip()


def _secrets_mtime_ns(path: str = ".streamlit/secrets.toml") -> int:
    """Return mtime_ns for secrets file, or 0 if missing."""
    p = Path(path)
    if not p.exists():
        return 0
    return p.stat().st_mtime_ns


def _load_bool_setting(name: str, default: bool) -> bool:
    """Load a boolean feature flag from Streamlit secrets/environment."""
    raw = _load_api_key(name)
    if not raw:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _is_mobile_client() -> bool:
    """Best-effort mobile client detection from request user-agent."""
    try:
        user_agent = str(st.context.headers.get("user-agent", "")).lower()
    except Exception:
        user_agent = ""

    mobile_tokens = (
        "android",
        "iphone",
        "ipad",
        "ipod",
        "mobile",
        "windows phone",
    )
    return any(token in user_agent for token in mobile_tokens)


def _macro_cache_token() -> str:
    """Build cache token including config + API key fingerprints."""
    from src.data_sources.cache_keys import build_macro_cache_token

    token_cfg = dict(macro_series_cfg)
    token_cfg["__market__"] = market_id
    if market_id == "US":
        ecos_key = _load_api_key("FRED_API_KEY")
        kosis_key = ""
    else:
        ecos_key = _load_api_key("ECOS_API_KEY")
        kosis_key = _load_api_key("KOSIS_API_KEY")
    return build_macro_cache_token(
        macro_series_cfg=token_cfg,
        ecos_key=ecos_key,
        kosis_key=kosis_key,
        secrets_mtime_ns=_secrets_mtime_ns(),
    )


def _krx_provider_configured() -> str:
    """Return configured KRX provider value (AUTO/OPENAPI/PYKRX)."""
    if market_id == "US":
        return str(getattr(market_profile, "price_provider", "YFINANCE")).strip().upper()
    from src.data_sources.krx_openapi import get_krx_provider

    return get_krx_provider(_load_api_key("KRX_PROVIDER"))


def _krx_provider_effective() -> str:
    """Return runtime-effective provider after AUTO resolution."""
    if market_id == "US":
        return str(getattr(market_profile, "price_provider", "YFINANCE")).strip().upper()
    configured = _krx_provider_configured()
    if configured == "AUTO":
        return "OPENAPI" if _load_api_key("KRX_OPENAPI_KEY") else "PYKRX"
    return configured


def _price_cache_token() -> str:
    """Build cache token for market price loader (KRX provider + key fingerprint)."""
    if market_id == "US":
        payload = {
            "market_id": market_id,
            "price_provider": _krx_provider_effective(),
            "secrets_mtime_ns": _secrets_mtime_ns(),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    from src.data_sources.cache_keys import build_price_cache_token

    return build_price_cache_token(
        krx_provider=_krx_provider_configured(),
        krx_openapi_key=_load_api_key("KRX_OPENAPI_KEY"),
        secrets_mtime_ns=_secrets_mtime_ns(),
    )


def _price_artifact_key() -> tuple:
    """Return cache-busting key for raw/warm price artifacts."""
    if market_id == "US":
        from src.data_sources.yfinance_sectors import get_price_artifact_key
    else:
        from src.data_sources.krx_indices import get_price_artifact_key

    return get_price_artifact_key()


def _investor_flow_artifact_key() -> tuple:
    """Return a cache-busting key for investor-flow / flow-proxy state."""
    if market_id == "US":
        return _price_artifact_key()
    if market_id != "KR":
        return (0, 0, "", "", "")
    from src.data_sources.krx_investor_flow import get_investor_flow_artifact_key

    return get_investor_flow_artifact_key()


def _macro_artifact_key() -> tuple:
    """Return cache-busting key for macro warehouse artifacts."""
    from src.data_sources.macro_sync import get_macro_artifact_key

    return get_macro_artifact_key(market=market_id)


def _probe_market_status() -> str:
    """Return current market-data availability from the warehouse."""
    if market_id == "US":
        from src.data_sources.yfinance_sectors import probe_market_status
    else:
        from src.data_sources.krx_indices import probe_market_status

    return probe_market_status()


def _probe_macro_status() -> str:
    """Return current macro-data availability from the warehouse."""
    from src.data_sources.macro_sync import probe_macro_status

    return probe_macro_status(market=market_id)


def _probe_investor_flow_status() -> str:
    """Return current investor-flow availability from the warehouse."""
    if market_id != "KR":
        return "SAMPLE"
    from src.data_sources.krx_investor_flow import probe_investor_flow_status

    return probe_investor_flow_status()


def _is_placeholder_kr_name(index_code: str, index_name: str) -> bool:
    code = str(index_code or "").strip()
    name = str(index_name or "").strip()
    return bool(code) and (not name or name == code)


def _kr_active_index_metadata_lookup() -> dict[str, dict[str, str]]:
    active_rows = read_active_index_dimension(market="KR")
    lookup = {
        str(row["index_code"]).strip(): {
            "index_name": str(row["index_name"]).strip(),
            "taxonomy_label": str(row.get("taxonomy_label", "")).strip(),
        }
        for _, row in active_rows.iterrows()
        if str(row.get("index_code", "")).strip()
    }
    missing_codes = [
        str(row["index_code"]).strip()
        for _, row in active_rows.iterrows()
        if _is_placeholder_kr_name(
            str(row.get("index_code", "")).strip(),
            str(row.get("index_name", "")).strip(),
        )
        or _is_placeholder_kr_name(
            str(row.get("index_code", "")).strip(),
            str(row.get("taxonomy_label", "")).strip(),
        )
    ]
    for code, meta in list(lookup.items()):
        if _is_placeholder_kr_name(code, meta.get("index_name", "")):
            meta["index_name"] = ""
        if _is_placeholder_kr_name(code, meta.get("taxonomy_label", "")):
            meta["taxonomy_label"] = ""

    try:
        from src.data_sources.krx_indices import discover_kr_index_rows

        discovered = {
            str(row.get("index_code", "")).strip(): {
                "index_name": str(row.get("index_name", "")).strip(),
                "taxonomy_label": str(row.get("taxonomy_label", "")).strip(),
            }
            for row in discover_kr_index_rows(str(settings.get("benchmark_code", "1001")))
            if str(row.get("index_code", "")).strip()
        }
    except Exception as exc:
        warning_key = " ".join(str(exc).split())[:500]
        if warning_key not in _KR_OFFICIAL_NAME_LOOKUP_WARNING_KEYS:
            _KR_OFFICIAL_NAME_LOOKUP_WARNING_KEYS.add(warning_key)
            logger.warning("KR official name lookup fallback failed: %s", exc)
        return lookup

    for code in missing_codes:
        discovered_meta = discovered.get(code)
        if not discovered_meta:
            continue
        current = lookup.setdefault(code, {"index_name": "", "taxonomy_label": ""})
        if not current.get("index_name") and not _is_placeholder_kr_name(code, discovered_meta.get("index_name", "")):
            current["index_name"] = discovered_meta["index_name"]
        if not current.get("taxonomy_label") and not _is_placeholder_kr_name(code, discovered_meta.get("taxonomy_label", "")):
            current["taxonomy_label"] = discovered_meta["taxonomy_label"]
    return lookup


def _kr_active_index_name_lookup() -> dict[str, str]:
    return {
        code: str(meta.get("index_name", "")).strip()
        for code, meta in _kr_active_index_metadata_lookup().items()
        if str(meta.get("index_name", "")).strip()
    }


def _normalize_kr_named_frame(
    frame: pd.DataFrame,
    *,
    code_col: str,
    name_col: str,
) -> pd.DataFrame:
    if frame.empty or code_col not in frame.columns or name_col not in frame.columns:
        return frame
    name_lookup = _kr_active_index_name_lookup()
    if not name_lookup:
        return frame

    normalized = frame.copy()
    normalized[code_col] = normalized[code_col].astype(str)
    normalized[name_col] = [
        name_lookup.get(code, name) if _is_placeholder_kr_name(code, name) else name
        for code, name in zip(
            normalized[code_col].astype(str).tolist(),
            normalized[name_col].astype(str).tolist(),
        )
    ]
    return normalized


def _normalize_kr_sector_universe_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rows:
        return rows
    metadata_lookup = _kr_active_index_metadata_lookup()
    if not metadata_lookup:
        return rows
    normalized: list[dict[str, Any]] = []
    for row in rows:
        code = str(dict(row or {}).get("index_code", "")).strip()
        name = str(dict(row or {}).get("index_name", "")).strip()
        taxonomy_label = str(dict(row or {}).get("taxonomy_label", "")).strip()
        meta = metadata_lookup.get(code, {})
        normalized.append(
            {
                **dict(row or {}),
                "index_code": code,
                "index_name": meta.get("index_name", name) if _is_placeholder_kr_name(code, name) else name,
                "taxonomy_label": (
                    meta.get("taxonomy_label", taxonomy_label)
                    if _is_placeholder_kr_name(code, taxonomy_label)
                    else taxonomy_label
                ),
            }
        )
    return normalized


def _canonicalize_kr_sector_universe_rows(
    rows: list[dict[str, Any]],
    *,
    benchmark_code: str = "",
    include_benchmark: bool = False,
) -> list[dict[str, Any]]:
    normalized_rows = _normalize_kr_sector_universe_rows(rows)
    if not normalized_rows:
        return normalized_rows
    return canonicalize_kr_sector_universe_rows(
        normalized_rows,
        benchmark_code=benchmark_code,
        include_benchmark=include_benchmark,
    )


def _all_sector_codes(benchmark_code: str) -> list[str]:
    """Return the unique sector universe used by the dashboard."""
    all_codes: list[str] = []
    for regime_data in sector_map.get("regimes", {}).values():
        for s in regime_data.get("sectors", []):
            code = str(s["code"])
            if code not in all_codes:
                all_codes.append(code)
    if benchmark_code and str(benchmark_code) not in all_codes:
        all_codes.append(str(benchmark_code))
    for code in settings.get("reference_index_codes", []) or []:
        normalized = str(code).strip()
        if normalized and normalized not in all_codes:
            all_codes.append(normalized)
    return all_codes


def get_market_index_universe_codes(benchmark_code: str, market_id_arg: str) -> list[str]:
    """Return the runtime market universe, using dim_index as KR authority."""
    normalized_market = str(market_id_arg or market_id).strip().upper() or market_id
    if normalized_market != "KR":
        return _all_sector_codes(benchmark_code)

    try:
        from src.data_sources.krx_indices import get_active_kr_index_universe_codes

        universe_codes = get_active_kr_index_universe_codes(str(benchmark_code))
    except Exception as exc:
        logger.warning("KR dim_index universe resolution failed: %s", exc)
        universe_codes = []
    if universe_codes:
        return list(dict.fromkeys(universe_codes))

    logger.warning(
        "KR dim_index universe unavailable after bootstrap/repair; falling back to legacy configured subset."
    )
    return _all_sector_codes(benchmark_code)


def _resolve_market_end_date(benchmark_code: str) -> date:
    """Resolve the market end date once per app run."""
    from src.transforms.calendar import get_last_business_day
    from src.data_sources.warehouse import get_market_latest_dates

    calendar_date = get_last_business_day(
        provider=_krx_provider_effective(),
        benchmark_code=benchmark_code,
    )
    normalized_market = str(market_id or "").strip().upper()
    if normalized_market != "KR":
        return calendar_date

    try:
        latest_dates = get_market_latest_dates([str(benchmark_code).strip()], market=normalized_market)
        latest_text = str(latest_dates.get(str(benchmark_code).strip(), "") or "").strip()
        latest_date = pd.Timestamp(latest_text).date() if latest_text else None
    except Exception as exc:
        logger.debug("Warehouse market-end-date fallback lookup failed: %s", exc)
        latest_date = None
    if latest_date is not None and calendar_date < latest_date <= date.today():
        return latest_date
    return calendar_date


def _build_regime_inflation_series(
    *,
    macro_df: pd.DataFrame,
    macro_series_cfg_obj: dict[str, Any],
    market_id_arg: str,
) -> pd.Series:
    """Backward-compatible wrapper around the shared regime inflation helper."""
    return build_regime_inflation_series(
        macro_df=macro_df,
        macro_series_cfg=macro_series_cfg_obj,
        market_id=str(market_id_arg or market_id),
    )


def _maybe_schedule_startup_krx_warm(
    benchmark_code: str,
    price_years: int,
    end_date: date,
) -> None:
    """Schedule a non-blocking warm job so first interactive load can stay cached."""
    if market_id != "KR":
        return
    if not _load_bool_setting("KRX_WARM_ON_STARTUP", True):
        return
    if _krx_provider_effective() == "OPENAPI":
        return


def _market_range_strings(end_date_str: str, price_years: int) -> tuple[str, str]:
    """Return the dashboard market-data lookback window."""
    end_date = pd.Timestamp(end_date_str).date()
    start_date = end_date - timedelta(days=365 * price_years)
    return start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")


def _investor_flow_range_strings(end_date_str: str, lookback_days: int = 120) -> tuple[str, str]:
    end_date = pd.Timestamp(end_date_str).date()
    start_date = end_date - timedelta(days=int(lookback_days))
    return start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")


def _format_yyyymmdd(value: str) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) != 8:
        return ""
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"


def _market_cache_label(warm_status: dict[str, object]) -> str:
    """Return a user-facing cache label preferring the warehouse coverage date."""
    warm_end = _format_yyyymmdd(
        str(warm_status.get("end", "") or warm_status.get("watermark_key", ""))
    )
    if warm_end:
        return f"market data ({warm_end})"
    return "market data"

def _build_market_refresh_notice(summary: dict[str, object]) -> tuple[str, str]:
    """Map manual refresh summary into a user-facing flash message."""
    status = str(summary.get("status", "")).strip().upper()
    coverage_complete = bool(summary.get("coverage_complete"))
    delta_codes = list(summary.get("delta_codes", []))
    failed_days = list(summary.get("failed_days", []))
    failed_codes = dict(summary.get("failed_codes") or {})
    provider_label = str(summary.get("provider", _krx_provider_effective()) or "").strip() or "provider"

    if bool(summary.get("warehouse_write_skipped")) and status == "LIVE":
        rows = int(summary.get("rows", 0) or 0)
        return (
            "warning",
            f"Market data refresh fetched live data via {provider_label}, but warehouse write lock blocked persistence. "
            f"Showing temporary preview ({rows} rows); close duplicate app sessions and refresh again to save it.",
        )
    if status == "LIVE" and delta_codes:
        return (
            "success",
            f"Market data refresh completed via {provider_label} ({len(delta_codes)} codes updated).",
        )
    if coverage_complete and not delta_codes:
        return ("info", "Market data already current; the latest local cache is in use.")
    if failed_days or failed_codes:
        detail = ""
        if failed_codes:
            first_code, first_reason = next(iter(failed_codes.items()))
            detail = f" First failure: {first_code}={first_reason}"
        elif failed_days:
            detail = f" Failed day: {failed_days[0]}"
        return (
            "warning",
            "Market data refresh fell back to cache after an incomplete live refresh. "
            f"Retry later or continue with cache.{detail}",
        )
    if status == "CACHED":
        return ("warning", "Market data refresh fell back to cache.")
    return ("error", "Market data refresh did not complete successfully.")


def _build_macro_refresh_notice(summary: dict[str, object]) -> tuple[str, str]:
    """Map macro warehouse refresh summary into a user-facing flash message."""
    status = str(summary.get("status", "")).strip().upper()
    coverage_complete = bool(summary.get("coverage_complete"))
    rows = int(summary.get("rows", 0) or 0)

    if status == "LIVE":
        return ("success", f"Macro data refresh completed ({rows} rows available).")
    if status == "CACHED" and coverage_complete:
        return ("info", "Macro data already current in the local warehouse.")
    if status == "CACHED":
        return ("warning", "Macro data refresh fell back to warehouse cache.")
    return ("error", "Macro data refresh did not complete successfully.")


def _build_investor_flow_refresh_notice(summary: dict[str, object]) -> tuple[str, str]:
    """Map manual investor-flow refresh summary into a user-facing flash message."""
    status = str(summary.get("status", "")).strip().upper()
    coverage_complete = bool(summary.get("coverage_complete"))
    rows = int(summary.get("rows", 0) or 0)
    collected_rows = int(summary.get("collected_rows", rows) or 0)
    warehouse_write_skipped = bool(summary.get("warehouse_write_skipped"))
    failed_codes: dict[str, str] = dict(summary.get("failed_codes") or {})
    window: dict[str, object] = dict(summary.get("window") or {})
    bootstrap_partial_preview = rows > 0 and not str(window.get("complete_cursor", "")).strip()
    sector_fail_count = sum(1 for k in failed_codes if k.startswith("sector:"))
    ticker_fail_count = sum(1 for k in failed_codes if not k.startswith("sector:"))
    auth_fail_count = sum(1 for v in failed_codes.values() if str(v).startswith("AUTH_REQUIRED:"))
    access_denied_count = sum(
        1
        for v in failed_codes.values()
        if str(v).startswith("ACCESS_DENIED:") or "access denied" in str(v).lower()
    )
    mode = str(window.get("mode", "")).strip()
    anchor_start = _format_yyyymmdd(str(window.get("anchor_start", "")))
    anchor_reason = str(window.get("anchor_reason", "")).strip()
    window_parts: list[str] = []
    if mode:
        window_parts.append(f"resolver={mode}")
    if anchor_start:
        window_parts.append(f"anchor={anchor_start}")
    if anchor_reason:
        window_parts.append(f"reason={anchor_reason}")
    window_detail = f" [{' / '.join(window_parts)}]" if window_parts else ""

    if warehouse_write_skipped and rows > 0:
        return (
            "warning",
            f"투자자수급 실시간 수집 {collected_rows}건은 완료됐지만 warehouse write lock으로 저장하지 못했습니다. 현재 세션에서 임시 preview만 표시하며, 이번 결과는 warehouse에 반영되지 않았습니다.",
        )
    if warehouse_write_skipped:
        return ("error", "투자자수급 실시간 수집은 완료됐지만 warehouse write lock으로 결과를 저장하지 못했습니다.")
    if status == "LIVE" and coverage_complete:
        return ("success", f"투자자수급 갱신 완료 ({rows}건 저장).")
    if status == "LIVE":
        parts: list[str] = []
        if sector_fail_count:
            parts.append(f"섹터 {sector_fail_count}개 구성종목 조회 실패")
        if ticker_fail_count:
            parts.append(f"종목 {ticker_fail_count}개 수급 수집 실패")
        detail = f" ({', '.join(parts)})" if parts else ""
        if bootstrap_partial_preview:
            return ("warning", f"투자자수급 갱신 완료 (부분 커버리지{detail}). complete cursor가 없어 이번 partial preview를 표시합니다.{window_detail}")
        return ("warning", f"투자자수급 갱신 완료 (부분 커버리지{detail}). complete cursor 기준 기존 데이터를 유지합니다.{window_detail}")
    if status == "CACHED":
        if coverage_complete:
            return ("info", "투자자수급 데이터가 이미 최신 complete cursor까지 반영되어 있습니다.")
        if auth_fail_count:
            return (
                "error",
                "투자자수급 갱신 실패: KRX 데이터마켓 로그인 세션이 필요합니다. "
                "환경변수 또는 Streamlit secrets에 KRX_ID / KRX_PW를 설정한 뒤 다시 시도하세요.",
            )
        if access_denied_count:
            return (
                "error",
                "투자자수급 갱신 실패: KRX가 수급 trading-value endpoint 접근을 차단했습니다. "
                "KRX_ID / KRX_PW를 설정한 인증 세션으로 다시 시도하세요. 인증 후에도 동일하면 KRX 정책상 비공식 endpoint가 차단된 상태입니다.",
            )
        if bootstrap_partial_preview:
            return ("warning", f"실시간 투자자수급 갱신은 완료되지 않았지만, bootstrap partial preview 데이터를 표시합니다.{window_detail}")
        snap_detail = ""
        if failed_codes:
            snap_detail = f" — {len(failed_codes)}개 항목 실패로 캐시 사용"
        return ("warning", f"투자자수급 갱신 실패 → 캐시 데이터 사용{snap_detail}.{window_detail}")
    return ("error", "투자자수급 갱신이 완료되지 않았습니다.")


def _legacy_show_notice_toast(notice: tuple[str, str] | None) -> None:
    """Render a transient toast for one-off refresh results."""
    if not notice:
        return

    level, message = notice
    prefix = {
        "success": "완료",
        "info": "안내",
        "warning": "주의",
        "error": "오류",
    }.get(level, "안내")
    st.toast(f"{prefix}: {message}")


def _legacy_render_dashboard_status_banner(banner: dict[str, object] | None) -> None:
    """Render the single top-of-page system status banner."""
    if not banner:
        return

    level = str(banner.get("level", "info")).strip().lower()
    title = str(banner.get("title", "")).strip()
    message = str(banner.get("message", "")).strip()
    details = [str(item).strip() for item in banner.get("details", []) if str(item).strip()]
    body = f"**{title}**\n\n{message}" if title else message

    if level == "error":
        st.error(body)
    elif level == "warning":
        st.warning(body)
    else:
        st.info(body)

    if details:
        with st.expander("상세 상태", expanded=False):
            for detail in details:
                st.write(f"- {detail}")


def _show_notice_toast(notice: tuple[str, str] | None) -> None:
    """Render a transient toast for one-off refresh results."""
    if not notice:
        return

    level, message = notice
    prefix = {
        "success": "Done",
        "info": "Info",
        "warning": "Warning",
        "error": "Error",
    }.get(level, "Info")
    st.toast(f"{prefix}: {message}")


def _render_dashboard_status_banner(banner: dict[str, object] | None) -> None:
    """Render top-of-page system status in the shared compact strip."""
    if not banner:
        return

    render_status_strip(banner)


def _openapi_cache_fallback_note(warm_status: dict[str, object]) -> str:
    """Return a short reason string for retryable OpenAPI cache fallback."""
    failed_days = [str(day) for day in warm_status.get("failed_days", []) if str(day).strip()]
    failed_codes = {
        str(code).strip(): str(detail).strip()
        for code, detail in dict(warm_status.get("failed_codes") or {}).items()
        if str(code).strip() and str(detail).strip()
    }

    if failed_days:
        preview = ", ".join(failed_days[:3])
        suffix = "" if len(failed_days) <= 3 else ", ..."
        return f"Latest OpenAPI warm was incomplete (failed days: {preview}{suffix})."
    if failed_codes:
        preview = ", ".join(sorted(failed_codes)[:3])
        suffix = "" if len(failed_codes) <= 3 else ", ..."
        return f"Latest OpenAPI warm failed for codes: {preview}{suffix}."

    warm_state = str(warm_status.get("status", "")).strip().upper()
    if warm_state:
        return f"Latest OpenAPI warm did not confirm current coverage (status={warm_state})."

    return "Latest OpenAPI warm did not confirm current coverage."


def _investor_flow_is_fresh(status_detail: dict[str, Any], *, market_end_date_str: str) -> bool:
    if not status_detail:
        return False
    latest = "".join(ch for ch in str(status_detail.get("end") or status_detail.get("watermark_key") or "") if ch.isdigit())[:8]
    return bool(status_detail.get("coverage_complete")) and latest == str(market_end_date_str)


def _investor_flow_is_actionable(
    *,
    market_id_arg: str,
    flow_status: str,
    flow_fresh: bool,
    flow_detail: dict[str, Any],
) -> bool:
    normalized_market = str(market_id_arg or market_id).strip().upper() or market_id
    if normalized_market != "KR":
        return False
    if str(flow_status).strip().upper() not in {"LIVE", "CACHED"}:
        return False
    if not flow_fresh:
        return False
    if bool(flow_detail.get("bootstrap_partial_preview")):
        return False
    if bool(flow_detail.get("warehouse_write_skipped")):
        return False
    return True


@st.cache_data(ttl=600)
def _cached_api_preflight(timeout_sec: int = 3, market_id_arg: str | None = None) -> dict:
    """Cached API endpoint reachability check (10 min TTL)."""
    from src.data_sources.preflight import run_api_preflight

    return run_api_preflight(timeout_sec=timeout_sec, market_id=market_id_arg or market_id)


# Named cache functions (R8)


@st.cache_data(ttl=CACHE_TTL)
def _cached_sector_prices(
    market_id_arg: str,
    end_date_str: str,
    benchmark_code: str,
    price_years: int,
    price_cache_token: str,
    price_artifact_key: tuple,
):
    """Fetch or load sector prices. Includes KRX provider/key cache token."""
    _ = (price_cache_token, price_artifact_key)
    if str(market_id_arg or "KR").strip().upper() == "US":
        from src.data_sources.yfinance_sectors import load_sector_prices
    else:
        from src.data_sources.krx_indices import load_sector_prices

    all_codes = get_market_index_universe_codes(benchmark_code, market_id_arg)
    start_str, end_str = _market_range_strings(end_date_str, price_years)

    transient = _load_market_price_transient_override(
        market_id_arg=market_id_arg,
        start_str=start_str,
        end_str=end_str,
        index_codes=all_codes,
    )
    if transient is not None:
        return transient

    status, df = load_sector_prices(all_codes, start_str, end_str)
    return status, df


def _filter_cached_sector_prices(
    sector_prices: pd.DataFrame,
    *,
    index_codes: list[str],
    end_date_str: str,
) -> pd.DataFrame:
    """Filter cached sector prices down to the requested universe and end date."""
    if sector_prices.empty or "index_code" not in sector_prices.columns or "close" not in sector_prices.columns:
        return pd.DataFrame()

    filtered = sector_prices.copy()
    filtered.index = pd.DatetimeIndex(filtered.index)
    filtered = filtered.sort_index()
    filtered["index_code"] = filtered["index_code"].astype(str)
    if "index_name" not in filtered.columns:
        filtered["index_name"] = filtered["index_code"]

    requested_codes = {str(code).strip() for code in index_codes if str(code).strip()}
    end_ts = pd.Timestamp(end_date_str).normalize()
    filtered = filtered[
        filtered["index_code"].isin(requested_codes)
        & (filtered.index <= end_ts)
    ]
    if filtered.empty:
        return pd.DataFrame()
    return filtered[["index_code", "index_name", "close"]]


def _load_analysis_sector_prices_from_cache(
    market_id_arg: str,
    end_date_str: str,
    benchmark_code: str,
) -> pd.DataFrame:
    """Load the widest cached analysis history without triggering live refreshes."""
    all_codes = get_market_index_universe_codes(benchmark_code, market_id_arg)
    cached = _filter_cached_sector_prices(
        read_market_prices(all_codes, "19000101", end_date_str, market=market_id_arg),
        index_codes=all_codes,
        end_date_str=end_date_str,
    )
    if not cached.empty:
        return cached

    if not CURATED_SECTOR_PRICES_PATH.exists():
        return pd.DataFrame()

    try:
        curated = pd.read_parquet(CURATED_SECTOR_PRICES_PATH)
    except Exception as exc:
        logger.warning("Analysis canvas curated cache load failed: %s", exc)
        return pd.DataFrame()

    return _filter_cached_sector_prices(
        curated,
        index_codes=all_codes,
        end_date_str=end_date_str,
    )


@st.cache_data(ttl=CACHE_TTL)
def _cached_analysis_sector_prices(
    market_id_arg: str,
    end_date_str: str,
    benchmark_code: str,
    price_years: int,
    price_artifact_key: tuple,
) -> pd.DataFrame:
    """Load analysis-canvas prices from cache/warehouse only."""
    _ = (price_years, price_artifact_key)
    return _load_analysis_sector_prices_from_cache(market_id_arg, end_date_str, benchmark_code)


@st.cache_data(ttl=CACHE_TTL)
def _cached_investor_flow(
    market_id_arg: str,
    end_date_str: str,
    flow_artifact_key: tuple,
):
    """Load KR investor-flow or US flow-proxy data for the dashboard."""
    _ = flow_artifact_key
    normalized_market = str(market_id_arg or market_id).strip().upper() or market_id
    start_str, end_str = _investor_flow_range_strings(end_date_str)
    if normalized_market == "US":
        from src.data_sources.us_flow_proxies import load_us_flow_proxies

        status, frame, status_detail = load_us_flow_proxies(
            sector_map=sector_map,
            start=start_str,
            end=end_str,
        )
        return status, _investor_flow_is_fresh(status_detail, market_end_date_str=end_str), status_detail, frame
    if normalized_market != "KR":
        return "SAMPLE", False, {}, pd.DataFrame()

    from src.data_sources.krx_investor_flow import load_sector_investor_flow, read_warm_status

    transient = _load_investor_flow_transient_override(
        market_id_arg=normalized_market,
        start_str=start_str,
        end_str=end_str,
    )
    if transient is not None:
        status, status_detail, frame = transient
        if not frame.empty:
            frame = _normalize_kr_named_frame(frame, code_col="sector_code", name_col="sector_name")
        return status, _investor_flow_is_fresh(status_detail, market_end_date_str=end_str), status_detail, frame

    status, frame = load_sector_investor_flow(
        sector_map=sector_map,
        start=start_str,
        end=end_str,
        market=normalized_market,
        allow_bootstrap_partial_preview=True,
    )
    status_detail = dict(read_warm_status())
    if status == "CACHED" and not frame.empty and not bool(status_detail.get("coverage_complete")):
        status_detail["bootstrap_partial_preview"] = True
    if not frame.empty:
        frame = _normalize_kr_named_frame(frame, code_col="sector_code", name_col="sector_name")
    return status, _investor_flow_is_fresh(status_detail, market_end_date_str=end_str), status_detail, frame


@st.cache_data(ttl=CACHE_TTL)
def _cached_theme_lens(
    market_id_arg: str,
    market_end_date_str: str,
    theme_lens_artifact_key: tuple,
) -> tuple[str, list[dict[str, Any]]]:
    """Load KR theme lens rows from warehouse cache only."""
    _ = theme_lens_artifact_key
    normalized_market = str(market_id_arg or market_id).strip().upper() or market_id
    if normalized_market != "KR":
        return "UNAVAILABLE", []
    return load_theme_lens_cache_only(asof_date=market_end_date_str)


@st.cache_data(ttl=CACHE_TTL)
def _cached_macro(market_id_arg: str, macro_cache_token: str, market_end_date_str: str):
    """Fetch or load macro data. Keyed by config + API key fingerprint token."""
    end_ym = str(market_end_date_str)[:6]
    start_ym = shift_month_token(end_ym, -59)
    normalized_market = str(market_id_arg or market_id).strip().upper() or market_id

    if normalized_market == "US":
        from src.data_sources.fred import load_fred_macro

        fred_cfg = build_enabled_fred_config(macro_series_cfg.get("fred", {}))
        if fred_cfg:
            return load_fred_macro(start_ym, end_ym, series_config=fred_cfg, market=normalized_market)
        return ("LIVE", pd.DataFrame())

    from src.data_sources.ecos import load_ecos_macro
    from src.data_sources.kosis import load_kosis_macro

    ecos_cfg = build_enabled_ecos_config(macro_series_cfg.get("ecos", {}))
    if ecos_cfg:
        ecos_status, ecos_df = load_ecos_macro(start_ym, end_ym, series_config=ecos_cfg)
    else:
        ecos_status, ecos_df = ("LIVE", pd.DataFrame())

    kosis_cfg = build_enabled_kosis_config(macro_series_cfg.get("kosis", {}))
    if kosis_cfg:
        kosis_status, kosis_df = load_kosis_macro(start_ym, end_ym, series_config=kosis_cfg)
    else:
        kosis_status, kosis_df = ("LIVE", pd.DataFrame())

    def _worst(s1, s2):
        order = {"LIVE": 0, "CACHED": 1, "SAMPLE": 2}
        return s1 if order.get(s1, 2) >= order.get(s2, 2) else s2

    combined_status = _worst(ecos_status, kosis_status)
    frames = []
    if not ecos_df.empty:
        frames.append(ecos_df)
    if not kosis_df.empty:
        frames.append(kosis_df)
    combined_df = pd.concat(frames) if frames else pd.DataFrame()
    return combined_status, combined_df


def _load_price_stage(
    *,
    market_id_arg: str,
    market_end_date_str: str,
    price_years: int,
    price_cache_token: str,
    price_artifact_key: tuple,
) -> tuple[str, pd.DataFrame, str]:
    normalized_market = str(market_id_arg or market_id).strip().upper() or market_id
    market_blocking_error = ""
    if normalized_market == "KR":
        from src.data_sources.krx_indices import (
            KRXInteractiveRangeLimitError,
            KRXMarketDataAccessDeniedError,
        )

        try:
            price_status, sector_prices = _cached_sector_prices(
                normalized_market,
                market_end_date_str,
                str(settings.get("benchmark_code", "1001")),
                price_years,
                price_cache_token,
                price_artifact_key,
            )
        except (KRXInteractiveRangeLimitError, KRXMarketDataAccessDeniedError) as exc:
            price_status = "BLOCKED"
            sector_prices = pd.DataFrame()
            market_blocking_error = str(exc)
        if not sector_prices.empty:
            sector_prices = _normalize_kr_named_frame(
                sector_prices,
                code_col="index_code",
                name_col="index_name",
            )
        return price_status, sector_prices, market_blocking_error

    price_status, sector_prices = _cached_sector_prices(
        normalized_market,
        market_end_date_str,
        str(settings.get("benchmark_code", "^GSPC")),
        price_years,
        price_cache_token,
        price_artifact_key,
    )
    return price_status, sector_prices, market_blocking_error


def _compute_macro_result(
    *,
    macro_df: pd.DataFrame,
    normalized_market: str,
    epsilon: float,
) -> pd.DataFrame:
    macro_result = pd.DataFrame()
    if not macro_df.empty:
        try:
            runtime_regime_settings = dict(settings)
            runtime_regime_settings["epsilon"] = float(epsilon)
            macro_result = build_regime_history_from_macro(
                macro_df=macro_df,
                macro_series_cfg=macro_series_cfg,
                settings=runtime_regime_settings,
                market_id=normalized_market,
                include_provisional=True,
                window_months=60,
            )
            macro_result = to_plotly_time_index(macro_result)
        except Exception as exc:
            logger.warning("compute_regime_history failed: %s", exc)

    if macro_result.empty:
        macro_result = pd.DataFrame(
            {
                "growth_dir": ["Flat"],
                "inflation_dir": ["Flat"],
                "regime": ["Indeterminate"],
                "confirmed_regime": ["Indeterminate"],
            },
            index=pd.DatetimeIndex([date.today()]),
        )
    return macro_result


def _compute_fx_change_pct(*, macro_df: pd.DataFrame) -> float:
    fx_change_pct = float("nan")
    if macro_df.empty:
        return fx_change_pct
    fx_series_alias = str(settings.get("fx_series_alias", "usdkrw"))
    fx_series = extract_macro_series(
        macro_df=macro_df,
        macro_series_cfg=macro_series_cfg,
        alias=fx_series_alias,
    )
    if len(fx_series) < 2:
        return fx_change_pct
    prev_fx = float(fx_series.iloc[-2])
    curr_fx = float(fx_series.iloc[-1])
    if pd.isna(prev_fx) or pd.isna(curr_fx) or prev_fx == 0:
        return fx_change_pct
    return float((curr_fx / prev_fx - 1) * 100)


def _resolve_price_reference_date(
    sector_prices: pd.DataFrame,
    *,
    benchmark_code: str,
) -> str:
    """Return the latest actual price date loaded for the benchmark or frame."""
    if not isinstance(sector_prices, pd.DataFrame) or sector_prices.empty:
        return ""

    frame = sector_prices.copy()
    try:
        frame.index = pd.DatetimeIndex(frame.index)
    except Exception:
        return ""

    if "index_code" in frame.columns:
        benchmark_rows = frame[frame["index_code"].astype(str) == str(benchmark_code)]
        if not benchmark_rows.empty:
            latest = pd.DatetimeIndex(benchmark_rows.index).max()
            if pd.notna(latest):
                return pd.Timestamp(latest).strftime("%Y-%m-%d")

    latest = pd.DatetimeIndex(frame.index).max()
    if pd.isna(latest):
        return ""
    return pd.Timestamp(latest).strftime("%Y-%m-%d")


def _build_signal_payload(
    *,
    normalized_market: str,
    market_end_date_str: str,
    price_status: str,
    sector_prices: pd.DataFrame,
    macro_df: pd.DataFrame,
    flow_status: str,
    flow_fresh: bool,
    flow_detail: dict[str, Any],
    sector_flow: pd.DataFrame,
    epsilon: float,
    rs_ma_period: int,
    ma_fast: int,
    ma_slow: int,
    momentum_method: str,
    momentum_skip_recent_days: int,
    momentum_lookback_6m_days: int,
    momentum_lookback_12m_days: int,
    momentum_rank_threshold_pct: float,
    momentum_abs_filter: str,
    price_years: int,
    flow_profile: str,
) -> tuple[list[Any], pd.DataFrame]:
    from src.signals.matrix import build_signal_table

    macro_result = _compute_macro_result(
        macro_df=macro_df,
        normalized_market=normalized_market,
        epsilon=epsilon,
    )

    bench_code = str(settings.get("benchmark_code", "1001"))
    bench_label = str(settings.get("benchmark_label", "")).strip()
    if not sector_prices.empty and "index_code" in sector_prices.columns:
        bench_mask = sector_prices["index_code"].astype(str) == bench_code
        if bench_mask.any():
            bench_series = sector_prices[bench_mask]["close"]
        elif normalized_market == "US" and bench_label and "index_name" in sector_prices.columns:
            label_mask = sector_prices["index_name"].astype(str) == bench_label
            bench_series = sector_prices[label_mask]["close"] if label_mask.any() else pd.Series(dtype=float)
        else:
            bench_series = pd.Series(dtype=float)
    else:
        bench_series = pd.Series(dtype=float)

    runtime_settings = dict(settings)
    runtime_settings.update(
        {
            "epsilon": float(epsilon),
            "rs_ma_period": int(rs_ma_period),
            "ma_fast": int(ma_fast),
            "ma_slow": int(ma_slow),
            "momentum_method": str(momentum_method),
            "momentum_skip_recent_days": int(momentum_skip_recent_days),
            "momentum_lookback_6m_days": int(momentum_lookback_6m_days),
            "momentum_lookback_12m_days": int(momentum_lookback_12m_days),
            "momentum_rank_threshold_pct": float(momentum_rank_threshold_pct),
            "momentum_abs_filter": str(momentum_abs_filter),
            "price_years": price_years,
        }
    )

    if price_status == "BLOCKED":
        return [], macro_result

    sector_universe_rows: list[dict[str, Any]] | None = None
    if normalized_market == "KR":
        sector_universe_rows = _canonicalize_kr_sector_universe_rows(
            read_active_index_dimension(market=normalized_market).to_dict("records"),
            benchmark_code=str(bench_code),
            include_benchmark=False,
        )
        try:
            theme_proxy_status, theme_proxy_prices, theme_proxy_rows = load_theme_proxy_signal_inputs(
                asof_date=market_end_date_str
            )
        except Exception as exc:
            logger.warning("KR theme proxy signal cache load failed: %s", exc)
            theme_proxy_status, theme_proxy_prices, theme_proxy_rows = "UNAVAILABLE", pd.DataFrame(), []
        if theme_proxy_status in {"CACHED", "PARTIAL"} and not theme_proxy_prices.empty and theme_proxy_rows:
            sector_prices = pd.concat([sector_prices, theme_proxy_prices]).sort_index()
            sector_universe_rows = [*(sector_universe_rows or []), *theme_proxy_rows]

    signals = build_signal_table(
        sector_prices=sector_prices,
        benchmark_prices=bench_series,
        macro_result=macro_result,
        sector_map=sector_map,
        settings=runtime_settings,
        market_id=normalized_market,
        sector_universe_rows=sector_universe_rows,
        fx_change_pct=_compute_fx_change_pct(macro_df=macro_df),
        sector_investor_flow=sector_flow,
        flow_profile=flow_profile,
        flow_enabled=_investor_flow_is_actionable(
            market_id_arg=normalized_market,
            flow_status=flow_status,
            flow_fresh=flow_fresh,
            flow_detail=flow_detail,
        ),
    )
    return signals, macro_result


@st.cache_data(ttl=CACHE_TTL)
def _cached_signals(
    market_id_arg: str,
    market_end_date_str: str,
    prices_key: tuple,
    macro_key: tuple,
    params_hash: str,
    macro_cache_token: str,
    price_cache_token: str,
    price_artifact_key: tuple,
    flow_artifact_key: tuple = (),
    epsilon: float = 0.0,
    rs_ma_period: int = 20,
    ma_fast: int = 20,
    ma_slow: int = 60,
    price_years: int = 3,
    flow_profile: str = "foreign_lead",
    momentum_method: str = "legacy_rs_ma_v0",
    momentum_skip_recent_days: int = 0,
    momentum_lookback_6m_days: int = 126,
    momentum_lookback_12m_days: int = 252,
    momentum_rank_threshold_pct: float = 0.60,
    momentum_abs_filter: str = "price_gt_200dma",
    theme_lens_artifact_key: tuple = (),
):
    """Compute signals. Keyed by parquet file metadata + params hash."""
    _ = theme_lens_artifact_key
    if not isinstance(flow_artifact_key, tuple):
        legacy_epsilon = float(flow_artifact_key)
        legacy_rs_ma_period = int(epsilon)
        legacy_ma_fast = int(rs_ma_period)
        legacy_ma_slow = int(ma_fast)
        legacy_price_years = int(ma_slow)
        flow_artifact_key = ()
        epsilon = legacy_epsilon
        rs_ma_period = legacy_rs_ma_period
        ma_fast = legacy_ma_fast
        ma_slow = legacy_ma_slow
        price_years = legacy_price_years

    normalized_market = str(market_id_arg or market_id).strip().upper() or market_id
    price_status, sector_prices, market_blocking_error = _load_price_stage(
        market_id_arg=normalized_market,
        market_end_date_str=market_end_date_str,
        price_years=price_years,
        price_cache_token=price_cache_token,
        price_artifact_key=price_artifact_key,
    )
    flow_status, flow_fresh, flow_detail, sector_flow = _cached_investor_flow(
        normalized_market,
        market_end_date_str,
        flow_artifact_key,
    )
    macro_status, macro_df = _cached_macro(normalized_market, macro_cache_token, market_end_date_str)
    signals, macro_result = _build_signal_payload(
        normalized_market=normalized_market,
        market_end_date_str=market_end_date_str,
        price_status=price_status,
        sector_prices=sector_prices,
        macro_df=macro_df,
        flow_status=flow_status,
        flow_fresh=flow_fresh,
        flow_detail=flow_detail,
        sector_flow=sector_flow,
        epsilon=epsilon,
        rs_ma_period=rs_ma_period,
        ma_fast=ma_fast,
        ma_slow=ma_slow,
        momentum_method=momentum_method,
        momentum_skip_recent_days=momentum_skip_recent_days,
        momentum_lookback_6m_days=momentum_lookback_6m_days,
        momentum_lookback_12m_days=momentum_lookback_12m_days,
        momentum_rank_threshold_pct=momentum_rank_threshold_pct,
        momentum_abs_filter=momentum_abs_filter,
        price_years=price_years,
        flow_profile=flow_profile,
    )
    return signals, macro_result, price_status, macro_status, market_blocking_error


def _compute_shared_flow_summary_map(
    flow_frame: pd.DataFrame | None,
    *,
    flow_profile: str,
    short_window: int,
    long_window: int,
) -> dict[str, object]:
    """Compute display-only investor-flow summary for downstream fallback surfaces."""
    if not isinstance(flow_frame, pd.DataFrame) or flow_frame.empty:
        return {}
    required_columns = {"sector_code", "sector_name", "investor_type", "net_flow_ratio"}
    if not required_columns.issubset(flow_frame.columns):
        return {}
    return dict(
        summarize_sector_investor_flow(
            flow_frame,
            flow_profile=flow_profile,
            short_window=short_window,
            long_window=long_window,
        )
    )


def load_dashboard_runtime_data(
    market_id_arg: str,
    market_end_date_str: str,
    prices_key: tuple,
    macro_key: tuple,
    params_hash: str,
    macro_cache_token: str,
    price_cache_token: str,
    price_artifact_key: tuple,
    flow_artifact_key: tuple = (),
    theme_lens_artifact_key: tuple = (),
    *,
    epsilon: float = 0.0,
    rs_ma_period: int = 20,
    ma_fast: int = 20,
    ma_slow: int = 60,
    price_years: int = 3,
    flow_profile: str = "foreign_lead",
    momentum_method: str = "legacy_rs_ma_v0",
    momentum_skip_recent_days: int = 0,
    momentum_lookback_6m_days: int = 126,
    momentum_lookback_12m_days: int = 252,
    momentum_rank_threshold_pct: float = 0.60,
    momentum_abs_filter: str = "price_gt_200dma",
    progress_callback: DashboardProgressCallback | None = None,
) -> dict[str, Any]:
    normalized_market = str(market_id_arg or market_id).strip().upper() or market_id
    task = "대시보드 데이터 로드"
    _emit_progress(
        progress_callback,
        task=task,
        phase="준비 중",
        pct=5,
        detail=f"{normalized_market} · {market_end_date_str}",
    )
    try:
        resolved_theme_lens_key = theme_lens_artifact_key or get_theme_lens_artifact_key()
        price_status, sector_prices, market_blocking_error = _load_price_stage(
            market_id_arg=normalized_market,
            market_end_date_str=market_end_date_str,
            price_years=price_years,
            price_cache_token=price_cache_token,
            price_artifact_key=price_artifact_key,
        )
        _emit_progress(
            progress_callback,
            task=task,
            phase="시장 데이터 로드 완료",
            pct=25,
            detail=f"상태 {price_status}",
        )
        flow_status, flow_fresh, flow_detail, sector_flow = _cached_investor_flow(
            normalized_market,
            market_end_date_str,
            flow_artifact_key,
        )
        _emit_progress(
            progress_callback,
            task=task,
            phase="수급 데이터 로드 완료",
            pct=45,
            detail=(
                f"상태 {flow_status}"
                f"{'' if (flow_fresh and not bool(flow_detail.get('warehouse_write_skipped'))) else ' · reference-only'}"
            ),
        )
        macro_status, macro_df = _cached_macro(
            normalized_market,
            macro_cache_token,
            market_end_date_str,
        )
        _emit_progress(
            progress_callback,
            task=task,
            phase="매크로 데이터 로드 완료",
            pct=65,
            detail=f"상태 {macro_status}",
        )
        signals, macro_result, price_status, macro_status, market_blocking_error = _cached_signals(
            normalized_market,
            market_end_date_str,
            prices_key,
            macro_key,
            params_hash,
            macro_cache_token,
            price_cache_token,
            price_artifact_key,
            flow_artifact_key,
            epsilon=epsilon,
            rs_ma_period=rs_ma_period,
            ma_fast=ma_fast,
            ma_slow=ma_slow,
            momentum_method=momentum_method,
            momentum_skip_recent_days=momentum_skip_recent_days,
            momentum_lookback_6m_days=momentum_lookback_6m_days,
            momentum_lookback_12m_days=momentum_lookback_12m_days,
            momentum_rank_threshold_pct=momentum_rank_threshold_pct,
            momentum_abs_filter=momentum_abs_filter,
            price_years=price_years,
            flow_profile=flow_profile,
            theme_lens_artifact_key=resolved_theme_lens_key,
        )
        flow_display_fresh = flow_fresh and not bool(flow_detail.get("warehouse_write_skipped"))
        theme_lens_status, theme_lens_rows = _cached_theme_lens(
            normalized_market,
            market_end_date_str,
            resolved_theme_lens_key,
        )
        shared_flow_summary_map = (
            _compute_shared_flow_summary_map(
                sector_flow,
                flow_profile=flow_profile,
                short_window=int(settings.get("investor_flow_short_window", 20)),
                long_window=int(settings.get("investor_flow_long_window", 60)),
            )
            if normalized_market == "KR"
            else {}
        )
        _emit_progress(
            progress_callback,
            task=task,
            phase="신호 계산 완료",
            pct=85,
            detail=f"{len(signals)} sectors",
        )
        payload = {
            "signals": signals,
            "macro_result": macro_result,
            "macro_df": macro_df,
            "market_data_reference_date": _resolve_price_reference_date(
                sector_prices,
                benchmark_code=str(settings.get("benchmark_code", "1001" if normalized_market == "KR" else "^GSPC")),
            ),
            "price_status": price_status,
            "macro_status": macro_status,
            "market_blocking_error": market_blocking_error,
            "investor_flow_status": flow_status,
            "investor_flow_fresh": flow_display_fresh,
            "investor_flow_detail": dict(flow_detail),
            "investor_flow_frame": sector_flow,
            "shared_flow_summary_map": shared_flow_summary_map,
            "theme_lens_status": theme_lens_status,
            "theme_lens_rows": list(theme_lens_rows),
            "theme_lens_artifact_key": resolved_theme_lens_key,
        }
        _emit_progress(
            progress_callback,
            task=task,
            phase="표시 데이터 준비 완료",
            pct=100,
            detail="대시보드 표시에 필요한 데이터가 준비되었습니다.",
            status="complete",
        )
        return payload
    except Exception as exc:
        _emit_progress(
            progress_callback,
            task=task,
            phase="로드 실패",
            pct=100,
            detail=str(exc),
            status="error",
        )
        raise


get_all_sector_codes = _all_sector_codes
build_market_refresh_notice = _build_market_refresh_notice
build_macro_refresh_notice = _build_macro_refresh_notice
build_investor_flow_refresh_notice = _build_investor_flow_refresh_notice
compute_shared_flow_summary_map = _compute_shared_flow_summary_map
cached_analysis_sector_prices = _cached_analysis_sector_prices
cached_api_preflight = _cached_api_preflight
cached_investor_flow = _cached_investor_flow
cached_macro = _cached_macro
cached_sector_prices = _cached_sector_prices
cached_signals = _cached_signals
is_mobile_client = _is_mobile_client
get_krx_provider_configured = _krx_provider_configured
get_krx_provider_effective = _krx_provider_effective
load_api_key = _load_api_key
load_analysis_sector_prices_from_cache = _load_analysis_sector_prices_from_cache
get_macro_artifact_key = _macro_artifact_key
get_macro_cache_token = _macro_cache_token
maybe_schedule_startup_krx_warm = _maybe_schedule_startup_krx_warm
get_market_range_strings = _market_range_strings
get_investor_flow_range_strings = _investor_flow_range_strings
get_openapi_cache_fallback_note = _openapi_cache_fallback_note
get_parquet_key = _parquet_key
get_investor_flow_artifact_key = _investor_flow_artifact_key
get_price_artifact_key = _price_artifact_key
get_price_cache_token = _price_cache_token
probe_investor_flow_status = _probe_investor_flow_status
probe_macro_status = _probe_macro_status
probe_market_status = _probe_market_status
render_dashboard_status_banner = _render_dashboard_status_banner
resolve_market_end_date = _resolve_market_end_date
get_secrets_mtime_ns = _secrets_mtime_ns
show_notice_toast = _show_notice_toast


__all__ = [
    "build_macro_refresh_notice",
    "build_market_refresh_notice",
    "build_investor_flow_refresh_notice",
    "cached_analysis_sector_prices",
    "cached_api_preflight",
    "cached_investor_flow",
    "cached_macro",
    "cached_sector_prices",
    "cached_signals",
    "clear_investor_flow_transient_override",
    "clear_market_price_transient_override",
    "compute_shared_flow_summary_map",
    "configure_dashboard_env",
    "get_all_sector_codes",
    "get_investor_flow_artifact_key",
    "get_investor_flow_range_strings",
    "get_krx_provider_configured",
    "get_krx_provider_effective",
    "get_macro_artifact_key",
    "get_macro_cache_token",
    "get_market_index_universe_codes",
    "get_market_range_strings",
    "get_openapi_cache_fallback_note",
    "get_parquet_key",
    "get_price_artifact_key",
    "get_price_cache_token",
    "get_secrets_mtime_ns",
    "is_mobile_client",
    "load_dashboard_runtime_data",
    "load_analysis_sector_prices_from_cache",
    "load_api_key",
    "maybe_schedule_startup_krx_warm",
    "probe_investor_flow_status",
    "probe_macro_status",
    "probe_market_status",
    "render_dashboard_status_banner",
    "resolve_market_end_date",
    "set_investor_flow_transient_override",
    "set_market_price_transient_override",
    "show_notice_toast",
]
