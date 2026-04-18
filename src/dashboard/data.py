"""Dashboard data/cache helpers extracted from app.py."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.data_sources.common import shift_month_token
from src.macro.series_utils import (
    build_regime_history_from_macro,
    build_regime_inflation_series,
    build_enabled_ecos_config,
    build_enabled_fred_config,
    build_enabled_kosis_config,
    extract_macro_series,
    to_plotly_time_index,
)
from src.data_sources.warehouse import read_market_prices
from src.ui.data_status import (
    resolve_dashboard_status_banner,
    resolve_price_cache_banner_case,
)
from src.ui.components import render_status_strip

logger = logging.getLogger(__name__)

settings: dict[str, Any] = {}
sector_map: dict[str, Any] = {}
macro_series_cfg: dict[str, Any] = {}
market_id = "KR"
market_profile: Any | None = None
CACHE_TTL = 21600
CURATED_SECTOR_PRICES_PATH = Path("data/curated/sector_prices.parquet")
INVESTOR_FLOW_TRANSIENT_OVERRIDE_KEY = "_investor_flow_transient_override"


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
    return all_codes


def _resolve_market_end_date(benchmark_code: str) -> date:
    """Resolve the market end date once per app run."""
    from src.transforms.calendar import get_last_business_day

    return get_last_business_day(
        provider=_krx_provider_effective(),
        benchmark_code=benchmark_code,
    )


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

    if status == "LIVE" and delta_codes:
        return (
            "success",
            f"Market data refresh completed via {provider_label} ({len(delta_codes)} codes updated).",
        )
    if coverage_complete and not delta_codes:
        return ("info", "Market data already current; the latest local cache is in use.")
    if failed_days or failed_codes:
        return (
            "warning",
            "Market data refresh fell back to cache after an incomplete live refresh. "
            "Retry later or continue with cache.",
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
            return ("warning", f"투자자수급 갱신 완료 (부분 커버리지{detail}). complete cursor가 없어 이번 partial preview를 표시합니다.")
        return ("warning", f"투자자수급 갱신 완료 (부분 커버리지{detail}). complete cursor 기준 기존 데이터를 유지합니다.")
    if status == "CACHED":
        if coverage_complete:
            return ("info", "투자자수급 데이터가 이미 최신 complete cursor까지 반영되어 있습니다.")
        if auth_fail_count:
            return (
                "error",
                "투자자수급 갱신 실패: KRX 데이터마켓 로그인 세션이 필요합니다. "
                "환경변수 또는 Streamlit secrets에 KRX_ID / KRX_PW를 설정한 뒤 다시 시도하세요.",
            )
        if bootstrap_partial_preview:
            return ("warning", "실시간 투자자수급 갱신은 완료되지 않았지만, bootstrap partial preview 데이터를 표시합니다.")
        snap_detail = ""
        if failed_codes:
            snap_detail = f" — {len(failed_codes)}개 항목 실패로 캐시 사용"
        return ("warning", f"투자자수급 갱신 실패 → 캐시 데이터 사용{snap_detail}.")
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
    """Render blocking errors prominently and lower-priority states compactly."""
    if not banner:
        return

    level = str(banner.get("level", "info")).strip().lower()
    if level == "error":
        _legacy_render_dashboard_status_banner(banner)
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

    all_codes = _all_sector_codes(benchmark_code)
    start_str, end_str = _market_range_strings(end_date_str, price_years)

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
    all_codes = _all_sector_codes(benchmark_code)
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
    return status, _investor_flow_is_fresh(status_detail, market_end_date_str=end_str), status_detail, frame


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
):
    """Compute signals. Keyed by parquet file metadata + params hash."""
    from src.signals.matrix import build_signal_table

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
    market_blocking_error = ""
    flow_status = "SAMPLE"
    flow_fresh = False
    flow_detail: dict[str, Any] = {}
    sector_flow = pd.DataFrame()
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
        flow_status, flow_fresh, flow_detail, sector_flow = _cached_investor_flow(
            normalized_market,
            market_end_date_str,
            flow_artifact_key,
        )
    else:
        price_status, sector_prices = _cached_sector_prices(
            normalized_market,
            market_end_date_str,
            str(settings.get("benchmark_code", "SPY")),
            price_years,
            price_cache_token,
            price_artifact_key,
        )
        flow_status, flow_fresh, flow_detail, sector_flow = _cached_investor_flow(
            normalized_market,
            market_end_date_str,
            flow_artifact_key,
        )

    macro_status, macro_df = _cached_macro(normalized_market, macro_cache_token, market_end_date_str)

    # Benchmark prices from sector_prices (benchmark_code row)
    bench_code = str(settings.get("benchmark_code", "1001"))
    if not sector_prices.empty and "index_code" in sector_prices.columns:
        bench_mask = sector_prices["index_code"].astype(str) == bench_code
        bench_series = sector_prices[bench_mask]["close"] if bench_mask.any() else pd.Series(dtype=float)
    else:
        bench_series = pd.Series(dtype=float)

    # Macro regime history
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
        # Fallback: create minimal mock result for display
        macro_result = pd.DataFrame(
            {
                "growth_dir": ["Flat"],
                "inflation_dir": ["Flat"],
                "regime": ["Indeterminate"],
                "confirmed_regime": ["Indeterminate"],
            },
            index=pd.DatetimeIndex([date.today()]),
        )

    runtime_settings = dict(settings)
    runtime_settings.update(
        {
            "epsilon": float(epsilon),
            "rs_ma_period": int(rs_ma_period),
            "ma_fast": int(ma_fast),
            "ma_slow": int(ma_slow),
            "price_years": price_years,
        }
    )
    fx_change_pct = float("nan")
    if not macro_df.empty:
        fx_series_alias = str(settings.get("fx_series_alias", "usdkrw"))
        fx_series = extract_macro_series(
            macro_df=macro_df,
            macro_series_cfg=macro_series_cfg,
            alias=fx_series_alias,
        )
        if len(fx_series) >= 2:
            prev_fx = float(fx_series.iloc[-2])
            curr_fx = float(fx_series.iloc[-1])
            if not (pd.isna(prev_fx) or pd.isna(curr_fx)) and prev_fx != 0:
                fx_change_pct = float((curr_fx / prev_fx - 1) * 100)

    if price_status == "BLOCKED":
        signals = []
    else:
        signals = build_signal_table(
            sector_prices=sector_prices,
            benchmark_prices=bench_series,
            macro_result=macro_result,
            sector_map=sector_map,
            settings=runtime_settings,
            fx_change_pct=fx_change_pct,
            sector_investor_flow=sector_flow,
            flow_profile=flow_profile,
            flow_enabled=_investor_flow_is_actionable(
                market_id_arg=normalized_market,
                flow_status=flow_status,
                flow_fresh=flow_fresh,
                flow_detail=flow_detail,
            ),
        )

    return signals, macro_result, price_status, macro_status, market_blocking_error


get_all_sector_codes = _all_sector_codes
build_market_refresh_notice = _build_market_refresh_notice
build_macro_refresh_notice = _build_macro_refresh_notice
build_investor_flow_refresh_notice = _build_investor_flow_refresh_notice
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
    "configure_dashboard_env",
    "get_all_sector_codes",
    "get_investor_flow_artifact_key",
    "get_investor_flow_range_strings",
    "get_krx_provider_configured",
    "get_krx_provider_effective",
    "get_macro_artifact_key",
    "get_macro_cache_token",
    "get_market_range_strings",
    "get_openapi_cache_fallback_note",
    "get_parquet_key",
    "get_price_artifact_key",
    "get_price_cache_token",
    "get_secrets_mtime_ns",
    "is_mobile_client",
    "load_analysis_sector_prices_from_cache",
    "load_api_key",
    "maybe_schedule_startup_krx_warm",
    "probe_investor_flow_status",
    "probe_macro_status",
    "probe_market_status",
    "render_dashboard_status_banner",
    "resolve_market_end_date",
    "set_investor_flow_transient_override",
    "show_notice_toast",
]
