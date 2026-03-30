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
    epsilon: float,
    rs_ma_period: int,
    ma_fast: int,
    ma_slow: int,
    price_years: int,
):
    """Compute signals. Keyed by parquet file metadata + params hash."""
    from src.macro.regime import compute_regime_history
    from src.signals.matrix import build_signal_table

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
    else:
        price_status, sector_prices = _cached_sector_prices(
            normalized_market,
            market_end_date_str,
            str(settings.get("benchmark_code", "SPY")),
            price_years,
            price_cache_token,
            price_artifact_key,
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
        growth_series = extract_macro_series(
            macro_df=macro_df,
            macro_series_cfg=macro_series_cfg,
            alias="leading_index",
        )
        # CPI MoM (전월비) as primary inflation signal; fall back to YoY
        inflation_series = extract_macro_series(
            macro_df=macro_df,
            macro_series_cfg=macro_series_cfg,
            alias="cpi_mom",
        )
        if inflation_series.empty:
            inflation_series = extract_macro_series(
                macro_df=macro_df,
                macro_series_cfg=macro_series_cfg,
                alias="cpi_yoy",
            )

        long_alias = str(settings.get("yield_curve_long", "bond_3y"))
        short_alias = str(settings.get("yield_curve_short", "base_rate"))
        _bond_s = extract_macro_series(macro_df, macro_series_cfg, long_alias)
        _base_s = extract_macro_series(macro_df, macro_series_cfg, short_alias)
        yield_curve_spread = None
        if not _bond_s.empty and not _base_s.empty:
            try:
                _spread = (_bond_s - _base_s).dropna()
                if not _spread.empty:
                    yield_curve_spread = _spread
            except Exception as _yc_exc:
                logger.debug("Yield curve spread computation skipped: %s", _yc_exc)

        if not growth_series.empty and not inflation_series.empty:
            aligned = pd.concat(
                {"growth": growth_series, "inflation": inflation_series},
                axis=1,
                join="inner",
            ).dropna()
            if not aligned.empty:
                try:
                    macro_result = compute_regime_history(
                        aligned["growth"],
                        aligned["inflation"],
                        epsilon=epsilon,
                        use_adaptive_epsilon=bool(settings.get("use_adaptive_epsilon", True)),
                        epsilon_factor=float(settings.get("epsilon_factor", 0.5)),
                        confirmation_periods=int(settings.get("confirmation_periods", 2)),
                        yield_curve_spread=yield_curve_spread,
                        yield_curve_threshold=float(settings.get("yield_curve_spread_threshold", 0.0)),
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
        )

    return signals, macro_result, price_status, macro_status, market_blocking_error
