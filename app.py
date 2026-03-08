"""
Korea Sector Rotation Dashboard
Streamlit SPA ??app.py

Architecture:
- Three named @st.cache_data functions for sector prices, macro, and signals (R8).
- Each button clears only its own cache (R8 ??no cross-cache pollution).
- SAMPLE mode shows full-width st.error banner + disables recompute (R9).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from config.theme import THEME_SESSION_KEY, get_theme_mode, set_theme_mode

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Page config (must be first Streamlit call)
st.set_page_config(
    page_title="Korea Sector Rotation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Local imports
from src.ui.styles import inject_css
from src.ui.components import (
    ALL_ACTION_OPTION,
    HEATMAP_PALETTE_OPTIONS,
    build_sector_detail_figure,
    build_sector_strength_heatmap,
    format_heatmap_palette_label,
    format_cycle_phase_label,
    infer_range_preset,
    normalize_range_preset,
    render_action_summary,
    render_analysis_toolbar,
    render_cycle_timeline_panel,
    render_decision_hero,
    render_page_header,
    render_panel_header,
    render_rs_momentum_bar,
    render_rs_scatter,
    render_returns_heatmap,
    render_sector_detail_panel,
    render_signal_table,
    render_status_card_row,
    render_status_strip,
    render_top_bar_filters,
    render_top_picks_table,
    resolve_range_from_preset,
)
from src.ui.data_status import (
    get_button_states,
    is_sample_mode,
    resolve_dashboard_status_banner,
    resolve_price_cache_banner_case,
)
from src.macro.series_utils import (
    build_enabled_ecos_config,
    build_enabled_kosis_config,
    extract_macro_series,
    to_plotly_time_index,
)
from src.data_sources.warehouse import read_market_prices

# Config loading


@st.cache_data(ttl=3600)
def _load_config() -> tuple[dict, dict, dict]:
    """Load YAML config files. Cached for 1 hour."""
    with open("config/settings.yml", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    with open("config/sector_map.yml", encoding="utf-8") as f:
        sector_map = yaml.safe_load(f)
    with open("config/macro_series.yml", encoding="utf-8") as f:
        macro_series = yaml.safe_load(f)
    return settings, sector_map, macro_series


settings, sector_map, macro_series_cfg = _load_config()
CACHE_TTL = int(settings.get("cache_ttl", 21600))
CURATED_SECTOR_PRICES_PATH = Path("data/curated/sector_prices.parquet")

# Cache key helper (R8)


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

    return build_macro_cache_token(
        macro_series_cfg=macro_series_cfg,
        ecos_key=_load_api_key("ECOS_API_KEY"),
        kosis_key=_load_api_key("KOSIS_API_KEY"),
        secrets_mtime_ns=_secrets_mtime_ns(),
    )


def _krx_provider_configured() -> str:
    """Return configured KRX provider value (AUTO/OPENAPI/PYKRX)."""
    from src.data_sources.krx_openapi import get_krx_provider

    return get_krx_provider(_load_api_key("KRX_PROVIDER"))


def _krx_provider_effective() -> str:
    """Return runtime-effective provider after AUTO resolution."""
    configured = _krx_provider_configured()
    if configured == "AUTO":
        return "OPENAPI" if _load_api_key("KRX_OPENAPI_KEY") else "PYKRX"
    return configured


def _price_cache_token() -> str:
    """Build cache token for market price loader (KRX provider + key fingerprint)."""
    from src.data_sources.cache_keys import build_price_cache_token

    return build_price_cache_token(
        krx_provider=_krx_provider_configured(),
        krx_openapi_key=_load_api_key("KRX_OPENAPI_KEY"),
        secrets_mtime_ns=_secrets_mtime_ns(),
    )


def _price_artifact_key() -> tuple:
    """Return cache-busting key for raw/warm price artifacts."""
    from src.data_sources.krx_indices import get_price_artifact_key

    return get_price_artifact_key()


def _macro_artifact_key() -> tuple:
    """Return cache-busting key for macro warehouse artifacts."""
    from src.data_sources.macro_sync import get_macro_artifact_key

    return get_macro_artifact_key()


def _probe_market_status() -> str:
    """Return current market-data availability from the warehouse."""
    from src.data_sources.krx_indices import probe_market_status

    return probe_market_status()


def _probe_macro_status() -> str:
    """Return current macro-data availability from the warehouse."""
    from src.data_sources.macro_sync import probe_macro_status

    return probe_macro_status()


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


def _build_sector_name_map(
    *,
    signals: list,
    sector_prices: pd.DataFrame,
    benchmark_code: str,
) -> dict[str, str]:
    """Build a code -> display name map from the current universe."""
    names: dict[str, str] = {str(benchmark_code): "KOSPI"}
    for signal in signals:
        names[str(signal.index_code)] = str(signal.sector_name)

    if not sector_prices.empty and "index_code" in sector_prices.columns and "index_name" in sector_prices.columns:
        latest_names = (
            sector_prices.reset_index()
            .sort_values(sector_prices.index.name or "trade_date")
            .drop_duplicates("index_code", keep="last")
        )
        for _, row in latest_names.iterrows():
            code = str(row["index_code"])
            if code not in names or not names[code].strip():
                names[code] = str(row["index_name"])
    return names


def _build_prices_wide(
    *,
    sector_prices: pd.DataFrame,
    sector_name_map: dict[str, str],
) -> pd.DataFrame:
    """Pivot long sector prices into a wide trade-date index frame."""
    if sector_prices.empty:
        return pd.DataFrame()

    prices_reset = sector_prices.reset_index().copy()
    date_col = sector_prices.index.name or "trade_date"
    prices_reset[date_col] = pd.to_datetime(prices_reset[date_col])
    prices_reset["series_name"] = prices_reset["index_code"].astype(str).map(
        lambda code: sector_name_map.get(code, code)
    )
    prices_wide = (
        prices_reset.pivot_table(
            index=date_col,
            columns="series_name",
            values="close",
            aggfunc="last",
        )
        .sort_index()
        .ffill()
    )
    prices_wide.index = pd.to_datetime(prices_wide.index)
    return prices_wide


def _build_monthly_sector_returns(
    *,
    prices_wide: pd.DataFrame,
    sector_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build monthly closes/returns from the full available analysis history."""
    monthly_close_full = prices_wide[sector_columns].resample("ME").last() if sector_columns else pd.DataFrame()
    monthly_returns_full = monthly_close_full.pct_change() * 100 if not monthly_close_full.empty else pd.DataFrame()
    return monthly_close_full, monthly_returns_full


def _build_monthly_return_views(
    *,
    prices_wide: pd.DataFrame,
    sector_columns: list[str],
    benchmark_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build absolute and excess monthly return views for the analysis canvas."""
    monthly_close_full, monthly_returns_full = _build_monthly_sector_returns(
        prices_wide=prices_wide,
        sector_columns=sector_columns,
    )
    benchmark_monthly_return = pd.Series(dtype="float64")
    if benchmark_label in prices_wide.columns:
        benchmark_monthly_close = prices_wide[[benchmark_label]].resample("ME").last()
        benchmark_monthly_return = benchmark_monthly_close[benchmark_label].pct_change() * 100
    monthly_excess_returns_full = (
        monthly_returns_full.sub(benchmark_monthly_return, axis=0)
        if not monthly_returns_full.empty and not benchmark_monthly_return.empty
        else pd.DataFrame(index=monthly_returns_full.index, columns=monthly_returns_full.columns)
    )
    return monthly_close_full, monthly_returns_full, benchmark_monthly_return, monthly_excess_returns_full


def _filter_monthly_frame_for_analysis(
    *,
    monthly_frame: pd.DataFrame,
    start_date: date,
    end_date: date,
    selected_cycle_phase: str,
    phase_by_month: pd.Series,
) -> pd.DataFrame:
    """Filter a monthly frame to the visible analysis range and cycle phase."""
    if monthly_frame.empty:
        return pd.DataFrame()

    start_ts = pd.Timestamp(start_date).to_period("M").to_timestamp("M")
    end_ts = pd.Timestamp(end_date).normalize()
    filtered = monthly_frame.loc[
        (monthly_frame.index >= start_ts)
        & (monthly_frame.index <= end_ts)
    ]
    if filtered.empty or selected_cycle_phase == "ALL" or phase_by_month.empty:
        return filtered

    valid_months = set(
        phase_by_month[phase_by_month.astype(str) == str(selected_cycle_phase)]
        .index.to_period("M")
        .astype(str)
    )
    if not valid_months:
        return filtered.iloc[0:0]
    return filtered.loc[
        filtered.index.to_period("M").astype(str).isin(valid_months)
    ]


def _build_heatmap_display(monthly_frame: pd.DataFrame) -> pd.DataFrame:
    """Transpose a monthly frame into sector x month display shape."""
    if monthly_frame.empty:
        return pd.DataFrame()
    display = monthly_frame.copy()
    display.index = display.index.to_period("M").astype(str)
    return display.T


def _extract_heatmap_selection(heatmap_event) -> tuple[str, str] | None:
    """Return the selected (month, sector) pair from a Plotly heatmap selection."""
    if heatmap_event is None:
        return None
    selection_state = getattr(heatmap_event, "selection", None)
    if selection_state is None and isinstance(heatmap_event, dict):
        selection_state = heatmap_event.get("selection", {})
    if not selection_state:
        return None

    selected_points = list(getattr(selection_state, "points", []) or selection_state.get("points", []))
    if not selected_points:
        return None
    point = selected_points[-1]
    customdata = point.get("customdata") if isinstance(point, dict) else None
    if isinstance(customdata, (list, tuple)) and len(customdata) >= 2:
        return str(customdata[0]), str(customdata[1])
    return None


def _build_cycle_segments(
    *,
    macro_result: pd.DataFrame,
    monthly_close: pd.DataFrame,
) -> tuple[list[dict[str, object]], pd.Series]:
    """Split macro regime history into early/late contiguous cycle segments."""
    if macro_result.empty:
        return [], pd.Series(dtype=object)

    regime_col = "confirmed_regime" if "confirmed_regime" in macro_result.columns else "regime"
    regime_series = macro_result[regime_col].copy()
    regime_series.index = pd.to_datetime(regime_series.index).to_period("M").to_timestamp(how="start")
    regime_series = regime_series.sort_index()
    if regime_series.empty:
        return [], pd.Series(dtype=object)

    phase_by_month = pd.Series(index=regime_series.index, dtype=object)
    segments: list[dict[str, object]] = []
    color_key = {
        "Recovery": "RECOVERY",
        "Expansion": "EXPANSION",
        "Slowdown": "SLOWDOWN",
        "Contraction": "CONTRACTION",
    }

    for _, run in regime_series.groupby((regime_series != regime_series.shift()).cumsum()):
        regime_value = run.iloc[0]
        if pd.isna(regime_value):
            continue
        regime = str(regime_value)
        run_months = list(run.index)
        if not run_months:
            continue

        split_index = max(1, int((len(run_months) + 1) / 2))
        partitions = [
            ("EARLY", run_months[:split_index]),
            ("LATE", run_months[split_index:]),
        ]
        for stage, months in partitions:
            if not months:
                continue

            phase_key = f"{color_key.get(regime, regime.upper())}_{stage}" if regime in color_key else regime.upper()
            phase_label = format_cycle_phase_label(phase_key) if phase_key in {
                "RECOVERY_EARLY",
                "RECOVERY_LATE",
                "EXPANSION_EARLY",
                "EXPANSION_LATE",
                "SLOWDOWN_EARLY",
                "SLOWDOWN_LATE",
                "CONTRACTION_EARLY",
                "CONTRACTION_LATE",
            } else regime
            start_month = months[0]
            end_month = months[-1]
            phase_by_month.loc[months] = phase_key
            start_date = pd.Timestamp(start_month).to_period("M").to_timestamp(how="start")
            end_date = pd.Timestamp(end_month).to_period("M").to_timestamp(how="end")

            top_summary = "No sector summary available."
            if not monthly_close.empty:
                segment_slice = monthly_close.loc[
                    (monthly_close.index >= start_date.normalize())
                    & (monthly_close.index <= end_date.normalize() + pd.offsets.MonthEnd(0))
                ]
                if len(segment_slice) >= 2:
                    segment_return = segment_slice.iloc[-1] / segment_slice.iloc[0] - 1
                    segment_return = segment_return.dropna().sort_values(ascending=False)
                    if not segment_return.empty:
                        top_sector = str(segment_return.index[0])
                        top_summary = f"Top sector: {top_sector} ({segment_return.iloc[0] * 100:+.1f}%)"

            segments.append(
                {
                    "phase_key": phase_key,
                    "label": phase_label,
                    "regime": regime,
                    "start": start_date,
                    "end": end_date,
                    "summary": top_summary,
                }
            )

    if segments:
        current_phase_key = str(phase_by_month.dropna().iloc[-1]) if not phase_by_month.dropna().empty else ""
        for segment in segments:
            segment["is_current"] = str(segment.get("phase_key", "")) == current_phase_key

    return segments, phase_by_month


def _filter_prices_for_phase(
    *,
    prices_wide: pd.DataFrame,
    phase_by_month: pd.Series,
    selected_cycle_phase: str,
) -> pd.DataFrame:
    """Filter daily prices to the months belonging to the selected cycle phase."""
    if prices_wide.empty or not selected_cycle_phase or selected_cycle_phase == "ALL":
        return prices_wide
    if phase_by_month.empty:
        return prices_wide.iloc[0:0]

    valid_months = set(
        phase_by_month[phase_by_month.astype(str) == str(selected_cycle_phase)]
        .index.to_period("M")
        .astype(str)
    )
    if not valid_months:
        return prices_wide.iloc[0:0]
    mask = prices_wide.index.to_period("M").astype(str).isin(valid_months)
    return prices_wide.loc[mask]


def _build_market_refresh_notice(summary: dict[str, object]) -> tuple[str, str]:
    """Map manual refresh summary into a user-facing flash message."""
    status = str(summary.get("status", "")).strip().upper()
    coverage_complete = bool(summary.get("coverage_complete"))
    delta_codes = list(summary.get("delta_codes", []))
    failed_days = list(summary.get("failed_days", []))
    failed_codes = dict(summary.get("failed_codes") or {})

    if status == "LIVE" and delta_codes:
        return (
            "success",
            f"Market data refresh completed via OpenAPI ({len(delta_codes)} codes updated).",
        )
    if coverage_complete and not delta_codes:
        return ("info", "Market data already current; latest KRX raw cache is in use.")
    if failed_days or failed_codes:
        return (
            "warning",
            "Market data refresh fell back to cache after an incomplete OpenAPI warm. "
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
    """Render the single top-of-page system status strip."""
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
def _cached_api_preflight(timeout_sec: int = 3) -> dict:
    """Cached API endpoint reachability check (10 min TTL)."""
    from src.data_sources.preflight import run_api_preflight

    return run_api_preflight(timeout_sec=timeout_sec)


# Named cache functions (R8)


@st.cache_data(ttl=CACHE_TTL)
def _cached_sector_prices(
    end_date_str: str,
    benchmark_code: str,
    price_years: int,
    price_cache_token: str,
    price_artifact_key: tuple,
):
    """Fetch or load sector prices. Includes KRX provider/key cache token."""
    _ = (price_cache_token, price_artifact_key)
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
    end_date_str: str,
    benchmark_code: str,
) -> pd.DataFrame:
    """Load the widest cached analysis history without triggering live refreshes."""
    all_codes = _all_sector_codes(benchmark_code)
    cached = _filter_cached_sector_prices(
        read_market_prices(all_codes, "19000101", end_date_str),
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
    end_date_str: str,
    benchmark_code: str,
    price_years: int,
    price_artifact_key: tuple,
) -> pd.DataFrame:
    """Load analysis-canvas prices from cache/warehouse only."""
    _ = (price_years, price_artifact_key)
    return _load_analysis_sector_prices_from_cache(end_date_str, benchmark_code)


@st.cache_data(ttl=CACHE_TTL)
def _cached_macro(macro_cache_token: str):
    """Fetch or load macro data. Keyed by config + API key fingerprint token."""
    from src.data_sources.ecos import load_ecos_macro
    from src.data_sources.kosis import load_kosis_macro

    # Date range: last 5 years monthly
    end_ym = date.today().strftime("%Y%m")
    start_ym = (date.today() - timedelta(days=365 * 5)).strftime("%Y%m")

    # ECOS
    ecos_cfg = build_enabled_ecos_config(macro_series_cfg.get("ecos", {}))
    if ecos_cfg:
        ecos_status, ecos_df = load_ecos_macro(start_ym, end_ym, series_config=ecos_cfg)
    else:
        ecos_status, ecos_df = ("LIVE", pd.DataFrame())

    # KOSIS
    kosis_cfg = build_enabled_kosis_config(macro_series_cfg.get("kosis", {}))
    if kosis_cfg:
        kosis_status, kosis_df = load_kosis_macro(start_ym, end_ym, series_config=kosis_cfg)
    else:
        kosis_status, kosis_df = ("LIVE", pd.DataFrame())

    # Combine: worst status wins
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
    market_end_date_str: str,
    prices_key: tuple,
    macro_key: tuple,
    params_hash: str,
    macro_cache_token: str,
    price_cache_token: str,
    price_artifact_key: tuple,
):
    """Compute signals. Keyed by parquet file metadata + params hash."""
    from src.macro.regime import compute_regime_history
    from src.data_sources.krx_indices import (
        KRXInteractiveRangeLimitError,
        KRXMarketDataAccessDeniedError,
    )
    from src.signals.matrix import build_signal_table

    price_years = int(st.session_state.get("price_years", settings.get("price_years", 3)))
    market_blocking_error = ""
    try:
        price_status, sector_prices = _cached_sector_prices(
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
    macro_status, macro_df = _cached_macro(macro_cache_token)

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

        # Yield curve spread: KTB 3Y - base rate (monthly)
        _bond_s = extract_macro_series(macro_df, macro_series_cfg, "bond_3y")
        _base_s = extract_macro_series(macro_df, macro_series_cfg, "base_rate")
        yield_curve_spread = None
        if not _bond_s.empty and not _base_s.empty:
            try:
                _bond_monthly = _bond_s.resample("M").last()
                _spread = (_bond_monthly - _base_s).dropna()
                if not _spread.empty:
                    yield_curve_spread = _spread
            except Exception as _yc_exc:
                logger.debug("Yield curve spread computation skipped: %s", _yc_exc)

        epsilon = float(st.session_state.get("epsilon", settings.get("epsilon", 0)))

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
            "epsilon": float(st.session_state.get("epsilon", settings.get("epsilon", 0))),
            "rs_ma_period": int(st.session_state.get("rs_ma_period", settings.get("rs_ma_period", 20))),
            "ma_fast": int(st.session_state.get("ma_fast", settings.get("ma_fast", 20))),
            "ma_slow": int(st.session_state.get("ma_slow", settings.get("ma_slow", 60))),
            "price_years": price_years,
        }
    )
    fx_change_pct = float("nan")
    if not macro_df.empty:
        fx_series = extract_macro_series(
            macro_df=macro_df,
            macro_series_cfg=macro_series_cfg,
            alias="usdkrw",
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


# Session state defaults

if THEME_SESSION_KEY not in st.session_state:
    st.session_state[THEME_SESSION_KEY] = get_theme_mode()
if "asof_date_str" not in st.session_state:
    st.session_state["asof_date_str"] = date.today().strftime("%Y%m%d")
if "epsilon" not in st.session_state:
    st.session_state["epsilon"] = float(settings.get("epsilon", 0))
if "rs_ma_period" not in st.session_state:
    st.session_state["rs_ma_period"] = int(settings.get("rs_ma_period", 20))
if "ma_fast" not in st.session_state:
    st.session_state["ma_fast"] = int(settings.get("ma_fast", 20))
if "ma_slow" not in st.session_state:
    st.session_state["ma_slow"] = int(settings.get("ma_slow", 60))
if "price_years" not in st.session_state:
    st.session_state["price_years"] = int(settings.get("price_years", 3))
if "filter_action_global" not in st.session_state:
    st.session_state["filter_action_global"] = "전체"
if "filter_regime_only_global" not in st.session_state:
    st.session_state["filter_regime_only_global"] = False
if "selected_sector" not in st.session_state:
    st.session_state["selected_sector"] = ""
if "selected_month" not in st.session_state:
    st.session_state["selected_month"] = ""
if "selected_cycle_phase" not in st.session_state:
    st.session_state["selected_cycle_phase"] = "ALL"
if "selected_range_preset" not in st.session_state:
    st.session_state["selected_range_preset"] = "1Y"
if "analysis_start_date" not in st.session_state:
    st.session_state["analysis_start_date"] = None
if "analysis_end_date" not in st.session_state:
    st.session_state["analysis_end_date"] = None
if "analysis_heatmap_palette" not in st.session_state:
    st.session_state["analysis_heatmap_palette"] = "classic"

theme_mode = get_theme_mode()
if st.session_state.get(THEME_SESSION_KEY) != theme_mode:
    st.session_state[THEME_SESSION_KEY] = theme_mode
analysis_heatmap_palette = str(st.session_state.get("analysis_heatmap_palette", "classic")).strip().lower()
if analysis_heatmap_palette not in HEATMAP_PALETTE_OPTIONS:
    analysis_heatmap_palette = "classic"
    st.session_state["analysis_heatmap_palette"] = analysis_heatmap_palette

if "filter_action_global" not in st.session_state:
    st.session_state["filter_action_global"] = ALL_ACTION_OPTION
elif st.session_state.get("filter_action_global") == "All":
    st.session_state["filter_action_global"] = ALL_ACTION_OPTION

normalized_range_preset = normalize_range_preset(st.session_state.get("selected_range_preset"))
if st.session_state.get("selected_range_preset") != normalized_range_preset:
    st.session_state["selected_range_preset"] = normalized_range_preset

inject_css(theme_mode)

# Sidebar

macro_cache_token = _macro_cache_token()
price_cache_token = _price_cache_token()
krx_provider_configured = _krx_provider_configured()
krx_provider_effective = _krx_provider_effective()
krx_openapi_key_present = bool(_load_api_key("KRX_OPENAPI_KEY"))

probe_price_status = _probe_market_status()
probe_macro_status = _probe_macro_status()
probe_data_status = {"price": probe_price_status, "macro": probe_macro_status}
btn_states = get_button_states(probe_data_status)

try:
    asof_default = date(
        int(st.session_state["asof_date_str"][:4]),
        int(st.session_state["asof_date_str"][4:6]),
        int(st.session_state["asof_date_str"][6:8]),
    )
except Exception:
    asof_default = date.today()

with st.sidebar:
    st.title("Korea Sector Rotation")
    st.caption("기준일, 테마, 데이터 작업")

    st.subheader("기본 설정")
    use_light_theme = st.toggle(
        "라이트 테마",
        value=theme_mode == "light",
        help="현재 대시보드 테마를 전환합니다.",
    )
    selected_theme_mode = "light" if use_light_theme else "dark"
    if selected_theme_mode != theme_mode:
        set_theme_mode(selected_theme_mode)
        st.rerun()

    selected_heatmap_palette = st.selectbox(
        "Heatmap palette",
        options=list(HEATMAP_PALETTE_OPTIONS),
        index=list(HEATMAP_PALETTE_OPTIONS).index(analysis_heatmap_palette),
        format_func=format_heatmap_palette_label,
        help="Experiment with diverging palettes for the monthly analysis heatmaps.",
    )
    if selected_heatmap_palette != analysis_heatmap_palette:
        st.session_state["analysis_heatmap_palette"] = selected_heatmap_palette
        st.rerun()

    asof_date = st.date_input(
        "기준일",
        value=asof_default,
        max_value=date.today(),
    )
    st.session_state["asof_date_str"] = asof_date.strftime("%Y%m%d")

    st.divider()
    with st.expander("고급 설정", expanded=False):
        with st.form("model_params_form"):
            st.caption("슬라이더 숫자를 직접 클릭해 세밀하게 조정할 수 있습니다.")
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                slider_epsilon = st.slider(
                    "Epsilon (방향 민감도)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state["epsilon"]),
                    step=0.05,
                    help="3MA 방향 판정 최소 변화량입니다. 0이면 모든 변화를 반영합니다.",
                )
                slider_rs_ma = st.slider(
                    "RS MA 기간",
                    min_value=5,
                    max_value=60,
                    value=int(st.session_state["rs_ma_period"]),
                    step=1,
                )
                slider_price_years = st.slider(
                    "가격 데이터 기간(년)",
                    min_value=1,
                    max_value=5,
                    value=int(st.session_state["price_years"]),
                    step=1,
                )
            with param_col2:
                slider_ma_fast = st.slider(
                    "빠른 MA",
                    min_value=5,
                    max_value=60,
                    value=int(st.session_state["ma_fast"]),
                    step=1,
                )
                slider_ma_slow = st.slider(
                    "느린 MA",
                    min_value=20,
                    max_value=120,
                    value=int(st.session_state["ma_slow"]),
                    step=1,
                )

            apply_params = st.form_submit_button("적용", width="stretch")

        if apply_params:
            st.session_state["epsilon"] = float(slider_epsilon)
            st.session_state["rs_ma_period"] = int(slider_rs_ma)
            st.session_state["ma_fast"] = int(slider_ma_fast)
            st.session_state["ma_slow"] = int(slider_ma_slow)
            st.session_state["price_years"] = int(slider_price_years)
            st.rerun()

    st.divider()
    st.subheader("데이터 작업")
    st.caption(f"시장: {probe_price_status} · 매크로: {probe_macro_status}")
    refresh_market = st.button(
        "시장데이터 갱신",
        disabled=not btn_states["refresh_market"],
        width="stretch",
    )
    refresh_macro = st.button(
        "매크로데이터 갱신",
        disabled=not btn_states["refresh_macro"],
        width="stretch",
    )
    recompute = st.button(
        "전체 재계산",
        disabled=not btn_states["recompute"],
        width="stretch",
        help="SAMPLE 데이터에서는 비활성화됩니다." if not btn_states["recompute"] else "",
    )

    st.caption("Korea Sector Rotation Dashboard")

rs_ma_period = int(st.session_state.get("rs_ma_period", settings.get("rs_ma_period", 20)))
ma_fast = int(st.session_state.get("ma_fast", settings.get("ma_fast", 20)))
ma_slow = int(st.session_state.get("ma_slow", settings.get("ma_slow", 60)))
price_years = int(st.session_state.get("price_years", settings.get("price_years", 3)))
benchmark_code = str(settings.get("benchmark_code", "1001"))
market_end_date = _resolve_market_end_date(benchmark_code)
market_end_date_str = market_end_date.strftime("%Y%m%d")

_maybe_schedule_startup_krx_warm(benchmark_code, price_years, market_end_date)


# Button handlers, each clears only its own cache (R8)

market_refresh_notice: tuple[str, str] | None = None
macro_refresh_notice: tuple[str, str] | None = None

if refresh_market:
    from src.data_sources.krx_indices import run_manual_price_refresh

    refresh_start_str, refresh_end_str = _market_range_strings(market_end_date_str, price_years)
    try:
        _cached_api_preflight.clear()
        with st.spinner("Refreshing market data..."):
            (_, _), refresh_summary = run_manual_price_refresh(
                _all_sector_codes(benchmark_code),
                refresh_start_str,
                refresh_end_str,
            )
        _cached_sector_prices.clear()
        _cached_analysis_sector_prices.clear()
        _cached_signals.clear()
        market_refresh_notice = _build_market_refresh_notice(refresh_summary)
    except Exception as exc:
        logger.exception("Manual market refresh failed")
        market_refresh_notice = ("error", f"Market data refresh failed: {exc}")

if refresh_macro:
    from src.data_sources.macro_sync import sync_macro_warehouse

    macro_end_ym = date.today().strftime("%Y%m")
    macro_start_ym = (date.today() - timedelta(days=365 * 10)).strftime("%Y%m")
    try:
        with st.spinner("Refreshing macro data..."):
            _, _, macro_summary = sync_macro_warehouse(
                start_ym=macro_start_ym,
                end_ym=macro_end_ym,
                macro_series_cfg=macro_series_cfg,
                reason="manual_refresh",
                force=False,
            )
        _cached_macro.clear()
        _cached_signals.clear()
        macro_refresh_notice = _build_macro_refresh_notice(macro_summary)
    except Exception as exc:
        logger.exception("Manual macro refresh failed")
        macro_refresh_notice = ("error", f"Macro data refresh failed: {exc}")

if recompute:
    shutil.rmtree("data/features", ignore_errors=True)
    Path("data/features").mkdir(exist_ok=True)
    _cached_signals.clear()
    st.rerun()


# Load data via cache functions

with st.spinner("데이터 로딩 중..."):
    try:
        prices_key = _price_artifact_key()
        macro_key = _macro_artifact_key()
        params = {
            "epsilon": float(st.session_state["epsilon"]),
            "rs_ma_period": rs_ma_period,
            "ma_fast": ma_fast,
            "ma_slow": ma_slow,
            "price_years": price_years,
        }
        params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]

        signals, macro_result, price_status, macro_status, market_blocking_error = _cached_signals(
            market_end_date_str,
            prices_key,
            macro_key,
            params_hash,
            macro_cache_token,
            price_cache_token,
            prices_key,
        )

        data_status = {"price": price_status, "macro": macro_status}
    except Exception as exc:
        logger.error("Data load failed: %s", exc)
        signals = []
        macro_result = pd.DataFrame(
            {
                "growth_dir": ["Flat"],
                "inflation_dir": ["Flat"],
                "regime": ["Indeterminate"],
                "confirmed_regime": ["Indeterminate"],
            },
            index=pd.DatetimeIndex([date.today()]),
        )
        price_status = "SAMPLE"
        macro_status = "SAMPLE"
        market_blocking_error = ""
        data_status = {"price": price_status, "macro": macro_status}


price_warm_status: dict[str, object] = {}
price_cache_case = None
if price_status == "CACHED":
    from src.data_sources.krx_indices import read_warm_status

    price_warm_status = read_warm_status()
    price_cache_case = resolve_price_cache_banner_case(
        price_status=price_status,
        provider_mode=krx_provider_effective,
        openapi_key_present=krx_openapi_key_present,
        market_end_date_str=market_end_date_str,
        warm_status=price_warm_status,
    )


# SAMPLE mode warning (R9)

try:
    preflight_status = _cached_api_preflight(timeout_sec=3)
except Exception as exc:
    preflight_status = {
        "PRECHECK": {
            "status": "HTTP_ERROR",
            "detail": str(exc),
            "url": "",
            "checked_at": "",
        }
    }

preflight_issues = {
    name: info for name, info in preflight_status.items() if info.get("status") != "OK"
}
openapi_missing_key_warning_shown = (
    krx_provider_configured == "OPENAPI" and not krx_openapi_key_present
)
_show_notice_toast(market_refresh_notice)
_show_notice_toast(macro_refresh_notice)

dashboard_status_banner = resolve_dashboard_status_banner(
    data_status=data_status,
    market_blocking_error=market_blocking_error,
    price_cache_case=price_cache_case,
    openapi_key_warning=openapi_missing_key_warning_shown,
    preflight_status=preflight_status,
    price_warm_status=price_warm_status,
)
_render_dashboard_status_banner(dashboard_status_banner)


current_regime = "Indeterminate"
regime_is_confirmed = False
yield_curve_status: str | None = None
if not macro_result.empty:
    if "confirmed_regime" in macro_result.columns:
        current_regime = str(macro_result["confirmed_regime"].iloc[-1])
        raw_regime = str(macro_result["regime"].iloc[-1]) if "regime" in macro_result.columns else current_regime
        regime_is_confirmed = current_regime == raw_regime and current_regime != "Indeterminate"
    elif "regime" in macro_result.columns:
        current_regime = str(macro_result["regime"].iloc[-1])
    if "yield_curve" in macro_result.columns:
        yield_curve_status = str(macro_result["yield_curve"].iloc[-1])

is_provisional = any(
    getattr(s, "is_provisional", False) for s in signals
)

# Extract latest values for macro tile display
_, macro_df = _cached_macro(macro_cache_token)  # cached ??no extra API call
growth_val: float | None = None
inflation_val: float | None = None
fx_change: float | None = None
if not macro_df.empty:
    _growth_s = extract_macro_series(macro_df, macro_series_cfg, "leading_index")
    if not _growth_s.empty:
        growth_val = float(_growth_s.iloc[-1])
    _inflation_s = extract_macro_series(macro_df, macro_series_cfg, "cpi_yoy")
    if not _inflation_s.empty:
        inflation_val = float(_inflation_s.iloc[-1])
    _fx_s = extract_macro_series(macro_df, macro_series_cfg, "usdkrw")
    if len(_fx_s) >= 2:
        fx_change = float((_fx_s.iloc[-1] / _fx_s.iloc[-2] - 1) * 100)

render_page_header(
    title="Korea Sector Rotation",
    description="Move from range selection to cycle context, sector comparison, and linked detail tracking without leaving the main canvas.",
    pills=[
        {"label": "Regime", "value": current_regime, "tone": "success" if regime_is_confirmed else "warning"},
        {"label": "Market", "value": price_status, "tone": "danger" if price_status == "SAMPLE" else "warning" if price_status == "CACHED" else "success"},
        {"label": "Macro", "value": macro_status, "tone": "danger" if macro_status == "SAMPLE" else "warning" if macro_status == "CACHED" else "success"},
        {"label": "Provider", "value": "KRX OpenAPI" if krx_provider_effective == "OPENAPI" else "pykrx", "tone": "info"},
    ],
)

try:
    if price_status == "BLOCKED":
        sector_prices_canvas = pd.DataFrame()
    else:
        sector_prices_canvas = _cached_analysis_sector_prices(
            market_end_date_str,
            benchmark_code,
            price_years,
            prices_key if "prices_key" in locals() else _price_artifact_key(),
        )
except Exception as exc:
    logger.warning("Analysis canvas price load fallback: %s", exc)
    sector_prices_canvas = pd.DataFrame()

sector_name_map = _build_sector_name_map(
    signals=list(signals),
    sector_prices=sector_prices_canvas,
    benchmark_code=benchmark_code,
)
prices_wide = _build_prices_wide(
    sector_prices=sector_prices_canvas,
    sector_name_map=sector_name_map,
)
benchmark_label = sector_name_map.get(benchmark_code, "KOSPI")
sector_columns = [col for col in prices_wide.columns if col != benchmark_label]
monthly_close_full, monthly_returns_full, _benchmark_monthly_return, monthly_excess_returns_full = _build_monthly_return_views(
    prices_wide=prices_wide,
    sector_columns=sector_columns,
    benchmark_label=benchmark_label,
)
cycle_segments_all, phase_by_month = _build_cycle_segments(
    macro_result=macro_result,
    monthly_close=monthly_close_full,
)

if not prices_wide.empty:
    analysis_min_date = prices_wide.index.min().date()
    analysis_max_date = prices_wide.index.max().date()
else:
    analysis_min_date = market_end_date - timedelta(days=365)
    analysis_max_date = market_end_date

if not st.session_state.get("analysis_end_date"):
    st.session_state["analysis_end_date"] = analysis_max_date
if not st.session_state.get("analysis_start_date"):
    preset_start, preset_end = resolve_range_from_preset(
        max_date=analysis_max_date,
        min_date=analysis_min_date,
        preset=normalize_range_preset(st.session_state.get("selected_range_preset", "1Y")),
    )
    st.session_state["analysis_start_date"] = preset_start
    st.session_state["analysis_end_date"] = preset_end

analysis_start_date = pd.Timestamp(st.session_state["analysis_start_date"]).date()
analysis_end_date = pd.Timestamp(st.session_state["analysis_end_date"]).date()
current_range_preset = infer_range_preset(
    start_date=analysis_start_date,
    end_date=analysis_end_date,
    min_date=analysis_min_date,
    max_date=analysis_max_date,
)
st.session_state["selected_range_preset"] = current_range_preset

toolbar_selected_sector = str(st.session_state.get("selected_sector", "")).strip() or "Auto"
toolbar_selected_phase = str(st.session_state.get("selected_cycle_phase", "ALL")).strip() or "ALL"
toolbar_selected_preset = normalize_range_preset(st.session_state.get("selected_range_preset", "1Y"))
resolved_start, resolved_end, resolved_preset, toolbar_submitted = render_analysis_toolbar(
    min_date=analysis_min_date,
    max_date=analysis_max_date,
    start_date=analysis_start_date,
    end_date=analysis_end_date,
    selected_range_preset=toolbar_selected_preset if current_range_preset == "CUSTOM" else current_range_preset,
    selected_cycle_phase=toolbar_selected_phase,
    selected_sector=toolbar_selected_sector,
)
if toolbar_submitted:
    st.session_state["analysis_start_date"] = resolved_start
    st.session_state["analysis_end_date"] = resolved_end
    st.session_state["selected_range_preset"] = resolved_preset
    st.rerun()

analysis_prices = prices_wide.loc[
    (prices_wide.index >= pd.Timestamp(st.session_state["analysis_start_date"]))
    & (prices_wide.index <= pd.Timestamp(st.session_state["analysis_end_date"]))
]
phase_by_month_visible = phase_by_month.loc[
    (phase_by_month.index >= pd.Timestamp(st.session_state["analysis_start_date"]).to_period("M").to_timestamp("M"))
    & (phase_by_month.index <= pd.Timestamp(st.session_state["analysis_end_date"]).to_period("M").to_timestamp("M"))
] if not phase_by_month.empty else pd.Series(dtype=object)

selected_cycle_phase = str(st.session_state.get("selected_cycle_phase", "ALL") or "ALL")
analysis_prices_phase = _filter_prices_for_phase(
    prices_wide=analysis_prices,
    phase_by_month=phase_by_month_visible,
    selected_cycle_phase=selected_cycle_phase,
)

heatmap_return_source = _filter_monthly_frame_for_analysis(
    monthly_frame=monthly_returns_full,
    start_date=analysis_start_date,
    end_date=analysis_end_date,
    selected_cycle_phase=selected_cycle_phase,
    phase_by_month=phase_by_month_visible,
)
heatmap_strength_source = _filter_monthly_frame_for_analysis(
    monthly_frame=monthly_excess_returns_full,
    start_date=analysis_start_date,
    end_date=analysis_end_date,
    selected_cycle_phase=selected_cycle_phase,
    phase_by_month=phase_by_month_visible,
)
heatmap_return_display = _build_heatmap_display(heatmap_return_source)
heatmap_strength_display = _build_heatmap_display(heatmap_strength_source)

visible_months = list(heatmap_return_display.columns) if not heatmap_return_display.empty else []
if visible_months:
    if st.session_state.get("selected_month") not in visible_months:
        st.session_state["selected_month"] = visible_months[-1]
else:
    st.session_state["selected_month"] = ""

visible_segments = [
    segment
    for segment in cycle_segments_all
    if pd.Timestamp(segment["end"]) >= pd.Timestamp(st.session_state["analysis_start_date"]).to_period("M").to_timestamp("M")
    and pd.Timestamp(segment["start"]) <= pd.Timestamp(st.session_state["analysis_end_date"]).to_period("M").to_timestamp("M")
]

with st.container(border=True):
    render_panel_header(
        eyebrow="Sector comparison",
        title="Monthly sector return",
        description="Scan absolute monthly sector returns first. Clicking a cell pins the sector and month for the detail panel below.",
        badge=format_cycle_phase_label(selected_cycle_phase),
    )
    heatmap_fig = build_sector_strength_heatmap(
        heatmap_return_display,
        selected_sector=str(st.session_state.get("selected_sector", "")),
        selected_month=str(st.session_state.get("selected_month", "")),
        theme_mode=theme_mode,
        palette=analysis_heatmap_palette,
        title="Monthly sector return",
        empty_message="No monthly sector return data is available for the active filters.",
        helper_metric_label="monthly return",
        hover_value_suffix="%",
    )
    heatmap_event = st.plotly_chart(
        heatmap_fig,
        width="stretch",
        key="analysis_sector_return_heatmap",
        on_select="rerun",
        selection_mode=("points",),
        config={"displayModeBar": False},
    )
    absolute_selection = _extract_heatmap_selection(heatmap_event)

with st.container(border=True):
    render_panel_header(
        eyebrow="Relative strength",
        title="Monthly sector strength vs KOSPI",
        description="Read each cell as monthly excess return versus KOSPI, using the same linked sector/month selection.",
        badge=format_cycle_phase_label(selected_cycle_phase),
    )
    strength_fig = build_sector_strength_heatmap(
        heatmap_strength_display,
        selected_sector=str(st.session_state.get("selected_sector", "")),
        selected_month=str(st.session_state.get("selected_month", "")),
        theme_mode=theme_mode,
        palette=analysis_heatmap_palette,
        title="Monthly sector strength vs KOSPI",
        empty_message="No monthly sector strength vs KOSPI data is available for the active filters.",
        helper_metric_label="monthly excess return",
        hover_value_suffix="%p vs KOSPI",
    )
    strength_event = st.plotly_chart(
        strength_fig,
        width="stretch",
        key="analysis_sector_strength_heatmap",
        on_select="rerun",
        selection_mode=("points",),
        config={"displayModeBar": False},
    )
    relative_selection = _extract_heatmap_selection(strength_event)

shared_heatmap_selection = relative_selection or absolute_selection
if shared_heatmap_selection is not None:
    month_value, sector_value = shared_heatmap_selection
    if (
        month_value != str(st.session_state.get("selected_month", ""))
        or sector_value != str(st.session_state.get("selected_sector", ""))
    ):
        st.session_state["selected_month"] = month_value
        st.session_state["selected_sector"] = sector_value
        st.rerun()

with st.container(border=True):
    render_panel_header(
        eyebrow="Cycle context",
        title="Cycle timeline context",
        description="Use early/late cycle filters to compress the heatmap and the detail chart down to a specific macro phase.",
        badge=current_regime,
    )
    chosen_cycle_phase = render_cycle_timeline_panel(
        segments=visible_segments,
        selected_cycle_phase=selected_cycle_phase,
        theme_mode=theme_mode,
    )
    if chosen_cycle_phase != selected_cycle_phase:
        st.session_state["selected_cycle_phase"] = chosen_cycle_phase
        st.rerun()

detail_prices = analysis_prices_phase if not analysis_prices_phase.empty else analysis_prices
if not detail_prices.empty:
    detail_columns = [col for col in sector_columns if col in detail_prices.columns]
    if benchmark_label in detail_prices.columns:
        detail_columns.append(benchmark_label)
    detail_prices = detail_prices[detail_columns]
else:
    detail_prices = pd.DataFrame()
detail_normalized = pd.DataFrame()
ranking_rows: list[dict[str, object]] = []
if not detail_prices.empty:
    detail_prices = detail_prices.dropna(how="all")
    detail_prices = detail_prices.ffill().dropna(axis=1, how="all")
    if len(detail_prices) >= 2:
        detail_normalized = detail_prices.div(detail_prices.iloc[0]).mul(100.0)
        sector_detail_returns = (
            detail_normalized[[col for col in detail_normalized.columns if col != benchmark_label]].iloc[-1] - 100.0
        )
        sector_detail_returns = sector_detail_returns.dropna().sort_values(ascending=False)
        ranking_rows = [
            {"sector": str(name), "return_pct": float(value)}
            for name, value in sector_detail_returns.items()
        ]

if ranking_rows:
    selected_sector_state = str(st.session_state.get("selected_sector", ""))
    if selected_sector_state not in [row["sector"] for row in ranking_rows]:
        st.session_state["selected_sector"] = str(ranking_rows[0]["sector"])
else:
    st.session_state["selected_sector"] = ""

selected_sector = str(st.session_state.get("selected_sector", ""))
comparison_sectors = [
    str(row["sector"])
    for row in ranking_rows
    if str(row["sector"]) != selected_sector
][:2]
detail_figure = build_sector_detail_figure(
    detail_normalized,
    selected_sector=selected_sector,
    benchmark_label=benchmark_label if benchmark_label in detail_normalized.columns else None,
    comparison_sectors=comparison_sectors,
    selected_month=str(st.session_state.get("selected_month", "")),
    theme_mode=theme_mode,
)

with st.container(border=True):
    detail_badge = selected_sector or "No selection"
    render_panel_header(
        eyebrow="Linked detail",
        title="Selected sector detail tracking",
        description="Rank the sectors on the left, then compare the selected sector against the benchmark and the strongest peers on the right.",
        badge=detail_badge,
    )
    chosen_sector, chosen_preset = render_sector_detail_panel(
        ranking_rows=ranking_rows,
        detail_figure=detail_figure,
        selected_sector=selected_sector,
        selected_range_preset=normalize_range_preset(st.session_state.get("selected_range_preset", "1Y")),
    )
    if chosen_sector and chosen_sector != selected_sector:
        st.session_state["selected_sector"] = chosen_sector
        st.rerun()
    if chosen_preset != normalize_range_preset(st.session_state.get("selected_range_preset", "1Y")):
        preset_start, preset_end = resolve_range_from_preset(
            max_date=analysis_max_date,
            min_date=analysis_min_date,
            preset=chosen_preset,
        )
        st.session_state["selected_range_preset"] = chosen_preset
        st.session_state["analysis_start_date"] = preset_start
        st.session_state["analysis_end_date"] = preset_end
        st.rerun()

# Global filters

is_mobile_client = _is_mobile_client()
filter_action_global, filter_regime_only_global = render_top_bar_filters(
    current_regime=current_regime,
    action_options=[ALL_ACTION_OPTION, "Strong Buy", "Watch", "Hold", "Avoid", "N/A"],
    is_mobile=is_mobile_client,
)

signals_filtered = list(signals)
if filter_regime_only_global:
    signals_filtered = [s for s in signals_filtered if s.macro_regime == current_regime]
if filter_action_global != ALL_ACTION_OPTION:
    signals_filtered = [s for s in signals_filtered if s.action == filter_action_global]


def _rs_divergence_pct(signal) -> float:
    if pd.isna(signal.rs) or pd.isna(signal.rs_ma) or signal.rs_ma == 0:
        return float("nan")
    return float((signal.rs - signal.rs_ma) / signal.rs_ma * 100)


action_priority = {
    "Strong Buy": 0,
    "Watch": 1,
    "Hold": 2,
    "Avoid": 3,
    "N/A": 4,
}


def _top_pick_sort_key(signal) -> tuple[int, float]:
    rs_div = _rs_divergence_pct(signal)
    rs_div_rank = -rs_div if not pd.isna(rs_div) else float("inf")
    return action_priority.get(signal.action, 99), rs_div_rank


top_pick_signals = sorted(signals_filtered, key=_top_pick_sort_key)

# Tabs interface

tab_summary, tab_charts, tab_all_signals = st.tabs([
    "대시보드 요약",
    "모멘텀/차트 분석",
    "전체 종목 데이터",
])

with tab_summary:
    from src.ui.components import (
        render_action_summary,
        render_decision_hero,
        render_status_card_row,
        render_top_picks_table,
    )

    render_decision_hero(
        regime=current_regime,
        regime_is_confirmed=regime_is_confirmed,
        growth_val=growth_val,
        inflation_val=inflation_val,
        fx_change=fx_change,
        is_provisional=is_provisional,
        theme_mode=theme_mode,
    )
    render_status_card_row(
        current_regime=current_regime,
        regime_is_confirmed=regime_is_confirmed,
        price_status=price_status,
        macro_status=macro_status,
        yield_curve_status=yield_curve_status,
    )

    with st.container(border=True):
        render_panel_header(
            eyebrow="Priority board",
            title="Top picks",
            description="The highest-ranked sectors after the active filters.",
            badge=f"{min(5, len(top_pick_signals))} shown",
        )
        render_top_picks_table(top_pick_signals, limit=5)

    st.divider()
    st.subheader("액션 분포")
    with st.container(border=True):
        render_panel_header(
            eyebrow="Breadth",
            title="Action distribution",
            description="See how the filtered universe spreads across the action ladder.",
        )
        render_action_summary(signals_filtered, theme_mode=theme_mode)

with tab_charts:
    from src.ui.components import (
        render_returns_heatmap,
        render_rs_momentum_bar,
        render_rs_scatter,
    )

    render_panel_header(
        eyebrow="Momentum map",
        title="RS scatter and momentum bars",
        description="Relative strength and RS gap panels use the same visual shell for easier scanning.",
    )
    st.markdown(
        """
<div class="compact-note">
<b>읽는 법</b>
RS 산점도는 벤치마크 대비 상대강도와 추세를 함께 보여줍니다. 오른쪽 위로 갈수록 강도가 높고 모멘텀이 가속되는 섹터입니다.
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("차트 해석 상세", expanded=False):
        st.markdown(
            """
**X축 (RS)**: 섹터 종가 대비 벤치마크(KOSPI) 비율입니다. 값이 높을수록 벤치마크 대비 상대강도가 높습니다.

**Y축 (RS MA)**: RS의 이동평균(기본 20)입니다. RS의 추세를 부드럽게 보여줍니다.

**점선 대각선**: `RS = RS MA` 기준선입니다. 기준선 위/아래로 모멘텀 방향을 판단합니다.

| 위치 | 의미 |
|------|------|
| 대각선 위 (RS > RS MA) | RS가 평균을 상회해 모멘텀이 가속되는 강세 신호 |
| 대각선 아래 (RS < RS MA) | RS가 평균을 하회해 모멘텀이 둔화되는 약세 신호 |
| 오른쪽 | 벤치마크 대비 상대강도가 강한 섹터 |
| 왼쪽 | 벤치마크 대비 상대강도가 약한 섹터 |
"""
        )

    if signals_filtered:
        benchmark_missing = any(
            "Benchmark Missing" in getattr(signal, "alerts", []) for signal in signals_filtered
        )
        if benchmark_missing:
            st.warning(
                "벤치마크(KOSPI, 1001) 데이터 누락으로 RS 산점도를 계산할 수 없습니다. 시장데이터 갱신 후 다시 시도하세요."
            )
        else:
            scatter_height = 520 if is_mobile_client else 700
            scatter_margin = (
                dict(l=44, r=18, t=56, b=50)
                if is_mobile_client
                else dict(l=72, r=32, t=64, b=64)
            )
            fig_scatter = render_rs_scatter(
                signals_filtered,
                height=scatter_height,
                margin=scatter_margin,
                theme_mode=theme_mode,
            )
            if is_mobile_client:
                st.plotly_chart(fig_scatter, width="stretch")
            else:
                _, scatter_col_c, _ = st.columns([0.7, 3.6, 0.7])
                with scatter_col_c:
                    st.plotly_chart(fig_scatter, width="stretch")

            st.markdown(
                """
<div class="compact-note">
<b>RS 이격도</b>
RS 이격도는 현재 RS가 이동평균 대비 얼마나 위나 아래에 있는지를 수치로 보여줍니다. 양수는 가속, 음수는 둔화를 뜻합니다.
</div>
""",
                unsafe_allow_html=True,
            )
            fig_bar = render_rs_momentum_bar(signals_filtered, theme_mode=theme_mode)
            if fig_bar.data:
                st.plotly_chart(fig_bar, width="stretch")
            else:
                st.info("RS/RS MA 데이터가 충분하지 않습니다.")

        st.markdown(
            """
<div class="compact-note">
<b>수익률 히트맵</b>
기간별 수익률을 한눈에 비교해 최근 강도 지속 여부를 확인할 수 있습니다.
</div>
""",
            unsafe_allow_html=True,
        )
        render_panel_header(
            eyebrow="Cross-section",
            title="Return heatmap",
            description="Compare multi-horizon sector returns with the same panel treatment used across the dashboard.",
        )
        fig_heatmap = render_returns_heatmap(signals_filtered, theme_mode=theme_mode)
        st.plotly_chart(fig_heatmap, width="stretch")
    else:
        st.info("글로벌 필터 조건에 맞는 신호가 없습니다.")

with tab_all_signals:
    render_panel_header(
        eyebrow="Full table",
        title="All sector signals",
        description="Native Streamlit grid with the same shell and filter feedback as the summary panels.",
    )
    from src.ui.components import render_signal_table

    st.caption(
        f"적용 필터: 액션={filter_action_global}, "
        f"현재 국면만 보기={'ON' if filter_regime_only_global else 'OFF'}"
    )
    with st.expander("적합/비적합 판정 기준", expanded=False):
        st.markdown(
            """
- `국면 적합`은 현재 시점의 확정 국면에서 해당 섹터가 맵핑되는지(`macro_fit`)로 판정합니다.
- 현재 시점 국면은 `confirmed_regime` 기준입니다. 아직 확정 전이면 잠정 상태로 해석합니다.
- 맵핑 기준은 `config/sector_map.yml`의 `regimes -> {국면} -> sectors`입니다.
- 최종 `액션`(Strong Buy/Watch/Hold/Avoid)은 국면 적합 여부와 모멘텀 조건(RS, 추세)을 결합해 계산합니다.
- `Indeterminate` 국면에서는 맵핑 섹터가 없어 전체가 `비적합`으로 표시될 수 있습니다.
"""
        )
    with st.expander("알림 카테고리 설명", expanded=False):
        rsi_overbought = int(settings.get("rsi_overbought", 70))
        rsi_oversold = int(settings.get("rsi_oversold", 30))
        fx_shock_pct = float(settings.get("fx_shock_pct", 3.0))

        st.markdown(
            f"""
- **Overheat**: 일간 RSI(`rsi_d`)가 `{rsi_overbought}` 이상이면 추가됩니다.
- **Oversold**: 일간 RSI(`rsi_d`)가 `{rsi_oversold}` 이하이면 추가됩니다.
- **FX Shock**: `|USD/KRW 변화율| > {fx_shock_pct:.1f}%`이고 수출 섹터이며 현재 액션이 `Strong Buy`이면 알림을 추가하고 액션을 `Watch`로 조정합니다.
- **Benchmark Missing**: 벤치마크 가격 데이터가 비어 있으면 모든 섹터에 추가됩니다 (액션 `N/A`).
- **RS Data Insufficient**: 특정 섹터의 RS/RS MA 계산이 불가능할 때 해당 섹터에 추가됩니다 (액션 `N/A`).
"""
        )
        st.caption(
            "참고: `FX Shock`은 신호 계산 시점의 최신 USD/KRW 변화율 기준으로 판정합니다. "
            "USD/KRW 시계열이 2개 미만이면 해당 시점에서는 `FX Shock`이 계산되지 않습니다."
        )

    render_signal_table(
        signals,
        filter_action=filter_action_global,
        filter_regime_only=filter_regime_only_global,
        current_regime=current_regime,
        theme_mode=theme_mode,
    )

# Footer

st.divider()
st.caption(
    f"기준일: {asof_date} | 데이터: "
    f"{'KRX OpenAPI' if krx_provider_effective == 'OPENAPI' else 'pykrx'} (KRX), "
    "ECOS (한국은행), KOSIS (통계청) | "
    f"국면: {current_regime}"
)

