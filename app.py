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

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ?? Page config (must be first Streamlit call) ??????????????????????????????
st.set_page_config(
    page_title="Korea Sector Rotation",
    page_icon="?뱤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ?? Local imports ????????????????????????????????????????????????????????????
from src.ui.styles import get_table_style_tokens, inject_css
from src.ui.data_status import is_sample_mode, get_button_states
from src.macro.series_utils import (
    build_enabled_ecos_config,
    build_enabled_kosis_config,
    extract_macro_series,
    to_plotly_time_index,
)

# ?? Config loading ????????????????????????????????????????????????????????????


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

# ?? Cache key helper (R8) ?????????????????????????????????????????????????????


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


@st.cache_data(ttl=600)
def _cached_api_preflight(timeout_sec: int = 3) -> dict:
    """Cached API endpoint reachability check (10 min TTL)."""
    from src.data_sources.preflight import run_api_preflight

    return run_api_preflight(timeout_sec=timeout_sec)


# ?? Named cache functions (R8) ????????????????????????????????????????????????


@st.cache_data(ttl=CACHE_TTL)
def _cached_sector_prices(
    asof_date_str: str,
    benchmark_code: str,
    price_years: int,
    price_cache_token: str,
):
    """Fetch or load sector prices. Includes KRX provider/key cache token."""
    _ = price_cache_token  # explicit cache key input for provider/key invalidation
    from src.data_sources.krx_indices import load_sector_prices
    from src.transforms.calendar import get_last_business_day

    # Gather all sector codes from map
    all_codes: list[str] = []
    for regime_data in sector_map.get("regimes", {}).values():
        for s in regime_data.get("sectors", []):
            code = str(s["code"])
            if code not in all_codes:
                all_codes.append(code)
    if benchmark_code and str(benchmark_code) not in all_codes:
        all_codes.append(str(benchmark_code))

    end_date = get_last_business_day()
    start_date = end_date - timedelta(days=365 * price_years)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    status, df = load_sector_prices(all_codes, start_str, end_str)
    return status, df


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
    prices_key: tuple,
    macro_key: tuple,
    params_hash: str,
    macro_cache_token: str,
    price_cache_token: str,
):
    """Compute signals. Keyed by parquet file metadata + params hash."""
    from src.macro.regime import compute_regime_history
    from src.signals.matrix import build_signal_table

    price_years = int(st.session_state.get("price_years", settings.get("price_years", 3)))
    price_status, sector_prices = _cached_sector_prices(
        st.session_state.get("asof_date_str", date.today().strftime("%Y%m%d")),
        str(settings.get("benchmark_code", "1001")),
        price_years,
        price_cache_token,
    )
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
        # CPI MoM (?꾩썡鍮? as primary inflation signal; fall back to YoY
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

        # Yield curve spread: KTB 3Y ??base rate (monthly)
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

    signals = build_signal_table(
        sector_prices=sector_prices,
        benchmark_prices=bench_series,
        macro_result=macro_result,
        sector_map=sector_map,
        settings=runtime_settings,
        fx_change_pct=fx_change_pct,
    )

    return signals, macro_result, price_status, macro_status


# ?? Session state defaults ?????????????????????????????????????????????????????

if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"
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
    st.session_state["filter_action_global"] = "?꾩껜"
if "filter_regime_only_global" not in st.session_state:
    st.session_state["filter_regime_only_global"] = False

theme_mode = str(st.session_state.get("theme_mode", "dark")).strip().lower()
if theme_mode not in {"dark", "light"}:
    theme_mode = "dark"
    st.session_state["theme_mode"] = theme_mode

inject_css(theme_mode)

# ?? Sidebar ???????????????????????????????????????????????????????????????????

prices_parquet = "data/curated/sector_prices.parquet"
macro_parquet = "data/curated/macro_monthly.parquet"
macro_cache_token = _macro_cache_token()
price_cache_token = _price_cache_token()
krx_provider_configured = _krx_provider_configured()
krx_provider_effective = _krx_provider_effective()
krx_openapi_key_present = bool(_load_api_key("KRX_OPENAPI_KEY"))

probe_price_status = "CACHED" if Path(prices_parquet).exists() else "SAMPLE"
probe_macro_status = "CACHED" if Path(macro_parquet).exists() else "SAMPLE"
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
    st.caption("UI / Theme Control Panel")

    st.subheader("鍮좊Ⅸ ?곹깭")
    use_light_theme = st.toggle(
        "?쇱씠???뚮쭏",
        value=theme_mode == "light",
        help="?댁젣 ???ㅽ겕 ?뚮쭏媛 ?곸슜?⑸땲??",
    )
    selected_theme_mode = "light" if use_light_theme else "dark"
    if selected_theme_mode != theme_mode:
        st.session_state["theme_mode"] = selected_theme_mode
        st.rerun()

    asof_date = st.date_input(
        "기준일",
        value=asof_default,
        max_value=date.today(),
    )
    st.session_state["asof_date_str"] = asof_date.strftime("%Y%m%d")

    quick_col1, quick_col2 = st.columns(2)
    with quick_col1:
        st.metric("媛寃??곹깭", probe_price_status)
    with quick_col2:
        st.metric("留ㅽ겕濡??곹깭", probe_macro_status)

    st.divider()
    st.subheader("湲濡쒕쾶 ?꾪꽣")
    st.selectbox(
        "?≪뀡 ?꾪꽣",
        options=["?꾩껜", "Strong Buy", "Watch", "Hold", "Avoid", "N/A"],
        key="filter_action_global",
    )
    st.checkbox(
        "?꾩옱 援?㈃ ?뱁꽣留?蹂닿린",
        key="filter_regime_only_global",
    )

    st.divider()
    with st.expander("紐⑤뜽 ?뚮씪誘명꽣", expanded=False):
        with st.form("model_params_form"):
            slider_epsilon = st.slider(
                "Epsilon (諛⑺뼢 誘쇨컧??",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state["epsilon"]),
                step=0.05,
                help="3MA 諛⑺뼢 ?먮퀎 理쒖냼 蹂?붾웾. 0 = 紐⑤뱺 蹂??諛섏쁺",
            )
            slider_rs_ma = st.slider(
                "RS MA 湲곌컙",
                min_value=5,
                max_value=60,
                value=int(st.session_state["rs_ma_period"]),
                step=1,
            )
            slider_ma_fast = st.slider(
                "鍮좊Ⅸ MA",
                min_value=5,
                max_value=60,
                value=int(st.session_state["ma_fast"]),
                step=1,
            )
            slider_ma_slow = st.slider(
                "?먮┛ MA",
                min_value=20,
                max_value=120,
                value=int(st.session_state["ma_slow"]),
                step=1,
            )
            slider_price_years = st.slider(
                "?곗씠??湲곌컙 (??",
                min_value=1,
                max_value=5,
                value=int(st.session_state["price_years"]),
                step=1,
            )

            with st.expander("怨좉툒 吏곸젒 ?낅젰", expanded=False):
                use_advanced_inputs = st.checkbox("怨좉툒 媛믪쑝濡??곸슜", value=False)
                adv_epsilon = st.number_input(
                    "Epsilon 吏곸젒 ?낅젰",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state["epsilon"]),
                    step=0.01,
                    format="%.2f",
                )
                adv_rs_ma = st.number_input(
                    "RS MA 吏곸젒 ?낅젰",
                    min_value=5,
                    max_value=60,
                    value=int(st.session_state["rs_ma_period"]),
                    step=1,
                )
                adv_ma_fast = st.number_input(
                    "鍮좊Ⅸ MA 吏곸젒 ?낅젰",
                    min_value=5,
                    max_value=60,
                    value=int(st.session_state["ma_fast"]),
                    step=1,
                )
                adv_ma_slow = st.number_input(
                    "?먮┛ MA 吏곸젒 ?낅젰",
                    min_value=20,
                    max_value=120,
                    value=int(st.session_state["ma_slow"]),
                    step=1,
                )
                adv_price_years = st.number_input(
                    "?곗씠??湲곌컙 吏곸젒 ?낅젰",
                    min_value=1,
                    max_value=5,
                    value=int(st.session_state["price_years"]),
                    step=1,
                )

            apply_params = st.form_submit_button("?곸슜", width='stretch')

        if apply_params:
            if use_advanced_inputs:
                st.session_state["epsilon"] = float(adv_epsilon)
                st.session_state["rs_ma_period"] = int(adv_rs_ma)
                st.session_state["ma_fast"] = int(adv_ma_fast)
                st.session_state["ma_slow"] = int(adv_ma_slow)
                st.session_state["price_years"] = int(adv_price_years)
            else:
                st.session_state["epsilon"] = float(slider_epsilon)
                st.session_state["rs_ma_period"] = int(slider_rs_ma)
                st.session_state["ma_fast"] = int(slider_ma_fast)
                st.session_state["ma_slow"] = int(slider_ma_slow)
                st.session_state["price_years"] = int(slider_price_years)
            st.rerun()

    st.divider()
    st.subheader("?곗씠???묒뾽")
    refresh_market = st.button(
        "?쒖옣?곗씠??媛깆떊",
        disabled=not btn_states["refresh_market"],
        width='stretch',
    )
    refresh_macro = st.button(
        "留ㅽ겕濡쒕뜲?댄꽣 媛깆떊",
        disabled=not btn_states["refresh_macro"],
        width='stretch',
    )
    recompute = st.button(
        "전체 재계산",
        disabled=not btn_states["recompute"],
        width='stretch',
        help="SAMPLE ?곗씠?곗뿉?쒕뒗 鍮꾪솢?깊솕?⑸땲??" if not btn_states["recompute"] else "",
    )

    st.caption("Korea Sector Rotation Dashboard")

rs_ma_period = int(st.session_state.get("rs_ma_period", settings.get("rs_ma_period", 20)))
ma_fast = int(st.session_state.get("ma_fast", settings.get("ma_fast", 20)))
ma_slow = int(st.session_state.get("ma_slow", settings.get("ma_slow", 60)))
price_years = int(st.session_state.get("price_years", settings.get("price_years", 3)))


# ?? Button handlers ??each clears only its own cache (R8) ????????????????????

if refresh_market:
    Path(prices_parquet).unlink(missing_ok=True)
    _cached_sector_prices.clear()
    st.rerun()

if refresh_macro:
    Path(macro_parquet).unlink(missing_ok=True)
    _cached_macro.clear()
    st.rerun()

if recompute:
    shutil.rmtree("data/features", ignore_errors=True)
    Path("data/features").mkdir(exist_ok=True)
    _cached_signals.clear()
    st.rerun()


# ?? Load data via cache functions ?????????????????????????????????????????????

with st.spinner("?곗씠??濡쒕뵫 以?.."):
    try:
        prices_key = _parquet_key(prices_parquet)
        macro_key = _parquet_key(macro_parquet)
        params = {
            "epsilon": float(st.session_state["epsilon"]),
            "rs_ma_period": rs_ma_period,
            "ma_fast": ma_fast,
            "ma_slow": ma_slow,
            "price_years": price_years,
        }
        params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]

        signals, macro_result, price_status, macro_status = _cached_signals(
            prices_key,
            macro_key,
            params_hash,
            macro_cache_token,
            price_cache_token,
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
        data_status = {"price": price_status, "macro": macro_status}


# ?? SAMPLE mode warning (R9) ??????????????????????????????????????????????????

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
if preflight_issues:
    summary = " | ".join(
        f"{name}:{info.get('status')} ({info.get('detail', 'n/a')})"
        for name, info in preflight_issues.items()
    )
    st.warning(f"API preflight warning: {summary}")
else:
    st.caption("API preflight: ECOS/KOSIS/KRX endpoints reachable.")

if krx_provider_configured == "OPENAPI" and not krx_openapi_key_present:
    st.warning(
        "KRX_PROVIDER is set to OPENAPI but KRX_OPENAPI_KEY is not configured. "
        "Market data will fall back to cache/SAMPLE until the key is provided.",
        icon="⚠️",
    )

if is_sample_mode(data_status):
    st.error(
        "⚠️ **SAMPLE 데이터 모드**: 실제 시장 데이터를 불러오지 못해 합성 데이터를 표시합니다. "
        "API 설정 또는 네트워크 상태를 확인한 뒤 새로고침을 시도하세요.",
        icon="⚠️",
    )
elif price_status == "CACHED" or macro_status == "CACHED":
    _cache_parts = []
    if price_status == "CACHED":
        _p = Path(prices_parquet)
        if _p.exists():
            import datetime as _dt

            _mtime = _dt.datetime.fromtimestamp(_p.stat().st_mtime).strftime("%Y-%m-%d")
            _cache_parts.append(f"시장데이터({_mtime})")
        else:
            _cache_parts.append("시장데이터")
    if macro_status == "CACHED":
        _cache_parts.append("매크로데이터")

    provider_note = ""
    if price_status == "CACHED":
        if krx_provider_effective == "OPENAPI":
            provider_note = (
                "KRX OpenAPI live fetch failed (or key permission is incomplete). "
                "Check KRX_OPENAPI_KEY and service approval on openapi.krx.co.kr."
            )
        else:
            provider_note = (
                "pykrx path may require authenticated KRX sessions (pykrx#276). "
                "Set KRX_PROVIDER=OPENAPI with KRX_OPENAPI_KEY for a more stable live path."
            )

    st.warning(
        f"캐시 데이터를 사용 중: {', '.join(_cache_parts)}. {provider_note} "
        "Use refresh buttons to retry or continue with cache.",
        icon="⚠️",
    )


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

# ?? Global filters ?????????????????????????????????????????????????????????????

filter_action_global = str(st.session_state.get("filter_action_global", "?꾩껜"))
filter_regime_only_global = bool(st.session_state.get("filter_regime_only_global", False))

signals_filtered = list(signals)
if filter_regime_only_global:
    signals_filtered = [s for s in signals_filtered if s.macro_regime == current_regime]
if filter_action_global != "?꾩껜":
    signals_filtered = [s for s in signals_filtered if s.action == filter_action_global]


def _format_return_pct(returns: dict[str, float], period: str) -> str:
    value = returns.get(period, float("nan"))
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:+.1f}%"


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

# ?? Tabs Interface ?????????????????????????????????????????????????????????????

tab_decision, tab_evidence, tab_all_signals = st.tabs([
    "寃곕줎 (Decision)",
    "洹쇨굅 (Evidence)",
    "?좏샇 (Signals)",
])

with tab_decision:
    from src.ui.components import (
        format_action_label,
        render_action_summary,
        render_macro_tile,
        render_returns_heatmap,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    render_macro_tile(
        regime=current_regime,
        growth_val=growth_val,
        inflation_val=inflation_val,
        fx_change=fx_change,
        is_provisional=is_provisional,
        theme_mode=theme_mode,
    )

    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        _regime_label = "?꾩옱 援?㈃ (?뺤젙)" if regime_is_confirmed else "?꾩옱 援?㈃ (?좎젙)"
        st.metric(_regime_label, current_regime)
        if yield_curve_status == "Inverted":
            st.warning("?섏씡瑜?怨≪꽑 ??쟾 (援?퀬梨?Y < 湲곗?湲덈━)", icon="?좑툘")
    with status_col2:
        st.metric("?곗씠???곹깭 (媛寃?", price_status)
    with status_col3:
        st.metric("?곗씠???곹깭 (留ㅽ겕濡?", macro_status)

    st.divider()
    st.subheader("Action 遺꾪룷")
    render_action_summary(signals_filtered, theme_mode=theme_mode)

    st.divider()
    st.subheader("Top Picks")
    if top_pick_signals:
        top_rows = []
        for rank, signal in enumerate(top_pick_signals[:8], start=1):
            rs_div = _rs_divergence_pct(signal)
            top_rows.append(
                {
                    "?쒖쐞": rank,
                    "?뱁꽣": signal.sector_name + (" *" if signal.is_provisional else ""),
                    "?≪뀡": format_action_label(signal.action),
                    "RS 이탈률": f"{rs_div:+.2f}%" if not pd.isna(rs_div) else "N/A",
                    "1M": _format_return_pct(signal.returns, "1M"),
                    "3M": _format_return_pct(signal.returns, "3M"),
                    "?뚮┝": ", ".join(signal.alerts) if signal.alerts else "-",
                }
            )
        top_df = pd.DataFrame(top_rows)
        table_tokens = get_table_style_tokens(theme_mode)
        action_col = top_df.columns[2] if len(top_df.columns) > 2 else None

        def _style_top_rows(row: pd.Series) -> list[str]:
            row_idx = int(row.name) if isinstance(row.name, int) else 0
            row_bg = (
                table_tokens["row_bg_even"]
                if row_idx % 2 == 0
                else table_tokens["row_bg_odd"]
            )
            base = (
                f"background-color: {row_bg}; "
                f"color: {table_tokens['row_text']}; "
                f"border-bottom: 1px solid {table_tokens['grid']};"
            )
            styles = [base for _ in range(len(row))]
            if action_col and action_col in row.index:
                styles[row.index.get_loc(action_col)] = (
                    f"background-color: {row_bg}; "
                    f"color: {table_tokens['row_text']}; "
                    f"font-weight: 700; "
                    f"border-bottom: 1px solid {table_tokens['grid']};"
                )
            return styles

        top_styled = (
            top_df.style.apply(_style_top_rows, axis=1).set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", table_tokens["header_bg"]),
                            ("color", table_tokens["header_text"]),
                            ("font-weight", "700"),
                            ("border-bottom", f"1px solid {table_tokens['grid']}"),
                        ],
                    }
                ],
                overwrite=False,
            )
        )
        st.dataframe(top_styled, width='stretch', hide_index=True)
    else:
        st.info("湲濡쒕쾶 ?꾪꽣 議곌굔??留욌뒗 ?뱁꽣媛 ?놁뒿?덈떎.")

    st.divider()
    if signals_filtered:
        fig_heatmap = render_returns_heatmap(signals_filtered, theme_mode=theme_mode)
        st.plotly_chart(fig_heatmap, width='stretch')
    else:
        st.info("?섏씡瑜??덊듃留듭쓣 ?쒖떆???좏샇媛 ?놁뒿?덈떎.")

with tab_evidence:
    from src.ui.components import render_rs_momentum_bar, render_rs_scatter

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="app-summary-card">
<b>?듭떖 ?붿빟</b><br/>
RS ?곗젏?꾨뒗 ?뱁꽣???곷?媛뺣룄(RS)? RS ?대룞?됯퇏???숈떆??蹂댁뿬以띾땲??<br/>
?ㅻⅨ履쎌씪?섎줉 踰ㅼ튂留덊겕 ?鍮?媛뺥븯怨? ?媛곸꽑 ?꾨옒(RS > RS MA)?쇱닔濡?紐⑤찘? 媛??援ш컙?낅땲??
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("李⑦듃 ?댁꽍 ?곸꽭", expanded=False):
        st.markdown("""
**X異?(RS)** ???뱁꽣 醫낃? 첨 踰ㅼ튂留덊겕(KOSPI) 鍮꾩쑉. 媛믪씠 ?댁닔濡?踰ㅼ튂留덊겕 ?鍮??덈? 媛뺣룄媛 ?믪뒿?덈떎.

**Y異?(RS MA)** ??RS???대룞?됯퇏(湲곕낯 20??. RS??異붿꽭 ?섏????섑??낅땲??

**?먯꽑 ?媛곸꽑** ??RS = RS MA 湲곗??? ???좎쓣 湲곗??쇰줈 ?꽷룹븘?섍? ?듭떖 ?좏샇?낅땲??

| ?꾩튂 | ?섎? |
|------|------|
| ???媛곸꽑 ?꾨옒 (RS > RS MA) | RS媛 ?됯퇏??珥덇낵 ??紐⑤찘? **媛??以?* ??媛뺤꽭 ?좏샇 |
| ???媛곸꽑 ??(RS < RS MA) | RS媛 ?됯퇏 誘몃떖 ??紐⑤찘? **媛먯냽 以?* ???쎌꽭 ?좏샇 |
| ???ㅻⅨ履?| 踰ㅼ튂留덊겕 ?鍮?**媛뺥븳** ?뱁꽣 |
| ???쇱そ | 踰ㅼ튂留덊겕 ?鍮?**?쏀븳** ?뱁꽣 |

**???됱긽** ??Strong Buy (珥덈줉) ??Watch (?뚮옉) ??Hold (?뚯깋) ??Avoid (鍮④컯)
""")

    if signals_filtered:
        benchmark_missing = any(
            "Benchmark Missing" in getattr(s, "alerts", []) for s in signals_filtered
        )
        if benchmark_missing:
            st.warning("踰ㅼ튂留덊겕(KOSPI, 1001) ?곗씠???꾨씫?쇰줈 紐⑤찘? 李⑦듃瑜?怨꾩궛?????놁뒿?덈떎. ?쒖옣?곗씠??媛깆떊 ???ъ떆?꾪븯?몄슂.")
        else:
            is_mobile_client = _is_mobile_client()
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
                st.plotly_chart(fig_scatter, width='stretch')
            else:
                _, scatter_col_c, _ = st.columns([0.7, 3.6, 0.7])
                with scatter_col_c:
                    st.plotly_chart(fig_scatter, width='stretch')

            st.markdown("---")
            st.markdown(
                """
**RS ?댄깉??(RS Divergence)**

- **怨꾩궛??*: `(RS 첨 RS ?대룞?됯퇏 - 1) 횞 100`
- **?묒닔 (+)**: RS媛 ?대룞?됯퇏蹂대떎 ?꾩뿉 ?덉뼱 紐⑤찘???**媛??* 以?- **?뚯닔 (-)**: RS媛 ?대룞?됯퇏蹂대떎 ?꾨옒???덉뼱 紐⑤찘???**媛먯냽** 以?- **?댁꽍 ?ъ씤??*: ???곗젏?꾩쓽 ?媛곸꽑(RS = RS MA)?먯꽌 ?쇰쭏???댄깉?덈뒗吏 ?섏튂濡?蹂댁뿬以띾땲??
"""
            )
            fig_bar = render_rs_momentum_bar(signals_filtered, theme_mode=theme_mode)
            if fig_bar.data:
                st.plotly_chart(fig_bar, width='stretch')
            else:
                st.info("RS/RS MA ?곗씠?곌? 異⑸텇?섏? ?딆뒿?덈떎.")
    else:
        st.info("湲濡쒕쾶 ?꾪꽣 議곌굔??留욌뒗 ?좏샇媛 ?놁뒿?덈떎.")

with tab_all_signals:
    from src.ui.components import render_signal_table

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        f"?곸슜 ?꾪꽣: ?≪뀡={filter_action_global}, "
        f"?꾩옱 援?㈃留?蹂닿린={'ON' if filter_regime_only_global else 'OFF'}"
    )
    with st.expander("?곹빀/鍮꾩쟻???먯젙 湲곗?", expanded=False):
        st.markdown(
            """
- `?곹빀`? **?꾩옱 ?쒖젏???뺤젙 援?㈃** ??留ㅽ븨???뱁꽣?몄?(`macro_fit`)濡??먯젙?⑸땲??
- ?꾩옱 ?쒖젏 援?㈃? `confirmed_regime` (N媛쒖썡 ?곗냽 ?숈씪 援?㈃ ?뺤젙) 湲곗??낅땲?? 援?㈃???꾩쭅 ?뺤젙?섏? ?딆? 寃쎌슦 `(?좎젙)` ?쒖떆媛 ?섑??⑸땲??
- 留ㅽ븨 湲곗?? `config/sector_map.yml`??`regimes -> {援?㈃} -> sectors`?낅땲??
- ?꾩옱 援?㈃???ы븿?섏? ?딆? ?뱁꽣??`鍮꾩쟻???쇰줈 ?쒖떆?⑸땲??
- 李멸퀬: 理쒖쥌 `?≪뀡`(Strong Buy/Watch/Hold/Avoid)? `?곹빀/鍮꾩쟻????紐⑤찘? 議곌굔(RS, 異붿꽭)??寃고빀??怨꾩궛?⑸땲??
- `Indeterminate` 援?㈃?대㈃ ?대떦 ?쒖젏?먮뒗 留ㅽ븨 ?뱁꽣媛 ?놁뼱 ?꾩껜媛 `鍮꾩쟻???쇰줈 ?쒖떆?????덉뒿?덈떎.
"""
        )
    with st.expander("?뚮┝ 移댄뀒怨좊━ ?ㅻ챸", expanded=False):
        rsi_overbought = int(settings.get("rsi_overbought", 70))
        rsi_oversold = int(settings.get("rsi_oversold", 30))
        fx_shock_pct = float(settings.get("fx_shock_pct", 3.0))

        st.markdown(
            f"""
- **?쒖떆 洹쒖튃**: ?뚮┝???섎굹 ?댁긽?대㈃ ?쇳몴(,)濡??④퍡 ?쒖떆?섍퀬, ?놁쑝硫?`-`濡??쒖떆?⑸땲??
- **Overheat**: ?쇨컙 RSI(`rsi_d`)媛 `{rsi_overbought}` ?댁긽????異붽??⑸땲??
- **Oversold**: ?쇨컙 RSI(`rsi_d`)媛 `{rsi_oversold}` ?댄븯????異붽??⑸땲??
- **FX Shock**: `|USD/KRW 蹂?붿쑉| > {fx_shock_pct:.1f}%` ?닿퀬 ?섏텧 ?뱁꽣?대ŉ ?꾩옱 ?≪뀡??`Strong Buy`????異붽??섎ŉ, ?≪뀡? `Watch`濡?媛뺣벑?⑸땲??
- **Benchmark Missing**: 踰ㅼ튂留덊겕 媛寃??곗씠?곌? 鍮꾩뼱 ?덉쓣 ??紐⑤뱺 ?뱁꽣??異붽??⑸땲???≪뀡 `N/A`).
- **RS Data Insufficient**: ?뱀젙 ?뱁꽣??RS/RS MA 怨꾩궛??遺덇??ν븷 ???대떦 ?뱁꽣??異붽??⑸땲???≪뀡 `N/A`).
"""
        )
        st.caption(
            "李멸퀬: `FX Shock`???좏샇 怨꾩궛 ???꾨떖??理쒓렐 USD/KRW 蹂?붿쑉??湲곗??쇰줈 ?먯젙?⑸땲?? "
            "USD/KRW ?쒓퀎?댁씠 2媛?誘몃쭔?대㈃ ?대떦 ?뚯감?먯꽌??`FX Shock`媛 怨꾩궛?섏? ?딆뒿?덈떎."
        )

    render_signal_table(
        signals_filtered,
        current_regime=current_regime,
        theme_mode=theme_mode,
    )

# ?? Footer ?????????????????????????????????????????????????????????????????????

st.divider()
st.caption(
    f"湲곗??? {asof_date} | ?곗씠?? "
    f"{'KRX OpenAPI' if krx_provider_effective == 'OPENAPI' else 'pykrx'} (KRX), "
    "ECOS (?쒓뎅???, KOSIS (?듦퀎泥? | "
    f"援?㈃: {current_regime}"
)

