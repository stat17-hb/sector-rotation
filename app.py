"""
Korea Sector Rotation Dashboard
Streamlit SPA — app.py

Architecture:
- Three named @st.cache_data functions for sector prices, macro, and signals (R8).
- Each button clears only its own cache (R8 — no cross-cache pollution).
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

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Korea Sector Rotation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ────────────────────────────────────────────────────────────
from src.ui.styles import get_table_style_tokens, inject_css
from src.ui.data_status import is_sample_mode, get_button_states
from src.macro.series_utils import (
    build_enabled_ecos_config,
    build_enabled_kosis_config,
    extract_macro_series,
    to_plotly_time_index,
)

# ── Config loading ────────────────────────────────────────────────────────────


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

# ── Cache key helper (R8) ─────────────────────────────────────────────────────


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


@st.cache_data(ttl=600)
def _cached_api_preflight(timeout_sec: int = 3) -> dict:
    """Cached API endpoint reachability check (10 min TTL)."""
    from src.data_sources.preflight import run_api_preflight

    return run_api_preflight(timeout_sec=timeout_sec)


# ── Named cache functions (R8) ────────────────────────────────────────────────


@st.cache_data(ttl=CACHE_TTL)
def _cached_sector_prices(asof_date_str: str, benchmark_code: str, price_years: int):
    """Fetch or load sector prices. Keyed by asof_date + benchmark + price_years."""
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
def _cached_signals(prices_key: tuple, macro_key: tuple, params_hash: str, macro_cache_token: str):
    """Compute signals. Keyed by parquet file metadata + params hash."""
    from src.macro.regime import compute_regime_history
    from src.signals.matrix import build_signal_table

    price_years = int(st.session_state.get("price_years", settings.get("price_years", 3)))
    price_status, sector_prices = _cached_sector_prices(
        st.session_state.get("asof_date_str", date.today().strftime("%Y%m%d")),
        str(settings.get("benchmark_code", "1001")),
        price_years,
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
        inflation_series = extract_macro_series(
            macro_df=macro_df,
            macro_series_cfg=macro_series_cfg,
            alias="cpi_yoy",
        )

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
                    )
                    macro_result = to_plotly_time_index(macro_result)
                except Exception as exc:
                    logger.warning("compute_regime_history failed: %s", exc)

    if macro_result.empty:
        # Fallback: create minimal mock result for display
        macro_result = pd.DataFrame(
            {"growth_dir": ["Flat"], "inflation_dir": ["Flat"], "regime": ["Indeterminate"]},
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


# ── Session state defaults ─────────────────────────────────────────────────────

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
    st.session_state["filter_action_global"] = "전체"
if "filter_regime_only_global" not in st.session_state:
    st.session_state["filter_regime_only_global"] = False

theme_mode = str(st.session_state.get("theme_mode", "dark")).strip().lower()
if theme_mode not in {"dark", "light"}:
    theme_mode = "dark"
    st.session_state["theme_mode"] = theme_mode

inject_css(theme_mode)

# ── Sidebar ───────────────────────────────────────────────────────────────────

prices_parquet = "data/curated/sector_prices.parquet"
macro_parquet = "data/curated/macro_monthly.parquet"
macro_cache_token = _macro_cache_token()

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

    st.subheader("빠른 상태")
    use_light_theme = st.toggle(
        "라이트 테마",
        value=theme_mode == "light",
        help="해제 시 다크 테마가 적용됩니다.",
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
        st.metric("가격 상태", probe_price_status)
    with quick_col2:
        st.metric("매크로 상태", probe_macro_status)

    st.divider()
    st.subheader("글로벌 필터")
    st.selectbox(
        "액션 필터",
        options=["전체", "Strong Buy", "Watch", "Hold", "Avoid", "N/A"],
        key="filter_action_global",
    )
    st.checkbox(
        "현재 국면 섹터만 보기",
        key="filter_regime_only_global",
    )

    st.divider()
    with st.expander("모델 파라미터", expanded=False):
        with st.form("model_params_form"):
            slider_epsilon = st.slider(
                "Epsilon (방향 민감도)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state["epsilon"]),
                step=0.05,
                help="3MA 방향 판별 최소 변화량. 0 = 모든 변화 반영",
            )
            slider_rs_ma = st.slider(
                "RS MA 기간",
                min_value=5,
                max_value=60,
                value=int(st.session_state["rs_ma_period"]),
                step=1,
            )
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
            slider_price_years = st.slider(
                "데이터 기간 (년)",
                min_value=1,
                max_value=5,
                value=int(st.session_state["price_years"]),
                step=1,
            )

            with st.expander("고급 직접 입력", expanded=False):
                use_advanced_inputs = st.checkbox("고급 값으로 적용", value=False)
                adv_epsilon = st.number_input(
                    "Epsilon 직접 입력",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state["epsilon"]),
                    step=0.01,
                    format="%.2f",
                )
                adv_rs_ma = st.number_input(
                    "RS MA 직접 입력",
                    min_value=5,
                    max_value=60,
                    value=int(st.session_state["rs_ma_period"]),
                    step=1,
                )
                adv_ma_fast = st.number_input(
                    "빠른 MA 직접 입력",
                    min_value=5,
                    max_value=60,
                    value=int(st.session_state["ma_fast"]),
                    step=1,
                )
                adv_ma_slow = st.number_input(
                    "느린 MA 직접 입력",
                    min_value=20,
                    max_value=120,
                    value=int(st.session_state["ma_slow"]),
                    step=1,
                )
                adv_price_years = st.number_input(
                    "데이터 기간 직접 입력",
                    min_value=1,
                    max_value=5,
                    value=int(st.session_state["price_years"]),
                    step=1,
                )

            apply_params = st.form_submit_button("적용", use_container_width=True)

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
    st.subheader("데이터 작업")
    refresh_market = st.button(
        "시장데이터 갱신",
        disabled=not btn_states["refresh_market"],
        use_container_width=True,
    )
    refresh_macro = st.button(
        "매크로데이터 갱신",
        disabled=not btn_states["refresh_macro"],
        use_container_width=True,
    )
    recompute = st.button(
        "전체 재계산",
        disabled=not btn_states["recompute"],
        use_container_width=True,
        help="SAMPLE 데이터에서는 비활성화됩니다." if not btn_states["recompute"] else "",
    )

    st.caption("Korea Sector Rotation Dashboard")

rs_ma_period = int(st.session_state.get("rs_ma_period", settings.get("rs_ma_period", 20)))
ma_fast = int(st.session_state.get("ma_fast", settings.get("ma_fast", 20)))
ma_slow = int(st.session_state.get("ma_slow", settings.get("ma_slow", 60)))
price_years = int(st.session_state.get("price_years", settings.get("price_years", 3)))


# ── Button handlers — each clears only its own cache (R8) ────────────────────

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


# ── Load data via cache functions ─────────────────────────────────────────────

with st.spinner("데이터 로딩 중..."):
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
            prices_key, macro_key, params_hash, macro_cache_token
        )

        data_status = {"price": price_status, "macro": macro_status}
    except Exception as exc:
        logger.error("Data load failed: %s", exc)
        signals = []
        macro_result = pd.DataFrame(
            {"growth_dir": ["Flat"], "inflation_dir": ["Flat"], "regime": ["Indeterminate"]},
            index=pd.DatetimeIndex([date.today()]),
        )
        price_status = "SAMPLE"
        macro_status = "SAMPLE"
        data_status = {"price": price_status, "macro": macro_status}


# ── SAMPLE mode warning (R9) ──────────────────────────────────────────────────

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

if is_sample_mode(data_status):
    st.error(
        "⚠️ **SAMPLE 데이터 모드**: 실제 시장 데이터를 불러올 수 없어 합성 데이터를 표시합니다. "
        "API 키 설정 또는 네트워크를 확인하세요. "
        "'시장데이터 갱신' 또는 '매크로데이터 갱신' 버튼으로 다시 시도하세요.",
        icon="⚠️",
    )


# ── Current regime ─────────────────────────────────────────────────────────────

current_regime = "Indeterminate"
if not macro_result.empty and "regime" in macro_result.columns:
    current_regime = str(macro_result["regime"].iloc[-1])

is_provisional = any(
    getattr(s, "is_provisional", False) for s in signals
)

# Extract latest values for macro tile display
_, macro_df = _cached_macro(macro_cache_token)  # cached — no extra API call
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

# ── Global filters ─────────────────────────────────────────────────────────────

filter_action_global = str(st.session_state.get("filter_action_global", "전체"))
filter_regime_only_global = bool(st.session_state.get("filter_regime_only_global", False))

signals_filtered = list(signals)
if filter_regime_only_global:
    signals_filtered = [s for s in signals_filtered if s.macro_regime == current_regime]
if filter_action_global != "전체":
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

# ── Tabs Interface ─────────────────────────────────────────────────────────────

tab_decision, tab_evidence, tab_all_signals = st.tabs([
    "결론 (Decision)",
    "근거 (Evidence)",
    "신호 (Signals)",
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
        st.metric("현재 국면", current_regime)
    with status_col2:
        st.metric("데이터 상태 (가격)", price_status)
    with status_col3:
        st.metric("데이터 상태 (매크로)", macro_status)

    st.divider()
    st.subheader("Action 분포")
    render_action_summary(signals_filtered, theme_mode=theme_mode)

    st.divider()
    st.subheader("Top Picks")
    if top_pick_signals:
        top_rows = []
        for rank, signal in enumerate(top_pick_signals[:8], start=1):
            rs_div = _rs_divergence_pct(signal)
            top_rows.append(
                {
                    "순위": rank,
                    "섹터": signal.sector_name + (" *" if signal.is_provisional else ""),
                    "액션": format_action_label(signal.action),
                    "RS 이탈도": f"{rs_div:+.2f}%" if not pd.isna(rs_div) else "N/A",
                    "1M": _format_return_pct(signal.returns, "1M"),
                    "3M": _format_return_pct(signal.returns, "3M"),
                    "알림": ", ".join(signal.alerts) if signal.alerts else "-",
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
        st.dataframe(top_styled, use_container_width=True, hide_index=True)
    else:
        st.info("글로벌 필터 조건에 맞는 섹터가 없습니다.")

    st.divider()
    if signals_filtered:
        fig_heatmap = render_returns_heatmap(signals_filtered, theme_mode=theme_mode)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("수익률 히트맵을 표시할 신호가 없습니다.")

with tab_evidence:
    from src.ui.components import render_rs_momentum_bar, render_rs_scatter

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="app-summary-card">
<b>핵심 요약</b><br/>
RS 산점도는 섹터의 상대강도(RS)와 RS 이동평균을 동시에 보여줍니다.<br/>
오른쪽일수록 벤치마크 대비 강하고, 대각선 아래(RS > RS MA)일수록 모멘텀 가속 구간입니다.
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("차트 해석 상세", expanded=False):
        st.markdown("""
**X축 (RS)** — 섹터 종가 ÷ 벤치마크(KOSPI) 비율. 값이 클수록 벤치마크 대비 절대 강도가 높습니다.

**Y축 (RS MA)** — RS의 이동평균(기본 20일). RS의 추세 수준을 나타냅니다.

**점선 대각선** — RS = RS MA 기준선. 이 선을 기준으로 위·아래가 핵심 신호입니다.

| 위치 | 의미 |
|------|------|
| ▼ 대각선 아래 (RS > RS MA) | RS가 평균을 초과 → 모멘텀 **가속 중** → 강세 신호 |
| ▲ 대각선 위 (RS < RS MA) | RS가 평균 미달 → 모멘텀 **감속 중** → 약세 신호 |
| → 오른쪽 | 벤치마크 대비 **강한** 섹터 |
| ← 왼쪽 | 벤치마크 대비 **약한** 섹터 |

**점 색상** — Strong Buy (초록) › Watch (파랑) › Hold (회색) › Avoid (빨강)
""")

    if signals_filtered:
        benchmark_missing = any(
            "Benchmark Missing" in getattr(s, "alerts", []) for s in signals_filtered
        )
        if benchmark_missing:
            st.warning("벤치마크(KOSPI, 1001) 데이터 누락으로 모멘텀 차트를 계산할 수 없습니다. 시장데이터 갱신 후 재시도하세요.")
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
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                _, scatter_col_c, _ = st.columns([0.7, 3.6, 0.7])
                with scatter_col_c:
                    st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("---")
            st.markdown(
                """
**RS 이탈도 (RS Divergence)**

- **계산식**: `(RS ÷ RS 이동평균 - 1) × 100`
- **양수 (+)**: RS가 이동평균보다 위에 있어 모멘텀이 **가속** 중
- **음수 (-)**: RS가 이동평균보다 아래에 있어 모멘텀이 **감속** 중
- **해석 포인트**: 위 산점도의 대각선(RS = RS MA)에서 얼마나 이탈했는지 수치로 보여줍니다.
"""
            )
            fig_bar = render_rs_momentum_bar(signals_filtered, theme_mode=theme_mode)
            if fig_bar.data:
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("RS/RS MA 데이터가 충분하지 않습니다.")
    else:
        st.info("글로벌 필터 조건에 맞는 신호가 없습니다.")

with tab_all_signals:
    from src.ui.components import render_signal_table

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        f"적용 필터: 액션={filter_action_global}, "
        f"현재 국면만 보기={'ON' if filter_regime_only_global else 'OFF'}"
    )
    with st.expander("적합/비적합 판정 기준", expanded=False):
        st.markdown(
            """
- `적합`은 **현재 시점의 국면(최신 매크로 판정)** 에 매핑된 섹터인지(`macro_fit`)로 판정합니다.
- 현재 시점 국면은 매크로 결과의 최신 행(`macro_result["regime"].iloc[-1]`)을 사용합니다.
- 매핑 기준은 `config/sector_map.yml`의 `regimes -> {국면} -> sectors`입니다.
- 현재 국면에 포함되지 않은 섹터는 `비적합`으로 표시됩니다.
- 참고: 최종 `액션`(Strong Buy/Watch/Hold/Avoid)은 `적합/비적합`에 모멘텀 조건(RS, 추세)을 결합해 계산됩니다.
- `Indeterminate` 국면이면 해당 시점에는 매핑 섹터가 없어 전체가 `비적합`으로 표시될 수 있습니다.
"""
        )
    with st.expander("알림 카테고리 설명", expanded=False):
        rsi_overbought = int(settings.get("rsi_overbought", 70))
        rsi_oversold = int(settings.get("rsi_oversold", 30))
        fx_shock_pct = float(settings.get("fx_shock_pct", 3.0))

        st.markdown(
            f"""
- **표시 규칙**: 알림이 하나 이상이면 쉼표(,)로 함께 표시되고, 없으면 `-`로 표시됩니다.
- **Overheat**: 일간 RSI(`rsi_d`)가 `{rsi_overbought}` 이상일 때 추가됩니다.
- **Oversold**: 일간 RSI(`rsi_d`)가 `{rsi_oversold}` 이하일 때 추가됩니다.
- **FX Shock**: `|USD/KRW 변화율| > {fx_shock_pct:.1f}%` 이고 수출 섹터이며 현재 액션이 `Strong Buy`일 때 추가되며, 액션은 `Watch`로 강등됩니다.
- **Benchmark Missing**: 벤치마크 가격 데이터가 비어 있을 때 모든 섹터에 추가됩니다(액션 `N/A`).
- **RS Data Insufficient**: 특정 섹터의 RS/RS MA 계산이 불가능할 때 해당 섹터에 추가됩니다(액션 `N/A`).
"""
        )
        st.caption(
            "참고: `FX Shock`는 신호 계산 시 전달된 최근 USD/KRW 변화율을 기준으로 판정됩니다. "
            "USD/KRW 시계열이 2개 미만이면 해당 회차에서는 `FX Shock`가 계산되지 않습니다."
        )

    render_signal_table(
        signals_filtered,
        current_regime=current_regime,
        theme_mode=theme_mode,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"기준일: {asof_date} | 데이터: pykrx (KRX), ECOS (한국은행), KOSIS (통계청) | "
    f"국면: {current_regime}"
)
