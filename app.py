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
from src.ui.styles import inject_css
from src.ui.data_status import is_sample_mode, get_button_states
from src.macro.series_utils import (
    build_enabled_ecos_config,
    build_enabled_kosis_config,
    extract_macro_series,
    to_plotly_time_index,
)

inject_css()

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

    price_status, sector_prices = _cached_sector_prices(
        st.session_state.get("asof_date_str", date.today().strftime("%Y%m%d")),
        str(settings.get("benchmark_code", "1001")),
        int(settings.get("price_years", 3)),
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

    signals = build_signal_table(
        sector_prices=sector_prices,
        benchmark_prices=bench_series,
        macro_result=macro_result,
        sector_map=sector_map,
        settings=settings,
    )

    return signals, macro_result, price_status, macro_status


# ── Session state defaults ─────────────────────────────────────────────────────

if "asof_date_str" not in st.session_state:
    st.session_state["asof_date_str"] = date.today().strftime("%Y%m%d")
if "epsilon" not in st.session_state:
    st.session_state["epsilon"] = float(settings.get("epsilon", 0))
if "filter_action_global" not in st.session_state:
    st.session_state["filter_action_global"] = "전체"
if "filter_regime_only_global" not in st.session_state:
    st.session_state["filter_regime_only_global"] = False


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 설정")
    st.divider()

    # Date picker
    asof_date = st.date_input(
        "기준일",
        value=date.today(),
        max_value=date.today(),
    )
    st.session_state["asof_date_str"] = asof_date.strftime("%Y%m%d")

    from src.ui.components import render_slider_with_input

    # Epsilon slider for regime sensitivity
    epsilon = render_slider_with_input(
        label="Epsilon (방향 민감도)",
        min_value=0.0,
        max_value=1.0,
        value=float(settings.get("epsilon", 0)),
        step=0.05,
        key="epsilon_ctrl",
        help="3MA 방향 판별 최소 변화량. 0 = 모든 변화 반영",
    )
    st.session_state["epsilon"] = epsilon

    # Momentum windows
    rs_ma_period = render_slider_with_input(
        label="RS MA 기간",
        min_value=5,
        max_value=60,
        value=int(settings.get("rs_ma_period", 20)),
        step=1,
        key="rs_ma_period_ctrl"
    )
    ma_fast = render_slider_with_input(
        label="빠른 MA",
        min_value=5,
        max_value=60,
        value=int(settings.get("ma_fast", 20)),
        step=1,
        key="ma_fast_ctrl"
    )
    ma_slow = render_slider_with_input(
        label="느린 MA",
        min_value=20,
        max_value=120,
        value=int(settings.get("ma_slow", 60)),
        step=1,
        key="ma_slow_ctrl"
    )
    price_years = render_slider_with_input(
        label="데이터 기간 (년)",
        min_value=1,
        max_value=5,
        value=int(settings.get("price_years", 3)),
        step=1,
        key="price_years_ctrl"
    )

    st.divider()

    # --- Load data to determine SAMPLE mode before rendering buttons ---
    macro_cache_token = _macro_cache_token()

    prices_parquet = "data/curated/sector_prices.parquet"
    macro_parquet = "data/curated/macro_monthly.parquet"

    # Compute button states — need data_status
    # We do a lightweight probe: if parquets exist → CACHED, else SAMPLE likely
    probe_price_status = "CACHED" if Path(prices_parquet).exists() else "SAMPLE"
    probe_macro_status = "CACHED" if Path(macro_parquet).exists() else "SAMPLE"
    probe_data_status = {"price": probe_price_status, "macro": probe_macro_status}
    btn_states = get_button_states(probe_data_status)

    refresh_market = st.button(
        "🔄 시장데이터 갱신",
        disabled=not btn_states["refresh_market"],
        use_container_width=True,
    )
    refresh_macro = st.button(
        "📈 매크로데이터 갱신",
        disabled=not btn_states["refresh_macro"],
        use_container_width=True,
    )
    recompute = st.button(
        "⚙️ 전체 재계산",
        disabled=not btn_states["recompute"],
        use_container_width=True,
        help="SAMPLE 데이터에서는 비활성화됩니다." if not btn_states["recompute"] else "",
    )

    st.divider()
    st.caption("Korea Sector Rotation Dashboard")


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

with st.sidebar:
    st.divider()
    st.subheader("🔎 글로벌 필터")
    st.selectbox(
        "액션 필터",
        options=["전체", "Strong Buy", "Watch", "Hold", "Avoid", "N/A"],
        key="filter_action_global",
    )
    st.checkbox(
        f"현재 국면 섹터만 보기 ({current_regime})",
        key="filter_regime_only_global",
    )

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
    "✅ 오늘의 결론 (Decision)",
    "📈 근거 분석 (Evidence)",
    "🔍 전체 신호 (Signals)",
])

with tab_decision:
    from src.ui.components import (
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
    render_action_summary(signals_filtered)

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
                    "액션": signal.action,
                    "RS 이탈도": f"{rs_div:+.2f}%" if not pd.isna(rs_div) else "N/A",
                    "1M": _format_return_pct(signal.returns, "1M"),
                    "3M": _format_return_pct(signal.returns, "3M"),
                    "알림": ", ".join(signal.alerts) if signal.alerts else "-",
                }
            )
        st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)
    else:
        st.info("글로벌 필터 조건에 맞는 섹터가 없습니다.")

    st.divider()
    if signals_filtered:
        fig_heatmap = render_returns_heatmap(signals_filtered)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("수익률 히트맵을 표시할 신호가 없습니다.")

with tab_evidence:
    from src.ui.components import render_rs_momentum_bar, render_rs_scatter

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📌 차트 읽는 법", expanded=False):
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
            fig_bar = render_rs_momentum_bar(signals_filtered)
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
    render_signal_table(signals_filtered, current_regime=current_regime)

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"기준일: {asof_date} | 데이터: pykrx (KRX), ECOS (한국은행), KOSIS (통계청) | "
    f"국면: {current_regime}"
)
