"""Dashboard UI entrypoints extracted from app.py."""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from config.theme import set_theme_mode
from src.dashboard.analysis import _extract_heatmap_selection
from src.ui.components import (
    DEFAULT_UI_LOCALE,
    FLOW_PROFILE_IDS,
    HEATMAP_PALETTE_OPTIONS,
    build_sector_strength_heatmap,
    describe_signal_decision,
    get_action_filter_label,
    format_cycle_phase_label,
    get_flow_profile_label,
    get_flow_state_label,
    format_heatmap_palette_label,
    get_ui_text,
    normalize_range_preset,
    render_action_summary,
    render_cycle_timeline_panel,
    render_decision_hero,
    render_investor_decision_boards,
    render_investor_flow_summary,
    render_panel_header,
    render_returns_heatmap,
    render_rs_momentum_bar,
    render_rs_scatter,
    render_sector_detail_panel,
    render_signal_table,
    render_status_card_row,
    render_top_bar_filters,
    render_top_picks_table,
)


def render_sidebar_controls(
    *,
    market_id: str,
    ui_labels: dict[str, str],
    theme_mode: str,
    analysis_heatmap_palette: str,
    probe_price_status: str,
    probe_macro_status: str,
    btn_states: dict[str, bool],
    asof_default: date,
    probe_investor_flow_status: str = "SAMPLE",
    flow_profile: str = "foreign_lead",
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> tuple[str, date, str, bool, bool, bool, bool]:
    market_options = ["KR", "US"]
    selected_market = st.radio(
        ui_labels.get("market_selector", "시장"),
        options=market_options,
        index=market_options.index(str(market_id or "KR").strip().upper() or "KR"),
        horizontal=True,
    )
    st.title(ui_labels.get("sidebar_title", "Sector Rotation"))
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
        "히트맵 색상",
        options=list(HEATMAP_PALETTE_OPTIONS),
        index=list(HEATMAP_PALETTE_OPTIONS).index(analysis_heatmap_palette),
        format_func=lambda value: format_heatmap_palette_label(value, locale=ui_locale),
        help="월간 분석 히트맵의 발산형 색상을 변경합니다.",
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
    with st.popover("⚙️ 고급 설정", use_container_width=True):
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
    if market_id == "KR":
        st.subheader(get_ui_text("flow_profile_label", ui_locale))
        selected_flow_profile = st.selectbox(
            get_ui_text("flow_profile_label", ui_locale),
            options=list(FLOW_PROFILE_IDS),
            index=list(FLOW_PROFILE_IDS).index(str(flow_profile or "foreign_lead")),
            format_func=lambda value: get_flow_profile_label(value, ui_locale),
            help=get_ui_text("flow_sidebar_caption", ui_locale),
        )
    else:
        selected_flow_profile = str(flow_profile or "foreign_lead")

    st.divider()
    st.subheader("데이터 작업")
    if market_id == "KR":
        st.caption(f"시장: {probe_price_status} · 매크로: {probe_macro_status} · 수급: {probe_investor_flow_status}")
    else:
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
    refresh_flow = False
    if market_id == "KR":
        refresh_flow = st.button(
            get_ui_text("flow_refresh_button", ui_locale),
            width="stretch",
        )
    recompute = st.button(
        "전체 재계산",
        disabled=not btn_states["recompute"],
        width="stretch",
        help="SAMPLE 데이터에서는 비활성화됩니다." if not btn_states["recompute"] else "",
    )

    st.caption(ui_labels.get("sidebar_title", "섹터 로테이션"))
    return selected_market, asof_date, selected_flow_profile, refresh_market, refresh_macro, refresh_flow, recompute


def render_analysis_canvas(
    *,
    heatmap_return_display: pd.DataFrame,
    heatmap_strength_display: pd.DataFrame,
    selected_cycle_phase: str,
    theme_mode: str,
    analysis_heatmap_palette: str,
    visible_segments: list[dict[str, Any]],
    current_regime: str,
    analysis_prices_phase: pd.DataFrame,
    analysis_prices: pd.DataFrame,
    sector_columns: list[str],
    benchmark_label: str,
    analysis_max_date: date,
    analysis_min_date: date,
    build_sector_detail_figure,
    resolve_range_from_preset,
    signal_lookup: dict[str, Any] | None = None,
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> None:
    with st.container(border=True):
        render_panel_header(
            eyebrow="섹터 비교",
            title="월간 섹터 수익률",
            description="절대 월간 섹터 수익률을 먼저 확인하세요. 셀 클릭 시 해당 섹터와 월이 아래 상세 패널에 고정됩니다.",
            badge=format_cycle_phase_label(selected_cycle_phase, locale=ui_locale),
        )
        heatmap_fig = build_sector_strength_heatmap(
            heatmap_return_display,
            selected_sector=str(st.session_state.get("selected_sector", "")),
            selected_month=str(st.session_state.get("selected_month", "")),
            theme_mode=theme_mode,
            palette=analysis_heatmap_palette,
            title="월간 섹터 수익률",
            empty_message="활성 필터에 해당하는 월간 섹터 수익률 데이터가 없습니다.",
            helper_metric_label="월간 수익률",
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
            eyebrow="상대강도",
            title=f"월간 섹터 강도 vs {benchmark_label}",
            description=f"각 셀은 {benchmark_label} 대비 월간 초과 수익률입니다. 동일한 섹터/월 연동 선택을 사용합니다.",
            badge=format_cycle_phase_label(selected_cycle_phase, locale=ui_locale),
        )
        strength_fig = build_sector_strength_heatmap(
            heatmap_strength_display,
            selected_sector=str(st.session_state.get("selected_sector", "")),
            selected_month=str(st.session_state.get("selected_month", "")),
            theme_mode=theme_mode,
            palette=analysis_heatmap_palette,
            title=f"월간 섹터 강도 vs {benchmark_label}",
            empty_message=f"활성 필터에 해당하는 {benchmark_label} 대비 월간 섹터 강도 데이터가 없습니다.",
            helper_metric_label="월간 초과 수익률",
            hover_value_suffix=f"%p vs {benchmark_label}",
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
            eyebrow="사이클 맥락",
            title="사이클 타임라인 맥락",
            description="초기/후기 사이클 필터를 사용해 히트맵과 상세 차트를 특정 매크로 국면으로 압축하세요.",
            badge=current_regime,
        )
        chosen_cycle_phase = render_cycle_timeline_panel(
            segments=visible_segments,
            selected_cycle_phase=selected_cycle_phase,
            theme_mode=theme_mode,
            locale=ui_locale,
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
        detail_badge = selected_sector or "선택 없음"
        detail_summary = None
        if signal_lookup and selected_sector in signal_lookup:
            detail_summary = describe_signal_decision(
                signal_lookup[selected_sector],
                st.session_state.get("held_sectors", []),
                locale=ui_locale,
            )
        render_panel_header(
            eyebrow="연동 상세",
            title="선택 섹터 상세 추적",
            description="왼쪽에서 섹터 순위를 확인하고, 선택 섹터를 벤치마크 및 상위 섹터와 비교하세요.",
            badge=detail_badge,
        )
        chosen_sector, chosen_preset = render_sector_detail_panel(
            ranking_rows=ranking_rows,
            detail_figure=detail_figure,
            selected_sector=selected_sector,
            selected_range_preset=normalize_range_preset(st.session_state.get("selected_range_preset", "1Y")),
            detail_summary=detail_summary,
            locale=ui_locale,
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


def render_decision_first_sections(
    *,
    current_regime: str,
    regime_is_confirmed: bool,
    growth_val: float | None,
    inflation_val: float | None,
    fx_change: float | None,
    fx_label: str,
    is_provisional: bool,
    theme_mode: str,
    price_status: str,
    macro_status: str,
    investor_flow_status: str,
    investor_flow_fresh: bool,
    investor_flow_profile: str,
    yield_curve_status: str | None,
    signals: list[Any],
    held_sector_options: list[str],
    action_options: list[str],
    is_mobile_client: bool,
    analysis_canvas_kwargs: dict[str, Any],
    market_id: str = "KR",
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> tuple[list[str], str, bool, str, bool]:
    """Render the decision-first main-page stack and return the active filters."""
    render_decision_hero(
        regime=current_regime,
        regime_is_confirmed=regime_is_confirmed,
        growth_val=growth_val,
        inflation_val=inflation_val,
        fx_change=fx_change,
        fx_label=fx_label,
        is_provisional=is_provisional,
        theme_mode=theme_mode,
        locale=ui_locale,
    )
    render_status_card_row(
        current_regime=current_regime,
        regime_is_confirmed=regime_is_confirmed,
        price_status=price_status,
        macro_status=macro_status,
        yield_curve_status=yield_curve_status,
        locale=ui_locale,
    )
    if str(market_id).strip().upper() == "KR" and (
        investor_flow_status != "SAMPLE" or any(
        str(getattr(signal, "flow_state", "unavailable")) != "unavailable" for signal in signals
        )
    ):
        render_investor_flow_summary(
            signals=signals,
            investor_flow_status=investor_flow_status,
            investor_flow_fresh=investor_flow_fresh,
            investor_flow_profile=investor_flow_profile,
            locale=ui_locale,
        )
    held_sectors = render_investor_decision_boards(
        signals=signals,
        held_sector_options=held_sector_options,
        locale=ui_locale,
    )
    filter_action, filter_regime_only, position_mode, show_alerted_only = render_top_bar_filters(
        current_regime=current_regime,
        action_options=action_options,
        is_mobile=is_mobile_client,
        locale=ui_locale,
    )
    render_analysis_canvas(**analysis_canvas_kwargs)
    return held_sectors, filter_action, filter_regime_only, position_mode, show_alerted_only


def render_summary_tab(
    *,
    tab,
    current_regime: str,
    regime_is_confirmed: bool,
    growth_val: float | None,
    inflation_val: float | None,
    fx_change: float | None,
    fx_label: str,
    is_provisional: bool,
    theme_mode: str,
    price_status: str,
    macro_status: str,
    yield_curve_status: str | None,
    top_pick_signals: list[Any],
    signals_filtered: list[Any],
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> None:
    with tab:
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
            fx_label=fx_label,
            is_provisional=is_provisional,
            theme_mode=theme_mode,
            locale=ui_locale,
        )
        render_status_card_row(
            current_regime=current_regime,
            regime_is_confirmed=regime_is_confirmed,
            price_status=price_status,
            macro_status=macro_status,
            yield_curve_status=yield_curve_status,
            locale=ui_locale,
        )

        with st.container(border=True):
            render_panel_header(
                eyebrow="우선순위 보드",
                title="상위 추천",
                description="활성 필터 적용 후 가장 높은 순위의 섹터입니다.",
                badge=f"{min(5, len(top_pick_signals))}개 표시",
            )
            render_top_picks_table(
                top_pick_signals,
                held_sectors=st.session_state.get("held_sectors", []),
                limit=5,
                locale=ui_locale,
            )

        st.divider()
        st.subheader("액션 분포")
        with st.container(border=True):
            render_panel_header(
                eyebrow="폭",
                title="액션 분포",
                description="필터 적용 후 유니버스가 액션 사다리에 걸쳐 어떻게 분포하는지 확인하세요.",
            )
            render_action_summary(signals_filtered, theme_mode=theme_mode, locale=ui_locale)


def render_charts_tab(
    *,
    tab,
    signals_filtered: list[Any],
    theme_mode: str,
    is_mobile_client: bool,
) -> None:
    with tab:
        from src.ui.components import (
            render_returns_heatmap,
            render_rs_momentum_bar,
            render_rs_scatter,
        )

        render_panel_header(
            eyebrow="모멘텀 맵",
            title="RS 산점도 및 모멘텀 막대",
            description="상대강도와 RS 이격도 패널은 동일한 시각적 구조를 사용해 비교가 쉽습니다.",
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
                st.markdown(
                    '<div class="empty-state-card">'
                    '<h4>벤치마크 데이터 누락</h4>'
                    '<p>KOSPI(1001) 가격 데이터 누락으로 RS 산점도를 계산할 수 없습니다.<br>시장데이터 갱신 후 다시 시도하세요.</p>'
                    '</div>',
                    unsafe_allow_html=True,
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
                eyebrow="횡단면",
                title="수익률 히트맵",
                description="대시보드 전체에서 사용하는 동일한 패널 구조로 복수 기간 섹터 수익률을 비교하세요.",
            )
            fig_heatmap = render_returns_heatmap(signals_filtered, theme_mode=theme_mode)
            st.plotly_chart(fig_heatmap, width="stretch")
        else:
            st.markdown(
                '<div class="empty-state-card">'
                '<h4>표시할 데이터 없음</h4>'
                '<p>글로벌 필터 조건에 맞는 신호가 없습니다.<br>선택된 액션 및 국면 필터를 확인해 주세요.</p>'
                '</div>',
                unsafe_allow_html=True,
            )


def render_all_signals_tab(
    *,
    tab,
    signals: list[Any],
    filter_action_global: str,
    filter_regime_only_global: bool,
    current_regime: str,
    held_sectors: list[str],
    position_mode: str,
    show_alerted_only: bool,
    theme_mode: str,
    settings: dict[str, Any],
    fx_label: str,
    etf_map: dict[str, list] | None = None,
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> None:
    with tab:
        render_panel_header(
            eyebrow="전체 테이블",
            title="전체 섹터 신호",
            description="요약 패널과 동일한 구조와 필터 피드백을 사용하는 Streamlit 그리드입니다.",
        )
        from src.ui.components import render_signal_table

        st.caption(
            f"적용 필터: 액션={get_action_filter_label(filter_action_global, ui_locale)}, "
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
            held_sectors=held_sectors,
            position_mode=position_mode,
            show_alerted_only=show_alerted_only,
            theme_mode=theme_mode,
            etf_map=etf_map,
            locale=ui_locale,
        )


def render_screening_tab(
    *,
    tab,
    signals: list[Any],
    settings: dict[str, Any],
    benchmark_code: str = "1001",
) -> None:
    """Render the 종목 스크리닝 tab: constituent stocks of Strong Buy sectors."""
    from src.data_sources.krx_stock_screening import load_screened_stocks

    with tab:
        render_panel_header(
            eyebrow="종목 스크리닝",
            title="Strong Buy 섹터 구성종목",
            description="현재 Strong Buy 섹터의 구성종목을 RS·RSI·SMA 기준으로 필터링한 매수 후보 리스트",
        )

        # Derive Strong Buy sectors from current signals
        strong_buy_sectors = [
            {"code": sig.index_code, "name": sig.sector_name}
            for sig in signals
            if getattr(sig, "action", "") == "Strong Buy"
        ]

        if not strong_buy_sectors:
            st.markdown(
                '<div class="empty-state-card">'
                '<h4>투자 유니버스 대기 중</h4>'
                '<p>현재 <b>Strong Buy</b> 섹터가 없습니다.<br>매크로 국면이 확정되거나 필터 조건이 완화되면 종목 스크리닝이 활성화됩니다.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        sector_labels = ", ".join(s["name"] for s in strong_buy_sectors)
        st.caption(f"대상 섹터: **{sector_labels}**")

        col_refresh, col_filter = st.columns([1, 3])
        with col_refresh:
            force_refresh = st.button("데이터 갱신", key="screening_refresh")
        with col_filter:
            show_momentum_only = st.toggle(
                "모멘텀 통과 종목만", value=True,
                help="RS > RS_MA AND SMA20 > SMA60 조건을 모두 충족하는 종목만 표시",
            )

        with st.spinner("구성종목 로딩 중..."):
            status, rows = load_screened_stocks(
                strong_buy_sectors=strong_buy_sectors,
                benchmark_code=benchmark_code,
                settings=settings,
                force_refresh=force_refresh,
            )

        if status == "UNAVAILABLE" or not rows:
            st.markdown(
                '<div class="empty-state-card">'
                '<h4>데이터 조회 불가</h4>'
                '<p>구성종목 데이터를 가져올 수 없습니다.<br>주말·공휴일 또는 서비스 API 점검 중일 수 있습니다.<br>평일 장중/장후에 <b>데이터 갱신</b>을 눌러주세요.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        status_label = {"LIVE": "실시간", "CACHED": "캐시(24h)"}
        st.caption(f"데이터 상태: **{status_label.get(status, status)}** | 총 {len(rows)}개 종목")

        if show_momentum_only:
            rows = [r for r in rows if r.get("momentum_ok")]

        if not rows:
            st.markdown(
                '<div class="empty-state-card">'
                '<h4>모멘텀 충족 종목 없음</h4>'
                '<p>설정된 모멘텀 조건(RS 상승 + SMA 추세 양호)을 충족하는 종목이 현재 없습니다.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        df = pd.DataFrame([
            {
                "종목코드": r["ticker"],
                "종목명": r["name"],
                "섹터": r["sector_name"],
                "RS": r["rs"],
                "RSI": r["rsi"],
                "RS↑": r["rs_strong"],
                "추세↑": r["trend_ok"],
                "1M(%)": r["ret_1m"],
                "3M(%)": r["ret_3m"],
                "알림": r["alerts"],
            }
            for r in rows
        ])

        def _color_momentum(val):
            if not isinstance(val, (int, float)) or pd.isna(val):
                return ""
            if val > 0:
                return "color: #ff4b4b; font-weight: 600;"
            elif val < 0:
                return "color: #2b7af0; font-weight: 600;"
            return ""

        styled_df = df.style.map(_color_momentum, subset=["1M(%)", "3M(%)"])

        st.dataframe(
            styled_df,
            width="stretch",
            hide_index=True,
            height=min(700, 76 + len(df) * 35),
            column_config={
                "종목코드": st.column_config.TextColumn("종목코드", width="small"),
                "종목명": st.column_config.TextColumn("종목명", width="medium"),
                "섹터": st.column_config.TextColumn("섹터", width="medium"),
                "RS": st.column_config.NumberColumn("RS", format="%.4f"),
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                "RS↑": st.column_config.CheckboxColumn("RS↑", width="small"),
                "추세↑": st.column_config.CheckboxColumn("추세↑", width="small"),
                "1M(%)": st.column_config.NumberColumn("1M(%)", format="%.1f%%"),
                "3M(%)": st.column_config.NumberColumn("3M(%)", format="%.1f%%"),
                "알림": st.column_config.TextColumn("알림", width="small"),
            },
        )

        # CSV download
        csv = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="CSV 다운로드",
            data=csv.encode("utf-8-sig"),
            file_name=f"screening_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


def render_investor_flow_tab(
    *,
    tab,
    signals: list[Any],
    investor_flow_frame: pd.DataFrame,
    investor_flow_status: str,
    investor_flow_fresh: bool,
    investor_flow_profile: str,
    investor_flow_detail: dict[str, Any] | None = None,
    market_id: str = "KR",
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> None:
    with tab:
        normalized_market = str(market_id).strip().upper() or "KR"
        if normalized_market == "US":
            state_labels = {
                "elevated": "활동 확대",
                "normal": "보통",
                "subdued": "활동 둔화",
                "unavailable": "사용 불가",
            }
            render_panel_header(
                eyebrow="Flow Proxies",
                title="US Flow Proxies",
                description="공개 ETF 데이터 기반 프록시입니다. 한국식 투자자별 수급과 1:1 대응이 아니라, 섹터 ETF 자금/거래 압력 확인 레이어로 보세요.",
                badge=investor_flow_status,
            )
            st.info("이 탭은 `ETF wrapper-flow proxy`를 보여 줍니다. participant-segmented 현물 순매수 데이터는 아닙니다.")
            if investor_flow_frame.empty:
                st.info("US flow proxy 데이터를 불러오지 못했습니다.")
                return

            latest_snapshot = investor_flow_frame.sort_values(
                by=["activity_zscore", "dollar_volume"],
                ascending=[False, False],
            ).copy()
            latest_snapshot["state_label"] = latest_snapshot["activity_state"].astype(str).map(
                lambda value: state_labels.get(str(value), str(value))
            )
            st.dataframe(
                latest_snapshot[
                    [
                        "sector_name",
                        "state_label",
                        "activity_zscore",
                        "dollar_volume",
                        "dollar_volume_short_mean",
                        "dollar_volume_long_mean",
                        "shares_outstanding",
                        "assets_under_management",
                        "nav",
                        "net_cash_amount",
                    ]
                ].rename(
                    columns={
                        "sector_name": "Sector",
                        "state_label": "Activity state",
                        "activity_zscore": "Activity z-score",
                        "dollar_volume": "Latest $Vol",
                        "dollar_volume_short_mean": "5D avg $Vol",
                        "dollar_volume_long_mean": "20D avg $Vol",
                        "shares_outstanding": "Shares Out",
                        "assets_under_management": "AUM",
                        "nav": "NAV",
                        "net_cash_amount": "Net Cash",
                    }
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "Sector": st.column_config.TextColumn("Sector", width="medium"),
                    "Activity state": st.column_config.TextColumn("Activity state", width="small"),
                    "Activity z-score": st.column_config.NumberColumn("Activity z-score", format="%.2f"),
                    "Latest $Vol": st.column_config.NumberColumn("Latest $Vol", format="%.0f"),
                    "5D avg $Vol": st.column_config.NumberColumn("5D avg $Vol", format="%.0f"),
                    "20D avg $Vol": st.column_config.NumberColumn("20D avg $Vol", format="%.0f"),
                    "Shares Out": st.column_config.NumberColumn("Shares Out", format="%.0f"),
                    "AUM": st.column_config.NumberColumn("AUM", format="%.0f"),
                    "NAV": st.column_config.NumberColumn("NAV", format="%.2f"),
                    "Net Cash": st.column_config.NumberColumn("Net Cash", format="%.0f"),
                },
            )
            latest_date = investor_flow_frame.index.max()
            ref_date_str = pd.Timestamp(latest_date).strftime("%Y-%m-%d")
            source_label = {"LIVE": "실시간", "CACHED": "캐시", "SAMPLE": "샘플"}.get(
                str(investor_flow_status).upper(), investor_flow_status
            )
            st.caption(f"데이터 기준일: {ref_date_str} · 소스: {source_label}")
            missing_tickers = [str(code) for code in investor_flow_frame.attrs.get("missing_tickers", []) if str(code).strip()]
            if missing_tickers:
                preview = ", ".join(missing_tickers[:4])
                suffix = "" if len(missing_tickers) <= 4 else ", ..."
                st.caption(f"누락 섹터: {preview}{suffix}")
            if not investor_flow_fresh:
                st.caption("최근 영업일까지 완전 커버된 값이 아닐 수 있습니다.")

            ownership_context = dict((investor_flow_detail or {}).get("ownership_context") or {})
            ici_context = dict(ownership_context.get("ici_weekly_flows") or {})
            ici_table = ici_context.get("table")
            if isinstance(ici_table, pd.DataFrame) and not ici_table.empty:
                render_panel_header(
                    eyebrow="ICI",
                    title="Weekly ETF Net Issuance",
                    description="ICI 주간 ETF 순발행 추정치입니다. 섹터별 flow가 아니라 미국 ETF 시장 전체 흐름 컨텍스트입니다.",
                    badge=str(ici_context.get("as_of", "")),
                )
                st.dataframe(
                    ici_table.rename(columns={"category": "Category", "value": "Net issuance (USD mn)"}),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Category": st.column_config.TextColumn("Category", width="medium"),
                        "Net issuance (USD mn)": st.column_config.NumberColumn("Net issuance (USD mn)", format="%.0f"),
                    },
                )

            positions_context = dict(ownership_context.get("sec_13f_positions") or {})
            positions_table = positions_context.get("table")
            if isinstance(positions_table, pd.DataFrame) and not positions_table.empty:
                render_panel_header(
                    eyebrow="SEC 13F",
                    title="Latest 13F Sector ETF Positioning",
                    description="최신 SEC 13F dataset에서 sector ETF 자체에 대한 institutional position을 집계한 값입니다. 분기 지연이 있는 ownership context입니다.",
                    badge=str(positions_context.get("dataset_label", "Latest 13F")),
                )
                st.dataframe(
                    positions_table.rename(
                        columns={
                            "sector_code": "ETF",
                            "sector_name": "Sector",
                            "filing_count": "Filings",
                            "manager_value_total_usd": "Reported value (USD)",
                            "manager_shares_total": "Reported shares",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "ETF": st.column_config.TextColumn("ETF", width="small"),
                        "Sector": st.column_config.TextColumn("Sector", width="medium"),
                        "cusip": st.column_config.TextColumn("CUSIP", width="small"),
                        "Filings": st.column_config.NumberColumn("Filings", format="%d"),
                        "Reported value (USD)": st.column_config.NumberColumn("Reported value (USD)", format="%.0f"),
                        "Reported shares": st.column_config.NumberColumn("Reported shares", format="%.0f"),
                    },
                )

            events_context = dict(ownership_context.get("sec_13dg_events") or {})
            events_table = events_context.get("table")
            if isinstance(events_table, pd.DataFrame) and not events_table.empty:
                render_panel_header(
                    eyebrow="SEC 13D/13G",
                    title="Recent 13D/13G Events",
                    description="최근 13D/13G 이벤트입니다. sector ETF top holdings 기준의 event-driven beneficial ownership context이며, participant flow가 아닙니다.",
                    badge=f"{int(events_context.get('lookback_days', 0))}D",
                )
                st.dataframe(
                    events_table.rename(
                        columns={
                            "sector_code": "ETF",
                            "sector_name": "Sector",
                            "matched_top_holdings": "Matched holdings",
                            "recent_13dg_events": "13D/G events",
                            "sample_events": "Examples",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "ETF": st.column_config.TextColumn("ETF", width="small"),
                        "Sector": st.column_config.TextColumn("Sector", width="medium"),
                        "Matched holdings": st.column_config.NumberColumn("Matched holdings", format="%d"),
                        "13D/G events": st.column_config.NumberColumn("13D/G events", format="%d"),
                        "Examples": st.column_config.TextColumn("Examples", width="large"),
                    },
                )

            form_sho_context = dict(ownership_context.get("form_sho_context") or {})
            if form_sho_context:
                render_panel_header(
                    eyebrow="Form SHO",
                    title="Form SHO Context",
                    description="SEC short-position transparency 레이어입니다. 현재는 공식 정책/availability 컨텍스트만 표시합니다.",
                    badge=str(form_sho_context.get("status", "")),
                )
                st.caption(str(form_sho_context.get("note", "")))

            context_errors = dict(ownership_context.get("errors") or {})
            if context_errors:
                preview = "; ".join(f"{key}: {value}" for key, value in sorted(context_errors.items()))
                st.caption(f"추가 컨텍스트 일부 로드 실패: {preview}")
            return

        render_panel_header(
            eyebrow=get_ui_text("flow_status_label", ui_locale),
            title=get_ui_text("flow_summary_title", ui_locale),
            description=get_ui_text("flow_summary_description", ui_locale),
            badge=f"{investor_flow_status} · {get_flow_profile_label(investor_flow_profile, ui_locale)}",
        )
        st.warning(get_ui_text("flow_tab_warning", ui_locale))

        if investor_flow_frame.empty:
            st.info(get_ui_text("flow_tab_empty", ui_locale))
            return

        latest_rows = []
        for signal in sorted(signals, key=lambda item: float(getattr(item, "flow_score", 0.0)), reverse=True):
            latest_rows.append(
                {
                    "Sector": str(getattr(signal, "sector_name", "")),
                    "Profile": get_flow_profile_label(str(getattr(signal, "flow_profile", investor_flow_profile)), ui_locale),
                    "Flow state": get_flow_state_label(str(getattr(signal, "flow_state", "unavailable")), ui_locale),
                    "Flow score": float(getattr(signal, "flow_score", 0.0)),
                    "Action change": f"{getattr(signal, 'base_action', getattr(signal, 'action', 'N/A'))} -> {getattr(signal, 'action', 'N/A')}",
                    "Foreign": get_flow_state_label(str(getattr(signal, "foreign_flow_state", "unavailable")), ui_locale),
                    "Institutional": get_flow_state_label(str(getattr(signal, "institutional_flow_state", "unavailable")), ui_locale),
                    "Retail": get_flow_state_label(str(getattr(signal, "retail_flow_state", "unavailable")), ui_locale),
                }
            )

        latest_df = pd.DataFrame(latest_rows)
        st.dataframe(
            latest_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Sector": st.column_config.TextColumn(get_ui_text("col_sector", ui_locale), width="medium"),
                "Profile": st.column_config.TextColumn(get_ui_text("flow_col_profile", ui_locale), width="medium"),
                "Flow state": st.column_config.TextColumn(get_ui_text("flow_col_state", ui_locale), width="small"),
                "Flow score": st.column_config.NumberColumn(get_ui_text("flow_col_score", ui_locale), format="%.2f"),
                "Action change": st.column_config.TextColumn(get_ui_text("flow_col_adjustment", ui_locale), width="medium"),
                "Foreign": st.column_config.TextColumn(get_ui_text("flow_col_foreign", ui_locale), width="small"),
                "Institutional": st.column_config.TextColumn(get_ui_text("flow_col_institutional", ui_locale), width="small"),
                "Retail": st.column_config.TextColumn(get_ui_text("flow_col_retail", ui_locale), width="small"),
            },
        )

        latest_date = investor_flow_frame.index.max()
        latest_snapshot = investor_flow_frame.loc[investor_flow_frame.index == latest_date].copy()
        if not latest_snapshot.empty:
            latest_snapshot["state_label"] = latest_snapshot["investor_type"].astype(str)
            latest_snapshot = latest_snapshot.rename(
                columns={
                    "sector_name": "Sector",
                    "investor_type": "Investor",
                    "net_flow_ratio": "Latest ratio",
                }
            )
            st.dataframe(
                latest_snapshot[["Sector", "Investor", "Latest ratio", "net_buy_amount"]].reset_index(drop=True),
                width="stretch",
                hide_index=True,
                column_config={
                    "Sector": st.column_config.TextColumn(get_ui_text("col_sector", ui_locale), width="medium"),
                    "Investor": st.column_config.TextColumn(get_ui_text("flow_status_label", ui_locale), width="small"),
                    "Latest ratio": st.column_config.NumberColumn(get_ui_text("flow_col_latest", ui_locale), format="%.4f"),
                    "net_buy_amount": st.column_config.NumberColumn("Net", format="%d"),
                },
            )

        # Data reference date and freshness caption
        try:
            ref_date_str = pd.Timestamp(latest_date).strftime("%Y-%m-%d")
        except Exception:
            ref_date_str = str(latest_date)
        source_label = {"LIVE": "실시간", "CACHED": "캐시", "SAMPLE": "샘플"}.get(
            str(investor_flow_status).upper(), investor_flow_status
        )
        st.caption(f"데이터 기준일: {ref_date_str} · 소스: {source_label}")
        if not investor_flow_fresh:
            st.caption(get_ui_text("flow_unavailable", ui_locale))


def _build_etf_map(sector_map: dict | None) -> dict[str, list]:
    """Build {index_code: [{"code":..., "name":...}, ...]} from sector_map config."""
    if not sector_map:
        return {}
    etf_map: dict[str, list] = {}
    for regime_data in sector_map.get("regimes", {}).values():
        for sector in regime_data.get("sectors", []):
            code = str(sector.get("code", ""))
            etfs = sector.get("etfs") or []
            if code and etfs:
                etf_map[code] = [{"code": str(e["code"]), "name": str(e["name"])} for e in etfs]
    return etf_map


@st.cache_data(ttl=60, show_spinner=False)
def _cached_monitoring_data(market_id: str) -> dict:
    """Load investor flow monitoring data with a 60-second TTL cache."""
    from src.data_sources.krx_investor_flow import read_warm_status
    from src.data_sources.warehouse import (
        read_investor_flow_raw_date_bounds,
        read_investor_flow_run_history,
    )
    warm = read_warm_status()
    bounds = read_investor_flow_raw_date_bounds(market=market_id)
    history = read_investor_flow_run_history(market=market_id, limit=15)
    return {"warm": warm, "bounds": bounds, "history": history}


def render_monitoring_tab(
    *,
    tab,
    market_id: str = "KR",
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> None:
    """Render the '데이터 모니터링' tab for investor-flow ingestion health."""
    with tab:
        render_panel_header(
            eyebrow="운영 현황",
            title="투자자 수급 데이터 수집 모니터링",
            description="KRX 투자자 수급 데이터의 수집 상태, 커버리지, 오류 이력을 확인합니다.",
            badge=market_id,
        )

        data = _cached_monitoring_data(str(market_id).strip().upper())
        warm: dict = data["warm"]
        bounds: dict = data["bounds"]
        history: pd.DataFrame = data["history"]

        # ── 섹션 1: 수집 상태 요약 ──────────────────────────────────────────
        st.subheader("수집 상태 요약")
        status_val = str(warm.get("status", "")).upper() or "UNKNOWN"
        watermark = str(warm.get("watermark_key", warm.get("end", "")) or "")
        coverage_ok = bool(warm.get("coverage_complete", False))
        predicted = int(warm.get("predicted_requests", 0) or 0)
        processed = int(warm.get("processed_requests", 0) or 0)
        completion_pct = round(processed / predicted * 100, 1) if predicted > 0 else 0.0

        status_label = {"LIVE": "🟢 LIVE", "CACHED": "🟡 CACHED", "SAMPLE": "⚪ SAMPLE"}.get(
            status_val, f"❓ {status_val}"
        )
        coverage_label = "✅ 완료" if coverage_ok else "❌ 미완료"
        watermark_label = watermark if watermark else "—"
        completion_label = f"{completion_pct}%" if predicted > 0 else "—"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("상태", status_label)
        col2.metric("워터마크 (마지막 성공일)", watermark_label)
        col3.metric("커버리지", coverage_label)
        col4.metric("최근 수집 완료율", completion_label)

        # ── 섹션 2: 데이터 커버리지 범위 ────────────────────────────────────
        st.subheader("데이터 커버리지 범위")
        min_date = str(bounds.get("min_trade_date", "") or "")
        max_date = str(bounds.get("max_trade_date", "") or "")
        if min_date and max_date:
            try:
                min_dt = pd.Timestamp(min_date)
                max_dt = pd.Timestamp(max_date)
                days = (max_dt - min_dt).days
                st.info(
                    f"수집 시작: **{min_dt.strftime('%Y-%m-%d')}** · "
                    f"수집 최신: **{max_dt.strftime('%Y-%m-%d')}** · "
                    f"보유 기간: **{days}일**"
                )
            except Exception:
                st.info(f"수집 범위: {min_date} ~ {max_date}")
        else:
            st.info("warehouse에 저장된 원시 데이터가 없습니다.")

        # ── 섹션 3: 미수집 날짜 / 오류 종목 ────────────────────────────────
        st.subheader("미수집 현황")
        failed_days: list = warm.get("failed_days") or []
        failed_codes: dict = warm.get("failed_codes") or {}
        aborted = bool(warm.get("aborted", False))
        abort_reason = str(warm.get("abort_reason", "") or "")

        if aborted and abort_reason:
            st.error(f"마지막 수집 중단됨: {abort_reason}")

        if failed_days:
            preview = ", ".join(failed_days[:10])
            suffix = f" 외 {len(failed_days) - 10}건" if len(failed_days) > 10 else ""
            st.warning(f"미수집 날짜 {len(failed_days)}건: {preview}{suffix}")
        if failed_codes:
            err_df = pd.DataFrame(
                [{"섹터코드": k, "오류": v} for k, v in failed_codes.items()]
            )
            st.warning(f"오류 섹터 {len(failed_codes)}건")
            st.dataframe(err_df, hide_index=True, use_container_width=True)
        if not failed_days and not failed_codes and not aborted:
            st.success("최근 수집 실행에서 미수집 항목 없음")

        # ── 섹션 4: 수집 이력 테이블 ────────────────────────────────────────
        st.subheader("수집 이력 (최근 15건)")
        if history.empty:
            st.info("수집 이력이 없습니다.")
        else:
            display = history.copy()
            display["요청범위"] = display["requested_start"].fillna("") + " ~ " + display["requested_end"].fillna("")
            display = display.rename(
                columns={
                    "created_at": "수집일시",
                    "reason": "이유",
                    "status": "상태",
                    "coverage_complete": "커버리지완료",
                    "aborted": "중단",
                    "predicted_requests": "예상요청",
                    "processed_requests": "처리요청",
                    "row_count": "저장행수",
                    "completion_pct": "완료율(%)",
                }
            )
            cols_ordered = [
                "수집일시", "이유", "요청범위", "상태",
                "커버리지완료", "중단", "완료율(%)", "예상요청", "처리요청", "저장행수",
            ]
            st.dataframe(
                display[cols_ordered].reset_index(drop=True),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "수집일시": st.column_config.DatetimeColumn("수집일시", format="YYYY-MM-DD HH:mm:ss"),
                    "이유": st.column_config.TextColumn("이유", width="small"),
                    "요청범위": st.column_config.TextColumn("요청범위", width="medium"),
                    "상태": st.column_config.TextColumn("상태", width="small"),
                    "커버리지완료": st.column_config.CheckboxColumn("커버리지완료"),
                    "중단": st.column_config.CheckboxColumn("중단"),
                    "완료율(%)": st.column_config.NumberColumn("완료율(%)", format="%.1f"),
                    "예상요청": st.column_config.NumberColumn("예상요청", format="%d"),
                    "처리요청": st.column_config.NumberColumn("처리요청", format="%d"),
                    "저장행수": st.column_config.NumberColumn("저장행수", format="%d"),
                },
            )

        if warm:
            provider = str(warm.get("provider", "") or "")
            st.caption(f"데이터 소스: {provider or '—'} · 60초 캐시")


def render_dashboard_tabs(
    *,
    current_regime: str,
    regime_is_confirmed: bool,
    growth_val: float | None,
    inflation_val: float | None,
    fx_change: float | None,
    fx_label: str,
    is_provisional: bool,
    theme_mode: str,
    price_status: str,
    macro_status: str,
    yield_curve_status: str | None,
    top_pick_signals: list[Any],
    signals_filtered: list[Any],
    signals: list[Any],
    filter_action_global: str,
    filter_regime_only_global: bool,
    held_sectors: list[str],
    position_mode: str,
    show_alerted_only: bool,
    settings: dict[str, Any],
    is_mobile_client: bool,
    market_id: str = "KR",
    investor_flow_status: str = "SAMPLE",
    investor_flow_fresh: bool = False,
    investor_flow_profile: str = "foreign_lead",
    investor_flow_frame: pd.DataFrame | None = None,
    investor_flow_detail: dict[str, Any] | None = None,
    sector_map: dict[str, Any] | None = None,
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> None:
    etf_map = _build_etf_map(sector_map)
    tab_labels = [
        "대시보드 요약",
        "모멘텀/차트 분석",
        "전체 종목 데이터",
        "종목 스크리닝",
    ]
    normalized_market = str(market_id).strip().upper() or "KR"
    include_flow_tab = normalized_market in {"KR", "US"}
    include_monitoring_tab = str(market_id).strip().upper() == "KR"
    if include_flow_tab:
        tab_labels.append("투자자 수급" if normalized_market == "KR" else "US Flow Proxies")
    if include_monitoring_tab:
        tab_labels.append("데이터 모니터링")
    tabs = st.tabs(tab_labels)
    tab_summary, tab_charts, tab_all_signals, tab_screening = tabs[:4]
    render_summary_tab(
        tab=tab_summary,
        current_regime=current_regime,
        regime_is_confirmed=regime_is_confirmed,
        growth_val=growth_val,
        inflation_val=inflation_val,
        fx_change=fx_change,
        fx_label=fx_label,
        is_provisional=is_provisional,
        theme_mode=theme_mode,
        price_status=price_status,
        macro_status=macro_status,
        yield_curve_status=yield_curve_status,
        top_pick_signals=top_pick_signals,
        signals_filtered=signals_filtered,
        ui_locale=ui_locale,
    )
    render_charts_tab(
        tab=tab_charts,
        signals_filtered=signals_filtered,
        theme_mode=theme_mode,
        is_mobile_client=is_mobile_client,
    )
    render_all_signals_tab(
        tab=tab_all_signals,
        signals=signals,
        filter_action_global=filter_action_global,
        filter_regime_only_global=filter_regime_only_global,
        current_regime=current_regime,
        held_sectors=held_sectors,
        position_mode=position_mode,
        show_alerted_only=show_alerted_only,
        theme_mode=theme_mode,
        settings=settings,
        fx_label=fx_label,
        etf_map=etf_map,
        ui_locale=ui_locale,
    )
    render_screening_tab(
        tab=tab_screening,
        signals=signals,
        settings=settings,
        benchmark_code=str(settings.get("benchmark_code", "1001")),
    )
    if include_flow_tab:
        render_investor_flow_tab(
            tab=tabs[4],
            signals=signals,
            investor_flow_frame=investor_flow_frame if investor_flow_frame is not None else pd.DataFrame(),
            investor_flow_status=investor_flow_status,
            investor_flow_fresh=investor_flow_fresh,
            investor_flow_profile=investor_flow_profile,
            investor_flow_detail=investor_flow_detail,
            market_id=normalized_market,
            ui_locale=ui_locale,
        )
    if include_monitoring_tab:
        monitoring_tab_idx = 4 + (1 if include_flow_tab else 0)
        render_monitoring_tab(
            tab=tabs[monitoring_tab_idx],
            market_id=str(market_id),
            ui_locale=ui_locale,
        )
