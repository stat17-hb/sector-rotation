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
    HEATMAP_PALETTE_OPTIONS,
    build_sector_strength_heatmap,
    describe_signal_decision,
    get_action_filter_label,
    format_cycle_phase_label,
    format_heatmap_palette_label,
    normalize_range_preset,
    render_action_summary,
    render_cycle_timeline_panel,
    render_decision_hero,
    render_investor_decision_boards,
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
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> tuple[str, date, bool, bool, bool]:
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

    st.caption(ui_labels.get("sidebar_title", "섹터 로테이션"))
    return selected_market, asof_date, refresh_market, refresh_macro, recompute


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
    yield_curve_status: str | None,
    signals: list[Any],
    held_sector_options: list[str],
    action_options: list[str],
    is_mobile_client: bool,
    analysis_canvas_kwargs: dict[str, Any],
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
                eyebrow="횡단면",
                title="수익률 히트맵",
                description="대시보드 전체에서 사용하는 동일한 패널 구조로 복수 기간 섹터 수익률을 비교하세요.",
            )
            fig_heatmap = render_returns_heatmap(signals_filtered, theme_mode=theme_mode)
            st.plotly_chart(fig_heatmap, width="stretch")
        else:
            st.info("글로벌 필터 조건에 맞는 신호가 없습니다.")


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
            st.info("현재 Strong Buy 섹터가 없습니다. 매크로 국면이 확정되면 종목 스크리닝이 활성화됩니다.")
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
            st.warning(
                "구성종목 데이터를 가져올 수 없습니다. "
                "주말·공휴일 또는 KRX API 점검 중에는 조회가 불가능합니다. "
                "평일 장중/장후에 '데이터 갱신'을 눌러주세요."
            )
            return

        status_label = {"LIVE": "실시간", "CACHED": "캐시(24h)"}
        st.caption(f"데이터 상태: **{status_label.get(status, status)}** | 총 {len(rows)}개 종목")

        if show_momentum_only:
            rows = [r for r in rows if r.get("momentum_ok")]

        if not rows:
            st.info("모멘텀 조건(RS 상승 + SMA 추세 양호)을 충족하는 종목이 없습니다.")
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

        st.dataframe(
            df,
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
    sector_map: dict[str, Any] | None = None,
    ui_locale: str = DEFAULT_UI_LOCALE,
) -> None:
    etf_map = _build_etf_map(sector_map)
    tab_summary, tab_charts, tab_all_signals, tab_screening = st.tabs([
        "대시보드 요약",
        "모멘텀/차트 분석",
        "전체 종목 데이터",
        "종목 스크리닝",
    ])
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
