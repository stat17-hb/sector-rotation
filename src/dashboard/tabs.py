"""Dashboard UI entrypoints extracted from app.py."""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from config.theme import set_theme_mode
from src.dashboard.analysis import _extract_heatmap_selection
from src.ui.components import (
    HEATMAP_PALETTE_OPTIONS,
    build_sector_strength_heatmap,
    format_cycle_phase_label,
    format_heatmap_palette_label,
    normalize_range_preset,
    render_action_summary,
    render_cycle_timeline_panel,
    render_decision_hero,
    render_panel_header,
    render_returns_heatmap,
    render_rs_momentum_bar,
    render_rs_scatter,
    render_sector_detail_panel,
    render_signal_table,
    render_status_card_row,
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
) -> tuple[str, date, bool, bool, bool]:
    market_options = ["KR", "US"]
    selected_market = st.radio(
        ui_labels.get("market_selector", "Market"),
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

    st.caption(ui_labels.get("sidebar_title", "Sector Rotation"))
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
) -> None:
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
            title=f"Monthly sector strength vs {benchmark_label}",
            description=f"Read each cell as monthly excess return versus {benchmark_label}, using the same linked sector/month selection.",
            badge=format_cycle_phase_label(selected_cycle_phase),
        )
        strength_fig = build_sector_strength_heatmap(
            heatmap_strength_display,
            selected_sector=str(st.session_state.get("selected_sector", "")),
            selected_month=str(st.session_state.get("selected_month", "")),
            theme_mode=theme_mode,
            palette=analysis_heatmap_palette,
            title=f"Monthly sector strength vs {benchmark_label}",
            empty_message=f"No monthly sector strength vs {benchmark_label} data is available for the active filters.",
            helper_metric_label="monthly excess return",
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


def render_all_signals_tab(
    *,
    tab,
    signals: list[Any],
    filter_action_global: str,
    filter_regime_only_global: bool,
    current_regime: str,
    theme_mode: str,
    settings: dict[str, Any],
    fx_label: str,
) -> None:
    with tab:
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
    settings: dict[str, Any],
    is_mobile_client: bool,
) -> None:
    tab_summary, tab_charts, tab_all_signals = st.tabs([
        "대시보드 요약",
        "모멘텀/차트 분석",
        "전체 종목 데이터",
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
        theme_mode=theme_mode,
        settings=settings,
        fx_label=fx_label,
    )
