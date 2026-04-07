"""Table-oriented UI renderers."""
from __future__ import annotations

from src.ui.base import *


def _empty_top_pick_message(
    position_mode: str,
    held_sectors: Sequence[str] | None = None,
    *,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> str:
    normalized = normalize_position_mode(position_mode)
    if normalized == "held" and not list(held_sectors or []):
        return get_ui_text("top_picks_empty_held_missing", locale)
    if normalized == "held":
        return get_ui_text("top_picks_empty_held", locale)
    if normalized == "new":
        return get_ui_text("top_picks_empty_new", locale)
    return get_ui_text("top_picks_empty_all", locale)


def render_top_picks_table(
    signals: Sequence,
    *,
    held_sectors: Sequence[str] | None = None,
    position_mode: str = "all",
    limit: int = 5,
    include_held: bool = True,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> None:
    """Render a compact top-picks table using Streamlit's native dataframe."""
    filtered = filter_signals_for_display(
        signals,
        held_sectors=held_sectors,
        position_mode=position_mode,
    )
    if not filtered:
        msg = _empty_top_pick_message(position_mode, held_sectors, locale=locale)
        st.markdown(f'<div class="empty-state-card"><h4>데이터 없음</h4><p>{msg}</p></div>', unsafe_allow_html=True)
        return

    def _color_momentum(val):
        if not isinstance(val, (int, float)) or pd.isna(val):
            return ""
        if val > 0:
            return "color: #ff4b4b; font-weight: 600;"
        elif val < 0:
            return "color: #2b7af0; font-weight: 600;"
        return ""

    rows: list[dict[str, object]] = []
    filtered = sorted(filtered, key=lambda signal: signal_display_sort_key(signal, held_sectors))
    for rank, signal in enumerate(list(filtered)[:limit], start=1):
        thesis = describe_signal_decision(signal, held_sectors, locale=locale)
        row = {
            "Rank": rank,
            "Sector": signal.sector_name + (" *" if signal.is_provisional else ""),
            "Decision": thesis["decision"],
            "Reason": thesis["reason"],
            "Risk": thesis["risk"],
            "Invalidation": thesis["invalidation"],
            "3M": _pct_value(signal.returns.get("3M")),
            "Alerts": thesis["alerts_text"],
        }
        if include_held:
            row["Held"] = bool(thesis["held"])
        rows.append(row)

    df_display = pd.DataFrame(rows)
    height = 76 + len(df_display) * 35
    
    styled_df = df_display.style.map(
        _color_momentum, subset=["3M"]
    )
    
    column_config: dict[str, object] = {
        "Rank": st.column_config.NumberColumn(get_ui_text("col_rank", locale), format="%d", width="small"),
        "Sector": st.column_config.TextColumn(get_ui_text("col_sector", locale), width="medium"),
        "Decision": st.column_config.TextColumn(get_ui_text("col_decision", locale), width="medium"),
        "Reason": st.column_config.TextColumn(get_ui_text("col_reason", locale), width="large"),
        "Risk": st.column_config.TextColumn(get_ui_text("col_risk", locale), width="large"),
        "Invalidation": st.column_config.TextColumn(get_ui_text("col_invalidation", locale), width="large"),
        "3M": st.column_config.NumberColumn(get_ui_text("period_3m", locale), format="%.1f%%"),
        "Alerts": st.column_config.TextColumn(get_ui_text("col_alerts", locale), width="medium"),
    }
    if include_held:
        column_config["Held"] = st.column_config.CheckboxColumn(get_ui_text("col_held", locale), width="small")
    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        height=height,
        column_config=column_config,
    )

    if len(filtered) > limit:
        st.caption(get_ui_text("top_picks_showing", locale, total=len(filtered), limit=limit))
    if any(getattr(signal, "is_provisional", False) for signal in filtered):
        st.caption(get_ui_text("provisional_caption", locale))

def _format_etfs(etfs: list) -> str:
    """Format ETF list as 'NAME (CODE) / NAME (CODE)'."""
    if not etfs:
        return ""
    return " / ".join(f"{e['name']} ({e['code']})" for e in etfs[:2])


def render_signal_table(
    signals: Sequence,
    filter_action: str | None = None,
    filter_regime_only: bool = False,
    current_regime: str | None = None,
    held_sectors: Sequence[str] | None = None,
    position_mode: str = "all",
    show_alerted_only: bool = False,
    theme_mode: str = "dark",
    etf_map: dict | None = None,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> None:
    """Render the full signal table using Streamlit's native dataframe."""
    del theme_mode  # native dataframe rendering does not need a theme argument

    if not signals:
        st.markdown(f'<div class="empty-state-card"><h4>데이터 없음</h4><p>{get_ui_text("signals_empty", locale)}</p></div>', unsafe_allow_html=True)
        return

    filtered = filter_signals_for_display(
        signals,
        filter_action=filter_action,
        filter_regime_only=filter_regime_only,
        current_regime=current_regime,
        held_sectors=held_sectors,
        position_mode=position_mode,
        show_alerted_only=show_alerted_only,
    )

    if not filtered:
        st.markdown(f'<div class="empty-state-card"><h4>필터된 신호 없음</h4><p>{get_ui_text("signals_filtered_empty", locale)}</p></div>', unsafe_allow_html=True)
        return

    filtered = sorted(filtered, key=lambda signal: signal_display_sort_key(signal, held_sectors))

    rows: list[dict[str, object]] = []
    for signal in filtered:
        thesis = describe_signal_decision(signal, held_sectors, locale=locale)
        alerts = thesis["alerts_text"]
        row: dict[str, object] = {
            "Sector": signal.sector_name + (" *" if signal.is_provisional else ""),
            "Held": bool(thesis["held"]),
            "Decision": thesis["decision"],
            "In Regime": bool(signal.macro_fit),
            "Action": format_action_label(signal.action, locale=locale),
            "ETF": _format_etfs((etf_map or {}).get(signal.index_code, [])),
            "Reason": thesis["reason"],
            "Invalidation": thesis["invalidation"],
            "RSI": _safe_float(signal.rsi_d),
            "1M": _pct_value(signal.returns.get("1M")),
            "3M": _pct_value(signal.returns.get("3M")),
            "Volatility": _pct_value(signal.volatility_20d),
            "MDD (3M)": _pct_value(signal.mdd_3m),
            "Alerts": alerts,
        }
        rows.append(row)

    df_display = pd.DataFrame(rows)
    height = min(760, 76 + len(df_display) * 35)

    def _color_momentum(val):
        if not isinstance(val, (int, float)) or pd.isna(val):
            return ""
        if val > 0:
            return "color: #ff4b4b; font-weight: 600;"
        elif val < 0:
            return "color: #2b7af0; font-weight: 600;"
        return ""

    styled_df = df_display.style.map(
        _color_momentum, subset=["1M", "3M"]
    )

    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        height=height,
        column_config={
            "Sector": st.column_config.TextColumn(get_ui_text("col_sector", locale), width="medium"),
            "Held": st.column_config.CheckboxColumn(get_ui_text("col_held", locale), width="small"),
            "Decision": st.column_config.TextColumn(get_ui_text("col_decision", locale), width="medium"),
            "In Regime": st.column_config.CheckboxColumn(get_ui_text("col_in_regime", locale), width="small"),
            "Action": st.column_config.TextColumn(get_ui_text("col_action", locale), width="small"),
            "ETF": st.column_config.TextColumn(get_ui_text("col_etf", locale), width="medium"),
            "Reason": st.column_config.TextColumn(get_ui_text("col_reason", locale), width="large"),
            "Invalidation": st.column_config.TextColumn(get_ui_text("col_invalidation", locale), width="large"),
            "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
            "1M": st.column_config.NumberColumn(get_ui_text("period_1m", locale), format="%.1f%%"),
            "3M": st.column_config.NumberColumn(get_ui_text("period_3m", locale), format="%.1f%%"),
            "Volatility": st.column_config.NumberColumn(get_ui_text("col_volatility", locale), format="%.1f%%"),
            "MDD (3M)": st.column_config.NumberColumn(get_ui_text("col_mdd_3m", locale), format="%.1f%%"),
            "Alerts": st.column_config.TextColumn(get_ui_text("col_alerts", locale), width="large"),
        },
    )

    if any(signal.is_provisional for signal in filtered):
        st.caption(get_ui_text("provisional_caption", locale))
