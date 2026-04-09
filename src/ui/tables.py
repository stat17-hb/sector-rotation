"""Table-oriented UI renderers."""
from __future__ import annotations

from src.ui.base import *  # noqa: F401,F403


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
    """Render a compact top-picks list using HTML cards instead of a wide dataframe to avoid horizontal scrolling."""
    filtered = filter_signals_for_display(
        signals,
        held_sectors=held_sectors,
        position_mode=position_mode,
    )
    if not filtered:
        st.info(_empty_top_pick_message(position_mode, held_sectors, locale=locale))
        return

    filtered = sorted(filtered, key=lambda signal: signal_display_sort_key(signal, held_sectors))
    
    cards_html = []
    for rank, signal in enumerate(list(filtered)[:limit], start=1):
        thesis = describe_signal_decision(signal, held_sectors, locale=locale)
        
        sector_name = signal.sector_name + (" *" if signal.is_provisional else "")
        decision = thesis.get("decision", "")
        reason = thesis.get("reason", "")
        risk = thesis.get("risk", "")
        invalidation = thesis.get("invalidation", "")
        alerts_text = thesis.get("alerts_text", "")
        
        ret_3m = _pct_value(signal.returns.get("3M"))
        ret_3m_str = f"{ret_3m:+.1f}%" if ret_3m is not None else "N/A"
        
        def _row(label: str, value: object) -> str:
            str_val = str(value).strip() if value else ""
            if not str_val or str_val == "None" or str_val == get_ui_text("alerts_none", locale):
                return ""
            return (
                '<div class="top-pick-card__row">'
                f'<span class="top-pick-card__label">{html.escape(label)}</span>'
                f'<span class="top-pick-card__value">{html.escape(str_val)}</span>'
                '</div>'
            )
            
        held_badge = ""
        if include_held:
            is_held = bool(thesis.get("held"))
            if is_held:
                held_badge = f'<span class="top-pick-card__held-badge">{html.escape(get_ui_text("col_held", locale))}</span>'

        card = (
            '<div class="top-pick-card">'
            '<div class="top-pick-card__header">'
            '<div class="top-pick-card__title">'
            f'<span class="top-pick-card__rank">{rank}.</span>'
            f'{html.escape(sector_name)}'
            f'{held_badge}'
            '</div>'
            f'<div class="top-pick-card__decision">{html.escape(str(decision))}</div>'
            '</div>'
            '<div class="top-pick-card__body">'
            f'{_row(get_ui_text("col_reason", locale), reason)}'
            f'{_row(get_ui_text("col_risk", locale), risk)}'
            f'{_row(get_ui_text("col_invalidation", locale), invalidation)}'
            '<div class="top-pick-card__metrics">'
            f'<span class="top-pick-card__metric"><strong>{get_ui_text("period_3m", locale)}</strong>{html.escape(ret_3m_str)}</span>'
            f'<span class="top-pick-card__metric"><strong>{get_ui_text("col_alerts", locale)}</strong>{html.escape(str(alerts_text))}</span>'
            '</div>'
            '</div>'
            '</div>'
        )
        cards_html.append(card)

    st.markdown(f'<div class="top-picks-container">{"".join(cards_html)}</div>', unsafe_allow_html=True)

    if len(filtered) > limit:
        st.caption(get_ui_text("top_picks_showing", locale, total=len(filtered), limit=limit))
    if any(getattr(signal, "is_provisional", False) for signal in filtered):
        st.caption(get_ui_text("provisional_caption", locale))


def _format_etfs(etfs: list) -> str:
    """Format ETF list as 'NAME (CODE) / NAME (CODE)'."""
    if not etfs:
        return ""
    return " / ".join(f"{item['name']} ({item['code']})" for item in etfs[:2])


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
        st.info(get_ui_text("signals_empty", locale))
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
        st.info(get_ui_text("signals_filtered_empty", locale))
        return

    filtered = sorted(filtered, key=lambda signal: signal_display_sort_key(signal, held_sectors))

    rows: list[dict[str, object]] = []
    for signal in filtered:
        thesis = describe_signal_decision(signal, held_sectors, locale=locale)
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
            "Alerts": thesis["alerts_text"],
        }
        rows.append(row)

    df_display = pd.DataFrame(rows)
    height = min(760, 76 + len(df_display) * 35)

    st.dataframe(
        df_display,
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
