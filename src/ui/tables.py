"""Table-oriented UI renderers."""
from __future__ import annotations

from src.ui.base import *
def _build_top_pick_reason(signal) -> str:
    reasons: list[str] = []
    reasons.append("Regime fit" if signal.macro_fit else "Regime mismatch")
    rs_div = _rs_divergence_pct(signal)
    if rs_div is not None:
        reasons.append("RS above trend" if rs_div >= 0 else "RS below trend")
    reasons.append("Trend intact" if signal.trend_ok else "Trend weakened")
    if signal.alerts:
        reasons.append(signal.alerts[0])
    return " | ".join(reasons[:3])


def render_top_picks_table(
    signals: Sequence,
    *,
    limit: int = 5,
) -> None:
    """Render a compact top-picks table using Streamlit's native dataframe."""
    if not signals:
        st.info("No sectors match the current filter set.")
        return

    rows: list[dict[str, object]] = []
    for rank, signal in enumerate(list(signals)[:limit], start=1):
        rows.append(
            {
                "Rank": rank,
                "Sector": signal.sector_name + (" *" if signal.is_provisional else ""),
                "Action": format_action_label(signal.action),
                "Why": _build_top_pick_reason(signal),
                "RS Gap": _rs_divergence_pct(signal),
                "1M": _pct_value(signal.returns.get("1M")),
                "3M": _pct_value(signal.returns.get("3M")),
                "Alerts": ", ".join(signal.alerts) if signal.alerts else "-",
            }
        )

    df_display = pd.DataFrame(rows)
    height = 76 + len(df_display) * 35
    st.dataframe(
        df_display,
        width="stretch",
        hide_index=True,
        height=height,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "Sector": st.column_config.TextColumn("Sector", width="medium"),
            "Action": st.column_config.TextColumn("Action", width="small"),
            "Why": st.column_config.TextColumn("Why", width="large"),
            "RS Gap": st.column_config.NumberColumn("RS Gap", format="%.2f%%"),
            "1M": st.column_config.NumberColumn("1M", format="%.1f%%"),
            "3M": st.column_config.NumberColumn("3M", format="%.1f%%"),
            "Alerts": st.column_config.TextColumn("Alerts", width="medium"),
        },
    )

    if len(signals) > limit:
        st.caption(f"Showing top {limit} of {len(signals)} matching sectors.")
    if any(getattr(signal, "is_provisional", False) for signal in signals):
        st.caption("* Includes sectors influenced by provisional macro data.")

def render_signal_table(
    signals: Sequence,
    filter_action: str | None = None,
    filter_regime_only: bool = False,
    current_regime: str | None = None,
    theme_mode: str = "dark",
) -> None:
    """Render the full signal table using Streamlit's native dataframe."""
    del theme_mode  # native dataframe rendering does not need a theme argument

    if not signals:
        st.info("No signal data available.")
        return

    filtered = list(signals)
    if filter_regime_only and current_regime:
        filtered = [signal for signal in filtered if signal.macro_regime == current_regime]
    if filter_action and not _is_all_action_filter(filter_action):
        filtered = [signal for signal in filtered if signal.action == filter_action]

    if not filtered:
        st.info("No sectors match the active filters.")
        return

    filtered.sort(
        key=lambda signal: (
            ACTION_PRIORITY.get(signal.action, 99),
            -(_safe_float(signal.returns.get("3M")) or -999.0),
            signal.sector_name,
        )
    )

    rows: list[dict[str, object]] = []
    for signal in filtered:
        alerts = ", ".join(signal.alerts) if signal.alerts else ("Data missing" if signal.action == "N/A" else "-")
        rows.append(
            {
                "Sector": signal.sector_name + (" *" if signal.is_provisional else ""),
                "In Regime": bool(signal.macro_fit),
                "Action": format_action_label(signal.action),
                "RSI": _safe_float(signal.rsi_d),
                "1M": _pct_value(signal.returns.get("1M")),
                "3M": _pct_value(signal.returns.get("3M")),
                "Volatility": _pct_value(signal.volatility_20d),
                "MDD (3M)": _pct_value(signal.mdd_3m),
                "Alerts": alerts,
            }
        )

    df_display = pd.DataFrame(rows)
    height = min(760, 76 + len(df_display) * 35)
    st.dataframe(
        df_display,
        width="stretch",
        hide_index=True,
        height=height,
        column_config={
            "Sector": st.column_config.TextColumn("Sector", width="medium"),
            "In Regime": st.column_config.CheckboxColumn("In Regime", width="small"),
            "Action": st.column_config.TextColumn("Action", width="small"),
            "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
            "1M": st.column_config.NumberColumn("1M", format="%.1f%%"),
            "3M": st.column_config.NumberColumn("3M", format="%.1f%%"),
            "Volatility": st.column_config.NumberColumn("Volatility", format="%.1f%%"),
            "MDD (3M)": st.column_config.NumberColumn("MDD (3M)", format="%.1f%%"),
            "Alerts": st.column_config.TextColumn("Alerts", width="large"),
        },
    )

    if any(signal.is_provisional for signal in filtered):
        st.caption("* Includes sectors influenced by provisional macro data.")
