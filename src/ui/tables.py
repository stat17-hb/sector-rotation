"""Table-oriented UI renderers."""
from __future__ import annotations

from src.ui.base import *


def _empty_top_pick_message(position_mode: str, held_sectors: Sequence[str] | None = None) -> str:
    normalized = normalize_position_mode(position_mode)
    if normalized == "held" and not list(held_sectors or []):
        return "Add held sectors first to enable portfolio action recommendations."
    if normalized == "held":
        return "No held sectors match the current decision rules."
    if normalized == "new":
        return "No new-buy ideas match the current decision rules."
    return "No sectors match the current filter set."


def render_top_picks_table(
    signals: Sequence,
    *,
    held_sectors: Sequence[str] | None = None,
    position_mode: str = "all",
    limit: int = 5,
    include_held: bool = True,
) -> None:
    """Render a compact top-picks table using Streamlit's native dataframe."""
    filtered = filter_signals_for_display(
        signals,
        held_sectors=held_sectors,
        position_mode=position_mode,
    )
    if not filtered:
        st.info(_empty_top_pick_message(position_mode, held_sectors))
        return

    rows: list[dict[str, object]] = []
    filtered = sorted(filtered, key=lambda signal: signal_display_sort_key(signal, held_sectors))
    for rank, signal in enumerate(list(filtered)[:limit], start=1):
        thesis = describe_signal_decision(signal, held_sectors)
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
    column_config: dict[str, object] = {
        "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
        "Sector": st.column_config.TextColumn("Sector", width="medium"),
        "Decision": st.column_config.TextColumn("Decision", width="medium"),
        "Reason": st.column_config.TextColumn("Reason", width="large"),
        "Risk": st.column_config.TextColumn("Risk", width="large"),
        "Invalidation": st.column_config.TextColumn("Invalidation", width="large"),
        "3M": st.column_config.NumberColumn("3M", format="%.1f%%"),
        "Alerts": st.column_config.TextColumn("Alerts", width="medium"),
    }
    if include_held:
        column_config["Held"] = st.column_config.CheckboxColumn("Held", width="small")
    st.dataframe(
        df_display,
        width="stretch",
        hide_index=True,
        height=height,
        column_config=column_config,
    )

    if len(filtered) > limit:
        st.caption(f"Showing top {limit} of {len(filtered)} matching sectors.")
    if any(getattr(signal, "is_provisional", False) for signal in filtered):
        st.caption("* Includes sectors influenced by provisional macro data.")

def render_signal_table(
    signals: Sequence,
    filter_action: str | None = None,
    filter_regime_only: bool = False,
    current_regime: str | None = None,
    held_sectors: Sequence[str] | None = None,
    position_mode: str = "all",
    show_alerted_only: bool = False,
    theme_mode: str = "dark",
) -> None:
    """Render the full signal table using Streamlit's native dataframe."""
    del theme_mode  # native dataframe rendering does not need a theme argument

    if not signals:
        st.info("No signal data available.")
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
        st.info("No sectors match the active filters.")
        return

    filtered = sorted(filtered, key=lambda signal: signal_display_sort_key(signal, held_sectors))

    rows: list[dict[str, object]] = []
    for signal in filtered:
        thesis = describe_signal_decision(signal, held_sectors)
        alerts = thesis["alerts_text"]
        rows.append(
            {
                "Sector": signal.sector_name + (" *" if signal.is_provisional else ""),
                "Held": bool(thesis["held"]),
                "Decision": thesis["decision"],
                "In Regime": bool(signal.macro_fit),
                "Action": format_action_label(signal.action),
                "Reason": thesis["reason"],
                "Invalidation": thesis["invalidation"],
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
            "Held": st.column_config.CheckboxColumn("Held", width="small"),
            "Decision": st.column_config.TextColumn("Decision", width="medium"),
            "In Regime": st.column_config.CheckboxColumn("In Regime", width="small"),
            "Action": st.column_config.TextColumn("Action", width="small"),
            "Reason": st.column_config.TextColumn("Reason", width="large"),
            "Invalidation": st.column_config.TextColumn("Invalidation", width="large"),
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
