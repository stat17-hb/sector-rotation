"""Reusable Streamlit UI components for the dashboard."""
from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.theme import get_chart_tokens, get_signal_tokens
from src.ui.styles import (
    get_action_colors,
    get_plotly_template,
    get_theme_tokens,
)


ALL_ACTION_OPTION = "전체"
LEGACY_ALL_ACTION_OPTIONS = {ALL_ACTION_OPTION, "All"}

ACTION_LABELS: dict[str, str] = {
    "Strong Buy": "[+] Strong Buy",
    "Watch": "[~] Watch",
    "Hold": "[=] Hold",
    "Avoid": "[x] Avoid",
    "N/A": "[-] N/A",
}

ACTION_PRIORITY: dict[str, int] = {
    "Strong Buy": 0,
    "Watch": 1,
    "Hold": 2,
    "Avoid": 3,
    "N/A": 4,
}

REGIME_SUBLABELS: dict[str, str] = {
    "Recovery": "Early-cycle rebound",
    "Expansion": "Risk-on growth phase",
    "Slowdown": "Growth cooling",
    "Contraction": "Defensive cycle",
    "Indeterminate": "Signal mix is inconclusive",
}

RANGE_PRESET_MONTHS: dict[str, int | None] = {
    "1Y": 12,
    "3Y": 36,
    "5Y": 60,
    "ALL": None,
    "CUSTOM": None,
}

RANGE_PRESET_LABELS: dict[str, str] = {
    "1Y": "1Y",
    "3Y": "3Y",
    "5Y": "5Y",
    "ALL": "All",
    "CUSTOM": "Custom",
}

LEGACY_RANGE_PRESET_MAP: dict[str, str] = {
    "3M": "CUSTOM",
    "6M": "CUSTOM",
    "12M": "1Y",
    "18M": "CUSTOM",
}

CYCLE_PHASE_ORDER: list[str] = [
    "ALL",
    "RECOVERY_EARLY",
    "RECOVERY_LATE",
    "EXPANSION_EARLY",
    "EXPANSION_LATE",
    "SLOWDOWN_EARLY",
    "SLOWDOWN_LATE",
    "CONTRACTION_EARLY",
    "CONTRACTION_LATE",
]

CYCLE_PHASE_LABELS: dict[str, str] = {
    "ALL": "All phases",
    "RECOVERY_EARLY": "Recovery / Early",
    "RECOVERY_LATE": "Recovery / Late",
    "EXPANSION_EARLY": "Expansion / Early",
    "EXPANSION_LATE": "Expansion / Late",
    "SLOWDOWN_EARLY": "Slowdown / Early",
    "SLOWDOWN_LATE": "Slowdown / Late",
    "CONTRACTION_EARLY": "Contraction / Early",
    "CONTRACTION_LATE": "Contraction / Late",
}

CYCLE_REGIME_PALETTE_LABELS: list[tuple[str, str]] = [
    ("Recovery", "cycle-recovery"),
    ("Expansion", "cycle-expansion"),
    ("Slowdown", "cycle-slowdown"),
    ("Contraction", "cycle-contraction"),
    ("Indeterminate", "cycle-indeterminate"),
]

HEATMAP_PALETTE_OPTIONS: tuple[str, ...] = ("classic", "contrast", "blue_orange")
HEATMAP_PALETTE_LABELS: dict[str, str] = {
    "classic": "Classic red/green",
    "contrast": "High-contrast red/green",
    "blue_orange": "Blue/orange diverging",
}


def format_action_label(action: str) -> str:
    """Return explicit text for action status so color is never the only cue."""
    return ACTION_LABELS.get(action, f"[?] {action}")


def format_cycle_phase_label(phase_key: str) -> str:
    """Return the user-facing label for a cycle phase selection key."""
    return CYCLE_PHASE_LABELS.get(str(phase_key), str(phase_key))


def format_range_preset_label(preset: str) -> str:
    """Return the compact display label for a range preset."""
    normalized = normalize_range_preset(preset)
    return RANGE_PRESET_LABELS.get(normalized, normalized)


def format_heatmap_palette_label(palette: str) -> str:
    """Return the user-facing label for a heatmap palette preset."""
    normalized = normalize_heatmap_palette(palette)
    return HEATMAP_PALETTE_LABELS.get(normalized, normalized)


def normalize_range_preset(preset: str | None) -> str:
    """Map legacy range presets onto the current 1Y/3Y/5Y/ALL/CUSTOM contract."""
    normalized = str(preset or "CUSTOM").strip().upper()
    if normalized in RANGE_PRESET_MONTHS:
        return normalized
    return LEGACY_RANGE_PRESET_MAP.get(normalized, "CUSTOM")


def normalize_heatmap_palette(palette: str | None) -> str:
    """Map unknown heatmap palette values back to the default preset."""
    normalized = str(palette or "classic").strip().lower()
    return normalized if normalized in HEATMAP_PALETTE_OPTIONS else "classic"


def get_analysis_heatmap_colorscale(
    *,
    theme_mode: str,
    palette: str | None,
) -> list[list[object]]:
    """Return a diverging colorscale preset for analysis heatmaps."""
    chart_tokens = get_chart_tokens(theme_mode)
    normalized = normalize_heatmap_palette(palette)
    return [list(stop) for stop in chart_tokens["analysis_heatmaps"][normalized]]


def resolve_range_from_preset(
    *,
    max_date: date,
    min_date: date,
    preset: str | None,
) -> tuple[date, date]:
    """Resolve a preset into a clamped [start, end] date range."""
    end_ts = pd.Timestamp(max_date).normalize()
    min_ts = pd.Timestamp(min_date).normalize()
    months = RANGE_PRESET_MONTHS.get(normalize_range_preset(preset))
    if months is None:
        return min_ts.date(), end_ts.date()
    start_ts = (end_ts - pd.DateOffset(months=months)) + pd.Timedelta(days=1)
    start_ts = max(start_ts.normalize(), min_ts)
    return start_ts.date(), end_ts.date()


def infer_range_preset(
    *,
    start_date: date,
    end_date: date,
    min_date: date,
    max_date: date,
) -> str:
    """Infer the closest preset label for the supplied date range."""
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    min_ts = pd.Timestamp(min_date).normalize()
    max_ts = pd.Timestamp(max_date).normalize()

    if start_ts <= min_ts and end_ts >= max_ts:
        return "ALL"

    for preset in ("1Y", "3Y", "5Y"):
        preset_start, preset_end = resolve_range_from_preset(
            max_date=max_ts.date(),
            min_date=min_ts.date(),
            preset=preset,
        )
        if start_ts == pd.Timestamp(preset_start) and end_ts == pd.Timestamp(preset_end):
            return preset
    return "CUSTOM"


def _safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _is_all_action_filter(value: str | None) -> bool:
    return str(value).strip() in LEGACY_ALL_ACTION_OPTIONS


def _pct_value(value: float | int | None) -> float | None:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return numeric * 100


def _rs_divergence_pct(signal) -> float | None:
    rs = _safe_float(getattr(signal, "rs", None))
    rs_ma = _safe_float(getattr(signal, "rs_ma", None))
    if rs is None or rs_ma in {None, 0.0}:
        return None
    return (rs - rs_ma) / rs_ma * 100


def _status_tone(value: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in {"LIVE", "OK"}:
        return "success"
    if normalized in {"CACHED", "WARNING"}:
        return "warning"
    if normalized in {"BLOCKED", "SAMPLE", "ERROR"}:
        return "danger"
    return "info"


def _action_tone(action: str) -> str:
    if action == "Strong Buy":
        return "success"
    if action == "Watch":
        return "info"
    if action == "Hold":
        return "warning"
    if action == "Avoid":
        return "danger"
    return "info"


def _render_card_html(
    *,
    eyebrow: str,
    value: str,
    detail: str = "",
    tone: str = "info",
    extra_html: str = "",
) -> str:
    detail_html = (
        f'<div class="status-card__detail">{html.escape(detail)}</div>'
        if detail
        else ""
    )
    return (
        f'<div class="status-card" data-tone="{html.escape(tone)}">'
        f'<div class="status-card__eyebrow">{html.escape(eyebrow)}</div>'
        f'<div class="status-card__value">{html.escape(value)}</div>'
        f"{detail_html}"
        f"{extra_html}"
        "</div>"
    )


def _render_cards_grid(cards: Sequence[str], class_name: str) -> None:
    if not cards:
        return
    markup = "".join(cards)
    st.markdown(f'<div class="{class_name}">{markup}</div>', unsafe_allow_html=True)


def render_page_header(
    *,
    title: str,
    description: str,
    pills: Sequence[Mapping[str, str]] | None = None,
) -> None:
    """Render the app-level page shell header."""
    pill_markup = "".join(
        (
            '<span class="page-shell__pill" '
            f'data-tone="{html.escape(str(item.get("tone", "info")))}">'
            f'<span>{html.escape(str(item.get("label", "")))}</span>'
            f"<strong>{html.escape(str(item.get('value', '')))}</strong>"
            "</span>"
        )
        for item in (pills or [])
        if str(item.get("label", "")).strip() and str(item.get("value", "")).strip()
    )
    pills_html = (
        f'<div class="page-shell__pills">{pill_markup}</div>' if pill_markup else ""
    )
    st.markdown(
        (
            '<section class="page-shell">'
            '<div class="page-shell__eyebrow">Sector rotation cockpit</div>'
            f'<div class="page-shell__title">{html.escape(title)}</div>'
            f'<div class="page-shell__description">{html.escape(description)}</div>'
            f"{pills_html}"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render_status_strip(banner: Mapping[str, object] | None) -> None:
    """Render a persistent top-of-page status strip for system state."""
    if not banner:
        return

    tone = str(banner.get("level", "info")).strip().lower() or "info"
    title = str(banner.get("title", "")).strip()
    message = str(banner.get("message", "")).strip()
    details = [str(item).strip() for item in banner.get("details", []) if str(item).strip()]
    detail_count = len(details)
    detail_html = (
        f'<span class="status-strip__meta">{detail_count} detail'
        f"{'' if detail_count == 1 else 's'}</span>"
        if detail_count
        else ""
    )

    st.markdown(
        (
            '<div class="status-strip" '
            f'data-tone="{html.escape(tone)}">'
            f'<div class="status-strip__badge">{html.escape(tone.upper())}</div>'
            '<div class="status-strip__body">'
            f'<div class="status-strip__title">{html.escape(title)}</div>'
            f'<div class="status-strip__message">{html.escape(message)}</div>'
            "</div>"
            f"{detail_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if details:
        with st.expander("System details", expanded=False):
            for detail in details:
                st.write(f"- {detail}")


def render_panel_header(
    *,
    eyebrow: str,
    title: str,
    description: str,
    badge: str = "",
) -> None:
    """Render a compact header used above chart and table panels."""
    badge_html = (
        f'<span class="panel-header__badge">{html.escape(badge)}</span>' if badge else ""
    )
    st.markdown(
        (
            '<div class="panel-header">'
            '<div class="panel-header__copy">'
            f'<div class="panel-header__eyebrow">{html.escape(eyebrow)}</div>'
            f'<div class="panel-header__title">{html.escape(title)}</div>'
            f'<div class="panel-header__description">{html.escape(description)}</div>'
            "</div>"
            f"{badge_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_analysis_toolbar(
    *,
    min_date: date,
    max_date: date,
    start_date: date,
    end_date: date,
    selected_range_preset: str,
    selected_cycle_phase: str,
    selected_sector: str,
) -> tuple[date, date, str, bool]:
    """Render the top analysis toolbar and return the committed selection."""
    current_preset = normalize_range_preset(selected_range_preset)

    summary_markup = (
        '<div class="analysis-toolbar__summary">'
        '<div class="analysis-toolbar__summary-item"><span>Window</span>'
        f"<strong>{html.escape(str(start_date))} - {html.escape(str(end_date))}</strong></div>"
        '<div class="analysis-toolbar__summary-item"><span>Cycle</span>'
        f"<strong>{html.escape(format_cycle_phase_label(selected_cycle_phase))}</strong></div>"
        '<div class="analysis-toolbar__summary-item"><span>Sector</span>'
        f"<strong>{html.escape(selected_sector or 'Auto')}</strong></div>"
        "</div>"
    )

    st.markdown(
        (
            '<div class="analysis-toolbar">'
            '<div class="analysis-toolbar__eyebrow">Analysis controls</div>'
            '<div class="analysis-toolbar__title">Pin the period first, then scan regime context and sector leadership.</div>'
            f"{summary_markup}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    with st.form("analysis_toolbar_form"):
        start_col, end_col, preset_col, apply_col = st.columns([1.2, 1.2, 1.6, 0.72])
        with start_col:
            start_input = st.date_input(
                "Start",
                value=start_date,
                min_value=min_date,
                max_value=max_date,
            )
        with end_col:
            end_input = st.date_input(
                "End",
                value=end_date,
                min_value=min_date,
                max_value=max_date,
            )
        with preset_col:
            preset_input = st.segmented_control(
                "Quick range",
                options=["1Y", "3Y", "5Y", "ALL", "CUSTOM"],
                default=current_preset,
                format_func=format_range_preset_label,
                selection_mode="single",
                label_visibility="visible",
                width="stretch",
            )
        with apply_col:
            submitted = st.form_submit_button("Apply", width="stretch", type="primary")

    if not submitted:
        return start_date, end_date, current_preset, False

    end_final = min(pd.Timestamp(end_input).date(), max_date)
    selected_preset = normalize_range_preset(str(preset_input or "CUSTOM"))
    if selected_preset != "CUSTOM":
        start_final, end_final = resolve_range_from_preset(
            max_date=end_final,
            min_date=min_date,
            preset=selected_preset,
        )
        return start_final, end_final, selected_preset, True

    start_final = max(pd.Timestamp(start_input).date(), min_date)
    if start_final > end_final:
        start_final = end_final

    inferred = infer_range_preset(
        start_date=start_final,
        end_date=end_final,
        min_date=min_date,
        max_date=max_date,
    )
    return start_final, end_final, inferred, True


def render_top_bar_filters(
    *,
    current_regime: str,
    action_options: Sequence[str],
    filter_action_key: str = "filter_action_global",
    filter_regime_key: str = "filter_regime_only_global",
    is_mobile: bool = False,
) -> tuple[str, bool]:
    """Render high-frequency filters in the main content area."""
    with st.container(border=True):
        st.markdown(
            (
                '<div class="command-bar">'
                '<div class="command-bar__eyebrow">Quick filters</div>'
                '<div class="command-bar__title">Refine the signal board without leaving the main canvas.</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        if is_mobile:
            st.selectbox(
                "Action filter",
                options=list(action_options),
                key=filter_action_key,
            )
            st.toggle(
                "Current regime only",
                key=filter_regime_key,
            )
        else:
            filter_col, toggle_col, summary_col = st.columns([1.6, 1.2, 2.4])
            with filter_col:
                st.selectbox(
                    "Action filter",
                    options=list(action_options),
                    key=filter_action_key,
                )
            with toggle_col:
                st.toggle(
                    "Current regime only",
                    key=filter_regime_key,
                )
            with summary_col:
                current_action = str(st.session_state.get(filter_action_key, action_options[0]))
                regime_only = bool(st.session_state.get(filter_regime_key, False))
                scope_label = "Matching current regime" if regime_only else "Full universe"
                st.markdown(
                    (
                        '<div class="top-bar-summary">'
                        '<div class="top-bar-summary__item"><span>Regime</span>'
                        f"<strong>{html.escape(current_regime)}</strong></div>"
                        '<div class="top-bar-summary__item"><span>Action</span>'
                        f"<strong>{html.escape(current_action)}</strong></div>"
                        '<div class="top-bar-summary__item"><span>Scope</span>'
                        f"<strong>{html.escape(scope_label)}</strong></div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

        if is_mobile:
            current_action = str(st.session_state.get(filter_action_key, action_options[0]))
            regime_only = bool(st.session_state.get(filter_regime_key, False))
            scope_label = "Matching current regime" if regime_only else "Full universe"
            st.markdown(
                (
                    '<div class="top-bar-summary">'
                    '<div class="top-bar-summary__item"><span>Regime</span>'
                    f"<strong>{html.escape(current_regime)}</strong></div>"
                    '<div class="top-bar-summary__item"><span>Action</span>'
                    f"<strong>{html.escape(current_action)}</strong></div>"
                    '<div class="top-bar-summary__item"><span>Scope</span>'
                    f"<strong>{html.escape(scope_label)}</strong></div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    return (
        str(st.session_state.get(filter_action_key, action_options[0])),
        bool(st.session_state.get(filter_regime_key, False)),
    )


def render_decision_hero(
    *,
    regime: str,
    regime_is_confirmed: bool,
    growth_val: float | None = None,
    inflation_val: float | None = None,
    fx_change: float | None = None,
    is_provisional: bool = False,
    theme_mode: str = "dark",
) -> None:
    """Render the hero card summarizing the current macro regime."""
    tokens = get_theme_tokens(theme_mode)
    regime_colors = {
        "Recovery": tokens["success"],
        "Expansion": tokens["primary"],
        "Slowdown": tokens["warning"],
        "Contraction": tokens["danger"],
        "Indeterminate": tokens["text_muted"],
    }
    accent = regime_colors.get(regime, tokens["primary"])
    regime_subtitle = REGIME_SUBLABELS.get(regime, regime)
    regime_state = "Confirmed regime" if regime_is_confirmed else "Provisional regime"

    chips = [
        f'<span class="decision-hero__chip">{html.escape(regime_state)}</span>',
    ]
    if is_provisional:
        chips.append('<span class="decision-hero__badge">Includes provisional macro prints</span>')

    def _hero_metric(label: str, value: float | None, suffix: str) -> str:
        numeric = _safe_float(value)
        display = f"{numeric:.2f}{suffix}" if numeric is not None else "N/A"
        return (
            '<div class="decision-hero__stat">'
            f'<span class="decision-hero__stat-label">{html.escape(label)}</span>'
            f'<strong>{html.escape(display)}</strong>'
            "</div>"
        )

    fx_numeric = _safe_float(fx_change)
    fx_display = f"{fx_numeric:+.1f}%" if fx_numeric is not None else "N/A"
    hero_metrics = "".join(
        [
            _hero_metric("Leading index", growth_val, "p"),
            _hero_metric("CPI YoY", inflation_val, "%"),
            _hero_metric("USD/KRW move", fx_numeric, "%"),
        ]
    )
    if fx_display == "N/A":
        hero_metrics = hero_metrics.replace("nan%", "N/A")

    st.markdown(
        (
            f'<div class="decision-hero" style="--decision-hero-accent: {accent};">'
            '<div class="decision-hero__copy">'
            '<div class="decision-hero__eyebrow">Macro regime</div>'
            f'<div class="decision-hero__title">{html.escape(regime)}</div>'
            f'<div class="decision-hero__subtitle">{html.escape(regime_subtitle)}</div>'
            f'<div class="decision-hero__chips">{"".join(chips)}</div>'
            "</div>"
            f'<div class="decision-hero__stats">{hero_metrics}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_macro_tile(
    regime: str,
    growth_val: float | None = None,
    inflation_val: float | None = None,
    fx_change: float | None = None,
    is_provisional: bool = False,
    theme_mode: str = "dark",
) -> None:
    """Backward-compatible wrapper for the newer decision hero."""
    render_decision_hero(
        regime=regime,
        regime_is_confirmed=regime != "Indeterminate",
        growth_val=growth_val,
        inflation_val=inflation_val,
        fx_change=fx_change,
        is_provisional=is_provisional,
        theme_mode=theme_mode,
    )


def render_status_card_row(
    *,
    current_regime: str,
    regime_is_confirmed: bool,
    price_status: str,
    macro_status: str,
    yield_curve_status: str | None = None,
) -> None:
    """Render consistent status cards under the hero section."""
    regime_detail = "Confirmed" if regime_is_confirmed else "Provisional"
    yield_inverted = str(yield_curve_status or "").strip().lower() == "inverted"

    cards = [
        _render_card_html(
            eyebrow="Current regime",
            value=current_regime,
            detail=regime_detail,
            tone="info" if current_regime == "Indeterminate" else "success",
        ),
        _render_card_html(
            eyebrow="Market data",
            value=price_status,
            detail="Warehouse / live price path",
            tone=_status_tone(price_status),
        ),
        _render_card_html(
            eyebrow="Macro data",
            value=macro_status,
            detail="Monthly macro warehouse",
            tone=_status_tone(macro_status),
        ),
    ]

    if yield_curve_status:
        cards.append(
            _render_card_html(
                eyebrow="Yield curve",
                value="Inverted" if yield_inverted else "Normal",
                detail="3Y government bond versus base rate",
                tone="warning" if yield_inverted else "success",
            )
        )

    _render_cards_grid(cards, "status-card-grid")


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


def render_rs_scatter(
    signals: Sequence,
    *,
    height: int = 680,
    margin: dict[str, int] | None = None,
    theme_mode: str = "dark",
) -> go.Figure:
    """Render a relative-strength scatter plot."""
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)
    action_colors = get_action_colors(theme_mode)

    x_vals: list[float] = []
    y_vals: list[float] = []
    texts: list[str] = []
    colors: list[str] = []
    hovers: list[str] = []

    for signal in signals:
        rs = _safe_float(getattr(signal, "rs", None))
        rs_ma = _safe_float(getattr(signal, "rs_ma", None))
        if getattr(signal, "action", "N/A") == "N/A":
            continue
        if rs is None or rs_ma is None:
            continue

        x_vals.append(rs)
        y_vals.append(rs_ma)
        texts.append(signal.sector_name.split(" ")[-1])
        colors.append(action_colors.get(signal.action, tokens["text_muted"]))
        hovers.append(
            "<b>{}</b><br>Action: {}<br>RS: {:.4f}<br>RS MA: {:.4f}<br>RSI(D): {:.1f}<br>"
            "Trend: {}<br>Alerts: {}".format(
                html.escape(signal.sector_name),
                html.escape(signal.action),
                rs,
                rs_ma,
                float(signal.rsi_d),
                "Healthy" if signal.trend_ok else "Weakening",
                html.escape(", ".join(signal.alerts) or "None"),
            )
        )

    fig = go.Figure()
    axis_range = None
    if x_vals:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                text=texts,
                textposition="top center",
                marker=dict(
                    color=colors,
                    size=12,
                    line=dict(width=1, color=tokens["surface"]),
                ),
                hovertext=hovers,
                hoverinfo="text",
            )
        )

        mn_raw = min(min(x_vals), min(y_vals))
        mx_raw = max(max(x_vals), max(y_vals))
        span = max(mx_raw - mn_raw, 1e-6)
        pad = span * 0.06
        mn = mn_raw - pad
        mx = mx_raw + pad
        axis_range = [mn, mx]
        fig.add_shape(
            type="line",
            x0=mn,
            y0=mn,
            x1=mx,
            y1=mx,
            line=dict(color=tokens["border"], dash="dot", width=1.5),
            layer="below",
        )
    else:
        fig.add_annotation(
            text="No valid RS / RS MA points are available. Check benchmark coverage first.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": tokens["text"]},
        )

    fig.update_layout(
        **template,
        title="Relative Strength versus RS Moving Average",
        xaxis_title="RS",
        yaxis_title="RS MA",
        height=height,
        showlegend=False,
    )
    fig.update_layout(margin=margin or dict(l=72, r=32, t=64, b=64))
    if axis_range:
        fig.update_xaxes(range=axis_range, constrain="domain")
        fig.update_yaxes(
            range=axis_range,
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        )

    return fig


def render_rs_momentum_bar(signals: Sequence, theme_mode: str = "dark") -> go.Figure:
    """Render a horizontal bar chart of RS divergence."""
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)
    action_colors = get_action_colors(theme_mode)

    filtered = []
    for signal in signals:
        if getattr(signal, "action", "N/A") == "N/A":
            continue
        rs = _safe_float(getattr(signal, "rs", None))
        rs_ma = _safe_float(getattr(signal, "rs_ma", None))
        if rs is None or rs_ma in {None, 0.0}:
            continue
        filtered.append(signal)

    if not filtered:
        return go.Figure()

    def rs_div(signal) -> float:
        assert signal.rs_ma != 0
        return (signal.rs - signal.rs_ma) / signal.rs_ma * 100

    filtered_sorted = sorted(filtered, key=rs_div)
    names = [signal.sector_name.split(" ")[-1] for signal in filtered_sorted]
    values = [rs_div(signal) for signal in filtered_sorted]
    colors = [
        action_colors.get(signal.action, tokens["text_muted"]) for signal in filtered_sorted
    ]
    hovers = [
        "<b>{}</b><br>RS gap: {:+.2f}%<br>RS: {:.4f} / RS MA: {:.4f}<br>Action: {}".format(
            html.escape(signal.sector_name),
            rs_div(signal),
            signal.rs,
            signal.rs_ma,
            html.escape(signal.action),
        )
        for signal in filtered_sorted
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            hovertext=hovers,
            hoverinfo="text",
            text=[f"{value:+.2f}%" for value in values],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line=dict(color=tokens["border"], width=1.5))
    fig.update_layout(
        **template,
        title="RS gap by sector",
        xaxis_title="RS gap (%)",
        yaxis_title="",
        height=max(300, len(filtered_sorted) * 36 + 80),
        showlegend=False,
    )
    fig.update_xaxes(ticksuffix="%")
    return fig


def render_returns_heatmap(signals: Sequence, theme_mode: str = "dark") -> go.Figure:
    """Render a multi-period sector returns heatmap."""
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)

    periods = ["1W", "1M", "3M", "6M", "12M"]
    sector_names: list[str] = []
    z_values: list[list[float | None]] = []

    for signal in signals:
        if getattr(signal, "action", "N/A") == "N/A" or not getattr(signal, "returns", {}):
            continue
        sector_names.append(signal.sector_name.split()[-1])
        row = []
        for period in periods:
            raw = _safe_float(signal.returns.get(period))
            row.append(raw * 100 if raw is not None else None)
        z_values.append(row)

    if not sector_names:
        fig = go.Figure()
        fig.update_layout(**template, title="No return data available")
        return fig

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=periods,
            y=sector_names,
            colorscale=[
                [0.0, tokens["danger"]],
                [0.5, tokens["border"]],
                [1.0, tokens["success"]],
            ],
            zmid=0,
            texttemplate="%{z:.1f}",
            textfont={"size": 11},
            hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **template,
        title="Multi-period return heatmap (%)",
        height=max(300, len(sector_names) * 40),
    )
    return fig


def _resolve_heatmap_density_mode(
    *,
    month_count: int,
    row_count: int,
) -> dict[str, object]:
    """Return deterministic label/text settings for the analysis heatmap."""
    if month_count <= 36:
        label_step = 1
        tickangle = 0
        bottom_margin = 64
        tickfont_size = 11
    elif month_count <= 72:
        label_step = 1
        tickangle = -90
        bottom_margin = 128
        tickfont_size = 10
    else:
        label_step = max(1, math.ceil(month_count / 48))
        tickangle = -90
        bottom_margin = 128
        tickfont_size = 9

    show_cell_text = month_count <= 36 and (month_count * row_count) <= 432
    helper_text = ""
    if not show_cell_text:
        helper_text = "<br><sup>Hover or click a cell to inspect exact monthly return values.</sup>"

    return {
        "label_step": label_step,
        "tickangle": tickangle,
        "bottom_margin": bottom_margin,
        "tickfont_size": tickfont_size,
        "show_cell_text": show_cell_text,
        "helper_text": helper_text,
    }


def build_sector_strength_heatmap(
    heatmap_df: pd.DataFrame,
    *,
    selected_sector: str | None = None,
    selected_month: str | None = None,
    theme_mode: str = "dark",
    palette: str = "classic",
    title: str = "Monthly sector return",
    empty_message: str = "No monthly sector return data is available for the active filters.",
    helper_metric_label: str = "monthly return",
    hover_value_suffix: str = "%",
) -> go.Figure:
    """Build a monthly sector heatmap used by the analysis canvas."""
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)

    if heatmap_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=empty_message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": tokens["text"]},
        )
        fig.update_layout(**template, title=title, height=320)
        return fig

    x_labels = [str(col) for col in heatmap_df.columns]
    y_labels = [str(idx) for idx in heatmap_df.index]
    x_positions = list(range(len(x_labels)))
    y_positions = list(range(len(y_labels)))
    density_mode = _resolve_heatmap_density_mode(
        month_count=len(x_labels),
        row_count=len(y_labels),
    )
    helper_text = str(density_mode["helper_text"] or "")
    if helper_text:
        helper_text = (
            helper_text.replace("exact monthly return values", f"exact {helper_metric_label} values")
            .replace("monthly return values", f"{helper_metric_label} values")
        )
    label_step = int(density_mode["label_step"])
    ticktext = [
        label if idx % label_step == 0 else ""
        for idx, label in enumerate(x_labels)
    ]
    z_values = heatmap_df.fillna(float("nan")).to_numpy()
    customdata = [
        [[x_labels[col_idx], y_labels[row_idx]] for col_idx in range(len(x_labels))]
        for row_idx in range(len(y_labels))
    ]
    colorscale = get_analysis_heatmap_colorscale(
        theme_mode=theme_mode,
        palette=palette,
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_positions,
            y=y_positions,
            customdata=customdata,
            colorscale=colorscale,
            zmid=0,
            zmin=float(pd.DataFrame(z_values).min().min(skipna=True))
            if pd.notna(pd.DataFrame(z_values).min().min(skipna=True))
            else -1.0,
            zmax=float(pd.DataFrame(z_values).max().max(skipna=True))
            if pd.notna(pd.DataFrame(z_values).max().max(skipna=True))
            else 1.0,
            xgap=1,
            ygap=1,
            texttemplate="%{z:.1f}" if bool(density_mode["show_cell_text"]) else None,
            textfont={"size": 10},
            colorbar=dict(
                orientation="h",
                y=1.12,
                x=1.0,
                xanchor="right",
                len=0.28,
                thickness=16,
                title=dict(text="%"),
            ),
            hovertemplate=f"%{{customdata[1]}}<br>%{{customdata[0]}}: %{{z:.1f}}{hover_value_suffix}<extra></extra>",
        )
    )

    fig.update_layout(
        **template,
        title=f"{title}{helper_text}",
        height=max(360, len(y_labels) * 34 + 130),
        clickmode="event+select",
        dragmode="select",
    )
    fig.update_layout(margin=dict(l=108, r=28, t=84, b=int(density_mode["bottom_margin"])))
    fig.update_xaxes(
        tickmode="array",
        tickvals=x_positions,
        ticktext=ticktext,
        side="bottom",
        showgrid=False,
        tickangle=int(density_mode["tickangle"]),
        tickfont={"size": int(density_mode["tickfont_size"])},
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_positions,
        ticktext=y_labels,
        autorange="reversed",
        showgrid=False,
    )

    if selected_sector in y_labels:
        row_index = y_labels.index(str(selected_sector))
        fig.add_shape(
            type="rect",
            x0=-0.5,
            x1=len(x_labels) - 0.5,
            y0=row_index - 0.5,
            y1=row_index + 0.5,
            line=dict(width=0),
            fillcolor=get_chart_tokens(theme_mode)["selection_row_fill"],
            layer="below",
        )
    if selected_month in x_labels:
        col_index = x_labels.index(str(selected_month))
        fig.add_shape(
            type="rect",
            x0=col_index - 0.5,
            x1=col_index + 0.5,
            y0=-0.5,
            y1=len(y_labels) - 0.5,
            line=dict(width=0),
            fillcolor=get_chart_tokens(theme_mode)["selection_col_fill"],
            layer="below",
        )
    if selected_sector in y_labels and selected_month in x_labels:
        row_index = y_labels.index(str(selected_sector))
        col_index = x_labels.index(str(selected_month))
        fig.add_shape(
            type="rect",
            x0=col_index - 0.5,
            x1=col_index + 0.5,
            y0=row_index - 0.5,
            y1=row_index + 0.5,
            line=dict(color=get_chart_tokens(theme_mode)["selection_outline"], width=2),
            fillcolor="rgba(0,0,0,0)",
        )

    return fig


def render_cycle_timeline_panel(
    *,
    segments: Sequence[Mapping[str, object]],
    selected_cycle_phase: str,
    theme_mode: str = "dark",
) -> str:
    """Render cycle chips plus a full-width regime timeline card."""
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)
    signal_tokens = get_signal_tokens(theme_mode)
    palette_markup = "".join(
        (
            '<span class="cycle-palette__item">'
            f'<span class="cycle-palette__swatch {css_class}"></span>'
            f"<span>{html.escape(label)}</span>"
            "</span>"
        )
        for label, css_class in CYCLE_REGIME_PALETTE_LABELS
    )

    st.markdown('<div class="phase-chip-row"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="cycle-palette"><span class="cycle-palette__label">Cycle palette</span>{palette_markup}</div>',
        unsafe_allow_html=True,
    )
    selected_phase = st.segmented_control(
        "Cycle phase",
        options=CYCLE_PHASE_ORDER,
        default=selected_cycle_phase if selected_cycle_phase in CYCLE_PHASE_ORDER else "ALL",
        format_func=format_cycle_phase_label,
        selection_mode="single",
        key="cycle_phase_segmented_control",
        label_visibility="collapsed",
        width="stretch",
    )

    fig = go.Figure()
    if not segments:
        fig.add_annotation(
            text="No regime history is available for the selected window.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": tokens["text"]},
        )
        fig.update_layout(**template, title="Cycle timeline (monthly)", height=240)
        fig.update_xaxes(title="", type="date", tickformat="%Y-%m", dtick="M1", tickangle=-45)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        return str(selected_phase or "ALL")

    phase_styles = signal_tokens["cycle_phase_styles"]

    y0 = 0.18
    y1 = 0.82

    for segment in segments:
        phase_key = str(segment.get("phase_key", ""))
        start = pd.Timestamp(segment.get("start"))
        end = pd.Timestamp(segment.get("end"))
        if pd.isna(start) or pd.isna(end):
            continue
        is_selected = selected_phase not in {None, "", "ALL"} and phase_key == selected_phase
        is_current = bool(segment.get("is_current", False))
        style = phase_styles.get(
            phase_key,
            phase_styles["INDETERMINATE"],
        )
        if phase_key == "INDETERMINATE":
            segment_state = "Indeterminate"
        elif is_selected:
            segment_state = "Selected"
        elif is_current:
            segment_state = "Current"
        else:
            segment_state = "Context"
        line_width = 4 if is_selected else 3 if is_current else 1.25
        line_color = (
            tokens["text"]
            if is_selected
            else signal_tokens["cycle_current_line"] if is_current else style["line"]
        )
        trace_opacity = 1.0 if is_selected else 0.92 if is_current else 0.62
        end_display = end

        fig.add_trace(
            go.Scatter(
                x=[start, end_display, end_display, start, start],
                y=[y0, y0, y1, y1, y0],
                mode="lines",
                fill="toself",
                fillcolor=style["fill"],
                line=dict(color=line_color, width=line_width),
                opacity=trace_opacity,
                hovertemplate=(
                    f"{html.escape(str(segment.get('label', phase_key)))}"
                    "<br>%{customdata[0]} -> %{customdata[1]}"
                    "<br>%{customdata[2]}"
                    "<br>Status: %{customdata[3]}<extra></extra>"
                ),
                customdata=[[
                    start.strftime("%Y-%m"),
                    end.strftime("%Y-%m"),
                    str(segment.get("summary", "No sector summary available.")),
                    segment_state,
                ]] * 5,
                name=str(segment.get("label", phase_key)),
                showlegend=False,
            )
        )

    fig.update_layout(
        **template,
        title="Cycle timeline (monthly)",
        height=270,
    )
    fig.update_layout(margin=dict(l=24, r=24, t=52, b=82))
    fig.update_xaxes(
        title="",
        type="date",
        tickformat="%Y-%m",
        dtick="M1",
        tickangle=-45,
        showgrid=True,
        ticklabelmode="period",
    )
    fig.update_yaxes(title="", visible=False, range=[0, 1], fixedrange=True)

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    return str(selected_phase or "ALL")


def build_sector_detail_figure(
    series_df: pd.DataFrame,
    *,
    selected_sector: str,
    benchmark_label: str | None = None,
    comparison_sectors: Sequence[str] | None = None,
    selected_month: str | None = None,
    theme_mode: str = "dark",
) -> go.Figure:
    """Build the linked sector detail line chart."""
    template = get_plotly_template(theme_mode)
    tokens = get_theme_tokens(theme_mode)

    fig = go.Figure()
    if series_df.empty or selected_sector not in series_df.columns:
        fig.add_annotation(
            text="No sector detail data is available for the current selection.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": tokens["text"]},
        )
        fig.update_layout(**template, title="Selected sector detail", height=360)
        return fig

    compare_order: list[str] = []
    if benchmark_label and benchmark_label in series_df.columns and benchmark_label != selected_sector:
        compare_order.append(benchmark_label)
    for sector in comparison_sectors or []:
        if sector in series_df.columns and sector not in compare_order and sector != selected_sector:
            compare_order.append(sector)

    muted_colors = list(get_chart_tokens(theme_mode)["muted_lines"])
    for index, sector in enumerate(compare_order):
        fig.add_trace(
            go.Scatter(
                x=series_df.index,
                y=series_df[sector],
                mode="lines",
                line=dict(color=muted_colors[index % len(muted_colors)], width=2, dash="solid" if sector != benchmark_label else "dash"),
                name=sector,
                hovertemplate=f"{html.escape(sector)}<br>%{{x|%Y-%m-%d}}: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=series_df.index,
            y=series_df[selected_sector],
            mode="lines",
            line=dict(color=tokens["primary"], width=3.5),
            name=selected_sector,
            hovertemplate=f"{html.escape(selected_sector)}<br>%{{x|%Y-%m-%d}}: %{{y:.2f}}<extra></extra>",
        )
    )

    if selected_month:
        month_mask = series_df.index.to_period("M").astype(str) == str(selected_month)
        if month_mask.any():
            selected_points = series_df.loc[month_mask]
            fig.add_trace(
                go.Scatter(
                    x=selected_points.index,
                    y=selected_points[selected_sector],
                    mode="markers",
                    marker=dict(color=tokens["primary"], size=8, line=dict(color=tokens["surface"], width=1)),
                    name="Pinned month",
                    showlegend=False,
                    hovertemplate=f"{html.escape(selected_sector)}<br>%{{x|%Y-%m-%d}}: %{{y:.2f}}<extra></extra>",
                )
            )

    fig.update_layout(
        **template,
        title="Selected sector detail",
        height=400,
    )
    fig.update_layout(margin=dict(l=48, r=20, t=58, b=48))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_yaxes(title="Indexed performance")
    fig.update_xaxes(title="")
    return fig


def render_sector_detail_panel(
    *,
    ranking_rows: Sequence[Mapping[str, object]],
    detail_figure: go.Figure,
    selected_sector: str,
    selected_range_preset: str,
    preset_options: Sequence[str] = ("1Y", "3Y", "5Y", "ALL"),
) -> tuple[str, str]:
    """Render the linked sector ranking list and detail chart."""
    selected_sector_value = str(selected_sector or "")
    normalized_preset = normalize_range_preset(selected_range_preset)
    ranking_col, chart_col = st.columns([1.05, 2.35], gap="large")

    with ranking_col:
        st.markdown(
            (
                '<div class="sector-rank-list__header">'
                '<div class="sector-rank-list__eyebrow">Sector ranking</div>'
                '<div class="sector-rank-list__title">Current return rank in the active window.</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        for rank, row in enumerate(ranking_rows, start=1):
            sector_label = str(row.get("sector", ""))
            return_pct = _safe_float(row.get("return_pct"))
            is_selected = sector_label == selected_sector_value
            button_col, metric_col = st.columns([3.2, 1], gap="small")
            with button_col:
                clicked = st.button(
                    f"{rank}. {sector_label}",
                    key=f"sector_rank_button_{rank}_{sector_label}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                )
                if clicked:
                    selected_sector_value = sector_label
            with metric_col:
                tone = "positive" if (return_pct or 0.0) >= 0 else "negative"
                metric_text = f"{return_pct:+.1f}%" if return_pct is not None else "N/A"
                st.markdown(
                    (
                        '<div class="sector-rank-list__metric" '
                        f'data-tone="{tone}" data-selected="{str(is_selected).lower()}">'
                        f"{html.escape(metric_text)}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

    with chart_col:
        preset_choice = st.segmented_control(
            "Detail range",
            options=list(preset_options),
            default=normalized_preset if normalized_preset in preset_options else None,
            format_func=format_range_preset_label,
            label_visibility="collapsed",
            width="content",
        )
        st.plotly_chart(detail_figure, width="stretch", config={"displayModeBar": False})

    return selected_sector_value, normalize_range_preset(str(preset_choice or normalized_preset))


def render_action_summary(signals: Sequence, theme_mode: str = "dark") -> None:
    """Render action counts plus a distribution chart."""
    if not signals:
        st.info("No signal data available.")
        return

    full_order = ["Strong Buy", "Watch", "Hold", "Avoid", "N/A"]
    action_counts = {action: 0 for action in full_order}
    for signal in signals:
        action_counts[signal.action] = action_counts.get(signal.action, 0) + 1

    display_order = (
        full_order
        if action_counts.get("N/A", 0) > 0
        else [action for action in full_order if action != "N/A"]
    )
    cards = [
        _render_card_html(
            eyebrow="Universe",
            value=str(sum(action_counts.values())),
            detail="Filtered sectors in view",
            tone="info",
        )
    ]
    for action in display_order:
        cards.append(
            _render_card_html(
                eyebrow=action,
                value=str(action_counts[action]),
                detail=format_action_label(action),
                tone=_action_tone(action),
            )
        )
    _render_cards_grid(cards, "summary-kpi-grid")

    template = get_plotly_template(theme_mode)
    action_colors = get_action_colors(theme_mode)
    fig = go.Figure(
        data=go.Bar(
            x=display_order,
            y=[action_counts[action] for action in display_order],
            marker_color=[action_colors[action] for action in display_order],
            text=[str(action_counts[action]) for action in display_order],
            textposition="outside",
        )
    )
    fig.update_layout(
        **template,
        title="Action distribution",
        xaxis_title="",
        yaxis_title="Sector count",
        height=280,
        showlegend=False,
    )
    max_count = max(action_counts.values()) if action_counts else 1
    fig.update_yaxes(dtick=1, range=[0, max_count * 1.25])
    st.plotly_chart(fig, width="stretch")


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
