"""Shared UI constants and helper functions."""
from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.theme import get_chart_tokens, get_signal_tokens
from src.dashboard.metrics import compute_rs_divergence_pct
from src.ui.copy import (
    ALL_ACTION_KEY,
    DEFAULT_UI_LOCALE,
    UiLocale,
    get_action_filter_label,
    get_action_label,
    get_all_action_label,
    get_cycle_palette_items,
    get_cycle_phase_label,
    get_decision_label,
    get_heatmap_palette_label,
    get_position_mode_label,
    get_range_preset_label,
    get_regime_subtitle,
    get_ui_text,
    is_all_action_filter,
    normalize_action_filter,
)
from src.ui.styles import (
    get_action_colors,
    get_plotly_template,
    get_theme_tokens,
)
ALL_ACTION_OPTION = get_all_action_label()
LEGACY_ALL_ACTION_OPTIONS = {
    ALL_ACTION_KEY,
    ALL_ACTION_OPTION,
    get_all_action_label("en"),
    "All",
    "전체",
}

ACTION_PRIORITY: dict[str, int] = {
    "Strong Buy": 0,
    "Watch": 1,
    "Hold": 2,
    "Avoid": 3,
    "N/A": 4,
}

POSITION_MODE_OPTIONS: tuple[str, ...] = ("all", "held", "new")

HELD_ACTION_PRIORITY: dict[str, int] = {
    "Strong Buy": 0,
    "Avoid": 1,
    "Hold": 2,
    "Watch": 3,
    "N/A": 4,
}

NEW_ACTION_PRIORITY: dict[str, int] = dict(ACTION_PRIORITY)

RANGE_PRESET_MONTHS: dict[str, int | None] = {
    "1Y": 12,
    "3Y": 36,
    "5Y": 60,
    "ALL": None,
    "CUSTOM": None,
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

CYCLE_REGIME_PALETTE_LABELS: list[tuple[str, str]] = [
    ("Recovery", "cycle-recovery"),
    ("Expansion", "cycle-expansion"),
    ("Slowdown", "cycle-slowdown"),
    ("Contraction", "cycle-contraction"),
    ("Indeterminate", "cycle-indeterminate"),
]

HEATMAP_PALETTE_OPTIONS: tuple[str, ...] = ("classic", "contrast", "blue_orange")


def format_action_label(action: str, locale: UiLocale = DEFAULT_UI_LOCALE) -> str:
    """Return explicit text for action status so color is never the only cue."""
    return get_action_label(action, locale)


def format_cycle_phase_label(phase_key: str, locale: UiLocale = DEFAULT_UI_LOCALE) -> str:
    """Return the user-facing label for a cycle phase selection key."""
    return get_cycle_phase_label(str(phase_key), locale)


def format_range_preset_label(preset: str, locale: UiLocale = DEFAULT_UI_LOCALE) -> str:
    """Return the compact display label for a range preset."""
    normalized = normalize_range_preset(preset)
    return get_range_preset_label(normalized, locale)


def format_heatmap_palette_label(palette: str, locale: UiLocale = DEFAULT_UI_LOCALE) -> str:
    """Return the user-facing label for a heatmap palette preset."""
    normalized = normalize_heatmap_palette(palette)
    return get_heatmap_palette_label(normalized, locale)


def normalize_position_mode(value: str | None) -> str:
    """Map unknown position filters back to the all-sectors default."""
    normalized = str(value or "all").strip().lower()
    return normalized if normalized in POSITION_MODE_OPTIONS else "all"


def format_position_mode_label(value: str, locale: UiLocale = DEFAULT_UI_LOCALE) -> str:
    """Return the user-facing label for a position-mode key."""
    normalized = normalize_position_mode(value)
    return get_position_mode_label(normalized, locale)


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
    return is_all_action_filter(value)


def _pct_value(value: float | int | None) -> float | None:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return numeric * 100


def _rs_divergence_pct(signal) -> float | None:
    return compute_rs_divergence_pct(signal)


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


def is_signal_held(signal, held_sectors: Sequence[str] | None = None) -> bool:
    """Return True when the signal belongs to the user's held-sector list."""
    held = {
        str(item).strip()
        for item in (held_sectors or [])
        if str(item).strip()
    }
    sector_name = str(getattr(signal, "sector_name", "")).strip()
    return bool(sector_name) and sector_name in held


def describe_signal_decision(
    signal,
    held_sectors: Sequence[str] | None = None,
    *,
    locale: UiLocale = DEFAULT_UI_LOCALE,
) -> dict[str, object]:
    """Return reusable decision-copy fields for one signal."""
    held = is_signal_held(signal, held_sectors)
    action = str(getattr(signal, "action", "N/A"))
    decision = get_decision_label(action, held=held, locale=locale)

    rs_div = _rs_divergence_pct(signal)
    ret_3m = _pct_value(getattr(signal, "returns", {}).get("3M"))
    volatility = _pct_value(getattr(signal, "volatility_20d", None))
    alerts = [str(item).strip() for item in getattr(signal, "alerts", []) if str(item).strip()]

    positive_parts: list[str] = []
    if bool(getattr(signal, "macro_fit", False)):
        positive_parts.append(get_ui_text("reason_regime_fit", locale))
    if rs_div is not None:
        positive_parts.append(get_ui_text("reason_rs_vs_trend", locale, value=rs_div))
    if bool(getattr(signal, "trend_ok", False)):
        positive_parts.append(get_ui_text("reason_trend_intact", locale))
    if ret_3m is not None:
        positive_parts.append(get_ui_text("reason_return_3m", locale, value=ret_3m))
    reason = " | ".join(positive_parts[:3]) if positive_parts else get_ui_text("reason_need_confirming_strength", locale)

    risk_parts: list[str] = []
    if not bool(getattr(signal, "macro_fit", False)):
        risk_parts.append(get_ui_text("risk_regime_mismatch", locale))
    if rs_div is not None and rs_div < 0:
        risk_parts.append(get_ui_text("risk_rs_below_trend", locale, value=rs_div))
    if not bool(getattr(signal, "trend_ok", False)):
        risk_parts.append(get_ui_text("risk_trend_weakened", locale))
    if volatility is not None and volatility >= 25.0:
        risk_parts.append(get_ui_text("risk_volatility", locale, value=volatility))
    risk_parts.extend(alerts[:2])
    deduped_risks: list[str] = []
    for item in risk_parts:
        if item and item not in deduped_risks:
            deduped_risks.append(item)
    risk = " | ".join(deduped_risks[:3]) if deduped_risks else get_ui_text("risk_none", locale)

    if action == "N/A":
        invalidation = get_ui_text("invalid_wait_for_data", locale)
    elif bool(getattr(signal, "macro_fit", False)) and bool(getattr(signal, "trend_ok", False)):
        invalidation = get_ui_text("invalid_break_regime_fit", locale)
    elif bool(getattr(signal, "macro_fit", False)):
        invalidation = get_ui_text("invalid_rs_below_trend", locale)
    elif held:
        invalidation = get_ui_text("invalid_regime_mismatch_persists", locale)
    else:
        invalidation = get_ui_text("invalid_promote_after_improve", locale)

    rs_trend = (
        get_ui_text("rs_trend_above", locale)
        if rs_div is not None and rs_div >= 0
        else get_ui_text("rs_trend_below", locale)
        if rs_div is not None
        else "N/A"
    )
    return_3m = f"{ret_3m:+.1f}%" if ret_3m is not None else "N/A"
    volatility_20d = f"{volatility:.1f}%" if volatility is not None else "N/A"
    alerts_text = ", ".join(alerts) if alerts else get_ui_text("alerts_none", locale)
    regime_fit = get_ui_text("regime_fit_yes", locale) if bool(getattr(signal, "macro_fit", False)) else get_ui_text("regime_fit_no", locale)
    conclusion = get_ui_text(
        "conclusion_template",
        locale,
        decision=decision,
        regime_fit=regime_fit,
        rs_trend=rs_trend,
        return_3m=return_3m,
        volatility_20d=volatility_20d,
        alerts_text=alerts_text,
    )

    return {
        "held": held,
        "decision": decision,
        "reason": reason,
        "risk": risk,
        "invalidation": invalidation,
        "regime_fit": regime_fit,
        "rs_trend": rs_trend,
        "return_3m": return_3m,
        "volatility_20d": volatility_20d,
        "alerts_text": alerts_text,
        "conclusion": conclusion,
    }


def filter_signals_for_display(
    signals: Sequence,
    *,
    filter_action: str | None = None,
    filter_regime_only: bool = False,
    current_regime: str | None = None,
    held_sectors: Sequence[str] | None = None,
    position_mode: str | None = None,
    show_alerted_only: bool = False,
) -> list:
    """Apply the dashboard's user-controlled filters to a signal sequence."""
    filtered = list(signals)
    if filter_regime_only and current_regime:
        filtered = [signal for signal in filtered if getattr(signal, "macro_regime", None) == current_regime]
    if filter_action and not _is_all_action_filter(filter_action):
        filtered = [signal for signal in filtered if getattr(signal, "action", None) == filter_action]

    normalized_position_mode = normalize_position_mode(position_mode)
    if normalized_position_mode == "held":
        filtered = [signal for signal in filtered if is_signal_held(signal, held_sectors)]
    elif normalized_position_mode == "new":
        filtered = [signal for signal in filtered if not is_signal_held(signal, held_sectors)]

    if show_alerted_only:
        filtered = [signal for signal in filtered if bool(getattr(signal, "alerts", []))]

    return filtered


def signal_display_sort_key(signal, held_sectors: Sequence[str] | None = None) -> tuple[object, ...]:
    """Return a stable display-order key that prioritizes held sectors first."""
    held = is_signal_held(signal, held_sectors)
    priority_map = HELD_ACTION_PRIORITY if held else NEW_ACTION_PRIORITY
    trailing_return = _safe_float(getattr(signal, "returns", {}).get("3M"))
    return (
        0 if held else 1,
        priority_map.get(str(getattr(signal, "action", "N/A")), 99),
        -(trailing_return if trailing_return is not None else -999.0),
        str(getattr(signal, "sector_name", "")),
    )


__all__ = [
    "html",
    "math",
    "Mapping",
    "Sequence",
    "date",
    "pd",
    "go",
    "st",
    "get_chart_tokens",
    "get_signal_tokens",
    "get_action_colors",
    "get_plotly_template",
    "get_theme_tokens",
    "UiLocale",
    "DEFAULT_UI_LOCALE",
    "ALL_ACTION_KEY",
    "ALL_ACTION_OPTION",
    "LEGACY_ALL_ACTION_OPTIONS",
    "ACTION_PRIORITY",
    "POSITION_MODE_OPTIONS",
    "HELD_ACTION_PRIORITY",
    "NEW_ACTION_PRIORITY",
    "RANGE_PRESET_MONTHS",
    "LEGACY_RANGE_PRESET_MAP",
    "CYCLE_PHASE_ORDER",
    "CYCLE_REGIME_PALETTE_LABELS",
    "HEATMAP_PALETTE_OPTIONS",
    "get_action_filter_label",
    "get_all_action_label",
    "get_cycle_palette_items",
    "get_regime_subtitle",
    "get_ui_text",
    "normalize_action_filter",
    "format_action_label",
    "format_cycle_phase_label",
    "format_position_mode_label",
    "format_range_preset_label",
    "format_heatmap_palette_label",
    "normalize_position_mode",
    "normalize_range_preset",
    "normalize_heatmap_palette",
    "get_analysis_heatmap_colorscale",
    "resolve_range_from_preset",
    "infer_range_preset",
    "_safe_float",
    "_is_all_action_filter",
    "_pct_value",
    "_rs_divergence_pct",
    "_status_tone",
    "_action_tone",
    "is_signal_held",
    "describe_signal_decision",
    "filter_signals_for_display",
    "signal_display_sort_key",
]
