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
    "ALL_ACTION_OPTION",
    "LEGACY_ALL_ACTION_OPTIONS",
    "ACTION_LABELS",
    "ACTION_PRIORITY",
    "REGIME_SUBLABELS",
    "RANGE_PRESET_MONTHS",
    "RANGE_PRESET_LABELS",
    "LEGACY_RANGE_PRESET_MAP",
    "CYCLE_PHASE_ORDER",
    "CYCLE_PHASE_LABELS",
    "CYCLE_REGIME_PALETTE_LABELS",
    "HEATMAP_PALETTE_OPTIONS",
    "HEATMAP_PALETTE_LABELS",
    "format_action_label",
    "format_cycle_phase_label",
    "format_range_preset_label",
    "format_heatmap_palette_label",
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
]
