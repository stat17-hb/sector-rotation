"""Pure dashboard analysis helpers extracted from app.py."""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from src.dashboard.metrics import compute_rs_divergence_pct
from src.ui.components import format_cycle_phase_label


ACTION_PRIORITY: dict[str, int] = {
    "Strong Buy": 0,
    "Watch": 1,
    "Hold": 2,
    "Avoid": 3,
    "N/A": 4,
}


def _build_sector_name_map(
    *,
    signals: list[Any],
    sector_prices: pd.DataFrame,
    benchmark_code: str,
    benchmark_label: str = "KOSPI",
) -> dict[str, str]:
    """Build a code -> display name map from the current universe."""
    names: dict[str, str] = {str(benchmark_code): str(benchmark_label)}
    for signal in signals:
        names[str(signal.index_code)] = str(signal.sector_name)

    if not sector_prices.empty and "index_code" in sector_prices.columns and "index_name" in sector_prices.columns:
        latest_names = (
            sector_prices.reset_index()
            .sort_values(sector_prices.index.name or "trade_date")
            .drop_duplicates("index_code", keep="last")
        )
        for _, row in latest_names.iterrows():
            code = str(row["index_code"])
            if code not in names or not names[code].strip():
                names[code] = str(row["index_name"])
    return names


def _build_prices_wide(
    *,
    sector_prices: pd.DataFrame,
    sector_name_map: dict[str, str],
) -> pd.DataFrame:
    """Pivot long sector prices into a wide trade-date index frame."""
    if sector_prices.empty:
        return pd.DataFrame()

    prices_reset = sector_prices.reset_index().copy()
    date_col = sector_prices.index.name or "trade_date"
    prices_reset[date_col] = pd.to_datetime(prices_reset[date_col])
    prices_reset["series_name"] = prices_reset["index_code"].astype(str).map(
        lambda code: sector_name_map.get(code, code)
    )
    prices_wide = (
        prices_reset.pivot_table(
            index=date_col,
            columns="series_name",
            values="close",
            aggfunc="last",
        )
        .sort_index()
        .ffill()
    )
    prices_wide.index = pd.to_datetime(prices_wide.index)
    return prices_wide


def _build_monthly_sector_returns(
    *,
    prices_wide: pd.DataFrame,
    sector_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build monthly closes/returns from the full available analysis history."""
    monthly_close_full = prices_wide[sector_columns].resample("ME").last() if sector_columns else pd.DataFrame()
    monthly_returns_full = monthly_close_full.pct_change() * 100 if not monthly_close_full.empty else pd.DataFrame()
    return monthly_close_full, monthly_returns_full


def _build_monthly_return_views(
    *,
    prices_wide: pd.DataFrame,
    sector_columns: list[str],
    benchmark_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build absolute and excess monthly return views for the analysis canvas."""
    monthly_close_full, monthly_returns_full = _build_monthly_sector_returns(
        prices_wide=prices_wide,
        sector_columns=sector_columns,
    )
    benchmark_monthly_return = pd.Series(dtype="float64")
    if benchmark_label in prices_wide.columns:
        benchmark_monthly_close = prices_wide[[benchmark_label]].resample("ME").last()
        benchmark_monthly_return = benchmark_monthly_close[benchmark_label].pct_change() * 100
    monthly_excess_returns_full = (
        monthly_returns_full.sub(benchmark_monthly_return, axis=0)
        if not monthly_returns_full.empty and not benchmark_monthly_return.empty
        else pd.DataFrame(index=monthly_returns_full.index, columns=monthly_returns_full.columns)
    )
    return monthly_close_full, monthly_returns_full, benchmark_monthly_return, monthly_excess_returns_full


def _filter_monthly_frame_for_analysis(
    *,
    monthly_frame: pd.DataFrame,
    start_date: date,
    end_date: date,
    selected_cycle_phase: str,
    phase_by_month: pd.Series,
) -> pd.DataFrame:
    """Filter a monthly frame to the visible analysis range and cycle phase."""
    if monthly_frame.empty:
        return pd.DataFrame()

    start_ts = pd.Timestamp(start_date).to_period("M").to_timestamp("M")
    end_ts = pd.Timestamp(end_date).normalize()
    filtered = monthly_frame.loc[
        (monthly_frame.index >= start_ts)
        & (monthly_frame.index <= end_ts)
    ]
    if filtered.empty or selected_cycle_phase == "ALL" or phase_by_month.empty:
        return filtered

    valid_months = set(
        phase_by_month[phase_by_month.astype(str) == str(selected_cycle_phase)]
        .index.to_period("M")
        .astype(str)
    )
    if not valid_months:
        return filtered.iloc[0:0]
    return filtered.loc[
        filtered.index.to_period("M").astype(str).isin(valid_months)
    ]


def _build_heatmap_display(monthly_frame: pd.DataFrame) -> pd.DataFrame:
    """Transpose a monthly frame into sector x month display shape."""
    if monthly_frame.empty:
        return pd.DataFrame()
    display = monthly_frame.copy()
    display.index = display.index.to_period("M").astype(str)
    return display.T


def _extract_heatmap_selection(heatmap_event: Any) -> tuple[str, str] | None:
    """Return the selected (month, sector) pair from a Plotly heatmap selection."""
    if heatmap_event is None:
        return None
    selection_state = getattr(heatmap_event, "selection", None)
    if selection_state is None and isinstance(heatmap_event, dict):
        selection_state = heatmap_event.get("selection", {})
    if not selection_state:
        return None

    selected_points = list(getattr(selection_state, "points", []) or selection_state.get("points", []))
    if not selected_points:
        return None
    point = selected_points[-1]
    customdata = point.get("customdata") if isinstance(point, dict) else None
    if isinstance(customdata, (list, tuple)) and len(customdata) >= 2:
        return str(customdata[0]), str(customdata[1])
    return None


def _build_cycle_segments(
    *,
    macro_result: pd.DataFrame,
    monthly_close: pd.DataFrame,
) -> tuple[list[dict[str, object]], pd.Series]:
    """Split macro regime history into early/late contiguous cycle segments."""
    if macro_result.empty:
        return [], pd.Series(dtype=object)

    regime_col = "confirmed_regime" if "confirmed_regime" in macro_result.columns else "regime"
    regime_series = macro_result[regime_col].copy()
    regime_series.index = pd.to_datetime(regime_series.index).to_period("M").to_timestamp(how="start")
    regime_series = regime_series.sort_index()
    if regime_series.empty:
        return [], pd.Series(dtype=object)

    phase_by_month = pd.Series(index=regime_series.index, dtype=object)
    segments: list[dict[str, object]] = []
    color_key = {
        "Recovery": "RECOVERY",
        "Expansion": "EXPANSION",
        "Slowdown": "SLOWDOWN",
        "Contraction": "CONTRACTION",
    }

    for _, run in regime_series.groupby((regime_series != regime_series.shift()).cumsum()):
        regime_value = run.iloc[0]
        if pd.isna(regime_value):
            continue
        regime = str(regime_value)
        run_months = list(run.index)
        if not run_months:
            continue

        split_index = max(1, int((len(run_months) + 1) / 2))
        partitions = [
            ("EARLY", run_months[:split_index]),
            ("LATE", run_months[split_index:]),
        ]
        for stage, months in partitions:
            if not months:
                continue

            phase_key = f"{color_key.get(regime, regime.upper())}_{stage}" if regime in color_key else regime.upper()
            phase_label = format_cycle_phase_label(phase_key) if phase_key in {
                "RECOVERY_EARLY",
                "RECOVERY_LATE",
                "EXPANSION_EARLY",
                "EXPANSION_LATE",
                "SLOWDOWN_EARLY",
                "SLOWDOWN_LATE",
                "CONTRACTION_EARLY",
                "CONTRACTION_LATE",
            } else regime
            start_month = months[0]
            end_month = months[-1]
            phase_by_month.loc[months] = phase_key
            start_date = pd.Timestamp(start_month).to_period("M").to_timestamp(how="start")
            end_date = pd.Timestamp(end_month).to_period("M").to_timestamp(how="end")

            top_summary = "No sector summary available."
            if not monthly_close.empty:
                segment_slice = monthly_close.loc[
                    (monthly_close.index >= start_date.normalize())
                    & (monthly_close.index <= end_date.normalize() + pd.offsets.MonthEnd(0))
                ]
                if len(segment_slice) >= 2:
                    segment_return = segment_slice.iloc[-1] / segment_slice.iloc[0] - 1
                    segment_return = segment_return.dropna().sort_values(ascending=False)
                    if not segment_return.empty:
                        top_sector = str(segment_return.index[0])
                        top_summary = f"Top sector: {top_sector} ({segment_return.iloc[0] * 100:+.1f}%)"

            segments.append(
                {
                    "phase_key": phase_key,
                    "label": phase_label,
                    "regime": regime,
                    "start": start_date,
                    "end": end_date,
                    "summary": top_summary,
                }
            )

    if segments:
        current_phase_key = str(phase_by_month.dropna().iloc[-1]) if not phase_by_month.dropna().empty else ""
        for segment in segments:
            segment["is_current"] = str(segment.get("phase_key", "")) == current_phase_key

    return segments, phase_by_month


def _filter_prices_for_phase(
    *,
    prices_wide: pd.DataFrame,
    phase_by_month: pd.Series,
    selected_cycle_phase: str,
) -> pd.DataFrame:
    """Filter daily prices to the months belonging to the selected cycle phase."""
    if prices_wide.empty or not selected_cycle_phase or selected_cycle_phase == "ALL":
        return prices_wide
    if phase_by_month.empty:
        return prices_wide.iloc[0:0]

    valid_months = set(
        phase_by_month[phase_by_month.astype(str) == str(selected_cycle_phase)]
        .index.to_period("M")
        .astype(str)
    )
    if not valid_months:
        return prices_wide.iloc[0:0]
    mask = prices_wide.index.to_period("M").astype(str).isin(valid_months)
    return prices_wide.loc[mask]


def _rs_divergence_pct(signal: Any) -> float:
    value = compute_rs_divergence_pct(signal)
    return float("nan") if value is None else float(value)


def _top_pick_sort_key(signal: Any) -> tuple[int, float]:
    rs_div = _rs_divergence_pct(signal)
    rs_div_rank = -rs_div if not pd.isna(rs_div) else float("inf")
    return ACTION_PRIORITY.get(signal.action, 99), rs_div_rank
