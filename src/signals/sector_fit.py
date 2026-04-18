"""Helpers for loading additive sector-fit evidence from validation artifacts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _latest_sector_fit_artifact() -> Path | None:
    target = Path("docs") / "regime-validity-dashboard-parity-current-rankings.csv"
    return target if target.exists() else None


def load_sector_fit_artifact(path: Path | None = None) -> pd.DataFrame:
    """Load the latest sector-fit ranking artifact if available."""
    target = path or _latest_sector_fit_artifact()
    if target is None or not target.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(target)
    except Exception:
        return pd.DataFrame()
    required = {"lag_months", "regime", "rank", "code", "avg_excess_pct"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame["code"] = frame["code"].astype(str)
    frame["regime"] = frame["regime"].astype(str)
    return frame


def build_sector_fit_lookup(
    *,
    regime: str,
    lag_months: int = 0,
) -> dict[str, dict[str, object]]:
    """Return additive sector-fit metadata keyed by sector code."""
    frame = load_sector_fit_artifact()
    if frame.empty or not regime or regime == "Indeterminate":
        return {}

    scoped = frame[(frame["lag_months"] == lag_months) & (frame["regime"] == str(regime))].copy()
    if scoped.empty:
        return {}
    total = int(len(scoped))
    top_half_cutoff = max(total // 2, 1)
    lag_scope = frame[frame["lag_months"] == lag_months].copy()

    lookup: dict[str, dict[str, object]] = {}
    for row in scoped.itertuples(index=False):
        code = str(row.code)
        sector_rows = lag_scope[lag_scope["code"] == code].copy()
        strong_regimes = [
            str(value)
            for value in sector_rows.loc[sector_rows["rank"] <= top_half_cutoff, "regime"].tolist()
        ]
        strong_regimes = list(dict.fromkeys(strong_regimes))
        note_parts: list[str] = []
        if int(row.rank) <= top_half_cutoff:
            note_parts.append("Top-half empirical fit")
        if len(strong_regimes) > 1:
            note_parts.append("Cross-regime leader")
        lookup[code] = {
            "sector_fit_rank": int(row.rank),
            "sector_fit_total": total,
            "sector_fit_avg_excess_pct": float(row.avg_excess_pct),
            "sector_fit_note": " | ".join(note_parts),
            "sector_fit_cross_regimes": tuple(strong_regimes),
        }
    return lookup
