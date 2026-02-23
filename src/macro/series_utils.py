"""
Macro series selection helpers for app wiring.

These functions keep provider-specific series IDs out of UI code and make
regime inputs deterministic across ECOS/KOSIS config changes.
"""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def _is_enabled(cfg: Mapping | None) -> bool:
    if not cfg:
        return False
    return bool(cfg.get("enabled", True))


def build_enabled_ecos_config(raw_cfg: Mapping | None) -> dict[str, dict]:
    """Build ECOS loader config, excluding disabled entries."""
    result: dict[str, dict] = {}
    for alias, cfg in (raw_cfg or {}).items():
        if not _is_enabled(cfg):
            continue
        row = {
            "stat_code": cfg["stat_code"],
            "item_code": cfg.get("item_code"),
        }
        if cfg.get("item_codes"):
            row["item_codes"] = list(cfg["item_codes"])
        if cfg.get("cycle"):
            row["cycle"] = str(cfg["cycle"])
        result[str(alias)] = row
    return result


def build_enabled_kosis_config(raw_cfg: Mapping | None) -> dict[str, dict]:
    """Build KOSIS loader config, excluding disabled entries."""
    result: dict[str, dict] = {}
    for alias, cfg in (raw_cfg or {}).items():
        if not _is_enabled(cfg):
            continue
        row = {
            "org_id": cfg["org_id"],
            "tbl_id": cfg["tbl_id"],
            "item_id": cfg["item_id"],
        }
        if cfg.get("obj_params"):
            row["obj_params"] = dict(cfg["obj_params"])
        result[str(alias)] = row
    return result


def _extract_kosis_series(macro_df: pd.DataFrame, cfg: Mapping | None) -> pd.Series:
    if not _is_enabled(cfg):
        return pd.Series(dtype=float)
    expected = f"{cfg['org_id']}/{cfg['tbl_id']}/{cfg['item_id']}"
    mask = macro_df["series_id"].astype(str) == expected
    if not mask.any():
        return pd.Series(dtype=float)
    values = macro_df.loc[mask, "value"].copy().sort_index()
    if values.index.has_duplicates:
        values = values.groupby(level=0).last()
    return values.astype("float64")


def _extract_ecos_series(macro_df: pd.DataFrame, cfg: Mapping | None) -> pd.Series:
    if not _is_enabled(cfg):
        return pd.Series(dtype=float)

    stat_code = str(cfg.get("stat_code", "")).strip()
    if not stat_code:
        return pd.Series(dtype=float)

    item_codes = cfg.get("item_codes") or []
    if not item_codes and cfg.get("item_code"):
        item_codes = [cfg["item_code"]]
    item_codes = [str(x).strip() for x in item_codes if str(x).strip()]

    series_ids = macro_df["series_id"].astype(str)
    if item_codes:
        expected = f"{stat_code}/{'/'.join(item_codes)}"
        mask = series_ids == expected
    else:
        mask = series_ids.str.startswith(f"{stat_code}/", na=False)

    if not mask.any():
        return pd.Series(dtype=float)

    values = macro_df.loc[mask, "value"].copy().sort_index()
    if values.index.has_duplicates:
        values = values.groupby(level=0).last()
    return values.astype("float64")


def extract_macro_series(
    macro_df: pd.DataFrame,
    macro_series_cfg: Mapping,
    alias: str,
) -> pd.Series:
    """Extract one configured macro series as a numeric series by alias."""
    if macro_df.empty:
        return pd.Series(dtype=float)
    if "series_id" not in macro_df.columns or "value" not in macro_df.columns:
        return pd.Series(dtype=float)

    kosis_cfg = (macro_series_cfg.get("kosis", {}) if macro_series_cfg else {}).get(alias)
    kosis_series = _extract_kosis_series(macro_df, kosis_cfg)
    if not kosis_series.empty:
        return kosis_series

    ecos_cfg = (macro_series_cfg.get("ecos", {}) if macro_series_cfg else {}).get(alias)
    ecos_series = _extract_ecos_series(macro_df, ecos_cfg)
    if not ecos_series.empty:
        return ecos_series

    return pd.Series(dtype=float)


def to_plotly_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a Plotly-serializable time index."""
    if df.empty:
        return df
    out = df.copy()
    if isinstance(out.index, pd.PeriodIndex):
        out.index = out.index.to_timestamp(how="end")
    return out

