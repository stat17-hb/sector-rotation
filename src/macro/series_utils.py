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


def build_enabled_fred_config(raw_cfg: Mapping | None) -> dict[str, dict]:
    """Build FRED loader config, excluding disabled entries."""
    result: dict[str, dict] = {}
    for alias, cfg in (raw_cfg or {}).items():
        if not _is_enabled(cfg):
            continue
        result[str(alias)] = {
            "series_id": cfg["series_id"],
            "transform": str(cfg.get("transform", "none")),
        }
    return result


def _extract_kosis_series(macro_df: pd.DataFrame, cfg: Mapping | None) -> pd.Series:
    if not _is_enabled(cfg):
        return pd.Series(dtype=float)
    _obj = cfg.get("obj_params") or {}
    _l1 = _obj.get("objL1") if isinstance(_obj, dict) else None
    expected = f"{cfg['org_id']}/{cfg['tbl_id']}/{cfg['item_id']}" + (f"/{_l1}" if _l1 else "")
    mask = macro_df["series_id"].astype(str) == expected
    if not mask.any():
        return pd.Series(dtype=float)
    values = macro_df.loc[mask, "value"].copy().sort_index()
    if values.index.has_duplicates:
        values = values.groupby(level=0).last()
    return values.astype("float64")


def _extract_alias_series(macro_df: pd.DataFrame, alias: str) -> pd.Series:
    if "series_alias" not in macro_df.columns:
        return pd.Series(dtype=float)
    mask = macro_df["series_alias"].astype(str) == str(alias)
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


def _extract_fred_series(macro_df: pd.DataFrame, cfg: Mapping | None) -> pd.Series:
    if not _is_enabled(cfg):
        return pd.Series(dtype=float)
    expected = str(cfg.get("series_id", "")).strip()
    if not expected:
        return pd.Series(dtype=float)
    mask = macro_df["series_id"].astype(str) == expected
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

    alias_series = _extract_alias_series(macro_df, alias)
    if not alias_series.empty:
        return alias_series

    kosis_cfg = (macro_series_cfg.get("kosis", {}) if macro_series_cfg else {}).get(alias)
    kosis_series = _extract_kosis_series(macro_df, kosis_cfg)
    if not kosis_series.empty:
        return kosis_series

    ecos_cfg = (macro_series_cfg.get("ecos", {}) if macro_series_cfg else {}).get(alias)
    ecos_series = _extract_ecos_series(macro_df, ecos_cfg)
    if not ecos_series.empty:
        return ecos_series

    fred_cfg = (macro_series_cfg.get("fred", {}) if macro_series_cfg else {}).get(alias)
    fred_series = _extract_fred_series(macro_df, fred_cfg)
    if not fred_series.empty:
        return fred_series

    return pd.Series(dtype=float)


def to_plotly_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a Plotly-serializable time index."""
    if df.empty:
        return df
    out = df.copy()
    if isinstance(out.index, pd.PeriodIndex):
        out.index = out.index.to_timestamp(how="end")
    return out


def filter_macro_provisional_rows(
    macro_df: pd.DataFrame,
    *,
    include_provisional: bool,
) -> pd.DataFrame:
    """Return the macro frame with optional provisional-row exclusion."""
    if include_provisional or macro_df.empty or "is_provisional" not in macro_df.columns:
        return macro_df
    filtered = macro_df[~macro_df["is_provisional"].fillna(False)].copy()
    return filtered


def build_regime_inflation_series(
    *,
    macro_df: pd.DataFrame,
    macro_series_cfg: Mapping,
    market_id: str,
) -> pd.Series:
    """Return the inflation series used for macro regime classification.

    KR keeps a homogeneous YoY-scale history by preferring direct `cpi_yoy`
    and backfilling older months from the legacy CPI index. This mirrors the
    dashboard path without importing dashboard code into scripts.
    """
    normalized_market = str(market_id or "KR").strip().upper() or "KR"
    if normalized_market != "KR":
        inflation_series = extract_macro_series(
            macro_df=macro_df,
            macro_series_cfg=macro_series_cfg,
            alias="cpi_mom",
        )
        if inflation_series.empty:
            inflation_series = extract_macro_series(
                macro_df=macro_df,
                macro_series_cfg=macro_series_cfg,
                alias="cpi_yoy",
            )
        return inflation_series

    inflation_series = extract_macro_series(
        macro_df=macro_df,
        macro_series_cfg=macro_series_cfg,
        alias="cpi_yoy",
    )
    cpi_legacy_idx = extract_macro_series(
        macro_df=macro_df,
        macro_series_cfg=macro_series_cfg,
        alias="cpi_index_legacy",
    )

    if not cpi_legacy_idx.empty:
        legacy_yoy = (cpi_legacy_idx / cpi_legacy_idx.shift(12) - 1) * 100
        legacy_yoy = legacy_yoy.dropna()
        if not legacy_yoy.empty:
            if inflation_series.empty:
                return legacy_yoy
            cutoff = inflation_series.index.min()
            legacy_part = legacy_yoy[legacy_yoy.index < cutoff]
            if not legacy_part.empty:
                return pd.concat([legacy_part, inflation_series]).sort_index()

    if not inflation_series.empty:
        return inflation_series

    return extract_macro_series(
        macro_df=macro_df,
        macro_series_cfg=macro_series_cfg,
        alias="cpi_mom",
    )


def build_regime_history_from_macro(
    *,
    macro_df: pd.DataFrame,
    macro_series_cfg: Mapping,
    settings: Mapping,
    market_id: str,
    include_provisional: bool,
    window_months: int | None = None,
) -> pd.DataFrame:
    """Build a macro regime history frame from a macro fact table.

    This is the smallest shared helper needed to keep dashboard and validation
    scripts on the same regime-construction contract.
    """
    from src.macro.regime import compute_regime_history

    filtered_macro = filter_macro_provisional_rows(
        macro_df,
        include_provisional=include_provisional,
    )
    if filtered_macro.empty:
        return pd.DataFrame()

    growth_series = extract_macro_series(
        macro_df=filtered_macro,
        macro_series_cfg=macro_series_cfg,
        alias="leading_index",
    )
    inflation_series = build_regime_inflation_series(
        macro_df=filtered_macro,
        macro_series_cfg=macro_series_cfg,
        market_id=market_id,
    )
    if growth_series.empty or inflation_series.empty:
        return pd.DataFrame()

    aligned = pd.concat(
        {"growth": growth_series, "inflation": inflation_series},
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return pd.DataFrame()

    if window_months is not None and window_months > 0 and len(aligned) > window_months:
        aligned = aligned.iloc[-window_months:]

    long_alias = str(settings.get("yield_curve_long", "bond_3y"))
    short_alias = str(settings.get("yield_curve_short", "base_rate"))
    bond_series = extract_macro_series(filtered_macro, macro_series_cfg, long_alias)
    base_series = extract_macro_series(filtered_macro, macro_series_cfg, short_alias)
    yield_curve_spread = None
    if not bond_series.empty and not base_series.empty:
        spread = (bond_series - base_series).dropna()
        if not spread.empty:
            yield_curve_spread = spread

    return compute_regime_history(
        aligned["growth"],
        aligned["inflation"],
        epsilon=float(settings.get("epsilon", 0.0)),
        use_adaptive_epsilon=bool(settings.get("use_adaptive_epsilon", True)),
        epsilon_factor=float(settings.get("epsilon_factor", 0.5)),
        confirmation_periods=int(settings.get("confirmation_periods", 2)),
        carry_single_flat_regime=bool(settings.get("carry_single_flat_regime", False)),
        yield_curve_spread=yield_curve_spread,
        yield_curve_threshold=float(settings.get("yield_curve_spread_threshold", 0.0)),
    )
