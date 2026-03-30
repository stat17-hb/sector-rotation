"""
FRED macro-data loader for US market support.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Literal

import pandas as pd

from src.data_sources.common import load_secret_or_env, request_json_with_retry
from src.data_sources.macro_sync import sync_provider_macro

logger = logging.getLogger(__name__)

DataStatus = Literal["LIVE", "CACHED", "SAMPLE"]
LoaderResult = tuple[DataStatus, pd.DataFrame]

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
VALID_TRANSFORMS = {"none", "pct_change_1m", "pct_change_12m"}


def _get_api_key() -> str:
    return load_secret_or_env("FRED_API_KEY")


def _normalize_transform(transform: str | None) -> str:
    normalized = str(transform or "none").strip().lower()
    if normalized not in VALID_TRANSFORMS:
        raise ValueError(f"Unsupported FRED transform: {transform!r}")
    return normalized


def _transform_series(values: pd.Series, transform: str) -> pd.Series:
    transform_name = _normalize_transform(transform)
    if transform_name == "none":
        return values
    if transform_name == "pct_change_1m":
        return values.pct_change(periods=1).mul(100.0)
    return values.pct_change(periods=12).mul(100.0)


def fetch_fred_series(
    series_id: str,
    start_ym: str,
    end_ym: str,
    transform: str = "none",
) -> pd.DataFrame:
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("FRED_API_KEY not configured")

    start_period = pd.Period(str(start_ym)[:6], freq="M")
    end_period = pd.Period(str(end_ym)[:6], freq="M")
    params = {
        "series_id": str(series_id).strip(),
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_period.to_timestamp(how="start").strftime("%Y-%m-%d"),
        "observation_end": end_period.to_timestamp(how="end").strftime("%Y-%m-%d"),
    }
    payload = request_json_with_retry(FRED_BASE_URL, params=params, client_name="FRED")
    if not isinstance(payload, dict):
        raise ValueError(f"FRED returned unexpected payload type: {type(payload).__name__}")
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError(f"No FRED data for {series_id}")

    rows: list[dict[str, object]] = []
    for item in observations:
        if not isinstance(item, dict):
            continue
        date_token = str(item.get("date", "")).strip()
        raw_value = str(item.get("value", "")).strip()
        if not date_token or raw_value in {"", "."}:
            continue
        try:
            period = pd.Period(pd.Timestamp(date_token), freq="M")
            value = float(raw_value)
        except (ValueError, TypeError):
            continue
        rows.append({"period": period, "value": value})

    if not rows:
        raise ValueError(f"Could not parse FRED observations for {series_id}")

    frame = pd.DataFrame(rows).drop_duplicates(subset=["period"], keep="last").sort_values("period")
    values = pd.Series(frame["value"].to_numpy(dtype="float64"), index=pd.PeriodIndex(frame["period"], freq="M"))
    transformed = _transform_series(values, transform).dropna()
    if transformed.empty:
        raise ValueError(f"FRED transform removed all rows for {series_id}")

    now = datetime.now(timezone.utc)
    result = pd.DataFrame(
        {
            "series_id": str(series_id).strip(),
            "value": transformed.astype("float64"),
            "source": "FRED",
            "fetched_at": now,
            "is_provisional": False,
        },
        index=transformed.index,
    )
    result.index = pd.PeriodIndex(result.index, freq="M")
    return result.astype({"value": "float64", "is_provisional": "bool"})


def _make_sample_macro() -> pd.DataFrame:
    periods = pd.period_range(end=pd.Period.now("M"), periods=36, freq="M")
    values = pd.Series(range(len(periods)), index=periods, dtype="float64")
    values = values.pct_change().fillna(0.0).mul(10.0)
    now = datetime.now(timezone.utc)
    frame = pd.DataFrame(
        {
            "series_id": "SAMPLE/FRED",
            "value": values.astype("float64"),
            "source": "FRED",
            "fetched_at": now,
            "is_provisional": False,
        },
        index=periods,
    )
    return frame.astype({"value": "float64", "is_provisional": "bool"})


def load_fred_macro(
    start_ym: str,
    end_ym: str,
    series_config: dict | None = None,
    *,
    market: str = "US",
) -> LoaderResult:
    if series_config is None:
        series_config = {
            "leading_index": {"series_id": "USALOLITONOSTSAM", "transform": "none", "enabled": True},
            "cpi_yoy": {"series_id": "CPIAUCSL", "transform": "pct_change_12m", "enabled": True},
            "cpi_mom": {"series_id": "CPIAUCSL", "transform": "pct_change_1m", "enabled": True},
            "fed_funds": {"series_id": "FEDFUNDS", "transform": "none", "enabled": True},
            "treasury_10y": {"series_id": "GS10", "transform": "none", "enabled": True},
            "treasury_2y": {"series_id": "GS2", "transform": "none", "enabled": True},
            "dxy": {"series_id": "DTWEXBGS", "transform": "none", "enabled": True},
        }

    def _fetch(alias: str, cfg: dict, provider_start: str, provider_end: str) -> pd.DataFrame:
        _ = alias
        return fetch_fred_series(
            series_id=cfg["series_id"],
            start_ym=provider_start,
            end_ym=provider_end,
            transform=str(cfg.get("transform", "none")),
        )

    status, result, summary = sync_provider_macro(
        provider="FRED",
        start_ym=start_ym,
        end_ym=end_ym,
        series_config=series_config,
        fetch_fn=_fetch,
        reason="load_fred_macro",
        force=False,
        market=market,
    )
    failed_aliases = dict(summary.get("failed_aliases") or {})
    if failed_aliases and status != "SAMPLE":
        logger.warning(
            "FRED partial success: %d aliases loaded, %d failed (%s)",
            len(summary.get("delta_aliases", [])),
            len(failed_aliases),
            ", ".join(sorted(failed_aliases)),
        )
    if status == "SAMPLE":
        logger.warning("Using SAMPLE data for FRED macro")
        return ("SAMPLE", _make_sample_macro())
    return status, result
