"""KR thematic lens backed by representative ETF OHLCV proxies.

Normal dashboard rendering is cache-only. Live pykrx access is reserved for
the explicit refresh function so theme ETF refresh cannot leak into ordinary
page render.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from src.data_sources.warehouse import (
    WAREHOUSE_PATH,
    read_stock_ohlcv,
    upsert_stock_ohlcv,
)


THEME_LENS_CONFIG_PATH = Path("config/theme_lens.yml")
THEME_RETURN_WINDOWS: tuple[tuple[str, int], ...] = (
    ("return_1d", 1),
    ("return_1m", 21),
    ("return_3m", 63),
)
MIN_LIVE_LOOKBACK_DAYS = 120
THEME_SIGNAL_LOOKBACK_DAYS = 420


@dataclass(frozen=True)
class ThemeLensDefinition:
    theme_id: str
    name: str
    price_source: str
    proxy_note: str
    classification_basis: tuple[dict[str, str], ...]
    representative_etfs: tuple[dict[str, str], ...]


def _coerce_text(value: object) -> str:
    return str(value or "").strip()


def _normalize_mapping_items(items: object, *, required_keys: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    if not isinstance(items, list):
        return ()
    normalized: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        row = {key: _coerce_text(item.get(key)) for key in required_keys}
        if all(row.values()):
            normalized.append(row)
    return tuple(normalized)


def load_theme_lens_config(path: Path = THEME_LENS_CONFIG_PATH) -> list[ThemeLensDefinition]:
    """Load and validate KR theme lens definitions."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    themes = payload.get("themes", []) if isinstance(payload, Mapping) else []
    if not isinstance(themes, list):
        raise ValueError("theme_lens.yml must contain a themes list")

    definitions: list[ThemeLensDefinition] = []
    seen_ids: set[str] = set()
    for item in themes:
        if not isinstance(item, Mapping):
            continue
        theme_id = _coerce_text(item.get("theme_id"))
        name = _coerce_text(item.get("name"))
        if not theme_id or not name:
            raise ValueError("theme entries require theme_id and name")
        if theme_id in seen_ids:
            raise ValueError(f"duplicate theme_id: {theme_id}")
        seen_ids.add(theme_id)

        etfs = _normalize_mapping_items(item.get("representative_etfs"), required_keys=("code", "name"))
        if not etfs:
            raise ValueError(f"theme {theme_id} requires at least one representative ETF")
        definitions.append(
            ThemeLensDefinition(
                theme_id=theme_id,
                name=name,
                price_source=_coerce_text(item.get("price_source")) or "ETF_OHLCV",
                proxy_note=_coerce_text(item.get("proxy_note")),
                classification_basis=_normalize_mapping_items(
                    item.get("classification_basis"),
                    required_keys=("provider", "label"),
                ),
                representative_etfs=etfs,
            )
        )
    return definitions


def _format_date(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _safe_return(close: pd.Series, offset: int) -> float:
    values = pd.to_numeric(close, errors="coerce").dropna()
    if len(values) <= offset:
        return float("nan")
    base = float(values.iloc[-(offset + 1)])
    latest = float(values.iloc[-1])
    if base == 0:
        return float("nan")
    return latest / base - 1.0


def build_theme_proxy_returns(frame: pd.DataFrame) -> dict[str, float]:
    """Compute proxy return metrics from one ETF OHLCV frame."""
    if frame.empty or "close" not in frame.columns:
        return {label: float("nan") for label, _offset in THEME_RETURN_WINDOWS}
    close = pd.to_numeric(frame.sort_index()["close"], errors="coerce")
    return {label: _safe_return(close, offset) for label, offset in THEME_RETURN_WINDOWS}


def _theme_tickers(definitions: list[ThemeLensDefinition]) -> list[str]:
    tickers: list[str] = []
    for definition in definitions:
        for etf in definition.representative_etfs:
            code = _coerce_text(etf.get("code"))
            if code and code not in tickers:
                tickers.append(code)
    return tickers


def get_theme_lens_artifact_key(path: Path = THEME_LENS_CONFIG_PATH) -> tuple[int, int, str, int, int]:
    """Return a cache key derived from config and warehouse artifacts."""
    config_hash = ""
    config_mtime = 0
    config_size = 0
    if path.exists():
        raw = path.read_bytes()
        config_hash = hashlib.sha256(raw).hexdigest()[:16]
        stat = path.stat()
        config_mtime = int(stat.st_mtime_ns)
        config_size = int(stat.st_size)

    warehouse_mtime = 0
    warehouse_size = 0
    if WAREHOUSE_PATH.exists():
        stat = WAREHOUSE_PATH.stat()
        warehouse_mtime = int(stat.st_mtime_ns)
        warehouse_size = int(stat.st_size)
    return (config_mtime, config_size, config_hash, warehouse_mtime, warehouse_size)


def _status_from_frame(frame: pd.DataFrame, *, asof_date: str) -> tuple[str, str]:
    if frame.empty:
        return "UNAVAILABLE", "대표 ETF 가격 캐시가 없습니다."
    if "close" not in frame.columns or pd.to_numeric(frame["close"], errors="coerce").dropna().empty:
        return "UNAVAILABLE", "대표 ETF 종가 데이터가 없습니다."
    latest = pd.Timestamp(frame.index.max()).normalize()
    requested = pd.Timestamp(asof_date).normalize()
    if latest < requested - pd.Timedelta(days=10):
        return "STALE", "대표 ETF 가격 캐시가 최신 기준일보다 오래되었습니다."
    return "CACHED", ""


def _row_for_theme(
    definition: ThemeLensDefinition,
    cached: pd.DataFrame,
    *,
    asof_date: str,
    status_override: str | None = None,
) -> dict[str, Any]:
    selected = definition.representative_etfs[0]
    selected_frame = pd.DataFrame()
    if not cached.empty and "ticker" in cached.columns:
        ticker_text = cached["ticker"].astype(str)
        for candidate in definition.representative_etfs:
            candidate_code = _coerce_text(candidate.get("code"))
            candidate_frame = cached[ticker_text == candidate_code].copy()
            candidate_status, _candidate_warning = _status_from_frame(candidate_frame, asof_date=asof_date)
            if candidate_status in {"CACHED", "STALE"}:
                selected = candidate
                selected_frame = candidate_frame
                break
            if selected_frame.empty and not candidate_frame.empty:
                selected = candidate
                selected_frame = candidate_frame
    selected_code = _coerce_text(selected.get("code"))
    selected_frame = selected_frame.sort_index()
    status, warning = _status_from_frame(selected_frame, asof_date=asof_date)
    if status_override and not selected_frame.empty:
        status = str(status_override).strip().upper()
        warning = ""
    returns = build_theme_proxy_returns(selected_frame)
    latest_date = _format_date(selected_frame.index.max()) if not selected_frame.empty else ""
    return {
        "theme_id": definition.theme_id,
        "theme_name": definition.name,
        "classification_basis": [dict(item) for item in definition.classification_basis],
        "representative_etfs": [dict(item) for item in definition.representative_etfs],
        "primary_proxy_code": selected_code,
        "primary_proxy_name": _coerce_text(selected.get("name")),
        "latest_date": latest_date,
        "price_source": definition.price_source,
        "proxy_note": definition.proxy_note,
        "status": status,
        "warning": warning,
        "reference_only": True,
        **returns,
    }


def load_theme_lens_cache_only(
    *,
    asof_date: str,
    lookback_days: int = MIN_LIVE_LOOKBACK_DAYS,
    config_path: Path = THEME_LENS_CONFIG_PATH,
) -> tuple[str, list[dict[str, Any]]]:
    """Load theme lens rows from warehouse cache only."""
    definitions = load_theme_lens_config(config_path)
    if not definitions:
        return "UNAVAILABLE", []
    end = pd.Timestamp(asof_date).strftime("%Y%m%d")
    start = (pd.Timestamp(asof_date) - pd.Timedelta(days=max(lookback_days, MIN_LIVE_LOOKBACK_DAYS))).strftime("%Y%m%d")
    cached = read_stock_ohlcv(_theme_tickers(definitions), start, end, market="KR")
    rows = [_row_for_theme(definition, cached, asof_date=asof_date) for definition in definitions]
    statuses = {str(row.get("status", "")) for row in rows}
    if statuses == {"CACHED"}:
        return "CACHED", rows
    if "CACHED" in statuses or "STALE" in statuses:
        return "PARTIAL", rows
    return "UNAVAILABLE", rows


def load_theme_proxy_signal_inputs(
    *,
    asof_date: str,
    lookback_days: int = THEME_SIGNAL_LOOKBACK_DAYS,
    config_path: Path = THEME_LENS_CONFIG_PATH,
) -> tuple[str, pd.DataFrame, list[dict[str, Any]]]:
    """Return cached theme ETF proxy rows in signal-compatible shape.

    This is intentionally cache-only. Live pykrx access belongs only in
    `refresh_theme_lens_etf_ohlcv()`.
    """
    definitions = load_theme_lens_config(config_path)
    if not definitions:
        return "UNAVAILABLE", pd.DataFrame(), []

    end = pd.Timestamp(asof_date).strftime("%Y%m%d")
    start = (pd.Timestamp(asof_date) - pd.Timedelta(days=max(lookback_days, THEME_SIGNAL_LOOKBACK_DAYS))).strftime(
        "%Y%m%d"
    )
    cached = read_stock_ohlcv(_theme_tickers(definitions), start, end, market="KR")
    if cached.empty or "ticker" not in cached.columns or "close" not in cached.columns:
        return "UNAVAILABLE", pd.DataFrame(), []

    ticker_text = cached["ticker"].astype(str)
    price_frames: list[pd.DataFrame] = []
    universe_rows: list[dict[str, Any]] = []
    used_codes: set[str] = set()

    for definition in definitions:
        row = _row_for_theme(definition, cached, asof_date=asof_date)
        if str(row.get("status", "")).strip().upper() != "CACHED":
            continue

        proxy_code = _coerce_text(row.get("primary_proxy_code"))
        if not proxy_code or proxy_code in used_codes:
            continue

        proxy_frame = cached[ticker_text == proxy_code].copy().sort_index()
        if proxy_frame.empty:
            continue

        signal_frame = pd.DataFrame(
            {
                "index_code": proxy_code,
                "index_name": definition.name,
                "close": pd.to_numeric(proxy_frame["close"], errors="coerce"),
            },
            index=pd.DatetimeIndex(proxy_frame.index).normalize(),
        ).dropna(subset=["close"])
        if signal_frame.empty:
            continue

        used_codes.add(proxy_code)
        price_frames.append(signal_frame)
        universe_rows.append(
            {
                "index_code": proxy_code,
                "index_name": definition.name,
                "family": "theme_lens_etf_proxy",
                "is_benchmark": False,
                "is_active": True,
                "export_sector": False,
                "taxonomy_kind": "THEME",
                "taxonomy_label": definition.name,
                "theme_id": definition.theme_id,
                "primary_proxy_code": proxy_code,
                "primary_proxy_name": _coerce_text(row.get("primary_proxy_name")),
                "reference_only": True,
            }
        )

    if not price_frames:
        return "UNAVAILABLE", pd.DataFrame(), []

    status = "CACHED" if len(universe_rows) == len(definitions) else "PARTIAL"
    return status, pd.concat(price_frames).sort_index(), universe_rows


def _normalize_live_ohlcv(raw: pd.DataFrame, *, ticker: str, ticker_name: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    column_map = {
        "시가": "open",
        "고가": "high",
        "저가": "low",
        "종가": "close",
        "거래량": "volume",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    normalized = pd.DataFrame(index=pd.DatetimeIndex(raw.index).normalize())
    for source, target in column_map.items():
        if source in raw.columns and target not in normalized.columns:
            normalized[target] = pd.to_numeric(raw[source], errors="coerce")
    if "close" not in normalized.columns and len(raw.columns) >= 4:
        normalized["close"] = pd.to_numeric(raw.iloc[:, 3], errors="coerce")
    if "volume" not in normalized.columns and len(raw.columns) >= 5:
        normalized["volume"] = pd.to_numeric(raw.iloc[:, 4], errors="coerce")
    if "close" not in normalized.columns:
        return pd.DataFrame()
    normalized["ticker"] = str(ticker)
    normalized["ticker_name"] = str(ticker_name or ticker)
    return normalized.sort_index().dropna(subset=["close"])


def refresh_theme_lens_etf_ohlcv(
    *,
    asof_date: str,
    lookback_days: int = THEME_SIGNAL_LOOKBACK_DAYS,
    config_path: Path = THEME_LENS_CONFIG_PATH,
) -> dict[str, Any]:
    """Explicit live refresh for theme ETF OHLCV rows."""
    from src.data_sources.pykrx_compat import ensure_pykrx_transport_compat

    ensure_pykrx_transport_compat()
    from pykrx import stock  # type: ignore[import]

    definitions = load_theme_lens_config(config_path)
    start = (pd.Timestamp(asof_date) - pd.Timedelta(days=max(lookback_days, MIN_LIVE_LOOKBACK_DAYS))).strftime("%Y%m%d")
    end = pd.Timestamp(asof_date).strftime("%Y%m%d")
    failed_codes: dict[str, str] = {}
    fetched_codes: list[str] = []
    refreshed_codes: list[str] = []
    live_frames: list[pd.DataFrame] = []
    row_count = 0
    etf_names = {
        _coerce_text(etf.get("code")): _coerce_text(etf.get("name"))
        for definition in definitions
        for etf in definition.representative_etfs
    }

    for ticker, ticker_name in etf_names.items():
        if not ticker:
            continue
        try:
            raw = stock.get_market_ohlcv_by_date(start, end, ticker)
            normalized = _normalize_live_ohlcv(raw, ticker=ticker, ticker_name=ticker_name)
        except Exception as exc:
            failed_codes[ticker] = str(exc)
            continue
        if normalized.empty:
            failed_codes[ticker] = "empty OHLCV response"
            continue
        fetched_codes.append(ticker)
        live_frames.append(normalized)
        row_count += int(len(normalized))
        try:
            upsert_stock_ohlcv(normalized, provider="PYKRX", market="KR")
        except Exception as exc:
            failed_codes[ticker] = f"warehouse upsert failed: {exc}"
            continue
        refreshed_codes.append(ticker)

    live_frame = pd.concat(live_frames).sort_index() if live_frames else pd.DataFrame()
    live_rows = [
        _row_for_theme(definition, live_frame, asof_date=asof_date, status_override="LIVE")
        for definition in definitions
    ]
    live_statuses = {str(row.get("status", "")) for row in live_rows}
    if live_statuses == {"LIVE"}:
        live_status = "LIVE"
    elif "LIVE" in live_statuses:
        live_status = "PARTIAL"
    else:
        live_status = "FAILED"
    status = "LIVE" if fetched_codes and not failed_codes else "PARTIAL" if fetched_codes else "FAILED"
    return {
        "status": status,
        "live_status": live_status,
        "requested_start": start,
        "requested_end": end,
        "fetched_codes": fetched_codes,
        "refreshed_codes": refreshed_codes,
        "failed_codes": failed_codes,
        "row_count": row_count,
        "live_rows": live_rows,
    }
