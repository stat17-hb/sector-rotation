"""
DuckDB-backed warehouse utilities for market and macro data.

The warehouse is the authoritative local store. Parquet exports are optional
compatibility artifacts and are never required for runtime reads.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal

import duckdb
import pandas as pd


WarehouseDataset = Literal["market_prices", "macro_data"]

WAREHOUSE_PATH = Path("data/warehouse.duckdb")
DEFAULT_PRICE_EXPORT_PATH = Path("data/curated/sector_prices.parquet")
DEFAULT_MACRO_EXPORT_PATH = Path("data/curated/macro_monthly.parquet")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _connect(*, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    if not read_only:
        WAREHOUSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(WAREHOUSE_PATH), read_only=read_only)


def ensure_warehouse_schema(connection: duckdb.DuckDBPyConnection | None = None) -> None:
    owns_connection = connection is None
    con = connection or _connect()
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS dim_index (
                index_code VARCHAR PRIMARY KEY,
                index_name VARCHAR,
                family VARCHAR,
                is_benchmark BOOLEAN NOT NULL DEFAULT FALSE,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                export_sector BOOLEAN,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS fact_krx_index_daily (
                trade_date DATE NOT NULL,
                index_code VARCHAR NOT NULL,
                close DOUBLE NOT NULL,
                provider VARCHAR NOT NULL,
                loaded_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (trade_date, index_code)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS dim_macro_series (
                series_alias VARCHAR PRIMARY KEY,
                provider VARCHAR NOT NULL,
                provider_series_id VARCHAR NOT NULL,
                enabled BOOLEAN NOT NULL,
                label VARCHAR,
                unit VARCHAR,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS fact_macro_monthly (
                period_month DATE NOT NULL,
                series_alias VARCHAR NOT NULL,
                provider VARCHAR NOT NULL,
                provider_series_id VARCHAR NOT NULL,
                value DOUBLE,
                is_provisional BOOLEAN NOT NULL,
                fetched_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (period_month, series_alias)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_runs (
                run_id VARCHAR PRIMARY KEY,
                dataset VARCHAR NOT NULL,
                reason VARCHAR NOT NULL,
                provider VARCHAR NOT NULL,
                requested_start VARCHAR NOT NULL,
                requested_end VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                coverage_complete BOOLEAN NOT NULL,
                failed_days_json VARCHAR NOT NULL,
                failed_codes_json VARCHAR NOT NULL,
                delta_keys_json VARCHAR NOT NULL,
                row_count BIGINT NOT NULL,
                aborted BOOLEAN NOT NULL,
                abort_reason VARCHAR NOT NULL,
                predicted_requests INTEGER NOT NULL,
                processed_requests INTEGER NOT NULL,
                summary_json VARCHAR NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_watermarks (
                dataset VARCHAR PRIMARY KEY,
                watermark_key VARCHAR NOT NULL,
                coverage_complete BOOLEAN NOT NULL,
                status VARCHAR NOT NULL,
                provider VARCHAR NOT NULL,
                details_json VARCHAR NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """
        )
    finally:
        if owns_connection:
            con.close()


def warehouse_exists() -> bool:
    return WAREHOUSE_PATH.exists()


def _serialize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalize_market_date(value: str) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) != 8:
        raise ValueError(f"Invalid market date: {value!r}")
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"


def _deserialize_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _register_frame(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
    frame: pd.DataFrame,
) -> None:
    con.register(view_name, frame)


def upsert_index_dimension(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    now = _utc_now()
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return
    frame["updated_at"] = now
    frame["index_code"] = frame["index_code"].astype(str)
    if "index_name" not in frame.columns:
        frame["index_name"] = pd.Series([""] * len(frame), dtype="object")
    if "family" not in frame.columns:
        frame["family"] = pd.Series([""] * len(frame), dtype="object")
    if "is_benchmark" not in frame.columns:
        frame["is_benchmark"] = pd.Series([False] * len(frame), dtype="bool")
    if "is_active" not in frame.columns:
        frame["is_active"] = pd.Series([True] * len(frame), dtype="bool")
    frame["index_name"] = frame["index_name"].astype(str)
    frame["family"] = frame["family"].astype(str)
    frame["is_benchmark"] = frame["is_benchmark"].astype(bool)
    frame["is_active"] = frame["is_active"].astype(bool)
    if "export_sector" not in frame.columns:
        frame["export_sector"] = pd.Series([None] * len(frame), dtype="object")

    con = _connect()
    try:
        ensure_warehouse_schema(con)
        _register_frame(con, "dim_index_upsert", frame)
        con.execute(
            """
            INSERT INTO dim_index AS dim
            SELECT
                index_code,
                index_name,
                family,
                is_benchmark,
                is_active,
                export_sector,
                updated_at
            FROM dim_index_upsert
            ON CONFLICT (index_code) DO UPDATE SET
                index_name = excluded.index_name,
                family = excluded.family,
                is_benchmark = excluded.is_benchmark,
                is_active = excluded.is_active,
                export_sector = excluded.export_sector,
                updated_at = excluded.updated_at
            """
        )
    finally:
        con.close()


def upsert_market_prices(frame: pd.DataFrame, *, provider: str) -> None:
    if frame.empty:
        return

    normalized = frame.copy()
    normalized.index = pd.DatetimeIndex(normalized.index)
    normalized = normalized.sort_index()
    normalized["index_code"] = normalized["index_code"].astype(str)
    normalized["close"] = normalized["close"].astype(float)
    normalized["trade_date"] = normalized.index.normalize()
    normalized["provider"] = str(provider or "").strip().upper() or "UNKNOWN"
    normalized["loaded_at"] = _utc_now()

    payload = normalized[["trade_date", "index_code", "close", "provider", "loaded_at"]]
    con = _connect()
    try:
        ensure_warehouse_schema(con)
        _register_frame(con, "market_prices_upsert", payload)
        con.execute(
            """
            INSERT INTO fact_krx_index_daily AS fact
            SELECT
                CAST(trade_date AS DATE) AS trade_date,
                index_code,
                close,
                provider,
                loaded_at
            FROM market_prices_upsert
            ON CONFLICT (trade_date, index_code) DO UPDATE SET
                close = excluded.close,
                provider = excluded.provider,
                loaded_at = excluded.loaded_at
            """
        )
    finally:
        con.close()


def read_market_prices(
    index_codes: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    if not warehouse_exists() or not index_codes:
        return pd.DataFrame()

    codes_frame = pd.DataFrame({"index_code": [str(code) for code in index_codes]})
    start_date = _normalize_market_date(start)
    end_date = _normalize_market_date(end)
    con = _connect(read_only=True)
    try:
        _register_frame(con, "requested_codes", codes_frame)
        result = con.execute(
            """
            SELECT
                fact.trade_date,
                fact.index_code,
                COALESCE(dim.index_name, fact.index_code) AS index_name,
                fact.close
            FROM fact_krx_index_daily AS fact
            INNER JOIN requested_codes AS req
                ON req.index_code = fact.index_code
            LEFT JOIN dim_index AS dim
                ON dim.index_code = fact.index_code
            WHERE fact.trade_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            ORDER BY fact.trade_date, fact.index_code
            """,
            [start_date, end_date],
        ).fetchdf()
    finally:
        con.close()

    if result.empty:
        return pd.DataFrame()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result = result.set_index("trade_date")
    result.index = pd.DatetimeIndex(result.index)
    return result[["index_code", "index_name", "close"]].astype(
        {"index_code": "object", "index_name": "object", "close": "float64"}
    )


def get_market_latest_dates(index_codes: list[str]) -> dict[str, str]:
    if not warehouse_exists() or not index_codes:
        return {}

    codes_frame = pd.DataFrame({"index_code": [str(code) for code in index_codes]})
    con = _connect(read_only=True)
    try:
        _register_frame(con, "latest_price_codes", codes_frame)
        result = con.execute(
            """
            SELECT
                fact.index_code,
                STRFTIME(MAX(fact.trade_date), '%Y%m%d') AS last_trade_date
            FROM fact_krx_index_daily AS fact
            INNER JOIN latest_price_codes AS req
                ON req.index_code = fact.index_code
            GROUP BY fact.index_code
            """
        ).fetchdf()
    finally:
        con.close()

    if result.empty:
        return {}
    return {
        str(row["index_code"]): str(row["last_trade_date"])
        for _, row in result.iterrows()
        if str(row["last_trade_date"]).strip()
    }


def is_market_coverage_complete(
    index_codes: list[str],
    start: str,
    end: str,
    *,
    benchmark_code: str,
) -> bool:
    frame = read_market_prices(index_codes, start, end)
    if frame.empty:
        return False

    benchmark = frame[frame["index_code"].astype(str) == str(benchmark_code)]
    if benchmark.empty:
        return False
    expected_dates = pd.Index(benchmark.index.unique()).sort_values()
    if expected_dates.empty:
        return False

    for code in [str(code) for code in index_codes]:
        code_dates = pd.Index(frame[frame["index_code"].astype(str) == code].index.unique())
        if len(code_dates) != len(expected_dates):
            return False
        if not code_dates.sort_values().equals(expected_dates):
            return False
    return True


def export_market_parquet(path: Path | None = None) -> Path | None:
    export_path = path or DEFAULT_PRICE_EXPORT_PATH
    if not warehouse_exists():
        return None

    con = _connect(read_only=True)
    try:
        result = con.execute(
            """
            SELECT
                fact.trade_date,
                fact.index_code,
                COALESCE(dim.index_name, fact.index_code) AS index_name,
                fact.close
            FROM fact_krx_index_daily AS fact
            LEFT JOIN dim_index AS dim
                ON dim.index_code = fact.index_code
            ORDER BY fact.trade_date, fact.index_code
            """
        ).fetchdf()
    finally:
        con.close()

    if result.empty:
        return None
    export_path.parent.mkdir(parents=True, exist_ok=True)
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    out = result.set_index("trade_date")
    out.index = pd.DatetimeIndex(out.index)
    out[["index_code", "index_name", "close"]].to_parquet(export_path)
    return export_path


def upsert_macro_dimension(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    now = _utc_now()
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return
    frame["updated_at"] = now
    frame["series_alias"] = frame["series_alias"].astype(str)
    frame["provider"] = frame["provider"].astype(str)
    frame["provider_series_id"] = frame["provider_series_id"].astype(str)
    frame["enabled"] = frame["enabled"].astype(bool)
    if "label" not in frame.columns:
        frame["label"] = pd.Series([""] * len(frame), dtype="object")
    if "unit" not in frame.columns:
        frame["unit"] = pd.Series([""] * len(frame), dtype="object")

    con = _connect()
    try:
        ensure_warehouse_schema(con)
        _register_frame(con, "dim_macro_upsert", frame)
        con.execute(
            """
            INSERT INTO dim_macro_series AS dim
            SELECT
                series_alias,
                provider,
                provider_series_id,
                enabled,
                label,
                unit,
                updated_at
            FROM dim_macro_upsert
            ON CONFLICT (series_alias) DO UPDATE SET
                provider = excluded.provider,
                provider_series_id = excluded.provider_series_id,
                enabled = excluded.enabled,
                label = excluded.label,
                unit = excluded.unit,
                updated_at = excluded.updated_at
            """
        )
    finally:
        con.close()


def upsert_macro_series_frame(
    *,
    series_alias: str,
    provider: str,
    provider_series_id: str,
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        return

    normalized = frame.copy()
    if isinstance(normalized.index, pd.PeriodIndex):
        normalized["period_month"] = normalized.index.to_timestamp(how="end").normalize()
    else:
        normalized["period_month"] = pd.to_datetime(normalized.index).to_period("M").to_timestamp(how="end").normalize()
    normalized["series_alias"] = str(series_alias)
    normalized["provider"] = str(provider).strip().upper()
    normalized["provider_series_id"] = str(provider_series_id)
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    normalized["is_provisional"] = normalized["is_provisional"].astype(bool)
    normalized["fetched_at"] = pd.to_datetime(normalized["fetched_at"], utc=True)

    payload = normalized[
        [
            "period_month",
            "series_alias",
            "provider",
            "provider_series_id",
            "value",
            "is_provisional",
            "fetched_at",
        ]
    ]

    con = _connect()
    try:
        ensure_warehouse_schema(con)
        _register_frame(con, "macro_series_upsert", payload)
        con.execute(
            """
            INSERT INTO fact_macro_monthly AS fact
            SELECT
                CAST(period_month AS DATE) AS period_month,
                series_alias,
                provider,
                provider_series_id,
                value,
                is_provisional,
                fetched_at
            FROM macro_series_upsert
            ON CONFLICT (period_month, series_alias) DO UPDATE SET
                provider = excluded.provider,
                provider_series_id = excluded.provider_series_id,
                value = excluded.value,
                is_provisional = excluded.is_provisional,
                fetched_at = excluded.fetched_at
            """
        )
    finally:
        con.close()


def read_macro_data(
    *,
    series_aliases: list[str],
    start_ym: str,
    end_ym: str,
) -> pd.DataFrame:
    if not warehouse_exists() or not series_aliases:
        return pd.DataFrame()

    alias_frame = pd.DataFrame({"series_alias": [str(alias) for alias in series_aliases]})
    start_date = pd.Period(str(start_ym)[:6], freq="M").to_timestamp(how="end").strftime("%Y-%m-%d")
    end_date = pd.Period(str(end_ym)[:6], freq="M").to_timestamp(how="end").strftime("%Y-%m-%d")

    con = _connect(read_only=True)
    try:
        _register_frame(con, "requested_aliases", alias_frame)
        result = con.execute(
            """
            SELECT
                fact.period_month,
                fact.series_alias,
                fact.provider_series_id,
                fact.value,
                fact.provider,
                fact.fetched_at,
                fact.is_provisional
            FROM fact_macro_monthly AS fact
            INNER JOIN requested_aliases AS req
                ON req.series_alias = fact.series_alias
            WHERE fact.period_month BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            ORDER BY fact.period_month, fact.series_alias
            """,
            [start_date, end_date],
        ).fetchdf()
    finally:
        con.close()

    if result.empty:
        return pd.DataFrame()

    result["period_month"] = pd.to_datetime(result["period_month"])
    result["fetched_at"] = pd.to_datetime(result["fetched_at"], utc=True)
    result = result.set_index(pd.PeriodIndex(result["period_month"], freq="M"))
    result.index.name = None
    return pd.DataFrame(
        {
            "series_alias": result["series_alias"].astype("object"),
            "series_id": result["provider_series_id"].astype("object"),
            "value": result["value"].astype("float64"),
            "source": result["provider"].astype("object"),
            "fetched_at": result["fetched_at"],
            "is_provisional": result["is_provisional"].astype("bool"),
        },
        index=result.index,
    )


def get_macro_latest_periods(series_aliases: list[str]) -> dict[str, str]:
    if not warehouse_exists() or not series_aliases:
        return {}

    alias_frame = pd.DataFrame({"series_alias": [str(alias) for alias in series_aliases]})
    con = _connect(read_only=True)
    try:
        _register_frame(con, "latest_macro_aliases", alias_frame)
        result = con.execute(
            """
            SELECT
                fact.series_alias,
                STRFTIME(MAX(fact.period_month), '%Y%m') AS last_period
            FROM fact_macro_monthly AS fact
            INNER JOIN latest_macro_aliases AS req
                ON req.series_alias = fact.series_alias
            GROUP BY fact.series_alias
            """
        ).fetchdf()
    finally:
        con.close()

    if result.empty:
        return {}
    return {
        str(row["series_alias"]): str(row["last_period"])
        for _, row in result.iterrows()
        if str(row["last_period"]).strip()
    }


def is_macro_coverage_complete(
    *,
    series_aliases: list[str],
    start_ym: str,
    end_ym: str,
) -> bool:
    frame = read_macro_data(series_aliases=series_aliases, start_ym=start_ym, end_ym=end_ym)
    if frame.empty:
        return False

    expected_periods = pd.period_range(str(start_ym)[:6], str(end_ym)[:6], freq="M")
    if expected_periods.empty:
        return False

    for alias in [str(alias) for alias in series_aliases]:
        alias_periods = pd.Index(frame[frame["series_alias"].astype(str) == alias].index.unique())
        if len(alias_periods) != len(expected_periods):
            return False
        if not alias_periods.sort_values().equals(pd.Index(expected_periods)):
            return False
    return True


def export_macro_parquet(path: Path | None = None) -> Path | None:
    export_path = path or DEFAULT_MACRO_EXPORT_PATH
    if not warehouse_exists():
        return None

    con = _connect(read_only=True)
    try:
        result = con.execute(
            """
            SELECT
                period_month,
                provider_series_id,
                value,
                provider,
                fetched_at,
                is_provisional
            FROM fact_macro_monthly
            ORDER BY period_month, provider_series_id
            """
        ).fetchdf()
    finally:
        con.close()

    if result.empty:
        return None
    export_path.parent.mkdir(parents=True, exist_ok=True)
    result["period_month"] = pd.to_datetime(result["period_month"])
    frame = pd.DataFrame(
        {
            "series_id": result["provider_series_id"].astype("object"),
            "value": result["value"].astype("float64"),
            "source": result["provider"].astype("object"),
            "fetched_at": pd.to_datetime(result["fetched_at"], utc=True),
            "is_provisional": result["is_provisional"].astype("bool"),
        },
        index=pd.PeriodIndex(result["period_month"], freq="M"),
    )
    frame.to_parquet(export_path)
    return export_path


def record_ingest_run(
    *,
    dataset: WarehouseDataset,
    reason: str,
    provider: str,
    requested_start: str,
    requested_end: str,
    status: str,
    coverage_complete: bool,
    failed_days: list[str] | None,
    failed_codes: dict[str, str] | None,
    delta_keys: list[str] | None,
    row_count: int,
    aborted: bool = False,
    abort_reason: str = "",
    predicted_requests: int = 0,
    processed_requests: int = 0,
    summary: dict[str, Any] | None = None,
) -> None:
    now = _utc_now()
    frame = pd.DataFrame(
        [
            {
                "run_id": f"{dataset}:{reason}:{now.isoformat()}",
                "dataset": dataset,
                "reason": reason,
                "provider": str(provider).strip().upper(),
                "requested_start": str(requested_start),
                "requested_end": str(requested_end),
                "status": str(status).strip().upper(),
                "coverage_complete": bool(coverage_complete),
                "failed_days_json": _serialize_json(list(failed_days or [])),
                "failed_codes_json": _serialize_json(dict(failed_codes or {})),
                "delta_keys_json": _serialize_json(list(delta_keys or [])),
                "row_count": int(row_count),
                "aborted": bool(aborted),
                "abort_reason": str(abort_reason or ""),
                "predicted_requests": int(predicted_requests or 0),
                "processed_requests": int(processed_requests or 0),
                "summary_json": _serialize_json(dict(summary or {})),
                "created_at": now,
            }
        ]
    )

    con = _connect()
    try:
        ensure_warehouse_schema(con)
        _register_frame(con, "ingest_run_insert", frame)
        con.execute("INSERT INTO ingest_runs SELECT * FROM ingest_run_insert")
    finally:
        con.close()


def update_ingest_watermark(
    *,
    dataset: WarehouseDataset,
    watermark_key: str,
    status: str,
    coverage_complete: bool,
    provider: str,
    details: dict[str, Any] | None = None,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "watermark_key": str(watermark_key or ""),
                "coverage_complete": bool(coverage_complete),
                "status": str(status).strip().upper(),
                "provider": str(provider).strip().upper(),
                "details_json": _serialize_json(dict(details or {})),
                "updated_at": _utc_now(),
            }
        ]
    )

    con = _connect()
    try:
        ensure_warehouse_schema(con)
        _register_frame(con, "ingest_watermark_upsert", frame)
        con.execute(
            """
            INSERT INTO ingest_watermarks AS target
            SELECT * FROM ingest_watermark_upsert
            ON CONFLICT (dataset) DO UPDATE SET
                watermark_key = excluded.watermark_key,
                coverage_complete = excluded.coverage_complete,
                status = excluded.status,
                provider = excluded.provider,
                details_json = excluded.details_json,
                updated_at = excluded.updated_at
            """
        )
    finally:
        con.close()


def read_dataset_status(dataset: WarehouseDataset) -> dict[str, Any]:
    if not warehouse_exists():
        return {}

    con = _connect(read_only=True)
    try:
        run_row = con.execute(
            """
            SELECT
                dataset,
                provider,
                requested_end,
                status,
                coverage_complete,
                failed_days_json,
                failed_codes_json,
                reason,
                delta_keys_json,
                aborted,
                abort_reason,
                predicted_requests,
                processed_requests,
                created_at
            FROM ingest_runs
            WHERE dataset = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [dataset],
        ).fetchone()
        watermark_row = con.execute(
            """
            SELECT
                watermark_key,
                coverage_complete,
                status,
                provider,
                details_json,
                updated_at
            FROM ingest_watermarks
            WHERE dataset = ?
            """,
            [dataset],
        ).fetchone()
    finally:
        con.close()

    result: dict[str, Any] = {}
    if run_row is not None:
        result.update(
            {
                "status": str(run_row[3] or "").strip().upper(),
                "provider": str(run_row[1] or "").strip().upper(),
                "end": "".join(ch for ch in str(run_row[2] or "") if ch.isdigit())[:8],
                "coverage_complete": bool(run_row[4]),
                "failed_days": [
                    str(item).strip()
                    for item in _deserialize_json(run_row[5], [])
                    if str(item).strip()
                ],
                "failed_codes": {
                    str(key).strip(): str(value).strip()
                    for key, value in _deserialize_json(run_row[6], {}).items()
                    if str(key).strip() and str(value).strip()
                },
                "reason": str(run_row[7] or "").strip(),
                "delta_codes": [
                    str(item).strip()
                    for item in _deserialize_json(run_row[8], [])
                    if str(item).strip()
                ],
                "aborted": bool(run_row[9]),
                "abort_reason": str(run_row[10] or "").strip(),
                "predicted_requests": int(run_row[11] or 0),
                "processed_requests": int(run_row[12] or 0),
            }
        )
    if watermark_row is not None:
        result.setdefault("status", str(watermark_row[2] or "").strip().upper())
        result.setdefault("provider", str(watermark_row[3] or "").strip().upper())
        watermark_key = str(watermark_row[0] or "").strip()
        if len("".join(ch for ch in watermark_key if ch.isdigit())) >= 6:
            result.setdefault("end", "".join(ch for ch in watermark_key if ch.isdigit())[:8])
        result.setdefault("coverage_complete", bool(watermark_row[1]))
        result["watermark_key"] = watermark_key
    return result


def get_dataset_artifact_key(dataset: WarehouseDataset) -> tuple[int, int, str, str, str]:
    if not warehouse_exists():
        return (0, 0, "", "", "")
    stat = WAREHOUSE_PATH.stat()
    status = read_dataset_status(dataset)
    return (
        int(stat.st_mtime_ns),
        int(stat.st_size),
        str(status.get("watermark_key", "")),
        str(status.get("status", "")),
        str(status.get("end", "")),
    )


def probe_dataset_mode(dataset: WarehouseDataset) -> str:
    if not warehouse_exists():
        return "SAMPLE"

    con = _connect(read_only=True)
    try:
        if dataset == "market_prices":
            count = int(con.execute("SELECT COUNT(*) FROM fact_krx_index_daily").fetchone()[0])
        else:
            count = int(con.execute("SELECT COUNT(*) FROM fact_macro_monthly").fetchone()[0])
    finally:
        con.close()
    return "CACHED" if count > 0 else "SAMPLE"


def market_row_count() -> int:
    if not warehouse_exists():
        return 0
    con = _connect(read_only=True)
    try:
        return int(con.execute("SELECT COUNT(*) FROM fact_krx_index_daily").fetchone()[0])
    finally:
        con.close()


def macro_row_count() -> int:
    if not warehouse_exists():
        return 0
    con = _connect(read_only=True)
    try:
        return int(con.execute("SELECT COUNT(*) FROM fact_macro_monthly").fetchone()[0])
    finally:
        con.close()
