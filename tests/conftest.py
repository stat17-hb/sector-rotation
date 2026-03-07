from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_duckdb_warehouse(tmp_path, monkeypatch):
    import src.data_sources.warehouse as warehouse

    monkeypatch.setattr(warehouse, "WAREHOUSE_PATH", tmp_path / "warehouse.duckdb")
    monkeypatch.setattr(warehouse, "DEFAULT_PRICE_EXPORT_PATH", tmp_path / "curated" / "sector_prices.parquet")
    monkeypatch.setattr(warehouse, "DEFAULT_MACRO_EXPORT_PATH", tmp_path / "curated" / "macro_monthly.parquet")
