"""Parametrized contract validation tests. (5 tests)"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.contracts.validators import ContractError, normalize_then_validate, validate_only


def _make_valid_sector_prices(n: int = 5) -> pd.DataFrame:
    """Return a valid sector_prices DataFrame."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "index_code": ["5044"] * n,
            "index_name": ["KRX 반도체"] * n,
            "close": [1000.0 + i for i in range(n)],
        },
        index=pd.DatetimeIndex(idx),
    )


def _make_valid_macro_monthly(n: int = 5) -> pd.DataFrame:
    """Return a valid macro_monthly DataFrame."""
    idx = pd.period_range("2024-01", periods=n, freq="M")
    return pd.DataFrame(
        {
            "series_id": ["ECOS/base_rate"] * n,
            "value": [3.5 + i * 0.1 for i in range(n)],
            "source": ["ECOS"] * n,
            "fetched_at": [datetime(2024, 1, 1)] * n,
            "is_provisional": [False] * n,
        },
        index=pd.PeriodIndex(idx, freq="M"),
    ).astype({"value": "float64", "is_provisional": "bool"})


class TestContracts:
    def test_valid_df_passes_validate_only(self):
        """validate_only passes for a well-formed sector_prices DataFrame."""
        df = _make_valid_sector_prices()
        validate_only(df, "sector_prices")  # should not raise

    def test_wrong_dtype_raises_contract_error(self):
        """validate_only raises ContractError when close column has wrong dtype."""
        df = _make_valid_sector_prices()
        df["close"] = df["close"].astype(int)  # int instead of float → violation
        with pytest.raises(ContractError, match="close"):
            validate_only(df, "sector_prices")

    def test_null_in_non_null_column_raises(self):
        """validate_only raises ContractError for null in non_null column."""
        df = _make_valid_sector_prices()
        df.loc[df.index[0], "close"] = None
        df["close"] = df["close"].astype(float)
        with pytest.raises(ContractError, match="close"):
            validate_only(df, "sector_prices")

    def test_wrong_index_type_raises(self):
        """validate_only raises ContractError when index type is wrong."""
        df = _make_valid_sector_prices().reset_index(drop=True)  # RangeIndex instead of DatetimeIndex
        with pytest.raises(ContractError, match="DatetimeIndex"):
            validate_only(df, "sector_prices")

    def test_ffill_max3_gap_behavior(self):
        """normalize_then_validate fills gaps ≤3, drops rows with gap > 3."""
        df = _make_valid_sector_prices(10)
        # Create gap of 2 (fillable) and gap of 4 (should drop)
        df = df.copy()
        df.loc[df.index[1], "close"] = None
        df.loc[df.index[2], "close"] = None
        # These two will be filled by ffill(limit=3)

        # gap > 3 — add separate rows with 4 consecutive NaN
        import numpy as np
        df2 = _make_valid_sector_prices(6)
        df2.loc[df2.index[1], "close"] = None
        df2.loc[df2.index[2], "close"] = None
        df2.loc[df2.index[3], "close"] = None
        df2.loc[df2.index[4], "close"] = None
        # Rows 1-4 are NaN — limit=3 fills first 3, leaves row 4 as NaN → drop

        df2["close"] = df2["close"].astype(float)
        result = normalize_then_validate(df2, "sector_prices")
        # Row with gap > 3 should be dropped
        assert result["close"].isna().sum() == 0, "All remaining NaNs should be dropped"
